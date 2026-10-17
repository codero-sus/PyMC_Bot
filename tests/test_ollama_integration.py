"""Ollama client + stub brain integration (no real model required)."""

from __future__ import annotations

import time

import pytest

from pymc_bot.agent import AgentLoop
from pymc_bot.bot import MinecraftBot
from pymc_bot.config import ConfigStore
from pymc_bot.events import EventLog
from pymc_bot.ollama import OllamaClient, OllamaError
from pymc_bot.ollama_stub import STUB_MODEL, serve


@pytest.fixture()
def stub_server():
    httpd = serve("127.0.0.1", 0)
    host, port = httpd.server_address[:2]
    yield f"http://{host}:{port}"
    httpd.shutdown()
    httpd.server_close()


def test_stub_reports_its_models(stub_server: str):
    client = OllamaClient(base_url=stub_server, timeout=5.0)
    ok, detail = client.ping()
    client.close()
    assert ok is True
    assert "model" in detail.lower()


def test_stub_answers_chat_and_generate(stub_server: str):
    client = OllamaClient(base_url=stub_server, timeout=5.0)
    state = '"health":20,"food":20,"players":[],"recent_chat":[],"position":{"x":0,"y":64,"z":0}'
    prompt = "Current Minecraft world state:\n{" + state + "}\n\nPick the single best next action."

    chat_reply = client.chat([{"role": "user", "content": prompt}])
    generate_reply = client.generate(prompt)
    models = [m["name"] for m in client.list_models()]
    client.close()

    assert STUB_MODEL in models
    assert '"action"' in chat_reply
    assert '"action"' in generate_reply


def test_client_reports_unreachable_ollama():
    client = OllamaClient(base_url="http://127.0.0.1:1", timeout=1.0)
    ok, detail = client.ping()
    assert ok is False and detail
    with pytest.raises(OllamaError):
        client.chat([{"role": "user", "content": "hi"}], timeout=1.0)
    client.close()


def test_agent_uses_ollama_when_enabled(store: ConfigStore, log: EventLog, stub_server: str):
    store.update(
        {
            "minecraft": {"backend": "simulated"},
            "ollama": {"enabled": True, "base_url": stub_server, "model": STUB_MODEL, "decision_interval": 0.5},
        }
    )
    bot = MinecraftBot(store, log, auto_reconnect=False, backend=None)
    bot.start()
    time.sleep(0.5)
    agent = AgentLoop(bot, store, log)
    try:
        ok, detail = agent.run_once()
        assert detail
        assert agent.status["last_decision"]["source"].startswith("ollama:")
        assert agent.status["errors"] == 0

        assert agent.start() is True
        time.sleep(1.5)
        assert agent.status["decisions"] >= 2
    finally:
        agent.stop()
        bot.stop()


def test_agent_falls_back_when_ollama_is_down(store: ConfigStore, log: EventLog):
    store.update(
        {
            "minecraft": {"backend": "simulated"},
            "ollama": {"enabled": True, "base_url": "http://127.0.0.1:1", "decision_interval": 0.5},
        }
    )
    bot = MinecraftBot(store, log, auto_reconnect=False)
    bot.start()
    time.sleep(0.5)
    agent = AgentLoop(bot, store, log)
    try:
        ok, detail = agent.run_once()
        assert isinstance(ok, bool) and detail
        assert agent.status["last_decision"]["source"] == "heuristic"
        messages = [event["message"] for event in log.tail(50)]
        assert any("failed" in message for message in messages)
    finally:
        agent.stop()
        bot.stop()


def test_stub_decision_is_always_executable(store: ConfigStore, log: EventLog, stub_server: str):
    """Whatever the stub answers must parse and run through the executor."""
    store.update({"minecraft": {"backend": "simulated"}, "ollama": {"enabled": True, "base_url": stub_server}})
    bot = MinecraftBot(store, log, auto_reconnect=False)
    bot.start()
    time.sleep(0.5)
    agent = AgentLoop(bot, store, log)
    client = OllamaClient(base_url=stub_server, timeout=5.0)
    try:
        agent._ollama = client
        for _ in range(3):
            state = bot.snapshot()
            decision = agent._ask_ollama(store.settings, state)
            assert decision.source.startswith("ollama:")
            ok, detail = agent.execute(decision)
            assert isinstance(ok, bool), detail
    finally:
        client.close()
        agent.stop()
        bot.stop()
