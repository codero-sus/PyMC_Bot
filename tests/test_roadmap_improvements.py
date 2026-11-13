"""Practical inference and persistent-waypoint reliability improvements."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import httpx

from pymc_bot.agent import AgentLoop
from pymc_bot.ollama import OllamaClient
from pymc_bot.skills import Places, _server_key


def test_ollama_json_mode_and_vision_are_sent_together():
    payloads = []
    def handler(request):
        payloads.append(json.loads(request.content))
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": '{"action":"wait"}'}})
        return httpx.Response(200, json={"response": '{"action":"wait"}'})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = OllamaClient(client=http)
        client.chat([{"role": "user", "content": "look"}], images=["image"], json_mode=True)
        client.chat([{"role": "user", "content": "look"}])
        client.generate("look", json_mode=True)
    assert payloads[0]["format"] == "json"
    assert payloads[0]["messages"][0]["images"] == ["image"]
    assert "format" not in payloads[1]
    assert payloads[2]["format"] == "json"


def test_agent_requests_json_mode_for_actions(live_bot, store, log):
    store.update({"ollama": {"json_mode": True}})
    agent = AgentLoop(live_bot, store, log)
    seen = []
    class Fake:
        def chat(self, messages, **kwargs):
            seen.append(kwargs)
            return '{"action":"wait"}'
    agent._ollama_client = lambda _: Fake()
    assert agent._ask_ollama(store.settings, live_bot.snapshot()).action == "wait"
    assert seen[0]["json_mode"] is True


def test_places_concurrent_bots_do_not_lose_updates(tmp_path):
    path = tmp_path / "places.json"
    def save(i):
        Places(path).set("server", f"bot{i}", "home", {"x": i, "y": 64, "z": 0, "dimension": "overworld"})
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(save, range(100)))
    assert len(json.loads(path.read_text())["server"]) == 100


def test_waypoints_use_connected_server_and_refuse_wrong_dimension(live_bot, store, monkeypatch):
    original = _server_key(live_bot)
    assert live_bot.skill("set_home")[0]
    store.update({"minecraft": {"host": "another.server"}})
    assert _server_key(live_bot) == original
    snapshot = live_bot.snapshot()
    monkeypatch.setattr(live_bot, "snapshot", lambda: snapshot | {"dimension": "the_nether"})
    monkeypatch.setattr(live_bot, "walk_to", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not walk")))
    ok, message = live_bot.skill("go_home")
    assert not ok and "dimension" in message


def test_invalid_waypoint_coordinates_are_not_navigated(live_bot, monkeypatch):
    Places(live_bot.store.path.parent / "pymc_bot_places.json").set(
        _server_key(live_bot), live_bot.username, "home",
        {"x": "nan", "y": 64, "z": 0, "dimension": "overworld"},
    )
    monkeypatch.setattr(live_bot, "walk_to", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not walk")))
    ok, message = live_bot.skill("go_home")
    assert not ok and "invalid" in message
