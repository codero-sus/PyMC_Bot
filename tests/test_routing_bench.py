"""Opt-in routing and read-only benchmark checks."""
from __future__ import annotations

import json

import httpx
import pytest

from pymc_bot.agent import AgentLoop
from pymc_bot.bench import benchmark
from pymc_bot.config import OllamaSettings
from pymc_bot.ollama import OllamaClient
from pymc_bot.routing import select_ollama_model


def config(**overrides):
    return OllamaSettings(adaptive_routing=True, model="main-vl", routine_model="turbo",
                          complex_model="thinking", **overrides)


def test_routing_is_optional_and_vision_keeps_explicit_model():
    cfg = config()
    assert select_ollama_model(cfg, False, {"health": 20}, "", [])[0] == "turbo"
    assert select_ollama_model(cfg, False, {"health": 5}, "", [])[0] == "thinking"
    assert select_ollama_model(cfg, False, {"health": 20}, "build a base", [])[0] == "thinking"
    assert select_ollama_model(cfg, False, {"health": 20, "entities": [
        {"hostile": True, "dist": 2}, {"hostile": True, "dist": 9}]}, "", [])[0] == "thinking"
    assert select_ollama_model(cfg, True, {"health": 1}, "plan", [])[0] == "main-vl"
    assert select_ollama_model(OllamaSettings(), False, {"health": 20}, "", [])[1] == "disabled"
    cfg.routine_model = ""
    assert select_ollama_model(cfg, False, {"health": 20}, "", [])[0] == "main-vl"


def test_agent_sends_selected_model_and_preserves_vl(live_bot, store, log, monkeypatch):
    store.update({"ollama": {"adaptive_routing": True, "model": "main-vl", "routine_model": "turbo",
                             "complex_model": "thinking", "json_mode": True}})
    agent = AgentLoop(live_bot, store, log)
    calls = []
    class Fake:
        def chat(self, messages, **kwargs):
            calls.append(kwargs)
            return '{"action":"wait"}'
    monkeypatch.setattr(agent, "_ollama_client", lambda _: Fake())
    decision = agent._ask_ollama(store.settings, live_bot.snapshot() | {"health": 20, "entities": []})
    assert decision.source == "ollama:turbo" and calls[-1]["model"] == "turbo"
    assert calls[-1]["json_mode"] is True
    assert agent.status["model_route"]["model"] == "turbo"
    store.update({"agent": {"vision": True}})
    monkeypatch.setattr(agent, "vision_frame", lambda _: ["test-image"])
    decision = agent._ask_ollama(store.settings, live_bot.snapshot() | {"health": 1})
    assert decision.source == "ollama:main-vl"
    assert calls[-1]["images"] == ["test-image"] and "model" not in calls[-1]


def test_benchmark_distinguishes_json_parsing_known_and_expected():
    outputs = iter(['{"action":"eat"}', 'Here: {"action":"fly"}', 'bad'])
    def handler(request):
        payload = json.loads(request.content)
        assert payload["format"] == "json"
        return httpx.Response(200, json={"message": {"content": next(outputs)}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        report = benchmark(OllamaClient(model="test", client=http), json_mode=True)
    assert report["cases"] == 3
    assert report["strict_json_rate"] == pytest.approx(1 / 3)
    assert report["parsed_rate"] == pytest.approx(2 / 3)
    assert report["known_action_rate"] == pytest.approx(1 / 3)
    assert report["expected_action_rate"] == pytest.approx(1 / 3)
    assert report["results"][2]["error"]


def test_benchmark_repeats_are_bounded():
    with pytest.raises(ValueError):
        benchmark(OllamaClient(), repeats=0)
