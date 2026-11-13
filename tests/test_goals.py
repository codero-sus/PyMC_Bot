"""Observation-driven gathering and its selected-bot API."""
from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pymc_bot.agent import AgentLoop, Decision
from pymc_bot.goals import GatherGoal
from pymc_bot.server import create_app

IDENTITY = ("localhost", 25565, "testbot")


def state(*items, **extra):
    return {"status": "connected", "health": 20, "inventory": list(items), **extra}


def step(goal, observation, **permissions):
    return goal.next(observation, IDENTITY, mining=permissions.get("mining", True),
                     movement=permissions.get("movement", True))


def test_gather_verifies_inventory_and_selects_drop():
    goal = GatherGoal()
    goal.start("minecraft:oak_log", 3, IDENTITY)
    kind, action = step(goal, state({"name": "minecraft:oak_log", "count": 1}))
    assert kind == "action" and action["action"] == "mine"
    goal.finish(action["token"], True, "mined", state({"name": "oak_log", "count": 2}))
    assert goal.snapshot()["status"] == "active"
    kind, action = step(goal, state({"name": "oak_log", "count": 2},
                                    ground_items=[{"item": "minecraft:oak_log", "dist": 2}]))
    assert kind == "action" and action["action"] == "collect"
    assert action["params"] == {"item": "oak_log"}
    goal.finish(action["token"], True, "collected", state({"name": "oak_log", "count": 3}))
    assert goal.snapshot()["status"] == "completed"


@pytest.mark.parametrize("unsafe", [
    {"health": None}, {"health": 8}, {"on_fire": True},
    {"in_water": True, "oxygen": None}, {"in_water": True, "oxygen": 4},
    {"entities": [{"hostile": True, "dist": 4}]}, {"status": "disconnected"},
])
def test_goal_pauses_for_unsafe_observations(unsafe):
    goal = GatherGoal()
    goal.start("oak_log", 2, IDENTITY)
    assert step(goal, state(**unsafe))[0] == "wait"
    assert goal.snapshot()["status"] == "paused"
    assert step(goal, state())[0] == "action"


def test_permissions_identity_and_observation():
    goal = GatherGoal()
    goal.start("coal", 1, IDENTITY)
    assert step(goal, state(), mining=False)[0] == "wait"
    assert step(goal, state(), movement=False)[0] == "wait"
    assert step(goal, {"health": 20})[0] == "wait"
    assert step(goal, state({"name": "coal", "count": "bad"}))[0] == "wait"
    assert goal.next(state(), ("other",), mining=True, movement=True)[0] == "wait"
    assert goal.snapshot()["status"] == "failed"


def test_failure_bounds_and_no_progress():
    goal = GatherGoal()
    goal.start("raw_iron", 1, IDENTITY)
    for _ in range(2):
        _, action = step(goal, state())
        goal.finish(action["token"], False, "not reachable", state())
    assert goal.snapshot()["status"] == "failed"
    goal.start("coal", 1, IDENTITY)
    for _ in range(4):
        _, action = step(goal, state())
        goal.finish(action["token"], True, "mined", state())
    assert step(goal, state())[0] == "wait"
    assert goal.snapshot()["status"] == "failed"


@pytest.mark.parametrize("item,first,second", [
    ("coal", "coal_ore", "deepslate_coal_ore"),
    ("raw_iron", "iron_ore", "deepslate_iron_ore"),
])
def test_alternate_ore_after_failure_or_no_progress(item, first, second):
    for success in (False, True):
        goal = GatherGoal()
        goal.start(item, 1, IDENTITY)
        _, action = step(goal, state())
        assert action["params"]["block"] == first
        goal.finish(action["token"], success, "attempted", state())
        _, action = step(goal, state())
        assert action["params"]["block"] == second
        goal.finish(action["token"], True, "mined", state({"name": item, "count": 1}))
        assert goal.snapshot()["status"] == "completed"


def test_targeted_collection_bridge():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js not installed")
    root = Path(__file__).resolve().parents[1]
    subprocess.run([node, str(root / "tests/fixtures/gather_bridge_harness.js"),
                    str(root / "pymc_bot/node/minecraft_bridge.js")], check=True, timeout=10)


def test_targeted_collection_simulation(live_bot):
    backend = live_bot.backend
    backend._ground = [
        {"name": "dirt", "count": 1, "x": 1, "z": 0, "at": 0},
        {"name": "oak_log", "count": 1, "x": 2, "z": 0, "at": 0},
    ]
    assert live_bot.collect_items(item="minecraft:oak_log")
    assert [drop["name"] for drop in backend._ground] == ["dirt"]


def test_inflight_claim_and_cancel_are_generation_safe():
    goal = GatherGoal()
    goal.start("oak_log", 1, IDENTITY)
    _, action = step(goal, state())
    assert step(goal, state())[0] == "wait"
    with pytest.raises(RuntimeError):
        goal.start("coal", 1, IDENTITY)
    assert goal.cancel()
    goal.finish(action["token"], True, "done", state({"name": "oak_log", "count": 1}))
    assert goal.snapshot()["status"] == "cancelled"
    goal.start("coal", 1, IDENTITY)
    assert goal.snapshot()["status"] == "active"


def test_agent_runs_goal_not_model_and_respects_recovery(live_bot, store, log, monkeypatch):
    agent = AgentLoop(live_bot, store, log)
    agent.start_gather_goal("oak_log", 5)
    monkeypatch.setattr(agent, "_decide", lambda *_: pytest.fail("model must not run during a goal"))
    previous = Decision("mine", {"block": "oak_log"}, source="heuristic")
    for _ in range(2):
        agent.recovery.record(previous, live_bot.snapshot(), False, "failed")
    assert agent.run_once()[0]
    assert agent.run_once()[0]
    assert agent.goal.snapshot()["status"] == "completed"


def test_agent_replans_after_mining_failure(live_bot, store, log, monkeypatch):
    agent = AgentLoop(live_bot, store, log)
    agent.start_gather_goal("raw_iron", 1)
    calls = []
    def mine(block):
        calls.append(block)
        if block == "deepslate_iron_ore":
            live_bot.backend._inv.add("raw_iron", 1)
            return True
        return False
    monkeypatch.setattr(live_bot, "mine", mine)
    assert not agent.run_once()[0]
    assert agent.run_once()[0]
    assert calls == ["iron_ore", "deepslate_iron_ore"]
    assert agent.goal.snapshot()["status"] == "completed"


def test_agent_cancellation_during_claim(live_bot, store, log, monkeypatch):
    agent = AgentLoop(live_bot, store, log)
    agent.start_gather_goal("oak_log", 4)
    claimed = threading.Event()
    resume = threading.Event()
    original = agent._act
    def delayed(decision):
        claimed.set()
        assert resume.wait(5)
        return original(decision)
    monkeypatch.setattr(agent, "_act", delayed)
    thread = threading.Thread(target=agent.run_once)
    thread.start()
    assert claimed.wait(5)
    assert agent.cancel_goal()
    resume.set()
    thread.join(5)
    assert not thread.is_alive()
    assert agent.goal.snapshot()["status"] == "cancelled"
    assert next(i["count"] for i in live_bot.snapshot()["inventory"] if i.get("name") == "oak_log") == 3


def test_goal_api(tmp_path):
    with TestClient(create_app(config_path=tmp_path / "cfg.json")) as client:
        assert client.put("/api/ai/goal", json={"item": "oak_log", "count": 2}).status_code == 409
        client.put("/api/config", json={"minecraft": {"backend": "simulated"}})
        assert client.post("/api/bot/start").status_code == 200
        import time
        for _ in range(100):
            if client.get("/api/status").json()["bot"]["state"] == "connected":
                break
            time.sleep(0.02)
        assert client.put("/api/ai/goal", json={"item": "diamond", "count": 1}).status_code == 422
        assert client.put("/api/ai/goal", json={"item": "oak_log", "count": True}).status_code == 422
        assert client.put("/api/ai/goal", json={"item": "oak_log", "count": 4}).status_code == 200
        assert client.get("/api/ai/goal").json()["task"]["status"] == "active"
        assert client.post("/api/ai/step").json()["ok"]
        assert client.get("/api/ai/goal").json()["task"]["status"] == "completed"
        assert client.delete("/api/ai/goal").status_code == 200
