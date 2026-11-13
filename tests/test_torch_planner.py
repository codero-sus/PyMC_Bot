"""Bounded prerequisite planning with real inventory checks."""
from __future__ import annotations

import pytest

from pymc_bot.agent import AgentLoop, Decision
from pymc_bot.goals import GatherGoal

IDENTITY = ("host", 25565, "bot")


def state(*items, **extra):
    return {"status": "connected", "health": 20, "inventory": list(items), **extra}


def next_step(goal, obs, **kwargs):
    return goal.next(obs, IDENTITY, mining=kwargs.get("mining", True), movement=kwargs.get("movement", True),
                     inventory=kwargs.get("inventory", True))


def test_prerequisites_and_verification():
    goal = GatherGoal()
    goal.start("torch", 4, IDENTITY)
    obs = state()
    _, action = next_step(goal, obs)
    assert action["action"] == "mine" and action["params"]["block"] == "oak_log"
    obs = state({"name": "oak_log", "count": 1})
    goal.finish(action["token"], True, "mined", obs)
    _, action = next_step(goal, obs)
    assert action["params"] == {"item": "oak_planks", "count": 4, "no_station": True}
    obs = state({"name": "oak_planks", "count": 4})
    goal.finish(action["token"], True, "crafted", obs)
    _, action = next_step(goal, obs)
    assert action["params"]["item"] == "stick"
    obs = state({"name": "oak_planks", "count": 2}, {"name": "stick", "count": 4})
    goal.finish(action["token"], True, "crafted", obs)
    assert next_step(goal, obs)[0] == "wait"
    assert "pickaxe" in goal.snapshot()["detail"]
    obs = state({"name": "stick", "count": 4}, {"name": "coal", "count": 1})
    _, action = next_step(goal, obs)
    assert action["params"]["item"] == "torch"
    goal.finish(action["token"], True, "claimed success", obs)
    assert goal.snapshot()["status"] != "completed"
    _, action = next_step(goal, obs)
    goal.finish(action["token"], True, "crafted", state({"name": "torch", "count": 4}))
    assert goal.snapshot()["status"] == "completed"


def test_torch_plan_permissions_safety_and_finite_work():
    goal = GatherGoal()
    goal.start("torch", 64, IDENTITY)
    assert next_step(goal, state(), inventory=False)[0] == "wait"
    assert "inventory" in goal.snapshot()["detail"]
    assert next_step(goal, state(health=3))[0] == "wait"
    for _ in range(4):
        _, action = next_step(goal, state())
        goal.finish(action["token"], True, "no change", state())
    assert next_step(goal, state())[0] == "wait"
    assert goal.snapshot()["status"] == "failed"


def test_torch_plan_uses_drop_without_pickaxe_and_charcoal():
    goal = GatherGoal()
    goal.start("torch", 4, IDENTITY)
    obs = state({"name": "stick", "count": 1}, ground_items=[{"item": "coal", "dist": 3}])
    _, action = next_step(goal, obs)
    assert action["action"] == "collect" and action["params"]["item"] == "coal"
    goal.finish(action["token"], True, "picked up", state({"name": "stick", "count": 1}, {"name": "charcoal", "count": 1}))
    _, action = next_step(goal, state({"name": "stick", "count": 1}, {"name": "charcoal", "count": 1}))
    assert action["params"]["item"] == "torch"


def test_ai_crafting_cannot_place_table_or_move_when_disallowed(live_bot, store, log, monkeypatch):
    agent = AgentLoop(live_bot, store, log)
    calls = []
    monkeypatch.setattr(live_bot, "skill", lambda name, params: (calls.append((name, params)) or (True, "ok")))
    agent.execute(Decision("craft", {"item": "stick", "no_station": False}, "ollama"))
    assert calls[-1][1]["no_station"] is True  # building disabled by default
    store.update({"agent": {"allow_building": True, "allow_movement": False}})
    agent.execute(Decision("craft", {"item": "stick"}, "ollama"))
    assert calls[-1][1]["no_station"] is True
    agent.execute(Decision("craft", {"item": "stick"}, "manual"), enforce_permissions=False)
    assert "no_station" not in calls[-1][1]


def test_agent_completes_torch_chain_in_simulation(live_bot, store, log, monkeypatch):
    bot = live_bot
    bot.backend._inv.add("coal", 2)
    agent = AgentLoop(bot, store, log)
    agent.start_gather_goal("torch", 16)
    monkeypatch.setattr(agent, "_decide", lambda *_: pytest.fail("LLM should not run during the goal"))
    for _ in range(8):
        assert agent.run_once()[0]
        if agent.goal.snapshot()["status"] == "completed":
            break
    assert agent.goal.snapshot()["status"] == "completed"
    assert bot.inventory().count("torch") >= 16
