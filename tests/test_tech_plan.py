"""Deterministic early-game prerequisites and verification, not live-game proof."""
from __future__ import annotations

import threading

import pytest

from pymc_bot.agent import AgentLoop
from pymc_bot.goals import GatherGoal
from pymc_bot.inventory import Inventory
from pymc_bot.tech_plan import RECIPES, next_tech_step

IDENTITY = ("server", 25565, "bot")


def state(*items, **extra):
    return {"status": "connected", "health": 20, "inventory": list(items), **extra}


def stack(name, count=1):
    return {"name": name, "count": count}


def turn(goal, snapshot, **permissions):
    return goal.next(snapshot, IDENTITY, mining=permissions.get("mining", True),
                     movement=permissions.get("movement", True), inventory=permissions.get("inventory", True),
                     building=permissions.get("building", True))


@pytest.mark.parametrize("target", RECIPES)
def test_tech_goal_requires_explicit_building_permission(target):
    goal = GatherGoal()
    goal.start(target, 1, IDENTITY)
    assert turn(goal, state(), building=False)[0] == "wait"
    assert "building" in goal.snapshot()["detail"]
    assert turn(goal, state(), inventory=False)[0] == "wait"
    assert "inventory" in goal.snapshot()["detail"]
    assert turn(goal, state(), mining=False)[0] == "wait"
    assert turn(goal, state(health=3))[0] == "wait"
    assert turn(goal, state(), building=True)[0] == "action"


def test_tech_goals_validate_item_and_bounds():
    goal = GatherGoal()
    with pytest.raises(ValueError):
        goal.start("stone_pickaxe", 9, IDENTITY)
    with pytest.raises(ValueError):
        goal.start("nether_portal", 1, IDENTITY)
    assert goal.start("furnace", 8, IDENTITY)["item"] == "furnace"


def test_plan_builds_wooden_pick_before_mining_stone():
    assert next_tech_step(state(), "stone_pickaxe") == ("", {}, "oak_log")
    action, params, progress = next_tech_step(state(stack("oak_log")), "stone_pickaxe")
    assert (action, params["item"], params["no_station"], progress) == ("craft", "oak_planks", True, "oak_planks")
    assert next_tech_step(state(stack("oak_planks", 2)), "stone_pickaxe")[1]["item"] == "stick"
    assert next_tech_step(state(stack("stick", 2), stack("oak_planks", 7)), "stone_pickaxe")[1]["item"] == "wooden_pickaxe"
    assert next_tech_step(state(stack("wooden_pickaxe")), "stone_pickaxe") == ("", {}, "cobblestone")
    action, params, output = next_tech_step(
        state(stack("wooden_pickaxe"), stack("cobblestone", 3), stack("stick", 2), stack("oak_planks", 4)), "stone_pickaxe")
    assert (action, params["item"], params["no_station"], output) == ("craft", "stone_pickaxe", False, "stone_pickaxe")


def test_planner_never_combines_single_planks_to_infer_a_recipe():
    action, _, target = next_tech_step(state(stack("oak_planks"), stack("birch_planks")), "stone_pickaxe")
    assert action == "" and target == "oak_log"


def test_progress_is_verified_on_each_step():
    goal = GatherGoal()
    goal.start("stone_pickaxe", 1, IDENTITY)
    _, step = turn(goal, state())
    goal.finish(step["token"], True, "not necessarily picked up", state())
    assert goal.snapshot()["no_progress"] == 1
    assert goal.snapshot()["status"] == "active"
    _, step = turn(goal, state())
    goal.finish(step["token"], True, "got log", state(stack("oak_log")))
    assert goal.snapshot()["no_progress"] == 0
    assert goal.snapshot()["status"] != "completed"
    assert turn(goal, state(stack("stone_pickaxe")))[0] == "wait"
    assert goal.snapshot()["status"] == "completed"


@pytest.mark.parametrize("hazard", [
    {"in_lava": True},
    {"entities": [{"type": "minecraft:arrow", "dx": 4, "dy": 0.9, "dz": 0, "vx": -0.5, "vy": 0, "vz": 0}]},
    {"entities": [{"dist": 3, "potion": {"form": "cloud", "kind": "unknown"}}]},
    {"entities": [{"dist": 2, "potion": {"form": "thrown", "kind": "bad"}}]},
])
def test_pauses_when_new_hazards_observed(hazard):
    goal = GatherGoal()
    goal.start("stone_pickaxe", 1, IDENTITY)
    assert turn(goal, state(**hazard))[0] == "wait"
    assert goal.snapshot()["status"] == "paused"
    assert turn(goal, state())[0] == "action"


def test_changed_identity_and_cancel_inflight():
    goal = GatherGoal()
    goal.start("stone_sword", 1, IDENTITY)
    _, step = turn(goal, state())
    assert goal.cancel()
    goal.finish(step["token"], True, "late result", state(stack("stone_sword")))
    assert goal.snapshot()["status"] == "cancelled"
    goal.start("furnace", 1, IDENTITY)
    assert goal.next(state(), ("other",), mining=True, movement=True, inventory=True, building=True)[0] == "wait"
    assert goal.snapshot()["status"] == "failed"


def test_concurrent_claim_only_once():
    goal = GatherGoal()
    goal.start("stone_pickaxe", 1, IDENTITY)
    barrier = threading.Barrier(3)
    results = []
    def claim():
        barrier.wait()
        results.append(turn(goal, state())[0])
    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(5)
    assert sorted(results) == ["action", "wait"]


def test_failed_action_can_still_show_verified_progress():
    goal = GatherGoal()
    goal.start("stone_pickaxe", 1, IDENTITY)
    _, step = turn(goal, state())
    goal.finish(step["token"], False, "response lost", state(stack("oak_log")))
    assert goal.snapshot()["status"] == "active"
    assert goal.snapshot()["failures"] == 0
    _, step = turn(goal, state(stack("oak_log")))
    goal.finish(step["token"], False, "response lost", state(stack("stone_pickaxe")))
    assert goal.snapshot()["status"] == "completed"


def test_iron_goal_bootstraps_tool_but_never_without_permission():
    goal = GatherGoal()
    goal.start("raw_iron", 1, IDENTITY)
    obs = state(stack("wooden_pickaxe"))
    assert turn(goal, obs, building=False)[0] == "wait"
    assert "pickaxe" in goal.snapshot()["detail"]
    _, step = turn(goal, obs, building=True)
    assert step["params"]["block"] == "stone"
    goal.finish(step["token"], True, "mined", state(stack("wooden_pickaxe"), stack("cobblestone")))
    assert goal.snapshot()["status"] == "active"


def test_existing_drops_can_be_collected_without_pickaxe():
    goal = GatherGoal()
    goal.start("raw_iron", 1, IDENTITY)
    obs = state(ground_items=[{"item": "minecraft:raw_iron", "dist": 2}])
    kind, step = turn(goal, obs, building=False, inventory=False)
    assert kind == "action" and step["action"] == "collect"
    assert step["params"] == {"item": "raw_iron"}


def test_simulator_respects_pickaxe_and_realistic_drops(live_bot):
    backend = live_bot.backend
    backend._inv = Inventory()
    assert backend.dig("stone")
    assert backend._inv.count("cobblestone") == 0
    backend._inv.add("wooden_pickaxe", 1)
    assert backend.dig("stone")
    assert backend._inv.count("cobblestone") == 1
    assert backend.dig("iron_ore")
    assert backend._inv.count("raw_iron") == 0
    backend._inv = Inventory()
    backend._inv.add("stone_pickaxe", 1)
    assert backend.dig("iron_ore")
    assert backend._inv.count("raw_iron") == 1


def test_goal_action_exception_is_bounded_and_claim_released(live_bot, store, log, monkeypatch):
    store.update({"agent": {"allow_building": True}})
    live_bot.backend._inv = Inventory()
    live_bot.backend._ground.clear()
    agent = AgentLoop(live_bot, store, log)
    agent.start_gather_goal("stone_pickaxe", 1)
    def fail(_):
        raise RuntimeError("boom")
    monkeypatch.setattr(agent, "_act", fail)
    ok, detail = agent.run_once()
    assert not ok and "boom" in detail
    assert not agent.goal.snapshot()["in_flight"]
    assert agent.goal.snapshot()["failures"] == 1
    assert not agent.run_once()[0]
    assert agent.goal.snapshot()["status"] == "failed"


@pytest.mark.parametrize("target", [*RECIPES, "cobblestone", "coal", "raw_iron"])
def test_agent_plans_from_empty_inventory_in_simulator(live_bot, store, log, monkeypatch, target):
    store.update({"agent": {"allow_building": True}})
    live_bot.backend._inv = Inventory()
    live_bot.backend._ground.clear()
    agent = AgentLoop(live_bot, store, log)
    agent.start_gather_goal(target, 1)
    monkeypatch.setattr(agent, "_decide", lambda *_: pytest.fail("goal must take precedence"))
    for _ in range(30):
        ok, detail = agent.run_once()
        assert ok, detail
        if agent.goal.snapshot()["status"] in {"completed", "failed"}:
            break
    assert agent.goal.snapshot()["status"] == "completed", agent.goal.snapshot()
    assert live_bot.inventory().count(target) == 1
