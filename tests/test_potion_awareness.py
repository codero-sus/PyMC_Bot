"""Potion observations drive bounded, conservative survival actions."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from pymc_bot.autopilot import Autopilot, Situation, evaluate
from pymc_bot.config import AutopilotSettings

ROOT = Path(__file__).resolve().parents[1]


def potion(kind="good", form="dropped", distance=4, entity_id=42):
    return {"id": entity_id, "type": "minecraft:item", "dist": distance,
            "potion": {"kind": kind, "form": form, "effects": []}}


def state(*entities, **extra):
    return {"health": 20, "food": 20, "pathfinder": True, "entities": list(entities), **extra}


def plan(snapshot, *, settings=None, busy=False):
    return evaluate(settings or AutopilotSettings(), Situation(snap=snapshot, busy=busy))


def test_seek_beneficial_dropped_potions_only_when_safe():
    result = plan(state(potion()))
    assert result.rule == "seek_potions" and result.params == {"mode": "seek", "entity_id": 42}
    assert plan(state(potion(kind="unknown"))) is None
    assert plan(state(potion(form="thrown"))) is None  # missing trajectory
    thrown = {**potion(form="thrown"), "dy": 1, "vy": -0.1}
    assert plan(state(thrown)).rule == "seek_potions"
    assert plan(state({**thrown, "dy": 4})) is None
    assert plan(state(potion(distance=1))) is None
    assert plan(state(potion(), health=8)) is None
    assert plan(state(potion(), pathfinder=False)) is None
    assert plan(state(potion()), busy=True) is None
    assert plan(state(potion()), settings=AutopilotSettings(auto_seek_potions=False)) is None


def test_bad_or_unknown_active_potions_take_priority_but_dropped_bad_is_harmless():
    for kind in ("bad", "unknown"):
        for form in ("thrown", "cloud"):
            result = plan(state(potion(kind, form), potion()), busy=True)
            assert result.rule == "avoid_potions" and result.interrupt and result.params["mode"] == "avoid"
    assert plan(state(potion("bad", "dropped"))) is None
    assert plan(state(potion("good", "cloud"))) is None
    assert plan(state(potion("bad", "thrown"), pathfinder=False)) is None
    assert plan(state(potion("bad", "thrown")), settings=AutopilotSettings(auto_avoid_potions=False)) is None
    assert plan(state(potion("bad", "thrown"), dead=True)).rule == "respawn"


def test_seeking_is_suppressed_by_hostiles_and_potion_hazards():
    zombie = {"type": "minecraft:zombie", "hostile": True, "dist": 8}
    result = plan(state(potion(), zombie))
    assert not result or result.rule != "seek_potions"
    assert plan(state(potion(), potion("unknown", "cloud", entity_id=43))).rule == "avoid_potions"


def test_potion_rules_available_in_panel(store):
    rules = {rule["name"]: rule for rule in Autopilot(lambda: [], store).status([])["rules"]}
    assert rules["avoid_potions"]["setting"] == "auto_avoid_potions"
    assert rules["seek_potions"]["setting"] == "auto_seek_potions"


def test_node_potion_classification_and_pathfinding():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    subprocess.run([node, str(ROOT / "tests/fixtures/potion_bridge_harness.js"),
                    str(ROOT / "pymc_bot/node/minecraft_bridge.js")], check=True, timeout=15)
