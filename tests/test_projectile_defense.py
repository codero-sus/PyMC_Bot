"""Projectile warning and shield reflex: prediction, priorities and opt-out."""

from __future__ import annotations

import math

import pytest

from pymc_bot.autopilot import Autopilot, Situation, evaluate, incoming_projectile
from pymc_bot.config import AutopilotSettings


def projectile(**changes):
    return {"type": "minecraft:arrow", "dx": 5, "dy": 0.9, "dz": 0,
            "vx": -0.5, "vy": 0, "vz": 0, **changes}


def situation(entity=None, *, offhand="shield", busy=False, **changes):
    return Situation(snap={"health": 20, "food": 20, "offhand": offhand,
                           "entities": [entity or projectile()], **changes}, busy=busy)


def test_projectile_prediction_is_bounded_and_requires_known_velocity():
    assert incoming_projectile(projectile()) == pytest.approx(10)
    assert incoming_projectile(projectile(vx=0.5)) is None
    assert incoming_projectile(projectile(vx=0)) is None
    assert incoming_projectile(projectile(dz=5)) is None
    assert incoming_projectile(projectile(dx=20)) is None
    assert incoming_projectile(projectile(vx=math.nan)) is None
    assert incoming_projectile(projectile(vx=None)) is None
    assert incoming_projectile(projectile(type="minecraft:zombie")) is None


def test_autopilot_only_uses_existing_shield_and_can_be_disabled():
    settings = AutopilotSettings()
    plan = evaluate(settings, situation(busy=True))
    assert plan is not None and plan.rule == "block_projectile" and plan.interrupt
    assert evaluate(settings, situation(offhand="totem_of_undying")) is None
    assert evaluate(settings, situation(offhand="")) is None
    assert evaluate(AutopilotSettings(auto_block_projectiles=False), situation()) is None
    assert evaluate(settings, situation(projectile(vx=0.5))) is None


def test_emergency_survival_has_priority_over_projectile():
    result = evaluate(AutopilotSettings(), situation(dead=True))
    assert result.rule == "respawn"
    result = evaluate(AutopilotSettings(), situation(in_water=True, oxygen=2))
    assert result.rule == "surface"


def test_reflex_is_exposed_in_status(store):
    status = Autopilot(lambda: [], store).status([])
    rule = next(item for item in status["rules"] if item["name"] == "block_projectile")
    assert rule["setting"] == "auto_block_projectiles" and rule["enabled"] and rule["emergency"]
