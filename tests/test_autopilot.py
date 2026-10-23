"""Autopilot: survival reflexes (pure rules + the keeper running them on simulated bots)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pymc_bot import autopilot as ap
from pymc_bot.bot import MinecraftBot
from pymc_bot.config import AutopilotSettings, ConfigStore


def snap(**extra):
    base = {
        "health": 20.0, "food": 20.0, "world_time": 1000, "dimension": "overworld", "oxygen": 20, "in_water": False,
        "on_fire": False, "dead": False, "sleeping": False, "block_light": 15, "sky_light": 15,
        "position": {"x": 0.0, "y": 64.0, "z": 0.0}, "entities": [], "ground_items": [],
        "slots": [None] * 36, "selected_slot": 0,
    }
    base.update(extra)
    return base


def slots(**items):
    out = [None] * 36
    for index, (name, count) in enumerate(items.items()):
        out[index] = {"name": name, "count": count}
    return out


def mob(kind, dist, hostile=True):
    return {"type": f"minecraft:{kind}", "dist": dist, "dx": dist, "dz": 0.0, "hostile": hostile, "player": False}


def plan(s, settings=None, **situation):
    return ap.evaluate(settings or AutopilotSettings(), ap.Situation(snap=s, **situation))


def test_nothing_to_do_on_a_calm_day():
    assert plan(snap()) is None
    assert plan(snap(dead=True), AutopilotSettings(enabled=False)) is None


@pytest.mark.parametrize(
    ("state", "rule"),
    [
        (snap(dead=True), "respawn"),
        (snap(in_water=True, oxygen=4), "surface"),
        (snap(on_fire=True), "extinguish"),
        (snap(health=8, slots=slots(totem_of_undying=1)), "totem"),
        (snap(health=4, entities=[mob("zombie", 3)]), "flee"),
        (snap(entities=[mob("creeper", 3)]), "flee"),
        (snap(food=10, slots=slots(bread=3)), "eat"),
        (snap(slots=slots(iron_helmet=1)), "armor"),
        (snap(slots=slots(dirt=4, iron_sword=1), entities=[mob("zombie", 9)]), "weapon"),
        (snap(slots=slots(iron_sword=1), entities=[mob("zombie", 3)]), "defend"),
        (snap(world_time=14000), "sleep"),
        (snap(sky_light=0, block_light=2, slots=slots(torch=5)), "torch"),
        (snap(ground_items=[{"item": "minecraft:oak_log", "count": 1, "dist": 3}]), "collect"),
    ],
)
def test_each_rule_fires(state, rule):
    result = plan(state)
    assert result is not None and result.rule == rule, result


def test_priorities_and_busy_bots():
    # drowning beats hunger, and urgent rules still run while the bot is busy
    both = snap(in_water=True, oxygen=3, food=5, slots=slots(bread=2))
    assert plan(both).rule == "surface"
    assert plan(both, busy=True).rule == "surface"
    # chores wait until the bot is idle; a nearly-full hunger bar is not urgent while busy
    assert plan(snap(world_time=14000), busy=True) is None
    assert plan(snap(food=12, slots=slots(bread=2)), busy=True) is None
    assert plan(snap(food=5, slots=slots(bread=2)), busy=True).rule == "eat"


def test_rules_that_hold_back():
    assert plan(snap(world_time=14000, entities=[mob("zombie", 6)], slots=slots(iron_sword=1)),
                AutopilotSettings(auto_defend=False, auto_weapon=False)) is None  # monsters close: no bed
    assert plan(snap(world_time=14000, dimension="minecraft:the_nether")) is None  # beds explode there
    assert plan(snap(entities=[mob("enderman", 2)])) is None  # never start a fight with neutral mobs
    assert plan(snap(food=5)) is None  # hungry but nothing to eat
    assert plan(snap(dead=True), cooldowns={"respawn": 50.0}, now=10.0) is None  # cooling down
    assert plan(snap(sky_light=15, block_light=0, slots=slots(torch=5))) is None  # outside, daylight


def test_shelter_home_tidy_and_idle_job():
    night = dict(world_time=15000, entities=[mob("zombie", 12)], slots=slots(cobblestone=64))
    settings = AutopilotSettings(auto_shelter=True, auto_defend=False, auto_weapon=False)
    assert plan(snap(**night), settings).rule == "sleep"  # the bed comes first
    assert plan(snap(**night), settings, failures={"sleep": 95.0}, cooldowns={"sleep": 200.0}, now=100.0).rule == "shelter"

    home = {"x": 100.0, "y": 64.0, "z": 0.0, "dimension": "overworld"}
    assert plan(snap(world_time=15000), AutopilotSettings(auto_home=True, auto_sleep=False), home=home).rule == "home"

    full = [{"name": "rotten_flesh", "count": 3}] + [{"name": "stone", "count": 64}] * 35
    tidy = plan(snap(slots=full))
    assert tidy.rule == "tidy" and tidy.action == "skill:toss_junk"
    tidy = plan(snap(slots=[{"name": "stone", "count": 64}] * 36))
    assert tidy.action == "skill:deposit"

    farm = AutopilotSettings(idle_job="farm")
    assert plan(snap(), farm, idle_seconds=120).action == "skill:harvest"
    assert plan(snap(), farm, idle_seconds=10) is None
    assert plan(snap(), farm, idle_seconds=120, ai_busy=True) is None  # the AI is in charge


def test_bad_idle_job_is_rejected():
    with pytest.raises(ValueError):
        AutopilotSettings(idle_job="dance")


def test_keeper_runs_plans_on_a_bot(live_bot: MinecraftBot, store: ConfigStore, log):
    keeper = ap.Autopilot(lambda: [live_bot], store, log)
    assert not keeper.start()  # PYMC_BOT_NO_AUTOPILOT=1 in the tests
    first = keeper.tick(wait=True)
    assert first == [(live_bot.username, "armor")]
    assert live_bot.snapshot()["armor_slots"]["head"]["name"] == "leather_helmet"
    memory = keeper.memory(live_bot)
    assert memory.counts["armor"] == 1 and memory.last_rule == "armor"

    keeper.pause(live_bot)
    assert keeper.tick(wait=True) == []
    keeper.pause(live_bot, False)

    status = keeper.status([live_bot])
    assert [rule["name"] for rule in status["rules"]] == list(ap.RULES)
    assert status["bots"][live_bot.username]["counts"]["armor"] == 1


def test_failed_rule_waits_longer(live_bot: MinecraftBot, store: ConfigStore, log):
    keeper = ap.Autopilot(lambda: [live_bot], store, log)
    result = ap.Plan("sleep", "sleep_in_bed", cooldown=1.0, fail_cooldown=90.0)
    ok, _ = keeper.run(live_bot, result)  # daytime: the bed refuses
    assert not ok
    memory = keeper.memory(live_bot)
    assert "sleep" in memory.failures and memory.cooldowns["sleep"] > memory.failures["sleep"] + 80


def test_autopilot_api(tmp_path):
    from pymc_bot.server import create_app

    with TestClient(create_app(config_path=tmp_path / "cfg.json")) as client:
        status = client.get("/api/autopilot/status").json()
        assert status["enabled"] and status["rule_order"][0] == "respawn" and status["settings"]["auto_sleep"]
        client.put("/api/config", json={"minecraft": {"backend": "simulated"}, "autopilot": {"idle_job": "chop"}})
        assert client.get("/api/autopilot/status").json()["idle_job"] == "chop"
        client.post("/api/bot/start")
        for _ in range(100):
            if client.get("/api/status").json().get("state") == "connected":
                break
            __import__("time").sleep(0.05)
        reply = client.post("/api/autopilot/run", json={}).json()
        assert reply["ok"] and reply["results"][0]["plan"]["rule"] == "armor"
        reply = client.post("/api/autopilot/run", json={"rule": "sleep"}).json()
        assert not reply["ok"]
        assert client.post("/api/autopilot/run", json={"rule": "dance"}).status_code == 422
        assert client.post("/api/autopilot/run", json={"paused": True}).json()["results"][0]["paused"] is True
