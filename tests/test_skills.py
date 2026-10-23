"""The 44 skills: catalogue, parameter cleaning, the simulated world, AI gating, places and the API."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from pymc_bot import skills
from pymc_bot.agent import ACTIONS, AgentLoop, Decision
from pymc_bot.backends import BaseBackend
from pymc_bot.bot import MinecraftBot
from pymc_bot.config import ConfigStore
from pymc_bot.mind import build_system_prompt

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_catalogue_has_more_than_forty_skills_in_every_group():
    assert len(skills.SKILLS) >= 40
    names = [s.name for s in skills.SKILLS]
    assert len(names) == len(set(names))
    assert {s.group for s in skills.SKILLS} == set(skills.GROUPS)
    for skill in skills.SKILLS:
        assert skill.summary and skill.permission in (*skills.PERMISSION_FLAGS, "none")
        assert (skill.bridge is None) == (skill.name not in skills.SKILL_COMMANDS)
        assert skills.get(skill.name) is skill
        for alias in skill.aliases:
            assert skills.get(alias) is skill
    assert len(skills.catalogue()) == len(skills.SKILLS)


def test_every_bridge_skill_is_a_backend_and_javascript_command():
    js = (ROOT / "pymc_bot" / "node" / "minecraft_bridge.js").read_text(encoding="utf-8")
    for command in skills.SKILL_COMMANDS:
        assert command in BaseBackend.LIFE_COMMANDS
        assert f"  {command}: cmd" in js, command


def test_clean_params_coerces_clamps_and_reads_synonyms():
    chop = skills.get("chop_tree")
    assert skills.clean_params(chop, {"count": "9999"}) == {"count": 64, "radius": 32}
    assert skills.clean_params(chop, {"amount": 3, "radius": "x"}) == {"count": 3, "radius": 32}
    craft = skills.get("craft")
    assert skills.clean_params(craft, {"name": "Minecraft:Oak Planks", "n": "4"})["item"] == "oak_planks"
    harvest = skills.get("harvest")
    assert skills.clean_params(harvest, {"replant": "no"})["replant"] is False


def test_skills_in_the_simulated_world(live_bot: MinecraftBot):
    ok, message = live_bot.skill("chop_tree")
    assert ok and "log" in message
    assert live_bot.stats["skills_used"] == 1 and live_bot.stats["blocks_mined"] >= 1
    ok, message = live_bot.skill("craft", {"item": "oak_planks", "count": 4})
    assert ok and "planks" in message
    assert live_bot.inventory().count("oak_planks") >= 4
    assert live_bot.skill("mine_ore", {"ore": "iron", "count": 2})[0]
    assert live_bot.skill("place_torch")[0]
    assert live_bot.skill("equip_weapon") == (True, "holding iron_sword")
    ok, message = live_bot.skill("fish")
    assert not ok and "fishing rod" in message
    ok, message = live_bot.skill("report")
    assert ok and "hp 20/20" in message
    assert live_bot.skill("nope") == (False, "unknown skill 'nope'")


def test_homes_and_waypoints_are_remembered_per_server(live_bot: MinecraftBot, store: ConfigStore):
    assert live_bot.skill("set_home")[0]
    assert live_bot.skill("set_waypoint", {"name": "Mine"})[0]
    data = json.loads((store.path.parent / "pymc_bot_places.json").read_text())
    server = next(iter(data))
    assert set(data[server]["testbot"]) == {"home", "mine"}
    ok, listing = live_bot.skill("waypoints")
    assert ok and "home" in listing and "mine" in listing
    ok, message = live_bot.skill("goto_waypoint", {"name": "nowhere"})
    assert not ok and "nowhere" in message


def test_agent_runs_skills_and_respects_the_switches(live_bot: MinecraftBot, store: ConfigStore, log):
    for skill in skills.SKILLS:
        assert skill.name in ACTIONS
    agent = AgentLoop(live_bot, store, log)
    ok, detail = agent.execute(Decision("chop_tree", {"count": 2}, source="test"))
    assert ok, detail
    store.update({"agent": {"allow_building": False}})
    ok, detail = agent.execute(Decision("build_shelter", {}, source="test"))
    assert not ok and "building" in detail
    ok, detail = agent.execute(Decision("shoot", {"mob": "villager"}, source="test"))
    assert not ok


def test_prompt_lists_only_allowed_skills(store: ConfigStore):
    text = skills.prompt_reference({"building": False})
    assert "chop_tree" in text and "build_shelter" not in text
    assert "build_shelter" in skills.prompt_reference({"building": True})
    prompt = build_system_prompt(store.settings.agent)
    assert "chop_tree" in prompt


def test_skills_api(tmp_path):
    from pymc_bot.server import create_app

    with TestClient(create_app(config_path=tmp_path / "cfg.json")) as client:
        catalogue = client.get("/api/skills").json()
        assert catalogue["count"] >= 40 and catalogue["groups"][0] == "survival"
        client.put("/api/config", json={"minecraft": {"backend": "simulated"}})
        client.post("/api/bot/start")
        for _ in range(100):
            if client.get("/api/status").json().get("state") == "connected":
                break
            __import__("time").sleep(0.05)
        reply = client.post("/api/bot/skill", json={"name": "chop_tree", "params": {"count": 3}}).json()
        assert reply["ok"], reply
        assert client.post("/api/bot/skill", json={"name": "nope"}).status_code == 404
        status = client.get("/api/status").json()
        assert "autopilot" in status and "bridge_setup" in status
