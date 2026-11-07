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


def test_shield_preserves_totem_at_low_health(live_bot):
    from pymc_bot.inventory import Inventory, Stack
    from pymc_bot.skills import get, run_skill

    inv = Inventory()
    inv.offhand = Stack("totem_of_undying", 1)
    inv.slots[0] = Stack("shield", 1)
    live_bot.inventory = lambda: inv
    live_bot.snapshot = lambda: {"health": 5}
    calls = []
    live_bot._perform = lambda command, params, **kwargs: (calls.append((command, params)) or True, {})
    ok, detail = run_skill(live_bot, get("shield"), {})
    assert not ok and "keeping the totem" in detail and not calls
    assert run_skill(live_bot, get("shield"), {"force": True})[0]
    assert [call[0] for call in calls] == ["equip", "use_item"]


def test_weapon_skills_are_exposed_and_fail_honestly_in_simulator(live_bot):
    for name in ("throw_trident", "spear_attack", "throw_projectile", "mace_smash"):
        assert skills.get(name) is not None
        ok, detail = live_bot.skill(name, {"confirm": True})
        assert not ok and "not simulated" in detail


def test_smooth_look_interpolates_and_preserves_pitch_on_yaw_only(live_bot):
    values = []
    live_bot.backend.look = lambda yaw, pitch: values.append((yaw, pitch))
    ok, detail = live_bot.skill("smooth_look", {"yaw": 90, "pitch": 20, "seconds": 0.1})
    assert ok, detail
    assert len(values) >= 2
    assert abs(values[-1][0] - __import__("math").pi / 2) < 0.001
    assert abs(values[-1][1] - __import__("math").radians(20)) < 0.001


def test_risky_ai_combat_requires_opt_in(live_bot, store, log):
    agent = AgentLoop(live_bot, store, log)
    for name, params in (("throw_trident", {"confirm": True}), ("mace_smash", {"mode": "wind_charge", "confirm": True})):
        ok, reason = agent.execute(Decision(name, params, source="ollama"))
        assert not ok and "risky combat is disabled" in reason
    store.update({"agent": {"allow_risky_combat": True, "allow_building": False}})
    ok, reason = agent.execute(Decision("mace_smash", {"mode": "tower", "confirm": True}, source="ollama"))
    assert not ok and "building" in reason


def test_smooth_move_combines_controls_and_cleans_up(live_bot):
    values, controls = [], []
    live_bot.backend.look = lambda yaw, pitch: values.append((yaw, pitch))
    live_bot.backend.set_control = lambda name, state: controls.append((name, state))
    ok, detail = live_bot.skill("smooth_move", {"direction": "forward+left", "yaw": 90, "seconds": 0.1})
    assert ok, detail
    assert len(values) >= 2
    assert ("forward", True) in controls and ("left", True) in controls
    assert ("forward", False) in controls and ("left", False) in controls
    ok, _ = live_bot.skill("smooth_move", {"direction": "forward+back", "yaw": 90})
    assert not ok


def test_stop_cancels_running_and_queued_skill_but_allows_new_work(live_bot):
    import threading
    from contextlib import contextmanager

    first_started, release, second_queued = threading.Event(), threading.Event(), threading.Event()
    seen = []
    original_scope = live_bot.action_scope
    count = 0
    count_lock = threading.Lock()

    @contextmanager
    def scope():
        nonlocal count
        with original_scope():
            with count_lock:
                count += 1
                if count == 2:
                    second_queued.set()
            yield

    live_bot.action_scope = scope

    def perform(cmd, params, timeout):
        seen.append(cmd)
        if len(seen) == 1:
            first_started.set()
            release.wait(3)
        return True, "", {"message": "completed"}

    live_bot.backend.perform = perform
    results = []
    first = threading.Thread(target=lambda: results.append(live_bot.skill("chop_tree")))
    second = threading.Thread(target=lambda: results.append(live_bot.skill("chop_tree")))
    first.start()
    assert first_started.wait(2)
    second.start()
    assert second_queued.wait(2)
    assert live_bot.stop_moving()
    release.set()
    first.join(3)
    second.join(3)
    assert not first.is_alive() and not second.is_alive()
    assert len(seen) == 1 and len(results) == 2
    assert all(not ok and "cancelled" in detail for ok, detail in results)
    assert live_bot.skill("chop_tree")[0]
