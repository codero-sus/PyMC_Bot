"""The mod's ``/pymcbot bot ...`` commands call the panel's HTTP API (mod/.../BotLink.java).

No Java compiler here, so this checks the contract from the Python side: every endpoint the mod calls
exists with that method, and the exact bodies the commands send work against a simulated bot.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pymc_bot import autopilot, skills
from pymc_bot.server import create_app

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "mod/src/client/java/dev/pymc/pymcb/client"
COMMANDS = (CLIENT / "PymcCommands.java").read_text(encoding="utf-8")
BOTLINK = (CLIENT / "BotLink.java").read_text(encoding="utf-8")


def mod_calls() -> set[tuple[str, str]]:
    calls = re.findall(r'BotLink\.(get|post|put)\("(/api/[^"]+)"', COMMANDS)
    return {(method.upper(), path) for method, path in calls}


def test_the_mod_only_calls_endpoints_the_panel_has(tmp_path: Path):
    calls = mod_calls()
    assert {("GET", "/api/status"), ("GET", "/api/fleet/status"), ("GET", "/api/skills"),
            ("POST", "/api/bot/skill"), ("POST", "/api/bot/action"), ("POST", "/api/bot/chat"),
            ("PUT", "/api/config"), ("GET", "/api/autopilot/status"),
            ("POST", "/api/ai/start"), ("POST", "/api/ai/stop")} <= calls
    app = create_app(config_path=tmp_path / "cfg.json")
    routes = {(method, route.path) for route in app.routes for method in getattr(route, "methods", None) or ()}
    missing = sorted(call for call in calls if call not in routes)
    assert not missing, f"the mod calls endpoints the panel does not have: {missing}"


def test_the_mod_stays_portable():
    """HttpURLConnection only: java.net.http and Gson are not safe across 1.21.1 - 26.3."""
    for source in (COMMANDS, BOTLINK):
        assert "java.net.http" not in source and "com.google.gson" not in source
    assert "setConnectTimeout" in BOTLINK and "setReadTimeout" in BOTLINK
    assert 'Thread(runnable, "pymcbot-botlink")' in BOTLINK and "setDaemon(true)" in BOTLINK
    # the reply is printed on the game thread
    assert "client.execute(" in BOTLINK
    # every command that makes a bot do something needs an attended (typed) run on 26.2+
    for word in ("do", "come", "home", "follow", "stop", "say", "on", "off", "start"):
        assert f'guarded("{word}")' in COMMANDS, word
    assert "panel <url>" in COMMANDS or "panel [url]" in COMMANDS


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(config_path=tmp_path / "cfg.json")
    with TestClient(app) as test_client:
        test_client.put("/api/config", json={"minecraft": {"backend": "simulated", "username": "ModBot"}})
        assert test_client.post("/api/bot/start").status_code == 200
        for _ in range(50):
            if test_client.get("/api/status").json()["bot"]["state"] == "connected":
                break
            time.sleep(0.1)
        yield test_client


def test_the_bodies_the_commands_send_work(client: TestClient):
    # /pymcbot bot do chop_tree count=2  ->  BotLink.options gives {"count":2}
    reply = client.post("/api/bot/skill", json={"name": "chop_tree", "params": {"count": 2}})
    assert reply.status_code == 200 and reply.json()["ok"], reply.json()
    assert "detail" in reply.json()
    # /pymcbot bot do mine_ore iron  ->  a bare word is the "name" option
    reply = client.post("/api/bot/skill", json={"name": "mine_ore", "params": {"name": "iron"}})
    assert reply.json()["ok"], reply.json()
    # /pymcbot bot come
    assert client.post("/api/bot/skill", json={"name": "come", "params": {"player": "Steve"}}).json()["ok"]
    # /pymcbot bot follow | stop
    assert client.post("/api/bot/action", json={"action": "follow", "player": "Steve"}).json()["ok"]
    assert client.post("/api/bot/action", json={"action": "stop"}).json()["ok"]
    # /pymcbot bot say hi
    assert client.post("/api/bot/chat", json={"message": "hi"}).json()["ok"]
    # /pymcbot bot auto off | on
    reply = client.put("/api/config", json={"autopilot": {"enabled": False}})
    assert reply.status_code == 200 and reply.json()["config"]["autopilot"]["enabled"] is False
    assert client.put("/api/config", json={"autopilot": {"enabled": True}}).json()["config"]["autopilot"]["enabled"]
    # an unknown skill: the mod shows the panel's "detail"
    reply = client.post("/api/bot/skill", json={"name": "nope", "params": {}})
    assert reply.status_code == 404 and "unknown skill" in reply.json()["detail"]


def test_the_fields_the_mod_reads_are_there(client: TestClient):
    status = client.get("/api/status").json()
    for key in ("username", "state", "health", "food", "position"):
        assert key in status["bot"], key
    assert "running" in status["agent"]
    assert "enabled" in status["autopilot"] and "last_rule" in status["autopilot"]["bot"]
    assert "connected" in status["fleet"]
    fleet = client.get("/api/fleet/status").json()
    assert {"username", "state", "health", "selected"} <= set(fleet["bots"][0])
    listed = client.get("/api/skills").json()
    assert len(listed["skills"]) == len(skills.SKILLS) >= 44
    assert {"name", "group"} <= set(listed["skills"][0])
    auto = client.get("/api/autopilot/status").json()
    assert {"enabled", "actions", "idle_job", "next"} <= set(auto)
    assert auto["next"] is None or auto["next"]["rule"] in autopilot.RULES
    assert {"rule", "reason"} <= set(auto["next"] or {"rule": 1, "reason": 1})
