"""Aternos & other free hosts: addresses, sleeping servers, kick hints, leave-when-alone."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from pymc_bot import hosting
from pymc_bot.backends import (
    BackendError,
    NodeBridgeBackend,
    SimulatedBackend,
    explain_connection_error,
)
from pymc_bot.bot import MinecraftBot
from pymc_bot.config import AppSettings, ConfigStore, MinecraftSettings
from pymc_bot.events import EventLog
from pymc_bot.hosting import (
    AloneWatcher,
    describe_probe,
    detect_host,
    host_tips,
    leave_rule_active,
    split_address,
)

FAKE_BRIDGE = Path(__file__).parent / "fixtures" / "fake_bridge.js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is required for the bridge protocol tests")


# ------------------------------------------------------------------ addresses
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("name.aternos.me", ("name.aternos.me", None)),
        ("name.aternos.me:12345", ("name.aternos.me", 12345)),
        ("  abc.aternos.host:25570 ", ("abc.aternos.host", 25570)),
        ("minecraft://play.example.net:25566/", ("play.example.net", 25566)),
        ("[::1]:25565", ("::1", 25565)),
        ("[2001:db8::1]", ("2001:db8::1", None)),
        ("2001:db8::1", ("2001:db8::1", None)),
        ("host:notaport", ("host:notaport", None)),
    ],
)
def test_split_address(text: str, expected: tuple[str, int | None]):
    assert split_address(text) == expected


def test_config_splits_host_and_port():
    settings = MinecraftSettings(host="name.aternos.me:12345")
    assert (settings.host, settings.port) == ("name.aternos.me", 12345)
    assert MinecraftSettings(host="name.aternos.me").port == 25565  # SRV record does the rest
    assert MinecraftSettings(host="[::1]:25570").host == "::1"


def test_config_store_update_splits_too(tmp_path: Path):
    store = ConfigStore(tmp_path / "cfg.json")
    store.load()
    store.update({"minecraft": {"host": "0.tcp.eu.ngrok.io:14567"}})
    assert store.settings.minecraft.host == "0.tcp.eu.ngrok.io"
    assert store.settings.minecraft.port == 14567


@pytest.mark.parametrize(
    ("host", "key"),
    [
        ("MyWorld.Aternos.me", "aternos"),
        ("abc.aternos.host:25570", "aternos"),
        ("srv.exaroton.me", "exaroton"),
        ("cool.minehut.gg", "minehut"),
        ("x.falixsrv.me", "falix"),
        ("happy-cat.joinmc.link", "playit"),
        ("0.tcp.eu.ngrok.io", "ngrok"),
        ("abc.e4mc.link", "e4mc"),
    ],
)
def test_detect_host(host: str, key: str):
    profile = detect_host(host)
    assert profile is not None and profile.key == key


def test_detect_host_unknown_and_by_motd():
    assert detect_host("play.example.net") is None
    assert detect_host("notaternos.me") is None
    assert detect_host("185.107.192.48", "This server is offline. Powered by aternos.org").key == "aternos"


def test_host_tips_mention_cracked_mode_only_for_offline_bots():
    offline = host_tips("name.aternos.me", "offline")
    assert offline[0] == "Aternos detected."
    assert any("Cracked" in tip for tip in offline)
    assert any("empty server" in tip for tip in offline)
    assert not any("Cracked" in tip for tip in host_tips("name.aternos.me", "microsoft"))
    assert host_tips("play.example.net") == []


def test_leave_rule_is_on_for_aternos_and_optional_elsewhere():
    settings = AppSettings()
    settings.minecraft = settings.minecraft.model_copy(update={"host": "name.aternos.me"})
    active, profile = leave_rule_active(settings)
    assert active and profile is not None and profile.key == "aternos"
    settings.minecraft = settings.minecraft.model_copy(update={"host": "play.example.net"})
    assert leave_rule_active(settings)[0] is False
    settings.minecraft = settings.minecraft.model_copy(update={"leave_when_alone": "always"})
    assert leave_rule_active(settings)[0] is True


# ---------------------------------------------------------------------- probe
def test_describe_probe_states():
    aternos = detect_host("name.aternos.me").as_dict()
    asleep = {"ok": True, "asleep": True, "state": "offline", "motd": "This server is offline.", "host": "name.aternos.me",
              "port": 25565, "hosting": aternos}
    text = describe_probe(asleep)
    assert "name.aternos.me:25565 (Aternos) is offline" in text and "Aternos dashboard" in text
    online = {"ok": True, "online": True, "asleep": False, "version": "Paper 1.21.4", "protocol": 769,
              "players": {"online": 2, "max": 20}, "latency": 87, "host": "name.aternos.me", "port": 25565,
              "resolved": {"host": "abc.aternos.host", "port": 25570}, "hosting": aternos}
    text = describe_probe(online)
    assert "is online via abc.aternos.host:25570" in text and "Paper 1.21.4" in text and "2/20 players" in text
    failed = {"ok": False, "error": "connect ECONNREFUSED", "host": "x", "port": 1, "hosting": None}
    assert describe_probe(failed) == "x:1 did not answer: connect ECONNREFUSED"


def test_probe_without_node_reports_it(monkeypatch):
    monkeypatch.setattr(hosting.shutil, "which", lambda name: None)
    result = hosting.probe_server("name.aternos.me:12345")
    assert result["ok"] is False and "Node.js" in result["error"]
    assert (result["host"], result["port"]) == ("name.aternos.me", 12345)
    assert result["hosting"]["key"] == "aternos"


@needs_node
def test_probe_of_a_closed_port():
    result = hosting.probe_server("127.0.0.1", 1, timeout=3.0)
    assert result["ok"] is False
    assert result["error"]


@needs_node
def test_probe_placeholder_detection_matches_the_js(tmp_path: Path):
    """The JS helper that recognises sleeping-server placeholders, run through node."""
    import json
    import subprocess

    script = (
        "const { sleepingState } = require(" + json.dumps(str(hosting.PROBE_SCRIPT)) + ");"
        "const cases = [['\\u00a7c\\u25cf Offline', -1, 'This server is offline. Powered by aternos.org'],"
        "['\\u25cf Starting', 0, 'This server is starting'], ['Paper 1.21.4', 769, 'A Minecraft Server'],"
        "['Velocity 3.4.0', 775, 'Lobby'], ['exaroton', -1, 'This server is currently offline.'],"
        "['Paper 1.21.4', 769, 'we are offline-mode friendly']];"
        "console.log(JSON.stringify(cases.map((c) => sleepingState(...c))));"
    )
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=20, check=True)
    assert json.loads(out.stdout) == ["offline", "starting", None, None, "offline", None]


# ---------------------------------------------------------------- kick hints
def test_asleep_and_host_specific_explanations():
    text = explain_connection_error("The server is offline", host="name.aternos.me", port=25565,
                                    asleep={"state": "offline", "motd": "This server is offline."})
    assert "name.aternos.me:25565 (Aternos) is offline" in text and "Wait for the server" in text

    kicked = explain_connection_error('{"translate":"multiplayer.disconnect.unverified_username"}',
                                      host="name.aternos.me", port=25565)
    assert "turn on Cracked" in kicked

    generic = explain_connection_error("Failed to verify username!", host="play.example.net", port=25565)
    assert "online-mode=false" in generic

    assert "Players > Whitelist" in explain_connection_error(
        "You are not white-listed on this server!", host="name.aternos.me", port=25565)
    assert "/whitelist add" in explain_connection_error("You are not whitelisted on this server!")
    assert "Gap between joins" in explain_connection_error("Connection throttled! Please wait before reconnecting.")
    assert "same name" in explain_connection_error("You logged in from another location")
    assert "starting" in explain_connection_error("This server is starting. Please wait.", host="a.exaroton.me")


@needs_node
def test_bridge_reports_a_sleeping_aternos_server(log: EventLog):
    backend = NodeBridgeBackend(log, node_binary=NODE, script=FAKE_BRIDGE)
    backend.start()
    try:
        settings = AppSettings().minecraft.model_copy(update={"host": "sleepy.aternos.me"})
        with pytest.raises(BackendError) as excinfo:
            backend.connect(settings)
        message = str(excinfo.value)
        assert "(Aternos) is offline" in message and "Aternos dashboard" in message
        assert backend.server_asleep["state"] == "offline"
        assert backend.snapshot()["server_state"] == "offline"

        with pytest.raises(BackendError) as excinfo:
            backend.connect(settings.model_copy(update={"host": "cracked.aternos.me"}))
        assert "turn on Cracked" in str(excinfo.value)
        assert backend.server_asleep is None
    finally:
        backend.stop()


# ------------------------------------------------------------ wait for server
class SleepyBackend(SimulatedBackend):
    """Simulated world whose first ``connect`` finds a sleeping free-host server."""

    def __init__(self, log: EventLog, sleeps: int = 1) -> None:
        super().__init__(log, ambient_chat=False)
        self.sleeps = sleeps
        self.server_asleep: dict[str, Any] | None = None

    def connect(self, settings: MinecraftSettings) -> None:
        if self.sleeps > 0:
            self.sleeps -= 1
            self.server_asleep = {"state": "offline", "motd": "This server is offline."}
            self.stop()
            raise BackendError("name.aternos.me:25565 (Aternos) is offline.")
        self.server_asleep = None


def _store(tmp_path: Path, **minecraft: Any) -> ConfigStore:
    store = ConfigStore(tmp_path / "cfg.json")
    store.load()
    store.update({"minecraft": {"host": "name.aternos.me", "backend": "simulated", **minecraft}})
    return store


def test_bot_waits_for_a_sleeping_server_then_joins(tmp_path: Path, log: EventLog, monkeypatch):
    probes: list[str] = []

    def fake_probe(host: str, port: int = 25565, timeout: float = 8.0) -> dict[str, Any]:
        probes.append(host)
        return {"ok": True, "online": len(probes) >= 2, "asleep": len(probes) < 2}

    monkeypatch.setattr(hosting, "probe_server", fake_probe)
    store = _store(tmp_path, wait_for_server=True, wait_minutes=1)
    bot = MinecraftBot(store, log, backend=SleepyBackend(log))
    bot.wait_poll_seconds = 0.1
    joined: list[bool] = []
    bot.on_connected = lambda: joined.append(True)
    try:
        bot.start()
        assert bot.state == "waiting"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not bot.connected:
            time.sleep(0.05)
        assert bot.connected and joined == [True]
        assert len(probes) == 2
        messages = [entry["message"] for entry in log.tail(50)]
        assert any("never starts the server" in message for message in messages)
        assert any("online now - joining" in message for message in messages)
    finally:
        bot.stop()


def test_bot_reports_a_sleeping_server_without_wait(tmp_path: Path, log: EventLog):
    store = _store(tmp_path)
    bot = MinecraftBot(store, log, backend=SleepyBackend(log))
    with pytest.raises(BackendError):
        bot.start()
    assert bot.state == "error"


def test_stop_cancels_waiting(tmp_path: Path, log: EventLog, monkeypatch):
    monkeypatch.setattr(hosting, "probe_server", lambda *a, **k: {"ok": True, "online": False, "asleep": True})
    store = _store(tmp_path, wait_for_server=True)
    bot = MinecraftBot(store, log, backend=SleepyBackend(log, sleeps=99))
    bot.wait_poll_seconds = 0.1
    bot.start()
    assert bot.state == "waiting"
    bot.stop()
    assert bot.state == "disconnected"
    time.sleep(0.3)
    assert bot.state == "disconnected"


# ------------------------------------------------------------ leave when alone
class FakeBot:
    def __init__(self, username: str, online: list[str] | None, connected: bool = True) -> None:
        self.username = username
        self.connected = connected
        self.stopped = False
        snapshot = {"players": [{"name": name} for name in (online or [])]}
        if online is not None:
            snapshot["online_players"] = online
        self.backend = type("B", (), {"snapshot": lambda _self: snapshot})()

    def stop(self) -> None:
        self.stopped = True
        self.connected = False


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_alone_watcher_leaves_aternos_when_only_bots_are_online(tmp_path: Path, log: EventLog):
    store = _store(tmp_path, alone_minutes=5)
    bots = [FakeBot("Bot_1", ["Bot_2"]), FakeBot("Bot_2", ["Bot_1"])]
    clock = Clock()
    watcher = AloneWatcher(lambda: bots, store, log, clock=clock)
    assert watcher.check() is False  # starts the timer
    clock.now += 4 * 60
    assert watcher.check() is False and watcher.alone_for == pytest.approx(240)
    clock.now += 61
    assert watcher.check() is True
    assert all(bot.stopped for bot in bots)
    assert any("Aternos does not allow bots" in entry["message"] for entry in log.tail(5))


def test_alone_watcher_stays_while_a_human_plays(tmp_path: Path, log: EventLog):
    store = _store(tmp_path, alone_minutes=1)
    bots = [FakeBot("Bot_1", ["Bot_2", "Steve"]), FakeBot("Bot_2", ["Bot_1", "Steve"])]
    clock = Clock()
    left: list[bool] = []
    watcher = AloneWatcher(lambda: bots, store, log, clock=clock, leave=lambda: left.append(True))
    for _ in range(5):
        assert watcher.check() is False
        clock.now += 60
    assert not left and watcher.alone_for == 0.0


def test_alone_watcher_is_off_on_normal_servers(tmp_path: Path, log: EventLog):
    store = _store(tmp_path, host="play.example.net", alone_minutes=1)
    bots = [FakeBot("Bot_1", [])]
    clock = Clock()
    watcher = AloneWatcher(lambda: bots, store, log, clock=clock)
    watcher.check()
    clock.now += 600
    assert watcher.check() is False and not bots[0].stopped
    store.update({"minecraft": {"leave_when_alone": "always"}})
    watcher.check()
    clock.now += 61
    assert watcher.check() is True and bots[0].stopped


def test_alone_watcher_uses_players_in_view_from_older_bridges(tmp_path: Path, log: EventLog):
    store = _store(tmp_path, alone_minutes=1)
    bots = [FakeBot("Bot_1", None)]
    bots[0].backend.snapshot = lambda: {"players": [{"name": "Alex"}]}
    clock = Clock()
    watcher = AloneWatcher(lambda: bots, store, log, clock=clock)
    watcher.check()
    clock.now += 120
    assert watcher.check() is False


def test_simulated_backend_lists_online_players(log: EventLog):
    backend = SimulatedBackend(log, ambient_chat=False)
    backend.start()
    try:
        snapshot = backend.snapshot()
        assert snapshot["online_players"] and set(snapshot["online_players"]) >= {p["name"] for p in snapshot["players"]}
    finally:
        backend.stop()


# ----------------------------------------------------------------------- API
def test_probe_api_and_hosting_status(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient

    from pymc_bot.server import create_app

    seen: dict[str, Any] = {}

    def fake_probe(host: str, port: int = 25565, timeout: float = 8.0) -> dict[str, Any]:
        seen.update(host=host, port=port)
        name, inline = split_address(host)
        return {"ok": True, "asleep": True, "online": False, "state": "starting", "motd": "", "host": name,
                "port": inline or port, "hosting": detect_host(name).as_dict()}

    monkeypatch.setattr(hosting, "probe_server", fake_probe)
    with TestClient(create_app(config_path=tmp_path / "cfg.json")) as client:
        client.put("/api/config", json={"minecraft": {"host": "name.aternos.me:12345"}})
        config = client.get("/api/config").json()
        minecraft = config.get("config", config)["minecraft"]
        assert (minecraft["host"], minecraft["port"]) == ("name.aternos.me", 12345)

        result = client.get("/api/server/probe").json()
        assert seen == {"host": "name.aternos.me", "port": 12345}
        assert "is starting" in result["summary"] and result["tips"][0] == "Aternos detected."

        client.get("/api/server/probe", params={"host": "other.exaroton.me"})
        assert seen == {"host": "other.exaroton.me", "port": 25565}

        status = client.get("/api/status").json()
        assert status["hosting"]["provider"]["key"] == "aternos"
        assert status["hosting"]["leave_when_alone"] is True
