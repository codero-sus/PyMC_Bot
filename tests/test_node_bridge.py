"""Python <-> Node bridge protocol tests.

These run against ``tests/fixtures/fake_bridge.js`` so the whole adapter can be
tested without mineflayer or a Minecraft server.  A test against the real
bridge (and a real server) lives in ``test_node_live.py`` and is opt-in.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from pymc_bot.backends import BackendError, NodeBridgeBackend, create_backend, node_available
from pymc_bot.config import AppSettings, ConfigStore  # noqa: F401
from pymc_bot.events import EventLog

FAKE_BRIDGE = Path(__file__).parent / "fixtures" / "fake_bridge.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is required for the bridge protocol tests")


@pytest.fixture()
def bridge(log: EventLog, wait_for):
    backend = NodeBridgeBackend(log, node_binary=NODE, script=FAKE_BRIDGE)
    backend.start()
    yield backend
    backend.stop()


def _settings(**overrides) -> AppSettings:
    settings = AppSettings()
    if overrides:
        settings.minecraft = settings.minecraft.model_copy(update=overrides)
    return settings.minecraft


def test_bridge_handshake(bridge: NodeBridgeBackend, log: EventLog, wait_for):
    assert bridge.capabilities["pathfinder"] is False
    assert bridge.supports_pathfinder is False

    def logged(text: str) -> bool:
        return any(text in event["message"] for event in log.tail(50))

    assert wait_for(lambda: logged("Bridge ready"), timeout=5)
    assert wait_for(lambda: logged("fake bridge ready"), timeout=5)


def test_connect_and_state_updates(bridge: NodeBridgeBackend, wait_for):
    bridge.connect(_settings(host="127.0.0.1", port=25565, username="ProtoBot"))
    assert wait_for(lambda: bridge.connected, timeout=5)
    assert wait_for(lambda: bridge.snapshot()["position"] is not None, timeout=5)

    assert wait_for(lambda: bridge.snapshot()["username"] == "ProtoBot", timeout=5)
    snapshot = bridge.snapshot()
    assert snapshot["status"] == "connected"
    assert snapshot["health"] == 19
    assert snapshot["server"]["host"] == "127.0.0.1"
    assert snapshot["server"]["username"] == "ProtoBot"
    assert snapshot["players"][0]["name"] == "Steve"
    assert snapshot["backend"] == "node"


def test_connect_error_is_reported(bridge: NodeBridgeBackend):
    with pytest.raises(BackendError) as excinfo:
        bridge.connect(_settings(host="badhost"))
    assert "badhost" in str(excinfo.value)
    assert bridge.connected is False
    assert bridge.snapshot()["status"] == "error"
    assert "badhost" in (bridge.snapshot()["last_error"] or "")


def test_primitives_reach_the_bridge(bridge: NodeBridgeBackend, wait_for):
    bridge.connect(_settings())
    assert wait_for(lambda: bridge.connected, timeout=5)

    bridge.say("bridge test")
    bridge.set_control("forward", True)
    bridge.look(1.0, -0.2)
    bridge.command("list")
    bridge.jump()
    assert bridge.swing_arm() is True
    assert bridge.dig("stone") is True
    assert bridge.attack("Steve") is True

    assert wait_for(lambda: bridge.snapshot()["position"]["z"] >= 1.0, timeout=5)
    assert wait_for(lambda: bridge.snapshot()["inventory"], timeout=5)
    assert bridge.snapshot()["yaw"] == 1.0
    assert bridge.snapshot()["inventory"][0]["name"] == "stone"

    bridge.stop_motion()
    time.sleep(0.3)


def test_chat_events_are_forwarded(bridge: NodeBridgeBackend, wait_for):
    bridge.connect(_settings())
    assert wait_for(lambda: bridge.connected, timeout=5)
    received: list[tuple[str, str]] = []
    bridge.on_chat(lambda username, message: received.append((username, message)))
    bridge.say("hi steve")
    assert wait_for(lambda: bool(received), timeout=5)
    assert received[0] == ("Steve", "nice")


def test_pathfinder_absent_falls_back(bridge: NodeBridgeBackend, wait_for):
    bridge.connect(_settings())
    assert wait_for(lambda: bridge.connected, timeout=5)
    assert bridge.pathfind_to(1, 64, 1) is False
    bridge.stop_path()  # must not raise


def test_premium_login_reports_the_device_code(bridge: NodeBridgeBackend, log: EventLog, wait_for):
    """auth='microsoft' must surface the device code and the real in-game name."""
    bridge.connect(_settings(host="127.0.0.1", username="buyer@example.com", auth="microsoft"))
    assert wait_for(lambda: bridge.connected, timeout=5)

    events = log.tail(60)
    auth_events = [event for event in events if event["level"] == "auth"]
    assert auth_events, "the device code must be logged for the user to enter"
    assert "FAKE-CODE-1234" in auth_events[0]["message"]
    assert auth_events[0]["data"]["verification_uri"].startswith("https://")

    assert wait_for(lambda: bridge.snapshot()["username"] == "PremiumPlayer", timeout=5)  # not the email
    snapshot = bridge.snapshot()
    assert snapshot["server"]["username"] == "buyer@example.com"
    assert any("Logged in as PremiumPlayer" in event["message"] for event in events)


def test_premium_failure_reason_is_surfaced(bridge: NodeBridgeBackend):
    with pytest.raises(BackendError) as excinfo:
        bridge.connect(_settings(host="127.0.0.1", username="bad@example.com", auth="microsoft"))
    assert "does the account own minecraft" in str(excinfo.value)


def test_unknown_control_is_rejected(bridge: NodeBridgeBackend):
    with pytest.raises(BackendError):
        bridge.set_control("teleport", True)


def test_commands_fail_when_process_is_gone(log: EventLog):
    backend = NodeBridgeBackend(log, node_binary=NODE, script=FAKE_BRIDGE)
    backend.start()
    assert backend._proc is not None
    backend._proc.kill()
    backend._proc.wait(timeout=5)
    time.sleep(0.2)
    backend.say("nobody is listening")  # best-effort call: logs a warning, does not raise
    assert backend.connected is False
    backend.stop()


def test_stop_terminates_the_process(bridge: NodeBridgeBackend):
    bridge.connect(_settings())
    time.sleep(0.3)
    bridge.stop()
    assert bridge.connected is False
    assert bridge.snapshot()["status"] == "disconnected"


def test_backend_factory_prefers_node_when_available(tmp_path: Path, log: EventLog):
    store = ConfigStore(tmp_path / "cfg.json")
    store.load()
    store.update({"minecraft": {"backend": "auto"}})
    backend = create_backend(store.settings, log)
    node_ok, _reason = node_available()
    assert backend.name == ("node" if node_ok else "simulated")


def test_backend_factory_respects_simulated(tmp_path: Path, log: EventLog):
    store = ConfigStore(tmp_path / "cfg.json")
    store.load()
    store.update({"minecraft": {"backend": "simulated"}})
    assert create_backend(store.settings, log).name == "simulated"


@pytest.mark.skipif(node_available()[0], reason="only meaningful without mineflayer installed")
def test_backend_factory_errors_for_missing_mineflayer(tmp_path: Path, log: EventLog):
    store = ConfigStore(tmp_path / "cfg.json")
    store.load()
    store.update({"minecraft": {"backend": "node"}})
    with pytest.raises(BackendError) as excinfo:
        create_backend(store.settings, log)
    assert "mineflayer" in str(excinfo.value)
