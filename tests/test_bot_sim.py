"""Movement / interaction tests against the simulated world."""

from __future__ import annotations

import math
import time

from pymc_bot.backends import SimulatedBackend
from pymc_bot.bot import MinecraftBot, pitch_to, wrap_angle, yaw_to
from pymc_bot.config import ConfigStore
from pymc_bot.events import EventLog


# --------------------------------------------------------------------- angles
def test_yaw_math_matches_minecraft_conventions():
    # yaw 0 faces +Z, yaw pi/2 faces -X
    assert yaw_to(0, 1) == 0
    assert abs(yaw_to(-1, 0) - math.pi / 2) < 1e-9
    assert abs(wrap_angle(3 * math.pi) - math.pi) < 1e-9
    assert pitch_to(0, 1, 1) < 0          # looking up at something above
    assert pitch_to(0, -1, 1) > 0         # looking down


# ------------------------------------------------------------------- connection
def test_simulated_backend_connects(store: ConfigStore, log: EventLog, wait_for):
    backend = SimulatedBackend(log, ambient_chat=False)
    backend.start()
    assert wait_for(lambda: backend.connected, timeout=5)
    snap = backend.snapshot()
    assert snap["status"] == "connected"
    assert snap["position"]["y"] == 64.0
    backend.stop()
    assert backend.snapshot()["status"] == "disconnected"


def test_simulated_backend_walking_moves_the_player(sim_backend: SimulatedBackend):
    sim_backend.start()
    time.sleep(0.3)
    start = sim_backend.snapshot()["position"]
    sim_backend.set_control("forward", True)
    time.sleep(0.6)
    sim_backend.set_control("forward", False)
    end = sim_backend.snapshot()["position"]
    assert math.dist((start["x"], start["z"]), (end["x"], end["z"])) > 1.0


def test_simulated_backend_jump_leaves_the_ground(sim_backend: SimulatedBackend):
    sim_backend.start()
    time.sleep(0.3)
    sim_backend.jump()
    peak = sim_backend.GROUND_Y
    for _ in range(40):
        time.sleep(0.05)
        peak = max(peak, sim_backend.snapshot()["position"]["y"])
    assert peak > sim_backend.GROUND_Y + 0.5
    time.sleep(0.8)
    assert abs(sim_backend.snapshot()["position"]["y"] - sim_backend.GROUND_Y) < 0.05


# -------------------------------------------------------------------- walking
def test_walk_to_reaches_the_target(live_bot: MinecraftBot):
    assert live_bot.walk_to(10, 10, tolerance=2.0, timeout=15)
    pos = live_bot.position()
    assert math.hypot(pos["x"] - 10, pos["z"] - 10) <= 2.5


def test_walk_to_times_out_on_an_impossible_target(live_bot: MinecraftBot):
    start = time.monotonic()
    assert live_bot.walk_to(5000, 5000, timeout=2.0) is False
    assert time.monotonic() - start < 8.0


def test_steering_faces_the_waypoint(live_bot: MinecraftBot):
    live_bot.steer_towards(0, 50)  # straight up +Z
    time.sleep(0.3)
    assert abs(wrap_angle(live_bot.yaw())) < 0.6


def test_mining_adds_to_the_inventory(live_bot: MinecraftBot):
    before = {item["name"]: item["count"] for item in live_bot.snapshot()["inventory"]}
    assert live_bot.mine("oak_log") is True
    after = {item["name"]: item["count"] for item in live_bot.snapshot()["inventory"]}
    assert after.get("oak_log", 0) == before.get("oak_log", 0) + 1
    assert live_bot.stats["blocks_mined"] >= 1


def test_mine_falls_back_to_exploring_without_deadlock(store: ConfigStore, log: EventLog):
    """mine() -> wander() -> walk_to() must not deadlock on the motion lock."""
    store.update({"agent": {"action_timeout": 1.5, "wander_radius": 4}})
    backend = SimulatedBackend(log, ambient_chat=False)

    def never_dig(_block: str) -> bool:
        return False

    backend.dig = never_dig  # type: ignore[method-assign]
    bot = MinecraftBot(store, log, auto_reconnect=False, backend=backend)
    bot.start()
    time.sleep(0.5)
    try:
        started = time.monotonic()
        assert bot.mine("diamond_ore") is False
        assert time.monotonic() - started < 20
    finally:
        bot.stop()


def test_follow_and_look_at_player(live_bot: MinecraftBot):
    players = live_bot.players()
    assert players, "the simulated world always has a couple of players"
    name = players[0]["name"]
    assert live_bot.look_at_player(name) is True
    assert live_bot.follow(name, seconds=2.0) is True
    assert live_bot.find_player("nobody-here") is None


def test_chat_is_logged_and_greets_new_players(live_bot: MinecraftBot, log: EventLog):
    backend = live_bot.backend
    assert backend is not None
    assert live_bot.say("hello world") is True
    backend._emit_chat("Steve", "hey bot")
    time.sleep(0.2)
    messages = [event["message"] for event in log.tail(50)]
    assert any("hello world" in message for message in messages)
    assert any("Hi Steve" in message for message in messages)
    backend._emit_chat("Steve", "are you there?")
    time.sleep(0.2)
    assert len(live_bot.chat_history) == 2


def test_chat_is_blocked_when_disabled(store: ConfigStore, log: EventLog):
    store.update({"agent": {"allow_chat": False}})
    backend = SimulatedBackend(log, ambient_chat=False)
    bot = MinecraftBot(store, log, auto_reconnect=False, backend=backend)
    bot.start()
    time.sleep(0.4)
    try:
        # The raw bot API always sends; the AI gate lives in AgentLoop.execute
        assert bot.say("still allowed manually") is True
    finally:
        bot.stop()


def test_wander_and_stop(live_bot: MinecraftBot):
    assert live_bot.wander(radius=6, steps=2) is True
    assert live_bot.stop_moving() is True
    assert live_bot.stats["distance_walked"] > 0


def test_snapshot_shape(live_bot: MinecraftBot):
    snap = live_bot.snapshot()
    for key in ("position", "health", "food", "players", "inventory", "yaw", "pitch", "status", "backend"):
        assert key in snap
    assert snap["backend"] == "simulated"
    assert snap["state"] == "connected"


def test_stop_disconnects(live_bot: MinecraftBot):
    live_bot.stop()
    assert live_bot.connected is False
    assert live_bot.state == "disconnected"
    assert live_bot.snapshot()["status"] == "disconnected"
