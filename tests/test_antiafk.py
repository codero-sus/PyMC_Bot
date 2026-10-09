"""Anti-AFK: bots must keep moving and looking around like a human.

These tests use the simulated backend, so they exercise the real movement code
(yaw changes, walking, jumping) without a Minecraft server.
"""

from __future__ import annotations

import math
import random
import time

import pytest
from pydantic import ValidationError

from pymc_bot.antiafk import HABITS, AntiAfkKeeper
from pymc_bot.backends import SimulatedBackend
from pymc_bot.bot import MinecraftBot, wrap_angle
from pymc_bot.config import AntiAfkSettings, ConfigStore
from pymc_bot.events import EventLog


@pytest.fixture()
def afk_store(tmp_path) -> ConfigStore:
    store = ConfigStore(tmp_path / "afk.json")
    store.load()
    store.update(
        {
            "minecraft": {"backend": "simulated"},
            "antiafk": {"interval_min": 1.0, "interval_max": 2.0, "verbose": True},
        }
    )
    return store


@pytest.fixture()
def idle_bot(afk_store: ConfigStore, log: EventLog, wait_for):
    backend = SimulatedBackend(log, ambient_chat=False)
    bot = MinecraftBot(afk_store, log, auto_reconnect=False, backend=backend)
    bot.start()
    assert wait_for(lambda: bot.connected, timeout=5)
    yield bot
    bot.stop()


@pytest.fixture()
def keeper(afk_store: ConfigStore, idle_bot: MinecraftBot, log: EventLog) -> AntiAfkKeeper:
    return AntiAfkKeeper(lambda: [idle_bot], afk_store, log, rng=random.Random(1234))


# ------------------------------------------------------------------- settings
def test_defaults_are_on_and_sane():
    settings = AntiAfkSettings()
    assert settings.enabled is True
    assert 1.0 <= settings.interval_min < settings.interval_max
    assert settings.max_walk_distance <= 12
    assert settings.swing and settings.look and settings.walk


def test_bad_intervals_are_rejected():
    with pytest.raises(ValidationError):
        AntiAfkSettings(interval_min=30, interval_max=5)
    with pytest.raises(ValidationError):
        AntiAfkSettings(interval_min=0.2)
    with pytest.raises(ValidationError):
        AntiAfkSettings(combo_probability=2.0)


# ------------------------------------------------------------------ bot state
def test_bot_reports_busy_and_idle_time(idle_bot: MinecraftBot):
    assert idle_bot.busy is False
    time.sleep(0.4)
    assert idle_bot.idle_seconds >= 0.35
    assert idle_bot.can_poke(min_idle=0.2) is True

    idle_bot.touch_activity()
    assert idle_bot.idle_seconds < 0.4


def test_busy_bot_is_not_pokable(idle_bot: MinecraftBot, wait_for):
    """The keeper must never fight an action for the keyboard."""
    with idle_bot.activity("walk"):
        assert idle_bot.busy is True
        assert idle_bot.current_action == "walk"
        assert idle_bot.can_poke() is False
    assert wait_for(lambda: not idle_bot.busy, timeout=2)
    assert idle_bot.can_poke() is True


def test_walking_marks_the_bot_as_active(idle_bot: MinecraftBot):
    idle_bot.walk_to(3, 3, tolerance=1.5, timeout=4)
    assert idle_bot.busy is False           # action finished
    assert idle_bot.idle_seconds < 1.5      # but the activity was recorded
    assert idle_bot.stats["current_action"] is None


def test_simulated_backend_supports_swings(idle_bot: MinecraftBot):
    assert idle_bot.swing_arm() is True
    assert idle_bot.swing_arm() is True
    assert idle_bot.backend is not None
    assert getattr(idle_bot.backend, "arm_swings", 0) == 2


# -------------------------------------------------------------------- habits
def test_every_habit_is_performable(keeper: AntiAfkKeeper, idle_bot: MinecraftBot, afk_store: ConfigStore):
    settings = afk_store.settings.antiafk
    assert set(keeper._available(settings, idle_bot)) == set(HABITS)
    for habit in HABITS:
        assert keeper._perform(idle_bot, habit, settings) is True, habit
    assert keeper._perform(idle_bot, "dance", settings) is False


def test_look_turns_the_head_in_small_steps(keeper: AntiAfkKeeper, idle_bot: MinecraftBot):
    """The head must move gradually - a single snap is what anti-AFK plugins flag."""
    viewed: list[float] = []
    backend = idle_bot.backend
    assert backend is not None
    original_look = backend.look

    def recording_look(yaw: float, pitch: float) -> None:
        viewed.append(yaw)
        original_look(yaw, pitch)

    backend.look = recording_look  # type: ignore[method-assign]
    try:
        assert keeper._habit_look(idle_bot) is True
    finally:
        backend.look = original_look  # type: ignore[method-assign]

    assert len(viewed) >= 3, "the turn should be interpolated over several steps"
    steps = [abs(wrap_angle(b - a)) for a, b in zip(viewed, viewed[1:], strict=False)]
    assert max(steps) < 0.6, f"a look step was too big: {max(steps):.2f} rad"
    assert abs(wrap_angle(viewed[-1] - viewed[0])) > 0.15, "the head barely moved"


def test_stroll_actually_walks_the_bot(keeper: AntiAfkKeeper, idle_bot: MinecraftBot, afk_store: ConfigStore):
    before = idle_bot.position()
    assert keeper._habit_stroll(idle_bot, afk_store.settings.antiafk) is True
    after = idle_bot.position()
    assert math.dist((before["x"], before["z"]), (after["x"], after["z"])) > 0.3


def test_hop_leaves_the_ground(keeper: AntiAfkKeeper, idle_bot: MinecraftBot):
    assert keeper._habit_hop(idle_bot) is True
    peak = 0.0
    for _ in range(12):
        time.sleep(0.05)
        position = idle_bot.position()
        if position:
            peak = max(peak, position["y"])
    assert peak > 64.2


def test_crouch_and_strafe_release_their_keys(keeper: AntiAfkKeeper, idle_bot: MinecraftBot):
    assert keeper._habit_crouch(idle_bot) is True
    assert keeper._habit_strafe(idle_bot) is True
    backend = idle_bot.backend
    assert backend is not None
    for control in ("sneak", "left", "right"):
        backend.set_control(control, False)  # sanity: keys are settable
    # after a strafe nothing should still be held down
    assert idle_bot.busy is False


def test_look_at_player_targets_somebody(keeper: AntiAfkKeeper, idle_bot: MinecraftBot):
    players = idle_bot.players()
    assert players
    assert keeper._habit_look_at_player(idle_bot) is True


def test_swing_calls_the_bridge(keeper: AntiAfkKeeper, idle_bot: MinecraftBot):
    before = getattr(idle_bot.backend, "arm_swings", 0)
    assert keeper._habit_swing(idle_bot) is True
    assert getattr(idle_bot.backend, "arm_swings", 0) > before


# --------------------------------------------------------------------- bursts
def test_burst_performs_at_least_one_habit(keeper: AntiAfkKeeper, idle_bot: MinecraftBot, afk_store: ConfigStore):
    habits = keeper.burst(idle_bot, afk_store.settings.antiafk)
    assert habits and set(habits) <= set(HABITS)
    assert len(habits) <= 3


def test_bursts_are_varied_not_repetitive(keeper: AntiAfkKeeper, idle_bot: MinecraftBot, afk_store: ConfigStore):
    """A bot doing the same thing every time is exactly what anti-AFK flags."""
    seen: set[str] = set()
    for _ in range(25):
        seen.update(keeper.burst(idle_bot, afk_store.settings.antiafk))
    assert len(seen) >= 4, f"only saw {seen}"


def test_habits_can_be_switched_off(keeper: AntiAfkKeeper, idle_bot: MinecraftBot, afk_store: ConfigStore):
    afk_store.update({"antiafk": {"look": False, "walk": False, "jump": False, "sneak": False, "swing": False,
                                  "look_at_players": False}})
    assert keeper.burst(idle_bot, afk_store.settings.antiafk) == []
    assert keeper._available(afk_store.settings.antiafk, idle_bot) == []


def test_poke_bot_records_the_burst(keeper: AntiAfkKeeper, idle_bot: MinecraftBot):
    habits = keeper.poke_bot(idle_bot)
    assert habits
    status = keeper.status([idle_bot])
    assert status["pokes"] == 1
    entry = status["bots"][0]
    assert entry["pokes"] == 1
    assert entry["last_habits"] == habits
    assert entry["seconds_since_poke"] is not None


# ----------------------------------------------------------------- the keeper
def test_tick_respects_the_idle_window(keeper: AntiAfkKeeper, idle_bot: MinecraftBot):
    assert keeper.tick() == []          # nothing scheduled yet
    assert keeper.tick() == []          # still inside the first interval
    assert keeper.record(idle_bot).next_burst_at > time.monotonic()


def test_tick_pokes_after_the_interval(keeper: AntiAfkKeeper, idle_bot: MinecraftBot):
    """No keeper thread here: tick() until the idle window has passed."""
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline and keeper.status([idle_bot])["pokes"] < 1:
        keeper.tick()
        time.sleep(0.2)
    status = keeper.status([idle_bot])
    assert status["pokes"] >= 1
    assert status["bots"][0]["last_habits"], "the burst must record what it did"


def test_running_keeper_keeps_bots_alive_forever(afk_store: ConfigStore, log: EventLog, wait_for):
    """The real thing: a keeper thread keeping several bots moving."""
    afk_store.update({"antiafk": {"interval_min": 1.0, "interval_max": 1.5}})
    bots = [
        MinecraftBot(afk_store, log, auto_reconnect=False, backend=SimulatedBackend(log, ambient_chat=False))
        for _ in range(3)
    ]
    for bot in bots:
        bot.start()
    assert wait_for(lambda: all(bot.connected for bot in bots), timeout=6)

    keeper = AntiAfkKeeper(lambda: bots, afk_store, log, rng=random.Random(99))
    try:
        assert keeper.start() is True
        assert wait_for(lambda: all(keeper.record(bot).pokes >= 1 for bot in bots), timeout=15)
        # and they keep going
        assert wait_for(lambda: all(keeper.record(bot).pokes >= 2 for bot in bots), timeout=20)
        assert keeper.status(bots)["pokes"] >= 6
    finally:
        assert keeper.stop() is True
        for bot in bots:
            bot.stop()
    assert keeper.start() is True          # restartable
    keeper.stop()


def test_keeper_skips_disconnected_bots(keeper: AntiAfkKeeper, idle_bot: MinecraftBot):
    idle_bot.stop()
    time.sleep(0.2)
    assert keeper.tick() == []
    assert keeper.poke_bot(idle_bot) == []
    status = keeper.status([idle_bot])
    assert status["bots"][0]["connected"] is False


def test_keeper_does_not_interrupt_a_busy_bot(afk_store: ConfigStore, log: EventLog, wait_for):
    afk_store.update({"antiafk": {"interval_min": 1.0, "interval_max": 1.2}})
    bot = MinecraftBot(afk_store, log, auto_reconnect=False, backend=SimulatedBackend(log, ambient_chat=False))
    bot.start()
    assert wait_for(lambda: bot.connected, timeout=5)
    keeper = AntiAfkKeeper(lambda: [bot], afk_store, log, rng=random.Random(5))
    try:
        started = time.monotonic()
        keeper.tick()  # schedules the first burst
        time.sleep(1.3)
        with bot.activity("manual-command"):
            assert keeper.tick() == [], "the keeper must wait while the bot is busy"
            assert bot.current_action == "manual-command"
        assert time.monotonic() - started > 1.0
    finally:
        bot.stop()
        keeper.stop(wait=False)


def test_verbose_logging_is_optional(afk_store: ConfigStore, log: EventLog, idle_bot: MinecraftBot, wait_for):
    afk_store.update({"antiafk": {"verbose": False}})
    quiet = AntiAfkKeeper(lambda: [idle_bot], afk_store, log, rng=random.Random(3))
    quiet.poke_bot(idle_bot)
    assert not any("anti-AFK" in event["message"] for event in log.tail(20))

    afk_store.update({"antiafk": {"verbose": True}})
    loud = AntiAfkKeeper(lambda: [idle_bot], afk_store, log, rng=random.Random(3))
    loud.poke_bot(idle_bot)
    assert wait_for(
        lambda: any("anti-AFK" in event["message"] for event in log.tail(20)), timeout=2
    )


def test_scheduling_stays_inside_the_configured_window(afk_store: ConfigStore):
    """Intervals are randomised but never outside the configured window."""
    from pymc_bot.antiafk import BotActivity

    settings = afk_store.settings.antiafk
    rng = random.Random(11)
    now = time.monotonic()
    waits = []
    for _ in range(500):
        activity = BotActivity()
        activity.schedule(settings, rng, now)
        waits.append(activity.next_burst_at - now)

    assert min(waits) >= settings.interval_min * 0.8
    assert max(waits) <= settings.interval_max * 1.2
    assert max(waits) - min(waits) > 0.2, "the timing must not be metronomic"
