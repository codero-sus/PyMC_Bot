"""Anti-AFK: keep every bot moving and looking around like a human.

Servers kick "AFK" players with a mix of heuristics — no movement, no head
rotation, no arm swings, no chat, or a perfectly regular movement pattern.  This
module runs one lightweight thread for the whole instance and, whenever a bot has
been idle for a while, performs a short randomised *burst* of human-ish habits:

* **look** – smooth head turns (many small steps with tiny pauses, not a snap),
  including occasional glances up/down;
* **look_at_player** – turn towards a nearby player for a moment, then away;
* **stroll** – walk one to three blocks in a random direction, then stop;
* **strafe** – a short sideways step while looking slightly off-axis;
* **hop** – one or two jumps in place;
* **crouch** – a brief sneak;
* **swing** – swing the arm a couple of times.

Timing is deliberately irregular (randomised intervals, occasional 2–3 habit
combos, never the same habit twice in a row) so the pattern does not look
scripted.  The keeper never interrupts the AI loop, a manual command or another
anti-AFK burst: it only acts on bots whose :class:`~pymc_bot.bot.MinecraftBot`
reports ``busy == False`` and which have been idle for at least
``antiafk.interval_min`` seconds.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pymc_bot.bot import MinecraftBot, wrap_angle
from pymc_bot.config import AntiAfkSettings, ConfigStore
from pymc_bot.events import EventLog

TICK = 0.4          # how often the keeper looks at the fleet
MAX_BURST = 3       # habits per burst at most
MAX_CONCURRENT = 32  # bursts in flight at once (a burst can take a few seconds)
HABITS = ("look", "look_at_player", "stroll", "strafe", "hop", "crouch", "swing")


@dataclass
class BotActivity:
    """Bookkeeping for one bot (last poke, habit history, next burst time)."""

    username: str = ""
    next_burst_at: float = 0.0
    last_poke_at: float = 0.0
    pokes: int = 0
    last_habits: list[str] = field(default_factory=list)
    in_flight: bool = False

    def schedule(self, settings: AntiAfkSettings, rng: random.Random, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        span = max(0.0, settings.interval_max - settings.interval_min)
        # Randomised, slightly skewed towards the short end so bots look awake.
        wait = settings.interval_min + span * (rng.random() ** 1.4)
        wait *= rng.uniform(0.85, 1.15)
        self.next_burst_at = now + max(1.0, wait)


class AntiAfkKeeper:
    """Runs the anti-AFK habits for every bot it is given."""

    def __init__(
        self,
        bots: Callable[[], Sequence[MinecraftBot]],
        store: ConfigStore,
        log: EventLog | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._bots = bots
        self.store = store
        self.log = log or EventLog()
        self._rng = rng or random.Random()
        # Keyed by object identity: two bots may share a name (or have none yet)
        # and each one still needs its own schedule.
        self._activity: dict[int, BotActivity] = {}
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pokes = 0
        self._last_habit: str | None = None
        self._workers: set[threading.Thread] = set()

    # ------------------------------------------------------------- lifecycle
    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def enabled(self) -> bool:
        return self.store.settings.antiafk.enabled

    def start(self) -> bool:
        with self._lock:
            if self.running:
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, name="antiafk", daemon=True)
            self._thread.start()
        if self.enabled:
            self.log.add(
                "Anti-AFK is on: idle bots will walk, hop, crouch, swing and look around like a human.",
                "info",
                "antiafk",
            )
        return True

    def stop(self, wait: bool = True) -> bool:
        with self._lock:
            if not self.running:
                return False
            self._stop_event.set()
            thread = self._thread
        if wait and thread is not None:
            thread.join(timeout=5.0)
        with self._lock:
            workers = list(self._workers)
        for worker in workers:
            worker.join(timeout=2.0)
        return True

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                settings = self.store.settings.antiafk
                if settings.enabled:
                    self.tick(settings)
            except Exception as exc:  # pragma: no cover - the keeper must never die
                self.log.add(f"Anti-AFK hiccup: {exc}", "warn", "antiafk")
            self._stop_event.wait(TICK)

    def tick(self, settings: AntiAfkSettings | None = None) -> list[str]:
        """Start a burst for every due bot.

        Bursts run in their own short-lived threads: a bot that walks for a few
        seconds must not delay the other 199 bots (or the keeper loop itself).
        Returns the names of the bots that were started.
        """
        settings = settings or self.store.settings.antiafk
        if not settings.enabled:
            return []
        now = time.monotonic()
        started: list[str] = []
        with self._lock:
            self._workers = {worker for worker in self._workers if worker.is_alive()}
            slots = MAX_CONCURRENT - len(self._workers)
        for bot in list(self._bots()):
            if slots <= 0:
                break
            if bot is None or not bot.connected:
                continue
            record = self.record(bot)
            if record.next_burst_at == 0.0:
                record.schedule(settings, self._rng, now)
                continue
            if now < record.next_burst_at or record.in_flight:
                continue
            if not bot.can_poke(min_idle=min(settings.interval_min, 5.0)):
                # Busy with the AI/human, or active very recently: check again soon.
                record.next_burst_at = now + 2.0
                continue
            record.in_flight = True
            worker = threading.Thread(
                target=self._burst_worker,
                args=(bot, settings, record),
                name=f"antiafk-{bot.username}",
                daemon=True,
            )
            with self._lock:
                self._workers.add(worker)
            worker.start()
            started.append(bot.username)
            slots -= 1
        return started

    def _burst_worker(self, bot: MinecraftBot, settings: AntiAfkSettings, record: BotActivity) -> None:
        try:
            habits = self.burst(bot, settings)
            record.last_poke_at = time.monotonic()
            if habits:
                record.pokes += 1
                record.last_habits = habits
                with self._lock:
                    self._pokes += 1
        except Exception as exc:  # pragma: no cover - a worker must never die loudly
            self.log.add(f"Anti-AFK burst failed for {bot.username}: {exc}", "debug", "antiafk")
        finally:
            record.in_flight = False
            record.schedule(settings, self._rng)

    def record(self, bot: MinecraftBot) -> BotActivity:
        key = id(bot)
        with self._lock:
            record = self._activity.get(key)
            if record is None:
                record = BotActivity(username=bot.username)
                self._activity[key] = record
            else:
                record.username = bot.username
            return record

    # ---------------------------------------------------------------- habits
    def burst(self, bot: MinecraftBot, settings: AntiAfkSettings) -> list[str]:
        """Perform one (or a short combo of) human-like habit(s)."""
        available = self._available(settings, bot)
        if not available:
            return []
        count = 1
        if self._rng.random() < settings.combo_probability:
            count = self._rng.choice([2, 2, 3])
        performed: list[str] = []
        for _ in range(min(count, MAX_BURST)):
            choices = [habit for habit in available if habit != (performed[-1] if performed else self._last_habit)]
            if not choices:
                choices = available
            habit = self._rng.choice(choices)
            try:
                if self._perform(bot, habit, settings):
                    performed.append(habit)
            except Exception as exc:  # pragma: no cover - defensive
                self.log.add(f"Anti-AFK {habit} failed for {bot.username}: {exc}", "debug", "antiafk")
            if not bot.connected:
                break
        if performed:
            self._last_habit = performed[-1]
            if settings.verbose:
                self.log.add(
                    f"anti-AFK {bot.username}: {', '.join(performed)}",
                    "debug",
                    "antiafk",
                )
        return performed

    @staticmethod
    def _available(settings: AntiAfkSettings, bot: MinecraftBot) -> list[str]:
        habits: list[str] = []
        if settings.look:
            habits.append("look")
        if settings.look_at_players and bot.players():
            habits.append("look_at_player")
        if settings.walk:
            habits.extend(["stroll", "strafe"])
        if settings.jump:
            habits.append("hop")
        if settings.sneak:
            habits.append("crouch")
        if settings.swing:
            habits.append("swing")
        return habits

    def _perform(self, bot: MinecraftBot, habit: str, settings: AntiAfkSettings) -> bool:
        if habit == "look":
            return self._habit_look(bot)
        if habit == "look_at_player":
            return self._habit_look_at_player(bot)
        if habit == "stroll":
            return self._habit_stroll(bot, settings)
        if habit == "strafe":
            return self._habit_strafe(bot)
        if habit == "hop":
            return self._habit_hop(bot)
        if habit == "crouch":
            return self._habit_crouch(bot)
        if habit == "swing":
            return self._habit_swing(bot)
        return False

    # --------------------------------------------------------- the habit impls
    def _habit_look(self, bot: MinecraftBot) -> bool:
        """A slow, smooth head turn with a small glance up or down."""
        pose = bot.position()
        if pose is None:
            return False
        turn = self._rng.uniform(0.35, 1.4) * self._rng.choice([-1, 1])
        pitch = self._rng.uniform(-0.28, 0.28)
        assert bot.backend is not None
        current_yaw = float(bot.snapshot().get("yaw") or 0.0)
        self._smooth_look(bot, wrap_angle(current_yaw + turn), pitch)
        # Sometimes glance back to roughly where we were.
        if self._rng.random() < 0.3:
            self._smooth_look(bot, wrap_angle(current_yaw + turn * self._rng.uniform(-0.4, 0.4)), 0.0)
        return True

    def _habit_look_at_player(self, bot: MinecraftBot) -> bool:
        players = bot.players()
        if not players:
            return False
        target = players[0]["name"]
        if not bot.look_at_player(target):
            return False
        time.sleep(self._rng.uniform(0.6, 1.4))
        if self._rng.random() < 0.5:
            self._habit_look(bot)
        return True

    def _habit_stroll(self, bot: MinecraftBot, settings: AntiAfkSettings) -> bool:
        """Walk a couple of blocks somewhere else, then stop."""
        pose = bot.position()
        if pose is None:
            return False
        angle = self._rng.uniform(0, 2 * math.pi)
        distance = self._rng.uniform(1.0, settings.max_walk_distance)
        target = (pose["x"] + math.cos(angle) * distance, pose["z"] + math.sin(angle) * distance)
        completed = bot.walk_to(target[0], target[1], tolerance=1.2, timeout=self._rng.uniform(4.0, 8.0))
        if self._rng.random() < 0.4:  # linger like someone who just walked somewhere
            bot.look(wrap_angle(angle))
            time.sleep(self._rng.uniform(0.3, 0.9))
        return True if completed else bot.can_poke()  # moving partway still counts

    def _habit_strafe(self, bot: MinecraftBot) -> bool:
        backend = bot.backend
        if backend is None or not backend.connected:
            return False
        side = self._rng.choice(["left", "right"])
        duration = self._rng.uniform(0.35, 0.9)
        yaw = float(bot.snapshot().get("yaw") or 0.0)
        with bot.activity("antiafk:strafe"):
            bot.look(wrap_angle(yaw + self._rng.uniform(-0.25, 0.25)))
            try:
                backend.set_control(side, True)
                time.sleep(duration)
            finally:
                backend.set_control(side, False)
        return True

    def _habit_hop(self, bot: MinecraftBot) -> bool:
        hops = self._rng.choice([1, 1, 2])
        with bot.activity("antiafk:hop"):
            for index in range(hops):
                if not bot.jump():
                    return False
                if index + 1 < hops:
                    time.sleep(self._rng.uniform(0.45, 0.9))
        return True

    def _habit_crouch(self, bot: MinecraftBot) -> bool:
        backend = bot.backend
        if backend is None or not backend.connected:
            return False
        with bot.activity("antiafk:crouch"):
            backend.set_control("sneak", True)
            time.sleep(self._rng.uniform(0.3, 0.8))
            backend.set_control("sneak", False)
        return True

    def _habit_swing(self, bot: MinecraftBot) -> bool:
        swings = self._rng.choice([1, 2, 3])
        with bot.activity("antiafk:swing"):
            for index in range(swings):
                if not bot.swing_arm():
                    return False
                if index + 1 < swings:
                    time.sleep(self._rng.uniform(0.25, 0.6))
        return True

    # ------------------------------------------------------------------ look
    def _smooth_look(
        self,
        bot: MinecraftBot,
        target_yaw: float,
        target_pitch: float,
        steps: int | None = None,
    ) -> None:
        """Turn the head in several small steps - nobody snaps their neck."""
        snap = bot.snapshot()
        if snap.get("position") is None:
            return
        yaw = float(snap.get("yaw") or 0.0)
        pitch = float(snap.get("pitch") or 0.0)
        delta_yaw = wrap_angle(target_yaw - yaw)
        steps = steps or max(3, int(abs(delta_yaw) / self._rng.uniform(0.08, 0.18)))
        for step in range(1, steps + 1):
            fraction = step / steps
            # ease-out: fast start, gentle finish (very human)
            eased = 1 - (1 - fraction) ** 2
            jitter = self._rng.uniform(-0.01, 0.01)
            bot.look(
                wrap_angle(yaw + delta_yaw * eased + jitter),
                pitch + (target_pitch - pitch) * eased,
            )
            time.sleep(self._rng.uniform(0.03, 0.07))

    # ---------------------------------------------------------------- status
    def poke_bot(self, bot: MinecraftBot, settings: AntiAfkSettings | None = None) -> list[str]:
        """Force a burst right now (used by the panel's "poke" button)."""
        settings = settings or self.store.settings.antiafk
        if bot is None or not bot.connected:
            return []
        habits = self.burst(bot, settings)
        if habits:
            record = self.record(bot)
            record.pokes += 1
            record.last_poke_at = time.monotonic()
            record.last_habits = habits
            self._pokes += 1
            record.schedule(settings, self._rng)
        return habits

    def status(self, bots: Sequence[MinecraftBot] | None = None) -> dict[str, Any]:
        settings = self.store.settings.antiafk
        bots = list(bots if bots is not None else self._bots())
        per_bot: list[dict[str, Any]] = []
        for bot in bots:
            if bot is None:
                continue
            record = self._activity.get(id(bot))
            per_bot.append(
                {
                    "username": bot.username,
                    "connected": bot.connected,
                    "busy": bot.busy,
                    "current_action": bot.current_action,
                    "idle_seconds": round(bot.idle_seconds, 1),
                    "pokes": record.pokes if record else 0,
                    "last_habits": record.last_habits if record else [],
                    "seconds_since_poke": round(time.monotonic() - record.last_poke_at, 1)
                    if record and record.last_poke_at
                    else None,
                }
            )
        return {
            "enabled": settings.enabled,
            "running": self.running,
            "pokes": self._pokes,
            "settings": settings.model_dump(),
            "bots": per_bot,
        }
