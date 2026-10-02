"""High level player controller: movement, mining, chat, stats.

Everything in here is plain Python and works against any
:class:`pymc_bot.backends.BaseBackend`, so the exact same code drives the
simulated world (tests, demos) and a real Minecraft server.

Minecraft angle conventions used throughout:
  * yaw 0 faces +Z, yaw pi/2 faces -X, increasing yaw turns left
  * pitch -pi/2 looks straight up, +pi/2 looks straight down
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pymc_bot.backends import BackendError, BaseBackend, create_backend
from pymc_bot.config import AppSettings, ConfigStore
from pymc_bot.events import EventLog

STATE_DISCONNECTED = "disconnected"
STATE_CONNECTING = "connecting"
STATE_CONNECTED = "connected"
STATE_ERROR = "error"

STEER_INTERVAL = 0.05
STUCK_WINDOW = 1.2
STUCK_DISTANCE = 0.6


def yaw_to(dx: float, dz: float) -> float:
    """Yaw that faces the vector ``(dx, dz)``."""
    return math.atan2(-dx, dz)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def pitch_to(dx: float, dy: float, dz: float) -> float:
    """Pitch that looks from the origin at the offset ``(dx, dy, dz)``."""
    horizontal = math.hypot(dx, dz)
    return math.atan2(-dy, horizontal)


class MinecraftBot:
    """Owns the backend connection and exposes player-ish verbs."""

    def __init__(
        self,
        store: ConfigStore,
        log: EventLog | None = None,
        auto_reconnect: bool = True,
        max_reconnects: int | None = 3,
        backend: BaseBackend | None = None,
        rejoin_seconds: float | None = None,
    ) -> None:
        self.store = store
        self.log = log or EventLog()
        self.auto_reconnect = auto_reconnect
        # None -> keep re-joining forever (what "populate the server" wants)
        self.max_reconnects = max_reconnects
        self.rejoin_seconds = rejoin_seconds
        self._backend: BaseBackend | None = backend
        self._injected_backend = backend is not None
        self._state = STATE_DISCONNECTED
        self._lock = threading.RLock()
        self._motion_lock = threading.RLock()  # re-entrant: mine() -> wander() -> walk_to()
        self._cancel = threading.Event()
        self._stop_requested = False
        self._configured_username: str | None = None
        self._monitor: threading.Thread | None = None
        self._reconnects = 0
        self._history: deque[dict[str, Any]] = deque(maxlen=100)
        self._greeted: set[str] = set()
        self._stats: dict[str, Any] = {
            "actions": 0,
            "chats_sent": 0,
            "blocks_mined": 0,
            "distance_walked": 0.0,
            "started_at": None,
        }
        self._last_position: tuple[float, float] | None = None
        self._active_actions = 0
        self._current_action: str | None = None
        self._last_activity = time.time()

    # -------------------------------------------------------------- activity
    @contextmanager
    def activity(self, name: str) -> Iterator[None]:
        """Mark the bot as busy with ``name`` while the block runs.

        The anti-AFK keeper watches this so it never fights the AI or a human
        command for the keyboard.
        """
        with self._lock:
            self._active_actions += 1
            self._current_action = name
        try:
            yield
        finally:
            with self._lock:
                self._active_actions -= 1
                if self._active_actions <= 0:
                    self._active_actions = 0
                    self._current_action = None
                self._last_activity = time.time()

    def touch_activity(self) -> None:
        """Record that something (AI, human, anti-AFK) just used this bot."""
        with self._lock:
            self._last_activity = time.time()

    @property
    def busy(self) -> bool:
        """True while an action owns the bot (walking, mining, following...)."""
        with self._lock:
            return self._active_actions > 0

    @property
    def current_action(self) -> str | None:
        with self._lock:
            return self._current_action

    @property
    def idle_seconds(self) -> float:
        with self._lock:
            return max(0.0, time.time() - self._last_activity)

    def can_poke(self, min_idle: float = 0.0) -> bool:
        """True when the anti-AFK keeper may take over for a moment."""
        return self.connected and not self.busy and self.idle_seconds >= min_idle

    # ---------------------------------------------------------- cancellation
    def cancel_actions(self) -> None:
        """Ask any in-flight movement/building action to stop early."""
        self._cancel.set()

    def clear_cancel(self) -> None:
        self._cancel.clear()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # ------------------------------------------------------------ properties
    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def backend(self) -> BaseBackend | None:
        with self._lock:
            return self._backend

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._state == STATE_CONNECTED and self._backend is not None and self._backend.connected

    @property
    def username(self) -> str:
        """In-game name of this bot (falls back to the configured username).

        With Microsoft auth the configured username is the account *email*, so
        the real name is read back from the server snapshot.
        """
        snapshot_name = ""
        backend = self.backend
        if backend is not None:
            snapshot_name = str(backend.snapshot().get("username") or "")
        with self._lock:
            return snapshot_name or self._configured_username or self.store.settings.minecraft.username

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            data = dict(self._stats)
            data["current_action"] = self._current_action
            data["idle_seconds"] = round(max(0.0, time.time() - self._last_activity), 1)
            started = data.get("started_at")
            data["uptime"] = round(time.time() - started, 1) if started else 0.0
            data["reconnects"] = self._reconnects
            return data

    @property
    def chat_history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def recent_chat(self, limit: int = 8) -> list[str]:
        return [f"<{entry['username']}> {entry['message']}" for entry in list(self._history)[-limit:]]

    # ------------------------------------------------------------- lifecycle
    def start(self, settings: AppSettings | None = None) -> None:
        settings = settings or self.store.settings
        with self._lock:
            if self._state in (STATE_CONNECTING, STATE_CONNECTED):
                return
            self._state = STATE_CONNECTING
            self._stop_requested = False
            self._configured_username = settings.minecraft.username
        self.log.add(
            f"Starting bot '{settings.minecraft.username}' -> {settings.minecraft.host}:{settings.minecraft.port} "
            f"({settings.minecraft.account_label})",
            "info",
            "bot",
        )
        try:
            self._connect(settings)
        except Exception as exc:
            with self._lock:
                self._state = STATE_ERROR
            self.log.add(f"Could not start the bot: {exc}", "error", "bot")
            raise
        if not self._wait_for_backend(timeout=min(20.0, settings.minecraft.connect_timeout)):
            with self._lock:
                self._state = STATE_ERROR
                backend = self._backend
            if backend is not None:
                backend.stop()
            raise BackendError(
                "The backend never reported a connection - check the server address and that it is online."
            )
        with self._lock:
            self._state = STATE_CONNECTED
            self._stats["started_at"] = time.time()
            self._reconnects = 0
        self._start_monitor()
        self.log.add("Bot is connected and ready.", "success", "bot")

    def _wait_for_backend(self, timeout: float = 10.0) -> bool:
        """Block until the backend reports a live connection."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            backend = self.backend
            if backend is None or backend.connected:
                return True
            if self._state == STATE_ERROR or self._stop_requested:
                return False
            time.sleep(0.05)
        return bool(self.backend and self.backend.connected)

    def _connect(self, settings: AppSettings) -> None:
        with self._lock:
            if self._backend is None:
                self._backend = create_backend(settings, self.log)
            elif not self._injected_backend:
                # A fresh backend per connection keeps the adapter state clean.
                self._backend.stop()
                self._backend = create_backend(settings, self.log)
            backend = self._backend
        backend.start()
        backend.on_chat(self._on_chat)
        if hasattr(backend, "connect"):  # NodeBridgeBackend
            backend.connect(settings.minecraft)  # type: ignore[attr-defined]
        elif hasattr(backend, "set_server"):
            backend.set_server(
                settings.minecraft.host,
                settings.minecraft.port,
                settings.minecraft.version,
                settings.minecraft.username,
            )

    def stop(self) -> None:
        with self._lock:
            if self._state == STATE_DISCONNECTED:
                return
            self._stop_requested = True
            backend = self._backend
            self._state = STATE_DISCONNECTED
        if backend is not None:
            try:
                backend.stop()
            except Exception as exc:  # pragma: no cover - shutdown must not raise
                self.log.add(f"Error while stopping the backend: {exc}", "warn", "bot")
        self._stats["started_at"] = None
        self.log.add("Bot stopped.", "info", "bot")

    # -------------------------------------------------------------- monitoring
    def _start_monitor(self) -> None:
        if self._monitor and self._monitor.is_alive():
            return
        self._monitor = threading.Thread(target=self._monitor_loop, name="bot-monitor", daemon=True)
        self._monitor.start()

    def _monitor_loop(self) -> None:
        while not self._stop_requested:
            time.sleep(2.0)
            backend = self.backend
            if backend is None or self._stop_requested:
                continue
            started = self._stats.get("started_at")
            if started is not None and time.time() - started < 3.0:
                continue  # grace period right after connecting
            if self._state == STATE_CONNECTED and not backend.connected:
                self.log.add("Lost the connection to the server.", "warn", "bot")
                with self._lock:
                    self._state = STATE_DISCONNECTED
                unlimited = self.max_reconnects is None
                if self.auto_reconnect and (unlimited or self._reconnects < self.max_reconnects):
                    self._try_reconnect()
                elif self.auto_reconnect:
                    self.log.add(
                        f"Gave up after {self._reconnects} reconnect attempts.", "error", "bot"
                    )
                    with self._lock:
                        self._state = STATE_ERROR

    def _try_reconnect(self) -> None:
        self._reconnects += 1
        base = self.rejoin_seconds or self.store.settings.minecraft.rejoin_seconds
        delay = min(60.0, base * min(self._reconnects, 3))
        attempt = f"{self._reconnects}" if self.max_reconnects is None else f"{self._reconnects}/{self.max_reconnects}"
        self.log.add(f"Reconnecting in {delay:.0f}s (attempt {attempt})...", "info", "bot")
        for _ in range(int(delay * 10)):
            if self._stop_requested:
                return
            time.sleep(0.1)
        if self._stop_requested:
            return
        try:
            self._connect(self._settings_for_reconnect())
            with self._lock:
                self._state = STATE_CONNECTED
            self._reconnects = 0
            self.log.add("Reconnected.", "success", "bot")
            self._start_monitor()
        except Exception as exc:
            self.log.add(f"Reconnect failed: {exc}", "error", "bot")
            with self._lock:
                self._state = STATE_ERROR

    def _settings_for_reconnect(self) -> AppSettings:
        """Config to reconnect with: the (possibly overridden) username we started with."""
        settings = self.store.settings
        with self._lock:
            username = self._configured_username
        if username and username != settings.minecraft.username:
            settings = settings.model_copy(
                update={"minecraft": settings.minecraft.model_copy(update={"username": username})}
            )
        return settings

    # ----------------------------------------------------------------- sensing
    def snapshot(self) -> dict[str, Any]:
        backend = self.backend
        if backend is None:
            return {"status": STATE_DISCONNECTED, "backend": None}
        data = backend.snapshot()
        data["state"] = self.state
        data["busy"] = self.busy
        data["idle_seconds"] = self.idle_seconds
        return data

    def position(self) -> dict[str, float] | None:
        return self.snapshot().get("position")

    def yaw(self) -> float:
        return float(self.snapshot().get("yaw") or 0.0)

    def players(self) -> list[dict[str, Any]]:
        return list(self.snapshot().get("players") or [])

    def find_player(self, name: str) -> dict[str, Any] | None:
        for player in self.players():
            if player["name"].lower() == (name or "").lower():
                return player
        return None

    def nearby_player(self, radius: float = 8.0) -> dict[str, Any] | None:
        for player in self.players():
            if player["distance"] <= radius:
                return player
        return None

    # ------------------------------------------------------------------ chat
    def _on_chat(self, username: str, message: str) -> None:
        entry = {"time": time.time(), "username": username, "message": message}
        self._history.append(entry)
        if not self.connected:
            return
        settings = self.store.settings
        if settings.agent.greet_players and username and username not in self._greeted:
            self._greeted.add(username)
            self.say(f"Hi {username}! I'm {self.username}, an AI player. Ask me anything.")
        elif message.strip().lower().endswith(("pymc_bot", "bot?")) or "pymc_bot" in message.lower():
            self.say("Yes? I'm listening.")

    def say(self, text: str) -> bool:
        backend = self.backend
        settings = self.store.settings
        if backend is None or not backend.connected:
            return False
        text = (text or "").strip().replace("\n", " ")[: settings.agent.max_chat_length]
        if not text:
            return False
        with self.activity("say"):
            backend.say(text)
            with self._lock:
                self._stats["chats_sent"] += 1
                self._stats["actions"] += 1
        self.log.add(f"<{self.username}> {text}", "bot", "bot")
        return True

    def command(self, text: str) -> bool:
        backend = self.backend
        if backend is None or not backend.connected:
            return False
        with self.activity("command"):
            backend.command(text)
        self.log.add(f"command -> /{text.lstrip('/')}", "info", "bot")
        return True

    # -------------------------------------------------------------- movement
    def _current_pose(self) -> tuple[float, float, float, float] | None:
        snap = self.snapshot()
        pos = snap.get("position")
        if not pos:
            return None
        return float(pos["x"]), float(pos["y"]), float(pos["z"]), float(snap.get("yaw") or 0.0)

    def _stop_motion(self) -> None:
        backend = self.backend
        if backend is not None and backend.connected:
            try:
                backend.stop_motion()
            except Exception as exc:  # pragma: no cover
                self.log.add(f"Could not stop motion: {exc}", "warn", "bot")

    def steer_towards(self, x: float, z: float, sprint: bool = True) -> bool:
        """One steering tick: face the waypoint and press forward if aligned."""
        backend = self.backend
        pose = self._current_pose()
        if backend is None or not backend.connected or pose is None:
            return False
        px, _py, pz, current_yaw = pose
        dx, dz = x - px, z - pz
        distance = math.hypot(dx, dz)
        if distance < 1e-3:
            return True
        target_yaw = yaw_to(dx, dz)
        try:
            backend.look(target_yaw, 0.0)
        except BackendError:
            return False
        error = wrap_angle(target_yaw - current_yaw)
        aligned = abs(error) < 0.55
        backend.set_control("forward", aligned)
        backend.set_control("sprint", bool(aligned and sprint and distance > 2.5))
        # Gently correct residual heading error by strafing.
        strafe = 0.0
        if aligned and abs(error) > 0.15:
            strafe = 1.0 if error > 0 else -1.0
        backend.set_control("right", strafe > 0)
        backend.set_control("left", strafe < 0)
        return True

    def walk_to(
        self,
        x: float,
        z: float,
        y: float | None = None,
        tolerance: float = 1.5,
        timeout: float | None = None,
    ) -> bool:
        """Walk to a point, preferring the backend pathfinder, else steering."""
        settings = self.store.settings
        timeout = timeout if timeout is not None else settings.agent.action_timeout
        backend = self.backend
        if backend is None or not backend.connected:
            return False
        if not self._motion_lock.acquire(timeout=timeout):
            self.log.add("Another movement action is already running.", "warn", "bot")
            return False
        try:
            start = time.monotonic()
            with self.activity("walk"), self._lock:
                self._stats["actions"] += 1
            destination = f"({x:.0f}, {y:.0f}, {z:.0f})" if y is not None else f"({x:.0f}, {z:.0f})"
            self.log.add(f"Walking to {destination}.", "info", "bot")
            if backend.supports_pathfinder and y is not None:
                if backend.pathfind_to(x, y, z, tolerance):
                    if self._wait_until_near(x, z, tolerance, timeout):
                        return True
                    backend.stop_path()
            return self._steer_to(x, z, tolerance, timeout, start)
        finally:
            self._stop_motion()
            self._track_distance()
            self._motion_lock.release()

    def _wait_until_near(self, x: float, z: float, tolerance: float, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.connected or self.cancelled:
                return False
            pose = self._current_pose()
            if pose and math.hypot(pose[0] - x, pose[2] - z) <= tolerance:
                return True
            time.sleep(0.1)
        return False

    def _steer_to(
        self, x: float, z: float, tolerance: float, timeout: float, start: float
    ) -> bool:
        """Iteratively press forward towards (x, z) until we arrive or get stuck."""
        history: deque[tuple[float, float, float]] = deque(maxlen=40)
        jumps = 0
        while time.monotonic() - start < timeout:
            if not self.connected or self.cancelled:
                return False
            pose = self._current_pose()
            if pose is None:
                time.sleep(STEER_INTERVAL)
                continue
            px, py, pz, _yaw = pose
            history.append((time.monotonic(), px, pz))
            distance = math.hypot(px - x, pz - z)
            if distance <= tolerance:
                return True
            # Aim a short way ahead so walls are not hugged too tightly.
            if distance > 3.0:
                ratio = 2.5 / distance
                self.steer_towards(px + (x - px) * ratio, pz + (z - pz) * ratio)
            else:
                self.steer_towards(x, z)
            if self._is_stuck(history):
                if jumps < 3:
                    jumps += 1
                    backend = self.backend
                    if backend is not None:
                        backend.jump()
                    time.sleep(0.4)
                else:
                    return False
            time.sleep(STEER_INTERVAL)
        return False

    @staticmethod
    def _is_stuck(history: deque[tuple[float, float, float]]) -> bool:
        if len(history) < int(STUCK_WINDOW / STEER_INTERVAL):
            return False
        first = history[0]
        last = history[-1]
        if last[0] - first[0] < STUCK_WINDOW:
            return False
        return math.hypot(last[1] - first[1], last[2] - first[2]) < STUCK_DISTANCE

    def _track_distance(self) -> None:
        pose = self._current_pose()
        if pose is None:
            return
        if self._last_position is not None:
            self._stats["distance_walked"] += math.dist(self._last_position, (pose[0], pose[2]))
        self._last_position = (pose[0], pose[2])

    def goto(self, x: float, z: float, y: float | None = None, tolerance: float = 1.5) -> bool:
        return self.walk_to(x, z, y, tolerance=tolerance)

    def wander(self, radius: int | None = None, steps: int = 3) -> bool:
        """Random exploration: a few short hops in random directions."""
        settings = self.store.settings
        radius = radius or settings.agent.wander_radius
        pose = self._current_pose()
        if pose is None:
            return False
        origin = (pose[0], pose[2])
        moved = False
        for _ in range(max(1, steps)):
            if self.cancelled:
                break
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(radius * 0.4, radius)
            tx = origin[0] + math.cos(angle) * distance
            tz = origin[1] + math.sin(angle) * distance
            if self.walk_to(tx, tz, tolerance=1.5, timeout=min(12.0, settings.agent.action_timeout)):
                moved = True
        if not moved:
            self.log.add("Explored a bit but could not move far.", "warn", "bot")
        return moved

    def follow(self, player: str, seconds: float | None = None, distance: float = 3.0) -> bool:
        settings = self.store.settings
        seconds = seconds or settings.agent.action_timeout
        deadline = time.monotonic() + seconds
        self.log.add(f"Following {player}.", "info", "bot")
        return self._run_follow(player, deadline, distance)

    def _run_follow(self, player: str, deadline: float, distance: float) -> bool:
        with self.activity("follow"):
            return self._follow_loop(player, deadline, distance)

    def _follow_loop(self, player: str, deadline: float, distance: float) -> bool:
        while time.monotonic() < deadline:
            if self.cancelled:
                break
            target = self.find_player(player)
            if target is None:
                self.log.add(f"{player} is not visible any more.", "warn", "bot")
                return False
            if target["distance"] <= distance:
                self.look_at_player(player)
                time.sleep(0.5)
                continue
            remaining = max(1.0, min(6.0, deadline - time.monotonic()))
            self.walk_to(target["x"], target["z"], target.get("y"), tolerance=distance, timeout=remaining)
        self._stop_motion()
        return True

    def look(self, yaw: float, pitch: float | None = None) -> bool:
        """Turn the head (yaw only when pitch is omitted)."""
        backend = self.backend
        pose = self._current_pose()
        if backend is None or not backend.connected or pose is None:
            return False
        try:
            backend.look(yaw, pose[3] if pitch is None else pitch)
        except BackendError:
            return False
        self.touch_activity()
        return True

    def swing_arm(self) -> bool:
        """Swing the arm (a very cheap 'I am here' signal)."""
        backend = self.backend
        if backend is None or not backend.connected:
            return False
        return bool(backend.swing_arm())

    def look_at_player(self, player: str) -> bool:
        backend = self.backend
        target = self.find_player(player)
        pose = self._current_pose()
        if backend is None or target is None or pose is None:
            return False
        dx = target["x"] - pose[0]
        dy = target.get("y", pose[1] + 1.6) + 0.5 - (pose[1] + 1.6)
        dz = target["z"] - pose[2]
        with self.activity("look_at_player"):
            try:
                backend.look(yaw_to(dx, dz), pitch_to(dx, dy, dz))
            except BackendError:
                return False
        return True

    def jump(self) -> bool:
        backend = self.backend
        if backend is None or not backend.connected:
            return False
        with self.activity("jump"), self._lock:
            backend.jump()
            self._stats["actions"] += 1
        self.log.add("Jumped.", "debug", "bot")
        return True

    def stop_moving(self) -> bool:
        self._stop_motion()
        self.touch_activity()
        self.log.add("Stopped moving.", "info", "bot")
        return True

    # ------------------------------------------------------------ interactions
    def mine(self, block: str, attempts: int = 2) -> bool:
        backend = self.backend
        if backend is None or not backend.connected:
            return False
        with self._motion_lock, self.activity("mine"):
            with self._lock:
                self._stats["actions"] += 1
            for attempt in range(max(1, attempts)):
                if self.cancelled:
                    return False
                if backend.dig(block):
                    with self._lock:
                        self._stats["blocks_mined"] += 1
                    self.log.add(f"Mined {block}.", "success", "bot")
                    return True
                if attempt + 1 < attempts:
                    self.log.add(f"No {block} nearby - exploring to find some.", "info", "bot")
                    self.wander(steps=1, radius=max(6, self.store.settings.agent.wander_radius // 2))
            self.log.add(f"Could not mine {block}.", "warn", "bot")
            return False

    def eat(self) -> bool:
        backend = self.backend
        if backend is None or not backend.connected:
            return False
        with self.activity("eat"):
            result = bool(getattr(backend, "eat", lambda: False)())
        self.log.add("Ate some food." if result else "Nothing to eat (or still full).", "info", "bot")
        return result

    def attack(self, player: str) -> bool:
        backend = self.backend
        if backend is None or not backend.connected:
            return False
        with self.activity("attack"), self._lock:
            self._stats["actions"] += 1
        return bool(backend.attack(player))

    def reconnect(self) -> None:
        self.stop()
        self._reconnects = 0
        self.start()
