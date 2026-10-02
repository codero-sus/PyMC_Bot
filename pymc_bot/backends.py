"""Bot backends: the simulated world and the real Minecraft (Node) bridge.

Two implementations of :class:`BaseBackend`:

``SimulatedBackend``
    A tiny pure-Python voxel-less world used for tests, demos and for running
    the control panel when Node/mineflayer is not installed.  It implements the
    exact same primitive API, including walking physics, so the high level
    movement code in :mod:`pymc_bot.bot` is exercised for real.

``NodeBridgeBackend``
    Spawns ``minecraft_bridge.js`` (mineflayer) as a child process and talks
    newline-delimited JSON over stdin/stdout.  This is what actually joins a
    cracked (offline-mode) Minecraft server.  All behaviour lives in Python;
    the bridge only translates primitives to the Minecraft protocol.
"""

from __future__ import annotations

import abc
import json
import math
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pymc_bot.config import AppSettings, MinecraftSettings
from pymc_bot.events import EventLog

NODE_DIR = Path(__file__).resolve().parent / "node"
BRIDGE_SCRIPT = NODE_DIR / "minecraft_bridge.js"
CONTROLS = ("forward", "back", "left", "right", "jump", "sneak", "sprint")


class BackendError(RuntimeError):
    """Raised when a backend cannot start or a primitive fails."""


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------
def empty_snapshot(status: str = "disconnected", backend: str = "") -> dict[str, Any]:
    return {
        "status": status,
        "backend": backend,
        "username": "",
        "position": None,
        "yaw": 0.0,
        "pitch": 0.0,
        "health": 20.0,
        "food": 20.0,
        "dimension": "overworld",
        "time_of_day": "day",
        "players": [],
        "inventory": [],
        "server": {"host": "", "port": 0, "version": ""},
        "last_error": None,
        "connected_at": None,
        "uptime": 0.0,
    }


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------
class BaseBackend(abc.ABC):
    """Primitive motor/sensor interface used by :class:`pymc_bot.bot.MinecraftBot`."""

    name = "base"
    supports_pathfinder = False

    def __init__(self, log: EventLog | None = None) -> None:
        self.log = log or EventLog()
        self._chat_callbacks: list[Callable[[str, str], None]] = []
        self._status_callbacks: list[Callable[[], None]] = []
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- events
    def on_chat(self, callback: Callable[[str, str], None]) -> Callable[[], None]:
        self._chat_callbacks.append(callback)

        def _unsubscribe() -> None:
            if callback in self._chat_callbacks:
                self._chat_callbacks.remove(callback)

        return _unsubscribe

    def on_status(self, callback: Callable[[], None]) -> Callable[[], None]:
        self._status_callbacks.append(callback)

        def _unsubscribe() -> None:
            if callback in self._status_callbacks:
                self._status_callbacks.remove(callback)

        return _unsubscribe

    def _emit_chat(self, username: str, message: str) -> None:
        for callback in list(self._chat_callbacks):
            try:
                callback(username, message)
            except Exception:  # pragma: no cover - callbacks must not kill the reader
                pass

    def _emit_status(self) -> None:
        for callback in list(self._status_callbacks):
            try:
                callback()
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------- lifecycle
    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @property
    @abc.abstractmethod
    def connected(self) -> bool: ...

    # ------------------------------------------------------------ primitives
    @abc.abstractmethod
    def say(self, text: str) -> None: ...

    @abc.abstractmethod
    def command(self, text: str) -> None: ...

    @abc.abstractmethod
    def set_control(self, control: str, state: bool) -> None: ...

    @abc.abstractmethod
    def look(self, yaw: float, pitch: float) -> None: ...

    def jump(self) -> None:
        self.set_control("jump", True)
        time.sleep(0.15)
        self.set_control("jump", False)

    def swing_arm(self) -> bool:
        """Swing the arm. Backends without animations may return False."""
        return False

    @abc.abstractmethod
    def stop_motion(self) -> None: ...

    def pathfind_to(self, x: float, y: float, z: float, tolerance: float = 1.5) -> bool:
        """Ask the backend's pathfinder to walk somewhere.

        Returns ``True`` when a pathfinder accepted the goal.  When it returns
        ``False`` the caller falls back to Python steering.
        """
        return False

    def stop_path(self) -> None:  # pragma: no cover - trivial
        return None

    @abc.abstractmethod
    def dig(self, block: str) -> bool: ...

    @abc.abstractmethod
    def attack(self, player: str) -> bool: ...

    @abc.abstractmethod
    def snapshot(self) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# simulated backend
# ---------------------------------------------------------------------------
class SimulatedBackend(BaseBackend):
    """Pure-Python pretend world (no Minecraft needed)."""

    name = "simulated"
    supports_pathfinder = False

    GROUND_Y = 64.0
    WALK_SPEED = 4.317
    SPRINT_SPEED = 5.612
    GRAVITY = 32.0
    JUMP_VELOCITY = 8.6
    TICK = 0.05
    AMBIENT_CHAT_EVERY = 25.0

    def __init__(self, log: EventLog | None = None, ambient_chat: bool = True) -> None:
        super().__init__(log)
        self.ambient_chat = ambient_chat
        self._status = "disconnected"
        self._controls: dict[str, bool] = {name: False for name in CONTROLS}
        self._position = [0.5, self.GROUND_Y, 0.5]
        self._velocity_y = 0.0
        self._yaw = 0.0
        self._pitch = 0.0
        self._health = 20.0
        self._food = 20.0
        self._inventory: dict[str, int] = {"oak_log": 3, "dirt": 2}
        self._connected_at: float | None = None
        self._last_error: str | None = None
        self._server = {"host": "", "port": 0, "version": ""}
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pending_jump = False
        self._rng = random.Random(1337)
        self._last_ambient = time.monotonic()
        # Fake neighbours so the AI has someone to talk to / follow.
        self._players: dict[str, dict[str, Any]] = {
            "Steve": {"x": 6.0, "y": self.GROUND_Y, "z": -4.0, "vx": 0.0, "vz": 0.0},
            "Alex": {"x": -8.0, "y": self.GROUND_Y, "z": 7.0, "vx": 0.0, "vz": 0.0},
        }

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._status in ("connected", "connecting"):
            return
        self._status = "connecting"
        self._emit_status()
        self.log.add("Connecting to simulated world...", "info", "sim")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="sim-backend", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        time.sleep(0.2)
        self._status = "connected"
        self._connected_at = time.time()
        self.log.add("Joined the simulated world as a lone player.", "success", "sim")
        self._emit_status()
        last = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            dt = min(0.25, now - last)
            last = now
            self._tick(dt)
            time.sleep(self.TICK)
        self._status = "disconnected"
        self._emit_status()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._status = "disconnected"
        self._connected_at = None
        for name in self._controls:
            self._controls[name] = False
        self.log.add("Left the simulated world.", "info", "sim")
        self._emit_status()

    @property
    def connected(self) -> bool:
        return self._status == "connected"

    # ------------------------------------------------------------------ tick
    def _tick(self, dt: float) -> None:
        # --- vertical physics -------------------------------------------------
        if self._pending_jump and self._position[1] <= self.GROUND_Y + 1e-6:
            self._velocity_y = self.JUMP_VELOCITY
            self._pending_jump = False
        self._velocity_y -= self.GRAVITY * dt
        self._position[1] += self._velocity_y * dt
        if self._position[1] <= self.GROUND_Y:
            self._position[1] = self.GROUND_Y
            self._velocity_y = 0.0

        # --- horizontal walking ----------------------------------------------
        forward = float(self._controls["forward"]) - float(self._controls["back"])
        strafe = float(self._controls["right"]) - float(self._controls["left"])
        if forward or strafe:
            speed = self.SPRINT_SPEED if self._controls["sprint"] else self.WALK_SPEED
            if self._controls["sneak"]:
                speed *= 0.3
            fx, fz = -math.sin(self._yaw), math.cos(self._yaw)
            rx, rz = -fz, fx
            vx = fx * forward + rx * strafe
            vz = fz * forward + rz * strafe
            length = math.hypot(vx, vz) or 1.0
            self._position[0] += (vx / length) * speed * dt
            self._position[2] += (vz / length) * speed * dt
            self._position[0] = max(-300.0, min(300.0, self._position[0]))
            self._position[2] = max(-300.0, min(300.0, self._position[2]))

        # --- needs -------------------------------------------------------------
        self._food = max(0.0, self._food - 0.02 * dt)
        if self._food > 6 and self._health < 20.0:
            self._health = min(20.0, self._health + 0.5 * dt)

        # --- neighbours ---------------------------------------------------------
        for player in self._players.values():
            if self._rng.random() < 0.05:
                player["vx"] = self._rng.uniform(-0.6, 0.6)
                player["vz"] = self._rng.uniform(-0.6, 0.6)
            player["x"] = max(-60.0, min(60.0, player["x"] + player["vx"] * dt * 4))
            player["z"] = max(-60.0, min(60.0, player["z"] + player["vz"] * dt * 4))

        if self.ambient_chat and time.monotonic() - self._last_ambient > self.AMBIENT_CHAT_EVERY:
            self._last_ambient = time.monotonic()
            speaker = self._rng.choice(list(self._players))
            line = self._rng.choice(
                [
                    "hey bot, what are you doing?",
                    "anyone got spare wood?",
                    "nice day for mining",
                    "PyMC_Bot, come help me build!",
                ]
            )
            self.log.add(f"<{speaker}> {line}", "chat", "sim")
            self._emit_chat(speaker, line)

    # ------------------------------------------------------------ primitives
    def say(self, text: str) -> None:
        self.log.add(f"<{self._server.get('username', 'PyMC_Bot')}> {text}", "bot", "sim")

    def command(self, text: str) -> None:
        line = text if text.startswith("/") else f"/{text}"
        self.log.add(f"command: {line}", "info", "sim")
        if line.startswith("/tp"):
            self._position = [2.0, self.GROUND_Y, 2.0]

    def set_control(self, control: str, state: bool) -> None:
        if control not in self._controls:
            raise BackendError(f"unknown control: {control}")
        self._controls[control] = bool(state)

    def look(self, yaw: float, pitch: float) -> None:
        self._yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        self._pitch = max(-math.pi / 2, min(math.pi / 2, pitch))

    def jump(self) -> None:
        self._pending_jump = True
        self._swings = getattr(self, "_swings", 0)

    def swing_arm(self) -> bool:
        self._swings = getattr(self, "_swings", 0) + 1
        self.arm_swings = self._swings
        return True

    def stop_motion(self) -> None:
        for name in ("forward", "back", "left", "right", "sneak", "sprint"):
            self._controls[name] = False
        self._pending_jump = False

    def dig(self, block: str) -> bool:
        block = (block or "stone").strip().lower().replace(" ", "_")
        time.sleep(0.05)
        self._inventory[block] = self._inventory.get(block, 0) + 1
        self.log.add(f"Mined 1 {block}.", "success", "sim")
        return True

    def attack(self, player: str) -> bool:
        if player not in self._players:
            return False
        self.log.add(f"Attacked {player}.", "warn", "sim")
        return True

    def snapshot(self) -> dict[str, Any]:
        snap = empty_snapshot(self._status, self.name)
        snap["position"] = {"x": round(self._position[0], 2), "y": round(self._position[1], 2), "z": round(self._position[2], 2)}
        snap["yaw"] = round(self._yaw, 4)
        snap["pitch"] = round(self._pitch, 4)
        snap["health"] = round(self._health, 1)
        snap["food"] = round(self._food, 1)
        snap["players"] = self._player_list()
        snap["inventory"] = [
            {"name": name, "count": count} for name, count in sorted(self._inventory.items()) if count > 0
        ]
        snap["server"] = dict(self._server)
        snap["last_error"] = self._last_error
        snap["connected_at"] = self._connected_at
        snap["uptime"] = round(time.time() - self._connected_at, 1) if self._connected_at else 0.0
        return snap

    def _player_list(self) -> list[dict[str, Any]]:
        out = []
        for name, player in self._players.items():
            distance = math.dist((self._position[0], self._position[2]), (player["x"], player["z"]))
            out.append(
                {
                    "name": name,
                    "x": round(player["x"], 1),
                    "y": round(player["y"], 1),
                    "z": round(player["z"], 1),
                    "distance": round(distance, 1),
                }
            )
        out.sort(key=lambda item: item["distance"])
        return out

    # ------------------------------------------------------------ test hooks
    def set_server(self, host: str, port: int, version: str, username: str) -> None:
        self._server = {"host": host, "port": port, "version": version, "username": username}

    def teleport(self, x: float, y: float, z: float) -> None:
        self._position = [float(x), float(y), float(z)]


# ---------------------------------------------------------------------------
# node / mineflayer backend
# ---------------------------------------------------------------------------
def node_available() -> tuple[bool, str]:
    """Check for node + the mineflayer dependency. Returns (ok, reason)."""
    node = shutil.which("node")
    if not node:
        return False, "Node.js was not found on PATH (needed for the real Minecraft protocol)."
    if not BRIDGE_SCRIPT.exists():
        return False, f"Bridge script missing: {BRIDGE_SCRIPT}"
    search = [NODE_DIR / "node_modules" / "mineflayer", Path.cwd() / "node_modules" / "mineflayer"]
    if not any(path.exists() for path in search):
        return False, "mineflayer is not installed. Run `npm install` in the project root."
    return True, "ok"


class NodeBridgeBackend(BaseBackend):
    """Talks to ``minecraft_bridge.js`` over newline-delimited JSON."""

    name = "node"

    def __init__(
        self,
        log: EventLog | None = None,
        node_binary: str | None = None,
        script: Path | str | None = None,
    ) -> None:
        super().__init__(log)
        self.node_binary = node_binary or shutil.which("node") or "node"
        self.script = Path(script) if script is not None else BRIDGE_SCRIPT
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._pending: dict[int, queue.Queue] = {}
        self._next_id = 1
        self._caps: dict[str, Any] = {}
        self._snapshot = empty_snapshot("disconnected", self.name)
        self._connected = False
        self._ready = threading.Event()
        self._spawned = threading.Event()
        self._connect_error: str | None = None
        self._host = ""
        self._port = 0
        self._version = ""
        self._username = ""

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._snapshot["status"] = "connecting"
        cmd = [self.node_binary, str(self.script)]
        self.log.add(f"Launching Minecraft bridge: {' '.join(cmd)}", "info", "node")
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=str(Path.cwd()),
            )
        except OSError as exc:  # pragma: no cover - depends on environment
            raise BackendError(f"Could not start Node.js: {exc}") from exc
        self._reader = threading.Thread(target=self._read_stdout, name="node-stdout", daemon=True)
        self._reader.start()
        threading.Thread(target=self._read_stderr, name="node-stderr", daemon=True).start()
        if not self._ready.wait(timeout=20.0):
            raise BackendError("Node bridge did not become ready in time.")
        if self._connect_error:
            raise BackendError(self._connect_error)

    def connect(self, settings: MinecraftSettings) -> None:
        """Send the ``connect`` command and wait for the spawn event."""
        self._host = settings.host
        self._port = settings.port
        self._version = settings.version
        self._username = settings.username
        version = None if settings.version in ("", "auto") else settings.version
        self._spawned.clear()
        self._connect_error = None
        response = self._request(
            "connect",
            {
                "host": settings.host,
                "port": settings.port,
                "username": settings.username,
                "version": version,
                "auth": settings.auth,
                "view_distance": settings.view_distance,
                "profiles_folder": self._profiles_folder(settings),
            },
            timeout=settings.connect_timeout + 5,
        )
        if not response.get("ok"):
            message = response.get("error") or "unknown error"
            self._connect_error = message
            self._snapshot["status"] = "error"
            self._snapshot["last_error"] = message
            self._connected = False
            raise BackendError(message)
        if not self._spawned.wait(timeout=settings.connect_timeout):
            raise BackendError(
                f"Timed out after {settings.connect_timeout:.0f}s waiting to spawn in {settings.host}:{settings.port}"
            )
        # 'spawn' set the event on success; an error/kick/end also releases it.
        if self._connect_error:
            raise BackendError(self._connect_error)
        if not self._connected:
            reason = self._snapshot.get("last_error") or "connection closed before the bot spawned"
            raise BackendError(str(reason))

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._request("disconnect", {}, timeout=3)
            except Exception:
                pass
            time.sleep(0.1)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._proc.kill()
        self._proc = None
        self._connected = False
        self._snapshot["status"] = "disconnected"
        self._snapshot["connected_at"] = None
        self._emit_status()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def capabilities(self) -> dict[str, Any]:
        return dict(self._caps)

    @property
    def supports_pathfinder(self) -> bool:  # type: ignore[override]
        return bool(self._caps.get("pathfinder"))

    # -------------------------------------------------------------- plumbing
    def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                self.log.add(f"bridge: {line}", "debug", "node")
                continue
            self._handle_message(payload)
        self._connected = False
        if self._snapshot["status"] not in ("error",):
            self._snapshot["status"] = "disconnected"
        self.log.add("Minecraft bridge process finished.", "warn", "node")
        self._ready.set()
        self._emit_status()

    def _read_stderr(self) -> None:
        if not self._proc or not self._proc.stderr:
            return
        for line in self._proc.stderr:
            line = line.strip()
            if line:
                self.log.add(f"node: {line}", "debug", "node")

    def _handle_message(self, payload: dict[str, Any]) -> None:
        request_id = payload.get("id")
        if request_id is not None:
            waiter = self._pending.pop(request_id, None)
            if waiter is not None:
                waiter.put(payload)
            return

        event = payload.get("event")
        if event == "ready":
            self._caps = payload.get("caps", {})
            self.log.add(
                "Bridge ready (pathfinder available)." if self._caps.get("pathfinder")
                else "Bridge ready (no pathfinder - Python steering will be used).",
                "info",
                "node",
            )
            self._ready.set()
        elif event == "log":
            self.log.add(payload.get("message", ""), payload.get("level", "info"), "node")
        elif event == "state":
            self._apply_state(payload.get("state", {}))
        elif event == "chat":
            username = str(payload.get("username", "?"))
            message = str(payload.get("message", ""))
            self.log.add(f"<{username}> {message}", "chat", "node")
            self._emit_chat(username, message)
        elif event == "msa_code":
            code = payload.get("user_code") or "?"
            uri = payload.get("verification_uri") or "https://www.microsoft.com/link"
            self.log.add(
                f"Premium login: open {uri} and enter the code {code} "
                f"(account: {self._username}). The code expires soon - do it now.",
                "auth",
                "node",
                data={"user_code": code, "verification_uri": uri, "username": self._username},
            )
        elif event == "login":
            self.log.add(f"Logged in as {payload.get('username', self._username)}.", "success", "node")
        elif event == "spawn":
            self._connected = True
            self._snapshot["status"] = "connected"
            self._snapshot["connected_at"] = time.time()
            self._snapshot["last_error"] = None
            self._spawned.set()
            self._emit_status()
        elif event in ("end", "kicked"):
            reason = payload.get("reason") or event
            self._connected = False
            self._snapshot["status"] = "disconnected" if event == "end" else "error"
            self._snapshot["last_error"] = str(reason)
            self._spawned.set()
            self.log.add(f"Disconnected from server: {reason}", "warn", "node")
            self._emit_status()
        elif event == "error":
            message = str(payload.get("message", "unknown error"))
            self._connect_error = message
            self._snapshot["status"] = "error"
            self._snapshot["last_error"] = message
            self.log.add(f"Bridge error: {message}", "error", "node")
            self._spawned.set()
            self._emit_status()

    def _apply_state(self, state: dict[str, Any]) -> None:
        snapshot = dict(self._snapshot)
        snapshot.update({key: value for key, value in state.items() if value is not None})
        snapshot["backend"] = self.name
        snapshot["server"] = {"host": self._host, "port": self._port, "version": self._version, "username": self._username}
        connected_at = self._snapshot.get("connected_at")
        snapshot["connected_at"] = connected_at
        snapshot["uptime"] = round(time.time() - connected_at, 1) if connected_at else 0.0
        self._snapshot = snapshot

    def _request(self, cmd: str, params: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
        if not self._proc or self._proc.poll() is not None:
            raise BackendError("Minecraft bridge is not running.")
        assert self._proc.stdin is not None
        request_id = self._next_id
        self._next_id += 1
        waiter: queue.Queue = queue.Queue(maxsize=1)
        self._pending[request_id] = waiter
        payload = json.dumps({"id": request_id, "cmd": cmd, "params": params})
        try:
            self._proc.stdin.write(payload + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            self._pending.pop(request_id, None)
            raise BackendError(f"Bridge stdin closed: {exc}") from exc
        try:
            response = waiter.get(timeout=timeout)
        except queue.Empty as exc:
            self._pending.pop(request_id, None)
            raise BackendError(f"Bridge command '{cmd}' timed out after {timeout:.0f}s") from exc
        return response

    def _profiles_folder(self, settings: MinecraftSettings) -> str | None:
        """Per-account token cache directory (premium accounts only)."""
        if settings.auth != "microsoft":
            return None
        base = Path(settings.profiles_folder).expanduser()
        if not base.is_absolute():
            base = Path.cwd() / base
        # One folder per account: tokens must never be shared between logins.
        safe = re.sub(r"[^A-Za-z0-9._@-]", "_", settings.username)[:64] or "account"
        folder = base / safe
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - depends on the filesystem
            self.log.add(f"Could not create the token cache folder {folder}: {exc}", "warn", "node")
            return str(base)
        return str(folder)

    def _fire(self, cmd: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Best-effort command: log failures instead of raising."""
        try:
            response = self._request(cmd, params or {}, timeout=5.0)
        except BackendError as exc:
            self.log.add(f"Bridge command '{cmd}' failed: {exc}", "warn", "node")
            return None
        if not response.get("ok"):
            self.log.add(f"Bridge command '{cmd}' rejected: {response.get('error')}", "warn", "node")
        return response

    # ------------------------------------------------------------ primitives
    def say(self, text: str) -> None:
        self._fire("say", {"text": text})

    def command(self, text: str) -> None:
        self._fire("command", {"text": text})

    def set_control(self, control: str, state: bool) -> None:
        if control not in CONTROLS:
            raise BackendError(f"unknown control: {control}")
        self._fire("control", {"name": control, "state": bool(state)})

    def look(self, yaw: float, pitch: float) -> None:
        self._fire("look", {"yaw": float(yaw), "pitch": float(pitch)})

    def stop_motion(self) -> None:
        self._fire("stop_motion", {})

    def pathfind_to(self, x: float, y: float, z: float, tolerance: float = 1.5) -> bool:
        if not self.supports_pathfinder:
            return False
        response = self._fire("pathfind_to", {"x": x, "y": y, "z": z, "range": tolerance})
        return bool(response and response.get("ok"))

    def stop_path(self) -> None:
        self._fire("stop_path", {})

    def swing_arm(self) -> bool:
        response = self._fire("swing_arm", {})
        return bool(response and response.get("ok"))

    def dig(self, block: str) -> bool:
        response = self._fire("dig", {"block": block, "timeout": 20})
        return bool(response and response.get("ok") and (response.get("result") or {}).get("dug"))

    def attack(self, player: str) -> bool:
        response = self._fire("attack", {"player": player})
        return bool(response and response.get("ok"))

    def snapshot(self) -> dict[str, Any]:
        snapshot = dict(self._snapshot)
        snapshot["backend"] = self.name
        snapshot["pathfinder"] = self.supports_pathfinder
        if self._proc is None or self._proc.poll() is not None:
            if snapshot["status"] == "connected":
                snapshot["status"] = "disconnected"
        return snapshot


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def create_backend(settings: AppSettings, log: EventLog) -> BaseBackend:
    """Build the backend requested by the config (``auto`` prefers Node)."""
    choice = settings.minecraft.backend
    if choice == "simulated":
        return SimulatedBackend(log)
    ok, reason = node_available()
    if choice == "node":
        if not ok:
            raise BackendError(reason)
        return NodeBridgeBackend(log)
    if ok:
        return NodeBridgeBackend(log)
    log.add(f"Falling back to the simulated world: {reason}", "warn", "backend")
    return SimulatedBackend(log)


def available_backends() -> dict[str, Any]:
    """Report which backends can run here (shown in the web UI)."""
    ok, reason = node_available()
    return {
        "node": {"available": ok, "reason": reason},
        "simulated": {"available": True, "reason": "always available"},
        "node_version": _node_version(),
        "python": sys.version.split()[0],
    }


def _node_version() -> str | None:
    binary = shutil.which("node")
    if not binary:
        return None
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None


def iter_controls() -> Iterable[str]:
    return CONTROLS


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")
