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
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pymc_bot import node_setup
from pymc_bot.config import AppSettings, MinecraftSettings
from pymc_bot.events import EventLog
from pymc_bot.inventory import (
    FOODS,
    INVENTORY_SIZE,
    Inventory,
    InventoryError,
    Stack,
    short,
    weapon_score,
)
from pymc_bot.skills import SKILL_COMMANDS

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
        "online_players": [],
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

    def view_image(self, width: int = 448) -> dict[str, Any]:
        """The bot's first-person view: ``{"image": base64, "mime": ...}`` (vision models).

        Raises :class:`BackendError` when the backend cannot render.
        """
        raise BackendError(f"the {self.name} backend cannot render what the bot sees")

    @abc.abstractmethod
    def dig(self, block: str) -> bool: ...

    @abc.abstractmethod
    def attack(self, player: str) -> bool: ...

    @abc.abstractmethod
    def snapshot(self) -> dict[str, Any]: ...

    # ------------------------------------------------- inventory & life actions
    #: Body actions beyond walking: hotbar, slots, armour, placing, sleeping, fighting mobs...
    #: Slots are numbered like a player sees them (0-8 hotbar, 9-35 backpack; pymc_bot.inventory).
    LIFE_COMMANDS: tuple[str, ...] = (
        "select_slot", "move_slot", "equip", "unequip", "swap_hands", "toss", "place", "use_item",
        "eat", "sleep", "wake", "collect", "attack_mob",
        *SKILL_COMMANDS,  # chop_tree, craft, harvest, deposit... (pymc_bot.skills)
    )

    def perform(self, cmd: str, params: dict[str, Any] | None = None, timeout: float = 10.0) -> tuple[bool, str, dict[str, Any]]:
        """Run one body action. Returns ``(ok, why_not, result)`` - never raises."""
        return False, f"the {self.name} backend cannot {cmd.replace('_', ' ')}", {}

    def eat(self, slot: int | None = None) -> bool:
        ok, _, _ = self.perform("eat", {} if slot is None else {"slot": slot}, timeout=10.0)
        return ok


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
        # A starter kit a player would have after a few minutes - a little untidy on purpose,
        # so the AI has a reason to organise the hotbar and put armour on.
        self._inv = Inventory()
        for slot, (name, count) in {
            0: ("iron_pickaxe", 1), 1: ("dirt", 2), 2: ("iron_sword", 1), 3: ("oak_log", 3),
            4: ("torch", 8), 12: ("bread", 3), 15: ("stone_axe", 1), 20: ("leather_helmet", 1),
        }.items():
            self._inv.slots[slot] = Stack(name, count)
        self._inv.selected = 2
        self._world_time = 1000.0  # Minecraft ticks: 0 sunrise, 13000 night, 24000 next day
        self._sleeping = False
        self._mobs: list[dict[str, Any]] = [dict(mob) for mob in self.MOBS]
        self._respawns: list[tuple[float, dict[str, Any]]] = []
        self._ground: list[dict[str, Any]] = [{"name": "oak_log", "count": 1, "x": 2.0, "z": 3.0, "at": 0.0}]
        self._placed: list[dict[str, Any]] = []
        # survival state the autopilot reacts to (the real bridge reports the same fields)
        self._oxygen = 20
        self._on_fire = False
        self._in_water = False
        self._dead = False
        self._riding = ""
        self._raining = False
        self._torches: list[tuple[float, float]] = []
        self._chest: dict[str, int] = {"bread": 4, "cobblestone": 32, "coal": 6}
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

        # --- day and night (20 ticks a second, like Minecraft) -------------------
        self._world_time = (self._world_time + 20.0 * dt) % 24000.0
        now = time.monotonic()
        with self._lock:
            for when, mob in list(self._respawns):
                if now >= when:
                    self._respawns.remove((when, mob))
                    self._mobs.append(mob)
            if not self._ground and self._rng.random() < 0.002:
                self._ground.append({"name": "oak_log", "count": 1, "x": self._rng.uniform(-8, 8), "z": self._rng.uniform(-8, 8), "at": 0.0})

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
        with self._lock:
            if self._inv.add(block, 1):
                self.log.add(f"Mined 1 {block}, but the inventory is full - it stays on the ground.", "warn", "sim")
                x, _, z = self._position
                self._ground.append({"name": block, "count": 1, "x": x, "z": z, "at": time.monotonic()})
                return True
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
        snap["online_players"] = list(self._players)
        with self._lock:
            # slots / hotbar / armour / off-hand + the flat list older code reads
            snap.update(self._inv.to_snapshot())
        snap["time_of_day"] = "night" if 13000 <= self._world_time < 23000 else "day"
        snap["world_time"] = int(self._world_time)
        snap["sleeping"] = self._sleeping
        snap["oxygen"] = self._oxygen
        snap["on_fire"] = self._on_fire
        snap["in_water"] = self._in_water
        snap["dead"] = self._dead
        snap["riding"] = self._riding
        snap["raining"] = self._raining
        x, _, z = self._position
        lit = any(math.dist((x, z), spot) <= 6 for spot in self._torches)
        snap["block_light"] = 12 if lit else 0
        snap["sky_light"] = 15
        snap["entities"] = self._entity_list()
        snap["ground_items"] = self._ground_items()
        snap["server"] = dict(self._server)
        snap["last_error"] = self._last_error
        snap["connected_at"] = self._connected_at
        snap["uptime"] = round(time.time() - self._connected_at, 1) if self._connected_at else 0.0
        return snap

    # The simulated neighbourhood: a couple of mobs, an animal and a dropped stack, so the
    # trained brain's entity and item inputs can be exercised without a Minecraft server.
    MOBS: tuple[dict[str, Any], ...] = (
        {"type": "minecraft:zombie", "x": -5.0, "y": GROUND_Y, "z": -9.0, "hostile": True, "health": 20.0},
        {"type": "minecraft:cow", "x": 9.0, "y": GROUND_Y, "z": 5.0, "hostile": False, "health": 10.0},
    )

    # --------------------------------------------------- inventory & life actions
    MOB_LOOT: dict[str, str] = {
        "zombie": "rotten_flesh", "husk": "rotten_flesh", "skeleton": "bone", "spider": "string",
        "creeper": "gunpowder", "cow": "beef", "pig": "porkchop", "sheep": "mutton", "chicken": "chicken",
    }

    def perform(self, cmd: str, params: dict[str, Any] | None = None, timeout: float = 10.0) -> tuple[bool, str, dict[str, Any]]:
        params = dict(params or {})
        handler = getattr(self, f"_do_{cmd}", None)
        if cmd not in self.LIFE_COMMANDS or handler is None:
            return False, f"unknown action {cmd!r}", {}
        if self._status != "connected":
            return False, "not connected", {}
        try:
            with self._lock:
                result = handler(params)
        except (InventoryError, ValueError, TypeError, KeyError) as exc:
            return False, str(exc).strip("'\"") or cmd, {}
        time.sleep(0.02)
        return True, "", result

    def _slot(self, params: dict[str, Any], key: str = "slot") -> int:
        index = int(params[key])
        if not 0 <= index < INVENTORY_SIZE:
            raise InventoryError(f"slot {index} does not exist (0-35)")
        if self._inv.slots[index] is None:
            raise InventoryError(f"slot {index} is empty")
        return index

    def _do_select_slot(self, params: dict[str, Any]) -> dict[str, Any]:
        self._inv.select(int(params["slot"]))
        return {"selected": self._inv.selected}

    def _do_move_slot(self, params: dict[str, Any]) -> dict[str, Any]:
        self._inv.move(self._slot(params, "from"), int(params["to"]))
        return {"moved": True}

    def _do_equip(self, params: dict[str, Any]) -> dict[str, Any]:
        index = self._slot(params)
        name = self._inv.slots[index].name
        self._inv.equip(index, str(params.get("destination") or "hand"))
        self.log.add(f"Equipped {name} ({params.get('destination') or 'hand'}).", "info", "sim")
        return {"equipped": name}

    def _do_unequip(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"slot": self._inv.unequip(str(params.get("destination") or ""))}

    def _do_swap_hands(self, params: dict[str, Any]) -> dict[str, Any]:
        self._inv.swap_hands()
        return {"swapped": True}

    def _do_toss(self, params: dict[str, Any]) -> dict[str, Any]:
        index = self._slot(params)
        stack = self._inv.slots[index]
        count = min(stack.count, int(params.get("count") or stack.count))
        name = stack.name
        stack.count -= count
        if stack.count <= 0:
            self._inv.slots[index] = None
        x, _, z = self._position
        self._ground.append({"name": name, "count": count, "x": x - math.sin(self._yaw) * 1.5,
                             "z": z + math.cos(self._yaw) * 1.5, "at": time.monotonic()})
        self.log.add(f"Dropped {count} {name}.", "info", "sim")
        return {"tossed": name, "count": count}

    def _do_place(self, params: dict[str, Any]) -> dict[str, Any]:
        index = self._slot(params)
        name = self._inv.slots[index].name
        self._inv.equip(index, "hand")
        self._inv.remove(name, 1)
        x, y, z = self._position
        spot = {"block": name, "x": math.floor(x - math.sin(self._yaw)), "y": int(y), "z": math.floor(z + math.cos(self._yaw))}
        self._placed.append(spot)
        self.log.add(f"Placed {name} at {spot['x']} {spot['y']} {spot['z']}.", "info", "sim")
        return {"placed": name, **{k: spot[k] for k in ("x", "y", "z")}}

    def _do_use_item(self, params: dict[str, Any]) -> dict[str, Any]:
        held = self._inv.held
        name = held.name if held else "hand"
        self.log.add(f"Used {name}.", "info", "sim")
        return {"used": name, "seconds": float(params.get("seconds") or 0.3)}

    def _do_eat(self, params: dict[str, Any]) -> dict[str, Any]:
        if params.get("slot") is not None:
            index = self._slot(params)
        else:
            index = self._inv.best_food(min(self._food, 14.0), self._health)
        if index is None:
            raise InventoryError("no food in the inventory")
        name = self._inv.slots[index].name
        if name not in FOODS:
            raise InventoryError(f"{name} is not food")
        if self._food >= 20 and name not in ("golden_apple", "enchanted_golden_apple", "chorus_fruit"):
            raise InventoryError("not hungry")
        stack = self._inv.slots[index]
        stack.count -= 1
        if stack.count <= 0:
            self._inv.slots[index] = None
        self._food = min(20.0, self._food + FOODS[name][0])
        if name in ("golden_apple", "enchanted_golden_apple"):
            self._health = min(20.0, self._health + 4.0)
        self.log.add(f"Ate {name}.", "success", "sim")
        return {"ate": True, "item": name}

    def _do_sleep(self, params: dict[str, Any]) -> dict[str, Any]:
        if not 12542 <= self._world_time < 23460:
            raise InventoryError("you can only sleep at night")
        self._world_time = 0.0  # the night passes for everyone
        self.log.add("Slept until morning.", "success", "sim")
        return {"sleeping": False, "slept": True}

    def _do_wake(self, params: dict[str, Any]) -> dict[str, Any]:
        self._sleeping = False
        return {"sleeping": False}

    def _do_collect(self, params: dict[str, Any]) -> dict[str, Any]:
        radius = float(params.get("radius") or 12)
        x, _, z = self._position
        now = time.monotonic()
        near = [d for d in self._ground if math.dist((x, z), (d["x"], d["z"])) <= radius and now - d["at"] > 2.0]
        if not near:
            raise InventoryError(f"nothing to pick up within {radius:g} blocks")
        picked = 0
        for drop in sorted(near, key=lambda d: math.dist((x, z), (d["x"], d["z"]))):
            left = self._inv.add(drop["name"], drop["count"])
            if left == drop["count"]:
                break
            picked += 1
            self._position[0], self._position[2] = drop["x"], drop["z"]
            if left:
                drop["count"] = left
            else:
                self._ground.remove(drop)
        if not picked:
            raise InventoryError("the inventory is full")
        self.log.add(f"Picked up {picked} item stack(s).", "info", "sim")
        return {"picked": picked}

    def _do_attack_mob(self, params: dict[str, Any]) -> dict[str, Any]:
        wanted = short(params.get("mob") or "")
        reach = float(params.get("range") or 16)
        x, y, z = self._position

        def kind(mob: dict[str, Any]) -> str:
            return short(mob["type"])

        candidates = [
            m for m in self._mobs
            if math.dist((x, z), (m["x"], m["z"])) <= reach and (kind(m) == wanted if wanted else m["hostile"])
        ]
        if not candidates:
            raise InventoryError(f"no {wanted or 'hostile mob'} within {reach:g} blocks")
        target = min(candidates, key=lambda m: math.dist((x, z), (m["x"], m["z"])))
        angle = math.atan2(x - target["x"], z - target["z"])
        self._position[0] = target["x"] + math.sin(angle) * 2.0
        self._position[2] = target["z"] + math.cos(angle) * 2.0
        held = self._inv.held
        damage = max(1.0, weapon_score(held.name) - 6.0) if held else 1.0
        hits = math.ceil(target["health"] / damage)
        self._mobs.remove(target)
        self._respawns.append((time.monotonic() + 60.0, target))
        loot = self.MOB_LOOT.get(kind(target))
        if loot:
            self._ground.append({"name": loot, "count": 1, "x": target["x"], "z": target["z"], "at": 0.0})
        self.log.add(f"Killed a {kind(target)} ({hits} hits).", "success", "sim")
        return {"target": kind(target), "hits": hits, "killed": True}

    # ------------------------------------------------------------- skills (pymc_bot.skills)
    # Pretend versions of the bridge's skill commands: they change the inventory the way the real
    # thing would, so the AI, the autopilot and the panel can be tried without a server.
    SIM_ORES: dict[str, str] = {
        "coal": "coal", "iron": "raw_iron", "copper": "raw_copper", "gold": "raw_gold", "redstone": "redstone",
        "lapis": "lapis_lazuli", "diamond": "diamond", "emerald": "emerald", "quartz": "quartz",
    }
    SIM_RECIPES: dict[str, tuple[int, dict[str, int]]] = {
        "stick": (4, {"#planks": 2}),
        "crafting_table": (1, {"#planks": 4}),
        "chest": (1, {"#planks": 8}),
        "torch": (4, {"#coal": 1, "stick": 1}),
        "wooden_pickaxe": (1, {"#planks": 3, "stick": 2}),
        "wooden_sword": (1, {"#planks": 2, "stick": 1}),
        "wooden_axe": (1, {"#planks": 3, "stick": 2}),
        "stone_pickaxe": (1, {"cobblestone": 3, "stick": 2}),
        "stone_sword": (1, {"cobblestone": 2, "stick": 1}),
        "stone_axe": (1, {"cobblestone": 3, "stick": 2}),
        "iron_pickaxe": (1, {"iron_ingot": 3, "stick": 2}),
        "iron_sword": (1, {"iron_ingot": 2, "stick": 1}),
        "furnace": (1, {"cobblestone": 8}),
        "bread": (1, {"wheat": 3}),
        "bucket": (1, {"iron_ingot": 3}),
        "shield": (1, {"#planks": 6, "iron_ingot": 1}),
        "white_bed": (1, {"white_wool": 3, "#planks": 3}),
    }
    SIM_SMELT: dict[str, str] = {
        "raw_iron": "iron_ingot", "raw_gold": "gold_ingot", "raw_copper": "copper_ingot", "beef": "cooked_beef",
        "porkchop": "cooked_porkchop", "chicken": "cooked_chicken", "mutton": "cooked_mutton", "cod": "cooked_cod",
        "salmon": "cooked_salmon", "rabbit": "cooked_rabbit", "potato": "baked_potato", "sand": "glass",
        "cobblestone": "stone", "oak_log": "charcoal", "kelp": "dried_kelp", "clay_ball": "brick",
    }
    SIM_FUELS: dict[str, float] = {"coal": 8, "charcoal": 8, "coal_block": 80, "lava_bucket": 100, "blaze_rod": 12,
                                   "#planks": 1.5, "#log": 1.5, "stick": 0.5}
    SIM_BREED_FOOD: dict[str, tuple[str, ...]] = {
        "cow": ("wheat",), "sheep": ("wheat",), "goat": ("wheat",), "pig": ("carrot", "potato", "beetroot"),
        "chicken": ("wheat_seeds", "beetroot_seeds", "melon_seeds", "pumpkin_seeds"), "rabbit": ("carrot", "dandelion"),
    }

    def _has(self, name: str) -> str | None:
        """First inventory item matching ``name`` (exact, ``#planks`` style tag, or a word)."""
        for _, stack in self._inv.items():
            if name.startswith("#") and stack.name.endswith("_" + name[1:]) or stack.name == name.lstrip("#"):
                return stack.name
        if name in ("#coal",):
            return self._has("coal") or self._has("charcoal")
        return None

    def _tag_count(self, name: str) -> int:
        if name.startswith("#"):
            return sum(stack.count for _, stack in self._inv.items()
                       if stack.name.endswith("_" + name[1:]) or (name == "#coal" and stack.name in ("coal", "charcoal")))
        return self._inv.count(name)

    def _take(self, name: str, count: int) -> None:
        left = count
        for _, stack in list(self._inv.items()):
            if left <= 0:
                break
            if name.startswith("#") and (stack.name.endswith("_" + name[1:]) or (name == "#coal" and stack.name in ("coal", "charcoal"))) \
                    or stack.name == name:
                left -= self._inv.remove(stack.name, left)
        if left > 0:
            raise InventoryError(f"not enough {name.lstrip('#')}")

    def _give(self, name: str, count: int) -> int:
        left = self._inv.add(name, count)
        if left:
            x, _, z = self._position
            self._ground.append({"name": name, "count": left, "x": x, "z": z, "at": time.monotonic()})
        return count - left

    def _need(self, name: str, why: str) -> str:
        found = self._has(name) or next((s.name for _, s in self._inv.items() if s.name.endswith(name)), None)
        if not found:
            raise InventoryError(f"I need {why}")
        return found

    def _do_drink(self, params: dict[str, Any]) -> dict[str, Any]:
        wanted = short(params.get("item") or "")
        names = [wanted] if wanted else ["potion", "milk_bucket", "honey_bottle"]
        for name in names:
            item = next((s.name for _, s in self._inv.items() if s.name == name or name in s.name), None)
            if item:
                self._inv.remove(item, 1)
                if item == "milk_bucket":
                    self._give("bucket", 1)
                elif item in ("potion", "honey_bottle"):
                    self._give("glass_bottle", 1)
                    self._health = min(20.0, self._health + 4.0)
                return {"drank": item, "message": f"drank {item.replace('_', ' ')}"}
        raise InventoryError("nothing to drink")

    def _do_surface(self, params: dict[str, Any]) -> dict[str, Any]:
        self._oxygen, self._in_water = 20, False
        return {"message": "swam up for air"}

    def _do_extinguish(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._on_fire:
            raise InventoryError("not on fire")
        self._on_fire = False
        how = "a water bucket" if self._inv.find("water_bucket") is not None else "the nearest water"
        return {"message": f"put the fire out with {how}"}

    def _do_respawn(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._dead:
            raise InventoryError("not dead")
        self._dead, self._health, self._food = False, 20.0, 20.0
        self._position = [0.5, self.GROUND_Y, 0.5]
        return {"message": "respawned"}

    def _do_find_block(self, params: dict[str, Any]) -> dict[str, Any]:
        block = short(params.get("block") or "oak_log")
        rng = random.Random(block)
        x, y, z = self._position
        spot = {"x": math.floor(x + rng.uniform(-20, 20)), "y": int(y) - (rng.randint(5, 40) if "ore" in block else 0),
                "z": math.floor(z + rng.uniform(-20, 20))}
        distance = round(math.dist((x, y, z), (spot["x"], spot["y"], spot["z"])), 1)
        return {"found": [{**spot, "block": block, "distance": distance}],
                "message": f"{block} at {spot['x']} {spot['y']} {spot['z']} ({distance:.0f} blocks away)"}

    def _mine_into_inventory(self, item: str, count: int, what: str) -> dict[str, Any]:
        got = self._give(item, count)
        self.log.add(f"Mined {count} {what}.", "success", "sim")
        return {"mined": count, "item": item, "message": f"mined {got} {what.replace('_', ' ')}"}

    def _do_chop_tree(self, params: dict[str, Any]) -> dict[str, Any]:
        logs = min(int(params.get("count") or 16), self._rng.randint(4, 7))
        return self._mine_into_inventory("oak_log", logs, "oak_log") | {"message": f"chopped a tree: {logs} oak logs"}

    def _do_mine_ore(self, params: dict[str, Any]) -> dict[str, Any]:
        ore = short(params.get("ore") or "coal").replace("_ore", "").replace("deepslate_", "")
        item = self.SIM_ORES.get(ore)
        if item is None:
            raise InventoryError(f"no {ore} ore around")
        if ore not in ("coal",) and not any(s.name.endswith("_pickaxe") and not s.name.startswith(("wooden", "golden"))
                                            for _, s in self._inv.items()):
            raise InventoryError(f"{ore} ore needs a stone pickaxe or better")
        count = int(params.get("count") or 4)
        return self._mine_into_inventory(item, count, f"{ore}_ore")

    def _do_gather(self, params: dict[str, Any]) -> dict[str, Any]:
        block = short(params.get("block") or "dirt")
        drops = {"stone": "cobblestone", "grass_block": "dirt", "deepslate": "cobbled_deepslate"}.get(block, block)
        return self._mine_into_inventory(drops, int(params.get("count") or 8), block)

    def _do_harvest(self, params: dict[str, Any]) -> dict[str, Any]:
        crops = self._rng.randint(3, 8)
        self._give("wheat", crops)
        self._give("wheat_seeds", crops)
        if params.get("replant", True):
            self._inv.remove("wheat_seeds", crops)
        return {"harvested": crops, "mined": crops, "message": f"harvested {crops} wheat" + (" and replanted" if params.get("replant", True) else "")}

    def _do_plant(self, params: dict[str, Any]) -> dict[str, Any]:
        seeds = ("wheat_seeds", "carrot", "potato", "beetroot_seeds", "melon_seeds", "pumpkin_seeds", "nether_wart")
        wanted = short(params.get("seed") or "")
        seed = wanted if wanted and self._inv.count(wanted) else next((s for s in seeds if self._inv.count(s)), None)
        if not seed:
            raise InventoryError("no seeds to plant")
        planted = min(self._inv.count(seed), int(params.get("count") or 16), 9)
        self._inv.remove(seed, planted)
        return {"planted": planted, "message": f"planted {planted} {seed.replace('_', ' ')}"}

    def _do_till(self, params: dict[str, Any]) -> dict[str, Any]:
        self._need("_hoe", "a hoe")
        count = min(int(params.get("count") or 8), 9)
        return {"tilled": count, "message": f"tilled {count} blocks of farmland"}

    def _do_bonemeal(self, params: dict[str, Any]) -> dict[str, Any]:
        self._need("bone_meal", "bone meal")
        used = min(self._inv.count("bone_meal"), int(params.get("count") or 4))
        self._inv.remove("bone_meal", used)
        return {"used": used, "message": f"used {used} bone meal on the crops"}

    def _do_fish(self, params: dict[str, Any]) -> dict[str, Any]:
        self._need("fishing_rod", "a fishing rod")
        caught = max(1, int(params.get("seconds") or 60) // 25)
        self._give("cod", caught)
        return {"caught": caught, "message": f"caught {caught} fish"}

    def _do_shear(self, params: dict[str, Any]) -> dict[str, Any]:
        self._need("shears", "shears")
        self._give("white_wool", 3)
        return {"sheared": 1, "message": "sheared a sheep: 3 white wool"}

    def _do_milk(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._inv.find("bucket") is None:
            raise InventoryError("I need an empty bucket")
        if not any(short(m["type"]) in ("cow", "mooshroom", "goat") for m in self._mobs):
            raise InventoryError("no cow nearby")
        self._inv.remove("bucket", 1)
        self._give("milk_bucket", 1)
        return {"message": "milked a cow"}

    def _do_breed(self, params: dict[str, Any]) -> dict[str, Any]:
        animal = short(params.get("animal") or "cow")
        foods = self.SIM_BREED_FOOD.get(animal)
        if not foods:
            raise InventoryError(f"I don't know what {animal}s eat")
        food = next((f for f in foods if self._inv.count(f) >= 2), None)
        if food is None:
            raise InventoryError(f"I need 2 {foods[0]} to breed {animal}s")
        self._inv.remove(food, 2)
        return {"fed": 2, "message": f"fed two {animal}s with {food} - a baby is on the way"}

    def _do_craft(self, params: dict[str, Any]) -> dict[str, Any]:
        item = short(params.get("item") or "")
        count = max(1, int(params.get("count") or 1))
        if not item:
            raise InventoryError("craft what? (item)")
        if item.endswith("_planks"):
            wood = item[: -len("_planks")]
            log_name = next((s.name for _, s in self._inv.items() if s.name in (f"{wood}_log", f"{wood}_stem")), None) \
                or next((s.name for _, s in self._inv.items() if s.name.endswith(("_log", "_stem"))), None)
            if log_name is None:
                raise InventoryError("I need logs to make planks")
            times = min(math.ceil(count / 4), self._inv.count(log_name))
            self._inv.remove(log_name, times)
            made = self._give(item, times * 4)
            return {"crafted": made, "item": item, "message": f"crafted {made} {item.replace('_', ' ')}"}
        recipe = self.SIM_RECIPES.get(item)
        if recipe is None:
            raise InventoryError(f"no recipe for {item} (in the simulated world)")
        per, needs = recipe
        times = math.ceil(count / per)
        for name, amount in needs.items():
            if self._tag_count(name) < amount * times:
                raise InventoryError(f"not enough {name.lstrip('#')} for {item} (need {amount * times})")
        for name, amount in needs.items():
            self._take(name, amount * times)
        made = self._give(item, per * times)
        return {"crafted": made, "item": item, "message": f"crafted {made} {item.replace('_', ' ')}"}

    def _do_smelt(self, params: dict[str, Any]) -> dict[str, Any]:
        item = short(params.get("item") or "")
        if not item:
            item = next((s.name for _, s in self._inv.items() if s.name in self.SIM_SMELT), "")
        result = self.SIM_SMELT.get(item)
        if not item:
            raise InventoryError("nothing to smelt")
        if result is None:
            raise InventoryError(f"{item} cannot be smelted")
        count = min(self._inv.count(item), int(params.get("count") or 8))
        if not count:
            raise InventoryError(f"I have no {item}")
        fuel = next((f for f in self.SIM_FUELS if self._tag_count(f) and f.lstrip("#") != item), None)
        if fuel is None:
            raise InventoryError("no fuel (coal, charcoal, planks or logs)")
        self._take(fuel, min(self._tag_count(fuel), math.ceil(count / self.SIM_FUELS[fuel])))
        self._inv.remove(item, count)
        made = self._give(result, count)
        return {"smelted": made, "item": result, "message": f"smelted {made} {item.replace('_', ' ')} into {result.replace('_', ' ')}"}

    def _keep(self, name: str) -> bool:
        from pymc_bot.skills import is_tool

        return is_tool(name) or name in FOODS or name in ("torch", "shield", "totem_of_undying", "bucket", "water_bucket")

    def _do_deposit(self, params: dict[str, Any]) -> dict[str, Any]:
        wanted = short(params.get("item") or "")
        stored = 0
        for index, stack in list(self._inv.items()):
            if wanted and wanted not in stack.name:
                continue
            if not wanted and not params.get("all") and self._keep(stack.name):
                continue
            self._chest[stack.name] = self._chest.get(stack.name, 0) + stack.count
            stored += stack.count
            self._inv.slots[index] = None
        if not stored:
            raise InventoryError("nothing to put away")
        return {"stored": stored, "message": f"put {stored} items into the chest"}

    def _do_withdraw(self, params: dict[str, Any]) -> dict[str, Any]:
        wanted = short(params.get("item") or "")
        name = next((n for n in self._chest if n == wanted), None) or next((n for n in self._chest if wanted in n), None)
        if not name:
            raise InventoryError(f"the chest has no {wanted or 'items'}")
        count = min(self._chest[name], int(params.get("count") or 16))
        got = self._give(name, count)
        self._chest[name] -= got
        if self._chest[name] <= 0:
            del self._chest[name]
        return {"took": got, "item": name, "message": f"took {got} {name.replace('_', ' ')} from the chest"}

    def _do_chest_contents(self, params: dict[str, Any]) -> dict[str, Any]:
        items = [{"name": n, "count": c} for n, c in sorted(self._chest.items())]
        summary = ", ".join(f"{i['count']} {i['name']}" for i in items) or "nothing"
        return {"items": items, "message": f"the chest has {summary}"}

    def _do_trade(self, params: dict[str, Any]) -> dict[str, Any]:
        offers = [{"index": 0, "give": "emerald x1", "get": "bread x6"}, {"index": 1, "give": "wheat x20", "get": "emerald x1"}]
        index = params.get("index")
        if index is None:
            return {"trades": offers, "message": "the villager offers: " + "; ".join(f"#{o['index']} {o['give']} -> {o['get']}" for o in offers)}
        index = int(index)
        times = int(params.get("times") or 1)
        if index == 0:
            self._take("emerald", times)
            self._give("bread", 6 * times)
        elif index == 1:
            self._take("wheat", 20 * times)
            self._give("emerald", times)
        else:
            raise InventoryError(f"the villager has no offer #{index}")
        return {"traded": times, "message": f"traded offer #{index} {times} time(s)"}

    def _place_torches(self, count: int) -> int:
        have = self._inv.count("torch")
        if not have:
            raise InventoryError("I have no torches")
        placed = min(have, count)
        self._inv.remove("torch", placed)
        x, y, z = self._position
        self._torches.extend((x + i, z) for i in range(placed))
        return placed

    def _do_place_torch(self, params: dict[str, Any]) -> dict[str, Any]:
        self._place_torches(1)
        return {"placed_count": 1, "message": "placed a torch"}

    def _do_light_up(self, params: dict[str, Any]) -> dict[str, Any]:
        placed = self._place_torches(min(int(params.get("count") or 6), 4))
        return {"placed_count": placed, "message": f"placed {placed} torches on dark spots"}

    def _building_stack(self, wanted: str) -> str:
        if wanted and self._inv.count(wanted):
            return wanted
        index = self._inv.building_block()
        if index is None:
            raise InventoryError("I have no blocks to build with")
        return self._inv.slots[index].name

    def _do_build_shelter(self, params: dict[str, Any]) -> dict[str, Any]:
        block = self._building_stack(short(params.get("block") or ""))
        placed = min(17, self._inv.count(block))
        self._inv.remove(block, placed)
        whole = placed >= 17
        return {"placed_count": placed, "complete": whole,
                "message": f"built a shelter from {placed} {block}" + ("" if whole else " (ran out of blocks - it has gaps)")}

    def _do_pillar_up(self, params: dict[str, Any]) -> dict[str, Any]:
        block = self._building_stack(short(params.get("block") or ""))
        height = min(int(params.get("height") or 4), self._inv.count(block))
        self._inv.remove(block, height)
        self._position[1] += height
        return {"placed_count": height, "message": f"towered up {height} blocks"}

    def _do_activate(self, params: dict[str, Any]) -> dict[str, Any]:
        block = short(params.get("block") or "door")
        return {"message": f"used the {block.replace('_', ' ')}"}

    def _do_ride(self, params: dict[str, Any]) -> dict[str, Any]:
        self._riding = "oak_boat"
        return {"riding": self._riding, "message": "got into an oak boat"}

    def _do_dismount(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._riding:
            raise InventoryError("not riding anything")
        self._riding = ""
        return {"message": "got off"}

    def _do_shoot(self, params: dict[str, Any]) -> dict[str, Any]:
        self._need("bow", "a bow")
        if not self._inv.count("arrow"):
            raise InventoryError("I have no arrows")
        shots = min(int(params.get("shots") or 3), self._inv.count("arrow"))
        result = self._do_attack_mob({"mob": params.get("mob"), "range": params.get("range") or 24})
        self._inv.remove("arrow", shots)
        return {**result, "shots": shots, "kills": 1, "message": f"shot a {result['target']} ({shots} arrows)"}

    def _entity_list(self) -> list[dict[str, Any]]:
        """Everything in range, in the same shape the mod and the bridge report it."""
        x, y, z = self._position
        entries: list[dict[str, Any]] = []
        for player in self._player_list():
            entries.append(
                {
                    "type": "minecraft:player",
                    "dx": round(player["x"] - x, 2),
                    "dy": round(player["y"] - y, 2),
                    "dz": round(player["z"] - z, 2),
                    "dist": player["distance"],
                    "hostile": False,
                    "player": True,
                    "health": 20.0,
                    "on_ground": True,
                    "count": 0,
                    "held_item": "minecraft:iron_sword",
                }
            )
        for mob in list(self._mobs):
            dx, dy, dz = mob["x"] - x, mob["y"] - y, mob["z"] - z
            entries.append(
                {
                    "type": mob["type"],
                    "dx": round(dx, 2),
                    "dy": round(dy, 2),
                    "dz": round(dz, 2),
                    "dist": round(math.sqrt(dx * dx + dy * dy + dz * dz), 2),
                    "hostile": bool(mob["hostile"]),
                    "player": False,
                    "health": mob["health"],
                    "on_ground": True,
                    "count": 0,
                    "held_item": "",
                }
            )
        entries.sort(key=lambda entry: entry["dist"])
        return entries[:24]

    def _ground_items(self) -> list[dict[str, Any]]:
        x, _, z = self._position
        items = [
            {"item": f"minecraft:{drop['name']}", "count": drop["count"], "dist": round(math.dist((x, z), (drop["x"], drop["z"])), 2)}
            for drop in list(self._ground)
        ]
        items.sort(key=lambda item: item["dist"])
        return items[:12]

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
SETUP_HINT = "run `python -m pymc_bot setup` or press Install in the panel (no npm needed)"


def node_module_dirs() -> list[Path]:
    """Where the bridge finds mineflayer: next to it, the working folder, the checkout, ~/.pymc_bot/runtime."""
    dirs = [NODE_DIR / "node_modules", Path.cwd() / "node_modules"]
    for path in node_setup.node_modules_dirs():
        if path not in dirs and path.parent != node_setup.NODE_DIR:
            dirs.append(path)
    return dirs


def find_node_binary() -> str | None:
    """``node`` on PATH, or the portable Node.js that ``pymc_bot setup`` downloaded."""
    return node_setup.find_node()


def node_available() -> tuple[bool, str]:
    """Check for node + the mineflayer dependency. Returns (ok, reason)."""
    node = find_node_binary()
    if not node:
        return False, f"Node.js was not found (needed for the real Minecraft protocol) - {SETUP_HINT}."
    if not BRIDGE_SCRIPT.exists():
        return False, f"Bridge script missing: {BRIDGE_SCRIPT}"
    if not any((path / "mineflayer").exists() for path in node_module_dirs()):
        return False, f"mineflayer is not installed - {SETUP_HINT}."
    return True, "ok"


_TESTED_VERSIONS = re.compile(r"testedVersions\s*=\s*\[([^\]]*)\]")


def mineflayer_versions() -> dict[str, Any] | None:
    """Minecraft versions the installed mineflayer supports, read from ``mineflayer/lib/version.js``.

    Returns ``{"oldest": "1.8.8", "latest": "26.1", "tested": [...]}`` or ``None`` when mineflayer is
    not installed. Shown in the panel and ``doctor`` so a too-new server is obvious before joining.
    """
    for root in node_module_dirs():
        path = root / "mineflayer" / "lib" / "version.js"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        match = _TESTED_VERSIONS.search(text)
        if not match:
            continue
        tested = re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))
        if tested:
            return {"oldest": tested[0], "latest": tested[-1], "tested": tested}
    return None


_NO_VERSION_DATA = re.compile(
    r"No data available for version (?P<name>[^\s'\"]+)"
    r"|Unsupported protocol version '?(?P<protocol>[\w.]+)'?"
)
_OUTDATED_CLIENT = re.compile(
    r"outdated[_ ]client|incompatible[_ ]client|outdated client|client is outdated|please use|"
    r"multiplayer\.disconnect\.(?:outdated|incompatible)",
    re.IGNORECASE,
)
_VIA_HINT = "install ViaVersion + ViaBackwards on the server"
_ASLEEP_KICK = re.compile(
    r"server is (?:currently )?(offline|starting|sleeping|stopping|asleep)|server is not (?:running|online)",
    re.IGNORECASE,
)
_ONLINE_MODE = re.compile(
    r"failed to verify username|unverified_username|not authenticated with|invalid session|"
    r"authservers? (?:are )?down|premium (?:account )?(?:is )?required|online[- ]mode",
    re.IGNORECASE,
)
_WHITELIST = re.compile(r"not white-?listed|multiplayer\.disconnect\.not_whitelisted", re.IGNORECASE)
_THROTTLE = re.compile(r"connection throttled|multiplayer\.disconnect\.throttled|logging in too fast", re.IGNORECASE)
_DUPLICATE = re.compile(r"logged in from another location|multiplayer\.disconnect\.duplicate_login", re.IGNORECASE)
_BANNED = re.compile(r"multiplayer\.disconnect\.banned|you are banned|banned from this server", re.IGNORECASE)


def explain_connection_error(
    message: str,
    *,
    host: str | None = None,
    port: int | None = None,
    latest: str | None = None,
    fallback: dict[str, Any] | None = None,
    asleep: dict[str, Any] | None = None,
) -> str:
    """Turn a raw mineflayer / Node error into a sentence that says what to do.

    Unknown messages are returned unchanged. ``latest`` is the newest Minecraft version mineflayer
    supports; ``fallback`` is the bridge's ``version_fallback`` event when the bot joined as an older
    version because the server is newer than mineflayer.
    """
    from pymc_bot.hosting import (
        GENERIC_CRACKED_HINT,
        GENERIC_START_HINT,
        GENERIC_WHITELIST_HINT,
        detect_host,
    )

    text = str(message or "").strip()
    where = f"{host}:{port}" if host and port else (host or "the server")
    newest = latest or (fallback or {}).get("latest") or "an older version"
    profile = detect_host(host or "", str((asleep or {}).get("motd") or text))
    named = f" ({profile.name})" if profile else ""

    if asleep or _ASLEEP_KICK.search(text):
        state = (asleep or {}).get("state")
        if not state:
            found = _ASLEEP_KICK.search(text)
            state = (found.group(1) if found and found.group(1) else "offline").lower()
        motd = (asleep or {}).get("motd") or ""
        shows = f' - it shows "{motd}"' if motd else ""
        start = profile.start_hint if profile else GENERIC_START_HINT
        return (
            f"{where}{named} is {state}{shows}. {start} "
            "Tip: turn on 'Wait for the server' and the bot joins as soon as it is up."
        )
    if _ONLINE_MODE.search(text):
        hint = profile.cracked_hint if profile else GENERIC_CRACKED_HINT
        return f"The server refused the offline-mode login. {hint} (Server said: {text})"
    if _WHITELIST.search(text):
        hint = (profile.whitelist_hint if profile and profile.whitelist_hint else GENERIC_WHITELIST_HINT)
        return f"The bot is not on the whitelist. {hint}"
    if _THROTTLE.search(text):
        return (
            "The server limits how fast players may (re)join. Wait a few seconds before connecting again; "
            "when adding many bots raise 'Gap between joins' to 4 s or more."
        )
    if _DUPLICATE.search(text):
        return "Someone else joined with the same name, so the server kicked this bot. Give every bot its own name."
    if _BANNED.search(text):
        return f"This bot name is banned on the server. (Server said: {text})"

    match = _NO_VERSION_DATA.search(text)
    if match:
        server = match.group("name") or (fallback or {}).get("server_version") or "this version"
        if match.group("protocol") and not match.group("name"):
            server = (fallback or {}).get("server_version") or f"protocol {match.group('protocol')}"
        return (
            f"The server runs Minecraft {server}, but the installed bot library (mineflayer) only supports up to "
            f"{newest}. To join anyway, {_VIA_HINT} (or run the server on {newest}) and set Version to auto. "
            f"Support for {server} arrives with a mineflayer update - run the updater or `npm update` later."
        )
    if _OUTDATED_CLIENT.search(text):
        server = (fallback or {}).get("server_version")
        using = (fallback or {}).get("using")
        joined = f" as Minecraft {using}" if using else ""
        runs = f"The server runs {server} and refused" if server else "The server refused"
        return (
            f"{runs} the bot because it joined{joined} - the installed mineflayer supports up to {newest}. "
            f"Fix: {_VIA_HINT} so older clients can join, or run the server on {newest}. (Server said: {text})"
        )
    if "ECONNREFUSED" in text or "WinError 10061" in text:
        return (
            f"Nothing is accepting connections at {where} - is the Minecraft server running, "
            "and are the host and port right?"
        )
    if "ENOTFOUND" in text or "EAI_AGAIN" in text:
        return f"Could not find the host '{host or text.split()[-1]}' - check the Host field ({text})."
    if "ETIMEDOUT" in text or "EHOSTUNREACH" in text:
        return f"{where} did not answer in time - check the address, the firewall, and that the server is online."
    return text


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
        self.node_binary = node_binary or find_node_binary() or "node"
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
        self._fallback: dict[str, Any] | None = None  # last version_fallback event from the bridge
        self._asleep: dict[str, Any] | None = None  # last server_asleep event (free host placeholder)
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
                # mineflayer installed by `pymc_bot setup` into ~/.pymc_bot/runtime is found via NODE_PATH
                env=node_setup.node_env(),
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
        self._fallback = None
        self._asleep = None
        self._snapshot.pop("server_state", None)
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
            message = self._explain(response.get("error") or "unknown error")
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
            minecraft = self._caps.get("minecraft") or {}
            supports = (
                f" Supports Minecraft {minecraft['oldest']} - {minecraft['latest']}."
                if minecraft.get("oldest") and minecraft.get("latest") else ""
            )
            self.log.add(
                ("Bridge ready (pathfinder available)." if self._caps.get("pathfinder")
                 else "Bridge ready (no pathfinder - Python steering will be used).") + supports,
                "info",
                "node",
            )
            self._ready.set()
        elif event == "server_asleep":
            self._asleep = dict(payload)
            self._snapshot["server_state"] = str(payload.get("state") or "offline")
        elif event == "version_fallback":
            self._fallback = dict(payload)
            # The bridge logs the details itself; remember them to explain a kick that may follow.
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
            raw = str(payload.get("reason") or event)
            reason = self._explain(raw)
            self._connected = False
            if event == "end" and self._snapshot.get("status") == "error" and self._snapshot.get("last_error"):
                # mineflayer always ends with 'end' after an error/kick; keep the real cause visible.
                self._spawned.set()
                self.log.add(f"Connection closed ({raw}).", "debug", "node")
                self._emit_status()
                return
            self._snapshot["status"] = "disconnected" if event == "end" else "error"
            self._snapshot["last_error"] = reason
            self._spawned.set()
            self.log.add(f"Disconnected from server: {reason}", "warn", "node")
            self._emit_status()
        elif event == "error":
            raw = str(payload.get("message", "unknown error"))
            message = self._explain(raw)
            self._connect_error = message
            self._snapshot["status"] = "error"
            self._snapshot["last_error"] = message
            self.log.add(f"Bridge error: {message}", "error", "node")
            if message != raw:
                self.log.add(f"Bridge error details: {raw}", "debug", "node")
            self._spawned.set()
            self._emit_status()

    def _explain(self, message: str) -> str:
        minecraft = (self._caps or {}).get("minecraft") or mineflayer_versions() or {}
        return explain_connection_error(
            message,
            host=self._host,
            port=self._port,
            latest=minecraft.get("latest"),
            fallback=self._fallback,
            asleep=self._asleep,
        )

    @property
    def server_asleep(self) -> dict[str, Any] | None:
        """The ``server_asleep`` details when the last connect found a sleeping/starting server."""
        return dict(self._asleep) if self._asleep else None

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

    def _fire(self, cmd: str, params: dict[str, Any] | None = None, timeout: float = 5.0) -> dict[str, Any] | None:
        """Best-effort command: log failures instead of raising."""
        try:
            response = self._request(cmd, params or {}, timeout=timeout)
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
        # The bridge walks to the block and digs it: give it the whole 20 s budget (+ slack).
        response = self._fire("dig", {"block": block, "timeout": 20}, timeout=25.0)
        return bool(response and response.get("ok") and (response.get("result") or {}).get("dug"))

    def attack(self, player: str) -> bool:
        response = self._fire("attack", {"player": player})
        return bool(response and response.get("ok"))

    def view_image(self, width: int = 448) -> dict[str, Any]:
        response = self._request("snapshot", {"width": int(width)}, timeout=20.0)
        if not response.get("ok"):
            raise BackendError(str(response.get("error") or "snapshot failed"))
        result = response.get("result") or {}
        if not result.get("image"):
            raise BackendError("the bridge returned no picture")
        return result

    def perform(self, cmd: str, params: dict[str, Any] | None = None, timeout: float = 10.0) -> tuple[bool, str, dict[str, Any]]:
        if cmd not in self.LIFE_COMMANDS:
            return False, f"unknown action {cmd!r}", {}
        try:
            response = self._request(cmd, params or {}, timeout=timeout)
        except BackendError as exc:
            return False, str(exc), {}
        if not response.get("ok"):
            return False, str(response.get("error") or f"{cmd} failed"), {}
        result = response.get("result")
        return True, "", result if isinstance(result, dict) else {}

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
        "minecraft_versions": mineflayer_versions() if ok else None,
        "python": sys.version.split()[0],
    }


def _node_version() -> str | None:
    binary = find_node_binary()
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
