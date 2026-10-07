"""Observation -> feature vector encoding, shared by the trainer and live inference.

The mod (:file:`mod/`, ``pymcplaytime``) writes one observation per sample::

    {"tick": 40, "activity": "forward", "x": 12.5, "y": 64.0, "z": -3.2,
     "vx": 0.1, "vy": 0.0, "vz": -0.2, "moved": 0.21, "yaw": 91.0, "pitch": 4.5,
     "on_ground": true, "sprinting": false, "sneaking": false,
     "selected_slot": 0, "health": 20.0, "food": 20, "dimension": "overworld",
     "blocks": [[dx, dy, dz, "minecraft:stone"], ... 27 cells ...],
     "nearby": [{"type": "minecraft:zombie", "dx": 3.1, "dy": 0.0, "dz": -1.4,
                 "dist": 3.4, "hostile": true, "health": 20.0}, ...]}

:func:`encode` turns exactly that -- and the live bot snapshot, through the same
code path -- into a fixed length ``float`` vector. Because both sides share this
function, a checkpoint trained on recorded playtime can be run live without any
mapping layer drifting apart. Bumping the layout means bumping
:data:`FEATURE_VERSION`, which is written into every checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

FEATURE_VERSION = 1

#: The action vocabulary the mod records and the model predicts.
ACTION_SPACE: tuple[str, ...] = (
    "forward",
    "back",
    "left",
    "right",
    "jump",
    "sneak",
    "look",
    "attack",
    "use",
    "hold",
    "move",
    "none",
)

#: Numeric targets regressed next to the action: view turn and movement deltas.
TARGET_NAMES: tuple[str, ...] = ("dyaw", "dpitch", "forward", "strafe")

DIMENSIONS: tuple[str, ...] = ("overworld", "nether", "end", "other")

#: 3x3x3 block neighbourhood, innermost axis last (matches how the mod scans it).
CELLS: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz) for dy in (-1, 0, 1) for dz in (-1, 0, 1) for dx in (-1, 0, 1)
)

BLOCK_CLASSES: tuple[str, ...] = ("air", "solid", "liquid", "hazard", "resource")

NEARBY_SLOTS = 6
NEARBY_CLASSES: tuple[str, ...] = ("player", "hostile", "passive", "other")

HAZARD_BLOCKS = (
    "lava",
    "fire",
    "soul_fire",
    "magma_block",
    "cactus",
    "sweet_berry_bush",
    "powder_snow",
    "campfire",
)
LIQUID_BLOCKS = ("water", "flowing_water", "bubble_column", "seagrass", "kelp")
RESOURCE_BLOCKS = (
    "log",
    "wood",
    "planks",
    "ore",
    "coal",
    "iron",
    "gold",
    "diamond",
    "emerald",
    "redstone",
    "lapis",
    "copper",
    "crafting_table",
    "furnace",
    "chest",
    "torch",
)
AIR_BLOCKS = ("air", "cave_air", "void_air", "light", "barrier", "structure_void")
HOSTILE_ENTITIES = (
    "zombie",
    "skeleton",
    "creeper",
    "spider",
    "enderman",
    "witch",
    "slime",
    "pillager",
    "warden",
    "blaze",
    "ghast",
    "husk",
    "drowned",
    "phantom",
    "hoglin",
    "piglin_brute",
)
PASSIVE_ENTITIES = (
    "cow",
    "sheep",
    "pig",
    "chicken",
    "horse",
    "rabbit",
    "villager",
    "cat",
    "wolf",
    "fox",
    "bee",
    "goat",
    "axolotl",
    "parrot",
    "turtle",
)


def _short_name(name: Any) -> str:
    """``"minecraft:oak_log"`` -> ``"oak_log"``; tolerates full ids and namespaces."""
    text = str(name or "").strip().lower()
    return text.split(":")[-1] if ":" in text else text


def block_class(name: Any) -> int:
    """Map a block id onto one of :data:`BLOCK_CLASSES` (an index)."""
    short = _short_name(name)
    if not short or short in AIR_BLOCKS or short.endswith("_air"):
        return 0
    if short in HAZARD_BLOCKS or any(part in short for part in ("lava", "fire")):
        return 3
    if short in LIQUID_BLOCKS or any(part in short for part in ("water", "kelp", "seagrass")):
        return 2
    if any(part in short for part in RESOURCE_BLOCKS):
        return 4
    return 1


def entity_class(entry: dict[str, Any]) -> int:
    """Map a nearby entity onto one of :data:`NEARBY_CLASSES` (an index)."""
    if entry.get("player") or _short_name(entry.get("type")) in ("player", "remoteplayer"):
        return 0
    if bool(entry.get("hostile")):
        return 1
    short = _short_name(entry.get("type"))
    if any(part in short for part in PASSIVE_ENTITIES):
        return 2
    if any(part in short for part in HOSTILE_ENTITIES):
        return 1
    return 3


def default_blocks(ground: bool = True) -> list[list[Any]]:
    """A plausible neighbourhood for callers that do not know the world.

    The mineflayer bridge used by the live bot does not ship a block scan, so live
    inference feeds the same encoder a neutral cell layout: air everywhere except
    the ground directly below the player.
    """
    cells: list[list[Any]] = []
    for dx, dy, dz in CELLS:
        solid = ground and dy == -1
        cells.append([dx, dy, dz, "minecraft:stone" if solid else "minecraft:air"])
    return cells


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _number(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return fallback if math.isnan(number) or math.isinf(number) else number


def _dimension_index(name: Any) -> int:
    short = _short_name(name) if name else "overworld"
    for index, known in enumerate(DIMENSIONS):
        if known == short:
            return index
    return len(DIMENSIONS) - 1


def feature_dim(include_blocks: bool = True) -> int:
    """Length of the vector :func:`encode` returns."""
    base = 3 + 3 + 1 + 4 + 4 + 9 + 2 + len(DIMENSIONS) + 2 + NEARBY_SLOTS * (len(NEARBY_CLASSES) + 4)
    blocks = len(CELLS) * len(BLOCK_CLASSES) if include_blocks else 0
    return base + blocks


def encode(observation: dict[str, Any], include_blocks: bool = True) -> list[float]:
    """Encode one observation (mod sample or live bot snapshot) as a feature vector."""
    observation = observation or {}
    out: list[float] = []

    # Position (bounded so a long walk cannot blow up the inputs), velocity, distance moved.
    out.append(_clamp(_number(observation.get("x")) / 256.0, -4.0, 4.0))
    out.append(_clamp(_number(observation.get("y"), 64.0) / 128.0, -2.0, 8.0))
    out.append(_clamp(_number(observation.get("z")) / 256.0, -4.0, 4.0))
    out.append(_clamp(_number(observation.get("vx")) / 10.0, -2.0, 2.0))
    out.append(_clamp(_number(observation.get("vy")) / 10.0, -2.0, 2.0))
    out.append(_clamp(_number(observation.get("vz")) / 10.0, -2.0, 2.0))
    out.append(_clamp(_number(observation.get("moved")) / 2.0, 0.0, 1.0))

    # View, as sin/cos so 359 degrees and 1 degree are neighbours.
    yaw = math.radians(_number(observation.get("yaw")))
    pitch = math.radians(_number(observation.get("pitch")))
    out.append(math.sin(yaw))
    out.append(math.cos(yaw))
    out.append(math.sin(pitch))
    out.append(_clamp(_number(observation.get("pitch")) / 90.0, -1.0, 1.0))

    # Posture flags.
    out.append(1.0 if observation.get("on_ground", True) else 0.0)
    out.append(1.0 if observation.get("sprinting") else 0.0)
    out.append(1.0 if observation.get("sneaking") else 0.0)
    out.append(1.0 if observation.get("in_water") else 0.0)

    # Hotbar selection (9 slots).
    slot = int(_clamp(_number(observation.get("selected_slot")), 0, 8))
    out.extend(1.0 if index == slot else 0.0 for index in range(9))

    # Vitals.
    out.append(_clamp(_number(observation.get("health"), 20.0) / 20.0, 0.0, 1.0))
    out.append(_clamp(_number(observation.get("food"), 20.0) / 20.0, 0.0, 1.0))

    # Dimension one-hot.
    dimension = _dimension_index(observation.get("dimension"))
    out.extend(1.0 if index == dimension else 0.0 for index in range(len(DIMENSIONS)))

    # Nearby entities: density, then the closest few described in detail.
    nearby = [entry for entry in (observation.get("nearby") or []) if isinstance(entry, dict)]
    nearby.sort(key=lambda entry: _number(entry.get("dist"), 999.0))
    out.append(_clamp(len(nearby) / 8.0, 0.0, 1.0))
    hostile_count = sum(1 for entry in nearby if entity_class(entry) == 1)
    out.append(_clamp(hostile_count / 4.0, 0.0, 1.0))
    for index in range(NEARBY_SLOTS):
        entry = nearby[index] if index < len(nearby) else None
        if entry is None:
            out.extend([0.0] * (len(NEARBY_CLASSES) + 4))
            continue
        kind = entity_class(entry)
        out.extend(1.0 if slot_index == kind else 0.0 for slot_index in range(len(NEARBY_CLASSES)))
        out.append(_clamp(_number(entry.get("dist"), 16.0) / 16.0, 0.0, 1.5))
        out.append(_clamp(_number(entry.get("dx")) / 16.0, -1.5, 1.5))
        out.append(_clamp(_number(entry.get("dy")) / 16.0, -1.5, 1.5))
        out.append(_clamp(_number(entry.get("health"), 20.0) / 20.0, 0.0, 1.0))

    # Block neighbourhood, one 5-way class encoding per cell.
    if include_blocks:
        known = {
            (int(cell[0]), int(cell[1]), int(cell[2])): block_class(cell[3])
            for cell in (observation.get("blocks") or [])
            if isinstance(cell, (list, tuple)) and len(cell) >= 4
        }
        for dx, dy, dz in CELLS:
            kind = known.get((dx, dy, dz))
            if kind is None:
                kind = block_class("minecraft:stone") if dy == -1 else block_class("minecraft:air")
            out.extend(1.0 if index == kind else 0.0 for index in range(len(BLOCK_CLASSES)))

    return out


def encode_many(observations: Iterable[dict[str, Any]], include_blocks: bool = True) -> list[list[float]]:
    return [encode(observation, include_blocks) for observation in observations]


def action_of(observation: dict[str, Any]) -> str:
    """The recorded action for a sample, normalised onto :data:`ACTION_SPACE`."""
    raw = str(observation.get("activity") or observation.get("action") or "none").strip().lower()
    return raw if raw in ACTION_SPACE else "none"


def wrap_degrees(delta: float) -> float:
    """Shortest signed angle, so 350 -> 10 is +20 instead of -340."""
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def movement_deltas(current: dict[str, Any], following: dict[str, Any]) -> tuple[float, float, float, float]:
    """Numeric targets learned next to the action label.

    Returns ``(dyaw, dpitch, forward, strafe)``: how the player turned and how far the
    next sample moved along (``forward``) and across (``strafe``) their own facing.
    """
    dyaw = wrap_degrees(_number(following.get("yaw")) - _number(current.get("yaw")))
    dpitch = _number(following.get("pitch")) - _number(current.get("pitch"))
    dx = _number(following.get("x")) - _number(current.get("x"))
    dz = _number(following.get("z")) - _number(current.get("z"))
    yaw = math.radians(_number(current.get("yaw")))
    # Minecraft yaws: 0 = +Z (south), 90 = -X (west).
    forward_axis = (-math.sin(yaw), math.cos(yaw))
    strafe_axis = (math.cos(yaw), math.sin(yaw))
    forward = dx * forward_axis[0] + dz * forward_axis[1]
    strafe = dx * strafe_axis[0] + dz * strafe_axis[1]
    return dyaw, dpitch, forward, strafe


def normalise_targets(values: Sequence[float]) -> list[float]:
    """Scale the numeric targets into a friendly range for regression."""
    dyaw, dpitch, forward, strafe = (list(values) + [0.0, 0.0, 0.0, 0.0])[:4]
    return [
        _clamp(dyaw / 15.0, -4.0, 4.0),
        _clamp(dpitch / 15.0, -4.0, 4.0),
        _clamp(forward / 0.6, -4.0, 4.0),
        _clamp(strafe / 0.6, -4.0, 4.0),
    ]


def denormalise_targets(values: Sequence[float]) -> list[float]:
    """Inverse of :func:`normalise_targets` (used when the model drives the bot)."""
    dyaw, dpitch, forward, strafe = (list(values) + [0.0, 0.0, 0.0, 0.0])[:4]
    return [dyaw * 15.0, dpitch * 15.0, forward * 0.6, strafe * 0.6]


def summarise(observations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Cheap dataset summary used by the CLI, the panel and the tests."""
    actions: dict[str, int] = {}
    dimensions: dict[str, int] = {}
    episodes: set[str] = set()
    hostile = 0
    for observation in observations:
        action = action_of(observation)
        actions[action] = actions.get(action, 0) + 1
        dimension = str(observation.get("dimension") or "overworld")
        dimensions[dimension] = dimensions.get(dimension, 0) + 1
        if observation.get("episode"):
            episodes.add(str(observation["episode"]))
        hostile += sum(1 for entry in (observation.get("nearby") or []) if entity_class(entry) == 1)
    return {
        "samples": len(observations),
        "episodes": len(episodes),
        "actions": dict(sorted(actions.items(), key=lambda item: -item[1])),
        "dimensions": dimensions,
        "hostile_entities_seen": hostile,
    }
