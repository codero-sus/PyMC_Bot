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
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# advanced training: entity and item vocabularies
# ---------------------------------------------------------------------------
#: Layout ids. v1 is the fixed, hand written encoding; v2 adds the entity and item
#: vocabularies that "advanced training" learns from the recorded playtime.
FEATURE_VERSION = 1
ADVANCED_FEATURE_VERSION = 2

DEFAULT_ENTITY_SLOTS = 24
DEFAULT_ITEM_SLOTS = 32
ENTITY_RANGE = 24.0
ITEM_COUNT_SCALE = 16.0

#: Entity ids that always get a slot when a vocabulary is learned (they exist in every
#: world and the ones a player reacts to), so a rare-but-important mob is never dropped
#: just because the recording was short.
SEED_ENTITIES: tuple[str, ...] = (
    "player",
    "zombie",
    "skeleton",
    "creeper",
    "spider",
    "enderman",
    "witch",
    "slime",
    "drowned",
    "husk",
    "phantom",
    "blaze",
    "ghast",
    "piglin",
    "hoglin",
    "cow",
    "sheep",
    "pig",
    "chicken",
    "villager",
    "wolf",
    "horse",
    "rabbit",
    "iron_golem",
)

#: Item *suffixes* that always get a slot: any sword, any pickaxe, any food.
SEED_ITEM_SUFFIXES: tuple[str, ...] = (
    "sword",
    "pickaxe",
    "axe",
    "shovel",
    "hoe",
    "bow",
    "shield",
    "apple",
    "bread",
    "beef",
    "porkchop",
    "chicken",
    "mutton",
    "carrot",
    "potato",
    "torch",
    "oak_log",
    "oak_planks",
    "cobblestone",
    "dirt",
    "stone",
    "coal",
    "iron_ingot",
    "diamond",
)

WEAPON_HINTS = ("sword", "axe", "bow", "crossbow", "trident", "mace", "arrow", "tnt", "firework")
TOOL_HINTS = ("pickaxe", "shovel", "hoe", "shears", "fishing_rod", "flint_and_steel", "bucket")
FOOD_HINTS = (
    "apple",
    "bread",
    "beef",
    "porkchop",
    "mutton",
    "chicken",
    "rabbit",
    "cod",
    "salmon",
    "potato",
    "carrot",
    "beetroot",
    "melon",
    "cookie",
    "cake",
    "stew",
    "soup",
    "berries",
    "kelp",
    "honey",
    "milk",
)
ARMOR_HINTS = ("helmet", "chestplate", "leggings", "boots", "shield", "elytra")

#: The global part of the advanced block (see :func:`advanced_features`).
ADVANCED_GLOBAL_DIM = 13


@dataclass(frozen=True)
class Vocabulary:
    """The entity and item words an advanced model learned from the playtime.

    The trainer learns which ids actually occur in the recording (plus the seeds above)
    and stores them in the checkpoint, so the live bot encodes its world into exactly the
    same slots. An empty vocabulary means "use the fixed :func:`encode` layout".
    """

    entities: tuple[str, ...] = ()
    items: tuple[str, ...] = ()

    @property
    def slots(self) -> tuple[int, int]:
        return len(self.entities), len(self.items)

    def entity_index(self, name: Any) -> int | None:
        short = _short_name(name)
        try:
            return self.entities.index(short)
        except ValueError:
            return None

    def item_index(self, name: Any) -> int | None:
        short = _short_name(name)
        try:
            return self.items.index(short)
        except ValueError:
            return None

    def to_dict(self) -> dict[str, Any]:
        return {"entities": list(self.entities), "items": list(self.items)}

    @classmethod
    def from_dict(cls, payload: Any) -> Vocabulary:
        if isinstance(payload, Vocabulary):
            return payload
        if not isinstance(payload, dict):
            return cls()
        return cls(
            entities=tuple(str(name) for name in payload.get("entities") or ()),
            items=tuple(str(name) for name in payload.get("items") or ()),
        )

    def describe(self) -> dict[str, Any]:
        return {"entities": len(self.entities), "items": len(self.items)}


def _top_names(counts: dict[str, float], seeds: Sequence[str], slots: int, *, seed_suffix: bool) -> tuple[str, ...]:
    """Pick ``slots`` ids: the seeds that occur first, then whatever is most frequent."""
    chosen: list[str] = []
    rest = sorted(counts, key=lambda name: (-counts[name], name))
    if seed_suffix:
        for suffix in seeds:
            for name in rest:
                if name.endswith(suffix) and name not in chosen:
                    chosen.append(name)
                    break
    else:
        for seed in seeds:
            if seed in counts:
                chosen.append(seed)
    if slots > 0:
        for name in rest:
            if len(chosen) >= slots:
                break
            if name not in chosen:
                chosen.append(name)
        return tuple(chosen[:slots])
    return tuple(chosen)


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


def observation_entities(observation: dict[str, Any]) -> list[dict[str, Any]]:
    """Every entity a sample knows about, from the mod or from the live bridge.

    Advanced recordings write the full list into ``entities``; a basic recording (and
    the older bridge) only has the closest few in ``nearby``; a live snapshot has the
    players in ``players``. The first key that carries data wins, so an entity is never
    counted twice.
    """
    observation = observation or {}
    for key in ("entities", "nearby", "players"):
        entries = [
            entry
            for entry in (observation.get(key) or [])
            if isinstance(entry, dict) and (entry.get("type") or entry.get("name") or entry.get("kind"))
        ]
        if entries:
            return entries
    return []


def _entity_delta(observation: dict[str, Any], entry: dict[str, Any]) -> tuple[float, float, float, float]:
    """``(dx, dy, dz, dist)`` of an entity relative to the observer."""
    dx = _number(entry.get("dx"))
    dy = _number(entry.get("dy"))
    dz = _number(entry.get("dz"))
    if dx == 0.0 and dy == 0.0 and dz == 0.0 and observation.get("x") is not None:
        dx = _number(entry.get("x")) - _number(observation.get("x"))
        dy = _number(entry.get("y")) - _number(observation.get("y"))
        dz = _number(entry.get("z")) - _number(observation.get("z"))
    distance = _number(entry.get("dist"), math.sqrt(dx * dx + dy * dy + dz * dz))
    return dx, dy, dz, distance


def _entity_name(entry: dict[str, Any]) -> str:
    if entry.get("player") or _short_name(entry.get("type")) in ("player", "remoteplayer"):
        return "player"
    return _short_name(entry.get("type") or entry.get("name") or entry.get("kind") or "unknown")


def observation_items(observation: dict[str, Any]) -> dict[str, int]:
    """Item id -> total count in the player's inventory (and hands/armour)."""
    observation = observation or {}
    counts: dict[str, int] = {}
    for entry in observation.get("items") or []:
        name: Any = None
        count = 1
        if isinstance(entry, dict):
            name, count = entry.get("name") or entry.get("item"), entry.get("count") or 1
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            # [slot, "minecraft:oak_log", 3] from the mod, or ["oak_log", 3] from a bridge
            if isinstance(entry[0], str):
                name = entry[0]
                count = entry[2] if len(entry) > 2 else entry[1]
            else:
                name = entry[1]
                count = entry[2] if len(entry) > 2 else 1
        if isinstance(name, str) and name:
            short = _short_name(name)
            counts[short] = counts.get(short, 0) + int(_number(count, 1.0))
    for entry in observation.get("inventory") or []:
        if isinstance(entry, dict) and entry.get("name"):
            short = _short_name(entry["name"])
            counts[short] = counts.get(short, 0) + int(_number(entry.get("count"), 1.0))
    for key in ("held_item", "offhand"):
        name = observation.get(key)
        if isinstance(name, str) and name:
            short = _short_name(name)
            counts.setdefault(short, 1)
    for name in observation.get("armor") or []:
        if isinstance(name, str) and name:
            short = _short_name(name)
            counts.setdefault(short, 1)
    return counts


def observation_ground_items(observation: dict[str, Any], vocab: Vocabulary | None = None) -> list[dict[str, Any]]:
    """Dropped item stacks around the player (``ground_items`` from the mod)."""
    observation = observation or {}
    stacks = [entry for entry in (observation.get("ground_items") or []) if isinstance(entry, dict)]
    if stacks or not observation.get("entities"):
        return stacks
    # A recording may keep drops inside ``entities`` (they are entities in Minecraft).
    return [
        entry
        for entry in observation["entities"]
        if isinstance(entry, dict) and _short_name(entry.get("type")) in ("item", "item_stack")
    ]


def _item_flags(item: str) -> tuple[bool, bool, bool, bool]:
    """``(weapon, tool, food, armour)`` hints for one item id."""
    name = _short_name(item)
    weapon = any(hint in name for hint in WEAPON_HINTS)
    tool = any(hint in name for hint in TOOL_HINTS)
    food = any(hint in name for hint in FOOD_HINTS)
    armor = any(hint in name for hint in ARMOR_HINTS)
    return weapon, tool, food, armor


def learn_vocabulary(
    observations: Iterable[dict[str, Any]],
    *,
    entity_slots: int = DEFAULT_ENTITY_SLOTS,
    item_slots: int = DEFAULT_ITEM_SLOTS,
) -> Vocabulary:
    """Learn which entities and items the recorded playtime actually contains.

    Ids are ranked by how often they show up (the seeds in :data:`SEED_ENTITIES` and
    :data:`SEED_ITEM_SUFFIXES` are kept even when they are rare), and the result is stored
    in the checkpoint so the live bot encodes its world into exactly the same slots.
    """
    entity_counts: dict[str, float] = {}
    item_counts: dict[str, float] = {}
    for observation in observations or []:
        for entry in observation_entities(observation):
            name = _entity_name(entry)
            entity_counts[name] = entity_counts.get(name, 0.0) + 1.0
        for name, count in observation_items(observation).items():
            item_counts[name] = item_counts.get(name, 0.0) + float(count)
    return Vocabulary(
        entities=_top_names(entity_counts, SEED_ENTITIES, entity_slots, seed_suffix=False),
        items=_top_names(item_counts, SEED_ITEM_SUFFIXES, item_slots, seed_suffix=True),
    )


def vocabulary_coverage(observations: Sequence[dict[str, Any]], vocab: Vocabulary) -> dict[str, Any]:
    """How much of a dataset the learned vocabulary can actually describe."""
    seen_entities: dict[str, int] = {}
    seen_items: dict[str, int] = {}
    described = 0
    for observation in observations:
        entities = observation_entities(observation)
        items = observation_items(observation)
        for entry in entities:
            name = _entity_name(entry)
            seen_entities[name] = seen_entities.get(name, 0) + 1
        for name, count in items.items():
            seen_items[name] = seen_items.get(name, 0) + count
        if entities or items:
            described += 1
    known = sum(count for name, count in seen_entities.items() if vocab.entity_index(name) is not None)
    total = sum(seen_entities.values()) or 1
    known_items = sum(count for name, count in seen_items.items() if vocab.item_index(name) is not None)
    total_items = sum(seen_items.values()) or 1
    return {
        "entity_types": dict(sorted(seen_entities.items(), key=lambda item: (-item[1], item[0]))[:20]),
        "item_types": dict(sorted(seen_items.items(), key=lambda item: (-item[1], item[0]))[:20]),
        "entity_coverage": round(known / total, 4),
        "item_coverage": round(known_items / total_items, 4),
        "samples_with_entities_or_items": described,
    }


def layout_version(vocab: Vocabulary | None = None) -> int:
    """The feature layout id a model with this vocabulary was trained with."""
    if vocab is None or (not vocab.entities and not vocab.items):
        return FEATURE_VERSION
    return ADVANCED_FEATURE_VERSION


def _bearing_sin_cos(dx: float, dz: float, yaw_degrees: float) -> tuple[float, float]:
    """Where an entity sits relative to the way the player is looking."""
    if dx == 0.0 and dz == 0.0:
        return 0.0, 0.0
    bearing = math.degrees(math.atan2(-dx, dz))
    relative = math.radians(wrap_degrees(bearing - yaw_degrees))
    return math.sin(relative), math.cos(relative)


def advanced_features(observation: dict[str, Any], vocab: Vocabulary) -> list[float]:
    """Entity and item features for the ids the model learned from the playtime.

    Per entity word: is it here, how close is it, where is it relative to the view.
    Per item word: is it carried, how many, is it in the player's hand.
    Then a handful of globals (weapons, food, armour, drops, closest threat/player).
    """
    observation = observation or {}
    yaw = _number(observation.get("yaw"))
    out: list[float] = []

    entities = observation_entities(observation)
    per_entity: dict[str, list[tuple[float, tuple[float, float]]]] = {}
    for entry in entities:
        name = _entity_name(entry)
        dx, _dy, dz, distance = _entity_delta(observation, entry)
        per_entity.setdefault(name, []).append((distance, _bearing_sin_cos(dx, dz, yaw)))

    for name in vocab.entities:
        found = per_entity.get(name) or []
        if not found:
            out.extend([0.0, 0.0, 0.0, 0.0])
            continue
        nearest_distance, (sin_bearing, cos_bearing) = min(found, key=lambda item: item[0])
        out.append(1.0)
        out.append(_clamp(1.0 - nearest_distance / ENTITY_RANGE, 0.0, 1.0))
        out.append(sin_bearing)
        out.append(cos_bearing)

    carried = observation_items(observation)
    held = _short_name(observation.get("held_item") or "")
    for name in vocab.items:
        count = carried.get(name, 0)
        out.append(1.0 if count else 0.0)
        out.append(_clamp(count / ITEM_COUNT_SCALE, 0.0, 1.0))
        out.append(1.0 if held and held == name else 0.0)

    # Globals: what the player is equipped to do, and what is lying on the floor.
    weapons = tools = foods = blockers = armor_pieces = 0
    for name, count in carried.items():
        weapon, tool, food, armor = _item_flags(name)
        weapons += count if weapon else 0
        tools += count if tool else 0
        foods += count if food else 0
        armor_pieces += 1 if armor else 0
        blockers += count if not (weapon or tool or food or armor) else 0
    ground = observation_ground_items(observation)
    nearest_ground = min((_number(entry.get("dist"), ENTITY_RANGE) for entry in ground), default=None)
    hostiles = [
        _number(entry.get("dist"), ENTITY_RANGE)
        for entry in entities
        if entity_class(entry) == 1
    ]
    players = [_number(entry.get("dist"), ENTITY_RANGE) for entry in entities if entity_class(entry) == 0]
    out.extend(
        [
            _clamp(sum(carried.values()) / 36.0, 0.0, 2.0),
            _clamp(len(carried) / 16.0, 0.0, 2.0),
            _clamp(weapons / 4.0, 0.0, 1.0),
            _clamp(tools / 4.0, 0.0, 1.0),
            _clamp(foods / 8.0, 0.0, 1.0),
            _clamp(blockers / 32.0, 0.0, 1.0),
            _clamp(armor_pieces / 4.0, 0.0, 1.0),
            _clamp(len(ground) / 4.0, 0.0, 1.0),
            _clamp(1.0 - nearest_ground / ENTITY_RANGE, 0.0, 1.0) if nearest_ground is not None else 0.0,
            _clamp(1.0 - min(hostiles) / ENTITY_RANGE, 0.0, 1.0) if hostiles else 0.0,
            _clamp(len(hostiles) / 4.0, 0.0, 1.0),
            _clamp(1.0 - min(players) / ENTITY_RANGE, 0.0, 1.0) if players else 0.0,
            _clamp(len(players) / 4.0, 0.0, 1.0),
        ]
    )
    return out


def basic_feature_dim(include_blocks: bool = True) -> int:
    """Length of the fixed part of the vector (layout v1)."""
    base = 3 + 3 + 1 + 4 + 4 + 9 + 2 + len(DIMENSIONS) + 2 + NEARBY_SLOTS * (len(NEARBY_CLASSES) + 4)
    blocks = len(CELLS) * len(BLOCK_CLASSES) if include_blocks else 0
    return base + blocks


def advanced_feature_dim(vocab: Vocabulary | None) -> int:
    """Extra floats the entity/item vocabularies add (0 without a vocabulary)."""
    if vocab is None or (not vocab.entities and not vocab.items):
        return 0
    return len(vocab.entities) * 4 + len(vocab.items) * 3 + ADVANCED_GLOBAL_DIM


def feature_dim(include_blocks: bool = True, vocab: Vocabulary | None = None) -> int:
    """Length of the vector :func:`encode` returns.

    ``vocab`` selects the advanced layout, so the trainer and the live bot can never
    disagree about what a column means.
    """
    return basic_feature_dim(include_blocks) + advanced_feature_dim(vocab)


def encode(
    observation: dict[str, Any],
    include_blocks: bool = True,
    vocab: Vocabulary | None = None,
) -> list[float]:
    """Encode one observation (mod sample or live bot snapshot) as a feature vector.

    With a ``vocab`` learned by :func:`learn_vocabulary` the entity and item words are
    appended after the fixed block (layout v2, "advanced training").
    """
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

    if vocab is not None and (vocab.entities or vocab.items):
        out.extend(advanced_features(observation, vocab))

    return out


def encode_many(
    observations: Iterable[dict[str, Any]],
    include_blocks: bool = True,
    vocab: Vocabulary | None = None,
) -> list[list[float]]:
    return [encode(observation, include_blocks, vocab) for observation in observations]


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
