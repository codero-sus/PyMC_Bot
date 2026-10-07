"""Reading, writing and synthesising playtime datasets.

Two file formats exist, both line delimited JSON (NDJSON):

* **episode files** -- ``<gameDir>/pymc-playtime/episodes/<run>.ndjsonl``, written by the
  Fabric mod as the player plays (one observation per sample, the action is the
  ``activity`` field).
* **dataset file** -- ``<gameDir>/pymc-playtime/dataset.jsonl``, produced by
  ``/pymc export`` in game or by :func:`export_dataset` here. Each line is one training
  example::

      {"instruction": "...", "input": {<observation>}, "output": "{\\"action\\": \\"forward\\"}",
       "episode": "run-20261007-101500"}

:func:`load_examples` accepts either format (and a directory of episode files), so the
trainer, the tests and the panel all read playtime through one function.

:func:`synthesize_playtime` fabricates a believable player session with the same schema --
used by the test suite and for headless demos, so the training pipeline can be exercised
without launching Minecraft.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pymc_bot.features import ACTION_SPACE, CELLS, movement_deltas

DEFAULT_DIRNAME = "pymc-playtime"
EPISODES_DIRNAME = "episodes"
DATASET_FILENAME = "dataset.jsonl"
META_FILENAME = "meta.json"

INSTRUCTION = (
    "You are a Minecraft player bot. Given the observation, reply with the action to take next."
)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _unwrap(line: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    """Turn one JSON line into ``(observation, action)`` or ``None`` when unusable."""
    if not isinstance(line, dict):
        return None
    if "input" in line and isinstance(line["input"], dict):
        observation = line["input"]
        output = line.get("output")
        action = ""
        if isinstance(output, dict):
            action = str(output.get("action") or "")
        elif isinstance(output, str):
            try:
                parsed = json.loads(output)
                action = str(parsed.get("action") or "") if isinstance(parsed, dict) else ""
            except json.JSONDecodeError:
                action = ""
        observation = {**observation, "activity": action or observation.get("activity")}
        episode = line.get("episode")
        if episode and not observation.get("episode"):
            observation["episode"] = episode
        return observation, (action if action in ACTION_SPACE else "none")
    if "activity" in line or "x" in line:
        action = str(line.get("activity") or "none").lower()
        return line, (action if action in ACTION_SPACE else "none")
    return None


def iter_ndjson(path: Path) -> Iterator[dict[str, Any]]:
    """Yield every JSON object in an NDJSON file, skipping blank/corrupt lines."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw or not raw.startswith("{"):
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def dataset_files(path: str | Path) -> list[Path]:
    """Resolve ``path`` (file or directory) into the list of NDJSON files to read."""
    target = Path(path).expanduser()
    if target.is_file():
        return [target]
    if not target.is_dir():
        return []
    files = sorted(target.glob("*.ndjsonl")) + sorted(target.glob("*.jsonl"))
    episodes = sorted((target / EPISODES_DIRNAME).glob("*.ndjsonl"))
    seen: list[Path] = []
    for file in files + episodes:
        if file not in seen:
            seen.append(file)
    return seen


def load_examples(path: str | Path, limit: int | None = None) -> list[tuple[dict[str, Any], str]]:
    """Load ``(observation, action)`` pairs from a dataset file, episode dir or game dir."""
    examples: list[tuple[dict[str, Any], str]] = []
    for file in dataset_files(path):
        for line in iter_ndjson(file):
            unwrapped = _unwrap(line)
            if unwrapped is None:
                continue
            examples.append(unwrapped)
            if limit is not None and len(examples) >= limit:
                return examples
    return examples


def write_dataset(
    path: str | Path,
    examples: Iterable[tuple[dict[str, Any], str]],
    *,
    instruction: str = INSTRUCTION,
) -> int:
    """Write ``(observation, action)`` pairs in the canonical dataset format."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("w", encoding="utf-8") as handle:
        for observation, action in examples:
            handle.write(
                json.dumps(
                    {
                        "instruction": instruction,
                        "input": observation,
                        "output": json.dumps({"action": action}),
                        "episode": observation.get("episode", "unknown"),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            written += 1
    return written


def export_dataset(
    source: str | Path = ".",
    destination: str | Path | None = None,
    *,
    append: bool = False,
) -> Path:
    """Mirror ``/pymc export`` outside the game (dataset dir -> dataset.jsonl)."""
    source_path = Path(source).expanduser()
    root = source_path if source_path.is_dir() else source_path.parent
    if destination is None:
        candidate = root if root.name == DEFAULT_DIRNAME else root / DEFAULT_DIRNAME
        destination = candidate / DATASET_FILENAME
    examples = load_examples(source_path)
    if append and Path(destination).exists():
        existing = load_examples(destination)
        return Path(destination) if write_dataset(destination, existing + examples) else Path(destination)
    write_dataset(destination, examples)
    return Path(destination)


# ---------------------------------------------------------------------------
# pairing samples into transitions (features -> next action)
# ---------------------------------------------------------------------------
def pair_transitions(
    examples: list[tuple[dict[str, Any], str]],
) -> list[tuple[dict[str, Any], str, list[float]]]:
    """Split ``examples`` into ``(observation, action, numeric_targets)``.

    Numeric targets come from the *next* sample in the same episode
    (:func:`pymc_bot.features.movement_deltas`), which is what lets the model reproduce
    the player's continuous motion instead of only the discrete action.
    """
    transitions: list[tuple[dict[str, Any], str, list[float]]] = []
    last_episode: str | None = None
    previous: tuple[dict[str, Any], str] | None = None
    for observation, action in examples:
        episode = str(observation.get("episode") or "")
        if previous is not None and episode == last_episode:
            targets = list(movement_deltas(previous[0], observation))
        else:
            targets = [0.0, 0.0, 0.0, 0.0]
        if previous is not None:
            transitions.append((previous[0], previous[1], targets))
        previous = (observation, action)
        last_episode = episode
    if previous is not None:
        transitions.append((previous[0], previous[1], [0.0, 0.0, 0.0, 0.0]))
    return transitions


# ---------------------------------------------------------------------------
# synthetic playtime (demos + tests; the real source is the mod)
# ---------------------------------------------------------------------------
PROFILES: dict[str, dict[str, float]] = {
    "explorer": {"explore": 0.55, "mine": 0.1, "combat": 0.05, "chat": 0.05},
    "builder": {"explore": 0.2, "mine": 0.35, "combat": 0.02, "chat": 0.03},
    "fighter": {"explore": 0.2, "mine": 0.05, "combat": 0.5, "chat": 0.02},
}


def synthesize_playtime(
    destination: str | Path,
    *,
    minutes: float = 6.0,
    sample_ticks: int = 2,
    profiles: Iterable[str] = ("explorer", "builder", "fighter"),
    seed: int = 7,
    instruction: str = INSTRUCTION,
    advanced: bool = False,
) -> dict[str, Any]:
    """Fabricate a player session and write it as a dataset file.

    Each profile is a plausible player style (exploring, mining, fighting); the
    generator walks a fake world, turns the view, jumps, sneaks, swings and switches
    hotbar slots, and emits samples with exactly the mod's schema. Handy for demos and
    for testing the trainer without a Minecraft client installed.

    ``advanced=True`` also writes the entity and item half of the schema (``entities``,
    ``items``, ``held_item``, ``armor``, ``ground_items``), which is what the mod records
    with ``/pymc advanced on`` and what an advanced model trains on.
    """
    rng = random.Random(seed)
    samples_per_minute = 60.0 / (sample_ticks * 0.05)
    total_samples = max(50, int(minutes * samples_per_minute))
    target = Path(destination).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    names = [name for name in profiles if name in PROFILES] or ["explorer"]
    written = 0
    action_counts: dict[str, int] = {}
    with target.open("w", encoding="utf-8") as handle:
        for profile in names:
            weights = PROFILES[profile]
            episode = f"sim-{profile}-{seed}"
            x, y, z = rng.uniform(-200, 200), 64.0, rng.uniform(-200, 200)
            yaw, pitch = rng.uniform(-180, 180), 0.0
            slot, health, food = 0, 20.0, 20.0
            count = max(20, total_samples // len(names))
            spawn_ticks = rng.randint(10, 30)  # a fresh spawn with nothing in hand
            mood = "explore"
            mood_left = rng.randint(20, 60)
            for tick in range(count):
                moved = 0.0
                action = "none"
                if mood_left <= 0:
                    mood = _pick_mood(rng, weights)
                    mood_left = rng.randint(20, 80)
                mood_left -= 1

                if mood == "explore":
                    action, moved, yaw, x, z = _step_walk(rng, yaw, x, z, sprint=rng.random() < 0.25)
                    if rng.random() < 0.06:
                        action = "look"
                        yaw += rng.uniform(-35, 35)
                    if rng.random() < 0.03:
                        action = "jump"
                        y += 0.4
                elif mood == "mine":
                    swing = rng.random()
                    if swing < 0.55:
                        action = "attack"
                    elif swing < 0.7:
                        action = "sneak"
                    elif swing < 0.8:
                        action = "hold"
                        slot = (slot + 1) % 9
                    else:
                        action = "look"
                        yaw += rng.uniform(-12, 12)
                    pitch = rng.uniform(-30, 25)
                    moved = rng.uniform(0.0, 0.05)
                    x += math.sin(math.radians(yaw)) * moved
                    z -= math.cos(math.radians(yaw)) * moved
                elif mood == "combat":
                    action = "attack" if rng.random() < 0.6 else "use"
                    yaw += rng.uniform(-25, 25)
                    if rng.random() < 0.3:
                        action = "jump"
                else:  # chat break
                    action = "look"
                    yaw += rng.uniform(-8, 8)
                    if rng.random() < 0.2:
                        action = "none"

                if action not in ACTION_SPACE:  # pragma: no cover - defensive
                    action = "none"
                # Health and hunger follow the mood instead of random-walking: a fight
                # costs health, everything else regenerates; eating restores hunger. A
                # session that decays to permanent near-death (the old behaviour) taught
                # the model that "nearly dead and starving" is the normal state.
                if mood == "combat":
                    health -= rng.uniform(0.0, 0.4)
                else:
                    health += rng.uniform(0.1, 0.5)  # regeneration between fights
                health = max(6.0, min(20.0, health))
                if action in ("use", "hold"):
                    food += rng.uniform(2.0, 6.0)  # ate something
                elif mood == "explore" and moved > 0.2:
                    food -= rng.uniform(0.0, 0.12)
                else:
                    food -= rng.uniform(0.0, 0.04)
                food = max(6.0, min(20.0, food))
                pitch = max(-60.0, min(60.0, pitch + rng.uniform(-3, 3)))

                observation = {
                    "tick": tick * sample_ticks,
                    "episode": episode,
                    "activity": action,
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "z": round(z, 3),
                    "vx": round(math.sin(math.radians(yaw)) * moved, 3),
                    "vy": 0.0,
                    "vz": round(-math.cos(math.radians(yaw)) * moved, 3),
                    "moved": round(moved, 3),
                    "yaw": round(yaw % 360.0, 2),
                    "pitch": round(pitch, 2),
                    "on_ground": action != "jump",
                    "sprinting": moved > 0.2,
                    "sneaking": action == "sneak",
                    "selected_slot": slot,
                    "health": round(health, 1),
                    "food": int(food),
                    "dimension": "overworld",
                    "blocks": _fake_blocks(rng, profile),
                    "nearby": _fake_nearby(rng, profile, x, y, z, yaw, mood),
                }
                if advanced:
                    entities = _fake_entities(rng, profile, mood, x, y, z, yaw)
                    if tick < spawn_ticks:
                        # Every session starts empty handed and unarmoured: without this
                        # state in the data a model never learns what to do when the bot
                        # spawns on a server with nothing in its inventory.
                        items, held, armor = _fresh_spawn_inventory(rng, slot)
                    else:
                        items, held, armor = _fake_inventory(rng, profile, mood, slot)
                        if mood == "chat" and rng.random() < 0.5:
                            held = ""  # weapons away while typing
                    observation["entities"] = entities
                    observation["items"] = items
                    observation["held_item"] = held
                    observation["armor"] = armor
                    observation["ground_items"] = _fake_ground_items(rng, profile, mood)
                action_counts[action] = action_counts.get(action, 0) + 1
                handle.write(
                    json.dumps(
                        {
                            "instruction": instruction,
                            "input": observation,
                            "output": json.dumps({"action": action}),
                            "episode": episode,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                written += 1

    return {
        "dataset": str(target),
        "samples": written,
        "episodes": len(names),
        "minutes": round(written * sample_ticks * 0.05 / 60.0, 2),
        "actions": dict(sorted(action_counts.items(), key=lambda item: -item[1])),
    }


def _pick_mood(rng: random.Random, weights: dict[str, float]) -> str:
    moods = ["explore", "mine", "combat", "chat"]
    roll = rng.random()
    cumulative = 0.0
    for mood in moods:
        cumulative += weights.get(mood, 0.0)
        if roll <= cumulative:
            return mood
    return "explore"


def _step_walk(
    rng: random.Random, yaw: float, x: float, z: float, *, sprint: bool
) -> tuple[str, float, float, float, float]:
    """Move one sample forward in the facing direction and pick the matching key."""
    speed = rng.uniform(0.15, 0.22) if sprint else rng.uniform(0.08, 0.16)
    drift = rng.uniform(-0.02, 0.02)
    yaw += rng.uniform(-4, 4)
    radians = math.radians(yaw)
    x += -math.sin(radians) * speed + math.cos(radians) * drift
    z += math.cos(radians) * speed + math.sin(radians) * drift
    action = "forward"
    if rng.random() < 0.08:
        action = rng.choice(["left", "right", "back"])
    return action, speed, yaw, x, z


def _fake_blocks(rng: random.Random, profile: str) -> list[list[Any]]:
    cells: list[list[Any]] = []
    for dx, dy, dz in CELLS:
        if dy == -1:
            name = "minecraft:stone"
            if profile == "builder" and rng.random() < 0.25:
                name = "minecraft:oak_log"
            elif rng.random() < 0.1:
                name = "minecraft:coal_ore"
        elif dy == 0 and abs(dx) + abs(dz) == 1 and rng.random() < 0.15:
            name = "minecraft:oak_log" if profile == "builder" else "minecraft:air"
        else:
            name = "minecraft:air"
        cells.append([dx, dy, dz, name])
    return cells


def _fake_nearby(
    rng: random.Random,
    profile: str,
    x: float,
    y: float,
    z: float,
    yaw: float,
    mood: str = "explore",
) -> list[dict[str, Any]]:
    """The neighbours a player in this mood would actually see.

    Who is around is what a player reacts to, so the two have to agree: mobs within
    reach while fighting, somebody standing next to you while chatting, animals or
    nothing at all while exploring or mining. Without this the model learns a
    correlation that does not exist ("anything nearby -> swing") and fights the other
    players on the server.
    """
    entries: list[dict[str, Any]] = []
    if mood == "combat":
        count = rng.choice([1, 1, 2, 2, 3])
        kinds = ["minecraft:zombie", "minecraft:skeleton", "minecraft:creeper", "minecraft:spider"]
        low, high = 1.5, 5.0
    elif mood == "chat":
        count = rng.choice([1, 1, 2])
        kinds = ["minecraft:player"]
        low, high = 2.5, 9.0
    elif mood == "mine":
        count = rng.choice([0, 0, 1])
        kinds = [
            {"explorer": "minecraft:cow", "builder": "minecraft:villager", "fighter": "minecraft:skeleton"}[
                profile
            ]
        ]
        low, high = 9.0, 16.0
    else:  # exploring: mostly alone, sometimes animals or another player in the distance
        count = rng.choice([0, 0, 0, 1, 1, 2])
        kinds = {
            "explorer": ["minecraft:cow", "minecraft:sheep", "minecraft:player"],
            "builder": ["minecraft:villager", "minecraft:player"],
            "fighter": ["minecraft:zombie", "minecraft:creeper"],
        }[profile]
        low, high = 6.0, 18.0
    for _ in range(count):
        angle = math.radians(yaw + rng.uniform(-90, 90))
        distance = rng.uniform(low, high)
        kind = rng.choice(kinds)
        entries.append(
            {
                "type": kind,
                "dx": round(-math.sin(angle) * distance, 2),
                "dy": round(rng.uniform(-1.0, 1.0), 2),
                "dz": round(math.cos(angle) * distance, 2),
                "dist": round(distance, 2),
                "hostile": kind.split(":")[-1]
                in ("zombie", "skeleton", "creeper", "spider", "enderman"),
                "health": 20.0 if kind == "minecraft:player" else round(rng.uniform(4, 20), 1),
            }
        )
    entries.sort(key=lambda entry: entry["dist"])
    return entries


HOTBAR_BY_MOOD: dict[str, tuple[str, ...]] = {
    "explore": ("minecraft:stone_sword", "minecraft:oak_planks", "minecraft:torch", "minecraft:bread"),
    "mine": ("minecraft:iron_pickaxe", "minecraft:stone_shovel", "minecraft:torch", "minecraft:cobblestone"),
    "combat": ("minecraft:iron_sword", "minecraft:shield", "minecraft:cooked_beef", "minecraft:bow"),
    "chat": ("minecraft:bread", "minecraft:oak_planks", "minecraft:torch", "minecraft:shield"),
}

ARMOR_BY_PROFILE: dict[str, tuple[str, ...]] = {
    "explorer": ("minecraft:leather_helmet", "minecraft:leather_boots"),
    "builder": ("minecraft:iron_helmet", "minecraft:iron_chestplate", "minecraft:iron_leggings"),
    "fighter": ("minecraft:diamond_helmet", "minecraft:diamond_chestplate", "minecraft:diamond_boots"),
}

GROUND_LOOT_BY_MOOD: dict[str, tuple[str, ...]] = {
    "explore": ("minecraft:wheat_seeds", "minecraft:bone"),
    "mine": ("minecraft:cobblestone", "minecraft:coal", "minecraft:raw_iron"),
    "combat": ("minecraft:rotten_flesh", "minecraft:bone", "minecraft:arrow"),
    "chat": ("minecraft:apple",),
}


def _fake_inventory(
    rng: random.Random, profile: str, mood: str, selected: int
) -> tuple[list[list[Any]], str, list[str]]:
    """A hotbar and armour that match what the player is doing (items are how Minecraft players act)."""
    hotbar = list(HOTBAR_BY_MOOD[mood])
    while len(hotbar) < 9:
        hotbar.append(hotbar[rng.randrange(len(hotbar))] if rng.random() < 0.6 else "minecraft:air")
    rng.shuffle(hotbar)
    items: list[list[Any]] = []
    for index, name in enumerate(hotbar):
        if name == "minecraft:air":
            continue
        count = 1 if name.endswith(("sword", "pickaxe", "shovel", "shield", "bow")) else rng.randint(1, 8)
        items.append([index, name, count])
    # A few things kept in the main inventory, the way a session accumulates junk.
    for index in range(9, 9 + rng.randint(1, 4)):
        name = rng.choice(
            ("minecraft:cobblestone", "minecraft:dirt", "minecraft:oak_log", "minecraft:coal", "minecraft:stick")
        )
        items.append([index, name, rng.randint(1, 12)])
    held = hotbar[selected % len(hotbar)]
    if held == "minecraft:air":
        held = next((entry[1] for entry in items), "minecraft:air")
    armor = list(ARMOR_BY_PROFILE.get(profile, ()))
    return items, held, armor


def _fresh_spawn_inventory(rng: random.Random, selected: int) -> tuple[list[list[Any]], str, list[str]]:
    """What a player carries right after joining: no armour, an empty hand, a bit of junk."""
    items: list[list[Any]] = []
    for index in range(rng.randint(0, 2)):
        items.append([index, rng.choice(("minecraft:dirt", "minecraft:cobblestone")), rng.randint(1, 12)])
    return items, "", []


def _fake_entities(
    rng: random.Random, profile: str, mood: str, x: float, y: float, z: float, yaw: float
) -> list[dict[str, Any]]:
    """Everything the player can see: the neighbours that matter plus distant scenery."""
    entities: list[dict[str, Any]] = []
    for entry in _fake_nearby(rng, profile, x, y, z, yaw, mood):
        entity = dict(entry)
        entity["player"] = _short_type(entity["type"]) == "player" or entity["type"].endswith("player")
        if entity["player"]:
            entity["health"] = 20.0
            entity["held_item"] = rng.choice(("minecraft:iron_sword", "minecraft:bow", "minecraft:shield"))
        else:
            entity["held_item"] = ""
        entity["on_ground"] = rng.random() < 0.9
        entity["yaw"] = round(rng.uniform(-180, 180), 1)
        entities.append(entity)
    # Distant animals/players wandering about - context the advanced encoder can learn from.
    for _ in range(rng.choice([0, 1, 1, 2])):
        angle = math.radians(yaw + rng.uniform(-180, 180))
        distance = rng.uniform(14.0, 26.0)
        kind = rng.choice(("minecraft:cow", "minecraft:sheep", "minecraft:zombie", "minecraft:player"))
        entities.append(
            {
                "type": kind,
                "dx": round(-math.sin(angle) * distance, 2),
                "dy": 0.0,
                "dz": round(math.cos(angle) * distance, 2),
                "dist": round(distance, 2),
                "hostile": kind.endswith("zombie"),
                "player": kind.endswith("player"),
                "health": 20.0,
                "held_item": "",
                "on_ground": True,
                "yaw": round(rng.uniform(-180, 180), 1),
            }
        )
    entities.sort(key=lambda entry: entry["dist"])
    return entities


def _fake_ground_items(rng: random.Random, profile: str, mood: str) -> list[dict[str, Any]]:
    """Drops lying around after mining/fighting - what a player walks over to pick up."""
    stacks: list[dict[str, Any]] = []
    for _ in range(rng.choice([0, 0, 1, 1, 2])):
        stacks.append(
            {
                "item": rng.choice(GROUND_LOOT_BY_MOOD[mood]),
                "count": rng.randint(1, 4),
                "dist": round(rng.uniform(1.0, 6.0), 2),
            }
        )
    stacks.sort(key=lambda entry: entry["dist"])
    return stacks


def _short_type(name: str) -> str:
    return str(name or "").split(":")[-1]


def dataset_summary(path: str | Path) -> dict[str, Any]:
    """Small report used by ``pymc_bot train --inspect`` and the panel."""
    from pymc_bot.features import learn_vocabulary, summarise, vocabulary_coverage

    examples = load_examples(path)
    observations = [observation for observation, _ in examples]
    summary = summarise(observations)
    summary["dataset"] = str(path)
    # What an advanced model could learn from this dataset.
    vocab = learn_vocabulary(observations)
    summary["vocabulary"] = vocab.describe()
    summary.update(
        {key: value for key, value in vocabulary_coverage(observations, vocab).items() if key != "samples_with_entities_or_items"}
    )
    summary["entities_and_items"] = bool(
        any(observation.get("entities") or observation.get("items") for observation in observations)
    )
    return summary
