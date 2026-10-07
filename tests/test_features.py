"""The observation encoder is the contract between the mod and the trained model.

If these break, a checkpoint can no longer be run, so the layout is pinned here.
"""

from __future__ import annotations

import math

from pymc_bot.features import (
    ACTION_SPACE,
    BLOCK_CLASSES,
    CELLS,
    DIMENSIONS,
    FEATURE_VERSION,
    NEARBY_CLASSES,
    NEARBY_SLOTS,
    action_of,
    block_class,
    default_blocks,
    denormalise_targets,
    encode,
    entity_class,
    feature_dim,
    movement_deltas,
    normalise_targets,
    summarise,
    wrap_degrees,
)

OBSERVATION = {
    "tick": 20,
    "episode": "run-1",
    "activity": "forward",
    "x": 12.5,
    "y": 64.0,
    "z": -3.25,
    "vx": 0.1,
    "vy": 0.0,
    "vz": -0.2,
    "moved": 0.21,
    "yaw": 91.0,
    "pitch": 4.5,
    "on_ground": True,
    "sprinting": False,
    "sneaking": False,
    "selected_slot": 3,
    "health": 17.0,
    "food": 15,
    "dimension": "overworld",
    "blocks": [[dx, dy, dz, "minecraft:stone" if dy == -1 else "minecraft:air"] for dx, dy, dz in CELLS],
    "nearby": [
        {"type": "minecraft:zombie", "dx": 3.1, "dy": 0.0, "dz": -1.4, "dist": 3.4, "hostile": True, "health": 20.0},
        {"type": "minecraft:player", "dx": -2.0, "dy": 0.0, "dz": 1.0, "dist": 2.2, "hostile": False, "health": 20.0},
    ],
}


def test_feature_layout_is_pinned():
    # Bumping the layout must bump FEATURE_VERSION (checkpoints carry it and refuse to load).
    assert FEATURE_VERSION == 1
    assert len(CELLS) == 27
    assert len(ACTION_SPACE) == 12
    assert feature_dim(include_blocks=True) == 215
    assert feature_dim(include_blocks=False) == 215 - 27 * len(BLOCK_CLASSES)


def test_encode_is_fixed_length_and_numeric():
    vector = encode(OBSERVATION)
    assert len(vector) == feature_dim(True)
    assert all(isinstance(value, float) for value in vector)
    assert all(math.isfinite(value) for value in vector)
    assert len(encode(OBSERVATION, include_blocks=False)) == feature_dim(False)


def test_encode_is_deterministic_and_position_sensitive():
    assert encode(OBSERVATION) == encode(dict(OBSERVATION))
    moved = encode({**OBSERVATION, "x": 40.0})
    assert moved != encode(OBSERVATION)


def test_encode_survives_missing_inputs():
    """A live bot snapshot has far fewer fields than a recorded sample."""
    vector = encode({"health": 20.0})
    assert len(vector) == feature_dim(True)
    assert all(math.isfinite(value) for value in vector)
    assert encode({}) == encode({})


def test_view_is_encoded_without_wraparound_discontinuity():
    near_zero = encode({**OBSERVATION, "yaw": 1.0})
    near_360 = encode({**OBSERVATION, "yaw": 359.0})
    far = encode({**OBSERVATION, "yaw": 180.0})
    zero_distance = sum((a - b) ** 2 for a, b in zip(near_zero, near_360, strict=True)) ** 0.5
    far_distance = sum((a - b) ** 2 for a, b in zip(near_zero, far, strict=True)) ** 0.5
    assert zero_distance < far_distance  # sin/cos, not raw degrees


def test_selected_slot_is_one_hot():
    slot = 5
    vector = encode({**OBSERVATION, "selected_slot": slot})
    start = 3 + 3 + 1 + 4 + 4  # pos, vel, moved, view, flags
    one_hot = vector[start : start + 9]
    assert one_hot.index(1.0) == slot
    assert sum(one_hot) == 1.0


def test_dimension_one_hot_covers_every_known_world():
    for index, dimension in enumerate(DIMENSIONS):
        vector = encode({**OBSERVATION, "dimension": dimension})
        offset = 3 + 3 + 1 + 4 + 4 + 9 + 2
        block = vector[offset : offset + len(DIMENSIONS)]
        assert block[index] == 1.0
        assert sum(block) == 1.0
    # Unknown (modded) dimensions land in the "other" bucket instead of blowing up.
    unknown = encode({**OBSERVATION, "dimension": "twilightforest:twilight_forest"})
    offset = 3 + 3 + 1 + 4 + 4 + 9 + 2
    assert unknown[offset + len(DIMENSIONS) - 1] == 1.0


def test_nearby_entities_are_sorted_and_capped():
    many = [{"type": "minecraft:cow", "dist": float(distance), "dx": 0.0, "dy": 0.0, "dz": 0.0} for distance in range(12, 0, -1)]
    vector = encode({**OBSERVATION, "nearby": many})
    slots_start = 3 + 3 + 1 + 4 + 4 + 9 + 2 + len(DIMENSIONS) + 2
    stride = len(NEARBY_CLASSES) + 4
    distances = [vector[slots_start + index * stride + len(NEARBY_CLASSES)] * 16 for index in range(NEARBY_SLOTS)]
    assert distances == sorted(distances)
    assert distances[0] == 1.0  # the closest of the fake cows (distance 1)


def test_block_and_entity_classification():
    assert block_class("minecraft:air") == 0
    assert block_class("minecraft:stone") == 1
    assert block_class("minecraft:water") == 2
    assert block_class("minecraft:lava") == 3
    assert block_class("minecraft:oak_log") == 4
    assert block_class("") == 0

    assert entity_class({"type": "minecraft:player", "player": True}) == 0
    assert entity_class({"type": "minecraft:zombie", "hostile": True}) == 1
    assert entity_class({"type": "minecraft:cow"}) == 2
    assert entity_class({"type": "minecraft:armor_stand"}) == 3


def test_default_blocks_look_like_solid_ground():
    cells = default_blocks()
    assert len(cells) == 27
    kinds = {(dx, dy, dz): block_class(name) for dx, dy, dz, name in cells}
    for (_dx, dy, _dz), kind in kinds.items():
        assert kind == (1 if dy == -1 else 0)


def test_action_normalisation():
    assert action_of({"activity": "sneak"}) == "sneak"
    assert action_of({"activity": "SPRINT"}) == "none"  # not in the recorded vocabulary
    assert action_of({"action": "look"}) == "look"
    assert action_of({}) == "none"


def test_movement_deltas_match_the_facing_direction():
    current = {"x": 0.0, "y": 64.0, "z": 0.0, "yaw": 0.0}
    # yaw 0 faces +Z in Minecraft, so walking one block south is purely "forward".
    following = {"x": 0.0, "y": 64.0, "z": 1.0, "yaw": 0.0}
    dyaw, dpitch, forward, strafe = movement_deltas(current, following)
    assert round(dyaw, 6) == 0.0 and round(dpitch, 6) == 0.0
    assert round(forward, 6) == 1.0
    assert round(strafe, 6) == 0.0

    # ... but facing west (yaw 90), the same step is a strafe.
    west = {"x": -1.0, "y": 64.0, "z": 0.0, "yaw": 90.0}
    _, _, forward_west, strafe_west = movement_deltas(current, west)
    assert round(forward_west, 6) == 0.0
    assert abs(round(strafe_west, 6)) == 1.0


def test_wrap_degrees_takes_the_short_way_round():
    assert wrap_degrees(350.0) == -10.0
    assert wrap_degrees(-350.0) == 10.0
    assert wrap_degrees(10.0) == 10.0


def test_target_scaling_round_trips():
    original = [12.0, -6.0, 0.4, -0.3]
    assert denormalise_targets(normalise_targets(original)) == original
    # Extreme values are clipped instead of exploding the regression head.
    clipped = normalise_targets([9999.0, 0.0, 0.0, 0.0])
    assert clipped[0] == 4.0


def test_summarise_counts_actions_and_episodes():
    summary = summarise([OBSERVATION, {**OBSERVATION, "activity": "look"}, {**OBSERVATION, "activity": "look"}])
    assert summary["samples"] == 3
    assert summary["episodes"] == 1
    assert summary["actions"] == {"look": 2, "forward": 1}
    assert summary["hostile_entities_seen"] == 3  # one zombie per sample
