"""The observation encoder is the contract between the mod and the trained model.

If these break, a checkpoint can no longer be run, so the layout is pinned here.
"""

from __future__ import annotations

import math

import pytest

from pymc_bot.features import (
    ACTION_SPACE,
    ADVANCED_FEATURE_VERSION,
    ADVANCED_GLOBAL_DIM,
    BLOCK_CLASSES,
    CELLS,
    DIMENSIONS,
    FEATURE_VERSION,
    NEARBY_CLASSES,
    NEARBY_SLOTS,
    Vocabulary,
    action_of,
    advanced_feature_dim,
    advanced_features,
    block_class,
    default_blocks,
    denormalise_targets,
    encode,
    entity_class,
    feature_dim,
    layout_version,
    learn_vocabulary,
    movement_deltas,
    normalise_targets,
    observation_entities,
    observation_items,
    summarise,
    vocabulary_coverage,
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


# --------------------------------------------------------------------- advanced
ADVANCED_OBSERVATION = {
    "x": 10.0,
    "y": 64.0,
    "z": -5.0,
    "yaw": 90.0,  # facing -X (west)
    "pitch": 0.0,
    "entities": [
        {"type": "minecraft:zombie", "dx": -3.0, "dy": 0.0, "dz": 0.0, "dist": 3.0, "hostile": True},
        {"type": "minecraft:player", "dx": 2.0, "dy": 0.0, "dz": 0.0, "dist": 2.0, "player": True},
    ],
    "items": [[0, "minecraft:iron_sword", 1], [1, "minecraft:cooked_beef", 5]],
    "held_item": "minecraft:iron_sword",
    "armor": ["minecraft:iron_helmet"],
    "ground_items": [{"item": "minecraft:oak_log", "count": 3, "dist": 2.5}],
}


def test_learn_vocabulary_keeps_what_the_playtime_contains():
    vocab = learn_vocabulary([ADVANCED_OBSERVATION], entity_slots=4, item_slots=4)
    assert vocab.entities[:2] == ("player", "zombie")
    assert "iron_sword" in vocab.items and "cooked_beef" in vocab.items
    assert len(vocab.entities) <= 4 and len(vocab.items) <= 4


def test_vocabulary_is_ranked_by_frequency_then_seeded():
    observations = [
        {"entities": [{"type": "minecraft:cow", "dist": 4.0}]} for _ in range(5)
    ] + [{"entities": [{"type": "minecraft:warden", "dist": 4.0, "hostile": True}]}]
    vocab = learn_vocabulary(observations, entity_slots=3, item_slots=1)
    assert vocab.entities[0] == "cow"  # most frequent wins
    assert "warden" in vocab.entities  # still kept within the slots


def test_advanced_layout_extends_the_basic_one():
    vocab = Vocabulary(entities=("zombie",), items=("iron_sword",))
    assert layout_version() == FEATURE_VERSION
    assert layout_version(vocab) == ADVANCED_FEATURE_VERSION
    assert advanced_feature_dim(None) == 0
    assert advanced_feature_dim(vocab) == 1 * 4 + 1 * 3 + ADVANCED_GLOBAL_DIM
    assert feature_dim(True, vocab) == feature_dim(True) + advanced_feature_dim(vocab)


def test_advanced_encode_matches_the_declared_dimension():
    vocab = learn_vocabulary([ADVANCED_OBSERVATION], entity_slots=4, item_slots=4)
    vector = encode(ADVANCED_OBSERVATION, True, vocab)
    assert len(vector) == feature_dim(True, vocab)
    # the basic half must stay byte-for-byte compatible
    assert vector[: feature_dim(True)] == encode(ADVANCED_OBSERVATION, True)


def test_advanced_features_place_entities_and_items_in_their_own_slots():
    observation = dict(
        ADVANCED_OBSERVATION,
        entities=[
            *ADVANCED_OBSERVATION["entities"],
            # 3 blocks to the side of a player facing west (-X): a pure lateral bearing
            {"type": "minecraft:cow", "dx": 0.0, "dy": 0.0, "dz": -3.0, "dist": 3.0},
        ],
    )
    vocab = Vocabulary(entities=("player", "zombie", "cow"), items=("iron_sword", "cooked_beef"))
    values = advanced_features(observation, vocab)
    present_player, closeness_player, sin_player, cos_player = values[0:4]
    present_zombie, closeness_zombie, sin_zombie, cos_zombie = values[4:8]
    present_cow, closeness_cow, sin_cow, cos_cow = values[8:12]
    assert present_player == present_zombie == present_cow == 1.0
    assert closeness_player > closeness_zombie  # the player is nearer
    # yaw 90 means facing -X: the zombie straight ahead, the player straight behind
    assert cos_zombie > 0.9 and cos_player < -0.9
    assert sin_zombie == pytest.approx(0.0, abs=0.01) and sin_player == pytest.approx(0.0, abs=0.01)
    # the cow is beside the view, so it is the lateral component that carries it
    assert abs(sin_cow) > 0.99 and abs(cos_cow) < 0.01
    assert closeness_cow == pytest.approx(closeness_zombie)
    tail = values[12:]  # the rest is the item and global half
    assert tail[0:3] == [1.0, pytest.approx(1 / 16), 1.0]  # sword: carried, counted, held
    assert tail[3:6] == [1.0, pytest.approx(5 / 16), 0.0]  # food: carried, not held
    globals_ = tail[6:]
    assert len(globals_) == ADVANCED_GLOBAL_DIM
    assert globals_[2] > 0.0  # a weapon is carried
    assert globals_[4] > 0.0  # food is carried
    assert globals_[6] > 0.0  # armour is worn
    assert globals_[7] > 0.0 and globals_[8] > 0.0  # something is lying on the ground nearby


def test_advanced_features_ignore_unknown_entities_and_items():
    vocab = Vocabulary(entities=("creeper",), items=("bread",))
    values = advanced_features(ADVANCED_OBSERVATION, vocab)
    assert values[0:4] == [0.0, 0.0, 0.0, 0.0]  # no creeper in sight
    assert values[4:7] == [0.0, 0.0, 0.0]  # bread is not carried


def test_observation_entities_prefers_the_richest_source():
    richer = dict(ADVANCED_OBSERVATION, nearby=[{"type": "minecraft:cow", "dist": 9.0}])
    assert observation_entities(richer)[0]["type"] == "minecraft:zombie"  # entities win over nearby
    only_nearby = {"nearby": [{"type": "minecraft:cow", "dist": 9.0}]}
    assert observation_entities(only_nearby) == only_nearby["nearby"]
    players = {"players": [{"name": "Steve", "x": 1.0, "y": 64.0, "z": 1.0}]}
    assert len(observation_entities(players)) == 1


def test_observation_items_reads_mod_and_bridge_shapes():
    from_bridge = {"items": [], "inventory": [{"name": "stone", "count": 3}], "held_item": "diamond_sword"}
    assert observation_items(from_bridge) == {"stone": 3, "diamond_sword": 1}
    from_mod = {"items": [[0, "minecraft:oak_log", 2], [5, "minecraft:bread", 1]]}
    assert observation_items(from_mod) == {"oak_log": 2, "bread": 1}


def test_vocabulary_coverage_reports_what_it_can_describe():
    vocab = learn_vocabulary([ADVANCED_OBSERVATION], entity_slots=4, item_slots=4)
    coverage = vocabulary_coverage([ADVANCED_OBSERVATION], vocab)
    assert coverage["entity_coverage"] == 1.0
    assert coverage["item_coverage"] == 1.0
    assert coverage["entity_types"]["zombie"] == 1


def test_vocabulary_round_trips_through_json():
    vocab = Vocabulary(entities=("player", "cow"), items=("bread",))
    assert Vocabulary.from_dict(vocab.to_dict()) == vocab
    assert Vocabulary.from_dict(None) == Vocabulary()
    assert vocab.describe() == {"entities": 2, "items": 1}
