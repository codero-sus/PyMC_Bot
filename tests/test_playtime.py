"""Reading playtime datasets: the mod's NDJSON, the exported training file and
the synthetic generator used by the demos and tests."""

from __future__ import annotations

import json
from pathlib import Path

from pymc_bot.features import ACTION_SPACE
from pymc_bot.playtime import (
    INSTRUCTION,
    dataset_files,
    dataset_summary,
    export_dataset,
    iter_ndjson,
    load_examples,
    pair_transitions,
    synthesize_playtime,
    write_dataset,
)

SAMPLE = {
    "tick": 4,
    "episode": "run-20260101-000000",
    "activity": "forward",
    "x": 1.0,
    "y": 64.0,
    "z": 2.0,
    "yaw": 10.0,
    "pitch": 0.0,
    "on_ground": True,
    "health": 20.0,
    "food": 20,
    "dimension": "overworld",
    "blocks": [],
    "nearby": [],
}


def test_synthetic_playtime_has_the_mod_schema(tmp_path: Path):
    target = tmp_path / "dataset.jsonl"
    info = synthesize_playtime(target, minutes=0.5, seed=5)
    assert info["samples"] >= 50
    assert target.is_file()
    lines = [line for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == info["samples"]

    first = json.loads(lines[0])
    assert first["instruction"] == INSTRUCTION
    assert json.loads(first["output"])["action"] in ACTION_SPACE
    observation = first["input"]
    for key in ("tick", "x", "y", "z", "yaw", "pitch", "on_ground", "blocks", "nearby", "dimension"):
        assert key in observation
    assert len(observation["blocks"]) == 27
    assert sum(info["actions"].values()) == info["samples"]


def test_load_examples_reads_datasets_and_episode_dirs(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    synthesize_playtime(dataset, minutes=0.2, seed=1)
    from_dataset = load_examples(dataset)
    assert from_dataset

    # episode files (raw mod output) are understood too
    episodes = tmp_path / "pymc-playtime" / "episodes"
    episodes.mkdir(parents=True)
    (episodes / "run-a.ndjsonl").write_text(
        "\n".join(json.dumps({**SAMPLE, "tick": tick}) for tick in range(3)) + "\n",
        encoding="utf-8",
    )
    from_episodes = load_examples(episodes)
    assert [action for _, action in from_episodes] == ["forward", "forward", "forward"]

    # a game directory is searched for both
    both = load_examples(tmp_path / "pymc-playtime")
    assert len(both) == 3
    # unknown actions are ignored rather than taught
    (episodes / "run-b.ndjsonl").write_text(json.dumps({**SAMPLE, "activity": "teleport"}) + "\n", encoding="utf-8")
    assert [action for _, action in load_examples(episodes)][-1] == "none"
    assert len(dataset_files(episodes)) == 2


def test_load_examples_survives_corrupt_lines(tmp_path: Path):
    target = tmp_path / "broken.jsonl"
    target.write_text('{"not json\n' + json.dumps({"input": SAMPLE, "output": '{"action": "jump"}'}) + "\n\n", encoding="utf-8")
    examples = load_examples(target)
    assert len(examples) == 1
    assert examples[0][1] == "jump"
    assert list(iter_ndjson(target))


def test_write_and_export_dataset_round_trip(tmp_path: Path):
    target = tmp_path / "pymc-playtime" / "dataset.jsonl"
    written = write_dataset(target, [(SAMPLE, "sneak"), (SAMPLE, "look")])
    assert written == 2
    examples = load_examples(target)
    assert [action for _, action in examples] == ["sneak", "look"]

    exported = export_dataset(target, append=True)
    assert exported == target
    assert len(load_examples(target)) == 4  # appended, not replaced

    bare = tmp_path / "game" / "playtime.jsonl"
    write_dataset(bare, [(SAMPLE, "jump")])
    export_dataset(bare)  # defaults to <dir>/pymc-playtime/dataset.jsonl
    assert (tmp_path / "game" / "pymc-playtime" / "dataset.jsonl").is_file()


def test_pair_transitions_uses_the_next_sample_as_the_motion_target(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    write_dataset(
        dataset,
        [
            ({**SAMPLE, "x": 0.0, "z": 0.0, "yaw": 0.0, "episode": "a"}, "forward"),
            ({**SAMPLE, "x": 0.0, "z": 0.4, "yaw": 0.0, "episode": "a"}, "look"),
        ],
    )
    transitions = pair_transitions(load_examples(dataset))
    assert [action for _, action, _ in transitions] == ["forward", "look"]
    first_targets = transitions[0][2]
    assert round(first_targets[2], 3) == 0.4  # walked 0.4 blocks forward
    assert transitions[1][2] == [0.0, 0.0, 0.0, 0.0]  # last sample has nothing to imitate


def test_pair_transitions_does_not_leak_between_episodes():
    examples = [
        ({**SAMPLE, "episode": "a"}, "forward"),
        ({**SAMPLE, "episode": "b", "z": 50.0}, "back"),
    ]
    transitions = pair_transitions(examples)
    assert transitions[0][2] == [0.0, 0.0, 0.0, 0.0]  # a different episode is not a continuation


def test_dataset_summary_reports_actions_and_episodes(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    synthesize_playtime(dataset, minutes=0.3, seed=2)
    summary = dataset_summary(dataset)
    assert summary["samples"] > 0
    assert summary["episodes"] >= 1
    assert set(summary["actions"]).issubset(set(ACTION_SPACE) | {"move"})


def test_synthetic_profiles_differ(tmp_path: Path):
    explorer = synthesize_playtime(tmp_path / "explorer.jsonl", minutes=0.3, profiles=["explorer"], seed=3)
    fighter = synthesize_playtime(tmp_path / "fighter.jsonl", minutes=0.3, profiles=["fighter"], seed=3)
    assert explorer["actions"].get("forward", 0) > fighter["actions"].get("forward", 0)
    assert fighter["actions"].get("attack", 0) > explorer["actions"].get("attack", 0)
