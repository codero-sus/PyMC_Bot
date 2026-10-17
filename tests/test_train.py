"""Training on recorded playtime: checkpoints, resume, metrics, the model card and
the background :class:`TrainingService` the panel uses."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from pymc_bot.models import Adam, NumpyMLP, balanced_accuracy, balanced_class_weights
from pymc_bot.playtime import synthesize_playtime
from pymc_bot.train import (
    CARD_FILENAME,
    CONFIG_FILENAME,
    METRICS_CSV,
    METRICS_JSONL,
    Dataset,
    TrainConfig,
    TrainingService,
    build_dataset,
    default_run_name,
    list_models,
    read_card,
    read_state,
    resolve_run_dir,
    train,
)


@pytest.fixture(scope="module")
def dataset_file(tmp_path_factory) -> Path:
    """One small synthetic playtime dataset shared by the training tests."""
    target = tmp_path_factory.mktemp("playtime") / "dataset.jsonl"
    synthesize_playtime(target, minutes=1.5, seed=11)
    return target


@pytest.fixture()
def models_dir(tmp_path: Path) -> Path:
    return tmp_path / "models"


def quick_config(dataset_file: Path, models_dir: Path, **overrides) -> TrainConfig:
    config = TrainConfig(
        dataset=str(dataset_file),
        run_name="t1",
        models_dir=str(models_dir),
        steps=40,
        checkpoint_every=20,
        batch_size=32,
        log_every=20,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# --------------------------------------------------------------------- dataset
def test_build_dataset_shapes(dataset_file: Path):
    dataset = build_dataset(dataset_file)
    assert len(dataset) > 100
    assert dataset.X.shape[0] == len(dataset.y) == len(dataset.T)
    assert dataset.X.shape[1] == 215
    assert dataset.T.shape[1] == 4
    assert set(np.unique(dataset.y)).issubset(set(range(len(dataset.actions))))
    assert dataset.summary["samples"] == len(dataset)


def test_build_dataset_rejects_missing_data(tmp_path: Path):
    with pytest.raises(ValueError, match="no playtime samples"):
        build_dataset(tmp_path / "does-not-exist.jsonl")


def test_split_is_disjoint_and_covers_everything(dataset_file: Path):
    dataset = build_dataset(dataset_file)
    train_set, val_set = dataset.split(0.2, seed=0)
    assert len(train_set) + len(val_set) == len(dataset)
    assert len(val_set) == pytest.approx(len(dataset) * 0.2, abs=1)
    # A tiny dataset cannot be split: everything stays in training.
    _, empty = Dataset(dataset.X[:5], dataset.y[:5], dataset.T[:5], dataset.actions, {}).split(0.2)
    assert len(empty) == 0


def test_default_run_name_is_filesystem_safe():
    assert default_run_name("/tmp/pymc-playtime/dataset.jsonl") == "playtime-dataset"
    assert default_run_name("/tmp/weird name!/x.jsonl") == "playtime-x"


# ------------------------------------------------------------------------ math
def test_balanced_class_weights_favour_rare_actions():
    labels = np.array([0] * 90 + [1] * 9 + [2], dtype=np.int64)
    weights = balanced_class_weights(labels, 3, power=1.0, cap=100.0)
    assert weights[2] > weights[1] > weights[0]
    assert weights.mean() == pytest.approx(1.0)
    # power=0 disables the correction entirely
    assert np.allclose(balanced_class_weights(labels, 3, power=0.0), 1.0)


def test_balanced_accuracy_ignores_class_frequency():
    labels = np.array([0, 0, 0, 0, 1], dtype=np.int64)
    predicted = np.array([0, 0, 0, 0, 1], dtype=np.int64)
    assert balanced_accuracy(predicted, labels) == pytest.approx(1.0)
    always_zero = np.zeros_like(labels)
    assert balanced_accuracy(always_zero, labels) == pytest.approx(0.5)  # 100% / 0%


def test_mlp_learns_a_tiny_problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 8)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    T = np.zeros((64, 4), dtype=np.float32)
    model = NumpyMLP(8, 2, hidden=(16,), lr=0.05)
    first, _ = model.evaluate(X, y, T, 0.0)
    for _ in range(60):
        model.optim_step(X, y, T, 0.05, 0.0)
    last, accuracy = model.evaluate(X, y, T, 0.0)
    assert last < first
    assert accuracy > 0.9


def test_adam_resumes_from_its_state():
    params = {"w": np.zeros(3, dtype=np.float32)}
    optimiser = Adam(params, lr=0.1)
    optimiser.step(params, {"w": np.ones(3, dtype=np.float32)})
    state = optimiser.state()
    resumed = Adam({"w": np.zeros(3, dtype=np.float32)}, lr=0.1)
    resumed.load_state(state)
    assert resumed.t == optimiser.t
    assert np.allclose(resumed.m["w"], optimiser.m["w"])


# -------------------------------------------------------------------- training
def test_train_writes_a_resumable_checkpoint_chain(dataset_file: Path, models_dir: Path):
    summary = train(quick_config(dataset_file, models_dir))
    run_dir = resolve_run_dir(models_dir, "t1")

    assert summary["steps"] == 40
    assert summary["val_loss"] is not None
    for name in ("latest.npz", CARD_FILENAME, CONFIG_FILENAME, METRICS_JSONL, METRICS_CSV):
        assert (run_dir / name).is_file(), name
    assert sorted(path.name for path in (run_dir / "checkpoints").iterdir()) == [
        "step-0000020.npz",
        "step-0000040.npz",
    ]

    card = read_card(models_dir, "t1")
    assert card["engine"] == "mlp"
    assert card["feature_version"] == 1
    assert card["actions"][0] == "forward"
    assert card["params"] > 0
    assert card["dataset_summary"]["samples"] > 0

    state = read_state(models_dir, "t1")
    assert state["step"] == 40
    assert state["resumable"] is True
    # two periodic checkpoints plus the final flush when training ends
    assert [entry["step"] for entry in state["history"]] == [20, 40, 40]

    metrics = [json.loads(line) for line in (run_dir / METRICS_JSONL).read_text(encoding="utf-8").splitlines()]
    assert [entry["step"] for entry in metrics] == [20, 40, 40]
    assert "val_loss" in metrics[-1]
    assert len((run_dir / METRICS_CSV).read_text(encoding="utf-8").splitlines()) == 4  # header + 3


def test_train_resumes_and_keeps_counting(dataset_file: Path, models_dir: Path):
    train(quick_config(dataset_file, models_dir))
    resumed = train(quick_config(dataset_file, models_dir, steps=60, resume=True))
    assert resumed["steps"] == 60
    state = read_state(models_dir, "t1")
    assert state["step"] == 60
    metrics = [json.loads(line) for line in (resolve_run_dir(models_dir, "t1") / METRICS_JSONL).read_text(encoding="utf-8").splitlines()]
    assert metrics[-1]["step"] == 60


def test_train_honours_max_seconds_and_saves_anyway(dataset_file: Path, models_dir: Path):
    summary = train(quick_config(dataset_file, models_dir, steps=100000, max_seconds=0.05, checkpoint_every=25))
    assert summary["interrupted"] is True
    assert resolve_run_dir(models_dir, "t1", ).joinpath("latest.npz").is_file()
    assert read_state(models_dir, "t1")["resumable"] is True


def test_train_can_be_stopped_from_another_thread(dataset_file: Path, models_dir: Path):
    stop = {"flag": False}

    def callback() -> bool:
        return stop["flag"]

    config = quick_config(dataset_file, models_dir, steps=5000, checkpoint_every=1000)
    stop["flag"] = True
    summary = train(config, should_stop=callback)
    assert summary["interrupted"] is True
    assert summary["steps"] == 0  # stopped before the first batch
    assert read_state(models_dir, "t1")["step"] == 0


def test_train_without_blocks_gives_a_smaller_vector(dataset_file: Path, models_dir: Path):
    summary = train(quick_config(dataset_file, models_dir, include_blocks=False, run_name="noblocks"))
    card = read_card(models_dir, summary["run"])
    assert card["include_blocks"] is False
    assert card["feature_dim"] == 80


def test_train_reports_the_dataset_it_used(dataset_file: Path, models_dir: Path):
    events: list[dict] = []
    train(quick_config(dataset_file, models_dir), on_event=events.append)
    kinds = [event["type"] for event in events]
    assert kinds[0] == "start"
    assert "step" in kinds and "checkpoint" in kinds
    assert kinds[-1] == "done"
    assert events[-1]["checkpoint"].endswith("latest.npz")


def test_list_models_newest_first(dataset_file: Path, models_dir: Path):
    train(quick_config(dataset_file, models_dir, run_name="first"))
    time.sleep(0.01)
    train(quick_config(dataset_file, models_dir, run_name="second", steps=20, checkpoint_every=20))
    runs = [card["run"] for card in list_models(models_dir)]
    assert runs[0] == "second"
    assert set(runs) == {"first", "second"}
    assert all(card["checkpoint_exists"] for card in list_models(models_dir))
    assert list_models(models_dir / "nope") == []


# --------------------------------------------------------------------- service
def test_training_service_reports_progress_and_stops(dataset_file: Path, models_dir: Path):
    service = TrainingService()
    assert service.start(quick_config(dataset_file, models_dir, steps=200, checkpoint_every=50, log_every=25))
    assert service.running
    assert service.start(quick_config(dataset_file, models_dir)) is False  # already running

    deadline = time.monotonic() + 30.0
    while service.running and time.monotonic() < deadline:
        time.sleep(0.05)
    status = service.status()
    assert status["error"] is None
    assert status["summary"]["steps"] == 200
    assert status["run"] == "t1"
    assert status["events"] > 0
    assert service.stop() is False  # finished already

    assert (resolve_run_dir(models_dir, "t1") / "latest.npz").is_file()


def test_training_service_surfaces_errors(dataset_file: Path, models_dir: Path):
    service = TrainingService()
    config = quick_config(models_dir / "missing.jsonl", models_dir)
    service.start(config)
    deadline = time.monotonic() + 20.0
    while service.running and time.monotonic() < deadline:
        time.sleep(0.05)
    assert "no playtime samples" in (service.status()["error"] or "")


def test_transformer_engine_requires_torch(dataset_file: Path, models_dir: Path):
    torch = pytest.importorskip("torch", reason="optional: --engine transformer needs PyTorch")
    del torch
    summary = train(quick_config(dataset_file, models_dir, engine="transformer", steps=20, checkpoint_every=20, run_name="tx"))
    assert summary["engine"] == "transformer"
    card = read_card(models_dir, "tx")
    assert card["checkpoint_file"] == "latest.pt"
    assert card["params"] > 1000


def test_unknown_engine_is_rejected(dataset_file: Path, models_dir: Path):
    with pytest.raises(ValueError, match="unknown engine"):
        train(quick_config(dataset_file, models_dir, engine="banana", steps=1))


# ------------------------------------------------------------- advanced training
@pytest.fixture(scope="module")
def advanced_dataset_file(tmp_path_factory) -> Path:
    """A synthetic session recorded with the advanced (entity + item) schema."""
    target = tmp_path_factory.mktemp("playtime-advanced") / "dataset.jsonl"
    synthesize_playtime(target, minutes=1.5, seed=13, advanced=True)
    return target


def test_build_dataset_learns_the_vocabulary(advanced_dataset_file: Path):
    from pymc_bot.features import feature_dim, layout_version

    basic = build_dataset(advanced_dataset_file)
    advanced = build_dataset(advanced_dataset_file, advanced=True)
    assert advanced.X.shape[0] == basic.X.shape[0]  # same samples, richer features
    assert advanced.X.shape[1] > basic.X.shape[1]
    assert advanced.X.shape[1] == feature_dim(True, advanced.vocab)
    assert basic.vocab.entities == () and basic.vocab.items == ()
    assert advanced.vocab.entities and advanced.vocab.items
    assert "zombie" in advanced.vocab.entities  # the combat profile fights them
    assert any(name.endswith("sword") or name.endswith("pickaxe") for name in advanced.vocab.items)
    assert advanced.summary["advanced"] is True
    assert advanced.summary["vocabulary"] == advanced.vocab.describe()
    assert advanced.summary["entity_coverage"] == 1.0
    assert layout_version(advanced.vocab) == 2
    # the basic half of an advanced vector is still the basic vector
    assert list(advanced.X[0][: basic.X.shape[1]]) == pytest.approx(list(basic.X[0]), abs=1e-6)


def test_advanced_slots_are_configurable(advanced_dataset_file: Path):
    small = build_dataset(advanced_dataset_file, advanced=True, entity_slots=2, item_slots=3)
    assert len(small.vocab.entities) == 2
    assert len(small.vocab.items) == 3


def test_train_writes_the_vocabulary_into_the_card_and_checkpoint(advanced_dataset_file: Path, tmp_path: Path):
    models_dir = tmp_path / "models"
    summary = train(
        TrainConfig(
            dataset=str(advanced_dataset_file),
            run_name="advanced-run",
            models_dir=str(models_dir),
            steps=20,
            checkpoint_every=20,
            batch_size=32,
            log_every=20,
            advanced=True,
            verbose=False,
        )
    )
    card = read_card(models_dir, "advanced-run")
    assert summary["run"] == "advanced-run"
    assert card["advanced"] is True
    assert card["feature_version"] == 2
    from pymc_bot.features import Vocabulary, feature_dim

    vocab = Vocabulary(entities=tuple(card["entity_vocabulary"]), items=tuple(card["item_vocabulary"]))
    assert card["feature_dim"] == feature_dim(True, vocab)
    assert card["item_vocabulary"]
    assert "Advanced training" in card["notes"]
    assert card["dataset_summary"]["advanced"] is True
    state = read_state(models_dir, "advanced-run")
    assert state["config"]["advanced"] is True
    # the checkpoint carries the same words, so a bare .npz still knows its layout
    import numpy as np

    from pymc_bot.models import _read_meta

    with np.load(models_dir / "advanced-run" / card["checkpoint_file"], allow_pickle=False) as data:
        meta = _read_meta(data)
    assert meta["feature_version"] == card["feature_version"]
    assert list(meta["entities"]) == card["entity_vocabulary"]
    assert list(meta["items"]) == card["item_vocabulary"]


def test_resume_keeps_advanced_mode(advanced_dataset_file: Path, tmp_path: Path):
    models_dir = tmp_path / "models"
    config = TrainConfig(
        dataset=str(advanced_dataset_file),
        run_name="resume-advanced",
        models_dir=str(models_dir),
        steps=10,
        checkpoint_every=10,
        batch_size=32,
        log_every=10,
        advanced=True,
        verbose=False,
    )
    train(config)
    # a fresh config that forgot the flag must still come back as an advanced run
    resumed = TrainConfig(
        dataset=str(advanced_dataset_file),
        run_name="resume-advanced",
        models_dir=str(models_dir),
        steps=20,
        checkpoint_every=10,
        batch_size=32,
        log_every=10,
        resume=True,
        verbose=False,
    )
    summary = train(resumed)
    assert resumed.advanced is True
    assert summary["steps"] == 20
    card = read_card(models_dir, "resume-advanced")
    assert card["advanced"] is True and card["step"] == 20
