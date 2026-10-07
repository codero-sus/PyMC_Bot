"""Train a Minecraft player model on recorded playtime, checkpointing as it goes.

The pipeline, end to end::

    Minecraft + mod (./mod)          python -m pymc_bot train
    ───────────────────────          ──────────────────────
    /pymc record start               dataset.jsonl ─▶ features (pymc_bot.features)
    ...play normally...                              ─▶ action + motion targets
    /pymc export                     ─▶ models/<run>/latest.npz   (self-checkpointed)
                                        models/<run>/checkpoints/step-000500.npz
                                        models/<run>/trainer_state.json   (resumable)
                                        models/<run>/metrics.jsonl|.csv
                                        models/<run>/model.json (card for the panel)

``--resume`` continues from ``trainer_state.json`` (weights, optimiser moments, step
counter, RNG state), so a run can be stopped and restarted at any point - which is also
exactly what the panel's *train* button and :class:`TrainingService` do.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pymc_bot.features import (
    ACTION_SPACE,
    DEFAULT_ENTITY_SLOTS,
    DEFAULT_ITEM_SLOTS,
    TARGET_NAMES,
    Vocabulary,
    encode,
    layout_version,
    learn_vocabulary,
    normalise_targets,
    vocabulary_coverage,
)
from pymc_bot.models import (
    balanced_accuracy,
    balanced_class_weights,
    build_model,
    checkpoint_filename,
    load_model,
    torch_available,
)
from pymc_bot.playtime import dataset_files, load_examples, pair_transitions

DEFAULT_MODELS_DIR = "models"
CONFIG_FILENAME = "trainer_state.json"
CARD_FILENAME = "model.json"
METRICS_JSONL = "metrics.jsonl"
METRICS_CSV = "metrics.csv"

EventCallback = Callable[[dict[str, Any]], None]
StopCallback = Callable[[], bool]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Everything that defines a training run (serialised into the checkpoint)."""

    dataset: str = ""
    run_name: str = ""
    models_dir: str = DEFAULT_MODELS_DIR
    engine: str = "mlp"
    hidden: tuple[int, ...] = (128, 64)
    d_model: int = 64
    heads: int = 4
    layers: int = 2
    batch_size: int = 64
    lr: float = 3e-3
    steps: int = 1000
    epochs: int = 0  # >0 overrides steps: train for whole passes over the data
    max_seconds: float = 0.0  # >0 stops after this long (long self-checkpointing runs)
    val_split: float = 0.1
    checkpoint_every: int = 200
    log_every: int = 50
    reg_weight: float = 0.5
    #: Inverse-frequency class weights, so walking does not drown out rare actions.
    class_weights: bool = True
    #: How strongly to balance (1.0 = full, 0.5 = tempered, 0.0 = off).
    class_weight_power: float = 0.5
    include_blocks: bool = True
    #: Advanced training: learn entity and item vocabularies from the playtime and feed
    #: them to the model (layout v2). The mod records them with ``/pymc advanced on``.
    advanced: bool = False
    entity_slots: int = DEFAULT_ENTITY_SLOTS
    item_slots: int = DEFAULT_ITEM_SLOTS
    seed: int = 0
    resume: bool = False
    from_checkpoint: str = ""
    shuffle: bool = True
    verbose: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hidden"] = list(self.hidden)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainConfig:
        known = {key: value for key, value in (payload or {}).items() if key in cls.__dataclass_fields__}
        if "hidden" in known and known["hidden"] is not None:
            known["hidden"] = tuple(int(size) for size in known["hidden"])
        return cls(**known)


# ---------------------------------------------------------------------------
# dataset preparation
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    """Feature matrix, action labels and motion targets ready for training."""

    X: np.ndarray
    y: np.ndarray
    T: np.ndarray
    actions: list[str]
    summary: dict[str, Any] = field(default_factory=dict)
    episodes: int = 0
    #: Entity/item words this dataset was encoded with (empty = basic layout).
    vocab: Vocabulary = field(default_factory=Vocabulary)

    def __len__(self) -> int:
        return int(len(self.y))

    def split(self, val_split: float, seed: int = 0) -> tuple[Dataset, Dataset]:
        if val_split <= 0.0 or len(self) < 10:
            empty = Dataset(self.X[:0], self.y[:0], self.T[:0], self.actions, {}, vocab=self.vocab)
            return self, empty
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self))
        cut = max(1, int(len(self) * val_split))
        val_idx, train_idx = order[:cut], order[cut:]
        train = Dataset(
            self.X[train_idx], self.y[train_idx], self.T[train_idx], self.actions, self.summary, vocab=self.vocab
        )
        val = Dataset(
            self.X[val_idx], self.y[val_idx], self.T[val_idx], self.actions, self.summary, vocab=self.vocab
        )
        train.episodes = val.episodes = self.episodes
        return train, val


def file_digest(path: str | Path, limit: int = 1 << 20) -> str:
    """Cheap content digest used to detect that a dataset changed."""
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if size >= limit:
                break
    return f"sha256:{digest.hexdigest()[:16]}:{size}"


def build_dataset(
    path: str | Path,
    *,
    include_blocks: bool = True,
    limit: int | None = None,
    advanced: bool = False,
    entity_slots: int = DEFAULT_ENTITY_SLOTS,
    item_slots: int = DEFAULT_ITEM_SLOTS,
) -> Dataset:
    """Turn a playtime dataset (or episode directory) into arrays.

    ``advanced=True`` first learns which entities and items the recording contains
    (:func:`pymc_bot.features.learn_vocabulary`) and appends those words to every
    feature vector. The vocabulary travels with the checkpoint, so the live bot encodes
    its world exactly the same way.
    """
    examples = load_examples(path, limit=limit)
    if not examples:
        raise ValueError(
            f"no playtime samples found in {path!r} - record some with the Fabric mod "
            "(/pymc record start ... /pymc export) or generate a demo set with "
            "'python -m pymc_bot train --simulate-out demo.jsonl'"
        )
    transitions = pair_transitions(examples)
    actions = list(ACTION_SPACE)
    index_of = {name: index for index, name in enumerate(actions)}
    vocab = (
        learn_vocabulary([observation for observation, _, _ in transitions], entity_slots=entity_slots, item_slots=item_slots)
        if advanced
        else Vocabulary()
    )
    X = np.asarray(
        [encode(observation, include_blocks, vocab) for observation, _, _ in transitions], dtype=np.float32
    )
    y = np.asarray([index_of[action] for _, action, _ in transitions], dtype=np.int64)
    T = np.asarray([normalise_targets(targets) for _, _, targets in transitions], dtype=np.float32)
    counts: dict[str, int] = {}
    for _, action, _ in transitions:
        counts[action] = counts.get(action, 0) + 1
    episodes = {str(observation.get("episode") or "") for observation, _, _ in transitions}
    summary = {
        "samples": len(transitions),
        "episodes": len(episodes),
        "actions": dict(sorted(counts.items(), key=lambda item: -item[1])),
        "files": [str(file) for file in dataset_files(path)],
        "advanced": bool(advanced),
        "vocabulary": vocab.describe(),
    }
    if advanced:
        summary.update(
            {
                key: value
                for key, value in vocabulary_coverage(
                    [observation for observation, _, _ in transitions], vocab
                ).items()
                if key != "samples_with_entities_or_items"
            }
        )
    return Dataset(X, y, T, actions, summary, episodes=len(episodes), vocab=vocab)


def dump_rng(rng: random.Random) -> list[Any]:
    """JSON-safe snapshot of ``random.Random`` state (so --resume shuffles identically)."""
    version, internal, gauss = rng.getstate()
    return [version, list(internal), gauss]


def load_rng(payload: Any, fallback_seed: int) -> random.Random:
    rng = random.Random(fallback_seed)
    if isinstance(payload, list) and len(payload) == 3:
        try:
            rng.setstate((int(payload[0]), tuple(int(value) for value in payload[1]), payload[2]))
        except (TypeError, ValueError):  # pragma: no cover - stale/corrupt state
            pass
    return rng


def default_run_name(dataset: str | Path) -> str:
    """``models/<name>`` derived from the dataset path."""
    target = Path(dataset)
    stem = target.stem if target.suffix else target.name
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in stem)
    cleaned = cleaned.strip("-") or "playtime"
    return f"playtime-{cleaned}"[:60]


# ---------------------------------------------------------------------------
# run directory bookkeeping
# ---------------------------------------------------------------------------
def resolve_run_dir(models_dir: str | Path, run_name: str) -> Path:
    return Path(models_dir).expanduser() / run_name


def read_card(models_dir: str | Path, run_name: str) -> dict[str, Any] | None:
    path = resolve_run_dir(models_dir, run_name) / CARD_FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - corrupt card
        return None


def read_state(models_dir: str | Path, run_name: str) -> dict[str, Any] | None:
    path = resolve_run_dir(models_dir, run_name) / CONFIG_FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - corrupt state
        return None


def list_models(models_dir: str | Path = DEFAULT_MODELS_DIR) -> list[dict[str, Any]]:
    """Every trained run in ``models_dir``, newest first (used by the panel)."""
    root = Path(models_dir).expanduser()
    if not root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        card = read_card(root, child.name)
        if card is None:
            continue
        card.setdefault("run", child.name)
        checkpoint = child / card.get("checkpoint_file", checkpoint_filename(card.get("engine", "mlp")))
        card["checkpoint"] = str(checkpoint)
        card["checkpoint_exists"] = checkpoint.is_file()
        card["checkpoints"] = sorted(
            str(path.name) for path in (child / "checkpoints").glob("*") if path.is_file()
        )[-5:]
        runs.append(card)
    runs.sort(key=lambda card: float(card.get("updated_at") or 0.0), reverse=True)
    return runs


def best_checkpoint(models_dir: str | Path, run_name: str) -> Path | None:
    """The most recent checkpoint for a run."""
    card = read_card(models_dir, run_name) or {}
    run_dir = resolve_run_dir(models_dir, run_name)
    latest = run_dir / card.get("checkpoint_file", checkpoint_filename(card.get("engine", "mlp")))
    return latest if latest.is_file() else None


def _balanced(model: Any, dataset: Dataset, reg_weight: float) -> float:
    """Mean per-class recall on the validation split (how well rare actions are learned)."""
    if not len(dataset):
        return 0.0
    probs, _ = model.predict(dataset.X)
    return balanced_accuracy(probs.argmax(axis=1), dataset.y)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train(
    config: TrainConfig,
    *,
    on_event: EventCallback | None = None,
    should_stop: StopCallback | None = None,
) -> dict[str, Any]:
    """Train (or resume) one run. Returns a summary dict; never raises for user input.

    Checkpoints are written every ``checkpoint_every`` steps and whenever training
    stops, so an interrupted run can be resumed with ``config.resume = True``.
    """
    started = time.monotonic()
    emit = on_event or (lambda _event: None)
    run_name = config.run_name or default_run_name(config.dataset)
    config.run_name = run_name
    run_dir = resolve_run_dir(config.models_dir, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    # Adopt the saved settings of the run *before* the dataset is built: a run that was
    # trained with the advanced entity/item layout must be resumed with the same columns.
    resume_from: Path | None = None
    state: dict[str, Any] = {}
    if config.resume:
        state = read_state(config.models_dir, run_name) or {}
        saved = state.get("config") or {}
        for key in (
            "hidden",
            "lr",
            "batch_size",
            "include_blocks",
            "reg_weight",
            "advanced",
            "entity_slots",
            "item_slots",
        ):
            if key in saved:
                value = saved[key]
                setattr(config, key, tuple(value) if key == "hidden" else value)
        config.engine = saved.get("engine", config.engine)
        candidate = (
            Path(config.from_checkpoint)
            if config.from_checkpoint
            else run_dir / checkpoint_filename(config.engine)
        )
        if candidate.is_file():
            resume_from = candidate
    elif config.from_checkpoint:
        resume_from = Path(config.from_checkpoint)

    if config.engine in ("transformer", "torch") and not torch_available():
        raise RuntimeError(
            "the transformer engine needs PyTorch (pip install -r requirements-train.txt); "
            "use --engine mlp for the dependency-free model"
        )

    dataset = build_dataset(
        config.dataset,
        include_blocks=config.include_blocks,
        advanced=config.advanced,
        entity_slots=config.entity_slots,
        item_slots=config.item_slots,
    )
    train_set, val_set = dataset.split(config.val_split, seed=config.seed)
    feature_dim = int(dataset.X.shape[1]) if len(dataset) else 0
    if not feature_dim:
        raise ValueError("the dataset produced no features")
    checkpoint_path = run_dir / checkpoint_filename(config.engine)

    model = load_model(resume_from, kind=config.engine) if resume_from else build_model(
        config.engine,
        feature_dim,
        len(dataset.actions),
        hidden=config.hidden,
        d_model=config.d_model,
        heads=config.heads,
        layers=config.layers,
        lr=config.lr,
        seed=config.seed,
    )

    # Balanced class weights: playtime is mostly walking, and without this the model
    # just learns to hold "forward" forever.
    weights = (
        balanced_class_weights(train_set.y, len(dataset.actions), config.class_weight_power)
        if config.class_weights
        else None
    )
    if weights is not None and hasattr(model, "class_weights"):
        model.class_weights = weights

    step = int(state.get("step", 0)) if resume_from else 0
    epoch = int(state.get("epoch", 0)) if resume_from else 0
    samples_seen = int(state.get("samples_seen", 0)) if resume_from else 0
    best_val = float(state.get("best_val", float("inf"))) if resume_from else float("inf")
    history: list[dict[str, Any]] = list(state.get("history", [])) if resume_from else []
    rng = load_rng(state.get("rng"), config.seed) if resume_from else random.Random(config.seed)

    total_steps = config.steps
    if config.epochs > 0 and len(train_set) > 0:
        steps_per_epoch = max(1, len(train_set) // max(1, config.batch_size))
        total_steps = config.epochs * steps_per_epoch
    deadline = started + config.max_seconds if config.max_seconds > 0 else None
    batch_size = max(1, min(config.batch_size, len(train_set)))

    emit(
        {
            "type": "start",
            "run": run_name,
            "engine": config.engine,
            "samples": len(dataset),
            "train_samples": len(train_set),
            "val_samples": len(val_set),
            "total_steps": total_steps,
            "resumed_from": str(resume_from) if resume_from else None,
            "feature_dim": feature_dim,
            "advanced": bool(config.advanced),
            "vocabulary": dataset.vocab.describe(),
        }
    )

    def snapshot(metrics: dict[str, Any], *, stopping: bool) -> dict[str, Any]:
        """Write checkpoint + metrics + model card (``stopping`` marks the final write)."""
        """Write the checkpoint, the metrics and the model card."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        layout = {
            "step": step,
            "epoch": epoch,
            "run": run_name,
            "engine": config.engine,
            "feature_version": layout_version(dataset.vocab),
            "advanced": bool(config.advanced),
            **dataset.vocab.to_dict(),
        }
        model.save(checkpoint_path, extra=layout)
        history_path = run_dir / "checkpoints" / f"step-{step:07d}{checkpoint_path.suffix}"
        model.save(history_path, extra={"step": step, "epoch": epoch, "run": run_name})
        metrics_line = {**metrics, "step": step, "epoch": epoch, "samples_seen": samples_seen}
        history.append(metrics_line)
        del history[:-200]
        with (run_dir / METRICS_JSONL).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics_line) + "\n")
        wrote_header = not (run_dir / METRICS_CSV).exists()
        with (run_dir / METRICS_CSV).open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics_line))
            if wrote_header:
                writer.writeheader()
            writer.writerow(metrics_line)

        config_payload = config.to_dict()
        state_payload = {
            "run": run_name,
            "step": step,
            "epoch": epoch,
            "samples_seen": samples_seen,
            "best_val": best_val,
            "rng": dump_rng(rng),
            "config": config_payload,
            "history": history[-50:],
            "updated_at": time.time(),
            "resumable": True,
        }
        (run_dir / CONFIG_FILENAME).write_text(json.dumps(state_payload, indent=2), encoding="utf-8")

        card = {
            "run": run_name,
            "engine": config.engine,
            "checkpoint_file": checkpoint_path.name,
            "feature_version": layout_version(dataset.vocab),
            "feature_dim": feature_dim,
            "include_blocks": config.include_blocks,
            "advanced": bool(config.advanced),
            "entity_vocabulary": list(dataset.vocab.entities),
            "item_vocabulary": list(dataset.vocab.items),
            "class_weights": config.class_weights,
            "class_weight_values": None if weights is None else [round(float(w), 4) for w in weights],
            "actions": dataset.actions,
            "targets": list(TARGET_NAMES),
            "hidden": list(config.hidden),
            "params": model.param_count(),
            "step": step,
            "epoch": epoch,
            "samples_seen": samples_seen,
            "dataset": str(config.dataset),
            "dataset_digest": dataset_digest,
            "dataset_summary": dataset.summary,
            "metrics": {
                "train_loss": metrics.get("loss"),
                "val_loss": metrics.get("val_loss"),
                "val_accuracy": metrics.get("val_acc"),
                "val_balanced_accuracy": metrics.get("val_balanced_acc"),
                "best_val_loss": None if best_val == float("inf") else best_val,
                "seconds": metrics.get("seconds"),
            },
            "history": history[-50:],
            "created_at": created_at,
            "updated_at": time.time(),
            "stopping": stopping,
            "notes": (
                "Trained on recorded playtime (Fabric mod ./mod) - run natively, no Ollama needed."
                + (
                    " Advanced training: entity and item vocabularies learned from the recording"
                    f" ({len(dataset.vocab.entities)} entities, {len(dataset.vocab.items)} items)."
                    if config.advanced
                    else ""
                )
            ),
        }
        (run_dir / CARD_FILENAME).write_text(json.dumps(card, indent=2), encoding="utf-8")
        return card

    dataset_digest = file_digest((dataset_files(config.dataset) or [Path(config.dataset)])[0])
    created_at = float((read_card(config.models_dir, run_name) or {}).get("created_at") or time.time())
    loss = 0.0
    metrics: dict[str, Any] = {"loss": 0.0, "val_loss": None, "val_acc": None, "seconds": 0.0}
    interrupted = False

    order = np.arange(len(train_set))
    while step < total_steps:
        epoch += 1
        if config.shuffle:
            rng.shuffle(order)
        for start in range(0, len(train_set) - batch_size + 1, batch_size):
            if step >= total_steps:
                break
            if deadline is not None and time.monotonic() >= deadline:
                interrupted = True
                break
            if should_stop is not None and should_stop():
                interrupted = True
                break
            idx = order[start : start + batch_size]
            loss, accuracy = model.optim_step(
                train_set.X[idx],
                train_set.y[idx],
                train_set.T[idx],
                config.lr,
                config.reg_weight,
                None if weights is None else weights[train_set.y[idx]],
            )
            step += 1
            samples_seen += batch_size
            if step % max(1, config.log_every) == 0 or step == total_steps:
                event = {
                    "type": "step",
                    "run": run_name,
                    "step": step,
                    "total_steps": total_steps,
                    "epoch": epoch,
                    "loss": round(loss, 5),
                    "train_acc": round(accuracy, 4),
                    "lr": config.lr,
                    "seconds": round(time.monotonic() - started, 2),
                }
                emit(event)
                if config.verbose:
                    print(json.dumps(event), flush=True)
            if config.checkpoint_every > 0 and step % config.checkpoint_every == 0:
                val_loss, val_acc = model.evaluate(
                    val_set.X, val_set.y, val_set.T, config.reg_weight
                )
                if len(val_set):
                    best_val = min(best_val, val_loss)
                metrics = {
                    "loss": round(loss, 5),
                    "val_loss": round(val_loss, 5),
                    "val_acc": round(val_acc, 4),
                    "val_balanced_acc": round(_balanced(model, val_set, config.reg_weight), 4),
                    "seconds": round(time.monotonic() - started, 2),
                }
                card = snapshot(metrics, stopping=False)
                emit({"type": "checkpoint", "run": run_name, "step": step, "checkpoint": str(checkpoint_path), **metrics, "card": card})
        if interrupted:
            break

    val_loss, val_acc = model.evaluate(val_set.X, val_set.y, val_set.T, config.reg_weight)
    if len(val_set):
        best_val = min(best_val, val_loss)
    metrics = {
        "loss": round(loss, 5),
        "val_loss": round(val_loss, 5),
        "val_acc": round(val_acc, 4),
        "val_balanced_acc": round(_balanced(model, val_set, config.reg_weight), 4),
        "seconds": round(time.monotonic() - started, 2),
    }
    card = snapshot(metrics, stopping=True)
    summary = {
        "run": run_name,
        "engine": config.engine,
        "steps": step,
        "epochs": epoch,
        "samples": len(dataset),
        "interrupted": interrupted,
        "checkpoint": str(checkpoint_path),
        "checkpoint_file": checkpoint_path.name,
        "run_dir": str(run_dir),
        **metrics,
    }
    emit({"type": "done", **summary, "card": card})
    return summary


# ---------------------------------------------------------------------------
# background training service (used by the panel and the CLI --follow mode)
# ---------------------------------------------------------------------------
class TrainingService:
    """Runs :func:`train` on a worker thread and exposes live progress."""

    def __init__(self, log: Any | None = None) -> None:
        self._log = log
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._status: dict[str, Any] = {
            "running": False,
            "run": None,
            "step": 0,
            "total_steps": 0,
            "loss": None,
            "val_loss": None,
            "val_acc": None,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "last_event": None,
            "summary": None,
            "events": 0,
        }

    # ------------------------------------------------------------- lifecycle
    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self, config: TrainConfig) -> bool:
        with self._lock:
            if self.running:
                return False
            self._stop.clear()
            self._status.update(
                {
                    "running": True,
                    "run": config.run_name or default_run_name(config.dataset),
                    "step": 0,
                    "total_steps": config.steps,
                    "loss": None,
                    "val_loss": None,
                    "val_acc": None,
                    "started_at": time.time(),
                    "finished_at": None,
                    "error": None,
                    "config": config.to_dict(),
                    "summary": None,
                    "events": 0,
                }
            )
            self._thread = threading.Thread(
                target=self._run, args=(config,), name="pymc-trainer", daemon=True
            )
            self._thread.start()
        self._add_log(
            f"Training '{self._status['run']}' on {config.dataset} "
            f"({config.engine}, {config.steps} steps, checkpoint every {config.checkpoint_every})",
            "ai",
        )
        return True

    def stop(self, wait: bool = False) -> bool:
        with self._lock:
            if not self.running:
                return False
            self._stop.set()
        if wait and self._thread is not None:
            self._thread.join(timeout=30.0)
        return True

    def _add_log(self, message: str, kind: str = "info") -> None:
        if self._log is None or not hasattr(self._log, "add"):
            return
        try:
            self._log.add(message, kind, "train")
        except Exception:  # pragma: no cover - logging must never break training
            pass

    # ---------------------------------------------------------------- worker
    def _run(self, config: TrainConfig) -> None:
        try:
            summary = train(config, on_event=self._on_event, should_stop=self._stop.is_set)
        except Exception as exc:
            with self._lock:
                self._status.update({"error": str(exc), "running": False, "finished_at": time.time()})
            self._add_log(f"Training failed: {exc}", "warn")
            return
        with self._lock:
            self._status.update({"running": False, "finished_at": time.time(), "summary": summary})
        self._add_log(
            f"Training finished: {summary['steps']} steps, val_loss={summary.get('val_loss')}, "
            f"checkpoint={summary['checkpoint']}",
            "success",
        )

    def _on_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._status["events"] = int(self._status.get("events", 0)) + 1
            self._status["last_event"] = event.get("type")
            if event.get("type") in ("step", "checkpoint"):
                self._status.update(
                    {
                        "step": event.get("step", self._status["step"]),
                        "total_steps": event.get("total_steps") or self._status.get("total_steps") or 0,
                        "loss": event.get("loss", self._status.get("loss")),
                        "val_loss": event.get("val_loss", self._status.get("val_loss")),
                        "val_acc": event.get("val_acc", self._status.get("val_acc")),
                    }
                )
            elif event.get("type") == "start":
                self._status.update(
                    {
                        "run": event.get("run"),
                        "total_steps": event.get("total_steps") or self._status.get("total_steps"),
                        "samples": event.get("samples"),
                        "engine": event.get("engine"),
                        "resumed_from": event.get("resumed_from"),
                    }
                )
            elif event.get("type") == "checkpoint":
                self._status["checkpoint"] = event.get("checkpoint")
                self._status["card"] = event.get("card")

    # ---------------------------------------------------------------- status
    def status(self) -> dict[str, Any]:
        with self._lock:
            status = dict(self._status)
        status["running"] = self.running
        if status.get("started_at") and status.get("running"):
            status["elapsed"] = round(time.time() - float(status["started_at"]), 2)
            step, total = int(status.get("step") or 0), int(status.get("total_steps") or 0)
            status["progress"] = round(step / total, 4) if total else None
        return status
