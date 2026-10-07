"""Run a trained playtime model: the local (no Ollama) player policy.

:class:`TrainedPolicy` loads a checkpoint written by :mod:`pymc_bot.train`, converts the
live bot snapshot into the same observation layout the mod recorded, predicts both an
action and the continuous motion deltas, and drives the bot with the low level controls
(``forward``/``back``/``left``/``right``/``jump``/``sneak`` plus ``look``) -- i.e. the
model plays the game the way the recorded player did.

Nothing here needs a GPU or an LLM server; PyTorch is only required when the checkpoint
came from the ``transformer`` engine.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pymc_bot.features import (
    ACTION_SPACE,
    FEATURE_VERSION,
    TARGET_NAMES,
    default_blocks,
    denormalise_targets,
    encode,
)
from pymc_bot.models import checkpoint_filename, load_model
from pymc_bot.train import read_card, resolve_run_dir

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pymc_bot.bot import MinecraftBot


def mask_actions(
    probabilities: dict[str, float], allowed: Collection[str] | None
) -> dict[str, float]:
    """Drop the actions a setting forbids and renormalise the rest.

    The trained brain predicts the player's own key presses, including left clicks. The
    permission gates still win: an action that is not in ``allowed`` gets probability 0,
    and if everything is forbidden the model is forced to idle.
    """
    if not allowed:
        return probabilities
    kept = {name: (value if name in allowed else 0.0) for name, value in probabilities.items()}
    total = sum(kept.values())
    if total <= 0.0:
        # Nothing the model wants is allowed, so it waits.
        idle = {name: 0.0 for name in probabilities}
        idle["none"] = 1.0
        return idle
    return {name: value / total for name, value in kept.items()}


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


class PolicyError(RuntimeError):
    """Raised when a checkpoint cannot be loaded or used."""


@dataclass
class Prediction:
    """One model output: the action, its probability and the motion deltas."""

    action: str
    probs: dict[str, float]
    dyaw: float = 0.0
    dpitch: float = 0.0
    forward: float = 0.0
    strafe: float = 0.0
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "probs": {key: round(value, 4) for key, value in self.probs.items()},
            "motion": {
                "dyaw": round(self.dyaw, 2),
                "dpitch": round(self.dpitch, 2),
                "forward": round(self.forward, 3),
                "strafe": round(self.strafe, 3),
            },
            "source": self.source,
        }


@dataclass
class PolicyRun:
    """Where a checkpoint lives plus its model card."""

    run: str
    run_dir: Path
    checkpoint: Path
    card: dict[str, Any] = field(default_factory=dict)

    @property
    def engine(self) -> str:
        return str(self.card.get("engine") or ("transformer" if self.checkpoint.suffix == ".pt" else "mlp"))


def find_run(run: str | Path, models_dir: str | Path = "models") -> PolicyRun:
    """Accept a run name, a run directory or a checkpoint file."""
    target = Path(run).expanduser()
    if target.is_file():
        run_dir = target.parent.parent if target.parent.name == "checkpoints" else target.parent
        card = read_card(run_dir.parent, run_dir.name) or {}
        return PolicyRun(run=run_dir.name, run_dir=run_dir, checkpoint=target, card=card)
    if target.is_dir() and (target / "model.json").exists():
        card = json.loads((target / "model.json").read_text(encoding="utf-8"))
        checkpoint = target / card.get("checkpoint_file", checkpoint_filename(card.get("engine", "mlp")))
        return PolicyRun(run=target.name, run_dir=target, checkpoint=checkpoint, card=card)
    run_dir = resolve_run_dir(models_dir, str(run))
    card = read_card(models_dir, str(run))
    if card is None:
        raise PolicyError(
            f"no trained model '{run}' in {Path(models_dir).expanduser()} - train one with "
            "'python -m pymc_bot train --dataset <playtime dataset>'"
        )
    checkpoint = run_dir / card.get("checkpoint_file", checkpoint_filename(card.get("engine", "mlp")))
    return PolicyRun(run=str(run), run_dir=run_dir, checkpoint=checkpoint, card=card)


class TrainedPolicy:
    """A playtime model wired to a live bot."""

    #: How long one predicted action is held when it presses movement keys.
    DEFAULT_STEP_SECONDS = 0.4
    #: One burst may not turn more than this (a runaway regression head would otherwise
    #: spin the bot on the spot instead of walking where the player walked).
    MAX_YAW_PER_BURST = 25.0
    MAX_PITCH_PER_BURST = 15.0

    def __init__(
        self,
        run: PolicyRun,
        model: Any,
        *,
        temperature: float = 0.0,
        step_seconds: float = DEFAULT_STEP_SECONDS,
        include_blocks: bool | None = None,
        seed: int | None = None,
    ) -> None:
        self.run = run
        self.model = model
        self.temperature = float(temperature)
        self.step_seconds = float(step_seconds)
        self.include_blocks = (
            bool(run.card.get("include_blocks", True)) if include_blocks is None else bool(include_blocks)
        )
        self._rng = random.Random(seed)
        self._last_position: tuple[float, float, float] | None = None
        self._predictions = 0
        self._last_prediction: dict[str, Any] | None = None
        actions = list(run.card.get("actions") or ACTION_SPACE)
        self.actions = actions
        feature_dim = int(run.card.get("feature_dim") or getattr(model, "feature_dim", 0))
        if feature_dim and feature_dim != getattr(model, "feature_dim", feature_dim):
            raise PolicyError("checkpoint feature size does not match its model.json card")
        version = run.card.get("feature_version")
        if version is not None and int(version) != FEATURE_VERSION:
            raise PolicyError(
                f"checkpoint was trained with feature layout v{version}, this build uses "
                f"v{FEATURE_VERSION} - retrain the model"
            )

    # ------------------------------------------------------------- factories
    @classmethod
    def load(
        cls,
        run: str | Path = "latest",
        models_dir: str | Path = "models",
        **kwargs: Any,
    ) -> TrainedPolicy:
        policy_run = find_run(run, models_dir)
        if not policy_run.checkpoint.is_file():
            raise PolicyError(
                f"checkpoint {policy_run.checkpoint} is missing - resume training with "
                f"--resume --run-name {policy_run.run}"
            )
        model = load_model(policy_run.checkpoint, kind=policy_run.engine)
        return cls(policy_run, model, **kwargs)

    # -------------------------------------------------------------- observing
    def observe(self, snapshot: dict[str, Any], *, dimension: str | None = None) -> dict[str, Any]:
        """Convert a live bot snapshot into the observation layout the model expects."""
        position = snapshot.get("position") or {}
        x = float(position.get("x") or 0.0)
        y = float(position.get("y") or 0.0)
        z = float(position.get("z") or 0.0)
        moved = 0.0
        if self._last_position is not None:
            dx, dz = x - self._last_position[0], z - self._last_position[2]
            dy = y - self._last_position[1]
            moved = math.sqrt(dx * dx + dy * dy + dz * dz)
        self._last_position = (x, y, z)

        nearby: list[dict[str, Any]] = []
        for player in snapshot.get("players") or []:
            if not isinstance(player, dict):
                continue
            px = float(player.get("x") or 0.0)
            py = float(player.get("y") or 0.0)
            pz = float(player.get("z") or 0.0)
            dx, dy, dz = px - x, py - y, pz - z
            distance = float(player.get("distance") or math.sqrt(dx * dx + dy * dy + dz * dz))
            nearby.append(
                {
                    "type": "minecraft:player",
                    "dx": round(dx, 2),
                    "dy": round(dy, 2),
                    "dz": round(dz, 2),
                    "dist": round(distance, 2),
                    "hostile": False,
                    "health": 20.0,
                    "player": True,
                }
            )
        nearby.sort(key=lambda entry: entry["dist"])

        observation = {
            "x": x,
            "y": y,
            "z": z,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "moved": round(moved, 4),
            # The mod records Minecraft degrees (player.getYRot()); the mineflayer bridge
            # and the simulated backend report radians, so everything the model sees is
            # converted to degrees here.
            "yaw": math.degrees(float(snapshot.get("yaw") or 0.0)),
            "pitch": math.degrees(float(snapshot.get("pitch") or 0.0)),
            "on_ground": bool(snapshot.get("on_ground", True)),
            "sprinting": bool(snapshot.get("sprinting", False)),
            "sneaking": bool(snapshot.get("sneaking", False)),
            "selected_slot": int(snapshot.get("selected_slot") or 0),
            "health": float(snapshot.get("health") or 20.0),
            "food": float(snapshot.get("food") or 20.0),
            "dimension": dimension or snapshot.get("dimension") or "overworld",
            "nearby": nearby,
            "activity": "none",
        }
        observation["blocks"] = (
            snapshot.get("blocks") if isinstance(snapshot.get("blocks"), list) else default_blocks()
        )
        return observation

    # -------------------------------------------------------------- inference
    def predict_observation(
        self,
        observation: dict[str, Any],
        *,
        sample: bool = True,
        allowed: Collection[str] | None = None,
    ) -> Prediction:
        features = np.asarray([encode(observation, self.include_blocks)], dtype=np.float32)
        probs, motion = self.model.predict(features)
        probabilities = {name: float(value) for name, value in zip(self.actions, probs[0], strict=False)}
        probabilities = mask_actions(probabilities, allowed)
        if sample and self.temperature > 0.0:
            action = self._sample(probabilities)
        else:
            action = max(probabilities.items(), key=lambda item: item[1])[0]
        deltas = denormalise_targets(motion[0])
        self._predictions += 1
        prediction = Prediction(
            action=action,
            probs=probabilities,
            dyaw=float(deltas[0]),
            dpitch=float(deltas[1]),
            forward=float(deltas[2]),
            strafe=float(deltas[3]),
            source=f"trained:{self.run.run}",
        )
        self._last_prediction = prediction.to_dict()
        return prediction

    def predict(self, snapshot: dict[str, Any], **kwargs: Any) -> Prediction:
        allowed = kwargs.pop("allowed", None)
        return self.predict_observation(self.observe(snapshot, **kwargs), allowed=allowed)

    def _sample(self, probabilities: dict[str, float]) -> str:
        temperature = max(1e-3, self.temperature)
        weights = [math.exp(math.log(max(p, 1e-9)) / temperature) for p in probabilities.values()]
        total = sum(weights)
        roll = self._rng.random() * total
        cumulative = 0.0
        for name, weight in zip(probabilities, weights, strict=False):
            cumulative += weight
            if roll <= cumulative:
                return name
        return max(probabilities.items(), key=lambda item: item[1])[0]

    def decide(self, snapshot: dict[str, Any], *, allowed: Collection[str] | None = None) -> Any:
        """Agent-compatible decision (imported lazily to avoid a circular import).

        ``allowed`` restricts the action space to what the permission gates permit, so a
        model trained on playtime can never attack, mine or move when the settings say no.
        """
        from pymc_bot.agent import Decision

        prediction = self.predict(snapshot, allowed=allowed)
        return Decision(
            action=prediction.action,
            params={
                "dyaw": prediction.dyaw,
                "dpitch": prediction.dpitch,
                "forward": prediction.forward,
                "strafe": prediction.strafe,
                "seconds": self.step_seconds,
                "confidence": round(prediction.probs.get(prediction.action, 0.0), 3),
            },
            source=prediction.source,
            raw=json.dumps(prediction.to_dict()),
        )

    # -------------------------------------------------------------- execution
    def act(self, bot: MinecraftBot, decision: Any) -> tuple[bool, str]:
        """Drive ``bot`` with one predicted action. Returns ``(ok, detail)``."""
        if not bot.connected:
            return False, "bot is not connected"
        action = getattr(decision, "action", str(decision))
        params = getattr(decision, "params", {}) or {}
        seconds = max(0.1, min(3.0, float(params.get("seconds") or self.step_seconds)))
        dyaw = _clamp(float(params.get("dyaw") or 0.0), -self.MAX_YAW_PER_BURST, self.MAX_YAW_PER_BURST)
        dpitch = _clamp(
            float(params.get("dpitch") or 0.0), -self.MAX_PITCH_PER_BURST, self.MAX_PITCH_PER_BURST
        )
        forward = float(params.get("forward") or 0.0)

        if action == "none":
            bot.stop_moving()
            return True, "trained: idle (the recorded player stood still here)"

        if action == "look":
            if abs(dyaw) < 1.0 and abs(dpitch) < 1.0:
                dyaw, dpitch = 10.0, 0.0  # a "look" sample with no learned delta still turns
            snapshot = bot.snapshot()
            yaw = math.degrees(float(snapshot.get("yaw") or 0.0)) + dyaw
            pitch = _clamp(math.degrees(float(snapshot.get("pitch") or 0.0)) + dpitch, -89.0, 89.0)
            ok = bot.look(math.radians(yaw), math.radians(pitch))
            return ok, f"trained: look {dyaw:+.1f}deg yaw {dpitch:+.1f}deg pitch"

        if action in ("attack", "use"):
            ok = bot.swing_arm()
            verb = "attack (swing)" if action == "attack" else "use (swing)"
            return ok, f"trained: {verb}"

        if action == "hold":
            return True, "trained: hold - hotbar switching is not exposed by the bridge yet"

        controls: dict[str, bool] = {}
        if action in ("forward", "move"):
            controls["forward"] = True
            controls["sprint"] = forward > 0.22
        elif action == "back":
            controls["back"] = True
        elif action == "left":
            controls["left"] = True
        elif action == "right":
            controls["right"] = True
        elif action == "jump":
            controls["jump"] = True
        elif action == "sneak":
            controls["sneak"] = True
        else:  # pragma: no cover - unknown label
            return False, f"trained: unknown action {action!r}"

        if dyaw or dpitch:
            snapshot = bot.snapshot()
            yaw = math.degrees(float(snapshot.get("yaw") or 0.0)) + dyaw
            pitch = _clamp(math.degrees(float(snapshot.get("pitch") or 0.0)) + dpitch, -89.0, 89.0)
            bot.look(math.radians(yaw), math.radians(pitch))
        ok = bot.control_burst(controls, seconds)
        pressed = ", ".join(sorted(name for name, state in controls.items() if state))
        return ok, f"trained: {pressed or action} for {seconds:.2f}s"

    # ----------------------------------------------------------------- status
    def describe(self) -> dict[str, Any]:
        card = self.run.card
        return {
            "run": self.run.run,
            "engine": self.run.engine,
            "checkpoint": str(self.run.checkpoint),
            "checkpoint_exists": self.run.checkpoint.is_file(),
            "step": card.get("step"),
            "params": card.get("params"),
            "actions": self.actions,
            "targets": [str(name) for name in (card.get("targets") or TARGET_NAMES)],
            "feature_dim": card.get("feature_dim"),
            "include_blocks": self.include_blocks,
            "trained_on": card.get("dataset"),
            "metrics": card.get("metrics"),
            "predictions": self._predictions,
            "last_prediction": self._last_prediction,
            "temperature": self.temperature,
            "step_seconds": self.step_seconds,
        }


# ---------------------------------------------------------------------------
# optional: distil the trained policy into an Ollama model (prompt export)
# ---------------------------------------------------------------------------
def policy_priors(policy: TrainedPolicy, observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Which action the model picks for a sample of observations (used for the prompt)."""
    priors: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for observation in observations[:200]:
        prediction = policy.predict_observation(observation, sample=False)
        priors[prediction.action] = priors.get(prediction.action, 0) + 1
        if len(examples) < 6:
            examples.append(prediction.to_dict())
    return {
        "priorities": dict(sorted(priors.items(), key=lambda item: -item[1])),
        "examples": examples,
    }


def build_ollama_modelfile(policy: TrainedPolicy, priors: dict[str, Any], base_model: str) -> str:
    """A Modelfile that turns an Ollama model into a persona agent for the trained player."""
    priorities = ", ".join(f"{name} ({count})" for name, count in priors.get("priorities", {}).items())
    lines = [
        f"FROM {base_model}",
        "PARAMETER temperature 0.4",
        "PARAMETER num_predict 96",
        "SYSTEM \"\"\""
        "You are PyMC_Bot, a Minecraft player whose behaviour was learned from a real player's "
        "recorded playtime. Given the current world state (position, look direction, health, food, "
        "nearby players, held item) reply with exactly one low level control action as compact JSON: "
        "{\"action\": \"forward\"} - allowed actions: "
        + ", ".join(ACTION_SPACE)
        + ". "
        + (f"On the recorded data the player's habits ranked: {priorities}. " if priorities else "")
        + "Turn with \"look\" and stay close to the style of the recorded player."
        "\"\"\"",
    ]
    return "\n".join(lines) + "\n"


def export_ollama_model(
    policy: TrainedPolicy,
    *,
    name: str,
    client: Any,
    base_model: str = "llama3.2",
    observations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write a Modelfile next to the checkpoint and create the Ollama model."""
    priors = policy_priors(policy, observations or [])
    modelfile = build_ollama_modelfile(policy, priors, base_model)
    target = policy.run.run_dir / f"Modelfile.{name}"
    target.write_text(modelfile, encoding="utf-8")
    response = client.create_model(name, base_model=base_model, modelfile=modelfile)
    return {"model": name, "modelfile": str(target), "priors": priors, "response": response}
