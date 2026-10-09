"""The player models: a dependency-free NumPy MLP and an optional tiny transformer.

Both implement the same small interface so :mod:`pymc_bot.train` and
:mod:`pymc_bot.local_model` do not care which one was trained::

    probs, reg = model.predict(X)                  # (batch, actions), (batch, 4)
    loss, acc  = model.optim_step(X, y, targets, lr, reg_weight)
    loss, acc  = model.evaluate(X, y, targets, reg_weight)
    model.save(path, extra={...})                  # checkpoints, incl. optimiser state
    model = load_model(path)

* :class:`NumpyMLP` -- two hidden layers, ReLU, softmax action head plus a regression
  head for the continuous motion targets. Trains on CPU in seconds and needs nothing
  beyond NumPy, which is why it is the default engine.
* :class:`TorchTransformer` -- a small from-scratch transformer encoder (the feature
  vector is cut into tokens, each embedded, attention + feed-forward, mean pooled).
  Used when ``--engine transformer`` is selected and PyTorch is installed
  (``pip install -r requirements-train.txt``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

N_TARGETS = 4


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy_grads(
    probs: np.ndarray, labels: np.ndarray, sample_weights: np.ndarray | None = None
) -> np.ndarray:
    """``d(loss)/d(logits)`` for a batch, already divided by the batch size."""
    grad = probs.copy()
    grad[np.arange(len(labels)), labels] -= 1.0
    if sample_weights is not None:
        grad *= sample_weights.reshape(-1, 1)
    return grad / len(labels)


def balanced_class_weights(
    labels: np.ndarray, n_classes: int, power: float = 0.5, cap: float = 4.0
) -> np.ndarray:
    """Inverse-frequency class weights (``n / (k * count)``), normalised to mean 1.

    Playtime is dominated by walking (``forward``), so plain cross-entropy learns to
    always walk. Weighting the rare actions keeps the model usable as a player;
    ``power`` tempers the correction (``1.0`` = full balancing, ``0.0`` = off) because
    full balancing costs a lot of raw accuracy.
    """
    counts = np.bincount(labels.astype(np.int64), minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = np.power(len(labels) / (n_classes * counts), float(power))
    if cap > 0:
        # An action seen once must not take over the whole loss.
        weights = np.clip(weights, None, cap)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def balanced_accuracy(predicted: np.ndarray, labels: np.ndarray) -> float:
    """Mean per-class recall (ignores how common each action is)."""
    present = np.unique(labels)
    if len(present) == 0:
        return 0.0
    recalls = [(predicted[labels == label] == label).mean() for label in present]
    return float(np.mean(recalls))


class Adam:
    """Adam optimiser for plain dict-of-arrays models (resumable)."""

    def __init__(
        self,
        params: dict[str, np.ndarray],
        lr: float = 3e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {key: np.zeros_like(value) for key, value in params.items()}
        self.v = {key: np.zeros_like(value) for key, value in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        bias1 = 1.0 - self.beta1**self.t
        bias2 = 1.0 - self.beta2**self.t
        for key, param in params.items():
            grad = grads[key]
            if self.weight_decay:
                grad = grad + self.weight_decay * param
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (grad * grad)
            param -= self.lr * (self.m[key] / bias1) / (np.sqrt(self.v[key] / bias2) + self.eps)

    def state(self, prefix: str = "adam_") -> dict[str, np.ndarray]:
        state: dict[str, np.ndarray] = {f"{prefix}t": np.array([self.t], dtype=np.int64)}
        for key in self.m:
            state[f"{prefix}m_{key}"] = self.m[key]
            state[f"{prefix}v_{key}"] = self.v[key]
        return state

    def load_state(self, state: dict[str, np.ndarray], prefix: str = "adam_") -> None:
        if f"{prefix}t" not in state:
            return
        self.t = int(np.asarray(state[f"{prefix}t"]).reshape(-1)[0])
        for key in self.m:
            if f"{prefix}m_{key}" in state:
                self.m[key] = np.asarray(state[f"{prefix}m_{key}"], dtype=np.float32)
                self.v[key] = np.asarray(state[f"{prefix}v_{key}"], dtype=np.float32)


# ---------------------------------------------------------------------------
# NumPy MLP (default engine)
# ---------------------------------------------------------------------------
class NumpyMLP:
    """Two hidden layers + action head + motion regression head."""

    kind = "mlp"

    def __init__(
        self,
        feature_dim: int,
        n_actions: int,
        hidden: tuple[int, ...] = (128, 64),
        n_targets: int = N_TARGETS,
        lr: float = 3e-3,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.feature_dim = int(feature_dim)
        self.n_actions = int(n_actions)
        self.hidden = tuple(int(size) for size in hidden)
        self.n_targets = int(n_targets)
        self.params: dict[str, np.ndarray] = {}

        sizes = (self.feature_dim, *self.hidden)
        for index in range(len(sizes) - 1):
            fan_in, fan_out = sizes[index], sizes[index + 1]
            self.params[f"W{index + 1}"] = (
                rng.standard_normal((fan_in, fan_out)).astype(np.float32) * np.sqrt(2.0 / fan_in)
            )
            self.params[f"b{index + 1}"] = np.zeros(fan_out, dtype=np.float32)
        last = self.hidden[-1]
        self.params["Wa"] = (rng.standard_normal((last, self.n_actions)) * 0.01).astype(np.float32)
        self.params["ba"] = np.zeros(self.n_actions, dtype=np.float32)
        self.params["Wr"] = (rng.standard_normal((last, self.n_targets)) * 0.01).astype(np.float32)
        self.params["br"] = np.zeros(self.n_targets, dtype=np.float32)

        self.optimiser = Adam(self.params, lr=lr)

    # -------------------------------------------------------------- inference
    def _activations(self, X: np.ndarray) -> list[np.ndarray]:
        acts = [X]
        hidden_layers = len(self.hidden)
        current = X
        for index in range(hidden_layers):
            current = np.maximum(
                current @ self.params[f"W{index + 1}"] + self.params[f"b{index + 1}"], 0.0
            )
            acts.append(current)
        return acts

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hiddens = self._activations(X)
        last = hiddens[-1]
        logits = last @ self.params["Wa"] + self.params["ba"]
        regression = last @ self.params["Wr"] + self.params["br"]
        return logits, regression

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        logits, regression = self.forward(X)
        return softmax(logits), regression

    # -------------------------------------------------------------- training
    def _loss(
        self,
        probs: np.ndarray,
        regression: np.ndarray,
        labels: np.ndarray,
        targets: np.ndarray,
        reg_weight: float,
        sample_weights: np.ndarray | None = None,
    ) -> float:
        logp = np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-9, 1.0))
        weighted = -logp if sample_weights is None else -logp * sample_weights
        classification = float(weighted.mean())
        if reg_weight <= 0.0:
            return classification
        return classification + 0.5 * reg_weight * float(((regression - targets) ** 2).mean())

    def optim_step(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        targets: np.ndarray,
        lr: float,
        reg_weight: float,
        sample_weights: np.ndarray | None = None,
    ) -> tuple[float, float]:
        X = np.asarray(X, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.float32)
        self.optimiser.lr = lr

        hiddens = self._activations(X)
        last = hiddens[-1]
        logits = last @ self.params["Wa"] + self.params["ba"]
        regression = last @ self.params["Wr"] + self.params["br"]
        probs = softmax(logits)
        loss = self._loss(probs, regression, labels, targets, reg_weight, sample_weights)

        grads: dict[str, np.ndarray] = {}
        dlogits = cross_entropy_grads(probs, labels, sample_weights)
        dreg = (
            (regression - targets) * (reg_weight / len(labels))
            if reg_weight > 0.0
            else np.zeros_like(regression)
        )
        grads["Wa"] = last.T @ dlogits
        grads["ba"] = dlogits.sum(axis=0)
        grads["Wr"] = last.T @ dreg
        grads["br"] = dreg.sum(axis=0)

        delta = dlogits @ self.params["Wa"].T + dreg @ self.params["Wr"].T
        for index in range(len(self.hidden), 0, -1):
            delta = delta * (hiddens[index] > 0.0)
            grads[f"W{index}"] = hiddens[index - 1].T @ delta
            grads[f"b{index}"] = delta.sum(axis=0)
            if index > 1:
                delta = delta @ self.params[f"W{index}"].T

        self.optimiser.step(self.params, grads)
        accuracy = float((probs.argmax(axis=1) == labels).mean())
        return loss, accuracy

    def evaluate(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        targets: np.ndarray,
        reg_weight: float,
        sample_weights: np.ndarray | None = None,
    ) -> tuple[float, float]:
        if len(X) == 0:
            return 0.0, 0.0
        probs, regression = self.predict(X)
        labels = np.asarray(labels, dtype=np.int64)
        loss = self._loss(
            probs, regression, labels, np.asarray(targets, dtype=np.float32), reg_weight, sample_weights
        )
        return loss, float((probs.argmax(axis=1) == labels).mean())

    # -------------------------------------------------------------- persistence
    def state(self) -> dict[str, np.ndarray]:
        state = dict(self.params)
        state.update(self.optimiser.state())
        return state

    def param_count(self) -> int:
        return int(sum(int(np.size(value)) for value in self.params.values()))

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.kind,
            "feature_dim": self.feature_dim,
            "n_actions": self.n_actions,
            "hidden": list(self.hidden),
            "n_targets": self.n_targets,
            **(extra or {}),
        }
        np.savez_compressed(target, **self.state(), **_meta_arrays(payload))
        return target

    @classmethod
    def load(cls, path: str | Path) -> NumpyMLP:
        with np.load(Path(path), allow_pickle=False) as data:
            meta = _read_meta(data)
            model = cls(
                feature_dim=int(meta["feature_dim"]),
                n_actions=int(meta["n_actions"]),
                hidden=tuple(meta.get("hidden", (128, 64))),
                n_targets=int(meta.get("n_targets", N_TARGETS)),
            )
            for key in list(model.params):
                model.params[key] = np.asarray(data[key], dtype=np.float32)
            model.optimiser.load_state({key: data[key] for key in data.files if key.startswith("adam_")})
        return model


# ---------------------------------------------------------------------------
# optional PyTorch transformer
# ---------------------------------------------------------------------------
def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:  # pragma: no cover - torch is optional
        return False
    return True


def _torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is optional
        raise RuntimeError(
            "the transformer engine needs PyTorch: pip install -r requirements-train.txt "
            "(the default --engine mlp needs nothing but NumPy)"
        ) from exc
    return torch


class TorchTransformer:
    """A small transformer encoder over the observation features.

    The feature vector is cut into ``n_tokens`` chunks, each chunk is projected to
    ``d_model`` (plus a learned positional embedding), two encoder layers with multi-head
    self-attention run over the tokens, and the mean pooled result feeds the same two
    heads the MLP has.
    """

    kind = "transformer"

    def __init__(
        self,
        feature_dim: int,
        n_actions: int,
        d_model: int = 64,
        heads: int = 4,
        layers: int = 2,
        token_dim: int = 16,
        n_targets: int = N_TARGETS,
        dropout: float = 0.1,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> None:
        torch = _torch()
        torch.manual_seed(seed)
        self.feature_dim = int(feature_dim)
        self.n_actions = int(n_actions)
        self.n_targets = int(n_targets)
        self.token_dim = int(token_dim)
        self.n_tokens = max(1, (self.feature_dim + self.token_dim - 1) // self.token_dim)
        self.padded_dim = self.n_tokens * self.token_dim
        self.d_model = int(d_model)
        self.heads = int(heads)
        self.layers = int(layers)
        self.dropout = float(dropout)

        nn = torch.nn

        class _Net(nn.Module):
            def __init__(
                self,
                token_dim: int,
                n_tokens: int,
                d_model: int,
                heads: int,
                layers: int,
                dropout: float,
                n_actions: int,
                n_targets: int,
            ) -> None:
                super().__init__()
                self.token_dim = token_dim
                self.n_tokens = n_tokens
                self.embed = nn.Linear(token_dim, d_model)
                self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=heads,
                    dim_feedforward=d_model * 2,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
                self.norm = nn.LayerNorm(d_model)
                self.head_action = nn.Linear(d_model, n_actions)
                self.head_motion = nn.Linear(d_model, n_targets)

            def forward(self, X):  # noqa: ANN001 - torch tensors
                tokens = X.view(X.shape[0], self.n_tokens, self.token_dim)
                hidden = self.embed(tokens) + self.pos
                hidden = self.encoder(hidden)
                hidden = self.norm(hidden).mean(dim=1)
                return self.head_action(hidden), self.head_motion(hidden)

        self.class_weights = None
        self.net = _Net(
            self.token_dim,
            self.n_tokens,
            self.d_model,
            self.heads,
            self.layers,
            self.dropout,
            self.n_actions,
            self.n_targets,
        )
        self.optimiser = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=0.01)
        self._lr = lr

    # -------------------------------------------------------------- inference
    def _prepare(self, X: np.ndarray):
        torch = _torch()
        array = np.asarray(X, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.shape[1] < self.padded_dim:
            array = np.pad(array, ((0, 0), (0, self.padded_dim - array.shape[1])))
        return torch.from_numpy(array)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = _torch()
        self.net.eval()
        with torch.no_grad():
            logits, motion = self.net(self._prepare(X))
            probs = torch.softmax(logits, dim=1)
        return probs.numpy(), motion.numpy()

    # -------------------------------------------------------------- training
    def optim_step(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        targets: np.ndarray,
        lr: float,
        reg_weight: float,
        sample_weights: np.ndarray | None = None,
    ) -> tuple[float, float]:
        torch = _torch()
        self.net.train()
        for group in self.optimiser.param_groups:
            group["lr"] = lr
        inputs = self._prepare(X)
        label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        target_tensor = torch.from_numpy(np.asarray(targets, dtype=np.float32))
        logits, motion = self.net(inputs)
        classification = torch.nn.functional.cross_entropy(
            logits, label_tensor, weight=self._torch_weights()
        )
        loss = classification
        if reg_weight > 0.0:
            loss = loss + 0.5 * reg_weight * torch.nn.functional.mse_loss(motion, target_tensor)
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimiser.step()
        accuracy = float((logits.argmax(dim=1) == label_tensor).float().mean())
        return float(loss.detach()), accuracy

    def evaluate(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        targets: np.ndarray,
        reg_weight: float,
        sample_weights: np.ndarray | None = None,
    ) -> tuple[float, float]:
        torch = _torch()
        if len(X) == 0:
            return 0.0, 0.0
        self.net.eval()
        with torch.no_grad():
            inputs = self._prepare(X)
            label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64))
            target_tensor = torch.from_numpy(np.asarray(targets, dtype=np.float32))
            logits, motion = self.net(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits, label_tensor, weight=self._torch_weights()
            )
            if reg_weight > 0.0:
                loss = loss + 0.5 * reg_weight * torch.nn.functional.mse_loss(motion, target_tensor)
            accuracy = float((logits.argmax(dim=1) == label_tensor).float().mean())
        return float(loss), accuracy

    def _torch_weights(self):
        """The balanced class weights as a torch tensor (lazily built)."""
        if self.class_weights is None:
            return None
        torch = _torch()
        return torch.from_numpy(np.asarray(self.class_weights, dtype=np.float32))

    def param_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.net.parameters()))

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
        torch = _torch()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": self.kind,
                "meta": {
                    "feature_dim": self.feature_dim,
                    "n_actions": self.n_actions,
                    "n_targets": self.n_targets,
                    "d_model": self.d_model,
                    "heads": self.heads,
                    "layers": self.layers,
                    "token_dim": self.token_dim,
                    "dropout": self.dropout,
                    **(extra or {}),
                },
                "state": self.net.state_dict(),
                "optimiser": self.optimiser.state_dict(),
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> TorchTransformer:
        torch = _torch()
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        meta = payload.get("meta", {})
        model = cls(
            feature_dim=int(meta["feature_dim"]),
            n_actions=int(meta["n_actions"]),
            d_model=int(meta.get("d_model", 64)),
            heads=int(meta.get("heads", 4)),
            layers=int(meta.get("layers", 2)),
            token_dim=int(meta.get("token_dim", 16)),
            n_targets=int(meta.get("n_targets", N_TARGETS)),
            dropout=float(meta.get("dropout", 0.1)),
        )
        model.net.load_state_dict(payload["state"])
        if payload.get("optimiser"):
            model.optimiser.load_state_dict(payload["optimiser"])
        return model

    def meta(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "n_actions": self.n_actions,
            "n_targets": self.n_targets,
            "d_model": self.d_model,
            "heads": self.heads,
            "layers": self.layers,
            "token_dim": self.token_dim,
            "dropout": self.dropout,
        }


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def build_model(
    engine: str,
    feature_dim: int,
    n_actions: int,
    *,
    hidden: tuple[int, ...] = (128, 64),
    d_model: int = 64,
    heads: int = 4,
    layers: int = 2,
    lr: float = 3e-3,
    seed: int = 0,
):
    """Create a fresh model for ``engine`` (``"mlp"`` or ``"transformer"``)."""
    if engine in ("mlp", "numpy", "linear"):
        return NumpyMLP(feature_dim, n_actions, hidden=hidden, lr=lr, seed=seed)
    if engine in ("transformer", "torch"):
        return TorchTransformer(
            feature_dim, n_actions, d_model=d_model, heads=heads, layers=layers, lr=lr, seed=seed
        )
    raise ValueError(f"unknown engine {engine!r} (use 'mlp' or 'transformer')")


def load_model(path: str | Path, kind: str | None = None):
    """Load a checkpoint by path, dispatching on the file suffix or ``kind``."""
    target = Path(path)
    if kind == "transformer" or target.suffix == ".pt":
        return TorchTransformer.load(target)
    return NumpyMLP.load(target)


def checkpoint_filename(engine: str) -> str:
    return "latest.pt" if engine in ("transformer", "torch") else "latest.npz"


# ---------------------------------------------------------------------------
# npz metadata helpers
# ---------------------------------------------------------------------------
def _meta_arrays(extra: dict[str, Any] | None) -> dict[str, np.ndarray]:
    if not extra:
        return {}
    return {"__meta__": np.frombuffer(json.dumps(extra, default=str).encode("utf-8"), dtype=np.uint8)}


def _read_meta(data: Any) -> dict[str, Any]:
    if "__meta__" not in data.files:
        return {}
    raw = bytes(np.asarray(data["__meta__"], dtype=np.uint8).tolist())
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):  # pragma: no cover - corrupt file
        return {}
