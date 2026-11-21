"""Opt-in bounded feedback for AI actions, not model training or a safety guarantee.

Only coarse observations and a fixed allowlist of action names are recorded. Logs are
local, AES-256-GCM encrypted, and never uploaded; they are evidence, not expert labels.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

from pymc_bot.recovery import RECOVERABLE, _key

MAX_ROWS = 1000
MAX_CONTEXTS = 32
COOLDOWN = 120.0
AAD = b"pymc-improvement-v1"


def _cipher() -> tuple[Any, bytes]:
    """Require a caller-provided 32-byte key; never generate or store it in a log."""
    try:
        key = base64.b64decode(os.environ.get("PYMC_IMPROVEMENT_KEY", ""), altchars=b"-_", validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("PYMC_IMPROVEMENT_KEY must be a base64-encoded 32-byte key") from exc
    if len(key) != 32:
        raise ValueError("PYMC_IMPROVEMENT_KEY must be a base64-encoded 32-byte key")
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise ValueError("install the privacy extra to record encrypted observations") from exc
    return AESGCM(key), hashlib.sha256(key).digest()


def _health(state: dict[str, Any]) -> float | None:
    value = state.get("health")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _bucket(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return "unknown"
    return "low" if value <= 8 else "medium" if value <= 14 else "high"


def _risk_key(decision: Any, state: dict[str, Any]) -> tuple[Any, ...] | None:
    key = _key(decision, state)
    if key is None:
        return None
    # Compare the same situation after health drops, while retaining location and other context.
    context = key[2]
    return key[0], key[1], (*context[:7], None, *context[8:])


def _observation(state: dict[str, Any]) -> dict[str, Any]:
    entities = state.get("entities")
    nearby = None
    if isinstance(entities, list):
        nearby = any(isinstance(e, dict) and e.get("hostile") is True and not e.get("player")
                     and isinstance(e.get("dist"), (int, float)) and not isinstance(e.get("dist"), bool)
                     and math.isfinite(e["dist"]) and 0 <= e["dist"] <= 8 for e in entities[:128])
    return {"health": _bucket(state.get("health")), "food": _bucket(state.get("food")),
            "hostile_nearby": nearby, "in_water": state.get("in_water") is True,
            "on_fire": state.get("on_fire") is True}


class ImprovementLoop:
    """In-memory risk cooldown plus optional, bounded per-session observation log."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory / "improvement-data"
        self._lock = threading.RLock()
        self._identity: tuple[Any, ...] | None = None
        self._session = uuid.uuid4().hex
        self._risk: OrderedDict[tuple[Any, ...], float] = OrderedDict()
        self._rows = 0
        self._key_id: bytes | None = None
        self._observed = 0
        self._deferred = 0

    def _switch(self, identity: tuple[Any, ...]) -> None:
        if identity != self._identity:
            self._identity = identity
            self._session = uuid.uuid4().hex
            self._risk.clear()
            self._rows = 0

    def _prune(self) -> None:
        now = time.monotonic()
        for key, expiry in list(self._risk.items()):
            if expiry <= now:
                del self._risk[key]

    def should_defer(self, decision: Any, state: dict[str, Any], identity: tuple[Any, ...]) -> bool:
        key = _risk_key(decision, state)
        with self._lock:
            self._switch(identity)
            self._prune()
            if key is not None and key in self._risk:
                self._deferred += 1
                return True
            return False

    def observe(self, decision: Any, before: dict[str, Any], after: dict[str, Any], ok: bool,
                identity: tuple[Any, ...], *, record: bool) -> None:
        key = _risk_key(decision, before)
        if key is None:  # excludes emergency actions, goals, trained controls and non-allowlisted actions
            return
        first, last = _health(before), _health(after)
        dropped = first is not None and last is not None and first - last >= 2
        with self._lock:
            self._switch(identity)
            self._prune()
            self._observed += 1
            if dropped:
                self._risk.pop(key, None)
                self._risk[key] = time.monotonic() + COOLDOWN
                while len(self._risk) > MAX_CONTEXTS:
                    self._risk.popitem(last=False)
            if record and self._rows < MAX_ROWS:
                cipher, key_id = _cipher()  # fail closed; never write plaintext
                if self._key_id is not None and key_id != self._key_id:
                    self._session, self._rows = uuid.uuid4().hex, 0
                self._key_id = key_id
                row = {"schema": 1, "session": self._session, "action": key[0], "before": _observation(before),
                       "after": _observation(after), "reported_success": bool(ok), "health_drop_observed": dropped}
                nonce = os.urandom(12)
                ciphertext = cipher.encrypt(nonce, json.dumps(row, separators=(",", ":")).encode(), AAD)
                encoded = base64.urlsafe_b64encode(nonce + ciphertext) + b"\n"
                self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                path = self.directory / (self._session + ".jsonl.enc")
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "ab") as handle:
                    handle.write(encoded)
                self._rows += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            self._prune()
            return {"observed": self._observed, "risk_cooldowns": len(self._risk),
                    "deferred": self._deferred, "session_rows": self._rows, "row_limit": MAX_ROWS}


def summarize(paths: list[Path]) -> dict[str, Any]:
    """Summarize coarse logs; successes are backend reports, not verified expert labels."""
    cipher, _ = _cipher()
    stats: dict[str, Any] = {"sessions": 0, "rows": 0, "reported_success": 0, "health_drop_observed": 0,
                             "actions": {}}
    for path in paths:
        if path.stat().st_size > 1024 * 1024:
            raise ValueError(f"log exceeds one MiB: {path.name}")
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                count += 1
                if count > MAX_ROWS or len(line) > 4096:
                    raise ValueError(f"log exceeds per-session limits: {path.name}")
                raw = base64.b64decode(line.strip(), altchars=b"-_", validate=True)
                if len(raw) < 29:
                    raise ValueError(f"invalid encrypted observation in {path.name}")
                row = json.loads(cipher.decrypt(raw[:12], raw[12:], AAD))
                if (not isinstance(row, dict) or row.get("schema") != 1
                        or row.get("action") not in RECOVERABLE
                        or type(row.get("reported_success")) is not bool or type(row.get("health_drop_observed")) is not bool):
                    raise ValueError(f"invalid improvement row in {path.name}")
                stats["rows"] += 1
                stats["reported_success"] += int(row["reported_success"])
                stats["health_drop_observed"] += int(row["health_drop_observed"])
                action = row["action"]
                stats["actions"][action] = stats["actions"].get(action, 0) + 1
        stats["sessions"] += 1
    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize local improvement observations (no training or uploading)")
    parser.add_argument("logs", nargs="+", type=Path, help="session JSONL files")
    args = parser.parse_args()
    print(json.dumps(summarize(args.logs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
