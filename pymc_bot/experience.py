"""Small, local, per-bot/server experience memory; never changes model weights."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from pymc_bot.mind import mentions

MAX_USERS = 8
MESSAGES_PER_USER = 3
MAX_OUTCOMES = 8
MAX_AGE = 7 * 24 * 60 * 60
MAX_FILE_BYTES = 32_768
# Avoid retaining obvious secrets or commands. This is not a complete secret detector.
SENSITIVE = re.compile(r"\b(password|passwd|token|api[ _-]?key|secret|credential|bearer)\b", re.I)
NAME = re.compile(r"^[A-Za-z0-9_]{1,16}$")


def _text(value: Any, limit: int = 120) -> str:
    return " ".join(str(value or "").replace("{", "(").replace("}", ")").split())[:limit]


class Experience:
    """Bounded episodic context. Entries older than seven days are excluded on access."""

    def __init__(self, directory: Path, identity: str) -> None:
        key = hashlib.sha256(identity.lower().encode()).hexdigest()[:24]
        self.path = directory / "bot-memory" / f"{key}.json"
        self.lock = threading.RLock()
        self.data: dict[str, Any] = {"users": {}, "outcomes": [], "last_chat": 0.0}
        self._load()

    def _load(self) -> None:
        try:
            if self.path.stat().st_size > MAX_FILE_BYTES:
                return
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
            users = raw.get("users", {})
            outcomes = raw.get("outcomes", [])
            if isinstance(users, dict) and isinstance(outcomes, list):
                self.data = {"users": users, "outcomes": outcomes, "last_chat": raw.get("last_chat", 0.0)}
                self._prune()
        except (OSError, ValueError, TypeError, OverflowError):
            self.data = {"users": {}, "outcomes": [], "last_chat": 0.0}

    def _prune(self) -> None:
        cutoff = time.time() - MAX_AGE
        users = {}
        for name, messages in list(self.data["users"].items())[-MAX_USERS:]:
            if not NAME.fullmatch(name) or not isinstance(messages, list):
                continue
            kept = []
            for row in messages[-MESSAGES_PER_USER:]:
                if isinstance(row, dict) and isinstance(row.get("ts"), (int, float)) and cutoff <= row["ts"] <= time.time() + 60:
                    text = _text(row.get("text"))
                    if text and not SENSITIVE.search(text) and not text.startswith("/"):
                        kept.append({"ts": row["ts"], "text": text})
            if kept:
                users[name] = kept
        outcomes = []
        for row in self.data["outcomes"][-MAX_OUTCOMES:]:
            if isinstance(row, dict) and isinstance(row.get("ts"), (int, float)) and cutoff <= row["ts"] <= time.time() + 60:
                action = _text(row.get("action"), 40)
                detail = _text(row.get("detail"), 100)
                if action and not SENSITIVE.search(detail):
                    thought = _text(row.get("thought"), 100)
                    if SENSITIVE.search(thought):
                        thought = ""
                    outcomes.append({"ts": row["ts"], "action": action, "ok": bool(row.get("ok")),
                                     "detail": detail, "thought": thought})
        last_chat = self.data.get("last_chat", 0.0)
        if not isinstance(last_chat, (int, float)) or not cutoff <= last_chat <= time.time() + 60:
            last_chat = 0.0
        self.data = {"users": users, "outcomes": outcomes, "last_chat": last_chat}

    def _save(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".memory-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def observe(self, history: list[dict[str, Any]], bot_name: str) -> None:
        with self.lock:
            self._prune()
            changed = False
            last_chat = self.data["last_chat"]
            newest = last_chat
            for row in history[-32:]:
                name, message = str(row.get("username") or ""), _text(row.get("message"))
                ts = row.get("time")
                if (not NAME.fullmatch(name) or name.lower() == bot_name.lower() or not isinstance(ts, (int, float))
                        or not time.time() - MAX_AGE <= ts <= time.time() + 60 or ts <= last_chat
                        or not mentions(message, bot_name)
                        or message.startswith("/") or SENSITIVE.search(message)):
                    continue
                users = self.data["users"]
                existing_name = next((key for key in users if key.lower() == name.lower()), name)
                messages = users.get(existing_name, [])
                item = {"ts": ts, "text": message}
                if item in messages:
                    continue
                messages.append(item)
                users.pop(existing_name, None)
                users[existing_name] = messages[-MESSAGES_PER_USER:]
                newest = max(newest, ts)
                changed = True
            if changed:
                self.data["last_chat"] = newest
                self._prune()
                self._save()

    def outcome(self, action: str, ok: bool, detail: str, thought: str = "") -> None:
        # Do not persist chat or its contents in action results.
        if action in {"say", "chat"}:
            return
        action, detail = _text(action, 40), _text(detail, 100)
        if not action or SENSITIVE.search(detail):
            return
        thought = _text(thought, 100)
        if SENSITIVE.search(thought):
            thought = ""
        with self.lock:
            self._prune()
            self.data["outcomes"].append({"ts": time.time(), "action": action, "ok": bool(ok),
                                          "detail": detail, "thought": thought})
            self._prune()
            self._save()

    def context(self, names: list[str]) -> tuple[list[str], list[str]]:
        with self.lock:
            self._prune()
            keys = {name.lower() for name in names}
            users = [f"<{name}> {row['text']}" for name, rows in self.data["users"].items()
                     if name.lower() in keys for row in rows][-6:]
            outcomes = [f"{row['action']}: {'worked' if row['ok'] else 'failed'} ({row['detail']})"
                        + (f"; thought: {row['thought']}" if row["thought"] else "")
                        for row in self.data["outcomes"][-4:]]
            return users, outcomes

    def clear(self) -> None:
        with self.lock:
            self.data = {"users": {}, "outcomes": [], "last_chat": time.time()}
            self.path.unlink(missing_ok=True)
