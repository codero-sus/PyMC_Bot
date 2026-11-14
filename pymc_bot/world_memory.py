"""Bounded, opt-in local spatial observations, stored separately for each bot/server.

Chest inventories are historical observations, not authoritative live state. This module
never navigates or opens a container, and does not record chat or model prompts.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

MAX_CHESTS = 64
MAX_ITEMS = 128
MAX_AGE_SECONDS = 30 * 86400
_NAME = re.compile(r"^[a-z0-9_.:-]{1,100}$")
_DIMENSION = re.compile(r"^[a-z0-9_.:/-]{1,100}$")


def _identity_key(identity: tuple[str, int, str]) -> str:
    host, port, username = identity
    # Structured encoding avoids ambiguity in host/user delimiters.
    raw = json.dumps([host.lower(), port, username.lower()], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _chest_data(result: Any) -> tuple[str, int, int, int, list[dict[str, Any]]] | None:
    if not isinstance(result, dict):
        return None
    dimension = str(result.get("dimension") or "").lower().removeprefix("minecraft:")
    position, items = result.get("position"), result.get("items")
    if not _DIMENSION.fullmatch(dimension) or not isinstance(position, dict) or not isinstance(items, list):
        return None
    coords = []
    for key, limit in (("x", 30_000_000), ("y", 4096), ("z", 30_000_000)):
        value = position.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        if int(value) != value or abs(value) > limit:
            return None
        coords.append(int(value))
    totals: dict[str, int] = {}
    if len(items) > 512:
        return None
    for item in items:
        if not isinstance(item, dict):
            return None
        name, count = str(item.get("name") or "").lower().removeprefix("minecraft:"), item.get("count")
        if not _NAME.fullmatch(name) or isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 1_000_000:
            return None
        totals[name] = min(1_000_000, totals.get(name, 0) + count)
    if len(totals) > MAX_ITEMS:
        return None
    entries = [{"name": name, "count": count} for name, count in sorted(totals.items())]
    return dimension, *coords, entries


class WorldMemory:
    """SQLite-backed chest snapshots. The identity is fixed when this instance is created."""

    def __init__(self, directory: Path, identity: tuple[str, int, str]) -> None:
        self.path = Path(directory) / "world-memory" / f"{_identity_key(identity)}.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=5.0)
        db.execute("PRAGMA busy_timeout = 5000")
        db.execute("PRAGMA secure_delete = ON")
        db.execute("CREATE TABLE IF NOT EXISTS chests (dimension TEXT NOT NULL, x INTEGER NOT NULL, "
                   "y INTEGER NOT NULL, z INTEGER NOT NULL, observed REAL NOT NULL, items TEXT NOT NULL, "
                   "PRIMARY KEY(dimension, x, y, z))")
        return db

    def remember_chest(self, result: Any) -> bool:
        parsed = _chest_data(result)
        if parsed is None:
            return False
        dimension, x, y, z, items = parsed
        now = time.time()
        with closing(self._connect()) as db, db:
            db.execute("INSERT OR REPLACE INTO chests VALUES (?, ?, ?, ?, ?, ?)",
                       (dimension, x, y, z, now, json.dumps(items, separators=(",", ":"))))
            db.execute("DELETE FROM chests WHERE observed < ?", (now - MAX_AGE_SECONDS,))
            db.execute("DELETE FROM chests WHERE rowid IN (SELECT rowid FROM chests ORDER BY observed DESC "
                       "LIMIT -1 OFFSET ?)", (MAX_CHESTS,))
        return True

    def chests(self, dimension: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        limit = max(1, min(64, int(limit)))
        cutoff = time.time() - MAX_AGE_SECONDS
        with closing(self._connect()) as db, db:
            if dimension:
                rows = db.execute("SELECT dimension,x,y,z,observed,items FROM chests WHERE observed >= ? "
                                  "AND dimension = ? ORDER BY observed DESC LIMIT ?",
                                  (cutoff, str(dimension).lower().removeprefix("minecraft:"), limit)).fetchall()
            else:
                rows = db.execute("SELECT dimension,x,y,z,observed,items FROM chests WHERE observed >= ? "
                                  "ORDER BY observed DESC LIMIT ?", (cutoff, limit)).fetchall()
        result = []
        for dim, x, y, z, observed, encoded in rows:
            try:
                items = json.loads(encoded)
                clean = _chest_data({"dimension": dim, "position": {"x": x, "y": y, "z": z}, "items": items})
                if clean is None or not math.isfinite(observed):
                    continue
            except (TypeError, ValueError, OverflowError):
                continue
            result.append({"dimension": dim, "position": {"x": x, "y": y, "z": z},
                           "observed_at": observed, "items": clean[4]})
        return result

    def clear(self) -> None:
        if not self.path.exists():
            return
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM chests")

    def prompt(self, dimension: str | None) -> str:
        if not dimension:
            return ""
        rows = self.chests(dimension=dimension, limit=4)
        if not rows:
            return ""
        lines = ["REMEMBERED CONTAINERS (old observations; contents may have changed, check before relying on them):"]
        for row in rows:
            pos = row["position"]
            items = ", ".join(f"{i['name']} x{i['count']}" for i in row["items"][:8]) or "empty"
            lines.append(f"- {row['dimension']} {pos['x']} {pos['y']} {pos['z']}: {items}")
        return "\n".join(lines)
