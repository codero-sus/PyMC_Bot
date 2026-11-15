"""Short-lived, context-sensitive recovery from repeated failed AI actions.

This is not model training: it keeps only a bounded in-memory cache and never blocks
manual actions or survival reflexes. Changing the situation or waiting lets the AI retry.
"""
from __future__ import annotations

import json
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

# Limit adaptation to repeatable, non-emergency jobs. Never suppress combat, eating,
# fleeing, respawning, shield/totem use, user chat, trained low-level controls, or stop.
RECOVERABLE = frozenset({"mine", "dig", "goto", "walk", "move", "follow", "follow_player", "collect", "pickup",
                         "pick_up", "sleep", "wear_armor", "armor", "organize", "sort_hotbar", "chop_tree",
                         "mine_ore", "gather", "craft", "smelt", "deposit", "withdraw", "go_home", "goto_waypoint"})
PARAMETERS = frozenset({"block", "target", "item", "ore", "player", "x", "y", "z", "name", "count", "slot"})


def _cell(value: Any) -> int | None:
    try:
        number = float(value)
        return math.floor(number / 8) if math.isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _context(state: dict[str, Any]) -> tuple[Any, ...]:
    pos = state.get("position") if isinstance(state.get("position"), dict) else {}
    server = state.get("server") if isinstance(state.get("server"), dict) else {}
    inventory = state.get("inventory") if isinstance(state.get("inventory"), list) else []
    items = sorted((str(row.get("name", ""))[:80], str(row.get("count", ""))[:12])
                   for row in inventory[:128] if isinstance(row, dict))
    entities = state.get("entities") if isinstance(state.get("entities"), list) else []
    nearby = sorted((str(row.get("type", ""))[:80], bool(row.get("hostile")), str(_cell(row.get("dist"))))
                    for row in entities[:24] if isinstance(row, dict))
    players = state.get("players") if isinstance(state.get("players"), list) else []
    visible_players = sorted(str(row.get("name", ""))[:32] for row in players[:24] if isinstance(row, dict))
    return (str(server.get("host", "")), str(server.get("port", "")), str(state.get("dimension", "")),
            _cell(pos.get("x")), _cell(pos.get("y")), _cell(pos.get("z")), str(state.get("time_of_day", "")),
            _cell(state.get("health")), _cell(state.get("food")), str(state.get("held_item", "")),
            tuple(items), tuple(nearby), tuple(visible_players))


def _key(decision: Any, state: dict[str, Any]) -> tuple[Any, ...] | None:
    action = str(getattr(decision, "action", "") or "").lower()
    if action not in RECOVERABLE or str(getattr(decision, "source", "")).startswith(("trained", "goal:")):
        return None
    params = getattr(decision, "params", None) or {}
    if not isinstance(params, dict):
        return None
    selected = {k: v for k, v in params.items() if k in PARAMETERS and isinstance(v, (str, int, float, bool, type(None)))}
    try:
        encoded = json.dumps(selected, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError):
        return None
    if len(encoded) > 512:
        return None
    return action, encoded, _context(state)


class RecoveryMemory:
    """At most 32 action/context keys, each expiring after 45 seconds."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._entries: OrderedDict[tuple[Any, ...], tuple[int, float]] = OrderedDict()
        self._lock = threading.Lock()
        self.skipped = 0

    def _prune(self, now: float) -> None:
        for key, (_, expires) in list(self._entries.items()):
            if expires <= now:
                del self._entries[key]

    def record(self, decision: Any, state: dict[str, Any], ok: bool, detail: str = "") -> None:
        key = _key(decision, state)
        if key is None:
            return
        now = self._clock()
        with self._lock:
            self._prune(now)
            if ok:
                self._entries.pop(key, None)
                return
            # Interruptions and connection loss do not teach us that the task is impossible.
            if any(word in str(detail).lower() for word in ("cancel", "interrupt", "not connected", "disconnected", "busy")):
                return
            count, _ = self._entries.pop(key, (0, 0.0))
            self._entries[key] = (min(2, count + 1), now + 45.0)
            while len(self._entries) > 32:
                self._entries.popitem(last=False)

    def should_defer(self, decision: Any, state: dict[str, Any]) -> bool:
        key = _key(decision, state)
        if key is None:
            return False
        with self._lock:
            self._prune(self._clock())
            return self._entries.get(key, (0, 0.0))[0] >= 2

    def note_skip(self) -> None:
        with self._lock:
            self.skipped += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            self._prune(self._clock())
            return {"deferred_contexts": sum(count >= 2 for count, _ in self._entries.values()), "skipped": self.skipped}
