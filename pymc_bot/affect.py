"""Small situational affect signals, not feelings or inferences about a player's character."""
from __future__ import annotations

import math
from typing import Any


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def affect(state: dict[str, Any]) -> dict[str, str]:
    """Derive a bounded, ephemeral signal only from current, observable conditions."""
    if state.get("status") != "connected":
        return {"state": "uncertain", "reason": "not connected"}
    health, food = _number(state.get("health")), _number(state.get("food"))
    if state.get("on_fire") or state.get("in_lava") or (health is not None and health <= 8):
        return {"state": "alarmed", "reason": "immediate survival risk"}
    entities = state.get("entities") if isinstance(state.get("entities"), list) else []
    for entity in entities[:24]:
        if not isinstance(entity, dict) or not entity.get("hostile") or entity.get("player"):
            continue
        distance = _number(entity.get("dist"))
        if distance is not None and 0 <= distance <= 8:
            return {"state": "alert", "reason": "hostile creature nearby"}
    if state.get("nearby_hazards"):
        return {"state": "cautious", "reason": "suspicious or dangerous blocks nearby"}
    if health is None or food is None:
        return {"state": "uncertain", "reason": "vitals not observed"}
    if food <= 6:
        return {"state": "strained", "reason": "low food"}
    players = state.get("players")
    if isinstance(players, list) and players:
        return {"state": "social", "reason": "another player is visible"}
    return {"state": "focused", "reason": "no immediate threat observed"}
