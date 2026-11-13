"""Optional, conservative model routing for Ollama decisions (not a survival reflex)."""
from __future__ import annotations

import math
import re
from typing import Any

_COMPLEX = re.compile(r"\b(craft|crafting|build|building|construct|recipe|recipes|smelt|nether|portal|plan|"
                      r"strategy|farm|farming|enchant|brew|redstone)\b", re.I)


def select_ollama_model(config: Any, vision_enabled: bool, state: dict[str, Any], goal: str,
                        recent_chat: list[str]) -> tuple[str, str]:
    """Select an installed model name without loading a model or executing an action.

    Heuristics are deliberately simple: they are not a safety system or a claim that the
    selected model is faster/better. VL stays on the explicitly configured model.
    """
    default = config.model
    if not config.adaptive_routing:
        return default, "disabled"
    if vision_enabled:
        return default, "vision uses configured model"
    routine, complex_model = config.routine_model.strip(), config.complex_model.strip()
    if not routine or not complex_model:
        return default, "routing models not configured"
    try:
        health = float(state.get("health"))
    except (TypeError, ValueError):
        health = math.nan
    danger = not math.isfinite(health) or health <= 10 or bool(state.get("on_fire")) or bool(state.get("in_water"))
    entities = state.get("entities") if isinstance(state.get("entities"), list) else []
    hostiles = 0
    for entity in entities:
        if not isinstance(entity, dict) or not entity.get("hostile"):
            continue
        try:
            distance = float(entity.get("dist"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(distance) and 0 <= distance <= 12:
            hostiles += 1
    if danger or hostiles >= 2:
        return complex_model, "danger or multiple hostiles"
    text = " ".join([str(goal or "")[:500], *(str(line)[:500] for line in recent_chat[-3:])])
    if _COMPLEX.search(text):
        return complex_model, "planning request"
    return routine, "routine"
