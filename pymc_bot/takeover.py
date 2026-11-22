"""Small, bounded decision contract for an explicitly controlled local player.

The mod, not this endpoint, owns the session and applies the controls. This module
never runs arbitrary code or operates the separate Mineflayer bot.
"""
from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pymc_bot.config import AppSettings
from pymc_bot.cortex import client_from_settings
from pymc_bot.fleet import primary_brain
from pymc_bot.model_identity import minecraftism_identity_prompt
from pymc_bot.ollama import OllamaClient


class TakeoverObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    health: float = Field(ge=0, le=2000, allow_inf_nan=False, strict=True)
    food: int = Field(ge=0, le=20, strict=True)
    yaw: float = Field(ge=-1000000, le=1000000, allow_inf_nan=False, strict=True)
    on_ground: bool = Field(strict=True)
    safe_ahead: bool = Field(strict=True)
    # Optional for older clients: no side visibility means no autonomous turn choice.
    safe_left: bool = Field(default=False, strict=True)
    safe_right: bool = Field(default=False, strict=True)


def mode(settings: AppSettings) -> str:
    return primary_brain(settings)


def available(settings: AppSettings) -> tuple[bool, str]:
    brain = mode(settings)
    if not settings.agent.allow_movement:
        return False, "Enable AI movement in the panel first."
    if brain not in ("heuristic", "ollama", "cortex"):
        return False, "Takeover supports heuristic, Ollama and Cortex brains; choose one in the panel."
    return True, brain


def sanitize(raw: Any, obs: TakeoverObservation) -> dict[str, Any]:
    """Never trust a model or client to bypass the small action allowlist."""
    if not isinstance(raw, dict) or set(raw) - {"move", "turn"}:
        raw = {}
    move = raw.get("move")
    if move not in ("forward", "stop"):
        move = "stop"
    turn = raw.get("turn", 0)
    if isinstance(turn, bool) or not isinstance(turn, (float, int)) or not math.isfinite(turn):
        turn = 0
    turn = max(-10.0, min(10.0, float(turn)))
    if obs.health <= 8 or obs.food <= 6 or not obs.on_ground:
        return {"move": "stop", "turn": 0.0}
    if not obs.safe_ahead:
        move = "stop"
    return {"move": move, "turn": turn}


def decide(settings: AppSettings, obs: TakeoverObservation) -> dict[str, Any]:
    ok, brain = available(settings)
    if not ok:
        raise ValueError(brain)
    if obs.health <= 8 or obs.food <= 6 or not obs.on_ground:
        return {"move": "stop", "turn": 0.0, "brain": brain}
    if brain == "heuristic":
        # Stop at obstacles; turn toward a sampled clear side rather than blindly spinning.
        # Alternate the preferred side by orientation when both are clear, without session state.
        turn = 0
        if not obs.safe_ahead:
            if obs.safe_left and obs.safe_right:
                turn = -10 if int((obs.yaw % 360) // 90) % 2 == 0 else 10
            elif obs.safe_left:
                turn = -10
            elif obs.safe_right:
                turn = 10
        result = {"move": "forward" if obs.safe_ahead else "stop", "turn": turn}
    else:
        if brain == "ollama":
            cfg = settings.ollama
            client = OllamaClient(base_url=cfg.base_url, model=cfg.model, temperature=cfg.temperature, timeout=10)
        else:
            client = client_from_settings(settings.cortex, timeout=10)
        try:
            text = client.chat(
                [{"role": "user", "content": json.dumps(obs.model_dump(), separators=(",", ":"))}],
                system=minecraftism_identity_prompt(
                    client.model,
                    'Control a Minecraft player conservatively. Reply ONLY as JSON: '
                    '{"move":"forward" or "stop","turn":number of degrees between -10 and 10}. '
                    'There is no attack, jump, inventory, chat or block interaction. '
                    'Stop if uncertain; only go forward when safe_ahead is true. '
                    'safe_left and safe_right sample the floor and clearance 45 degrees to either side; '
                    'they do not detect mobs or hidden traps. Negative turn is left, positive is right.'),
                timeout=10,
            )
        finally:
            client.close()
        try:
            result = json.loads(text)
        except (ValueError, TypeError):
            result = {}
    return {**sanitize(result, obs), "brain": brain}
