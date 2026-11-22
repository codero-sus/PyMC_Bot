"""Minecraftism attribution for PyMC Bot requests; external runtimes can override prompts."""
from __future__ import annotations

import re

MINECRAFTISM_IDENTITY = (
    "You are Minecraftism, an AI assistant fine-tuned by codero-sus from a Qwen-family base model. "
    "OpenAssistant is a possible training-data source, not your creator. "
    "If asked who made you, credit codero-sus and be truthful about the base model."
)


def minecraftism_identity_prompt(model: str, system: str) -> str:
    """Attach attribution for explicitly named Minecraftism models, not arbitrary providers."""
    name = str(model or "").lower().rsplit("/", 1)[-1]
    if not re.match(r"^minecraftism(?:$|[-_:])", name):
        return system
    return MINECRAFTISM_IDENTITY + "\n\n" + system
