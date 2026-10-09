"""PyMC_Bot -- a Python-controlled Minecraft player bot.

The bot connects to a Minecraft server in *cracked* (offline-mode) or
Microsoft-account mode, exposes a small REST/WebSocket config page served by
Uvicorn, and can optionally hand the wheel to a local Ollama model so the bot
decides for itself how to move around, chat, mine and follow players.

Layout
------
``pymc_bot.config``    pydantic settings + JSON persistence
``pymc_bot.events``    thread-safe event log with async subscribers
``pymc_bot.backends``  simulated backend + Node/mineflayer protocol adapter
``pymc_bot.bot``       high level player controller (move, goto, follow, mine...)
``pymc_bot.ollama``    tiny Ollama HTTP client
``pymc_bot.agent``     the LLM decision loop + heuristic fallback policy
``pymc_bot.server``    FastAPI app / Uvicorn entry point
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
