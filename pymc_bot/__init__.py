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

# --------------------------------------------------------------------- credits
# One source of truth for authorship: the panel footer and About dialog, the
# ``/api/about`` endpoint, ``python -m pymc_bot about``/``--version``, the
# OpenAPI docs and the mod metadata all point at these values.
__author__ = "codero-sus"
__author_url__ = "https://github.com/codero-sus"
__website__ = "https://the-codero.web.app/"
__repository__ = "https://github.com/codero-sus/PyMC_Bot"
__license__ = "PolyForm-Noncommercial-1.0.0"
__license_url__ = "https://polyformproject.org/licenses/noncommercial/1.0.0"
MADE_WITH = "Made with \u2764\ufe0f by codero-sus"

__all__ = [
    "MADE_WITH",
    "__author__",
    "__author_url__",
    "__license__",
    "__license_url__",
    "__repository__",
    "__version__",
    "__website__",
    "about",
    "credit_line",
]


def credit_line() -> str:
    """One line for banners: ``PyMC_Bot 0.1.0 by codero-sus (GitHub) - website``."""
    return f"PyMC_Bot {__version__} by {__author__} ({__author_url__}) \u00b7 {__website__}"


def about() -> dict:
    """Authorship and credits, as served by ``GET /api/about``."""
    return {
        "name": "PyMC_Bot",
        "version": __version__,
        "author": __author__,
        "author_github": __author_url__,
        "website": __website__,
        "repository": __repository__,
        "license": __license__,
        "license_url": __license_url__,
        "license_summary": "free for personal, non-commercial use",
        "made_with": MADE_WITH,
        "credits": [
            {
                "name": "PyMC_Bot",
                "role": "bot, control panel, fleet, anti-AFK, playtime training, Fabric mod",
                "by": __author__,
                "url": __repository__,
            },
            {
                "name": "Cortex LLMHoster",
                "role": "optional self-hosted LLM brain (OpenAI-compatible)",
                "by": __author__,
                "url": "https://github.com/codero-sus/Cortex_LLMHoster",
            },
        ],
        "built_with": [
            {"name": "FastAPI + Uvicorn", "url": "https://fastapi.tiangolo.com/"},
            {"name": "mineflayer", "url": "https://github.com/PrismarineJS/mineflayer"},
            {"name": "Fabric", "url": "https://fabricmc.net/"},
            {"name": "Ollama", "url": "https://ollama.com/"},
            {"name": "NumPy / PyTorch", "url": "https://numpy.org/"},
        ],
    }
