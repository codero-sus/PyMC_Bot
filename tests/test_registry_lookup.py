"""Read-only client-registry inspection (not a full modded-server dump)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pymc_bot.server import create_app


def test_node_registry_filter_pagination_and_validation():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed")
    root = Path(__file__).resolve().parents[1]
    subprocess.run([node, str(root / "tests/fixtures/registry_bridge_harness.js"),
                    str(root / "pymc_bot/node/minecraft_bridge.js")], check=True, timeout=10)


def test_registry_api_selected_bot_and_bounds(live_bot, store, monkeypatch):
    calls = []
    def perform(cmd, params, **kwargs):
        calls.append((cmd, params))
        return True, {"source": "mineflayer_registry", "entries": [], "total": 0}
    monkeypatch.setattr(live_bot, "_perform", perform)
    with TestClient(create_app(config_path=store.path, bot=live_bot)) as client:
        assert client.get("/api/bot/registry?kind=recipe").status_code == 422
        assert client.get("/api/bot/registry?limit=101").status_code == 422
        assert client.get("/api/bot/registry?query=bad%20name").status_code == 422
        result = client.get("/api/bot/registry?kind=block&query=stone&offset=5&limit=10")
        assert result.status_code == 200
        assert result.json()["source"] == "mineflayer_registry"
        assert calls[-1] == ("registry_lookup", {"kind": "block", "query": "stone", "offset": 5, "limit": 10})
        assert client.get("/api/bot/recipes?item=bad%20item").status_code == 422
        assert client.get("/api/bot/recipes?item=oak_planks&limit=51").status_code == 422
        assert client.get("/api/bot/recipes?item=oak_planks").status_code == 200
        assert calls[-1] == ("recipe_lookup", {"item": "oak_planks", "limit": 20})


def test_simulated_backend_is_explicitly_unsupported(live_bot):
    ok, why, _ = live_bot.backend.perform("registry_lookup", {})
    assert not ok and "unavailable" in why
    ok, why, _ = live_bot.backend.perform("recipe_lookup", {})
    assert not ok and "unavailable" in why
