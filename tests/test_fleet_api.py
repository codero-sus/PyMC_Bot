"""Fleet + premium endpoints of the Uvicorn control panel."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pymc_bot.server import create_app


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(config_path=tmp_path / "api_fleet.json")
    with TestClient(app) as test_client:
        test_client.put(
            "/api/config",
            json={
                "minecraft": {"backend": "simulated", "username": "MainBot"},
                "fleet": {"max_bots": 10, "count": 3, "name_pattern": "Pop_{n}", "stagger_seconds": 0.05},
            },
        )
        test_client.post("/api/bot/start")
        for _ in range(60):
            if test_client.get("/api/status").json()["bot"]["state"] == "connected":
                break
            time.sleep(0.1)
        yield test_client


def _wait(client: TestClient, predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate(client.get("/api/fleet/status").json()):
            return True
        time.sleep(0.15)
    return False


# ------------------------------------------------------------------- populate
def test_populate_endpoint_adds_players(client: TestClient):
    response = client.post("/api/fleet/populate", json={"count": 3, "stagger": 0.05})
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 3
    assert payload["queued"] == ["Pop_1", "Pop_2", "Pop_3"]

    assert _wait(client, lambda status: status["connected"] == 4)
    status = client.get("/api/fleet/status").json()
    assert status["size"] == 4
    assert status["extra_bots"] == 3
    assert {bot["username"] for bot in status["bots"]} == {"MainBot", "Pop_1", "Pop_2", "Pop_3"}


def test_populate_with_explicit_usernames(client: TestClient):
    response = client.post("/api/fleet/populate", json={"usernames": ["alpha", "beta"]})
    assert response.status_code == 200
    assert response.json()["queued"] == ["alpha", "beta"]
    assert _wait(client, lambda status: status["connected"] == 3)


def test_populate_validates_requests(client: TestClient):
    assert client.post("/api/fleet/populate", json={"count": 0}).status_code == 422
    assert client.post("/api/fleet/populate", json={"count": 99}).status_code == 422   # > max_bots
    assert client.post("/api/fleet/populate", json={"pattern": "NoIndex"}).status_code == 422
    assert client.post("/api/fleet/populate", json={"count": 2, "auth": "microsoft"}).status_code == 422  # generated names are not emails


# ---------------------------------------------------------------------- spawn
def test_add_premium_player(client: TestClient):
    response = client.post(
        "/api/fleet/spawn",
        json={"username": "buyer@example.com", "auth": "microsoft", "ai": "off"},
    )
    assert response.status_code == 200
    member = response.json()["member"]
    assert member["auth"] == "microsoft"
    assert member["configured_username"] == "buyer@example.com"

    status = client.get("/api/fleet/status").json()
    entry = next(bot for bot in status["bots"] if bot["configured_username"] == "buyer@example.com")
    assert entry["auth"] == "microsoft"


def test_spawn_validation_errors(client: TestClient):
    assert client.post("/api/fleet/spawn", json={"username": "bad name!"}).status_code == 422
    assert client.post("/api/fleet/spawn", json={"username": "not-an-email", "auth": "microsoft"}).status_code == 422
    assert client.post("/api/fleet/spawn", json={"username": "Dancer", "ai": "disco"}).status_code == 422
    assert client.post("/api/fleet/spawn", json={"username": "MainBot"}).status_code == 422  # primary already uses it


# ------------------------------------------------------------------- controls
def test_select_and_actions_across_the_fleet(client: TestClient):
    client.post("/api/fleet/populate", json={"count": 2, "stagger": 0.05})
    assert _wait(client, lambda status: status["connected"] == 3)

    assert client.post("/api/fleet/select", json={"bot": "Pop_1"}).json()["selected"] == "Pop_1"
    status = client.get("/api/status").json()
    assert status["bot"]["username"] == "Pop_1"          # the panel now controls Pop_1
    assert status["fleet"]["selected"] == "Pop_1"

    assert client.post("/api/bot/action", json={"action": "jump"}).json()["ok"] is True

    result = client.post("/api/bot/action", json={"action": "wander", "bot": "all"}).json()
    assert result["ok"] is True
    assert len(result["results"]) == 3

    assert client.post("/api/bot/chat", json={"message": "hi all", "bot": "all"}).json()["sent"] == 3
    assert client.post("/api/bot/chat", json={"message": "hi", "bot": "ghost"}).status_code == 404
    assert client.post("/api/fleet/select", json={"bot": "ghost"}).status_code == 404


def test_stop_start_and_remove_endpoints(client: TestClient):
    client.post("/api/fleet/populate", json={"count": 2, "stagger": 0.05})
    assert _wait(client, lambda status: status["connected"] == 3)

    assert client.post("/api/fleet/stop", json={"bot": "Pop_1"}).json()["ok"] is True
    assert client.post("/api/fleet/stop", json={"bot": "ghost"}).status_code == 404

    assert client.post("/api/fleet/start", json={"bot": "Pop_1"}).json()["started"] == 1
    assert _wait(client, lambda status: status["connected"] == 3)

    assert client.post("/api/fleet/remove", json={"bot": "Pop_2"}).json()["fleet"]["size"] == 2
    assert client.post("/api/fleet/remove", json={"bot": "Pop_2"}).status_code == 404

    stopped = client.post("/api/fleet/stop", json={"remove": True}).json()
    assert stopped["stopped"] == 1
    assert stopped["fleet"]["extra_bots"] == 0


def test_broadcast_endpoint(client: TestClient):
    # the main bot is connected, so it speaks too
    assert client.post("/api/fleet/broadcast", json={"message": "hello"}).json()["sent"] == 1

    client.post("/api/fleet/populate", json={"count": 2, "stagger": 0.05})
    assert _wait(client, lambda status: status["connected"] == 3)
    assert client.post("/api/fleet/broadcast", json={"message": "hello"}).json()["sent"] == 3

    # with nobody connected at all, there is nobody to speak through
    client.post("/api/fleet/stop")
    client.post("/api/bot/stop")
    time.sleep(0.3)
    assert client.post("/api/fleet/broadcast", json={"message": "anyone?"}).status_code == 409


def test_roster_endpoints(client: TestClient):
    client.post("/api/fleet/populate", json={"count": 2, "stagger": 0.05})
    roster = client.get("/api/fleet/roster").json()["roster"]
    assert [spec["username"] for spec in roster] == ["Pop_1", "Pop_2"]

    client.delete("/api/fleet/roster")
    assert client.get("/api/fleet/roster").json()["roster"] == []
    # clearing the roster must not disconnect anyone
    assert _wait(client, lambda status: status["connected"] == 3)


def test_fleet_status_shape(client: TestClient):
    status = client.get("/api/fleet/status").json()
    for key in ("enabled", "max_bots", "size", "extra_bots", "connected", "roster_size", "selected", "primary", "bots"):
        assert key in status
    assert status["size"] == 1 and status["extra_bots"] == 0
    assert status["primary"]["sponsor"] == "primary"
    assert status["primary"]["selected"] is True


def test_status_payload_includes_the_fleet(client: TestClient):
    payload = client.get("/api/status").json()
    assert "fleet" in payload
    assert payload["fleet"]["size"] == 1
    assert payload["bot"]["username"] == "MainBot"


def test_contract_lists_auth_and_ai_modes(client: TestClient):
    contract = client.get("/api/contract").json()
    assert contract["auth_modes"] == ["offline", "microsoft"]
    assert set(contract["ai_modes"]) == {"off", "heuristic", "ollama"}


def test_all_bots_stop_when_the_panel_shuts_down(tmp_path: Path):
    app = create_app(config_path=tmp_path / "shutdown.json")
    with TestClient(app) as client:
        client.put("/api/config", json={"minecraft": {"backend": "simulated"}, "fleet": {"stagger_seconds": 0.05}})
        client.post("/api/bot/start")
        for _ in range(50):
            if client.get("/api/status").json()["bot"]["state"] == "connected":
                break
            time.sleep(0.1)
        client.post("/api/fleet/populate", json={"count": 2})
        time.sleep(2.5)
        assert client.get("/api/fleet/status").json()["connected"] >= 2
    # after the context manager exits, everything must be shut down
    assert app.state.fleet.members == []
    assert app.state.fleet.primary.state == "disconnected"
