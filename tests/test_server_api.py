"""REST/WebSocket API tests for the Uvicorn config page."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pymc_bot.server import create_app


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(config_path=tmp_path / "cfg.json")
    with TestClient(app) as test_client:
        yield test_client


def _set_simulated(client: TestClient) -> None:
    response = client.put(
        "/api/config",
        json={"minecraft": {"backend": "simulated", "username": "ApiBot"}},
    )
    assert response.status_code == 200


def _connect(client: TestClient) -> None:
    _set_simulated(client)
    assert client.post("/api/bot/start").status_code == 200
    for _ in range(50):
        if client.get("/api/status").json()["bot"]["state"] == "connected":
            return
        time.sleep(0.1)
    raise AssertionError("bot never connected")


# ---------------------------------------------------------------------- pages
def test_index_page_renders(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "PyMC_Bot" in response.text
    assert "/static/app.js" in response.text


def test_static_assets_are_served(client: TestClient):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_healthz(client: TestClient):
    payload = client.get("/healthz").json()
    assert payload["ok"] is True and payload["version"]


# --------------------------------------------------------------------- status
def test_status_payload(client: TestClient):
    payload = client.get("/api/status").json()
    assert payload["bot"]["state"] == "disconnected"
    assert payload["agent"]["running"] is False
    assert "node" in payload["backends"]
    assert "ollama" in payload


def test_contract_lists_actions(client: TestClient):
    payload = client.get("/api/contract").json()
    assert "wander" in payload["actions"]
    assert "connect" in payload["bridge_commands"]
    assert payload["websocket"]["path"] == "/ws"


def test_logs_endpoint(client: TestClient):
    events = client.get("/api/logs?limit=50").json()["events"]
    assert isinstance(events, list)
    assert any("control panel ready" in event["message"] for event in events)
    assert client.post("/api/logs/clear").json()["ok"] is True
    assert client.get("/api/logs").json()["events"] == []


# --------------------------------------------------------------------- config
def test_config_roundtrip(client: TestClient):
    original = client.get("/api/config").json()["config"]
    assert original["minecraft"]["username"] == "PyMC_Bot"

    client.put("/api/config", json={"minecraft": {"port": 25599}, "agent": {"allow_attacking": True}})
    updated = client.get("/api/config").json()["config"]
    assert updated["minecraft"]["port"] == 25599
    assert updated["agent"]["allow_attacking"] is True
    assert updated["minecraft"]["username"] == "PyMC_Bot"  # merged, not replaced


def test_config_rejects_invalid_values(client: TestClient):
    response = client.put("/api/config", json={"minecraft": {"port": 123456}})
    assert response.status_code == 422


def test_config_reset(client: TestClient):
    client.put("/api/config", json={"minecraft": {"host": "mc.example.org"}})
    payload = client.post("/api/config/reset").json()
    assert payload["config"]["minecraft"]["host"] == "127.0.0.1"


# ------------------------------------------------------------------- commands
def test_commands_require_a_connection(client: TestClient):
    assert client.post("/api/bot/chat", json={"message": "hi"}).status_code == 409
    assert client.post("/api/bot/command", json={"command": "list"}).status_code == 409
    assert client.post("/api/ai/start").status_code == 409
    assert client.post("/api/ai/step").status_code == 409


def test_connect_chat_and_disconnect(client: TestClient):
    _connect(client)

    status = client.get("/api/status").json()
    assert status["bot"]["backend"] == "simulated"
    assert status["bot"]["position"] is not None

    assert client.post("/api/bot/chat", json={"message": "hello api"}).status_code == 200
    assert client.post("/api/bot/command", json={"command": "help"}).status_code == 200

    assert client.post("/api/bot/stop").status_code == 200
    assert client.get("/api/status").json()["bot"]["state"] == "disconnected"


def test_manual_actions(client: TestClient):
    _connect(client)

    response = client.post("/api/bot/action", json={"action": "jump"})
    assert response.status_code == 200 and response.json()["ok"] is True

    response = client.post("/api/bot/action", json={"action": "wander"})
    assert response.json()["detail"] == "explored the area"

    response = client.post("/api/bot/action", json={"raw": '{"action": "say", "message": "raw json"}'})
    assert response.json()["ok"] is True

    assert client.post("/api/bot/action", json={"action": "dance"}).status_code == 422
    assert client.post("/api/bot/action", json={"raw": "not json"}).status_code == 422
    assert client.post("/api/bot/action", json={}).status_code == 422


def test_ai_loop_endpoints(client: TestClient):
    _connect(client)
    client.put("/api/config", json={"ollama": {"enabled": False, "decision_interval": 0.5}})

    assert client.post("/api/ai/start").json()["running"] is True
    assert client.get("/api/ai/status").json()["running"] is True
    time.sleep(1.0)
    assert client.post("/api/ai/step").json()["decision"] is not None
    assert client.post("/api/ai/stop").json()["stopped"] is True
    deadline = time.monotonic() + 3
    while client.get("/api/ai/status").json()["running"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert client.get("/api/ai/status").json()["running"] is False


def test_backends_endpoint(client: TestClient):
    payload = client.get("/api/backends").json()
    assert payload["backends"]["simulated"]["available"] is True
    assert isinstance(payload["node_ready"], bool)


def test_ollama_health_endpoint(client: TestClient):
    payload = client.get("/api/ollama/health?force=true").json()
    assert isinstance(payload["ok"], bool)
    assert payload["detail"]


# ------------------------------------------------------------------ websocket
def test_websocket_sends_hello_status_and_events(client: TestClient):
    with client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json()["type"] == "hello"
        assert websocket.receive_json()["type"] == "status"

        client.app.state.log.add("socket test event", "info", "test")
        frame = None
        for _ in range(5):
            frame = websocket.receive_json()
            if frame["type"] == "event":
                break
        assert frame is not None
        assert frame["data"]["message"] == "socket test event"


def test_slow_health_probe_does_not_block_the_websocket_event_loop(tmp_path, monkeypatch):
    import threading

    from pymc_bot import server

    entered, release, handled = threading.Event(), threading.Event(), threading.Event()

    def slow_ping(_self):
        entered.set()
        assert release.wait(timeout=4)
        return True, "ok"

    monkeypatch.setattr(server.OllamaClient, "ping", slow_ping)
    app = create_app(config_path=tmp_path / "cfg.json")
    with TestClient(app) as client, client.websocket_connect("/ws") as socket:
        assert socket.receive_json()["type"] == "hello"
        assert entered.wait(2)

        def get_health():
            if client.get("/healthz").status_code == 200:
                handled.set()

        thread = threading.Thread(target=get_health, daemon=True)
        thread.start()
        try:
            assert handled.wait(2), "a slow model health check blocked the event loop"
        finally:
            release.set()
            thread.join(timeout=3)
        assert socket.receive_json()["type"] == "status"


def test_status_is_refreshed_during_a_continuous_event_stream(tmp_path, monkeypatch):
    import threading

    from pymc_bot import server
    from pymc_bot.events import EventLog

    monkeypatch.setattr(server, "WEBSOCKET_STATUS_INTERVAL", 0.05)
    monkeypatch.setattr(server.OllamaClient, "ping", lambda _self: (True, "ok"))
    log = EventLog()
    app = create_app(config_path=tmp_path / "cfg.json", log=log)
    with TestClient(app) as client, client.websocket_connect("/ws") as socket:
        assert socket.receive_json()["type"] == "hello"
        assert socket.receive_json()["type"] == "status"
        done = threading.Event()

        def flood():
            deadline = time.monotonic() + 0.7
            while time.monotonic() < deadline:
                log.add("busy")
                time.sleep(0.005)
            done.set()

        producer = threading.Thread(target=flood, daemon=True)
        producer.start()
        try:
            while True:
                frame = socket.receive_json()
                if frame["type"] == "status":
                    assert not done.is_set(), "status was starved until events stopped"
                    break
        finally:
            producer.join(timeout=2)


def test_ollama_models_prioritize_installed_minecraftism(client, monkeypatch):
    from pymc_bot import server

    class Client:
        def __init__(self, **kwargs):
            pass

        def list_models(self):
            return [{"name": "other"}, {"name": "minecraftism-vl:latest"}, {"name": "Minecraftism:latest"}]

        def close(self):
            pass

    monkeypatch.setattr(server, "OllamaClient", Client)
    models = client.get("/api/ollama/models").json()["models"]
    assert set(models[:2]) == {"Minecraftism:latest", "minecraftism-vl:latest"}
    assert models[-1] == "other"
