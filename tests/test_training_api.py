"""The panel API for training and running a model trained on playtime."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pymc_bot.server import create_app
from pymc_bot.train import list_models


@pytest.fixture()
def panel(tmp_path: Path):
    config_path = tmp_path / "config.json"
    models_dir = tmp_path / "models"
    dataset = tmp_path / "pymc-playtime" / "dataset.jsonl"
    config_path.write_text(
        json.dumps(
            {
                "minecraft": {"backend": "simulated", "username": "ApiBot"},
                "ollama": {"enabled": False},
                "antiafk": {"enabled": False},
                "fleet": {"enabled": False},
                "training": {"models_dir": str(models_dir), "dataset": str(dataset), "steps": 40, "checkpoint_every": 20},
            }
        ),
        encoding="utf-8",
    )
    app = create_app(config_path=config_path)
    with TestClient(app) as client:
        yield {
            "client": client,
            "config_path": config_path,
            "models_dir": models_dir,
            "dataset": dataset,
            "app": app,
        }


def _train_demo(panel, **payload) -> dict:
    body = {"simulate_minutes": 1.5, "steps": 40, "checkpoint_every": 20, "run_name": "api-demo", **payload}
    response = panel["client"].post("/api/training/start", json=body)
    assert response.status_code == 200, response.text
    deadline = time.time() + 60.0
    while time.time() < deadline:
        status = panel["client"].get("/api/training/status").json()["status"]
        if not status["running"]:
            assert status["error"] is None, status["error"]
            return status
        time.sleep(0.1)
    raise AssertionError("training did not finish in time")


# --------------------------------------------------------------------- contract
def test_contract_lists_brains_and_player_actions(panel):
    contract = panel["client"].get("/api/contract").json()
    assert "trained" in contract["brains"]
    assert set(contract["training_engines"]) == {"mlp", "transformer"}
    assert "forward" in contract["player_actions"]


# ---------------------------------------------------------------------- status
def test_training_status_describes_the_dataset_and_settings(panel):
    payload = panel["client"].get("/api/training/status").json()
    assert payload["status"]["running"] is False
    assert payload["settings"]["engine"] == "mlp"
    assert payload["dataset"]["exists"] is False  # nothing recorded yet
    assert Path(payload["models_dir"]) == panel["models_dir"]
    assert payload["brain"] == "auto"


def test_status_includes_a_training_summary(panel):
    training = panel["client"].get("/api/status").json()["training"]
    assert set(training) >= {"running", "brain", "models", "active_run"}


# --------------------------------------------------------------------- training
def test_training_start_creates_a_usable_checkpoint(panel):
    status = _train_demo(panel)
    assert status["step"] == 40
    assert status["run"] == "api-demo"
    assert status["summary"]["checkpoint"].endswith("latest.npz")
    assert panel["dataset"].is_file()
    assert list_models(panel["models_dir"])[0]["run"] == "api-demo"

    dataset = panel["client"].get("/api/training/dataset").json()
    assert dataset["exists"] is True
    assert dataset["samples"] > 0
    assert dataset["actions"]["forward"] > 0


def test_training_start_rejects_a_missing_dataset(panel):
    response = panel["client"].post("/api/training/start", json={"dataset": "/tmp/not-recorded.jsonl"})
    assert response.status_code == 400
    assert "no playtime dataset" in response.json()["detail"]


def test_training_stop_is_reported_when_idle(panel):
    assert panel["client"].post("/api/training/stop").status_code == 409


def test_training_respects_the_configured_settings(panel):
    _train_demo(panel, run_name="configured", engine="mlp", steps=20, checkpoint_every=10)
    card = list_models(panel["models_dir"])[0]
    assert card["run"] == "configured"
    assert card["step"] == 20
    # the panel remembers what it was asked to do
    settings = panel["client"].get("/api/training/status").json()["settings"]
    assert settings["steps"] == 20
    assert settings["active_run"] == "configured"


# ----------------------------------------------------------------------- models
def test_models_endpoints_list_and_activate(panel):
    _train_demo(panel)
    listing = panel["client"].get("/api/models").json()
    assert listing["models"][0]["run"] == "api-demo"
    assert listing["active_run"] == "api-demo"

    detail = panel["client"].get("/api/models/api-demo")
    assert detail.status_code == 200
    assert detail.json()["feature_version"] == 1
    assert panel["client"].get("/api/models/missing").status_code == 404

    activated = panel["client"].post("/api/models/activate", json={"run": "api-demo"})
    assert activated.status_code == 200
    body = activated.json()
    assert body["active_run"] == "api-demo"
    assert body["brain"] == "trained"
    assert body["policy"]["run"] == "api-demo"

    config = panel["client"].get("/api/config").json()["config"]
    assert config["agent"]["mode"] == "trained"
    assert config["training"]["active_run"] == "api-demo"

    assert panel["client"].post("/api/models/activate", json={"run": "ghost"}).status_code == 404
    assert panel["client"].post("/api/models/deactivate").json()["brain"] == "auto"


def test_preview_shows_what_the_model_would_do(panel):
    _train_demo(panel)
    panel["client"].post("/api/models/activate", json={"run": "api-demo"})
    preview = panel["client"].get("/api/policy/preview").json()
    assert preview["prediction"]["action"] in preview["policy"]["actions"]
    assert 0.0 <= preview["prediction"]["probs"][preview["prediction"]["action"]] <= 1.0
    assert "position" in preview["observation"]


def test_preview_without_a_model_explains_itself(panel):
    response = panel["client"].get("/api/policy/preview")
    assert response.status_code == 400
    assert "no trained model" in response.json()["detail"]


def test_export_to_ollama_needs_a_reachable_ollama(panel):
    _train_demo(panel)
    response = panel["client"].post("/api/models/export-ollama", json={"run": "api-demo"})
    # no Ollama in the test environment: the panel must say so, not crash
    assert response.status_code in (502, 200)


# --------------------------------------------------- trained brain plays the bot
def test_the_trained_brain_actually_drives_the_connected_bot(panel):
    _train_demo(panel)
    client = panel["client"]
    assert client.post("/api/models/activate", json={"run": "api-demo"}).status_code == 200
    assert client.post("/api/bot/start").status_code == 200
    assert client.post("/api/ai/start").status_code == 200

    deadline = time.time() + 25.0
    status = client.get("/api/status").json()
    while time.time() < deadline:
        status = client.get("/api/status").json()
        if status["agent"]["decisions"] >= 4:
            break
        time.sleep(0.2)

    assert status["agent"]["brain"] == "trained"
    assert status["agent"]["decisions"] >= 4
    assert status["agent"]["trained"]["run"] == "api-demo"
    assert status["agent"]["last_decision"]["source"] == "trained:api-demo"
    # and the bot really moved: the recorded player's walking is reproduced
    position = status["bot"]["position"]
    assert abs(position["x"]) + abs(position["z"]) > 0.5
    client.post("/api/ai/stop")
