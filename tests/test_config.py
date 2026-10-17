"""Config model + persistence tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pymc_bot.config import AppSettings, ConfigStore, default_config_path


def test_defaults_are_sane():
    settings = AppSettings()
    assert settings.minecraft.auth == "offline"          # cracked servers by default
    assert settings.minecraft.username == "PyMC_Bot"
    assert settings.minecraft.backend == "auto"
    assert settings.ollama.enabled is False              # bot works without an LLM
    assert settings.agent.allow_attacking is False       # no griefing by default
    assert settings.server.host == "0.0.0.0"


def test_store_roundtrip(tmp_path: Path):
    store = ConfigStore(tmp_path / "cfg.json")
    store.load()
    store.update({"minecraft": {"host": "mc.example.com", "port": 25566}, "ollama": {"enabled": True}})

    fresh = ConfigStore(tmp_path / "cfg.json")
    settings = fresh.load()
    assert settings.minecraft.host == "mc.example.com"
    assert settings.minecraft.port == 25566
    assert settings.ollama.enabled is True

    # partial merge keeps untouched sections
    assert settings.minecraft.username == "PyMC_Bot"
    assert settings.agent.allow_chat is True


def test_store_survives_corrupt_file(tmp_path: Path):
    path = tmp_path / "broken.json"
    path.write_text("{ not json at all", encoding="utf-8")
    store = ConfigStore(path)
    settings = store.load()
    assert settings.minecraft.username == "PyMC_Bot"
    assert store.load_error is not None
    assert "Could not read" in store.load_error


def test_store_write_is_atomic_and_creates_parents(tmp_path: Path):
    path = tmp_path / "nested" / "dir" / "cfg.json"
    store = ConfigStore(path)
    store.load()
    store.save()
    assert path.exists()
    assert not path.with_suffix(".tmp").exists()
    assert json.loads(path.read_text(encoding="utf-8"))["minecraft"]["auth"] == "offline"


def test_validation_rejects_bad_values():
    with pytest.raises(ValidationError):
        AppSettings.model_validate({"minecraft": {"port": 70000}})
    with pytest.raises(ValidationError):
        AppSettings.model_validate({"minecraft": {"username": "this-name-is-way-too-long"}})
    with pytest.raises(ValidationError):
        AppSettings.model_validate({"minecraft": {"auth": "mojang"}})
    with pytest.raises(ValidationError):
        AppSettings.model_validate({"minecraft": {"host": "   "}})


def test_update_rejects_invalid_patch(tmp_path: Path):
    store = ConfigStore(tmp_path / "cfg.json")
    store.load()
    with pytest.raises(ValidationError):
        store.update({"agent": {"max_chat_length": 10_000}})
    # the good settings are untouched
    assert store.settings.agent.max_chat_length == 200


def test_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    target = tmp_path / "custom.json"
    monkeypatch.setenv("PYMC_BOT_CONFIG", str(target))
    assert default_config_path() == target


def test_public_dict_is_json_serialisable():
    payload = json.dumps(AppSettings().public_dict())
    assert "minecraft" in payload
