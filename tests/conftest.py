"""Shared pytest fixtures."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pymc_bot.backends import SimulatedBackend
from pymc_bot.bot import MinecraftBot
from pymc_bot.config import ConfigStore
from pymc_bot.events import EventLog


@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    return tmp_path / "pymc_bot_config.json"


@pytest.fixture()
def store(config_path: Path) -> ConfigStore:
    store = ConfigStore(config_path)
    store.load()
    return store


@pytest.fixture()
def log() -> EventLog:
    return EventLog()


@pytest.fixture()
def sim_backend(log: EventLog) -> SimulatedBackend:
    backend = SimulatedBackend(log, ambient_chat=False)
    yield backend
    backend.stop()


@pytest.fixture()
def live_bot(store: ConfigStore, log: EventLog) -> MinecraftBot:
    """A connected bot running against the simulated world."""
    store.update({"minecraft": {"backend": "simulated", "username": "TestBot"}})
    backend = SimulatedBackend(log, ambient_chat=False)
    backend.ambient_chat = False
    bot = MinecraftBot(store, log, auto_reconnect=False, backend=backend)
    bot.start()
    _wait_for(lambda: bot.connected, timeout=5.0)
    yield bot
    bot.stop()


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture()
def wait_for():
    return _wait_for
