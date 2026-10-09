"""Opt-in end-to-end test against a real Minecraft server.

It is skipped unless ``PYMC_TEST_SERVER`` is set, e.g.:

    PYMC_TEST_SERVER=127.0.0.1:25565 python -m pytest tests/test_node_live.py -v

Use a throwaway cracked/offline-mode server (PaperMC with
``online-mode=false``) - the bot will join as a normal player.
"""

from __future__ import annotations

import os
import time

import pytest

from pymc_bot.backends import NodeBridgeBackend, node_available
from pymc_bot.config import AppSettings
from pymc_bot.events import EventLog

SERVER = os.environ.get("PYMC_TEST_SERVER", "")
VERSION = os.environ.get("PYMC_TEST_VERSION", "auto")

pytestmark = pytest.mark.skipif(
    not SERVER or not node_available()[0],
    reason="set PYMC_TEST_SERVER=host:port and install mineflayer to run the live test",
)


def test_join_a_real_server():
    host, _, port = SERVER.rpartition(":")
    port = int(port or 25565)
    log = EventLog()
    backend = NodeBridgeBackend(log)
    settings = AppSettings()
    settings.minecraft = settings.minecraft.model_copy(
        update={"host": host, "port": port, "version": VERSION, "username": "PyMC_TestBot", "backend": "node"}
    )
    try:
        backend.start()
        backend.connect(settings.minecraft)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not backend.connected:
            time.sleep(0.25)
        assert backend.connected, "bot never spawned"
        snapshot = backend.snapshot()
        assert snapshot["position"] is not None
        assert snapshot["status"] == "connected"

        backend.say("PyMC_Bot live test, hello!")
        assert backend.set_control("forward", True) is None
        time.sleep(1.5)
        backend.stop_motion()
        after = backend.snapshot()
        assert after["position"] != snapshot["position"], "the bot did not move"
    finally:
        backend.stop()
