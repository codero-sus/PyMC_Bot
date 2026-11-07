"""Exercise the real Java HTTP client with small Minecraft stubs and a local HTTP server.

Set PYMC_JAVA and PYMC_ECJ to a Java runtime and Eclipse compiler jar to enable this test.
This is not a substitute for compiling or running the Fabric mod.
"""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_java_botlink_http_and_priority_queue(tmp_path):
    java, ecj = os.getenv("PYMC_JAVA"), os.getenv("PYMC_ECJ")
    if not java or not ecj:
        pytest.skip("set PYMC_JAVA and PYMC_ECJ to run the Java client harness")
    stubs = {
        "dev/pymc/pymcb/PlaytimeConfig.java": """package dev.pymc.pymcb;
public class PlaytimeConfig { public String panelUrl, panelToken = "";
private static final PlaytimeConfig INSTANCE = new PlaytimeConfig();
public static PlaytimeConfig get() { return INSTANCE; }}""",
        "dev/pymc/pymcb/client/PymcBotScreen.java": """package dev.pymc.pymcb.client;
public class PymcBotScreen { public static final String CHAT_PREFIX = ""; }""",
        "dev/pymc/pymcb/client/compat/Mc.java": """package dev.pymc.pymcb.client.compat;
public class Mc { public static void message(Object player, Object text) {} }""",
        "net/minecraft/client/Minecraft.java": """package net.minecraft.client;
public class Minecraft { public Object player = new Object(); private static final Minecraft INSTANCE = new Minecraft();
public static Minecraft getInstance() { return INSTANCE; } public void execute(Runnable r) { r.run(); }}""",
        "net/minecraft/network/chat/Component.java": """package net.minecraft.network.chat;
public class Component { public static Component literal(String text) { return new Component(); }}""",
    }
    sources = []
    for name, content in stubs.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        sources.append(str(path))
    sources += [str(ROOT / "mod/src/main/java/dev/pymc/pymcb/Json.java"),
                str(ROOT / "mod/src/client/java/dev/pymc/pymcb/client/BotLink.java"),
                str(ROOT / "tests/fixtures/java/BotLinkHarness.java")]
    subprocess.run([java, "-jar", ecj, "-21", "-d", str(tmp_path), *sources], check=True, timeout=60)
    subprocess.run([java, "-cp", str(tmp_path), "dev.pymc.pymcb.client.BotLinkHarness"], check=True, timeout=20)

    import socket
    import threading
    import time

    import uvicorn

    from pymc_bot.server import create_app

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(config_path=tmp_path / "config.json"), log_level="error"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 8
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started
        subprocess.run([java, "-cp", str(tmp_path), "dev.pymc.pymcb.client.BotLinkHarness",
                        f"http://127.0.0.1:{port}"], check=True, timeout=30)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
