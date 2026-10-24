"""The Kaggle notebook's Discord webhook messages: secret, payloads, rate limits, server warnings, log events."""
import io
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
from queue import Queue

NOTEBOOK = Path(__file__).resolve().parents[1] / "pynb" / "PyMC_Minecraft_LLM_Kaggle.ipynb"
HOOK = "https://discord.com/api/webhooks/123456789012345678/abc-DEF_123"


def _cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in nb["cells"]]


def _block(cell, start, end):
    return cell[cell.index(start):cell.index(end)]


def _helper(monkeypatch, url=HOOK, mention="", replies=None, extra=None):
    """Run the notebook's Discord helper with a fake Kaggle secret and a fake Discord."""
    asked, sent = [], []
    replies = list(replies or [])

    class Secrets:
        def get_secret(self, name):
            asked.append(name)
            return url

    monkeypatch.setitem(sys.modules, "kaggle_secrets", types.SimpleNamespace(UserSecretsClient=Secrets))

    def urlopen(request, timeout=0):
        sent.append({"url": request.full_url, "headers": dict(request.header_items()),
                     "payload": json.loads(request.data.decode("utf-8"))})
        if replies:
            code = replies.pop(0)
            body = io.BytesIO(json.dumps({"retry_after": 0.01}).encode())
            raise urllib.error.HTTPError(request.full_url, code, "error", {}, body)
        return io.BytesIO(b"")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    env = {"re": re, "json": json, "time": time, "threading": threading, "urllib": urllib, "DiscordQueue": Queue,
           "DISCORD": True, "DISCORD_SECRET": "discord_webhook_inform_me1", "DISCORD_MENTION": mention,
           "__name__": "notebook"}
    env.update(extra or {})
    exec(_block(_cells()[3], "# ---- Discord: the webhook URL", "# ---- end Discord\n"), env)  # noqa: S102
    return env, asked, sent


def _wait(env):
    end = time.monotonic() + 5
    while env["DISCORD_OUTBOX"].unfinished_tasks and time.monotonic() < end:
        pass


def test_settings_and_secret_name():
    settings = _cells()[1]
    assert 'DISCORD_SECRET = "discord_webhook_inform_me1"' in settings
    assert "DISCORD = True" in settings and "DISCORD_EVERY_MIN = 30" in settings and 'NOTEBOOK_URL = ""' in settings
    assert "discord.com/api/webhooks/" not in settings  # the URL lives only in Kaggle's secrets
    assert "discord_webhook_inform_me1" in _cells()[0]  # the intro says how to set it up


def test_messages_use_the_secret_and_never_print_it(monkeypatch, capsys):
    env, asked, sent = _helper(monkeypatch)
    assert asked == ["discord_webhook_inform_me1"]
    assert env["discord"]("step 10/100", "minecraftism", "info") is True
    _wait(env)
    assert len(sent) == 1 and sent[0]["url"] == HOOK
    payload = sent[0]["payload"]
    assert payload["embeds"][0]["description"] == "step 10/100" and payload["embeds"][0]["title"] == "minecraftism"
    assert payload["allowed_mentions"] == {"parse": []} and "content" not in payload  # no ping by default
    assert "User-agent" in sent[0]["headers"]  # Discord's Cloudflare blocks urllib's default agent
    assert HOOK not in capsys.readouterr().out


def test_ping_only_for_important_messages(monkeypatch):
    env, _asked, sent = _helper(monkeypatch, mention="<@42>")
    env["discord"]("done", "result", "ok", ping=True)
    env["discord"]("status", "status", "info")
    _wait(env)
    assert sent[0]["payload"]["content"] == "<@42>" and "users" in sent[0]["payload"]["allowed_mentions"]["parse"]
    assert "content" not in sent[1]["payload"]


def test_rate_limit_is_waited_out_and_a_dead_webhook_stops(monkeypatch, capsys):
    env, _asked, sent = _helper(monkeypatch, replies=[429])
    env["discord"]("x")
    _wait(env)
    assert len(sent) == 2  # retried after Discord's retry_after
    env, _asked, sent = _helper(monkeypatch, replies=[404])
    env["discord"]("x")
    _wait(env)
    assert env["DISCORD_STATE"]["dead"] and env["discord"]("y") is False
    out = capsys.readouterr().out
    assert "HTTP 404" in out and HOOK not in out


def test_throttle_and_bad_or_missing_url(monkeypatch, capsys):
    env, _asked, _sent = _helper(monkeypatch)
    assert env["discord"]("a", key="k", every=600) is True
    assert env["discord"]("b", key="k", every=600) is False
    env, _asked, _sent = _helper(monkeypatch, url="https://example.com/hook")
    assert env["DISCORD_URL"] == "" and env["discord"]("a") is False
    assert "not a Discord webhook URL" in capsys.readouterr().out
    env, _asked, _sent = _helper(monkeypatch, extra={"DISCORD": False})
    assert env["discord"]("a") is False


def test_server_warnings_before_kaggle_closes_it(monkeypatch, tmp_path):
    env, _asked, _sent = _helper(monkeypatch)
    start = 1_000_000.0
    env.update({"SESSION_START": start, "DEADLINE": start + 11.5 * 3600, "NOTEBOOK_URL": "https://www.kaggle.com/code/me/nb",
                "DISCORD_EVERY_MIN": 0, "WORK": str(tmp_path), "TMP": str(tmp_path), "shutil": shutil,
                "subprocess": subprocess, "os": __import__("os"), "DISCORD_URL": ""})  # "" = no monitor thread
    exec(_block(_cells()[3], "# ---- Discord server monitor", "# ---- end Discord server monitor"), env)  # noqa: S102
    warned = {}
    assert env["server_warnings"](start + 10 * 3600, warned) == []
    first = env["server_warnings"](start + 11 * 3600 + 60, warned)
    assert [title for title, _text, ping in first if ping] == ["Server closes in 59 min"]
    assert "https://www.kaggle.com/code/me/nb" in first[0][1]
    assert env["server_warnings"](start + 11 * 3600 + 120, warned) == []  # said once
    titles = [t for t, _x, _p in env["server_warnings"](start + 12 * 3600 - 300, warned)]
    assert titles == ["Server closes in 5 min", "Server closes in 5 min"]  # the 30 and 10 minute warnings
    status = env["server_status"]()
    assert "https://www.kaggle.com/code/me/nb" in status and "Disk free" in status


def test_server_closing_message(monkeypatch):
    env, _asked, sent = _helper(monkeypatch, mention="<@1>")
    env.update({"SERVER_URL": "u"})
    exec(_block(_cells()[3], "def server_closing", "\n\nif DISCORD_URL:\n    import atexit"), env)  # noqa: S102
    env["server_closing"]()
    env["server_closing"]()  # only once
    _wait(env)
    assert len(sent) == 1 and sent[0]["payload"]["embeds"][0]["title"] == "Server closing"
    assert sent[0]["payload"]["content"] == "<@1>"
    cell12 = _cells()[12]
    assert 'DISCORD_STATE["finished"] = True' in cell12 and "discord_flush()" in cell12


def test_log_lines_become_discord_events(monkeypatch):
    env, _asked, sent = _helper(monkeypatch)
    exec(_block(_cells()[6], "# ---- Discord events", "\n\ndef discord_tail"), env)  # noqa: S102
    line = env["discord_line"]
    assert line("minecraftism", "===== minecraftism on GPU 0,1 =====")
    assert line("minecraftism", "uploaded the GGUF edition -> https://huggingface.co/me/Minecraftism-4B-GGUF")
    assert line("minecraftism", "learned: the loss stopped falling (~0.020) - stopping at step 1600")
    assert not line("minecraftism", "step 100/4230 (2%) · loss 0.400")  # status only, sent with the server status
    assert env["DISCORD_STATUS"]["minecraftism"].startswith("step 100/4230")
    assert line("minecraftism", "Traceback (most recent call last):")
    assert not line("minecraftism", "RuntimeError: CUDA error")  # the same kind of error again: throttled
    _wait(env)
    titles = [m["payload"]["embeds"][0]["description"] for m in sent]
    assert titles[0] == "minecraftism on GPU 0,1"
    assert any("huggingface.co/me/Minecraftism-4B-GGUF" in t for t in titles)
