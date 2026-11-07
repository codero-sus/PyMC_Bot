from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

from pymc_bot import __main__ as cli


def setup_source(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "pymc_bot").mkdir(parents=True)
    (root / "updater.py").write_text("")
    (root / "pyproject.toml").write_text("")
    monkeypatch.setattr(cli, "__file__", str(root / "pymc_bot" / "__main__.py"))
    monkeypatch.setattr(sys, "argv", ["pymc-bot", "serve"])
    monkeypatch.delenv("PYMC_BOT_UPDATE_RESTARTED", raising=False)
    return root


def test_startup_relaunches_after_update(tmp_path, monkeypatch):
    root = setup_source(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append((a, kw)) or SimpleNamespace(returncode=10))
    monkeypatch.setattr(cli.os, "execv", lambda *args: calls.append(args))
    cli._prepare_startup()
    assert calls[0][0][0][-1] == "--auto"
    assert calls[1][1][-1] == "serve"
    assert (root / ".pymc_update.lock").exists()
    assert cli.os.environ["PYMC_BOT_UPDATE_RESTARTED"] == "1"


def test_startup_skips_git_and_non_runtime_commands(tmp_path, monkeypatch):
    root = setup_source(tmp_path, monkeypatch)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected update")))
    (root / ".git").mkdir()
    cli._prepare_startup()
    (root / ".git").rmdir()
    monkeypatch.setattr(sys, "argv", ["pymc-bot", "train"])
    cli._prepare_startup()


def test_stale_lock_file_does_not_disable_checks(tmp_path, monkeypatch):
    root = setup_source(tmp_path, monkeypatch)
    (root / ".pymc_update.lock").write_text("old")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append(a) or SimpleNamespace(returncode=0))
    cli._prepare_startup()
    assert len(calls) == 1


def test_restart_guard_is_consumed(tmp_path, monkeypatch):
    setup_source(tmp_path, monkeypatch)
    monkeypatch.setenv("PYMC_BOT_UPDATE_RESTARTED", "1")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append(a) or SimpleNamespace(returncode=0))
    cli._prepare_startup()
    assert not calls and "PYMC_BOT_UPDATE_RESTARTED" not in cli.os.environ
    cli._prepare_startup()
    assert len(calls) == 1


def test_running_monitor_restarts_only_when_idle(tmp_path, monkeypatch):
    import threading

    root = setup_source(tmp_path, monkeypatch)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=11))
    bot = SimpleNamespace(state="disconnected")
    fleet = SimpleNamespace(all_bots=lambda: [bot], ai_running=lambda _bot: False)
    app = SimpleNamespace(state=SimpleNamespace(fleet=fleet, training_service=SimpleNamespace(running=False)))
    server = SimpleNamespace(should_exit=False)
    restart = threading.Event()

    class Once:
        calls = 0

        def wait(self, _interval):
            self.calls += 1
            return self.calls > 1

        def is_set(self):
            return False

    cli._watch_release(root, app, server, Once(), restart, 0.001)
    assert restart.is_set() and server.should_exit
    restart.clear()
    server.should_exit = False
    bot.state = "connected"
    cli._watch_release(root, app, server, Once(), restart, 0.001)
    assert not restart.is_set() and not server.should_exit
    bot.state = "disconnected"
    app.state.training_service.running = True
    cli._watch_release(root, app, server, Once(), restart, 0.001)
    assert not restart.is_set() and not server.should_exit


def test_running_monitor_does_not_restart_after_shutdown(tmp_path, monkeypatch):
    import threading

    root = setup_source(tmp_path, monkeypatch)
    finished, restart = threading.Event(), threading.Event()

    def probe(*_args, **_kwargs):
        finished.set()
        return SimpleNamespace(returncode=11)

    monkeypatch.setattr(subprocess, "run", probe)
    server = SimpleNamespace(should_exit=False)
    cli._watch_release(root, SimpleNamespace(), server, finished, restart, 0.001)
    assert not restart.is_set() and not server.should_exit
