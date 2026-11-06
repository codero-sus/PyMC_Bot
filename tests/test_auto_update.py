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
    assert not (root / ".pymc_update.lock").exists()
    assert cli.os.environ["PYMC_BOT_UPDATE_RESTARTED"] == "1"


def test_startup_skips_git_and_non_runtime_commands(tmp_path, monkeypatch):
    root = setup_source(tmp_path, monkeypatch)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected update")))
    (root / ".git").mkdir()
    cli._prepare_startup()
    (root / ".git").rmdir()
    monkeypatch.setattr(sys, "argv", ["pymc-bot", "train"])
    cli._prepare_startup()
