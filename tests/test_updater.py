"""Tests for updater.py - all offline (local zips, a temporary git remote, fake releases).

Standard library + pytest only: CI also runs this file with ``--noconftest`` on Windows and
macOS without the bot's packages, the way a fresh download meets the updater.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("updater", ROOT / "updater.py")
updater = importlib.util.module_from_spec(spec)
sys.modules["updater"] = updater
spec.loader.exec_module(updater)


def make_zip(files: dict[str, bytes | str], top: str = "PyMC_Bot-abc123", mode: int | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{top}/", "")
        for name, data in files.items():
            info = zipfile.ZipInfo(f"{top}/{name}")
            if mode is not None and name.endswith(".sh"):
                info.external_attr = (0o100000 | mode) << 16
            zf.writestr(info, data)
    return buf.getvalue()


def mod_jar(mc: str, mod_id: str = "pymcplaytime") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("fabric.mod.json", json.dumps({"id": mod_id, "depends": {"minecraft": mc}}))
    return buf.getvalue()


def release(tag: str, mcs: list[str], prerelease: bool = False, version: str | None = None) -> dict:
    version = version or tag[len(updater.MOD_TAG_PREFIX):]
    assets = []
    for mc in mcs:
        data = mod_jar(mc)
        assets.append({
            "name": f"pymcplaytime-{version}+{mc}.jar",
            "size": len(data),
            "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
            "browser_download_url": f"https://example.invalid/{version}/{mc}.jar",
            "_data": data,
        })
    return {"tag_name": tag, "prerelease": prerelease, "draft": False, "assets": assets}


# ---------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(("name", "expected"), [
    ("pymcplaytime-0.5.0+1.21.4.jar", ("0.5.0", "1.21.4")),
    ("pymcplaytime-0.5.0+26.3.jar", ("0.5.0", "26.3")),
    ("pymcplaytime-1.2+26.1.2.jar", ("1.2", "26.1.2")),
    ("pymcplaytime-0.5.0+1.21.4-gametest.jar", ("0.5.0", "1.21.4-gametest")),
    ("pymcplaytime-0.5.0+1.21.4.jar.old", None),
    ("pymcplaytime-0.5.0-sources.jar", None),
    ("fabric-api-0.119.0+1.21.4.jar", None),
])
def test_parse_jar_name(name, expected):
    assert updater.parse_jar_name(name) == expected


def test_version_tuple_orders_numerically():
    assert updater.version_tuple("0.10.0") > updater.version_tuple("0.9.9")
    assert updater.version_tuple("26.1") > updater.version_tuple("1.21.11")


def test_pick_release_prefers_stable_and_falls_back_to_pre():
    rels = [
        release("pymcplaytime-v0.6.0", ["1.21.4"], prerelease=True),
        release("pymcplaytime-v0.5.0", ["1.21.4"]),
        release("pymcplaytime-v0.10.0", ["1.21.4"], prerelease=True),
        {"tag_name": "v9.9.9", "prerelease": False, "draft": False, "assets": []},
        {"tag_name": "pymcplaytime-v9.0.0", "prerelease": False, "draft": True, "assets": []},
    ]
    assert updater.pick_release(rels, allow_pre=False)["tag_name"] == "pymcplaytime-v0.5.0"
    assert updater.pick_release(rels, allow_pre=True)["tag_name"] == "pymcplaytime-v0.10.0"
    only_pre = [release("pymcplaytime-v0.5.0", ["26.3"], prerelease=True)]
    assert updater.pick_release(only_pre, allow_pre=False)["tag_name"] == "pymcplaytime-v0.5.0"
    assert updater.pick_release([], allow_pre=True) is None


def test_asset_for_matches_exact_version():
    rel = release("pymcplaytime-v0.5.0", ["1.21.1", "1.21.10"])
    assert updater.asset_for(rel, "0.5.0", "1.21.1")["name"] == "pymcplaytime-0.5.0+1.21.1.jar"
    assert updater.asset_for(rel, "0.5.0", "1.21.10")["name"] == "pymcplaytime-0.5.0+1.21.10.jar"
    assert updater.asset_for(rel, "0.5.0", "1.21.2") is None


@pytest.mark.parametrize(("system", "expected_tail"), [
    ("Windows", ("Roaming", ".minecraft", "mods")),
    ("Darwin", ("Library", "Application Support", "minecraft", "mods")),
    ("Linux", (".minecraft", "mods")),
])
def test_default_mods_dir(monkeypatch, tmp_path, system, expected_tail):
    monkeypatch.setattr(updater.platform, "system", lambda: system)
    monkeypatch.setattr(updater.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    assert updater.default_mods_dir().parts[-len(expected_tail):] == expected_tail


def test_cli_parses_and_rejects_conflicts():
    args = updater.build_parser().parse_args(["--check", "--mod-only", "--mc", "26.3", "--pre"])
    assert args.check and args.mod_only and args.mc == "26.3" and args.pre
    with pytest.raises(SystemExit) as exc:
        updater.build_parser().parse_args(["--deps", "--no-deps"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------- zip install


def test_apply_zip_keeps_user_files_and_backs_up_edits(tmp_path):
    (tmp_path / "pymc_bot").mkdir()
    (tmp_path / "pymc_bot" / "agent.py").write_text("my local edit\n")
    (tmp_path / "README.md").write_bytes(b"same\n")
    (tmp_path / "CHANGES.md").write_bytes(b"one\r\ntwo\r\n")  # only the line endings differ
    (tmp_path / "pymc_bot_config.json").write_text('{"host": "mine"}')
    (tmp_path / "models" / "run1").mkdir(parents=True)
    (tmp_path / "models" / "run1" / "model.json").write_text("weights")
    (tmp_path / "notes.txt").write_text("not in the repo")

    archive = make_zip({
        "README.md": "same\n",
        "CHANGES.md": "one\ntwo\n",
        "pymc_bot/agent.py": "new upstream code\n",
        "pymc_bot/new_module.py": "x = 1\n",
        "pymc_bot_config.json": '{"host": "upstream"}',
        "models/run1/model.json": "upstream",
        "updater.sh": "#!/bin/sh\n",
    }, mode=0o755)
    report = updater.apply_zip(archive, tmp_path, stamp="T1")

    assert (tmp_path / "pymc_bot" / "agent.py").read_text() == "new upstream code\n"
    assert (tmp_path / "pymc_bot" / "new_module.py").read_text() == "x = 1\n"
    assert (tmp_path / ".pymc_backup" / "T1" / "pymc_bot" / "agent.py").read_text() == "my local edit\n"
    assert (tmp_path / "pymc_bot_config.json").read_text() == '{"host": "mine"}'
    assert (tmp_path / "models" / "run1" / "model.json").read_text() == "weights"
    assert (tmp_path / "notes.txt").read_text() == "not in the repo"
    assert report["backed_up"] == ["pymc_bot/agent.py"]
    assert "README.md" not in report["written"]  # unchanged files are not rewritten
    assert "CHANGES.md" in report["written"]  # CRLF-only difference: replaced, but no backup
    assert (tmp_path / "CHANGES.md").read_bytes() == b"one\ntwo\n"
    assert set(report["skipped"]) == {"pymc_bot_config.json", "models/run1/model.json"}
    if sys.platform != "win32":
        assert (tmp_path / "updater.sh").stat().st_mode & 0o111


@pytest.mark.parametrize("evil", ["../escape.txt", "a/../../escape.txt"])
def test_apply_zip_rejects_path_traversal(tmp_path, evil):
    target = tmp_path / "repo"
    target.mkdir()
    with pytest.raises(updater.UpdateError):
        updater.apply_zip(make_zip({evil: "boom"}), target)
    assert not (tmp_path / "escape.txt").exists()


def test_update_zip_uses_marker(monkeypatch, tmp_path):
    archive = make_zip({"pymc_bot/__init__.py": '__version__ = "9.9.9"\n'})
    calls = []
    monkeypatch.setattr(updater, "remote_commit",
                        lambda repo, branch: {"sha": "f" * 40, "date": "2026-10-16T00:00:00Z", "message": "m"})
    monkeypatch.setattr(updater, "http_get", lambda url, **kw: calls.append(url) or archive)

    assert updater.update_zip(tmp_path, "codero-sus/PyMC_Bot", "main", check=True) is False
    assert calls == []
    assert updater.update_zip(tmp_path, "codero-sus/PyMC_Bot", "main", check=False) is True
    assert calls == [f"{updater.CODELOAD}/codero-sus/PyMC_Bot/zip/{'f' * 40}"]
    assert updater.local_version(tmp_path) == "9.9.9"
    assert json.loads((tmp_path / updater.STATE_FILE).read_text())["sha"] == "f" * 40
    assert updater.update_zip(tmp_path, "codero-sus/PyMC_Bot", "main", check=False) is False
    assert len(calls) == 1


# ---------------------------------------------------------------------------- git checkout


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def git_pair(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    remote, upstream, local = tmp_path / "remote.git", tmp_path / "upstream", tmp_path / "local"
    git("init", "-q", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    git("clone", "-q", str(remote), str(upstream), cwd=tmp_path)
    git("config", "user.email", "t@example.com", cwd=upstream)
    git("config", "user.name", "t", cwd=upstream)
    (upstream / "requirements.txt").write_text("fastapi\n")
    (upstream / "app.py").write_text("v = 1\n")
    git("add", "-A", cwd=upstream)
    git("commit", "-q", "-m", "first", cwd=upstream)
    git("push", "-q", "origin", "HEAD:main", cwd=upstream)
    git("clone", "-q", str(remote), str(local), cwd=tmp_path)
    git("config", "user.email", "t@example.com", cwd=local)
    git("config", "user.name", "t", cwd=local)

    def publish(name: str, text: str, message: str = "change") -> None:
        (upstream / name).write_text(text)
        git("add", "-A", cwd=upstream)
        git("commit", "-q", "-m", message, cwd=upstream)
        git("push", "-q", "origin", "HEAD:main", cwd=upstream)

    return local, publish


def test_git_update_fast_forwards(git_pair):
    local, publish = git_pair
    assert updater.is_git_checkout(local)
    assert updater.update_git(local, check=False, autostash=False) is False
    publish("app.py", "v = 2\n", "bump app")
    assert updater.update_git(local, check=True, autostash=False) is False
    assert (local / "app.py").read_text() == "v = 1\n"
    assert updater.update_git(local, check=False, autostash=False) is True
    assert (local / "app.py").read_text() == "v = 2\n"


def test_git_update_refuses_dirty_tree_unless_autostash(git_pair):
    local, publish = git_pair
    publish("app.py", "v = 2\n")
    (local / "requirements.txt").write_text("fastapi\nmy-extra\n")
    with pytest.raises(updater.UpdateError, match="--autostash"):
        updater.update_git(local, check=False, autostash=False)
    assert (local / "app.py").read_text() == "v = 1\n"
    assert updater.update_git(local, check=False, autostash=True) is True
    assert (local / "app.py").read_text() == "v = 2\n"
    assert (local / "requirements.txt").read_text() == "fastapi\nmy-extra\n"


def test_git_update_refuses_diverged_branch(git_pair):
    local, publish = git_pair
    publish("app.py", "v = 2\n")
    (local / "mine.txt").write_text("local work\n")
    git("add", "-A", cwd=local)
    git("commit", "-q", "-m", "local", cwd=local)
    with pytest.raises(updater.UpdateError, match="local commit"):
        updater.update_git(local, check=False, autostash=False)


def test_main_reinstalls_deps_only_when_requirements_change(git_pair, monkeypatch):
    local, publish = git_pair
    seen = []
    monkeypatch.setattr(updater, "update_dependencies", lambda root, before, force, check: seen.append(
        [n for n in updater.DEPENDENCY_FILES if before.get(n) != updater.file_hashes(root).get(n)]))
    assert updater.main([], root=local) == 0
    assert seen == []  # nothing pulled, nothing installed
    publish("requirements.txt", "fastapi\nhttpx\n")
    assert updater.main([], root=local) == 0
    assert seen == [["requirements.txt"]]
    assert updater.main(["--no-deps"], root=local) == 0
    assert len(seen) == 1


def test_update_dependencies_uses_venv(tmp_path, monkeypatch):
    before = updater.file_hashes(tmp_path)
    (tmp_path / "requirements.txt").write_text("fastapi\n")
    py = tmp_path / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    calls = []
    monkeypatch.setattr(updater, "venv_python", lambda root: py)
    monkeypatch.setattr(updater.subprocess, "run",
                        lambda cmd, cwd=None, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    updater.update_dependencies(tmp_path, before, force=False, check=True)
    assert calls == []
    updater.update_dependencies(tmp_path, before, force=False, check=False)
    assert calls == [[str(py), "-m", "pip", "install", "--quiet", "-r", "requirements.txt"]]


# ---------------------------------------------------------------------------- mod


@pytest.fixture
def fake_github(monkeypatch):
    state = {"releases": [release("pymcplaytime-v0.5.0", ["1.21.4", "26.3"], prerelease=True)], "downloads": []}

    def http_json(url):
        assert "/releases" in url
        return state["releases"]

    def http_get(url, **kw):
        state["downloads"].append(url)
        for rel in state["releases"]:
            for asset in rel["assets"]:
                if asset["browser_download_url"] == url:
                    return state.get("tamper", asset["_data"])
        raise AssertionError(url)

    monkeypatch.setattr(updater, "http_json", http_json)
    monkeypatch.setattr(updater, "http_get", http_get)
    return state


def test_mod_update_replaces_old_jar(tmp_path, fake_github):
    mods = tmp_path / "mods"
    mods.mkdir()
    (mods / "pymcplaytime-0.4.0+1.21.4.jar").write_bytes(mod_jar("1.21.4"))
    (mods / "fabric-api-0.119.0+1.21.4.jar").write_bytes(b"api")

    assert updater.update_mod(mods, updater.REPO, None, allow_pre=False, check=True) is False
    assert fake_github["downloads"] == []
    assert updater.update_mod(mods, updater.REPO, None, allow_pre=False, check=False) is True
    names = sorted(p.name for p in mods.iterdir())
    assert names == ["fabric-api-0.119.0+1.21.4.jar", "pymcplaytime-0.4.0+1.21.4.jar.old",
                     "pymcplaytime-0.5.0+1.21.4.jar"]
    assert updater.update_mod(mods, updater.REPO, None, allow_pre=False, check=False) is False


def test_mod_install_with_mc_and_missing_version(tmp_path, fake_github):
    mods = tmp_path / "mods"
    assert updater.update_mod(mods, updater.REPO, "26.3", allow_pre=False, check=False) is True
    assert (mods / "pymcplaytime-0.5.0+26.3.jar").is_file()
    assert updater.update_mod(mods, updater.REPO, "1.20.1", allow_pre=False, check=False) is False
    with pytest.raises(updater.UpdateError, match="--mc"):
        updater.update_mod(tmp_path / "empty", updater.REPO, None, allow_pre=False, check=False)


def test_mod_update_rejects_bad_download(tmp_path, fake_github):
    mods = tmp_path / "mods"
    mods.mkdir()
    old = mods / "pymcplaytime-0.4.0+26.3.jar"
    old.write_bytes(b"old")
    fake_github["tamper"] = mod_jar("1.21.4")  # wrong jar
    with pytest.raises(updater.UpdateError):
        updater.update_mod(mods, updater.REPO, None, allow_pre=False, check=False)
    assert old.read_bytes() == b"old"
    assert sorted(p.name for p in mods.iterdir()) == [old.name]


def test_verify_jar_checks_metadata():
    data = mod_jar("26.3", mod_id="something-else")
    asset = {"size": len(data), "digest": "sha256:" + hashlib.sha256(data).hexdigest()}
    with pytest.raises(updater.UpdateError, match="not the recorder"):
        updater.verify_jar(data, asset, "26.3")
    with pytest.raises(updater.UpdateError, match="not a valid mod jar"):
        updater.verify_jar(b"junk", {"size": 4}, "26.3")
    good = mod_jar("26.3")
    updater.verify_jar(good, {"size": len(good), "digest": "sha256:" + hashlib.sha256(good).hexdigest()}, "26.3")


def test_install_jar_reports_locked_file(tmp_path, monkeypatch):
    mods = tmp_path / "mods"
    mods.mkdir()
    old = mods / "pymcplaytime-0.4.0+26.3.jar"
    old.write_bytes(b"old")

    def locked(self, target):
        raise PermissionError(13, "in use", str(self))

    monkeypatch.setattr(updater.Path, "rename", locked)
    with pytest.raises(updater.UpdateError, match="close Minecraft"):
        updater.install_jar(mods, b"new", "pymcplaytime-0.5.0+26.3.jar", [old])
    assert not any(p.name.endswith(".part") for p in mods.iterdir())


# ---------------------------------------------------------------------------- wrappers


def test_wrappers_exist_and_hand_over_safely():
    sh = (ROOT / "updater.sh").read_text()
    assert sh.startswith("#!/bin/sh")
    assert 'exec "$PY" updater.py "$@"' in sh
    cmd_raw = (ROOT / "updater.cmd").read_bytes()
    assert b"\r\n" in cmd_raw or b"\n" in cmd_raw
    cmd = cmd_raw.decode("ascii")
    launches = [line for line in cmd.splitlines() if "updater.py %*" in line]
    assert len(launches) == 3 and all(line.rstrip().endswith("& exit /b") for line in launches)
    assert "py -3 updater.py" in cmd and ".venv\\Scripts\\python.exe" in cmd


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("sh") is None, reason="no POSIX shell")
def test_updater_sh_syntax_and_help():
    subprocess.run(["sh", "-n", str(ROOT / "updater.sh")], check=True)
    out = subprocess.run(["sh", str(ROOT / "updater.sh"), "--help"], capture_output=True, text=True, check=True)
    assert "--mod-only" in out.stdout


def test_updater_output_is_ascii():
    (ROOT / "updater.py").read_text(encoding="ascii")
    (ROOT / "updater.cmd").read_text(encoding="ascii")
    (ROOT / "updater.sh").read_text(encoding="ascii")
