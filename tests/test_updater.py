"""Tests for updater.py - all offline (local zips, a fake GitHub, fake releases).

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


def make_zip(files: dict[str, bytes | str], top: str = "PyMC_Bot-abc123", mode: int | None = None,
             commit: str = "") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.comment = commit.encode("ascii")
        zf.writestr(f"{top}/", "")
        for name, data in files.items():
            info = zipfile.ZipInfo(f"{top}/{name}")
            if mode is not None and name.endswith(".sh"):
                info.external_attr = (0o100000 | mode) << 16
            zf.writestr(info, data)
    return buf.getvalue()


def mod_jar(mc: str, mod_id: str = "pymcbot") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("fabric.mod.json", json.dumps({"id": mod_id, "depends": {"minecraft": mc}}))
    return buf.getvalue()


def release(tag: str, mcs: list[str], prerelease: bool = False, version: str | None = None) -> dict:
    version = version or updater.release_version(tag)
    mod_id = tag.split("-v", 1)[0]
    assets = []
    for mc in mcs:
        data = mod_jar(mc, mod_id)
        assets.append({
            "name": f"{mod_id}-{version}+{mc}.jar",
            "size": len(data),
            "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
            "browser_download_url": f"https://example.invalid/{version}/{mc}.jar",
            "_data": data,
        })
    return {"tag_name": tag, "prerelease": prerelease, "draft": False, "assets": assets}


# ---------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(("name", "expected"), [
    ("pymcbot-0.6.0+1.21.4.jar", ("0.6.0", "1.21.4")),
    ("pymcbot-0.6.0+26.3.jar", ("0.6.0", "26.3")),
    ("pymcbot-gametest-0.6.0+26.3.jar", None),
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
        release("pymcbot-v0.6.0", ["1.21.4"], prerelease=True),
        release("pymcbot-v0.5.0", ["1.21.4"]),
        release("pymcbot-v0.10.0", ["1.21.4"], prerelease=True),
        {"tag_name": "v9.9.9", "prerelease": False, "draft": False, "assets": []},
        {"tag_name": "pymcbot-v9.0.0", "prerelease": False, "draft": True, "assets": []},
    ]
    assert updater.pick_release(rels, allow_pre=False)["tag_name"] == "pymcbot-v0.5.0"
    assert updater.pick_release(rels, allow_pre=True)["tag_name"] == "pymcbot-v0.10.0"
    only_pre = [release("pymcbot-v0.5.0", ["26.3"], prerelease=True)]
    assert updater.pick_release(only_pre, allow_pre=False)["tag_name"] == "pymcbot-v0.5.0"
    assert updater.pick_release([], allow_pre=True) is None


def test_release_version_knows_both_names():
    assert updater.release_version("pymcbot-v0.6.0") == "0.6.0"
    assert updater.release_version("pymcplaytime-v0.5.0") == "0.5.0"
    assert updater.release_version("v1.0.0") is None
    rels = [release("pymcplaytime-v0.5.0", ["26.3"]), release("pymcbot-v0.6.0", ["26.3"])]
    assert updater.pick_release(rels, allow_pre=False)["tag_name"] == "pymcbot-v0.6.0"
    old_only = [release("pymcplaytime-v0.5.0", ["26.3"], prerelease=True)]
    assert updater.asset_for(old_only[0], "0.5.0", "26.3")["name"] == "pymcplaytime-0.5.0+26.3.jar"


def test_asset_for_matches_exact_version():
    rel = release("pymcbot-v0.5.0", ["1.21.1", "1.21.10"])
    assert updater.asset_for(rel, "0.5.0", "1.21.1")["name"] == "pymcbot-0.5.0+1.21.1.jar"
    assert updater.asset_for(rel, "0.5.0", "1.21.10")["name"] == "pymcbot-0.5.0+1.21.10.jar"
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


class FakeGitHub:
    """commits API + codeload zips for one repository, with an optional exhausted API limit."""

    def __init__(self, monkeypatch, ref: str):
        self.ref = ref
        self.commits: list[tuple[str, dict]] = []
        self.calls: list[str] = []
        self.rate_limited = False
        monkeypatch.setattr(updater, "http_get", self.http_get)
        monkeypatch.setattr(updater, "http_json", lambda url: json.loads(self.http_get(url)))

    def publish(self, files: dict[str, str], message: str = "change") -> str:
        sha = hashlib.sha1(f"{len(self.commits)}{message}".encode()).hexdigest()
        self.commits.append((sha, dict(files)))
        return sha

    def http_get(self, url: str, accept: str = "", timeout: float = 0) -> bytes:
        self.calls.append(url)
        sha, files = self.commits[-1]
        if url.startswith(updater.API):
            if self.rate_limited:
                raise updater.RateLimited("rate limit")
            assert url.endswith("/commits/" + updater.urllib.parse.quote(self.ref, safe=""))
            return json.dumps({"sha": sha, "commit": {"committer": {"date": "2026-10-19T00:00:00Z"},
                                                      "message": "msg\nbody"}}).encode()
        wanted = url.rsplit("/zip/", 1)[1]
        match = next((f for s, f in self.commits if s == wanted), None)
        if match is None:  # refs/heads/<ref>: the newest commit
            assert wanted == "refs/heads/" + updater.urllib.parse.quote(self.ref, safe="/")
            wanted, match = sha, files
        return make_zip(match, top="PyMC_Bot-" + self.ref.replace("/", "-"), commit=wanted)


def test_update_zip_uses_marker_and_downloads_by_commit(monkeypatch, tmp_path):
    hub = FakeGitHub(monkeypatch, "feature/x")
    sha = hub.publish({"pymc_bot/__init__.py": '__version__ = "9.9.9"\n'})
    assert updater.update_zip(tmp_path, "codero-sus/PyMC_Bot", "feature/x", check=True) is False
    assert not any("/zip/" in call for call in hub.calls)
    assert updater.update_zip(tmp_path, "codero-sus/PyMC_Bot", "feature/x", check=False) is True
    assert hub.calls[-1] == f"{updater.CODELOAD}/codero-sus/PyMC_Bot/zip/{sha}"
    assert updater.local_version(tmp_path) == "9.9.9"
    state = json.loads((tmp_path / updater.STATE_FILE).read_text())
    assert state["sha"] == sha and state["channel"] == "custom" and "pymc_bot/__init__.py" in state["files"]
    assert "feature" not in json.dumps(state), "the state file never stores the branch"
    downloads = len([c for c in hub.calls if "/zip/" in c])
    assert updater.update_zip(tmp_path, "codero-sus/PyMC_Bot", "feature/x", check=False) is False
    assert len([c for c in hub.calls if "/zip/" in c]) == downloads


def test_update_zip_removes_files_deleted_upstream_unless_edited(monkeypatch, tmp_path):
    hub = FakeGitHub(monkeypatch, "b")
    hub.publish({"keep.py": "1", "gone.py": "old", "edited.py": "orig", "pkg/only.py": "x"})
    assert updater.update_zip(tmp_path, "r/r", "b", check=False)
    (tmp_path / "edited.py").write_text("my change")
    (tmp_path / "mine.txt").write_text("not from the project")
    hub.publish({"keep.py": "2"})
    assert updater.update_zip(tmp_path, "r/r", "b", check=False)
    assert (tmp_path / "keep.py").read_text() == "2"
    assert not (tmp_path / "gone.py").exists() and not (tmp_path / "pkg").exists()
    assert (tmp_path / "edited.py").read_text() == "my change"
    assert (tmp_path / "mine.txt").exists()


def test_update_zip_falls_back_to_the_zip_comment_when_rate_limited(monkeypatch, tmp_path):
    hub = FakeGitHub(monkeypatch, "b")
    sha = hub.publish({"a.py": "1"})
    hub.rate_limited = True
    assert updater.update_zip(tmp_path, "r/r", "b", check=False) is True
    assert json.loads((tmp_path / updater.STATE_FILE).read_text())["sha"] == sha
    assert updater.update_zip(tmp_path, "r/r", "b", check=False) is False


def test_download_archive_rejects_the_wrong_commit(monkeypatch):
    monkeypatch.setattr(updater, "http_get", lambda url, **kw: make_zip({"a": "1"}, commit="1" * 40))
    with pytest.raises(updater.UpdateError, match="expected commit"):
        updater.download_archive("r/r", "b", "2" * 40)
    monkeypatch.setattr(updater, "http_get", lambda url, **kw: b"not a zip")
    with pytest.raises(updater.UpdateError, match="zip"):
        updater.download_archive("r/r", "b", "2" * 40)


def test_git_folder_is_never_touched_or_used(monkeypatch, tmp_path, capsys):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    hub = FakeGitHub(monkeypatch, "b")
    hub.publish({"a.py": "1", ".git/HEAD": "evil"})
    monkeypatch.setattr(updater.subprocess, "run", lambda *a, **k: pytest.fail("no subprocess (no git)"))
    assert updater.update_zip(tmp_path, "r/r", "b", check=False)
    assert (tmp_path / ".git" / "HEAD").read_text() == "ref: refs/heads/main\n"
    assert "it is not used" in capsys.readouterr().out


def test_main_reinstalls_deps_only_when_requirements_change(monkeypatch, tmp_path):
    hub = FakeGitHub(monkeypatch, updater.update_channel())
    hub.publish({"requirements.txt": "fastapi\n", "app.py": "v = 1\n"})
    seen = []
    monkeypatch.setattr(updater, "update_dependencies", lambda root, before, force, check: seen.append(
        [n for n in updater.DEPENDENCY_FILES if before.get(n) != updater.file_hashes(root).get(n)]))
    assert updater.main(["--no-deps"], root=tmp_path) == 0
    assert seen == []
    assert updater.main([], root=tmp_path) == 0
    assert seen == []  # nothing new, nothing installed
    hub.publish({"requirements.txt": "fastapi\nhttpx\n", "app.py": "v = 1\n"})
    assert updater.main([], root=tmp_path) == 0
    assert seen == [["requirements.txt"]]


# ---------------------------------------------------------------------------- update branch: main


def test_updates_come_from_main(monkeypatch, tmp_path, capsys):
    assert updater.update_channel() == updater.DEFAULT_BRANCH == "main"
    assert updater.retired_channel() != "main"
    hub = FakeGitHub(monkeypatch, "main")
    hub.publish({"a.py": "1"})
    assert updater.update_zip(tmp_path, "r/r", None, check=False)
    out = capsys.readouterr().out
    assert "checking the main branch on GitHub" in out and "<update channel>" not in out
    assert updater.read_state(tmp_path)["channel"] == "default"
    updater.say("python -m pymc_bot.__main__ on main")  # "main" is public: nothing is redacted
    assert "python -m pymc_bot.__main__ on main" in capsys.readouterr().out


# ---------------------------------------------------------------------------- retired hidden channel


def test_retired_channel_is_sealed_and_round_trips():
    channel = updater.retired_channel()
    assert channel and "/" not in updater._RETIRED_CHANNEL
    assert updater.unseal(updater.seal(channel)) == channel
    assert updater.unseal(updater.seal("some/other-branch")) == "some/other-branch"
    assert updater.seal(channel) == updater._RETIRED_CHANNEL


def test_channel_never_appears_in_plain_text_in_the_project():
    """Leak guard: no project file (code, docs, workflows, tests) contains the retired hidden channel."""
    forms = updater._secret_forms(updater.retired_channel())
    skip = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache"}
    leaks = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or skip & set(path.relative_to(ROOT).parts) or path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        leaks += [f"{path.relative_to(ROOT)}: {form[:3]}..." for form in forms if form in text]
    assert not leaks, leaks


def test_output_and_errors_never_show_the_channel(monkeypatch, capsys):
    channel = updater.retired_channel()
    updater.say(f"url https://api.github.com/repos/x/y/commits/{updater.urllib.parse.quote(channel, safe='')}")
    updater.say(f"folder PyMC_Bot-{channel.replace('/', '-')}/README.md and {channel}")
    out = capsys.readouterr().out
    assert all(form not in out for form in updater._secret_forms(channel))
    assert out.count("<update channel>") == 3

    def boom(url, accept="", timeout=0):
        raise updater.UpdateError(f"GitHub answered HTTP 500 for {updater.redact(url)}")
    monkeypatch.setattr(updater, "http_get", boom)
    monkeypatch.setattr(updater, "http_json", lambda url: boom(url))
    assert updater.main(["--check"], root=ROOT) == 1
    out = capsys.readouterr().out
    assert "ERROR" in out and all(form not in out for form in updater._secret_forms(channel))


def test_help_does_not_reveal_the_channel(capsys):
    with pytest.raises(SystemExit):
        updater.main(["--help"])
    out = capsys.readouterr().out
    assert "instead of main" in out and "--autostash" not in out and "--seal" not in out
    assert all(form not in out for form in updater._secret_forms(updater.retired_channel()))


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
    state = {"releases": [release("pymcbot-v0.5.0", ["1.21.4", "26.3"], prerelease=True)], "downloads": []}

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
    # the pre-0.6 "pymcplaytime" jar is replaced by the renamed "pymcbot" jar
    assert names == ["fabric-api-0.119.0+1.21.4.jar", "pymcbot-0.5.0+1.21.4.jar",
                     "pymcplaytime-0.4.0+1.21.4.jar.old"]
    assert updater.update_mod(mods, updater.REPO, None, allow_pre=False, check=False) is False


def test_mod_install_with_mc_and_missing_version(tmp_path, fake_github):
    mods = tmp_path / "mods"
    assert updater.update_mod(mods, updater.REPO, "26.3", allow_pre=False, check=False) is True
    assert (mods / "pymcbot-0.5.0+26.3.jar").is_file()
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
    with pytest.raises(updater.UpdateError, match="not the PyMC Bot mod"):
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
        updater.install_jar(mods, b"new", "pymcbot-0.5.0+26.3.jar", [old])
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


def test_a_closed_output_pipe_never_stops_the_updater(monkeypatch):
    """`updater.py --check | grep -q ...`: grep exits at the first match; the updater must carry on."""

    class ClosedPipe:
        def write(self, _text):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(updater.sys, "stdout", ClosedPipe())
    updater.say("up to date")
    updater.say("done")  # would have raised before
    assert not isinstance(updater.sys.stdout, ClosedPipe)


def test_http_get_retries_short_github_hiccups(monkeypatch):
    calls = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def flaky(request, timeout=0):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise updater.urllib.error.HTTPError(request.full_url, 502, "Bad Gateway", {}, io.BytesIO(b""))
        if len(calls) == 2:
            raise updater.urllib.error.URLError("connection reset")
        return Response(b"ok")

    monkeypatch.setattr(updater, "RETRY_DELAYS", (0, 0))
    monkeypatch.setattr(updater.urllib.request, "urlopen", flaky)
    assert updater.http_get("https://api.github.com/x") == b"ok" and len(calls) == 3

    def missing(request, timeout=0):
        calls.append("404")
        raise updater.urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, io.BytesIO(b""))

    calls.clear()
    monkeypatch.setattr(updater.urllib.request, "urlopen", missing)
    with pytest.raises(updater.UpdateError, match="not found"):
        updater.http_get("https://api.github.com/x")
    assert calls == ["404"], "a real answer (404) is never retried"

    def down(request, timeout=0):
        calls.append("down")
        raise updater.urllib.error.URLError("offline")

    calls.clear()
    monkeypatch.setattr(updater.urllib.request, "urlopen", down)
    with pytest.raises(updater.UpdateError, match="are you online"):
        updater.http_get("https://api.github.com/x")
    assert len(calls) == 3


def test_an_empty_release_list_is_asked_again(monkeypatch):
    answers = [[], [], [{"tag_name": "pymcbot-v1.2.0", "prerelease": False, "draft": False, "assets": []}]]
    asked = []
    monkeypatch.setattr(updater, "RETRY_DELAYS", (0, 0))
    monkeypatch.setattr(updater, "http_json", lambda url: asked.append(url) or answers[len(asked) - 1])
    assert updater.newest_mod_release("me/repo", allow_pre=True)["tag_name"] == "pymcbot-v1.2.0" and len(asked) == 3
    asked.clear()
    monkeypatch.setattr(updater, "http_json", lambda url: asked.append(url) or [{"tag_name": "other-v1", "assets": []}])
    assert updater.newest_mod_release("me/repo", allow_pre=True) is None and len(asked) == 1, \
        "a real list without mod releases is not asked again"
