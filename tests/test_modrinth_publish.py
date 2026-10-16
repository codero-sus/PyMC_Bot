"""Modrinth publishing: the upload script, the project page and the icon."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import modrinth_publish as mp  # noqa: E402


def _jars(folder: Path, names: list[str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"PK\x03\x04jar")
    return folder


def test_find_jars_picks_only_release_jars_of_this_version(tmp_path: Path):
    folder = _jars(tmp_path, ["pymcbot-0.7.0+1.21.4.jar", "pymcbot-0.7.0+26.3.jar", "pymcbot-0.6.0+26.3.jar",
                              "pymcbot-gametest-0.7.0+26.3.jar", "pymcbot-0.7.0+26.3-sources.jar", "other.jar"])
    assert sorted(mp.find_jars(folder, "0.7.0")) == ["1.21.4", "26.3"]
    assert mp.find_jars(folder, "0.6.0") == {"26.3": folder / "pymcbot-0.6.0+26.3.jar"}


def test_version_payload_matches_modrinths_create_version_api():
    data = mp.version_payload("pymc-bot", "0.7.0", "26.3", channel="beta", changelog="hi", featured=True)
    assert data["version_number"] == "0.7.0+26.3"
    assert data["game_versions"] == ["26.3"] and data["loaders"] == ["fabric"]
    assert data["environment"] == "client_only"
    assert data["version_type"] == "beta" and data["featured"] is True
    assert {"project_id": "P7dR8mSH", "dependency_type": "required"} in data["dependencies"]  # Fabric API
    assert {"project_id": "mOgUt4GM", "dependency_type": "optional"} in data["dependencies"]  # Mod Menu
    assert data["file_parts"] == ["file"] and data["primary_file"] == "file"
    assert "PyMC Bot" in data["name"]


def test_multipart_body_carries_data_and_jar(tmp_path: Path):
    jar = _jars(tmp_path, ["pymcbot-0.7.0+26.3.jar"]) / "pymcbot-0.7.0+26.3.jar"
    body, content_type = mp.multipart({"a": 1}, jar)
    boundary = content_type.split("boundary=", 1)[1]
    assert content_type.startswith("multipart/form-data")
    assert body.count(f"--{boundary}".encode()) == 3
    assert b'name="data"' in body and json.dumps({"a": 1}).encode() in body
    assert b'filename="pymcbot-0.7.0+26.3.jar"' in body and b"PK\x03\x04jar" in body


def test_changelog_section_for_the_current_version():
    version = mp.mod_version()
    text = mp.changelog_for(version)
    assert text and not text.startswith("##")
    assert mp.changelog_for("0.0.0-nope") == ""


def test_dry_run_needs_no_token(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv("MODRINTH_TOKEN", raising=False)
    version = mp.mod_version()
    folder = _jars(tmp_path, [f"pymcbot-{version}+26.3.jar", f"pymcbot-{version}+1.21.1.jar"])
    assert mp.main(["--jars", str(folder), "--dry-run", "--update-page"]) == 0
    out = capsys.readouterr().out
    assert f"{version}+1.21.1" in out and f"{version}+26.3" in out
    assert out.index("1.21.1") < out.index("26.3")  # oldest first, newest last (featured)
    assert "would update the project page" in out


def test_real_upload_requires_project_and_token(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MODRINTH_TOKEN", raising=False)
    monkeypatch.delenv("MODRINTH_PROJECT", raising=False)
    folder = _jars(tmp_path, [f"pymcbot-{mp.mod_version()}+26.3.jar"])
    assert mp.main(["--jars", str(folder)]) == 2
    assert mp.main(["--jars", str(folder), "--project", "pymc-bot"]) == 2
    assert mp.main(["--jars", str(tmp_path / "empty")]) == 1


def test_already_published_versions_are_skipped(tmp_path: Path, monkeypatch, capsys):
    version = mp.mod_version()
    folder = _jars(tmp_path, [f"pymcbot-{version}+26.3.jar", f"pymcbot-{version}+26.2.jar"])
    calls: list[tuple[str, str]] = []

    def fake_request(method, path, token, body=None, content_type="application/json"):
        calls.append((method, path))
        if method == "GET":
            return [{"version_number": f"{version}+26.3"}]
        return {"id": "abc"}

    monkeypatch.setenv("MODRINTH_TOKEN", "mrp_test")
    monkeypatch.setattr(mp, "request", fake_request)
    assert mp.main(["--jars", str(folder), "--project", "pymc-bot"]) == 0
    assert calls == [("GET", "/project/pymc-bot/version"), ("POST", "/version")]
    assert "already on Modrinth" in capsys.readouterr().out


# ----------------------------------------------------------------- page + icon
def test_modrinth_page_is_about_pymc_bot_and_honest_about_privacy():
    page = (ROOT / "mod/MODRINTH.md").read_text(encoding="utf-8")
    assert page.startswith("# PyMC Bot")
    assert "1.21.1 to 26.3" in page
    assert "no network" in page.lower()
    assert "pymcbot/dataset.jsonl" in page
    assert "Playtime Recorder" not in page
    assert "codero-sus" in page and "MIT" in page
    assert "Cortex" not in page


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


@pytest.mark.parametrize(("path", "size"), [("mod/src/main/resources/assets/pymcbot/icon.png", 128),
                                            ("docs/modrinth-icon.png", 512)])
def test_icons_fit_modrinth(path: str, size: int):
    icon = ROOT / path
    assert _png_size(icon) == (size, size)
    assert icon.stat().st_size < 256 * 1024  # Modrinth's icon limit
