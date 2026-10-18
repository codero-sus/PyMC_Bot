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
    assert mp.main(["--jars", str(folder), "--dry-run", "--update-project"]) == 0
    out = capsys.readouterr().out
    assert f"{version}+1.21.1" in out and f"{version}+26.3" in out
    assert out.index("1.21.1") < out.index("26.3")  # oldest first, newest last (featured)
    assert "would update the project" in out and "summary" in out


def test_real_upload_requires_project_and_token(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MODRINTH_TOKEN", raising=False)
    monkeypatch.delenv("MODRINTH_PROJECT", raising=False)
    folder = _jars(tmp_path, [f"pymcbot-{mp.mod_version()}+26.3.jar"])
    assert mp.main(["--jars", str(folder)]) == 2
    assert mp.main(["--jars", str(folder), "--project", "pymc-bot"]) == 2
    assert mp.main(["--jars", str(tmp_path / "empty")]) == 1


def _zip_jar(path: Path, files: dict[str, bytes], when: tuple = (2026, 1, 1, 0, 0, 0)) -> Path:
    import zipfile

    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(zipfile.ZipInfo(name, when), data)
    return path


class FakeModrinth:
    """Records API calls; GET /project/<p>/version answers with ``versions``."""

    def __init__(self, versions: list[dict]) -> None:
        self.versions = versions
        self.calls: list[tuple[str, str, object]] = []

    def __call__(self, method, path, token, body=None, content_type="application/json"):
        payload = json.loads(body) if body and content_type == "application/json" else body
        self.calls.append((method, path, payload))
        return self.versions if method == "GET" else None

    def paths(self) -> list[tuple[str, str]]:
        return [(method, path) for method, path, _ in self.calls]


def _remote(number: str, vid: str, jar: Path, featured: bool = False) -> dict:
    return {"id": vid, "version_number": number, "featured": featured,
            "files": [{"primary": True, "url": f"https://cdn.modrinth.com/{vid}.jar",
                       "hashes": {"sha1": mp.sha1(jar.read_bytes())}}]}


def test_every_run_creates_missing_updates_existing_and_replaces_changed_jars(tmp_path: Path):
    version = mp.mod_version()
    new = tmp_path / "new"
    new.mkdir()
    same = _zip_jar(new / f"pymcbot-{version}+26.1.jar", {"a.class": b"same"})
    changed = _zip_jar(new / f"pymcbot-{version}+26.2.jar", {"a.class": b"new code"})
    fresh = _zip_jar(new / f"pymcbot-{version}+26.3.jar", {"a.class": b"fresh"})
    # On Modrinth: 26.1 is the same code built at another time, 26.2 holds old code, plus an old featured release.
    old_same = _zip_jar(tmp_path / "old-same.jar", {"a.class": b"same"}, when=(2025, 5, 5, 5, 5, 6))
    old_changed = _zip_jar(tmp_path / "old-changed.jar", {"a.class": b"old code"})
    api = FakeModrinth([
        _remote(f"{version}+26.1", "V261", old_same),
        _remote(f"{version}+26.2", "V262", old_changed),
        _remote("0.6.0+26.1", "OLD", old_changed, featured=True),
    ])
    cdn = {"https://cdn.modrinth.com/V261.jar": old_same.read_bytes(),
           "https://cdn.modrinth.com/V262.jar": old_changed.read_bytes()}
    payloads = [(mp.version_payload("p", version, mc, featured=mc == "26.3"), jar)
                for mc, jar in (("26.1", same), ("26.2", changed), ("26.3", fresh))]

    counts = mp.sync_versions("p", payloads, "mrp_x", call=api, fetch=cdn.__getitem__)

    assert counts == {"created": 1, "updated": 1, "replaced": 1, "skipped": 0, "failed": 0}
    assert api.paths() == [
        ("GET", "/project/p/version"),
        ("PATCH", "/version/V261"),                       # details only - same classes, other timestamps
        ("POST", "/version/V262/file"),                   # new jar ...
        ("PATCH", "/version/V262"),                       # ... made primary, details refreshed
        ("DELETE", f"/version_file/{mp.sha1(old_changed.read_bytes())}?algorithm=sha1&version_id=V262"),
        ("POST", "/version"),                             # 26.3 did not exist yet
        ("PATCH", "/version/OLD"),                        # older release loses "featured"
    ]
    details = api.calls[1][2]
    assert details["name"] == f"PyMC Bot {version} for Minecraft 26.1" and details["game_versions"] == ["26.1"]
    assert details["dependencies"][0]["dependency_type"] == "required" and "changelog" in details
    assert api.calls[3][2]["primary_file"] == ["sha1", mp.sha1(changed.read_bytes())]
    assert api.calls[-1][2] == {"featured": False}


def test_a_jar_already_uploaded_under_another_number_is_not_sent_twice(tmp_path: Path, capsys):
    version = mp.mod_version()
    jar = _zip_jar(tmp_path / f"pymcbot-{version}+26.3.jar", {"a.class": b"x"})
    api = FakeModrinth([_remote("0.7.0-manual", "MAN", jar)])
    counts = mp.sync_versions("p", [(mp.version_payload("p", version, "26.3"), jar)], "mrp_x", call=api)
    assert counts["skipped"] == 1 and api.paths() == [("GET", "/project/p/version")]
    assert "already on Modrinth as 0.7.0-manual" in capsys.readouterr().out


def test_a_hand_made_version_holding_several_jars_is_untangled(tmp_path: Path, capsys):
    """Like the real project: 0.7.0+1.21.1 made by hand, with its jar not primary and a 26.3 jar inside."""
    version = mp.mod_version()
    jar_1211 = _zip_jar(tmp_path / f"pymcbot-{version}+1.21.1.jar", {"a.class": b"1.21.1"})
    jar_263 = _zip_jar(tmp_path / f"pymcbot-{version}+26.3.jar", {"a.class": b"26.3"})
    hand = _zip_jar(tmp_path / "hand.jar", {"a.class": b"older 1.21.1 build"})
    api = FakeModrinth([{"id": "HAND", "version_number": f"{version}+1.21.1", "version_type": "beta", "files": [
        {"filename": f"pymcbot-{version}+1.21.1.jar", "primary": True, "url": "u", "hashes": {"sha1": mp.sha1(hand.read_bytes())}},
        {"filename": f"pymcbot-{version}+1.21.1.jar", "primary": False, "url": "u", "hashes": {"sha1": mp.sha1(jar_1211.read_bytes())}},
        {"filename": f"pymcbot-{version}+26.3.jar", "primary": False, "url": "u", "hashes": {"sha1": mp.sha1(jar_263.read_bytes())}},
    ]}])
    payloads = [(mp.version_payload("p", version, mc, featured=mc == "26.3"), jar)
                for mc, jar in (("1.21.1", jar_1211), ("26.3", jar_263))]

    counts = mp.sync_versions("p", payloads, "mrp_x", call=api, fetch=lambda url: b"")

    assert counts == {"created": 1, "updated": 1, "replaced": 0, "skipped": 0, "failed": 0}
    assert api.paths() == [
        ("GET", "/project/p/version"),
        ("DELETE", f"/version_file/{mp.sha1(jar_263.read_bytes())}?algorithm=sha1&version_id=HAND"),  # misplaced
        ("PATCH", "/version/HAND"),                                                                   # no re-upload
        ("DELETE", f"/version_file/{mp.sha1(hand.read_bytes())}?algorithm=sha1&version_id=HAND"),     # older build
        ("POST", "/version"),                                                                         # 26.3 gets its own
    ]
    patch = api.calls[2][2]
    assert patch["primary_file"] == ["sha1", mp.sha1(jar_1211.read_bytes())] and patch["version_type"] == "release"
    assert "Modrinth has 1 version(s)" in capsys.readouterr().out


def test_a_replacement_jar_never_reuses_the_name_of_the_file_it_replaces(tmp_path: Path):
    """The real failure: version 0.7.0+1.21.1 held pymcbot-0.7.0+1.21.1.jar (a different build) and
    Modrinth refused a second file of that name. The new jar goes up as -<sha7>, then the old one goes."""
    version = mp.mod_version()
    jar = _zip_jar(tmp_path / f"pymcbot-{version}+1.21.1.jar", {"a.class": b"new"})
    old = _zip_jar(tmp_path / "old.jar", {"a.class": b"old"})
    api = FakeModrinth([{"id": "V", "version_number": f"{version}+1.21.1", "files": [
        {"filename": jar.name, "primary": True, "url": "u", "hashes": {"sha1": mp.sha1(old.read_bytes())}}]}])
    counts = mp.sync_versions("p", [(mp.version_payload("p", version, "1.21.1"), jar)], "mrp_x",
                              call=api, fetch=lambda url: old.read_bytes())
    assert counts["replaced"] == 1 and counts["failed"] == 0
    upload = api.calls[1][2]
    new_name = f"pymcbot-{version}+1.21.1-{mp.sha1(jar.read_bytes())[:7]}.jar"
    assert f'filename="{new_name}"'.encode() in upload
    assert api.paths()[2:] == [("PATCH", "/version/V"),
                               ("DELETE", f"/version_file/{mp.sha1(old.read_bytes())}?algorithm=sha1&version_id=V")]
    # Next time the plain name is free again, and the -<sha7> file is still recognised as the 1.21.1 jar.
    assert mp.upload_name(jar, {new_name}, "f" * 40) == jar.name
    assert mp._jar_mc(new_name) == "1.21.1" and mp._jar_mc(f"pymcbot-{version}+26.1.2.jar") == "26.1.2"
    assert mp._jar_mc(f"pymcbot-{version}+1.21.1-sources.jar") is None


def test_one_refused_version_does_not_stop_the_others_but_fails_the_run(tmp_path: Path, monkeypatch, capsys):
    version = mp.mod_version()
    _zip_jar(tmp_path / f"pymcbot-{version}+26.2.jar", {"a.class": b"x"})
    _zip_jar(tmp_path / f"pymcbot-{version}+26.3.jar", {"a.class": b"y"})
    seen = []

    def call(method, path, token, body=None, content_type="application/json"):
        seen.append((method, path))
        if method == "GET":
            return []
        if method == "POST" and len([m for m, _ in seen if m == "POST"]) == 1:
            raise SystemExit('Modrinth POST /version failed: HTTP 400 {"description":"Duplicate files are not allowed"}')
        return None

    monkeypatch.setenv("MODRINTH_TOKEN", "mrp_test")
    monkeypatch.setattr(mp, "request", call)
    assert mp.main(["--jars", str(tmp_path), "--project", "p"]) == 1
    assert [m for m, _ in seen].count("POST") == 2  # the second version was still created
    out = capsys.readouterr().out
    assert "failed: Modrinth POST /version failed: HTTP 400" in out and "1 created" in out and "1 failed" in out


def test_ci_turns_red_when_publishing_fails():
    job = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8").split("\n  modrinth:\n", 1)[1]
    assert "status=${PIPESTATUS[0]}" in job and 'exit "$status"' in job  # tee must not hide the exit code


def test_project_settings_sync_title_summary_description_links_and_licence():
    settings = mp.project_settings()
    versions = mp.supported_versions()
    assert settings["title"] == "PyMC Bot"
    assert f"from {versions[0]} to {versions[-1]}" in settings["description"] and "{" not in settings["description"]
    assert len(settings["description"]) <= 256
    assert settings["body"] == (ROOT / "mod/MODRINTH.md").read_text(encoding="utf-8")
    assert settings["license_id"] == "PolyForm-Noncommercial-1.0.0"
    assert settings["source_url"] == "https://github.com/codero-sus/PyMC_Bot"
    listing = (ROOT / "docs/modrinth/LISTING.md").read_text(encoding="utf-8")
    assert settings["description"] in listing  # the copy-paste sheet shows the same summary


def test_a_rejected_optional_project_field_still_updates_the_text():
    calls = []

    def call(method, path, token, body=None, content_type="application/json"):
        data = json.loads(body)
        calls.append(sorted(data))
        if "categories" in data:
            raise SystemExit("Modrinth PATCH /project/p failed: HTTP 400 bad category")

    mp.update_project("p", "mrp_x", mp.project_settings(), call=call)
    assert calls[-1] == ["body", "description", "title"]


def test_main_updates_the_project_before_the_versions(tmp_path: Path, monkeypatch):
    version = mp.mod_version()
    _zip_jar(tmp_path / f"pymcbot-{version}+26.3.jar", {"a.class": b"x"})
    api = FakeModrinth([])
    monkeypatch.setenv("MODRINTH_TOKEN", "mrp_test")
    monkeypatch.setattr(mp, "request", api)
    assert mp.main(["--jars", str(tmp_path), "--project", "pymcbot", "--update-project"]) == 0
    assert api.paths() == [("PATCH", "/project/pymcbot"), ("GET", "/project/pymcbot/version"), ("POST", "/version")]


# ----------------------------------------------------------------- page + icon
def test_modrinth_page_is_about_pymc_bot_and_honest_about_privacy():
    page = (ROOT / "mod/MODRINTH.md").read_text(encoding="utf-8")
    assert "# PyMC Bot" in page.split("\n", 3)[2]
    assert "1.21.1 to 26.3" in page
    assert "no network" in page.lower()
    assert "pymcbot/dataset.jsonl" in page
    assert "Playtime Recorder" not in page
    assert "codero-sus" in page and "PolyForm Noncommercial" in page
    assert "Cortex" not in page


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


@pytest.mark.parametrize(("path", "size"), [("mod/src/main/resources/assets/pymcbot/icon.png", 128),
                                            ("docs/modrinth/icon.png", 512)])
def test_icons_fit_modrinth(path: str, size: int):
    icon = ROOT / path
    assert _png_size(icon) == (size, size)
    assert icon.stat().st_size < 256 * 1024  # Modrinth's icon limit


# ----------------------------------------------------------------- limits
class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def _http_error(code: int, headers: dict[str, str]):
    import email.message
    import io
    import urllib.error

    message = email.message.Message()
    for key, value in headers.items():
        message[key] = value
    return urllib.error.HTTPError("https://api.modrinth.com/v2/version", code, "x", message, io.BytesIO(b"{}"))


def test_rate_limits_are_waited_out_and_retried(monkeypatch):
    answers = [_http_error(429, {"X-Ratelimit-Reset": "7"}), _http_error(502, {}), _Response(b'{"id": "ok"}')]
    waits: list[float] = []

    def fake_urlopen(req, timeout):
        assert req.get_header("User-agent") == mp.USER_AGENT
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(mp.urllib.request, "urlopen", fake_urlopen)
    assert mp.request("POST", "/version", "mrp_x", b"{}", sleep=waits.append) == {"id": "ok"}
    assert waits == [7.0, 4.0]  # Modrinth's reset header, then exponential back-off


def test_client_errors_are_not_retried(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        raise _http_error(400, {})

    monkeypatch.setattr(mp.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(SystemExit, match="HTTP 400"):
        mp.request("POST", "/version", "mrp_x", b"{}", sleep=lambda _s: None)
    assert len(calls) == 1


def test_ci_publishes_to_modrinth_from_main_or_on_request():
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = text.split("\n  modrinth:\n", 1)[1]
    assert "needs: release" in job
    assert "github.ref == 'refs/heads/main'" in job and "vars.MODRINTH_PROJECT != ''" in job
    assert "contains(github.event.head_commit.message, '[modrinth]')" in job  # opt-in for other branches
    assert "github.event_name == 'push'" in job  # never from pull requests
    assert "secrets.MODRINTH_TOKEN" in job and "--dry-run" in job
    assert "--update-project" in job and "MODRINTH_UPDATE_PAGE" not in job  # page + summary every time
    assert "name: pymcbot-all-versions" in job
