"""The npm-free mineflayer installer (pymc_bot.node_setup) - offline, with a fake registry."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from pymc_bot import node_setup as ns

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------ semver
@pytest.mark.parametrize(
    ("version", "spec", "ok"),
    [
        ("4.39.0", "^4.20.0", True),
        ("5.0.0", "^4.20.0", False),
        ("0.2.5", "^0.2.1", True),
        ("0.3.0", "^0.2.1", False),
        ("1.2.9", "~1.2.3", True),
        ("1.3.0", "~1.2.3", False),
        ("2.0.0", ">=1.0.0 <2.0.0", False),
        ("1.9.9", ">=1.0.0 <2.0.0", True),
        ("1.4.0", "1.x", True),
        ("3.1.0", "1 || 2 || 3", True),
        ("1.0.0-beta.1", "^1.0.0", False),
        ("7.7.7", "*", True),
        ("1.2.3", "1.2.3", True),
        ("1.5.0", "1.2.3 - 1.6", True),
    ],
)
def test_satisfies(version, spec, ok):
    assert ns.satisfies(version, spec) is ok


def test_max_satisfying_picks_the_newest_match():
    assert ns.max_satisfying(["1.0.0", "1.4.2", "1.10.0", "2.0.0"], "^1.0.0") == "1.10.0"
    assert ns.max_satisfying(["1.0.0"], "^2") is None


# ------------------------------------------------------------------------------ fake registry
def _tarball(name: str, version: str, files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        payload = {"package.json": json.dumps({"name": name, "version": version}), **files}
        for rel, text in payload.items():
            data = text.encode()
            info = tarfile.TarInfo(f"package/{rel}")
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeNpm:
    """A tiny registry: ``{name: {version: (deps, files)}}`` with real sha512 integrity."""

    def __init__(self, packages):
        self.tarballs: dict[str, bytes] = {}
        self.docs: dict[str, dict] = {}
        for name, versions in packages.items():
            doc = {"name": name, "dist-tags": {"latest": max(versions, key=ns.Version.parse)}, "versions": {}}
            for version, (deps, files) in versions.items():
                data = _tarball(name, version, files)
                url = f"https://fake/{name}/-/{name}-{version}.tgz"
                self.tarballs[url] = data
                doc["versions"][version] = {
                    "name": name, "version": version, "dependencies": deps,
                    "dist": {"tarball": url, "integrity": "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()},
                }
            self.docs[name] = doc
        self.fetched: list[str] = []

    def fetch(self, url: str):
        name = url.rsplit("/", 1)[1].replace("%2F", "/")
        self.fetched.append(name)
        if name not in self.docs:
            raise ns.SetupError(f"404 {name}")
        return self.docs[name]

    def download(self, url: str) -> bytes:
        return self.tarballs[url]


@pytest.fixture()
def npm():
    return FakeNpm({
        "mineflayer": {"4.1.0": ({"vec3": "^0.1.7", "chat": "^1.0.0"}, {"index.js": "module.exports = 1"})},
        "vec3": {"0.1.8": ({}, {"index.js": "v"}), "0.2.0": ({}, {"index.js": "v2"})},
        "chat": {"1.2.0": ({"vec3": "^0.2.0"}, {"index.js": "c"})},
        "pathfinder": {"2.4.0": ({"vec3": "^0.1.0"}, {"index.js": "p"})},
    })


def test_resolver_hoists_and_nests_conflicting_versions(npm):
    packages = ns.resolve({"mineflayer": "^4.0.0", "pathfinder": "^2.0.0"}, ns.Registry("https://fake", npm.fetch),
                          log=lambda _m: None)
    paths = {p.path: p.version for p in packages}
    assert paths["node_modules/mineflayer"] == "4.1.0"
    assert paths["node_modules/vec3"] == "0.1.8"  # hoisted: what mineflayer (first) asked for
    assert paths["node_modules/chat/node_modules/vec3"] == "0.2.0"  # chat needs another vec3: nested copy
    assert len(packages) == 5


def test_install_python_unpacks_checks_and_writes_a_manifest(npm, tmp_path):
    deps = {"mineflayer": "^4.0.0"}
    count = ns.install_python(tmp_path, deps, registry=ns.Registry("https://fake", npm.fetch), log=lambda _m: None,
                              download=npm.download, workers=2)
    nm = tmp_path / "node_modules"
    assert count == 4
    assert (nm / "mineflayer" / "index.js").read_text() == "module.exports = 1"
    assert (nm / "chat" / "node_modules" / "vec3" / "index.js").read_text() == "v2"
    manifest = ns.read_manifest(nm)
    assert manifest["method"] == "python" and manifest["fingerprint"] == ns.deps_fingerprint(deps)
    assert not list(tmp_path.glob(".node_modules-*"))  # staging folder cleaned up


def test_corrupt_download_is_refused(npm, tmp_path):
    def broken(url: str) -> bytes:
        return npm.download(url) + b"x"

    with pytest.raises(ns.SetupError, match="checksum"):
        ns.install_python(tmp_path, {"vec3": "0.1.8"}, registry=ns.Registry("https://fake", npm.fetch),
                          log=lambda _m: None, download=broken)
    assert not (tmp_path / "node_modules").exists()


def test_extract_skips_bedrock_data_when_slim_and_blocks_path_escapes(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in ("package/data/pc/1.21/blocks.json", "package/data/bedrock/1.20/blocks.json",
                    "package/data/bedrock/common/x.json", "package/../../evil.txt"):
            info = tarfile.TarInfo(rel)
            info.size = 2
            tar.addfile(info, io.BytesIO(b"{}"))
    data = buf.getvalue()
    slim = tmp_path / "slim"
    ns.extract_tarball(data, slim / "minecraft-data", package_name="minecraft-data/minecraft-data", slim=True)
    assert (slim / "minecraft-data" / "data" / "pc" / "1.21" / "blocks.json").exists()
    assert (slim / "minecraft-data" / "data" / "bedrock" / "common" / "x.json").exists()
    assert not (slim / "minecraft-data" / "data" / "bedrock" / "1.20").exists()
    assert not (tmp_path / "evil.txt").exists()
    full = tmp_path / "full"
    ns.extract_tarball(data, full, package_name="minecraft-data/minecraft-data", slim=False)
    assert (full / "data" / "bedrock" / "1.20" / "blocks.json").exists()


def test_default_dependencies_match_package_json():
    deps = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))["dependencies"]
    assert deps == ns.DEFAULT_DEPENDENCIES
    assert ns.wanted_dependencies() == deps


def test_install_bridge_auto_falls_back_to_the_bundle(monkeypatch, tmp_path):
    calls = []

    def no_registry(target, *a, **k):
        calls.append("python")
        raise ns.SetupError("registry unreachable")

    def bundle(target, *a, **k):
        calls.append("bundle")
        return 3

    monkeypatch.setattr(ns, "install_python", no_registry)
    monkeypatch.setattr(ns, "install_bundle", bundle)
    assert ns.install_bridge("auto", tmp_path, log=lambda _m: None) == "bundle"
    assert calls == ["python", "bundle"]


def test_install_bundle_from_a_local_zip(tmp_path):
    archive = tmp_path / ns.BUNDLE_ASSET
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("node_modules/mineflayer/package.json", json.dumps({"version": "4.39.0"}))
        zf.writestr("node_modules/../escape.txt", "no")
    target = tmp_path / "rt"
    ns.install_bundle(target, archive=archive, log=lambda _m: None)
    assert (target / "node_modules" / "mineflayer" / "package.json").exists()
    assert not (tmp_path / "escape.txt").exists()
    assert ns.read_manifest(target / "node_modules")["method"] == "bundle"


def test_bridge_status_reports_what_is_installed(monkeypatch, tmp_path):
    nm = tmp_path / "node_modules"
    (nm / "mineflayer").mkdir(parents=True)
    (nm / "mineflayer" / "package.json").write_text(json.dumps({"version": "4.39.0"}))
    monkeypatch.setattr(ns, "node_modules_dirs", lambda: [nm])
    monkeypatch.setattr(ns, "find_node", lambda min_major=0: "/usr/bin/node")
    monkeypatch.setattr(ns, "node_version", lambda binary: "v22.11.0")
    status = ns.bridge_status()
    assert status["ready"] and status["mineflayer_version"] == "4.39.0" and status["node_version"] == "v22.11.0"


def test_cli_check_prints_status(monkeypatch, capsys):
    monkeypatch.setattr(ns, "bridge_status", lambda: {"ready": False, "node": None, "mineflayer": None,
                                                     "node_version": None, "mineflayer_version": None,
                                                     "installed_by": None, "current": False, "npm": None,
                                                     "target": "/x", "node_portable": False})
    parser = __import__("argparse").ArgumentParser()
    ns.add_arguments(parser)
    code = ns.run(parser.parse_args(["--check"]), log=print)
    assert code in (0, 1)
    assert "mineflayer" in capsys.readouterr().out.lower()


def test_updater_refreshes_the_bridge_without_npm(monkeypatch, tmp_path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("pymc_updater_for_test", ROOT / "updater.py")
    updater = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = updater
    spec.loader.exec_module(updater)

    calls = []

    class Helper:
        SetupError = ns.SetupError

        @staticmethod
        def install_bridge(method, target, log):
            calls.append((method, Path(target)))
            log("resolved 92 packages")
            return "python"

    monkeypatch.setattr(updater, "_load_node_setup", lambda root: Helper)
    monkeypatch.setattr(updater, "say", lambda message: calls.append(message))
    (tmp_path / "node_modules").mkdir()
    updater.update_bridge(tmp_path)
    assert ("auto", tmp_path) in calls and any("mineflayer updated" in str(c) for c in calls)

    calls.clear()
    (tmp_path / "node_modules").rmdir()
    updater.update_bridge(tmp_path)  # not installed here: only explain how
    assert not any(isinstance(c, tuple) for c in calls)
    assert any("pymc_bot setup" in str(c) for c in calls)
