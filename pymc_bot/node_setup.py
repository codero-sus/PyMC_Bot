#!/usr/bin/env python3
"""Install the Minecraft bridge (mineflayer) **without npm** - and Node.js itself if it is missing.

The real-server backend runs mineflayer, a Node.js library. Most people do not want to install npm
for one library, so this module does the job in pure Python (standard library only, so it also runs
before ``pip install`` and from the stand-alone updater)::

    python -m pymc_bot setup                  # everything that is missing, the best way available
    python -m pymc_bot setup --check          # only report what is installed
    python pymc_bot/node_setup.py             # the same without the bot's Python packages

Ways to get mineflayer, tried in this order by ``--method auto``:

``python``  a small npm in Python: resolves ``package.json`` against registry.npmjs.org (the same
            registry npm uses), checks every tarball's sha512, and lays out ``node_modules`` the
            way npm does (shared packages at the top, conflicting versions nested).
``bundle``  downloads ``pymc-bot-node-modules.zip`` from the newest GitHub release (built by CI
            with the ``python`` method) and checks it against the release's ``SHA256SUMS.txt``.
``npm``     plain ``npm install`` when npm happens to be there.

Node.js: when ``node`` (22+) is not on PATH, the official portable build for this OS/CPU is
downloaded from nodejs.org into ``~/.pymc_bot/runtime/node`` and verified against the release's
``SHASUMS256.txt``. Nothing is installed system wide and no admin rights are needed.

Made with love by codero-sus - https://github.com/codero-sus - https://the-codero.web.app/
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
NODE_DIR = PACKAGE_DIR / "node"
REGISTRY = os.environ.get("PYMC_NPM_REGISTRY", "https://registry.npmjs.org").rstrip("/")
NODE_MIRROR = os.environ.get("PYMC_NODE_MIRROR", "https://nodejs.org/dist").rstrip("/")
GITHUB_REPO = "codero-sus/PyMC_Bot"
BUNDLE_ASSET = "pymc-bot-node-modules.zip"
MANIFEST = ".pymc-install.json"
MIN_NODE_MAJOR = 22
NODE_LINE = 22  # the LTS line that is downloaded when node is missing
USER_AGENT = "PyMC_Bot-setup (+https://github.com/codero-sus/PyMC_Bot)"
RETRY_DELAYS = (1.5, 4.0)
METHODS = ("auto", "python", "bundle", "npm")

#: Used when the bot runs from a pip install (no package.json next to it). Kept equal to the
#: project's package.json by tests/test_node_setup.py.
DEFAULT_DEPENDENCIES: dict[str, str] = {
    "minecraft-data": "^3.117.0",
    "minecraft-protocol": "^1.68.0",
    "mineflayer": "^4.39.0",
    "mineflayer-pathfinder": "^2.4.5",
}

#: minecraft-data ships every Bedrock Edition version too (~330 MB) - the bridge only plays Java
#: Edition, so a "slim" install skips them (the shared ``bedrock/common`` folder stays).
SLIM_SKIP = re.compile(r"^minecraft-data/minecraft-data/data/bedrock/(?!common(?:/|$))[^/]+")

Log = Callable[[str], None]


class SetupError(RuntimeError):
    """Something the user can act on (no internet, unsupported platform, bad checksum...)."""


def runtime_dir() -> Path:
    """Where downloaded Node.js and pip-install bridges live (``PYMC_BOT_RUNTIME`` overrides it)."""
    override = os.environ.get("PYMC_BOT_RUNTIME")
    return Path(override).expanduser() if override else Path.home() / ".pymc_bot" / "runtime"


def _print(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------- http
def http_get(url: str, *, accept: str | None = None, timeout: float = 60.0, retries: Iterable[float] | None = None) -> bytes:
    """GET with a couple of retries for flaky networks (never retries a 4xx)."""
    delays = list(RETRY_DELAYS if retries is None else retries)
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    while True:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or not delays:
                raise SetupError(f"{url}: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if not delays:
                reason = getattr(exc, "reason", exc)
                raise SetupError(f"could not download {url} ({reason}) - check the internet connection") from exc
        time.sleep(delays.pop(0))


def http_json(url: str, *, accept: str | None = "application/json") -> Any:
    try:
        return json.loads(http_get(url, accept=accept).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SetupError(f"{url} did not return JSON") from exc


# ---------------------------------------------------------------------------- semver
_VERSION = re.compile(r"^\s*v?=?\s*(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?\s*$")
_PARTIAL = re.compile(r"^v?=?(\*|x|X|\d+)(?:\.(\*|x|X|\d+))?(?:\.(\*|x|X|\d+))?(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$")


@dataclass(frozen=True, order=False)
class Version:
    major: int
    minor: int
    patch: int
    pre: tuple[Any, ...] = ()

    @classmethod
    def parse(cls, text: str) -> Version | None:
        match = _VERSION.match(str(text))
        if not match:
            return None
        pre = tuple(int(p) if p.isdigit() else p for p in match.group(4).split(".")) if match.group(4) else ()
        return cls(int(match.group(1)), int(match.group(2)), int(match.group(3)), pre)

    def key(self) -> tuple[Any, ...]:
        # a release sorts after its pre-releases; numeric identifiers sort before text ones
        pre = tuple((0, p, "") if isinstance(p, int) else (1, 0, p) for p in self.pre)
        return (self.major, self.minor, self.patch, 0 if self.pre else 1, pre)

    def __lt__(self, other: Version) -> bool:
        return self.key() < other.key()

    def __le__(self, other: Version) -> bool:
        return self.key() <= other.key()

    def __gt__(self, other: Version) -> bool:
        return self.key() > other.key()

    def __ge__(self, other: Version) -> bool:
        return self.key() >= other.key()

    def __str__(self) -> str:
        core = f"{self.major}.{self.minor}.{self.patch}"
        return core + ("-" + ".".join(str(p) for p in self.pre) if self.pre else "")


def _wild(part: str | None) -> bool:
    return part is None or part in ("*", "x", "X")


def _expand(op: str, text: str) -> list[tuple[str, Version]]:
    """One range token (``^1.2``, ``~1.2.3``, ``>=2``, ``1.x``...) as plain comparators."""
    match = _PARTIAL.match(text)
    if not match:
        raise ValueError(f"bad version range part: {op}{text!r}")
    ma, mi, pa, pre = match.groups()
    pre_t = tuple(int(p) if p.isdigit() else p for p in pre.split(".")) if pre else ()
    if _wild(ma):
        return [] if op in ("", "=", "^", "~", ">=", "<=") else [("<", Version(0, 0, 0))] if op in ("<", ">") else []
    major = int(ma)
    if _wild(mi):
        low, high = Version(major, 0, 0), Version(major + 1, 0, 0)
        exact = None
    elif _wild(pa):
        minor = int(mi)
        low, high = Version(major, minor, 0), Version(major, minor + 1, 0)
        exact = None
    else:
        minor, patch = int(mi), int(pa)
        exact = Version(major, minor, patch, pre_t)
        low = exact
        if op == "^":
            if major:
                high = Version(major + 1, 0, 0)
            elif minor:
                high = Version(0, minor + 1, 0)
            else:
                high = Version(0, 0, patch + 1)
        elif op == "~":
            high = Version(major, minor + 1, 0)
        else:
            high = None
    if op == "^" and exact is None:
        if major:
            high = Version(major + 1, 0, 0)
        elif not _wild(mi):
            high = Version(0, int(mi) + 1, 0)
        else:
            high = Version(1, 0, 0)
        return [(">=", low), ("<", high)]
    if op in ("^", "~"):
        return [(">=", low), ("<", high)]
    if exact is not None:
        return [(op or "=", exact)]
    # partial versions with an operator: 1.2 means [1.2.0, 1.3.0)
    if op in ("", "="):
        return [(">=", low), ("<", high)]
    if op == ">":
        return [(">=", high)]
    if op == ">=":
        return [(">=", low)]
    if op == "<":
        return [("<", low)]
    if op == "<=":
        return [("<", high)]
    raise ValueError(f"bad operator {op!r}")


def parse_range(spec: str) -> list[list[tuple[str, Version]]]:
    """``"^1.2.3 || >=2 <3"`` -> a list of comparator sets (any set may match)."""
    spec = (spec or "").strip()
    sets: list[list[tuple[str, Version]]] = []
    for part in spec.split("||"):
        part = part.strip()
        if part in ("", "*", "x", "X", "latest"):
            sets.append([])
            continue
        hyphen = re.match(r"^(\S+)\s+-\s+(\S+)$", part)
        if hyphen:
            # "1.2.3 - 2.3" means >=1.2.3 <2.4.0 (a partial upper end includes its whole range)
            sets.append(_expand(">=", hyphen.group(1)) + _expand("<=", hyphen.group(2)))
            continue
        comparators: list[tuple[str, Version]] = []
        tokens = re.findall(r"(\^|~>?|>=|<=|>|<|=)?\s*([^\s<>=^~]+)", part)
        for op, text in tokens:
            comparators += _expand("~" if op == "~>" else op, text)
        sets.append(comparators)
    return sets


def _test(version: Version, op: str, bound: Version) -> bool:
    if op == "=":
        return version.key() == bound.key()
    if op == ">=":
        return version >= bound
    if op == ">":
        return version > bound
    if op == "<=":
        return version <= bound
    return version < bound


def satisfies(version: str | Version, spec: str) -> bool:
    parsed = version if isinstance(version, Version) else Version.parse(version)
    if parsed is None:
        return False
    for comparators in parse_range(spec):
        if not all(_test(parsed, op, bound) for op, bound in comparators):
            continue
        if parsed.pre:
            # like npm: a pre-release only matches a range that names a pre-release of the same x.y.z
            core = (parsed.major, parsed.minor, parsed.patch)
            if not any(b.pre and (b.major, b.minor, b.patch) == core for _, b in comparators):
                continue
        return True
    return False


def max_satisfying(versions: Iterable[str], spec: str) -> str | None:
    best: tuple[Version, str] | None = None
    for text in versions:
        parsed = Version.parse(text)
        if parsed is not None and satisfies(parsed, spec) and (best is None or parsed > best[0]):
            best = (parsed, text)
    return best[1] if best else None


# ---------------------------------------------------------------------------- resolver
@dataclass
class Package:
    name: str
    version: str
    tarball: str
    integrity: str = ""
    shasum: str = ""
    dependencies: dict[str, str] = field(default_factory=dict)
    optional: set[str] = field(default_factory=set)
    path: str = ""  # relative to node_modules' parent, e.g. node_modules/a/node_modules/b


class Registry:
    """Abbreviated npm metadata ("corgi" documents), cached per package."""

    ACCEPT = "application/vnd.npm.install-v1+json; q=1.0, application/json; q=0.8"

    def __init__(self, url: str = REGISTRY, fetch: Callable[[str], Any] | None = None) -> None:
        self.url = url.rstrip("/")
        self._fetch = fetch or (lambda u: http_json(u, accept=self.ACCEPT))
        self._cache: dict[str, Any] = {}

    def packument(self, name: str) -> dict[str, Any]:
        if name not in self._cache:
            quoted = urllib.parse.quote(name, safe="@")
            self._cache[name] = self._fetch(f"{self.url}/{quoted}")
        return self._cache[name]

    def prefetch(self, names: Iterable[str], workers: int = 8) -> None:
        missing = sorted({n for n in names if n not in self._cache})
        if len(missing) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for name, doc in zip(missing, pool.map(self._safe, missing), strict=True):
                    if doc is not None:
                        self._cache[name] = doc

    def _safe(self, name: str) -> Any:
        try:
            quoted = urllib.parse.quote(name, safe="@")
            return self._fetch(f"{self.url}/{quoted}")
        except SetupError:
            return None  # fetched again (with the real error) when it is needed

    def pick(self, name: str, spec: str) -> dict[str, Any]:
        doc = self.packument(name)
        versions = doc.get("versions") or {}
        spec = (spec or "*").strip()
        if spec.startswith("npm:"):
            raise SetupError(f"{name}: npm aliases ({spec}) are not supported - use --method npm")
        if re.match(r"^(git|github:|https?:|file:|link:)", spec) or "/" in spec and not spec.startswith("@"):
            raise SetupError(f"{name}: {spec} is not a registry version - use --method npm")
        tags = doc.get("dist-tags") or {}
        if spec in tags:
            chosen = tags[spec]
        else:
            latest = tags.get("latest")
            # npm prefers the "latest" tag when it satisfies the range
            if latest and latest in versions and satisfies(latest, spec):
                chosen = latest
            else:
                chosen = max_satisfying(versions, spec)
        if not chosen or chosen not in versions:
            raise SetupError(f"no version of {name} matches {spec!r}")
        return versions[chosen]


def _platform_ok(meta: dict[str, Any]) -> bool:
    def allowed(rules: Any, value: str) -> bool:
        if not rules:
            return True
        rules = [rules] if isinstance(rules, str) else list(rules)
        if f"!{value}" in rules:
            return False
        positive = [r for r in rules if not r.startswith("!")]
        return not positive or value in positive

    node_os = {"win32": "win32", "darwin": "darwin"}.get(sys.platform, "linux" if sys.platform.startswith("linux") else sys.platform)
    machine = platform.machine().lower()
    node_cpu = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(machine, machine)
    return allowed(meta.get("os"), node_os) and allowed(meta.get("cpu"), node_cpu)


def resolve(dependencies: dict[str, str], registry: Registry, log: Log = _print) -> list[Package]:
    """npm's node_modules layout: hoist every package to the top unless that version would clash.

    Breadth first, so direct dependencies get the top-level spots; a package that needs another
    version of something already visible to it gets its own copy in its nested ``node_modules``.
    """
    placed: dict[str, Package] = {}  # install path -> package
    root = ""

    def lookup(parent: str, name: str) -> tuple[str, Package | None]:
        # Node's require() walks up: parent/node_modules/name, then each ancestor's node_modules
        here = parent
        while True:
            path = f"{here}/node_modules/{name}" if here else f"node_modules/{name}"
            if path in placed:
                return path, placed[path]
            if not here:
                return "", None
            cut = here.rfind("/node_modules/")
            here = here[:cut] if cut >= 0 else ""

    queue: list[tuple[str, dict[str, str], set[str]]] = [(root, dict(dependencies), set())]
    while queue:
        registry.prefetch(name for _, deps, _ in queue for name in deps)
        next_queue: list[tuple[str, dict[str, str], set[str]]] = []
        for parent, deps, optional in queue:
            for name, spec in sorted(deps.items()):
                found_at, found = lookup(parent, name)
                if found is not None and satisfies(found.version, spec):
                    continue
                try:
                    meta = registry.pick(name, spec)
                except SetupError:
                    if name in optional:
                        log(f"  skipping optional {name}@{spec} (not available)")
                        continue
                    raise
                if name in optional and not _platform_ok(meta):
                    continue
                if found is None:
                    path = f"node_modules/{name}"
                else:
                    path = f"{parent}/node_modules/{name}" if parent else f"node_modules/{name}"
                    if path in placed:  # cannot happen with Node's lookup rules, but never overwrite
                        raise SetupError(f"conflicting versions of {name} under {parent or 'the project'}")
                dist = meta.get("dist") or {}
                if not dist.get("tarball"):
                    raise SetupError(f"{name}@{meta.get('version')} has no tarball in the registry")
                peers = meta.get("peerDependencies") or {}
                peer_meta = meta.get("peerDependenciesMeta") or {}
                child_deps = dict(meta.get("dependencies") or {})
                opt = dict(meta.get("optionalDependencies") or {})
                child_deps.update(opt)
                for peer, peer_spec in peers.items():
                    if not (peer_meta.get(peer) or {}).get("optional"):
                        child_deps.setdefault(peer, peer_spec)
                package = Package(
                    name=name, version=str(meta.get("version")), tarball=dist["tarball"],
                    integrity=str(dist.get("integrity") or ""), shasum=str(dist.get("shasum") or ""),
                    dependencies=child_deps, optional=set(opt), path=path,
                )
                placed[path] = package
                next_queue.append((path, child_deps, set(opt)))
        queue = next_queue
    return sorted(placed.values(), key=lambda p: p.path)


# ---------------------------------------------------------------------------- extraction
def verify(data: bytes, package: Package) -> None:
    """Check the tarball against the registry's SRI ``integrity`` (sha512) or legacy sha1 shasum."""
    for entry in package.integrity.split():
        algo, _, digest = entry.partition("-")
        if algo in ("sha512", "sha384", "sha256"):
            actual = base64.b64encode(hashlib.new(algo, data).digest()).decode()
            if actual != digest:
                raise SetupError(f"{package.name}@{package.version}: checksum mismatch - download corrupted?")
            return
    if package.shasum:
        if hashlib.sha1(data).hexdigest() != package.shasum:  # noqa: S324 - npm's legacy checksum
            raise SetupError(f"{package.name}@{package.version}: checksum mismatch - download corrupted?")
        return
    raise SetupError(f"{package.name}@{package.version} has no checksum in the registry")


def _safe_parts(name: str) -> list[str] | None:
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts) or re.match(r"^[A-Za-z]:", parts[0] or ""):
        return None
    return parts


def extract_tarball(data: bytes, dest: Path, *, package_name: str = "", slim: bool = False) -> int:
    """Unpack an npm tarball (top folder stripped) into ``dest``. Returns the number of files."""
    files = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        for member in archive:
            if not (member.isfile() or member.isdir()):
                continue  # npm ignores links/devices in tarballs as well
            parts = _safe_parts(member.name)
            if parts is None or len(parts) < 2 and member.isfile():
                continue
            rel = "/".join(parts[1:])
            if not rel:
                continue
            if slim and package_name and SLIM_SKIP.match(f"{package_name}/{rel}"):
                continue
            target = dest.joinpath(*parts[1:])
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, open(target, "wb") as out:
                shutil.copyfileobj(source, out)
            mode = 0o755 if member.mode & 0o111 else 0o644
            os.chmod(target, mode)
            files += 1
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def deps_fingerprint(dependencies: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(dependencies, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------- project
def project_package_json() -> Path | None:
    """``package.json`` of a source checkout (None for a pip install)."""
    path = PROJECT_ROOT / "package.json"
    return path if path.is_file() and (PROJECT_ROOT / "pymc_bot").is_dir() else None


def wanted_dependencies() -> dict[str, str]:
    path = project_package_json()
    if path is not None:
        try:
            deps = json.loads(path.read_text(encoding="utf-8")).get("dependencies") or {}
            if deps:
                return {str(k): str(v) for k, v in deps.items()}
        except (OSError, json.JSONDecodeError):
            pass
    return dict(DEFAULT_DEPENDENCIES)


def default_target() -> Path:
    """Folder that gets ``node_modules``: the checkout (next to package.json) or the runtime folder."""
    if project_package_json() is not None and os.access(PROJECT_ROOT, os.W_OK):
        return PROJECT_ROOT
    return runtime_dir()


def node_modules_dirs() -> list[Path]:
    """Every node_modules the bridge can load mineflayer from (same order Node searches + runtime)."""
    seen: list[Path] = []
    for base in (NODE_DIR, PACKAGE_DIR, PROJECT_ROOT, Path.cwd(), runtime_dir()):
        path = base / "node_modules"
        if path not in seen:
            seen.append(path)
    return seen


def find_mineflayer() -> Path | None:
    for folder in node_modules_dirs():
        if (folder / "mineflayer" / "package.json").is_file() or (folder / "mineflayer").is_dir():
            return folder
    return None


def read_manifest(node_modules: Path) -> dict[str, Any]:
    try:
        return json.loads((node_modules / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def bridge_status() -> dict[str, Any]:
    """What is installed: node, its version, where mineflayer lives and whether it is current."""
    node = find_node()
    folder = find_mineflayer()
    manifest = read_manifest(folder) if folder else {}
    deps = wanted_dependencies()
    mineflayer_version = None
    if folder is not None:
        try:
            mineflayer_version = json.loads((folder / "mineflayer" / "package.json").read_text(encoding="utf-8")).get("version")
        except (OSError, json.JSONDecodeError):
            mineflayer_version = None
    current = bool(folder) and (not manifest or manifest.get("fingerprint") == deps_fingerprint(deps))
    return {
        "node": node,
        "node_version": node_version(node) if node else None,
        "node_portable": bool(node) and str(runtime_dir()) in str(node),
        "mineflayer": str(folder) if folder else None,
        "mineflayer_version": mineflayer_version,
        "installed_by": manifest.get("method") or ("npm" if folder else None),
        "current": current,
        "npm": shutil.which("npm"),
        "target": str(default_target()),
        "ready": bool(node) and bool(folder),
    }


# ---------------------------------------------------------------------------- node.js
def _major(version: str | None) -> int:
    match = re.match(r"v?(\d+)", version or "")
    return int(match.group(1)) if match else 0


def node_version(binary: str | os.PathLike[str] | None) -> str | None:
    if not binary:
        return None
    try:
        out = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def portable_node() -> Path | None:
    base = runtime_dir() / "node"
    binary = base / "node.exe" if os.name == "nt" else base / "bin" / "node"
    return binary if binary.is_file() else None


def find_node(min_major: int = 0) -> str | None:
    """``node`` from PATH, else the portable one this module downloaded (``min_major`` = 22 to require it)."""
    for candidate in (shutil.which("node"), portable_node()):
        if candidate and (not min_major or _major(node_version(candidate)) >= min_major):
            return str(candidate)
    return None


def node_platform() -> tuple[str, str, str]:
    """``(os, arch, archive extension)`` in nodejs.org's file naming."""
    if sys.platform == "win32":
        system, ext = "win", "zip"
    elif sys.platform == "darwin":
        system, ext = "darwin", "tar.gz"
    elif sys.platform.startswith("linux"):
        system, ext = "linux", "tar.gz"
    else:
        raise SetupError(f"no portable Node.js build for {sys.platform} - install Node.js {MIN_NODE_MAJOR}+ yourself")
    machine = platform.machine().lower()
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64", "armv7l": "armv7l",
            "ppc64le": "ppc64le", "s390x": "s390x"}.get(machine)
    if arch is None:
        raise SetupError(f"no portable Node.js build for the {machine} CPU - install Node.js {MIN_NODE_MAJOR}+ yourself")
    return system, arch, ext


def latest_node_release(line: int = NODE_LINE, mirror: str = NODE_MIRROR) -> str:
    releases = http_json(f"{mirror}/index.json")
    for release in releases if isinstance(releases, list) else []:
        version = str(release.get("version") or "")
        if _major(version) == line and release.get("lts"):
            return version
    for release in releases if isinstance(releases, list) else []:
        if _major(str(release.get("version"))) == line:
            return str(release["version"])
    raise SetupError(f"nodejs.org lists no Node.js {line} release")


def _extract_archive(archive: Path, dest: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                parts = _safe_parts(info.filename)
                if parts is None:
                    continue
                target = dest.joinpath(*parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                mode = (info.external_attr >> 16) & 0o777
                if mode:
                    os.chmod(target, mode)
        return
    with tarfile.open(archive, mode="r:*") as tf:
        for member in tf:
            parts = _safe_parts(member.name)
            if parts is None:
                continue
            target = dest.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.issym():
                link = member.linkname
                if os.path.isabs(link):
                    continue
                resolved = (target.parent / link).resolve()
                if not str(resolved).startswith(str(dest.resolve())):
                    continue  # never point outside the runtime folder
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink() or target.exists():
                    target.unlink()
                os.symlink(link, target)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tf.extractfile(member)
                if source is None:
                    continue
                with source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out)
                os.chmod(target, member.mode & 0o777 or 0o644)


def install_node(*, line: int = NODE_LINE, mirror: str = NODE_MIRROR, log: Log = _print, force: bool = False) -> Path:
    """Download the official portable Node.js into ``<runtime>/node`` (checksum verified)."""
    existing = portable_node()
    if existing is not None and not force and _major(node_version(existing)) >= MIN_NODE_MAJOR:
        log(f"Node.js already downloaded: {existing}")
        return existing
    system, arch, ext = node_platform()
    version = latest_node_release(line, mirror)
    stem = f"node-{version}-{system}-{arch}"
    filename = f"{stem}.{ext}"
    sums = http_get(f"{mirror}/{version}/SHASUMS256.txt").decode("utf-8", "replace")
    expected = next((line_.split()[0] for line_ in sums.splitlines() if line_.strip().endswith(f"  {filename}")
                     or line_.split()[-1:] == [filename]), None)
    if not expected:
        raise SetupError(f"{filename} is not part of Node.js {version} (SHASUMS256.txt)")
    log(f"Downloading Node.js {version} for {system}-{arch} (nodejs.org, no installer, no admin rights)...")
    base = runtime_dir()
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base, prefix=".node-") as tmp:
        archive = Path(tmp) / filename
        archive.write_bytes(http_get(f"{mirror}/{version}/{filename}", timeout=300))
        if _sha256(archive) != expected:
            raise SetupError(f"{filename}: checksum mismatch - download corrupted, try again")
        unpacked = Path(tmp) / "unpacked"
        _extract_archive(archive, unpacked)
        folder = unpacked / stem
        if not folder.is_dir():
            entries = [p for p in unpacked.iterdir() if p.is_dir()]
            if len(entries) != 1:
                raise SetupError(f"unexpected layout in {filename}")
            folder = entries[0]
        final = base / "node"
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        shutil.move(str(folder), str(final))
    binary = portable_node()
    if binary is None:
        raise SetupError("Node.js was downloaded but the node binary is missing")
    if os.name != "nt":
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log(f"Node.js {node_version(binary) or version} ready: {binary}")
    return binary


# ---------------------------------------------------------------------------- installers
def _write_manifest(node_modules: Path, method: str, dependencies: dict[str, str], packages: int) -> None:
    node_modules.mkdir(parents=True, exist_ok=True)
    (node_modules / MANIFEST).write_text(json.dumps({
        "method": method,
        "fingerprint": deps_fingerprint(dependencies),
        "dependencies": dependencies,
        "packages": packages,
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n", encoding="utf-8")


def _swap_in(staging: Path, target: Path) -> None:
    """Replace ``target/node_modules`` with the freshly built one (old one kept until the new one is in place)."""
    final = target / "node_modules"
    old = target / f".node_modules-old-{os.getpid()}"
    if final.exists():
        final.rename(old)
    try:
        staging.rename(final)
    except OSError:
        if old.exists() and not final.exists():
            old.rename(final)
        raise
    shutil.rmtree(old, ignore_errors=True)


def install_python(target: Path, dependencies: dict[str, str] | None = None, *, registry: Registry | None = None,
                   slim: bool = True, log: Log = _print, workers: int = 8,
                   download: Callable[[str], bytes] | None = None) -> int:
    """The npm-free install: resolve, download (sha512 checked) and unpack into ``target/node_modules``."""
    dependencies = dict(dependencies or wanted_dependencies())
    registry = registry or Registry()
    fetch = download or (lambda url: http_get(url, timeout=180))
    log(f"Resolving {', '.join(f'{k}@{v}' for k, v in dependencies.items())} from {registry.url} ...")
    packages = resolve(dependencies, registry, log)
    log(f"{len(packages)} packages to download.")
    target.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".node_modules-", dir=target))
    try:
        def fetch_one(package: Package) -> tuple[Package, bytes]:
            data = fetch(package.tarball)
            verify(data, package)
            return package, data

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for package, data in pool.map(fetch_one, packages):
                rel = package.path.split("/", 1)[1]  # strip the leading node_modules/
                dest = staging.joinpath(*rel.split("/"))
                extract_tarball(data, dest, package_name=package.name, slim=slim)
                done += 1
                if done % 20 == 0 or done == len(packages):
                    log(f"  {done}/{len(packages)} unpacked")
        _write_manifest(staging, "python", dependencies, len(packages))
        _swap_in(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    log(f"mineflayer installed without npm in {target / 'node_modules'}")
    return len(packages)


def newest_bundle(repo: str = GITHUB_REPO) -> tuple[str, str, str | None]:
    """``(tag, zip url, SHA256SUMS url)`` of the newest release that carries the node_modules bundle."""
    releases = http_json(f"https://api.github.com/repos/{repo}/releases?per_page=20", accept="application/vnd.github+json")
    for release in releases if isinstance(releases, list) else []:
        assets = {a.get("name"): a.get("browser_download_url") for a in release.get("assets") or []}
        if assets.get(BUNDLE_ASSET):
            return str(release.get("tag_name")), assets[BUNDLE_ASSET], assets.get("SHA256SUMS.txt")
    raise SetupError(f"no GitHub release of {repo} carries {BUNDLE_ASSET} yet")


def install_bundle(target: Path, *, repo: str = GITHUB_REPO, log: Log = _print,
                   source: tuple[str, str, str | None] | None = None, archive: Path | None = None) -> int:
    """Unpack the prebuilt ``pymc-bot-node-modules.zip`` - downloaded, or ``archive`` saved by hand (offline)."""
    if archive is not None:
        log(f"Unpacking the Minecraft bridge from {archive} ...")
        data = Path(archive).read_bytes()
        sums_url = None
    else:
        tag, url, sums_url = source or newest_bundle(repo)
        log(f"Downloading the prebuilt Minecraft bridge from the {tag} release ...")
        data = http_get(url, timeout=300)
    if sums_url:
        sums = http_get(sums_url).decode("utf-8", "replace")
        expected = next((ln.split()[0] for ln in sums.splitlines() if ln.split()[-1:] == [BUNDLE_ASSET]), None)
        if expected and hashlib.sha256(data).hexdigest() != expected:
            raise SetupError(f"{BUNDLE_ASSET}: checksum mismatch - download corrupted, try again")
    target.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".node_modules-", dir=target))
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            count = 0
            for info in zf.infolist():
                parts = _safe_parts(info.filename)
                if parts is None:
                    continue
                if parts[0] == "node_modules":
                    parts = parts[1:]
                if not parts:
                    continue
                dest = staging.joinpath(*parts)
                if info.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
                count += 1
        if not (staging / "mineflayer").is_dir():
            raise SetupError(f"{BUNDLE_ASSET} does not contain mineflayer")
        manifest = read_manifest(staging)
        if not manifest:
            _write_manifest(staging, "bundle", wanted_dependencies(), count)
        else:
            manifest["method"] = "bundle"
            (staging / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        _swap_in(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    log(f"mineflayer installed from the release bundle in {target / 'node_modules'}")
    return count


def install_npm(target: Path, *, log: Log = _print) -> int:
    npm = shutil.which("npm")
    if npm is None:
        raise SetupError("npm is not installed")
    if not (target / "package.json").is_file():
        (target / "package.json").write_text(json.dumps({
            "name": "pymc-bot-runtime", "private": True, "dependencies": wanted_dependencies()}, indent=2) + "\n",
            encoding="utf-8")
    log(f"npm install (in {target})")
    if subprocess.run([npm, "install", "--no-audit", "--no-fund"], cwd=target).returncode != 0:
        raise SetupError("npm install failed - see the messages above")
    return 0


def install_bridge(method: str = "auto", target: Path | None = None, *, slim: bool = True, log: Log = _print,
                   archive: Path | None = None) -> str:
    """Install mineflayer with ``method``; ``auto`` tries python, then the release bundle, then npm.

    ``archive``: a ``pymc-bot-node-modules.zip`` already on disk (no internet needed at all).
    """
    target = Path(target) if target else default_target()
    if archive is not None:
        method = "bundle"
    order = {"auto": ("python", "bundle", "npm")}.get(method, (method,))
    errors: list[str] = []
    for name in order:
        if name == "npm" and method == "auto" and shutil.which("npm") is None:
            errors.append("npm: not installed")
            continue
        try:
            if name == "python":
                install_python(target, slim=slim, log=log)
            elif name == "bundle":
                install_bundle(target, log=log, archive=archive)
            elif name == "npm":
                install_npm(target, log=log)
            else:
                raise SetupError(f"unknown method {name!r} (use {', '.join(METHODS)})")
            return name
        except (SetupError, OSError) as exc:
            errors.append(f"{name}: {exc}")
            if method != "auto":
                break
            log(f"  {name} did not work ({exc}) - trying the next way")
    raise SetupError("could not install mineflayer - " + "; ".join(errors))


def setup(method: str = "auto", *, target: Path | None = None, node: str = "auto", slim: bool = True,
          force: bool = False, log: Log = _print, archive: Path | None = None) -> dict[str, Any]:
    """Everything the real-server backend needs. ``node``: auto (download if missing), download, skip."""
    report: dict[str, Any] = {}
    have = find_node(MIN_NODE_MAJOR)
    if node == "download" or (node == "auto" and have is None):
        if node == "auto" and find_node():
            log(f"node {node_version(find_node())} is older than {MIN_NODE_MAJOR} - downloading a portable Node.js")
        report["node"] = str(install_node(log=log, force=force and node == "download"))
    else:
        report["node"] = have
        if have:
            log(f"Node.js {node_version(have)} found: {have}")
        elif node == "skip":
            log("Node.js not found (skipped - the bridge needs Node.js 22+ to run)")
    status = bridge_status()
    if status["mineflayer"] and status["current"] and not force and archive is None:
        log(f"mineflayer {status['mineflayer_version'] or ''} already installed: {status['mineflayer']}")
        report["mineflayer"] = status["mineflayer"]
        report["method"] = "already installed"
    else:
        report["method"] = install_bridge(method, target, slim=slim, log=log, archive=archive)
        report["mineflayer"] = str((Path(target) if target else default_target()) / "node_modules")
    report["ready"] = bool(report.get("node")) and bool(report.get("mineflayer"))
    return report


def node_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the bridge process: the runtime node_modules on NODE_PATH, portable node on PATH."""
    env = dict(os.environ if base is None else base)
    extra = [str(path) for path in (runtime_dir() / "node_modules", PROJECT_ROOT / "node_modules") if path.is_dir()]
    if extra:
        current = [p for p in env.get("NODE_PATH", "").split(os.pathsep) if p]
        env["NODE_PATH"] = os.pathsep.join(current + [p for p in extra if p not in current])
    portable = portable_node()
    if portable is not None:
        env["PATH"] = str(portable.parent) + os.pathsep + env.get("PATH", "")
    return env


def make_bundle(out: Path, *, log: Log = _print) -> Path:
    """Build ``pymc-bot-node-modules.zip`` (the release asset) with the python installer."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pymc-bundle-") as tmp:
        install_python(Path(tmp), slim=True, log=log)
        root = Path(tmp) / "node_modules"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    zf.write(path, "node_modules/" + path.relative_to(root).as_posix())
    log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


# ---------------------------------------------------------------------------- cli
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=METHODS, default="auto",
                        help="how to get mineflayer: auto (python, then bundle, then npm), python (no npm), "
                             "bundle (prebuilt zip from the GitHub release), npm")
    parser.add_argument("--node", choices=("auto", "download", "skip"), default="auto",
                        help="auto: download a portable Node.js only when node 22+ is missing")
    parser.add_argument("--target", default=None, help="folder that gets node_modules (default: the project, "
                        "or ~/.pymc_bot/runtime for pip installs)")
    parser.add_argument("--full", action="store_true", help="keep minecraft-data's Bedrock files (+330 MB, not needed)")
    parser.add_argument("--force", action="store_true", help="reinstall even when everything is current")
    parser.add_argument("--check", action="store_true", help="only show what is installed")
    parser.add_argument("--zip", metavar="ZIP", dest="archive", default=None,
                        help=f"install from a {BUNDLE_ASSET} you downloaded yourself (works offline)")
    parser.add_argument("--bundle", metavar="ZIP", help="(CI) build the node_modules release zip and exit")


def run(args: argparse.Namespace, log: Log = _print) -> int:
    if args.bundle:
        make_bundle(Path(args.bundle), log=log)
        return 0
    if args.check:
        status = bridge_status()
        log(f"node        : {status['node_version'] or 'not found'}{' (portable)' if status['node_portable'] else ''}")
        log(f"mineflayer  : {status['mineflayer_version'] or 'not installed'}"
            + (f"  ({status['mineflayer']}, by {status['installed_by']})" if status["mineflayer"] else ""))
        log(f"npm         : {'found' if status['npm'] else 'not needed'}")
        log("ready       : " + ("yes" if status["ready"] else "no - run: python -m pymc_bot setup"))
        return 0 if status["ready"] else 1
    try:
        report = setup(args.method, target=Path(args.target) if args.target else None, node=args.node,
                       slim=not args.full, force=args.force, log=log,
                       archive=Path(args.archive) if getattr(args, "archive", None) else None)
    except (SetupError, OSError) as exc:
        log(f"setup failed: {exc}")
        return 1
    log("The Minecraft bridge is ready - pick backend 'node' (or auto) in the panel." if report["ready"]
        else "mineflayer is installed; install Node.js 22+ (or rerun without --node skip) to play on real servers.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pymc_bot setup", description=__doc__.split("\n\n")[0])
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
