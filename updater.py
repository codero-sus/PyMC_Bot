#!/usr/bin/env python3
"""PyMC_Bot updater - keeps the bot, its dependencies and the Fabric mod up to date.

    python updater.py                 # update PyMC_Bot (and its dependencies if they changed)
    python updater.py --check         # only show what would be updated
    python updater.py --mod           # also update the PyMC Playtime Recorder jar in .minecraft/mods
    python updater.py --mod-only      # only the mod
    python updater.py --mod-only --mc 1.21.4    # install the mod for 1.21.4 if it is not there yet

or use the wrappers next to this file: ``updater.cmd`` (Windows) / ``./updater.sh`` (macOS, Linux).

How it updates PyMC_Bot:

* **git checkout** - ``git fetch`` + ``git pull --ff-only``. Local edits to tracked files stop the
  update (nothing is overwritten) unless you pass ``--autostash``.
* **downloaded copy** (no ``.git``) - downloads the source zip of the branch from GitHub and copies
  it over this folder. Your files are kept: ``pymc_bot_config.json``, ``.venv``, ``models/``,
  recordings and anything else that is not part of the repository are never touched, and every
  repository file you changed is first copied to ``.pymc_backup/<time>/``.

Afterwards ``pip install -r requirements.txt`` (and ``npm install``) run only if those files
changed, using ``.venv`` when it exists. The mod update finds ``pymcplaytime-<version>+<mc>.jar``
in your mods folder, downloads the jar for the same Minecraft version from the newest GitHub
release, checks its size, SHA-256 and ``fabric.mod.json``, and keeps the old one as ``*.jar.old``.

Only the Python standard library is used, so the updater works before anything is installed.
Set ``GITHUB_TOKEN`` to raise GitHub's API limit (60 requests/hour without it).

Made with love by codero-sus - https://github.com/codero-sus - https://the-codero.web.app/
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
REPO = "codero-sus/PyMC_Bot"
DEFAULT_BRANCH = "main"
API = "https://api.github.com"
CODELOAD = "https://codeload.github.com"
USER_AGENT = "PyMC_Bot-updater"
STATE_FILE = ".pymc_update.json"
BACKUP_DIR = ".pymc_backup"
MOD_TAG_PREFIX = "pymcplaytime-v"
JAR_RE = re.compile(r"^pymcplaytime-(?P<version>\d+(?:\.\d+)*)\+(?P<mc>[0-9][0-9A-Za-z.\-]*)\.jar$")
# Files whose change means dependencies must be reinstalled.
DEPENDENCY_FILES = ("requirements.txt", "requirements-train.txt", "requirements-dev.txt", "package.json")
# Never written by a zip update, even if a future archive contained them.
PROTECTED = (".git", ".venv", "venv", "node_modules", "models", ".pymc_profiles", BACKUP_DIR,
             "pymc_bot_config.json", STATE_FILE, "pymc-playtime")


class UpdateError(Exception):
    """A problem the user can act on; printed without a traceback."""


def say(message: str = "") -> None:
    print(f"[update] {message}" if message else "", flush=True)


# ---------------------------------------------------------------------------- HTTP


def _request(url: str, accept: str = "application/vnd.github+json") -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT, "Accept": accept}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and url.startswith(API):
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def http_get(url: str, accept: str = "application/vnd.github+json", timeout: float = 60.0) -> bytes:
    """GET with a clear error message (rate limits, offline, missing branch...)."""
    try:
        with urllib.request.urlopen(_request(url, accept), timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 403 and "rate limit" in exc.read().decode("utf-8", "replace").lower():
            raise UpdateError("GitHub's API rate limit was reached - wait an hour or set GITHUB_TOKEN") from exc
        if exc.code == 404:
            raise UpdateError(f"not found on GitHub: {url}") from exc
        raise UpdateError(f"GitHub answered HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UpdateError(f"could not reach GitHub ({exc}) - are you online?") from exc


def http_json(url: str):
    return json.loads(http_get(url).decode("utf-8"))


# ---------------------------------------------------------------------------- helpers


def run(cmd: list[str], cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise UpdateError(f"`{' '.join(cmd)}` failed: {detail}")
    return result


def file_hashes(root: Path, names=DEPENDENCY_FILES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        path = root / name
        out[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return out


def version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text))


def local_version(root: Path = ROOT) -> str:
    init = root / "pymc_bot" / "__init__.py"
    match = re.search(r'__version__\s*=\s*"([^"]+)"', init.read_text(encoding="utf-8")) if init.is_file() else None
    return match.group(1) if match else "unknown"


def is_protected(rel: PurePosixPath) -> bool:
    return bool(rel.parts) and rel.parts[0] in PROTECTED


def venv_python(root: Path = ROOT) -> Path | None:
    for candidate in (root / ".venv" / "Scripts" / "python.exe", root / ".venv" / "bin" / "python"):
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------- bot: git


def is_git_checkout(root: Path = ROOT) -> bool:
    if not (root / ".git").exists() or shutil.which("git") is None:
        return False
    return run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root, check=False).stdout.strip() == "true"


def update_git(root: Path, check: bool, autostash: bool) -> bool:
    """Fast-forwards the checkout. Returns True when new commits were pulled."""
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root).stdout.strip()
    upstream = run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], cwd=root, check=False)
    if upstream.returncode != 0:
        raise UpdateError(f"branch '{branch}' does not track a remote branch - run `git branch -u origin/{branch}`")
    upstream_name = upstream.stdout.strip()
    say(f"git checkout on '{branch}' (tracking {upstream_name}) - fetching ...")
    run(["git", "fetch", "--quiet"], cwd=root)
    behind = int(run(["git", "rev-list", "--count", "HEAD..@{u}"], cwd=root).stdout.strip() or 0)
    ahead = int(run(["git", "rev-list", "--count", "@{u}..HEAD"], cwd=root).stdout.strip() or 0)
    if behind == 0:
        say(f"PyMC_Bot is up to date ({local_version(root)})" + (f", {ahead} local commit(s) not pushed" if ahead else ""))
        return False
    log = run(["git", "log", "--oneline", "--no-decorate", "-n", "10", "HEAD..@{u}"], cwd=root).stdout.strip()
    say(f"{behind} new commit(s) on {upstream_name}:")
    for line in log.splitlines():
        say(f"  {line}")
    if check:
        return False
    if ahead:
        raise UpdateError(f"your branch has {ahead} local commit(s) and the remote moved on - "
                          "merge or rebase yourself (`git pull --rebase`)")
    dirty = run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root).stdout.strip()
    if dirty and not autostash:
        raise UpdateError("you have local changes to repository files:\n" + dirty
                          + "\n  commit or stash them, or run the updater with --autostash")
    cmd = ["git", "pull", "--ff-only", "--quiet"] + (["--autostash"] if autostash else [])
    run(cmd, cwd=root)
    say(f"pulled {behind} commit(s) - PyMC_Bot is now {local_version(root)} "
        f"({run(['git', 'rev-parse', '--short', 'HEAD'], cwd=root).stdout.strip()})")
    return True


# ---------------------------------------------------------------------------- bot: zip


def remote_commit(repo: str, branch: str) -> dict:
    data = http_json(f"{API}/repos/{repo}/commits/{branch}")
    return {"sha": data["sha"], "date": data["commit"]["committer"]["date"],
            "message": data["commit"]["message"].splitlines()[0]}


def read_state(root: Path) -> dict:
    try:
        return json.loads((root / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def apply_zip(archive: bytes, root: Path, stamp: str | None = None) -> dict[str, list[str]]:
    """Copies the repository archive over ``root``.

    Files that are not in the archive (config, venv, models, recordings) are left alone, protected
    folders are never written, and a repository file whose content differs (beyond CRLF/LF line
    endings) is backed up to ``.pymc_backup/<stamp>/`` before it is replaced. Returns the written, backed-up and skipped paths.
    """
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    report: dict[str, list[str]] = {"written": [], "backed_up": [], "skipped": []}
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if not names:
            raise UpdateError("the downloaded archive is empty")
        top = PurePosixPath(names[0]).parts[0]
        for name in names:
            parts = PurePosixPath(name).parts
            if parts[0] != top or len(parts) < 2:
                continue
            rel = PurePosixPath(*parts[1:])
            if rel.is_absolute() or ".." in rel.parts:
                raise UpdateError(f"refusing unsafe path in archive: {name}")
            if is_protected(rel):
                report["skipped"].append(str(rel))
                continue
            data = zf.read(name)
            dest = root.joinpath(*rel.parts)
            if dest.is_file():
                current = dest.read_bytes()
                if current == data:
                    continue
                # Only back up real edits - not files an editor merely converted to CRLF.
                if current.replace(b"\r\n", b"\n") != data.replace(b"\r\n", b"\n"):
                    backup = root / BACKUP_DIR / stamp / Path(*rel.parts)
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dest, backup)
                    report["backed_up"].append(str(rel))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            mode = (zf.getinfo(name).external_attr >> 16) & 0o777
            if mode & 0o100 and os.name != "nt":
                dest.chmod(dest.stat().st_mode | 0o111)
            report["written"].append(str(rel))
    return report


def update_zip(root: Path, repo: str, branch: str, check: bool) -> bool:
    say(f"downloaded copy (no git) - checking {repo}@{branch} on GitHub ...")
    remote = remote_commit(repo, branch)
    state = read_state(root)
    if state.get("sha") == remote["sha"]:
        say(f"PyMC_Bot is up to date ({local_version(root)}, {remote['sha'][:7]})")
        return False
    known = f"installed {state['sha'][:7]}" if state.get("sha") else "installed version unknown"
    say(f"newest commit {remote['sha'][:7]} ({remote['date'][:10]}): {remote['message']} [{known}]")
    if check:
        return False
    say("downloading the source zip ...")
    archive = http_get(f"{CODELOAD}/{repo}/zip/{remote['sha']}", accept="application/zip", timeout=300)
    report = apply_zip(archive, root)
    (root / STATE_FILE).write_text(json.dumps({"repo": repo, "branch": branch, **remote}, indent=2) + "\n",
                                   encoding="utf-8")
    say(f"updated {len(report['written'])} file(s) - PyMC_Bot is now {local_version(root)} ({remote['sha'][:7]})")
    if report["backed_up"]:
        say(f"your edited copies of {len(report['backed_up'])} file(s) were saved in {BACKUP_DIR}/ first:")
        for rel in report["backed_up"][:15]:
            say(f"  {rel}")
    say("files deleted upstream are not removed - delete them by hand if they get in the way")
    return True


# ---------------------------------------------------------------------------- dependencies


def update_dependencies(root: Path, before: dict, force: bool, check: bool) -> None:
    after = file_hashes(root)
    changed = [name for name in DEPENDENCY_FILES if before.get(name) != after.get(name)]
    if not changed and not force:
        say("dependencies unchanged")
        return
    say(("dependency files changed: " + ", ".join(changed)) if changed else "reinstalling dependencies (--deps)")
    if check:
        return
    python = venv_python(root)
    if python is None and sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        python = Path(sys.executable)  # running inside some other virtualenv
    py_changed = force or any(n.startswith("requirements") for n in changed)
    if py_changed:
        if python is None:
            say("no .venv found - install the Python packages yourself:")
            say("  python -m venv .venv  then  .venv/bin/pip install -r requirements.txt   (Windows: .venv\\Scripts\\pip)")
        else:
            say(f"pip install -r requirements.txt  ({python})")
            result = subprocess.run([str(python), "-m", "pip", "install", "--quiet", "-r", "requirements.txt"], cwd=root)
            if result.returncode != 0:
                raise UpdateError("pip install failed - see the messages above")
            if "requirements-train.txt" in changed:
                say("requirements-train.txt changed - if you use the transformer engine run:")
                say(f"  {python} -m pip install -r requirements-train.txt")
            if "requirements-dev.txt" in changed and (root / ".venv").exists():
                say("requirements-dev.txt changed - developers: pip install -r requirements-dev.txt")
    if force or "package.json" in changed:
        npm = shutil.which("npm")
        if npm is None:
            say("package.json changed but npm is not installed - real servers need Node 18+ and `npm install`")
        else:
            say("npm install")
            if subprocess.run([npm, "install", "--no-audit", "--no-fund"], cwd=root).returncode != 0:
                raise UpdateError("npm install failed - see the messages above")


# ---------------------------------------------------------------------------- mod


@dataclass(frozen=True)
class InstalledJar:
    path: Path
    version: str
    mc: str


def default_mods_dir() -> Path:
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / ".minecraft"
    elif system == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "minecraft"
    else:
        base = Path.home() / ".minecraft"
    return base / "mods"


def parse_jar_name(name: str) -> tuple[str, str] | None:
    match = JAR_RE.match(name)
    return (match.group("version"), match.group("mc")) if match else None


def installed_jars(mods_dir: Path) -> list[InstalledJar]:
    jars = []
    if mods_dir.is_dir():
        for path in sorted(mods_dir.glob("pymcplaytime-*.jar")):
            parsed = parse_jar_name(path.name)
            if parsed:
                jars.append(InstalledJar(path, *parsed))
    return jars


def pick_release(releases: list[dict], allow_pre: bool) -> dict | None:
    """Newest mod release: stable first; pre-releases with --pre or when there is no stable one."""
    mod = [r for r in releases if not r.get("draft") and str(r.get("tag_name", "")).startswith(MOD_TAG_PREFIX)]
    mod.sort(key=lambda r: version_tuple(r["tag_name"][len(MOD_TAG_PREFIX):]), reverse=True)
    stable = [r for r in mod if not r.get("prerelease")]
    if allow_pre or not stable:
        return mod[0] if mod else None
    return stable[0]


def asset_for(release: dict, version: str, mc: str) -> dict | None:
    wanted = f"pymcplaytime-{version}+{mc}.jar"
    return next((a for a in release.get("assets", []) if a.get("name") == wanted), None)


def verify_jar(data: bytes, asset: dict, mc: str) -> None:
    if asset.get("size") is not None and len(data) != asset["size"]:
        raise UpdateError(f"download size {len(data)} does not match the release ({asset['size']})")
    digest = str(asset.get("digest") or "")
    if digest.startswith("sha256:") and hashlib.sha256(data).hexdigest() != digest.split(":", 1)[1]:
        raise UpdateError("the downloaded jar's SHA-256 does not match the release - try again")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            meta = json.loads(zf.read("fabric.mod.json").decode("utf-8"))
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise UpdateError(f"the downloaded file is not a valid mod jar ({exc})") from exc
    if meta.get("id") != "pymcplaytime" or meta.get("depends", {}).get("minecraft") != mc:
        raise UpdateError(f"the downloaded jar is not the recorder for Minecraft {mc}")


def install_jar(mods_dir: Path, data: bytes, name: str, replaces: list[Path]) -> Path:
    """Writes the new jar atomically and keeps each replaced jar as ``<name>.old``."""
    mods_dir.mkdir(parents=True, exist_ok=True)
    target = mods_dir / name
    tmp = mods_dir / (name + ".part")
    tmp.write_bytes(data)
    try:
        for old in replaces:
            if old.resolve() == target.resolve():
                continue
            backup = old.with_name(old.name + ".old")
            if backup.exists():
                backup.unlink()
            old.rename(backup)
        os.replace(tmp, target)
    except PermissionError as exc:
        tmp.unlink(missing_ok=True)
        raise UpdateError(f"{exc.filename or mods_dir} is in use - close Minecraft and run the updater again") from exc
    return target


def update_mod(mods_dir: Path, repo: str, mc_wanted: str | None, allow_pre: bool, check: bool,
               reinstall: bool = False) -> bool:
    say(f"mods folder: {mods_dir}")
    jars = installed_jars(mods_dir)
    targets: dict[str, list[InstalledJar]] = {}
    for jar in jars:
        targets.setdefault(jar.mc, []).append(jar)
    if mc_wanted:
        targets.setdefault(mc_wanted, [])
    if not targets:
        raise UpdateError("no pymcplaytime-*.jar in that folder - pass --mc <your Minecraft version> to install it, "
                          "or --mods-dir if your mods live elsewhere")
    release = pick_release(http_json(f"{API}/repos/{repo}/releases?per_page=30"), allow_pre)
    if release is None:
        raise UpdateError(f"{repo} has no {MOD_TAG_PREFIX}* release yet")
    newest = release["tag_name"][len(MOD_TAG_PREFIX):]
    say(f"newest mod release: {release['tag_name']}" + (" (pre-release)" if release.get("prerelease") else ""))
    changed = False
    for mc, current in sorted(targets.items()):
        best = max((j.version for j in current), key=version_tuple, default=None)
        if best and version_tuple(best) >= version_tuple(newest) and not reinstall:
            say(f"Minecraft {mc}: pymcplaytime {best} is up to date")
            continue
        asset = asset_for(release, newest, mc)
        if asset is None:
            supported = sorted({p[1] for a in release.get("assets", []) if (p := parse_jar_name(a.get("name", "")))},
                               key=version_tuple)
            say(f"Minecraft {mc}: {release['tag_name']} has no jar for it (available: {', '.join(supported)})")
            continue
        action = f"{best} -> {newest}" if best else f"install {newest}"
        say(f"Minecraft {mc}: {action}  ({asset['name']}, {asset.get('size', '?')} bytes)")
        if check:
            continue
        data = http_get(asset["browser_download_url"], accept="application/octet-stream", timeout=300)
        verify_jar(data, asset, mc)
        target = install_jar(mods_dir, data, asset["name"], [j.path for j in current])
        checked = "SHA-256" if str(asset.get("digest", "")).startswith("sha256:") else "size"
        say(f"Minecraft {mc}: installed {target.name} ({checked} verified)")
        if current:
            say("  the old jar was kept as *.jar.old - Fabric ignores it, delete it whenever you like")
        changed = True
    if changed:
        say("restart Minecraft to load the new mod; keep Fabric API for the same version in the mods folder")
    return changed


# ---------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="updater",
        description="Update PyMC_Bot, its dependencies and the PyMC Playtime Recorder mod from GitHub.",
        epilog="Examples:  updater --check   |   updater --mod   |   updater --mod-only --mc 1.21.4",
    )
    parser.add_argument("--check", action="store_true", help="only report available updates, change nothing")
    parser.add_argument("--mod", action="store_true", help="also update the Fabric mod jar in your mods folder")
    parser.add_argument("--mod-only", action="store_true", help="only update the Fabric mod")
    parser.add_argument("--mc", metavar="VERSION", help="Minecraft version to install the mod for (e.g. 1.21.4, 26.3)")
    parser.add_argument("--mods-dir", type=Path, help="mods folder (default: the official launcher's .minecraft/mods)")
    parser.add_argument("--pre", action="store_true", help="allow mod pre-releases even when a stable release exists")
    parser.add_argument("--reinstall", action="store_true", help="download the mod again even if it is up to date")
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help=f"branch for downloaded copies (default {DEFAULT_BRANCH})")
    parser.add_argument("--repo", default=REPO, help=argparse.SUPPRESS)
    parser.add_argument("--autostash", action="store_true", help="git: stash local edits, update, re-apply them")
    deps = parser.add_mutually_exclusive_group()
    deps.add_argument("--deps", action="store_true", help="reinstall dependencies even if unchanged")
    deps.add_argument("--no-deps", action="store_true", help="never touch dependencies")
    return parser


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    args = build_parser().parse_args(argv)
    if args.mc and not (args.mod or args.mod_only):
        args.mod = True
    say(f"PyMC_Bot updater - {args.repo}" + ("  (check only)" if args.check else ""))
    try:
        if not args.mod_only:
            before = file_hashes(root)
            if is_git_checkout(root):
                pulled = update_git(root, args.check, args.autostash)
            else:
                pulled = update_zip(root, args.repo, args.branch, args.check)
            if not args.no_deps and (pulled or args.deps):
                update_dependencies(root, before, args.deps, args.check)
            if pulled:
                say("restart PyMC_Bot (the panel / bot) to use the new version")
        if args.mod or args.mod_only:
            say()
            update_mod(args.mods_dir or default_mods_dir(), args.repo, args.mc, args.pre, args.check, args.reinstall)
    except UpdateError as exc:
        say(f"ERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        say("cancelled")
        return 130
    say("done")
    return 0


if __name__ == "__main__":
    # Plain ASCII output so old Windows consoles never choke on it.
    raise SystemExit(main())
