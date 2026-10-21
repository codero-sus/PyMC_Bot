#!/usr/bin/env python3
"""PyMC_Bot updater - keeps the bot, its dependencies and the Fabric mod up to date.

    python updater.py                 # update PyMC_Bot (and its dependencies if they changed)
    python updater.py --check         # only show what would be updated
    python updater.py --mod           # also update the PyMC Bot mod jar in .minecraft/mods
    python updater.py --mod-only      # only the mod
    python updater.py --mod-only --mc 1.21.4    # install the mod for 1.21.4 if it is not there yet

or use the wrappers next to this file: ``updater.cmd`` (Windows) / ``./updater.sh`` (macOS, Linux).

How it updates PyMC_Bot - always by downloading and extracting a ``.zip``, never with git:

1. asks GitHub which commit is newest on this build's update channel (one small API call);
2. downloads that exact commit as a source ``.zip`` and checks it (complete archive, and the commit
   id stored inside the zip must match);
3. extracts it over this folder. Your files are kept: ``pymc_bot_config.json``, ``.venv``,
   ``models/``, recordings, ``.git`` and anything else that is not part of the project are never
   touched, and every project file you changed is first copied to ``.pymc_backup/<time>/``;
4. removes files the project deleted since your last update - but only files you never edited.

A git clone works too: it is updated the same way (``.git`` is left alone, so ``git status`` will
show the new files as changes - commit, stash or reset them as you like).

Afterwards ``pip install -r requirements.txt`` (and ``npm install``) run only if those files
changed, using ``.venv`` when it exists. The mod update finds ``pymcbot-<version>+<mc>.jar``
(or a pre-0.6 ``pymcplaytime-*.jar``)
in your mods folder, downloads the jar for the same Minecraft version from the newest GitHub
release, checks its size, SHA-256 and ``fabric.mod.json``, and keeps the old one as ``*.jar.old``.

Only the Python standard library is used, so the updater works before anything is installed.
Set ``GITHUB_TOKEN`` to raise GitHub's API limit (60 requests/hour without it); without the API
the updater still works by downloading the zip and reading the commit id stored in it.

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
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
REPO = "codero-sus/PyMC_Bot"
API = "https://api.github.com"
CODELOAD = "https://codeload.github.com"
USER_AGENT = "PyMC_Bot-updater"
STATE_FILE = ".pymc_update.json"
BACKUP_DIR = ".pymc_backup"
# The mod is "PyMC Bot" (id pymcbot) since 0.6.0; 0.5.0 and older were "pymcplaytime".
# Both names are recognised so an old jar is replaced by the new one.
MOD_IDS = ("pymcbot", "pymcplaytime")
MOD_TAG_PREFIXES = tuple(f"{mod_id}-v" for mod_id in MOD_IDS)
JAR_RE = re.compile(r"^(?P<id>pymcbot|pymcplaytime)-(?P<version>\d+(?:\.\d+)*)\+(?P<mc>[0-9][0-9A-Za-z.\-]*)\.jar$")
# Files whose change means dependencies must be reinstalled.
DEPENDENCY_FILES = ("requirements.txt", "requirements-train.txt", "requirements-dev.txt", "package.json")
# Never written by a zip update, even if a future archive contained them.
PROTECTED = (".git", ".venv", "venv", "node_modules", "models", ".pymc_profiles", BACKUP_DIR,
             "pymc_bot_config.json", STATE_FILE, "pymc-playtime")


# ---------------------------------------------------------------------------- update channel
#
# The branch this build updates from is stored sealed (XOR with a fixed key, base85, reversed) so
# it does not show up as plain text in this file, in the console output or in error messages.
# This only hides the name from casual reading - it is not encryption. Maintainers re-seal a new
# branch with:  python updater.py --seal <branch>   and paste the result into _CHANNEL.
_SEAL_KEY = hashlib.sha256(b"PyMC_Bot:update-channel:v1").digest()
_CHANNEL = (
    "?43GhL@Uq!fF+s"
    "ERTSYozS}+{qdR1"
)


def _xor(data: bytes) -> bytes:
    return bytes(byte ^ _SEAL_KEY[index % len(_SEAL_KEY)] for index, byte in enumerate(data))


def seal(text: str) -> str:
    """Seal a branch name for ``_CHANNEL``."""
    return base64.b85encode(_xor(text.encode("utf-8"))).decode("ascii")[::-1]


def unseal(blob: str) -> str:
    return _xor(base64.b85decode(blob[::-1].encode("ascii"))).decode("utf-8")


def update_channel() -> str:
    """The branch this build follows (never printed)."""
    return unseal(_CHANNEL)


def _secret_forms(ref: str) -> set[str]:
    """Every way a ref can appear in URLs, archive folder names and messages."""
    forms = {ref, urllib.parse.quote(ref, safe=""), urllib.parse.quote(ref, safe="/"), ref.replace("/", "-")}
    return {form for form in forms if len(form) >= 4}


def redact(text: str, refs: tuple[str, ...] = ()) -> str:
    """``text`` with the update channel (and any other hidden ``refs``) replaced by a placeholder."""
    out = str(text)
    hidden = set(refs)
    try:
        hidden.add(update_channel())
    except (ValueError, UnicodeDecodeError):  # pragma: no cover - only if _CHANNEL was mangled
        pass
    for ref in hidden:
        for form in sorted(_secret_forms(ref), key=len, reverse=True):
            out = out.replace(form, "<update channel>")
    return out


class UpdateError(Exception):
    """A problem the user can act on; printed without a traceback."""


class RateLimited(UpdateError):
    """GitHub's API limit is used up (the zip download still works)."""


def say(message: str = "") -> None:
    # Every line goes through redact(), so the update channel can never reach the console.
    try:
        print(f"[update] {redact(message)}" if message else "", flush=True)
    except BrokenPipeError:
        # Whoever read our output stopped early (`| head`, `| grep -q`). Keep working silently:
        # an update must never stop half-way because nobody is listening any more.
        try:
            sys.stdout = open(os.devnull, "w")  # noqa: SIM115 - lives until the process exits
        except OSError:
            pass


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
        if exc.code == 429 or (exc.code == 403 and "rate limit" in exc.read().decode("utf-8", "replace").lower()):
            raise RateLimited("GitHub's API rate limit was reached - wait an hour or set GITHUB_TOKEN") from exc
        if exc.code == 404:
            raise UpdateError(f"not found on GitHub: {redact(url)}") from exc
        raise UpdateError(f"GitHub answered HTTP {exc.code} for {redact(url)}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UpdateError(f"could not reach GitHub ({exc}) - are you online?") from exc


def http_json(url: str):
    return json.loads(http_get(url).decode("utf-8"))


# ---------------------------------------------------------------------------- helpers


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


# ---------------------------------------------------------------------------- bot: zip


def git_checkout_present(root: Path = ROOT) -> bool:
    """A ``.git`` folder is there (it is never used or touched, only mentioned)."""
    return (root / ".git").exists()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def remote_commit(repo: str, ref: str) -> dict:
    """The newest commit of ``ref`` (one API call)."""
    data = http_json(f"{API}/repos/{repo}/commits/{urllib.parse.quote(ref, safe='')}")
    return {"sha": data["sha"], "date": data["commit"]["committer"]["date"],
            "message": data["commit"]["message"].splitlines()[0]}


def archive_commit(archive: bytes) -> str:
    """The commit id GitHub stores as the zip comment of a source archive ('' if missing)."""
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            comment = zf.comment.decode("ascii", "replace").strip()
    except zipfile.BadZipFile as exc:
        raise UpdateError("the download is not a valid zip archive - try again") from exc
    return comment if re.fullmatch(r"[0-9a-f]{40}", comment) else ""


def download_archive(repo: str, ref: str, expected_sha: str = "") -> tuple[bytes, str]:
    """Downloads the source zip (by commit id when known) and verifies it. Returns (zip, commit)."""
    target = expected_sha or f"refs/heads/{urllib.parse.quote(ref, safe='/')}"
    archive = http_get(f"{CODELOAD}/{repo}/zip/{target}", accept="application/zip", timeout=300)
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            broken = zf.testzip()
    except zipfile.BadZipFile as exc:
        raise UpdateError("the downloaded file is not a valid zip archive - try again") from exc
    if broken:
        raise UpdateError(f"the downloaded zip is damaged ({redact(broken)}) - try again")
    commit = archive_commit(archive)
    if expected_sha and commit and commit != expected_sha:
        raise UpdateError("the downloaded zip is not the expected commit - try again")
    return archive, commit or expected_sha


def read_state(root: Path) -> dict:
    try:
        return json.loads((root / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_file(dest: Path, data: bytes) -> None:
    """Write next to the target first, then swap it in (no half-written files on a crash)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".pymc-new")
    tmp.write_bytes(data)
    os.replace(tmp, dest)


def apply_zip(archive: bytes, root: Path, stamp: str | None = None) -> dict:
    """Extracts the repository archive over ``root``.

    Files that are not in the archive (config, venv, models, recordings) are left alone, protected
    folders are never written, and a repository file whose content differs (beyond CRLF/LF line
    endings) is backed up to ``.pymc_backup/<stamp>/`` before it is replaced. Returns the written,
    backed-up and skipped paths plus ``files``: a manifest (path -> SHA-256) of everything the
    archive contained, used next time to remove files the project deleted.
    """
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    report: dict = {"written": [], "backed_up": [], "skipped": [], "files": {}}
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
            if rel.is_absolute() or ".." in rel.parts or ":" in rel.parts[0]:
                raise UpdateError(f"refusing unsafe path in archive: {redact(name)}")
            if is_protected(rel):
                report["skipped"].append(str(rel))
                continue
            data = zf.read(name)
            report["files"][str(rel)] = _sha256(data)
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
            _write_file(dest, data)
            mode = (zf.getinfo(name).external_attr >> 16) & 0o777
            if mode & 0o100 and os.name != "nt":
                dest.chmod(dest.stat().st_mode | 0o111)
            report["written"].append(str(rel))
    return report


def remove_deleted(root: Path, old_files: dict, new_files: dict) -> dict[str, list[str]]:
    """Deletes files the project removed since the last zip update - only if you never edited them."""
    removed: list[str] = []
    kept: list[str] = []
    for rel, digest in sorted(old_files.items()):
        if rel in new_files:
            continue
        path_rel = PurePosixPath(rel)
        if path_rel.is_absolute() or ".." in path_rel.parts or is_protected(path_rel):
            continue
        path = root.joinpath(*path_rel.parts)
        if not path.is_file():
            continue
        if _sha256(path.read_bytes()) != digest:
            kept.append(rel)  # you changed it - leave it to you
            continue
        path.unlink()
        removed.append(rel)
        parent = path.parent
        while parent != root and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    return {"removed": removed, "kept": kept}


def update_zip(root: Path, repo: str, ref: str | None, check: bool) -> bool:
    """Updates PyMC_Bot from a downloaded source zip. ``ref`` None = this build's update channel."""
    channel = ref or update_channel()
    where = "the update channel" if ref is None else "the chosen branch"
    if git_checkout_present(root):
        say("a .git folder is here - it is not used: the update comes as a zip and .git is left alone")
    say(f"checking {where} on GitHub ...")
    state = read_state(root)
    archive: bytes | None = None
    try:
        remote = remote_commit(repo, channel)
    except RateLimited:
        # No API calls left: the zip itself says which commit it is.
        say("GitHub's API limit is used up - downloading the zip to compare instead")
        archive, sha = download_archive(repo, channel)
        if not sha:
            raise
        remote = {"sha": sha, "date": "", "message": "(details need the GitHub API)"}
    if state.get("sha") == remote["sha"]:
        say(f"PyMC_Bot is up to date ({local_version(root)}, {remote['sha'][:7]})")
        return False
    known = f"installed {state['sha'][:7]}" if state.get("sha") else "installed version unknown"
    when = f" ({remote['date'][:10]})" if remote.get("date") else ""
    say(f"newest commit {remote['sha'][:7]}{when}: {remote['message']} [{known}]")
    if check:
        return False
    if archive is None:
        say("downloading the source zip ...")
        archive, _ = download_archive(repo, channel, remote["sha"])
    say(f"extracting {len(archive) / 1048576:.1f} MB ...")
    report = apply_zip(archive, root)
    cleanup = remove_deleted(root, state.get("files") or {}, report["files"]) if state.get("files") else None
    (root / STATE_FILE).write_text(json.dumps({
        "repo": repo,
        "channel": "default" if ref is None else "custom",
        **remote,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": report["files"],
    }, indent=2) + "\n", encoding="utf-8")
    say(f"updated {len(report['written'])} file(s) - PyMC_Bot is now {local_version(root)} ({remote['sha'][:7]})")
    if report["backed_up"]:
        say(f"your edited copies of {len(report['backed_up'])} file(s) were saved in {BACKUP_DIR}/ first:")
        for rel in report["backed_up"][:15]:
            say(f"  {rel}")
    if cleanup is None:
        say("first zip update here: files the project deletes from now on are removed on the next update")
    else:
        if cleanup["removed"]:
            say(f"removed {len(cleanup['removed'])} file(s) the project deleted:")
            for rel in cleanup["removed"][:15]:
                say(f"  {rel}")
        if cleanup["kept"]:
            say(f"kept {len(cleanup['kept'])} deleted-upstream file(s) because you edited them:")
            for rel in cleanup["kept"][:15]:
                say(f"  {rel}")
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
            say("package.json changed but npm is not installed - real servers need Node 22+ and `npm install`")
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
        for path in sorted(p for mod_id in MOD_IDS for p in mods_dir.glob(f"{mod_id}-*.jar")):
            parsed = parse_jar_name(path.name)
            if parsed:
                jars.append(InstalledJar(path, *parsed))
    return jars


def release_version(tag: str) -> str | None:
    """'pymcbot-v0.6.0' -> '0.6.0' (also the old 'pymcplaytime-v' tags); None for other tags."""
    for prefix in MOD_TAG_PREFIXES:
        if tag.startswith(prefix):
            return tag[len(prefix):]
    return None


def pick_release(releases: list[dict], allow_pre: bool) -> dict | None:
    """Newest mod release: stable first; pre-releases with --pre or when there is no stable one."""
    mod = [r for r in releases if not r.get("draft") and release_version(str(r.get("tag_name", "")))]
    mod.sort(key=lambda r: version_tuple(release_version(r["tag_name"])), reverse=True)
    stable = [r for r in mod if not r.get("prerelease")]
    if allow_pre or not stable:
        return mod[0] if mod else None
    return stable[0]


def asset_for(release: dict, version: str, mc: str) -> dict | None:
    wanted = {f"{mod_id}-{version}+{mc}.jar" for mod_id in MOD_IDS}
    return next((a for a in release.get("assets", []) if a.get("name") in wanted), None)


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
    if meta.get("id") not in MOD_IDS or meta.get("depends", {}).get("minecraft") != mc:
        raise UpdateError(f"the downloaded jar is not the PyMC Bot mod for Minecraft {mc}")


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
        raise UpdateError("no pymcbot-*.jar in that folder - pass --mc <your Minecraft version> to install it, "
                          "or --mods-dir if your mods live elsewhere")
    release = pick_release(http_json(f"{API}/repos/{repo}/releases?per_page=30"), allow_pre)
    if release is None:
        raise UpdateError(f"{repo} has no {MOD_TAG_PREFIXES[0]}* release yet")
    newest = release_version(release["tag_name"])
    say(f"newest mod release: {release['tag_name']}" + (" (pre-release)" if release.get("prerelease") else ""))
    changed = False
    for mc, current in sorted(targets.items()):
        best = max((j.version for j in current), key=version_tuple, default=None)
        if best and version_tuple(best) >= version_tuple(newest) and not reinstall:
            say(f"Minecraft {mc}: PyMC Bot mod {best} is up to date")
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
        description="Update PyMC_Bot, its dependencies and the PyMC Bot mod from GitHub.",
        epilog="Examples:  updater --check   |   updater --mod   |   updater --mod-only --mc 1.21.4",
    )
    parser.add_argument("--check", action="store_true", help="only report available updates, change nothing")
    parser.add_argument("--mod", action="store_true", help="also update the Fabric mod jar in your mods folder")
    parser.add_argument("--mod-only", action="store_true", help="only update the Fabric mod")
    parser.add_argument("--mc", metavar="VERSION", help="Minecraft version to install the mod for (e.g. 1.21.4, 26.3)")
    parser.add_argument("--mods-dir", type=Path, help="mods folder (default: the official launcher's .minecraft/mods)")
    parser.add_argument("--pre", action="store_true", help="allow mod pre-releases even when a stable release exists")
    parser.add_argument("--reinstall", action="store_true", help="download the mod again even if it is up to date")
    parser.add_argument("--branch", default=None, metavar="NAME",
                        help="update from another branch instead of this build's update channel")
    parser.add_argument("--repo", default=REPO, help=argparse.SUPPRESS)
    parser.add_argument("--seal", metavar="BRANCH", help=argparse.SUPPRESS)  # maintainers: value for _CHANNEL
    deps = parser.add_mutually_exclusive_group()
    deps.add_argument("--deps", action="store_true", help="reinstall dependencies even if unchanged")
    deps.add_argument("--no-deps", action="store_true", help="never touch dependencies")
    return parser


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    args = build_parser().parse_args(argv)
    if args.seal:
        print(seal(args.seal))
        return 0
    if args.mc and not (args.mod or args.mod_only):
        args.mod = True
    say(f"PyMC_Bot updater - {args.repo}" + ("  (check only)" if args.check else ""))
    try:
        if not args.mod_only:
            before = file_hashes(root)
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
    except Exception as exc:  # never show a traceback (it could contain the update channel)
        say(f"ERROR: unexpected {type(exc).__name__}: {exc}")
        return 1
    except KeyboardInterrupt:
        say("cancelled")
        return 130
    say("done")
    return 0


if __name__ == "__main__":
    # Plain ASCII output so old Windows consoles never choke on it.
    raise SystemExit(main())
