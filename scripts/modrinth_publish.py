#!/usr/bin/env python3
"""Upload the PyMC Bot mod jars to Modrinth - one Modrinth version per Minecraft version.

Build the jars first (``cd mod && ./build-all.sh`` -> ``mod/dist/``) or download them from the
GitHub release, create the project on modrinth.com once, then::

    # see what would be uploaded (no token, no network)
    python scripts/modrinth_publish.py --jars mod/dist --dry-run

    # publish (token from https://modrinth.com/settings/pats - see mod/README.md for the scopes)
    MODRINTH_TOKEN=mrp_... python scripts/modrinth_publish.py --project pymcbot --jars mod/dist --update-project

Every run brings Modrinth fully up to date: missing versions are created; existing versions get the
current name, changelog, game version, loader, dependencies and channel, and their jar is replaced when
its contents changed. ``--update-project`` also sets the title, summary, description (mod/MODRINTH.md),
categories, links and licence from mod/modrinth.json. Re-running is safe.
Only the standard library is used.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "mod"
API = "https://api.modrinth.com/v2"
REPO = "https://github.com/codero-sus/PyMC_Bot"
# Modrinth asks every API client for a User-Agent that identifies the project and a contact.
USER_AGENT = "codero-sus/PyMC_Bot (https://the-codero.web.app/)"

FABRIC_API = "P7dR8mSH"  # modrinth.com/mod/fabric-api
MOD_MENU = "mOgUt4GM"  # modrinth.com/mod/modmenu (optional: adds the Configure button)

JAR_RE = re.compile(r"^pymcbot-(?P<mod>\d+\.\d+\.\d+(?:[-.][0-9A-Za-z.]+)?)\+(?P<mc>[0-9][0-9A-Za-z.\-]*)\.jar$")


# --------------------------------------------------------------------------- inputs
def mod_version(mod_dir: Path = MOD) -> str:
    for line in (mod_dir / "gradle.properties").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "mod_version":
            return value.strip()
    raise SystemExit("mod_version is missing from mod/gradle.properties")


def supported_versions(mod_dir: Path = MOD) -> list[str]:
    data = json.loads((mod_dir / "versions.json").read_text(encoding="utf-8"))
    return [entry["minecraft"] for entry in data["versions"]]


def changelog_for(mod: str, path: Path = MOD / "CHANGELOG.md") -> str:
    """The ``## <mod>`` section of mod/CHANGELOG.md (empty when there is none)."""
    if not path.is_file():
        return ""
    section: list[str] = []
    inside = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if inside:
                break
            inside = line[3:].strip() == mod
            continue
        if inside:
            section.append(line)
    return "\n".join(section).strip()


def find_jars(folder: Path, version: str | None = None) -> dict[str, Path]:
    """``{minecraft_version: jar}`` for every ``pymcbot-<mod>+<mc>.jar`` (gametest/sources jars are ignored)."""
    jars: dict[str, Path] = {}
    for jar in sorted(Path(folder).glob("*.jar")):
        if jar.stem.endswith(("-sources", "-dev", "-javadoc")):
            continue
        match = JAR_RE.match(jar.name)
        if not match or (version and match["mod"] != version):
            continue
        jars[match["mc"]] = jar
    return jars


def version_payload(
    project: str,
    mod: str,
    minecraft: str,
    *,
    channel: str = "release",
    changelog: str = "",
    featured: bool = False,
) -> dict[str, Any]:
    """The ``data`` part of Modrinth's create-version request for one jar."""
    return {
        "project_id": project,
        "name": f"PyMC Bot {mod} for Minecraft {minecraft}",
        "version_number": f"{mod}+{minecraft}",
        "changelog": changelog or f"See {REPO}/releases/tag/pymcbot-v{mod}",
        "game_versions": [minecraft],
        "loaders": ["fabric"],
        "version_type": channel,
        "featured": featured,
        "status": "listed",
        "environment": "client_only",
        "dependencies": [
            {"project_id": FABRIC_API, "dependency_type": "required"},
            {"project_id": MOD_MENU, "dependency_type": "optional"},
        ],
        "file_parts": ["file"],
        "primary_file": "file",
    }


def multipart(data: dict[str, Any], jar: Path, filename: str | None = None) -> tuple[bytes, str]:
    boundary = f"----pymcbot{uuid.uuid4().hex}"
    head = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"data\"\r\n"
        f"Content-Type: application/json\r\n\r\n{json.dumps(data)}\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename or jar.name}\"\r\n"
        f"Content-Type: application/java-archive\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + jar.read_bytes() + tail, f"multipart/form-data; boundary={boundary}"


# --------------------------------------------------------------------------- HTTP
RETRY_STATUS = {429, 500, 502, 503, 504}


def _retry_delay(headers: Any, attempt: int) -> float:
    """Seconds to wait before the next try: Retry-After / X-Ratelimit-Reset, else 2, 4, 8, 16 s."""
    for name in ("Retry-After", "X-Ratelimit-Reset"):
        value = headers.get(name) if headers is not None else None
        try:
            if value is not None:
                return min(max(float(value), 1.0), 90.0)
        except ValueError:
            pass
    return float(2 ** (attempt + 1))


def request(method: str, path: str, token: str | None, body: bytes | None = None,
            content_type: str = "application/json", attempts: int = 5, sleep: Any = time.sleep) -> Any:
    """One API call. Rate limits (HTTP 429, 300 requests/minute per IP) and server hiccups are retried."""
    for attempt in range(attempts):
        req = urllib.request.Request(API + path, data=body, method=method)
        req.add_header("User-Agent", USER_AGENT)
        if token:
            req.add_header("Authorization", token)
        if body is not None:
            req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                raw = response.read()
            return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if exc.code in RETRY_STATUS and attempt + 1 < attempts:
                delay = _retry_delay(exc.headers, attempt)
                print(f"  Modrinth answered HTTP {exc.code} - retrying in {delay:.0f}s", file=sys.stderr)
                sleep(delay)
                continue
            raise SystemExit(f"Modrinth {method} {path} failed: HTTP {exc.code} {detail}") from None
        except urllib.error.URLError as exc:
            if attempt + 1 < attempts:
                sleep(_retry_delay(None, attempt))
                continue
            raise SystemExit(f"Could not reach Modrinth ({exc.reason})") from None
    raise SystemExit(f"Modrinth {method} {path} failed after {attempts} attempts")


def project_settings(mod_dir: Path = MOD) -> dict[str, Any]:
    """The project fields synced on every run: mod/modrinth.json + mod/MODRINTH.md as the description."""
    raw = json.loads((mod_dir / "modrinth.json").read_text(encoding="utf-8"))
    versions = supported_versions(mod_dir)
    fill = {"first_mc": versions[0], "last_mc": versions[-1], "mod_version": mod_version(mod_dir)}
    fields: dict[str, Any] = {
        "title": raw["title"],
        "description": raw["summary"].format(**fill),  # Modrinth calls the summary "description"
        "body": (mod_dir / "MODRINTH.md").read_text(encoding="utf-8"),
    }
    for key in ("categories", "additional_categories", "source_url", "issues_url", "wiki_url", "license_id", "license_url"):
        if raw.get(key):
            fields[key] = raw[key]
    if len(fields["title"]) > 64 or not 3 <= len(fields["description"]) <= 256:
        raise SystemExit("mod/modrinth.json: title must be <= 64 and summary 3-256 characters")
    return fields


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()  # noqa: S324 - Modrinth identifies files by sha1


def jar_digest(data: bytes) -> str:
    """Hash of the jar *contents* (names + bytes), ignoring zip timestamps, so a rebuild of unchanged
    code is not mistaken for a new jar."""
    digest = hashlib.sha256()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith("/"):
                continue
            digest.update(name.encode() + b"\0" + archive.read(name) + b"\0")
    return digest.hexdigest()


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()


def primary_file(version: dict[str, Any]) -> dict[str, Any] | None:
    files = version.get("files") or []
    return next((f for f in files if f.get("primary")), files[0] if files else None)


def same_contents(remote: dict[str, Any], local: bytes, fetch: Callable[[str], bytes]) -> bool:
    """True when the file on Modrinth holds exactly the local jar's classes and resources."""
    if remote.get("hashes", {}).get("sha1") == sha1(local):
        return True
    try:
        return jar_digest(fetch(remote["url"])) == jar_digest(local)
    except Exception as exc:  # unreachable CDN or a broken zip -> treat as changed and replace it
        print(f"  could not compare with the file on Modrinth ({exc}) - replacing it", file=sys.stderr)
        return False


def metadata(data: dict[str, Any]) -> dict[str, Any]:
    """Fields of a create-version payload that PATCH /version/{id} accepts."""
    keys = ("name", "changelog", "game_versions", "loaders", "version_type", "featured", "dependencies", "status")
    return {key: data[key] for key in keys}


def update_project(project: str, token: str, settings: dict[str, Any], call: Callable[..., Any] = None) -> None:
    call = call or request
    try:
        call("PATCH", f"/project/{project}", token, json.dumps(settings).encode())
    except SystemExit as exc:
        if "HTTP 400" not in str(exc):
            raise
        # One rejected optional field (e.g. a category) must not block the description/summary update.
        print(f"  Modrinth rejected a project setting ({exc}); updating title, summary and description only",
              file=sys.stderr)
        basic = {key: settings[key] for key in ("title", "description", "body")}
        call("PATCH", f"/project/{project}", token, json.dumps(basic).encode())


def _file_hash(entry: dict[str, Any]) -> str:
    return (entry.get("hashes") or {}).get("sha1", "")


FILE_RE = re.compile(r"^pymcbot-(?P<mod>[^+]+)\+(?P<mc>[0-9][0-9A-Za-z.\-]*?)(?:-[0-9a-f]{7})?\.jar$")


def _jar_mc(filename: str) -> str | None:
    """Minecraft version of a ``pymcbot-<mod>+<mc>[-<sha7>].jar`` file name (None for any other file)."""
    if (filename or "").endswith(("-sources.jar", "-dev.jar", "-javadoc.jar")):
        return None
    match = FILE_RE.match(filename or "")
    return match["mc"] if match else None


def upload_name(jar: Path, taken: set[str], file_hash: str) -> str:
    """Modrinth refuses a second file with the same name in one version ("Duplicate files are not
    allowed"), so a replacement jar gets ``-<sha7>`` while the old one is still there."""
    if jar.name not in taken:
        return jar.name
    return f"{jar.stem}-{file_hash[:7]}{jar.suffix}"


def explain_duplicate(call: Callable[..., Any], token: str, file_hash: str) -> str:
    """Where Modrinth already has this exact file - for the log when it refuses an upload."""
    try:
        found = call("GET", f"/version_file/{file_hash}?algorithm=sha1", token) or {}
        return f" (this file is already in project {found.get('project_id')}, version {found.get('version_number')})"
    except SystemExit:
        return ""


def describe(remote: list[dict[str, Any]]) -> None:
    """Log what is on Modrinth right now - makes a failed sync easy to understand from the CI log."""
    print(f"Modrinth has {len(remote)} version(s):")
    for version in remote:
        files = ", ".join(
            f"{f.get('filename')}{' (primary)' if f.get('primary') else ''} sha1 {_file_hash(f)[:10]}"
            for f in version.get("files") or []
        )
        print(f"  {version.get('version_number')} [{version.get('version_type')}] id {version.get('id')}: {files or 'no files'}")


def _delete_file(call: Callable[..., Any], token: str, version_id: str, file_hash: str) -> None:
    call("DELETE", f"/version_file/{file_hash}?algorithm=sha1&version_id={version_id}", token)


def sync_versions(
    project: str,
    payloads: list[tuple[dict[str, Any], Path]],
    token: str,
    *,
    call: Callable[..., Any] = None,
    fetch: Callable[[str], bytes] = download,
) -> dict[str, int]:
    """Create missing versions and bring every existing one up to date (details + jar). Returns counts.

    A problem with one version (e.g. Modrinth refusing a file) is reported and the others still go
    through; ``counts["failed"]`` tells the caller to fail the run.
    """
    call = call or request
    remote = call("GET", f"/project/{project}/version", token) or []
    describe(remote)
    by_number = {v.get("version_number"): v for v in remote}
    ours = {data["version_number"]: (data, jar) for data, jar in payloads}
    counts = {"created": 0, "updated": 0, "replaced": 0, "skipped": 0, "failed": 0}

    # 1. Clean-up: a version of ours that also carries jars for *other* Minecraft versions (e.g. several
    #    jars uploaded by hand into one version) would block those jars - Modrinth refuses duplicates.
    for number, version in by_number.items():
        if number not in ours:
            continue
        mc = ours[number][0]["game_versions"][0]
        files = list(version.get("files") or [])
        kept = list(files)
        for entry in files:
            other = _jar_mc(entry.get("filename", ""))
            if other and other != mc and len(kept) > 1:
                _delete_file(call, token, version["id"], _file_hash(entry))
                kept.remove(entry)
                print(f"- removed {entry.get('filename')} from {number} (it belongs to Minecraft {other})")
        version["files"] = kept

    where = {_file_hash(f): v.get("version_number") for v in remote for f in v.get("files") or []}

    for data, jar in payloads:
        number, local = data["version_number"], jar.read_bytes()
        local_hash = sha1(local)
        existing = by_number.get(number)
        try:
            if existing is None:
                if local_hash in where:  # Modrinth refuses the same file twice
                    print(f"! {number}: this exact jar is already on Modrinth as {where[local_hash]} - skipped")
                    counts["skipped"] += 1
                    continue
                body, content_type = multipart(data, jar)
                call("POST", "/version", token, body, content_type)
                print(f"+ {number} created")
                counts["created"] += 1
                continue

            patch = metadata(data)
            files = existing.get("files") or []
            hashes = {_file_hash(f) for f in files}
            old = primary_file(existing)
            if local_hash in hashes:
                action = "details updated (jar already there)"
                counts["updated"] += 1
            elif old is not None and same_contents(old, local, fetch):
                local_hash = _file_hash(old)  # same classes, other build time: keep the file
                action = "details updated (jar unchanged)"
                counts["updated"] += 1
            else:
                name = upload_name(jar, {f.get("filename", "") for f in files}, local_hash)
                body, content_type = multipart({}, jar, name)
                call("POST", f"/version/{existing['id']}/file", token, body, content_type)
                action = "jar replaced and details updated"
                counts["replaced"] += 1
            patch["primary_file"] = ["sha1", local_hash]
            call("PATCH", f"/version/{existing['id']}", token, json.dumps(patch).encode())
            # Older builds of this jar are removed; files that are not PyMC Bot jars (e.g. sources) stay.
            this_mc = data["game_versions"][0]
            for entry in files:
                stale, jar_mc = _file_hash(entry), _jar_mc(entry.get("filename", ""))
                if stale and stale != local_hash and (jar_mc == this_mc or (jar_mc is None and entry.get("primary"))):
                    _delete_file(call, token, existing["id"], stale)
            print(f"~ {number} {action}")
        except SystemExit as exc:
            hint = explain_duplicate(call, token, local_hash) if "Duplicate files" in str(exc) else ""
            print(f"! {number} failed: {exc}{hint}")
            counts["failed"] += 1

    # Only the newest Minecraft version of the current release stays featured.
    for version in remote:
        if version.get("featured") and version.get("version_number") not in ours:
            try:
                call("PATCH", f"/version/{version['id']}", token, json.dumps({"featured": False}).encode())
                print(f"- {version.get('version_number')} no longer featured")
            except SystemExit as exc:
                print(f"! could not un-feature {version.get('version_number')}: {exc}")
    return counts


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--jars", type=Path, default=MOD / "dist", help="folder with pymcbot-<mod>+<mc>.jar (default mod/dist)")
    parser.add_argument("--project", default=os.environ.get("MODRINTH_PROJECT", ""),
                        help="Modrinth project slug or id (or MODRINTH_PROJECT)")
    parser.add_argument("--channel", choices=("release", "beta", "alpha"), default="release")
    parser.add_argument("--changelog", type=Path,
                        help="markdown file used as the changelog (default: this version's section of mod/CHANGELOG.md)")
    parser.add_argument("--only", action="append", default=[], metavar="MC", help="publish only this Minecraft version (repeatable)")
    parser.add_argument("--update-project", "--update-page", dest="update_project", action="store_true",
                        help="also set title, summary, description (mod/MODRINTH.md), links and licence from mod/modrinth.json")
    parser.add_argument("--dry-run", action="store_true", help="print what would be published; no token or network needed")
    args = parser.parse_args(argv)

    mod = mod_version()
    expected = supported_versions()
    jars = find_jars(args.jars, mod)
    if args.only:
        jars = {mc: jar for mc, jar in jars.items() if mc in args.only}
    if not jars:
        print(f"No pymcbot-{mod}+<mc>.jar in {args.jars} - build them with mod/build-all.sh first.", file=sys.stderr)
        return 1
    unknown = sorted(set(jars) - set(expected))
    if unknown:
        print(f"Not in mod/versions.json, skipped: {', '.join(unknown)}", file=sys.stderr)
    order = [mc for mc in expected if mc in jars]
    missing = [mc for mc in expected if mc not in jars and not args.only]
    if missing:
        print(f"Note: no jar for {', '.join(missing)}", file=sys.stderr)
    changelog = args.changelog.read_text(encoding="utf-8") if args.changelog else changelog_for(mod)
    newest = expected[-1]
    project = args.project or "<project>"
    settings = project_settings() if args.update_project else None

    token = os.environ.get("MODRINTH_TOKEN", "").strip()
    if not args.dry_run:
        if not args.project:
            print("--project (or MODRINTH_PROJECT) is required", file=sys.stderr)
            return 2
        if not token:
            print("Set MODRINTH_TOKEN to a Modrinth personal access token.", file=sys.stderr)
            return 2

    payloads = [
        (version_payload(project, mod, mc, channel=args.channel, changelog=changelog, featured=mc == newest), jars[mc])
        for mc in order
    ]
    if args.dry_run:
        for data, jar in payloads:
            print(f"* {data['version_number']:<16} {jar.name}  ({jar.stat().st_size // 1024} KiB) - create or update")
        if settings:
            print(f"would update the project: title {settings['title']!r}, summary {settings['description']!r}, "
                  f"description {len(settings['body'])} chars, {', '.join(sorted(set(settings) - {'title', 'description', 'body'}))}")
        print(f"dry run: {len(payloads)} jar(s).")
        return 0

    if settings:
        update_project(project, token, settings)
        print("updated the project: title, summary, description, links and licence")
    counts = sync_versions(project, payloads, token)
    print(f"{len(payloads)} jar(s): {counts['created']} created, {counts['replaced']} jar(s) replaced, "
          f"{counts['updated']} details-only updates, {counts['skipped']} skipped, {counts['failed']} failed.")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
