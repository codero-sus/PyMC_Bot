#!/usr/bin/env python3
"""Upload the PyMC Bot mod jars to Modrinth - one Modrinth version per Minecraft version.

Build the jars first (``cd mod && ./build-all.sh`` -> ``mod/dist/``) or download them from the
GitHub release, create the project on modrinth.com once, then::

    # see what would be uploaded (no token, no network)
    python scripts/modrinth_publish.py --jars mod/dist --dry-run

    # upload (token from https://modrinth.com/settings/pats with "Create versions" scope)
    MODRINTH_TOKEN=mrp_... python scripts/modrinth_publish.py --project pymc-bot --jars mod/dist

    # also replace the project page text with mod/MODRINTH.md ("Write projects" scope)
    MODRINTH_TOKEN=mrp_... python scripts/modrinth_publish.py --project pymc-bot --jars mod/dist --update-page

Versions already on Modrinth (same ``version_number``) are skipped, so re-running is safe.
Only the standard library is used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
import uuid
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


def multipart(data: dict[str, Any], jar: Path) -> tuple[bytes, str]:
    boundary = f"----pymcbot{uuid.uuid4().hex}"
    head = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"data\"\r\n"
        f"Content-Type: application/json\r\n\r\n{json.dumps(data)}\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{jar.name}\"\r\n"
        f"Content-Type: application/java-archive\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + jar.read_bytes() + tail, f"multipart/form-data; boundary={boundary}"


# --------------------------------------------------------------------------- HTTP
def request(method: str, path: str, token: str | None, body: bytes | None = None,
            content_type: str = "application/json") -> Any:
    req = urllib.request.Request(API + path, data=body, method=method)
    req.add_header("User-Agent", USER_AGENT)
    if token:
        req.add_header("Authorization", token)
    if body is not None:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise SystemExit(f"Modrinth {method} {path} failed: HTTP {exc.code} {detail}") from None
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach Modrinth ({exc.reason})") from None
    return json.loads(raw) if raw else None


def published_numbers(project: str, token: str | None) -> set[str]:
    return {entry.get("version_number", "") for entry in request("GET", f"/project/{project}/version", token) or []}


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--jars", type=Path, default=MOD / "dist", help="folder with pymcbot-<mod>+<mc>.jar (default mod/dist)")
    parser.add_argument("--project", default=os.environ.get("MODRINTH_PROJECT", ""),
                        help="Modrinth project slug or id (or MODRINTH_PROJECT)")
    parser.add_argument("--channel", choices=("release", "beta", "alpha"), default="release")
    parser.add_argument("--changelog", type=Path,
                        help="markdown file used as the changelog (default: this version's section of mod/CHANGELOG.md)")
    parser.add_argument("--only", action="append", default=[], metavar="MC", help="upload only this Minecraft version (repeatable)")
    parser.add_argument("--update-page", action="store_true", help="also set the project description to mod/MODRINTH.md")
    parser.add_argument("--dry-run", action="store_true", help="print what would be uploaded; no token or network needed")
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

    token = os.environ.get("MODRINTH_TOKEN", "").strip()
    if not args.dry_run:
        if not args.project:
            print("--project (or MODRINTH_PROJECT) is required", file=sys.stderr)
            return 2
        if not token:
            print("Set MODRINTH_TOKEN to a Modrinth personal access token.", file=sys.stderr)
            return 2
    done = set() if args.dry_run else published_numbers(project, token)

    uploaded = 0
    for mc in order:
        data = version_payload(project, mod, mc, channel=args.channel, changelog=changelog, featured=mc == newest)
        if data["version_number"] in done:
            print(f"= {data['version_number']} already on Modrinth")
            continue
        if args.dry_run:
            print(f"+ {data['version_number']:<16} {jars[mc].name}  ({jars[mc].stat().st_size // 1024} KiB)")
            continue
        body, content_type = multipart(data, jars[mc])
        request("POST", "/version", token, body, content_type)
        uploaded += 1
        print(f"+ uploaded {data['version_number']}")

    if args.update_page:
        page = (MOD / "MODRINTH.md").read_text(encoding="utf-8")
        if args.dry_run:
            print(f"would update the project page ({len(page)} chars from mod/MODRINTH.md)")
        else:
            request("PATCH", f"/project/{project}", token, json.dumps({"body": page}).encode())
            print("updated the project page")
    print(f"{'dry run: ' if args.dry_run else ''}{len(order)} jar(s), {uploaded} uploaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
