#!/usr/bin/env python3
"""Write the GitHub release notes for the PyMC Playtime Recorder mod.

CI's ``release`` job attaches one jar per Minecraft version to the release
``pymcplaytime-v<mod_version>`` and uses this script for the description, so the
"which jar do I download?" table always matches ``mod/versions.json``.

Usage::

    python scripts/release_notes.py                 # print the notes
    python scripts/release_notes.py --tag           # print only the tag name
    python scripts/release_notes.py -o notes.md     # write them to a file

Only the standard library is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "mod"
REPO = "https://github.com/codero-sus/PyMC_Bot"


def mod_version(mod_dir: Path = MOD) -> str:
    for line in (mod_dir / "gradle.properties").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "mod_version":
            return value.strip()
    raise SystemExit("mod_version is missing from mod/gradle.properties")


def tag_name(mod_dir: Path = MOD) -> str:
    return f"pymcplaytime-v{mod_version(mod_dir)}"


def jar_name(version: str, minecraft: str) -> str:
    return f"pymcplaytime-{version}+{minecraft}.jar"


def notes(mod_dir: Path = MOD) -> str:
    version = mod_version(mod_dir)
    table = json.loads((mod_dir / "versions.json").read_text(encoding="utf-8"))
    rows = []
    for entry in reversed(table["versions"]):
        tested = "launched in CI" if entry.get("gametest") else "compiled in CI"
        rows.append(
            f"| {entry['minecraft']} | `{jar_name(version, entry['minecraft'])}` "
            f"| {entry['fabric_api']} or newer | {entry['java']} | {tested} |"
        )
    first, last = table["versions"][0]["minecraft"], table["versions"][-1]["minecraft"]
    return "\n".join(
        [
            f"## PyMC Playtime Recorder {version}",
            "",
            f"Records your Minecraft playtime so PyMC Bot can learn to play like you. One jar per "
            f"Minecraft version, **{first} to {last}** (Fabric).",
            "",
            "### Install",
            "",
            "1. Install [Fabric Loader](https://fabricmc.net/use/installer/) "
            f"{table['loader']} or newer for your Minecraft version.",
            "2. Put **Fabric API** for the same version in `.minecraft/mods/`.",
            "3. Download the jar for your exact Minecraft version below and put it in `.minecraft/mods/`.",
            "4. In game: `/pymc record start`, play, `/pymc record stop`, `/pymc export` - "
            "or open `/pymc settings`.",
            "",
            "Each jar only loads on the Minecraft version in its name.",
            "",
            "| Minecraft | Jar | Fabric API | Java | Tested |",
            "|---|---|---|---|---|",
            *rows,
            "",
            "\"launched in CI\" means the CI starts that version of the game with the jar, records a "
            "few seconds of scripted play and checks the dataset and screenshots (Fabric's client game "
            "test API exists from 1.21.4). Older versions are compile-checked.",
            "",
            "### What's in it",
            "",
            "- On-screen **REC / PAUSED / IDLE / MENU** badge with the run clock and sample count "
            "(`/pymc hud on|off`, `/pymc hud corner <corner>`).",
            "- **Settings screen** (`/pymc settings` or a hotkey): start/stop, pause, the badge, menu "
            "pausing, AFK skipping, sample rate, entities + items, auto-record, chat notices, export.",
            "- Pause/resume, auto-record on join, menu and AFK filtering, entity + item recording.",
            "",
            f"Train on the dataset with PyMC Bot: see the [training guide]({REPO}/blob/main/docs/TRAINING.md). "
            f"Build it yourself: [compiling the mod]({REPO}/blob/main/mod/README.md#compiling-the-mod).",
            "",
            "Made with \u2764\ufe0f by [codero-sus](https://github.com/codero-sus) - https://the-codero.web.app/",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", action="store_true", help="print only the release tag")
    parser.add_argument("-o", "--output", help="write the notes to this file")
    args = parser.parse_args(argv)
    if args.tag:
        print(tag_name())
        return 0
    text = notes()
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
