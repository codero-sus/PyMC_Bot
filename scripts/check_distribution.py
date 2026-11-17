#!/usr/bin/env python3
"""Check the actual publishable archives, including the notebook exclusion."""
from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

REQUIRED = {
    "pymc_bot/__main__.py", "pymc_bot/node/minecraft_bridge.js", "pymc_bot/node/potions.js",
    "pymc_bot/node/probe.js", "pymc_bot/node/safety.js", "pymc_bot/node/package.json", "pymc_bot/data/mobs.json",
    "pymc_bot/web/index.html", "pymc_bot/web/login.html", "pymc_bot/web/login.js",
    "pymc_bot/web/app.js", "pymc_bot/web/styles.css", "pymc_bot/web/fx.js",
}


def check(directory: Path) -> None:
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1 or len(list(directory.iterdir())) != 2:
        raise ValueError("expected exactly one wheel and one sdist")
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = set(archive.namelist())
    with tarfile.open(sdists[0], "r:gz") as archive:
        entries = [entry.name for entry in archive.getmembers() if entry.isfile()]
    roots = {name.split("/", 1)[0] for name in entries}
    if len(roots) != 1 or any("/" not in name for name in entries):
        raise ValueError("unexpected sdist structure")
    sdist_names = {name.split("/", 1)[1] for name in entries}
    for label, names in (("wheel", wheel_names), ("sdist", sdist_names)):
        missing = REQUIRED - names
        if missing:
            raise ValueError(f"{label} is missing runtime files: {sorted(missing)}")
        forbidden = [name for name in names if name.lower().endswith(".ipynb") or
                     name.startswith(("pynb/", "tests/", "scripts/", "mod/"))]
        if forbidden:
            raise ValueError(f"{label} includes forbidden files: {sorted(forbidden)}")
        if not any(name == "LICENSE" or name.endswith("/LICENSE") for name in names):
            raise ValueError(f"{label} is missing its license")
    print(f"Checked {wheels[0].name} and {sdists[0].name}; runtime assets present, notebooks excluded")


if __name__ == "__main__":
    check(Path(sys.argv[1] if len(sys.argv) > 1 else "dist"))
