"""Update a virtualenv-installed wheel from verified GitHub release assets."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Any

REPO = "https://github.com/codero-sus/PyMC_Bot/releases/download/"
API = "https://api.github.com/repos/codero-sus/PyMC_Bot/releases?per_page=30"
WHEEL = re.compile(r"pymc_bot-([0-9]+(?:\.[0-9]+)+)-py3-none-any\.whl")


def eligible() -> bool:
    """Never mutate system Python, Git checkouts, or an editable installation."""
    root = Path(__file__).resolve().parent.parent
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix) or (root / ".git").exists():
        return False
    try:
        dist = importlib.metadata.distribution("pymc-bot")
        if dist.read_text("direct_url.json"):
            origin = json.loads(dist.read_text("direct_url.json") or "{}")
            if origin.get("dir_info", {}).get("editable"):
                return False
        return Path(dist.locate_file("pymc_bot/__init__.py")).resolve() == Path(__file__).resolve().parent / "__init__.py"
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError):
        return False


def _version(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", value):
        raise ValueError("unsupported version")
    parts = [int(part) for part in value.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _get(url: str, limit: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "PyMC-Bot-package-updater"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("release asset too large")
    return data


def update(*, check: bool = False) -> bool:
    """Return whether a newer release wheel is available or installed. Never run in a checkout."""
    if not eligible():
        return False
    try:
        installed = _version(importlib.metadata.version("pymc-bot"))
        releases: Any = json.loads(_get(API, 2_000_000))
        if not isinstance(releases, list):
            return False
        candidates = []
        for release in releases:
            if not isinstance(release, dict) or release.get("draft") or release.get("prerelease"):
                continue
            for asset in release.get("assets", []):
                if not isinstance(asset, dict):
                    continue
                match = WHEEL.fullmatch(str(asset.get("name", "")))
                if not match or _version(match.group(1)) <= installed:
                    continue
                url, digest, size = asset.get("browser_download_url"), asset.get("digest"), asset.get("size")
                parsed = urllib.parse.urlsplit(str(url))
                if (parsed.scheme != "https" or parsed.hostname != "github.com" or parsed.username or parsed.password
                        or not str(url).startswith(REPO) or not isinstance(digest, str)
                        or not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest)
                        or isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= 50_000_000):
                    continue
                candidates.append((_version(match.group(1)), asset))
        if not candidates:
            return False
        _, asset = max(candidates, key=lambda item: item[0])
        if check:
            return True
        data = _get(asset["browser_download_url"], 50_000_000)
        if len(data) != asset["size"] or hashlib.sha256(data).hexdigest().lower() != asset["digest"][7:].lower():
            return False
        with tempfile.TemporaryDirectory(prefix="pymcbot-wheel-") as temp:
            wheel = Path(temp) / asset["name"]
            wheel.write_bytes(data)
            with zipfile.ZipFile(wheel) as archive:
                names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
                if len(names) != 1 or archive.getinfo(names[0]).file_size > 1_000_000:
                    return False
                metadata = BytesParser().parsebytes(archive.read(names[0]))
                version = WHEEL.fullmatch(asset["name"]).group(1)
                if str(metadata.get("Name", "")).lower().replace("_", "-") != "pymc-bot" or metadata.get("Version") != version:
                    return False
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--no-input",
                            "--disable-pip-version-check", str(wheel)], check=True,
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        logging.getLogger(__name__).warning("Automatic update check failed (%s); retaining installed version", type(exc).__name__)
        return False


def restart() -> None:
    os.environ["PYMC_BOT_UPDATE_RESTARTED"] = "1"
    os.execv(sys.executable, [sys.executable, "-m", "pymc_bot", *sys.argv[1:]])
