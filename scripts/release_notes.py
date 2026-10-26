#!/usr/bin/env python3
"""Write the GitHub release notes for the PyMC Bot mod.

CI's ``release`` job attaches one jar per Minecraft version to the release
``pymcbot-v<mod_version>`` and uses this script for the description, so the
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
import os
import re
import subprocess
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
    return f"pymcbot-v{mod_version(mod_dir)}"


def jar_name(version: str, minecraft: str) -> str:
    return f"pymcbot-{version}+{minecraft}.jar"


def wheel_version() -> str:
    for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "version":
            return value.strip().strip('"')
    return "0.0.0"


NOTEBOOK_PATH = "pynb/PyMC_Minecraft_LLM_Kaggle.ipynb"


def commit_sha() -> str:
    """The full commit this release is built from (CI: GITHUB_SHA), or "" when unknown."""
    sha = os.environ.get("GITHUB_SHA", "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                 check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            sha = ""
    return sha if re.fullmatch(r"[0-9a-f]{40}", sha) else ""


def notebook_url(sha: str | None = None) -> str:
    """The Kaggle notebook pinned to a commit (never a branch). The notebook is not a release asset any more:
    Kaggle imports it straight from this link (File > Import Notebook)."""
    sha = commit_sha() if sha is None else sha
    return f"{REPO}/raw/{sha}/{NOTEBOOK_PATH}" if sha else f"{REPO}/blob/main/pynb/README.md"


def downloads(version: str) -> list[str]:
    """Every file on the release besides the per-version jars (built by scripts/release_kit.py)."""
    wheel = wheel_version()
    return [
        "### Downloads",
        "",
        "| File | What it is |",
        "|---|---|",
        "| `PyMC_Bot-source.zip` | **The bot** (Python): unzip, `pip install -r requirements.txt`, "
        "`python -m pymc_bot` opens the control panel on http://localhost:8000 |",
        f"| `pymc_bot-{wheel}-py3-none-any.whl` / `pymc_bot-{wheel}.tar.gz` | the bot as a pip package: "
        f"`pip install pymc_bot-{wheel}-py3-none-any.whl` |",
        "| `updater.py` · `updater.cmd` · `updater.sh` | **The updater**: put it in your PyMC_Bot folder and run it "
        "(Windows: double-click `updater.cmd`) - updates the bot and installs the right mod jar |",
        "| `pymcbot-<version>+<minecraft>.jar` | **The mod**, one per Minecraft version (table below) |",
        f"| `pymcbot-{version}-all-versions.zip` | all mod jars in one zip |",
        "| `pymc-minecraft-llm-kit.zip` | **Your own Minecraft LLM**: ready-made training data (mob database, "
        "game situations, Minecraft Wiki pages) - upload `llm_data` to Kaggle |",
        "| `train_minecraft_llm.py` | **Train Minecraftism** on Kaggle "
        "(GPU T4 x2): the trainer the notebook runs. The notebook is not attached - import it on Kaggle "
        f"(File → Import Notebook → paste) from `{notebook_url()}` (this release's commit). Pick what to train with "
        "`editions_to_be_trained = {txt, txt_think, vl, vl_think}` - only those are trained. **Both T4s train** every edition together "
        "(`GPU_MODE = \"parallel\"`: one edition per T4, both T4s merged on a single edition via HF Accelerate DDP, fp16; "
        "`\"merge\"` = both T4s on every edition). `DISTILL` runs Qwen3's "
        "strong-to-weak distillation: Qwen3-14B writes answers (on vLLM across both T4s, kept on Hugging Face for the next session), then Qwen3-8B teaches the 4B token by token (on-policy reverse KL). `MAX_HOURS` stops training in time for the export and upload "
        "before Kaggle's 12 h limit, `FAST` sizes batches by tokens, streams shared data once and makes the GGUFs on the "
        "CPU (uploading meanwhile) while the T4 trains the next edition, and `AUTO_EXIT` ends the session when everything is on Hugging Face. Results, logs and the shared data cache stay in `/kaggle/working`; downloads, GGUF work files and other throwaway files go to `/kaggle/tmp`. Every edition is uploaded to Hugging Face **twice**: "
        "**safetensors** first (`<you>/Minecraftism-4B-Uncensored`, the compact 16-bit merged model for "
        "transformers / vLLM / fine-tuning; `SAFETENSORS = True`), then **GGUF** quants made from it "
        "(`<you>/Minecraftism-4B-Uncensored-GGUF`: Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, each uploaded as soon as it "
        "is ready; `GGUF_QUANTS`), both with `tokenizer.json` and the configs. Afterwards `python -m pymc_bot llm import --hf <you>/Minecraftism-4B-Uncensored-GGUF` "
        "(guide: `docs/MINECRAFT_LLM.md` in the source zip) |",
        "| `pymc-bot-node-modules.zip` | **mineflayer, prebuilt** for players who can't (or don't want to) use npm: "
        "`python -m pymc_bot setup` fetches it by itself when needed, or offline: "
        "`python -m pymc_bot setup --zip pymc-bot-node-modules.zip` |",
        "| `SHA256SUMS.txt` | checksums of every file above |",
        "",
    ]


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
            f"## PyMC Bot {version}",
            "",
            *downloads(version),
            "## The mod",
            "",
            f"The PyMC Bot companion mod records your Minecraft playtime so PyMC Bot can learn to play "
            f"like you. One jar per "
            f"Minecraft version, **{first} to {last}** (Fabric).",
            "",
            "### Install",
            "",
            "1. Install [Fabric Loader](https://fabricmc.net/use/installer/) "
            f"{table['loader']} or newer for your Minecraft version.",
            "2. Put **Fabric API** for the same version in `.minecraft/mods/`.",
            "3. Download the jar for your exact Minecraft version below and put it in `.minecraft/mods/`.",
            "4. In game: open the **PyMC Bot settings page** (`/pymcbot settings`, a hotkey or Mod Menu's "
            "*Configure* button) and press *Start recording* - or use `/pymcbot record start`, play, "
            "`/pymcbot record stop`, `/pymcbot export`.",
            "5. Already have a `pymcplaytime-*.jar` (0.5.0 and older)? Delete it - this release replaces it "
            "(`updater --mod-only` does that for you). Your settings move over automatically.",
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
            "(`/pymcbot hud on|off`, `/pymcbot hud corner <corner>`).",
            "- **PyMC Bot settings page** (`/pymcbot settings`, a hotkey or Mod Menu): tabs for the recorder "
            "controls, recording (sample rate, menus, AFK skipping, auto-record, block area, nearby "
            "entities, thresholds), advanced entity + item limits, the HUD badge with a live preview, "
            "and an About page. Hover a button for help; each page has *Reset page*.",
            "- Pause/resume, auto-record on join, menu and AFK filtering, entity + item recording.",
            "",
            f"Train on the dataset with PyMC Bot: see the [training guide]({REPO}/blob/main/docs/TRAINING.md). "
            f"Build it yourself: [compiling the mod]({REPO}/blob/main/mod/README.md#compiling-the-mod).",
            "",
            "Made with \u2764\ufe0f by [codero-sus](https://github.com/codero-sus) - https://codero.is-a.dev - https://the-codero.web.app/",
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
