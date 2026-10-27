#!/usr/bin/env python3
"""Writes pynb/Minecraftism_Dataset_Kaggle.ipynb: settings, setup, pymc_bot/llm_webdata.py verbatim, run.

Run it after changing pymc_bot/llm_webdata.py (tests/test_webdata.py fails while the notebook is out of date).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "pymc_bot" / "llm_webdata.py"
NOTEBOOK = ROOT / "pynb" / "Minecraftism_Dataset_Kaggle.ipynb"

INTRO = """# Minecraftism dataset - trusted Minecraft Q&A, collected on Kaggle (CPU edition: no GPU hours)

The game's own data + web search + an LLM on llama.cpp (Qwen3-30B-A3B, a mixture of experts, in the 30 GB of RAM)
collect Minecraft questions and answers, check every one, and upload them to your Hugging Face dataset
**`<you>/Minecraftism-dataset`** - **private**: only you can see it (`PRIVATE = True`; a repo that already exists
as public is switched to private too).

**Kaggle setup:** Settings → *Accelerator*: **None** (free - no GPU quota used; `DEVICE = "cpu"` never touches a GPU)
· *Internet* **On** · Add-ons → Secrets: `HF_TOKEN` (a Hugging Face **write** token). Optional:
`discord_webhook_inform_me1` (status messages) and `ping_role` (the Discord role pinged on start, uploads, errors
and the end).

**Fast on a CPU - run 5 at once:** Kaggle runs up to **5 CPU sessions at the same time**. Leave `SESSIONS = 5`, set
`SESSION = 1` → *Save Version → Save & Run All*; change to `SESSION = 2` → *Save Version* again; ... up to 5. Each
saved version keeps its own settings, takes its own fifth of the topics and writes its own files (they never
overwrite each other). 5 sessions = 5x the rows in the same 12 hours. Run them again later and each goes on where it
stopped (`RESUME`).

**Instant rows:** session 1 first writes ~4,300 **exact** rows straight from the game's data (recipes, stack sizes,
durability, hardness, mining tools, food, enchantments, mobs, biomes) - no LLM, done in a second (`GAME_DATA`).

**How an answer gets in** (each step can reject it):
1. Pages only from sites whose licence allows commercial use - Minecraft Fandom wiki (CC BY-SA 3.0) and Wikipedia
   (CC BY-SA 4.0). minecraft.wiki is CC BY-NC-SA, so it is never read. Official Mojang pages only *confirm* answers.
2. The LLM writes question/answer pairs from the page text alone and copies the sentences that prove them.
3. Every quoted sentence must really be on the page, and every number in the answer must be in the quotes.
4. A second LLM pass sees only question, answer and quotes and must say SUPPORTED.
5. Crafting answers must match the game's own recipes (minecraft-data, MIT) - outdated recipes are caught.
6. Optional (`CORROBORATE = True`, off by default because it is the slowest part on a CPU): answers the game data
   did not confirm are searched again - an independent page that contradicts the answer rejects it, one that
   confirms it marks the row `trust: high` (otherwise `medium`).

**As many rows as possible:** after the ~1,700 game-data topics it reads every other article of the wiki
(`WIKI_PAGES`), each page whole (in parts of ~2,800 characters, 5 pairs per part), opened directly (no search).

Rejected pairs and the reason go to `meta/rejected-NN.jsonl` for review. The dataset uploads every 30 minutes, and
the next session goes on where this one stopped (`RESUME`). Pressing **Stop** keeps and uploads what was collected.
It uses the whole machine: llama-server works on 4 topics at once on all 4 CPU threads (it batches them), while web
search, downloads and page parsing run in separate worker processes. Before it starts, it measures the model with and
without n-gram speculation (the text stays exactly the same) and keeps the faster way. The log shows the rows/h and
the LLM's tokens/s. Load it with `load_dataset("<you>/Minecraftism-dataset")` (or `"high"` for the confirmed rows
only), or train with it: `train_minecraft_llm.py --minecraft-qa-dataset <you>/Minecraftism-dataset`.

Made with ❤️ by codero-sus · [the-codero.web.app](https://the-codero.web.app/) ·
[codero.is-a.dev](https://codero.is-a.dev) · part of [PyMC Bot](https://github.com/codero-sus/PyMC_Bot)
"""

SETTINGS = """# ============================== settings ==============================
MODEL = "auto"           # "auto" = "qwen3-30b-a3b": Qwen3-30B-A3B-Instruct-2507 (mixture of experts: 30B in ~23 GB
                         # RAM, only ~3B work per token - on a CPU as fast as a 4B, and far more careful)
                         # "qwen3.5-9b" / "qwen3.5-4b" / "qwen3-4b" / "qwen3-1.7b": smaller (on a CPU NOT much faster
                         # except 1.7B, which is the least careful)
DATASET_REPO = "Minecraftism-dataset"  # -> <your HF name>/Minecraftism-dataset (or "name/repo")
UPLOAD = True            # upload to Hugging Face (every UPLOAD_EVERY_MIN minutes, at the end and when stopped)
PRIVATE = True           # True = only you can see it (an existing public repo is switched to private);
                         # False = public - only once you have looked through it
RESUME = True            # go on from what earlier sessions uploaded (skips the topics already done)
KEEP = "medium"          # "medium" = every row that passed all checks (each has trust: high/medium);
                         # "high" = only rows confirmed by the game data or an independent page
CORROBORATE = False      # True = search each answer again and ask an independent page (catches outdated pages, makes
                         # more rows `high`) - but it is the slowest part on a CPU: about half the rows per hour
GAME_DATA = True         # session 1: ~4,300 exact rows from the game data first (no LLM, takes a second)
PARALLEL = 0             # topics worked on at once; 0 = auto: 4 on the CPU (1 = one at a time, slowest)
MIN_TOKENS_PER_S = 0     # 0 = quality first: always the biggest model that fits (the 30B on Kaggle). A number (e.g. 20)
                         # = speed first: measures each model here and keeps the biggest that writes that many tokens/s
PROCESSES = True         # web search, downloads and page parsing in PARALLEL worker processes (outside the main one)
SPECULATE = "auto"       # n-gram speculation (guesses the quotes it copies, checks many tokens at once):
                         # "auto" = measure with and without, keep the faster; "on" / "off"
DEVICE = "cpu"           # "cpu" = never uses a GPU (no GPU hours). "auto" / "gpu" = a GPU when attached (optional)
WIKI_PAGES = True        # after the ~1,700 game-data topics, EVERY other article of the Fandom wiki (thousands more)
PAGES_PER_TOPIC = 0      # wiki pages per topic (each read WHOLE, in parts of ~2,800 characters); 0 = auto: 1
PAIRS_PER_PAGE = 0       # question/answer pairs the LLM writes per part of a page (max 6); 0 = auto: 5
SESSIONS = 5             # Kaggle sessions you run AT THE SAME TIME (Kaggle allows 5 CPU ones); each its own share
SESSION = 1              # which one this is: 1 ... SESSIONS - Save Version with 1, then 2, ... (see the top)
MAX_TOPICS = 0           # 0 = as many as the time allows (84 mechanics, then mobs, items, blocks, enchantments...)
MAX_ROWS = 0             # 0 = no limit
MAX_HOURS = 11.75        # Kaggle stops a session after 12 h; the last upload happens before this
UPLOAD_EVERY_MIN = 30

HF_SECRET = "HF_TOKEN"                          # Kaggle secret with your Hugging Face write token
DISCORD = True                                  # status messages (only if the secret below is set)
DISCORD_SECRET = "discord_webhook_inform_me1"   # Kaggle secret with the webhook URL (never printed)
PING_ROLE_SECRET = "ping_role"                  # Kaggle secret with the role pinged on start, uploads, errors, end
"""

SETUP = """# ============================== setup ==============================
import os
import subprocess
import sys
import urllib.error
import urllib.request

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ddgs", "huggingface_hub>=0.30"], check=True)
WORK, TMP = "/kaggle/working", "/kaggle/tmp"  # the dataset in /kaggle/working; model, llama.cpp, page cache in /kaggle/tmp
for folder in (WORK, TMP):
    os.makedirs(folder, exist_ok=True)
try:
    gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
except Exception:
    gpus = []
ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
print(f"{os.cpu_count()} CPU threads, {ram:.0f} GB RAM" + (f" - {len(gpus)} GPU(s), llama.cpp will use them: {gpus[0]}"
      if gpus else " - no GPU: the CPU does the work (free; Accelerator GPU T4 x2 is ~10x faster)"))
for name, url in (("Hugging Face", "https://huggingface.co/api/whoami-v2"), ("GitHub", "https://api.github.com")):
    try:
        urllib.request.urlopen(url, timeout=20).close()
    except urllib.error.HTTPError:
        pass  # it answered: the internet works (401 without a token is fine)
    except Exception as exc:
        raise SystemExit(f"{name} is not reachable ({type(exc).__name__}) - turn Internet on in the notebook settings") from None
print("internet: OK")
"""

RUN = """# ============================== run ==============================
result = main({
    "model": MODEL, "dataset_repo": DATASET_REPO, "upload": UPLOAD, "private": PRIVATE, "resume": RESUME,
    "keep": KEEP, "corroborate": CORROBORATE, "game_data": GAME_DATA, "parallel": PARALLEL, "device": DEVICE,
    "min_tokens_per_s": MIN_TOKENS_PER_S, "speculate": SPECULATE,
    "processes": PROCESSES, "pages_per_topic": PAGES_PER_TOPIC, "pairs_per_page": PAIRS_PER_PAGE,
    "wiki_pages": WIKI_PAGES, "sessions": SESSIONS, "session": SESSION,
    "max_topics": MAX_TOPICS, "max_rows": MAX_ROWS, "max_hours": MAX_HOURS, "upload_every_min": UPLOAD_EVERY_MIN,
    "hf_secret": HF_SECRET, "discord": DISCORD, "discord_secret": DISCORD_SECRET,
    "ping_role_secret": PING_ROLE_SECRET, "work": WORK, "tmp": TMP,
})
print(f"\\n{'stopped early' if result.get('stopped') else 'finished'}: {result['rows']} rows ({result['high']} high trust, "
      f"{result['medium']} medium), {result['rejected']} rejected, {result['topics']} topics\\n-> {result['link']}")
for row in result["samples"]:
    print(f"\\n[{row['trust']}] Q: {row['question']}\\n         A: {row['answer']}")
"""


def cell(kind: str, text: str) -> dict:
    lines = text.splitlines(keepends=True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    out = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        out.update({"execution_count": None, "outputs": []})
    return out


def notebook() -> dict:
    module = "# ============================== the collector (pymc_bot/llm_webdata.py) ==============================\n"
    module += MODULE.read_text(encoding="utf-8")
    return {
        "cells": [cell("markdown", INTRO), cell("code", SETTINGS), cell("code", SETUP), cell("code", module),
                  cell("code", RUN)],
        "metadata": {
            "kaggle": {"accelerator": "none", "dataSources": [], "isGpuEnabled": False, "isInternetEnabled": True,
                       "language": "python", "sourceType": "notebook"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def render() -> str:
    return json.dumps(notebook(), indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        current = NOTEBOOK.read_text(encoding="utf-8") if NOTEBOOK.is_file() else ""
        print("up to date" if current == text else f"{NOTEBOOK.name} is out of date - run {Path(__file__).name}")
        return 0 if current == text else 1
    NOTEBOOK.write_text(text, encoding="utf-8")
    print(f"wrote {NOTEBOOK.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
