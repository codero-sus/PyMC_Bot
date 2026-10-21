#!/usr/bin/env python3
"""Fine-tune Minecraftism, the Minecraft LLM for PyMC Bot, on a Kaggle notebook (2x T4) - or any CUDA box.

This file is self-contained: ``python -m pymc_bot llm build`` copies it next to ``train.jsonl`` so
the Kaggle dataset you upload has everything. In the notebook (training/llm/*.ipynb):

    python train_minecraft_llm.py --preset minecraftism              # ~1-2 h on one T4
    python train_minecraft_llm.py --preset minecraftism-vl-thinking  # the vision + thinking edition
    python train_minecraft_llm.py --dry-run                          # check the data, no GPU needed

Minecraftism comes in four editions (``--list-presets``):

==============================  ========  ========  =====================================
preset                          vision    thinks    base model
==============================  ========  ========  =====================================
``minecraftism``                no        no        Qwen3-4B-Instruct-2507
``minecraftism-thinking``       no        yes       Qwen3-4B-Thinking-2507
``minecraftism-vl``             yes       no        Qwen3-VL-4B-Instruct
``minecraftism-vl-thinking``    yes       yes       Qwen3-VL-4B-Thinking
==============================  ========  ========  =====================================

Thinking editions learn each row's ``reasoning`` trace as ``<think>...</think>`` before the answer;
vision editions also learn from the screenshots in ``images/`` (recorded by the Fabric mod).

It trains a LoRA adapter with Unsloth (4-bit QLoRA, loss on the replies only), checks the result on
``eval.jsonl``, and writes a ready-to-run Ollama model to ``/kaggle/working/<preset>``: ``*.gguf``
(+ ``mmproj-*.gguf`` for vision) + ``Modelfile`` + ``pymc_model.json`` + a ``README.md`` model card.
With ``--hub-repo auto`` and the ``HF_TOKEN`` secret it is uploaded to Hugging Face as well.
Download the folder and run ``python -m pymc_bot llm import <folder>`` on your PC, or
``python -m pymc_bot llm import --hf <user>/<repo>``.

Only the standard library is imported at the top so presets and ``--dry-run`` work anywhere.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

COMPACT_TAG = "[PyMC Bot Minecraft brain]"

#: The size options. ``train_gb`` is the GPU memory training needs, ``run_gb`` what Ollama needs to
#: run the q4_k_m file on your PC. Kaggle's T4 has 16 GB and no bfloat16 (fp16 is used).
_MINECRAFTISM_WHY = {
    "minecraftism": "Recommended. The fast one: replies straight away with the action JSON. Trains in ~1-2 h on one "
                    "T4 and runs on most PCs (4 GB VRAM or CPU).",
    "minecraftism-thinking": "Thinks before it acts (a short <think> about health, mobs, chat and its goal), so it "
                             "makes fewer silly moves - but every turn takes a few seconds longer.",
    "minecraftism-vl": "Sees the game: learns from your recorded screenshots and gets the bot's rendered view while "
                       "playing (Agent > Vision). Needs screenshots in the data (/pymcbot screenshots on).",
    "minecraftism-vl-thinking": "Sees and thinks: the most capable edition and the slowest. Same data needs as VL.",
}
#: Qwen's sampling advice per mode (a little cooler than chat for reliable JSON).
RUN_THINKING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "num_ctx": 8192, "repeat_penalty": 1.0}
RUN_VISION = {"temperature": 0.6, "top_p": 0.8, "top_k": 20, "num_ctx": 8192, "repeat_penalty": 1.05}


def _edition(name: str, model: str, base: str, *, reasoning: bool, vision: bool, hub: str, file: str) -> dict[str, Any]:
    return {
        "model": model,
        "base_model": base,
        "chat_template": "qwen3-instruct" if not (reasoning or vision) else None,  # None = the model's own template
        "family": "chatml",
        "thinking": False,  # not a hybrid model: nothing to switch off
        "reasoning": reasoning,
        "vision": vision,
        "load_in_4bit": True,
        "batch": 1 if vision else 2,
        "accum": 8 if vision else 4,
        "train_gb": 9 if vision else 7,
        "run_gb": 4.5 if vision else 3.5,
        "gguf_gb": 3.3 if vision else 2.5,
        "speed": (2.2 if vision else 1.0) * (1.4 if reasoning else 1.0),
        "max_new_tokens": 1024 if reasoning else 256,
        "run_parameters": RUN_THINKING if reasoning else RUN_VISION if vision else None,
        "edition": name,
        "hub": hub,
        "file": file,
        "why": _MINECRAFTISM_WHY[name],
    }


PRESETS: dict[str, dict[str, Any]] = {
    "minecraftism": _edition("minecraftism", "unsloth/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B-Instruct-2507",
                             reasoning=False, vision=False, hub="Minecraftism-4B-GGUF", file="minecraftism-4b"),
    "minecraftism-thinking": _edition("minecraftism-thinking", "unsloth/Qwen3-4B-Thinking-2507", "Qwen/Qwen3-4B-Thinking-2507",
                                      reasoning=True, vision=False, hub="Minecraftism-4B-Thinking-GGUF",
                                      file="minecraftism-4b-thinking"),
    "minecraftism-vl": _edition("minecraftism-vl", "unsloth/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-4B-Instruct",
                                reasoning=False, vision=True, hub="Minecraftism-VL-4B-GGUF", file="minecraftism-vl-4b"),
    "minecraftism-vl-thinking": _edition("minecraftism-vl-thinking", "unsloth/Qwen3-VL-4B-Thinking", "Qwen/Qwen3-VL-4B-Thinking",
                                         reasoning=True, vision=True, hub="Minecraftism-VL-4B-Thinking-GGUF",
                                         file="minecraftism-vl-4b-thinking"),
    "qwen3-4b": {
        "model": "unsloth/Qwen3-4B-Instruct-2507",
        "chat_template": "qwen3-instruct",
        "family": "chatml",
        "thinking": False,
        "load_in_4bit": True,
        "batch": 2,
        "accum": 4,
        "train_gb": 7,
        "run_gb": 3.5,
        "gguf_gb": 2.5,
        "speed": 1.0,
        "why": "Same base as minecraftism, under the generic PyMC name (pymc-minecraft-qwen3-4b.gguf).",
    },
    "qwen3-8b": {
        "model": "unsloth/Qwen3-8B",
        "chat_template": "qwen3",
        "family": "chatml",
        "thinking": True,  # hybrid model: trained with thinking switched off
        "load_in_4bit": True,
        "batch": 2,
        "accum": 4,
        "train_gb": 11,
        "run_gb": 6.5,
        "gguf_gb": 5.0,
        "speed": 2.0,
        "why": "Smarter chat and better Minecraft answers. Still fits one T4 (about 2x slower to train). "
               "Needs ~6-8 GB VRAM to run smoothly.",
    },
    "qwen3-14b": {
        "model": "unsloth/Qwen3-14B",
        "chat_template": "qwen3",
        "family": "chatml",
        "thinking": True,
        "load_in_4bit": True,
        "batch": 1,
        "accum": 8,
        "train_gb": 15,
        "run_gb": 10,
        "gguf_gb": 9.0,
        "speed": 3.8,
        "why": "The biggest that trains on a 16 GB T4 (tight; use --split to spread it over both T4s). "
               "Best knowledge, but needs a 10-12 GB GPU to run and answers slower.",
    },
    "llama-3.2-3b": {
        "model": "unsloth/Llama-3.2-3B-Instruct",
        "chat_template": "llama-3.1",
        "family": "llama3",
        "thinking": False,
        "load_in_4bit": True,
        "batch": 2,
        "accum": 4,
        "train_gb": 6,
        "run_gb": 2.5,
        "gguf_gb": 2.0,
        "speed": 0.8,
        "why": "For weak PCs or many bots at once: small and fast, but makes more mistakes.",
    },
    "qwen3.5-4b": {
        "model": "unsloth/Qwen3.5-4B",
        "chat_template": None,  # use the model's own template
        "family": "chatml",
        "thinking": False,
        "load_in_4bit": False,  # Unsloth advises 16-bit LoRA for Qwen3.5
        "batch": 1,
        "accum": 8,
        "train_gb": 10,
        "run_gb": 4,
        "gguf_gb": 3.0,
        "speed": 4.0,
        "why": "Newest small Qwen. Experimental here: a T4 has no bfloat16 and Qwen3.5 overflows in fp16, so "
               "Unsloth trains it in float32 on Kaggle - several times slower.",
    },
}
DEFAULT_PRESET = "minecraftism"
#: The four Minecraftism editions, in the order the notebook trains them.
EDITIONS = ("minecraftism", "minecraftism-thinking", "minecraftism-vl", "minecraftism-vl-thinking")
MODEL_NAME = "Minecraftism"
AUTHOR = "codero-sus"
AUTHOR_URL = "https://github.com/codero-sus"
WEBSITE = "https://the-codero.web.app/"
PROJECT_URL = "https://github.com/codero-sus/PyMC_Bot"
#: How each family marks a user turn / an assistant reply (loss is computed on replies only).
TURN_MARKERS = {
    "chatml": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "llama3": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
}
STOP_TOKENS = {"chatml": ["<|im_end|>", "<|im_start|>"], "llama3": ["<|eot_id|>", "<|start_header_id|>"]}
#: Sampling for the finished model (Qwen's advice for non-thinking mode, a bit cooler for JSON).
RUN_PARAMETERS = {"temperature": 0.6, "top_p": 0.8, "top_k": 20, "num_ctx": 4096, "repeat_penalty": 1.05}
DATA_GLOBS = ("train.jsonl", "llm_data/train.jsonl", "/kaggle/input/**/train.jsonl")
#: Longest side (pixels) pictures are given to the vision model at (training and eval).
IMAGE_SIDE = 512


# --------------------------------------------------------------------------- data
def find_data(path: str | None) -> Path:
    if path:
        candidate = Path(path)
        if candidate.is_dir():
            candidate = candidate / "train.jsonl"
        if not candidate.is_file():
            raise SystemExit(f"no training data at {candidate}")
        return candidate
    for pattern in DATA_GLOBS:
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return Path(hits[0])
    raise SystemExit("train.jsonl not found - add your dataset to the notebook (Add Input) or pass --data")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_rows(rows: list[dict[str, Any]]) -> list[str]:
    problems = []
    for number, row in enumerate(rows, 1):
        messages = row.get("messages") or []
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            problems.append(f"row {number}: needs messages ending with an assistant reply")
        elif any(not str(m.get("content") or "").strip() for m in messages):
            problems.append(f"row {number}: empty message")
    return problems


def strip_thinking(text: str) -> str:
    """The answer part of a reply: ``<think>...</think>`` removed, also when the opening tag is missing
    (Qwen3 *Thinking* models start inside the think block, so only ``</think>`` shows up)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    if re.search(r"</think>", text, flags=re.I):
        text = re.split(r"</think>", text, flags=re.I)[-1]
    return text.strip()


def think_text(reasoning: str, answer: str) -> str:
    """What a Thinking edition is trained to write: the trace, then the answer."""
    reasoning = " ".join(str(reasoning or "").split()) or "Simple one, answer directly."
    return f"<think>\n{reasoning}\n</think>\n\n{answer}"


def prepare_messages(row: dict[str, Any], preset: dict[str, Any]) -> list[dict[str, str]]:
    """The chat a row is trained on for this edition (thinking editions get the trace)."""
    messages = [dict(m) for m in row["messages"]]
    if preset.get("reasoning"):
        messages[-1]["content"] = think_text(row.get("reasoning", ""), messages[-1]["content"])
    return messages


def select_rows(rows: list[dict[str, Any]], preset: dict[str, Any]) -> list[dict[str, Any]]:
    """Text editions skip picture-only rows ("what do you see?"); image turns keep their text."""
    if preset.get("vision"):
        return list(rows)
    return [row for row in rows if row.get("kind") != "vision"]


def vision_messages(row: dict[str, Any], preset: dict[str, Any], load_image: Any) -> list[dict[str, Any]]:
    """A row for a vision model: the pictures go in front of the last user message."""
    messages = prepare_messages(row, preset)
    images = [load_image(name) for name in row.get("images") or []]
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    out: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if index == last_user and images:
            parts = [{"type": "image", "image": image} for image in images]
            parts.append({"type": "text", "text": message["content"]})
            out.append({"role": message["role"], "content": parts})
        else:
            out.append({"role": message["role"], "content": [{"type": "text", "text": message["content"]}]})
    return out


def first_json(text: str) -> dict[str, Any] | None:
    """The first JSON object in a reply (``<think>`` blocks and code fences are ignored)."""
    text = strip_thinking(text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def score(row: dict[str, Any], reply: str) -> dict[str, Any]:
    """One eval row: valid JSON (turns), the expected action, the expected keywords in the answer."""
    result: dict[str, Any] = {"kind": row.get("kind", "qa"), "source": row.get("source", "?")}
    check = row.get("check") or {}
    text = reply or ""
    if result["kind"] == "turn":
        payload = first_json(text)
        result["json_ok"] = payload is not None and isinstance(payload.get("action"), str)
        if "action" in check:
            result["action_ok"] = bool(result["json_ok"] and str(payload.get("action", "")).lower() == check["action"])
        if payload is not None:
            text = str(payload.get("message") or "")
    if check.get("contains"):
        low = text.lower()
        result["contains_ok"] = all(str(word).lower() in low for word in check["contains"])
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(key: str) -> float | None:
        values = [r[key] for r in results if key in r]
        return round(100 * sum(values) / len(values), 1) if values else None

    return {"rows": len(results), "json_valid_pct": rate("json_ok"), "action_match_pct": rate("action_ok"),
            "answer_keywords_pct": rate("contains_ok")}


def estimate(rows: list[dict[str, Any]], preset: dict[str, Any], epochs: float, max_steps: int) -> dict[str, Any]:
    chars = sum(len(m.get("content", "")) for row in rows for m in row["messages"])
    if preset.get("reasoning"):
        chars += sum(len(row.get("reasoning") or "") for row in rows)
    tokens = chars / 3.6
    if preset.get("vision"):
        tokens += 150 * sum(len(row.get("images") or []) for row in rows)  # ~150 tokens per 512 px picture
    per_step = preset["batch"] * preset["accum"]
    steps = max_steps if max_steps > 0 else int(len(rows) * epochs / per_step) + 1
    # ~1.6k tokens/s for a 4B QLoRA on one T4, scaled by the preset's relative cost
    hours = (tokens / max(1, len(rows))) * per_step * steps / 1600 * preset["speed"] / 3600
    return {"rows": len(rows), "tokens": int(tokens), "avg_tokens": int(tokens / max(1, len(rows))), "steps": steps,
            "est_hours_t4": round(hours, 1)}


# --------------------------------------------------------------------------- Ollama files
def modelfile_text(gguf_name: str, family: str, template: str | None = None, mmproj: str | None = None,
                   parameters: dict[str, Any] | None = None) -> str:
    """An Ollama Modelfile. Vision models get a second ``FROM`` line with the projector (``mmproj``)."""
    lines = [f"FROM ./{gguf_name}"]
    if mmproj:
        lines.append(f"FROM ./{mmproj}")
    if template:
        lines.append(f'TEMPLATE """{template}"""')
    for key, value in (parameters or RUN_PARAMETERS).items():
        lines.append(f"PARAMETER {key} {value}")
    for stop in STOP_TOKENS.get(family, []):
        lines.append(f'PARAMETER stop "{stop}"')
    return "\n".join(lines) + "\n"


def template_from_modelfile(text: str) -> str | None:
    match = re.search(r'TEMPLATE\s+"""(.*?)"""', text, flags=re.S)
    return match.group(1) if match else None


# --------------------------------------------------------------------------- training
def render(tokenizer: Any, messages: list[dict[str, str]], preset: dict[str, Any], generation: bool = False) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": generation}
    if preset["thinking"]:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def output_name(preset_name: str, preset: dict[str, Any], quant: str) -> str:
    stem = preset.get("file") or f"pymc-minecraft-{preset_name}"
    return f"{stem}.{quant.upper()}.gguf"


def display_name(preset_name: str, preset: dict[str, Any]) -> str:
    if not preset.get("edition"):
        return f"PyMC Minecraft ({preset_name})"
    return MODEL_NAME + (" VL" if preset.get("vision") else "") + " 4B" + (" Thinking" if preset.get("reasoning") else "")


def model_card(preset_name: str, preset: dict[str, Any], report: dict[str, Any], repo: str) -> str:
    """README.md for the Hugging Face repo (also written next to the GGUF)."""
    name = display_name(preset_name, preset)
    vision, reasoning = bool(preset.get("vision")), bool(preset.get("reasoning"))
    base = preset.get("base_model") or preset["model"]
    quant = str(report.get("quant") or "Q4_K_M").upper()
    tags = ["minecraft", "gguf", "pymc-bot", "minecraftism", "qwen3", "unsloth", "game-agent"]
    tags += ["vision", "image-text-to-text"] if vision else []
    tags += ["reasoning", "thinking"] if reasoning else []
    files = [f"- `{report.get('gguf', output_name(preset_name, preset, quant))}` - the model ({quant})"]
    if report.get("mmproj"):
        files.append(f"- `{report['mmproj']}` - the vision projector (needed for pictures)")
    files += ["- `Modelfile` - for `ollama create`", "- `pymc_model.json` - training report PyMC Bot reads on import"]
    evaluation = report.get("eval") or {}
    eval_lines = ""
    if evaluation:
        def pct(key: str) -> str:
            value = evaluation.get(key)
            return "-" if value is None else f"{value}%"

        eval_lines = ("\n## Evaluation (held-out rows)\n\n| valid JSON | right action | right answers | rows |\n|---|---|---|---|\n"
                      f"| {pct('json_valid_pct')} | {pct('action_match_pct')} | {pct('answer_keywords_pct')} | "
                      f"{evaluation.get('rows', '-')} |\n")
    vision_text = ("\nIt also **sees**: it was trained on first-person screenshots recorded with the PyMC Bot Fabric mod, "
                   "paired with what the player did next and grounded descriptions of each picture. In PyMC Bot turn on "
                   "*let the model see (vision)* in the panel's AI settings and the bot sends its rendered view every few turns.\n") if vision else ""
    think_text_ = ("\nIt **thinks first**: before every answer it writes a short `<think>` about its health, the mobs "
                   "around it (with facts from the mob database), chat and its goal, then the JSON action. Smarter "
                   "choices, slower turns.\n") if reasoning else ""
    llama = (f"llama-server -m {report.get('gguf', 'model.gguf')} --mmproj {report.get('mmproj', 'mmproj.gguf')}"
             if vision else f"llama-server -m {report.get('gguf', 'model.gguf')}")
    return f"""---
license: other
license_name: polyform-noncommercial-1.0.0
license_link: https://polyformproject.org/licenses/noncommercial/1.0.0/
base_model: {base}
tags:
{chr(10).join('- ' + t for t in tags)}
language:
- en
pipeline_tag: {'image-text-to-text' if vision else 'text-generation'}
library_name: gguf
---

# {name}

**Minecraftism** is a small model that knows Minecraft and plays it: the brain of
[PyMC Bot]({PROJECT_URL}), a Minecraft player bot. Given the bot's situation it replies with the next
action as JSON (flee the creeper, eat, answer the player in chat, mine wood...), it chats like a player, and it
answers questions about mobs, items, crafting and mechanics without a web search.
{think_text_}{vision_text}
| | |
|---|---|
| Edition | {name} (`{preset_name}`) |
| Base model | [{base}](https://huggingface.co/{base}) (Apache-2.0) |
| Fine-tune | LoRA r={report.get('rank', 32)} with Unsloth on Kaggle (T4), {report.get('train_rows', '?')} rows, loss {report.get('loss', '?')} |
| Data | PyMC Bot rules + mob database, Minecraft Wiki text, recorded play{' and screenshots' if vision else ''} |
| Size | ~{preset.get('gguf_gb')} GB file, ~{preset.get('run_gb')} GB to run |
{eval_lines}
## Files

{chr(10).join(files)}

## Use it

With PyMC Bot (switches the bot to this model with the short Minecraft prompt):

```bash
python -m pymc_bot llm import --hf {repo}
```

With Ollama:

```bash
ollama run hf.co/{repo}:{quant}
```

With llama.cpp:

```bash
{llama}
```

## License and credits

- Made with ❤️ by [codero-sus]({AUTHOR_URL}) - {WEBSITE}
- **Personal, non-commercial use only.** The fine-tune is released under PolyForm Noncommercial 1.0.0, and the
  training data includes Minecraft Wiki text (CC BY-NC-SA 3.0, https://minecraft.wiki). The base model's own
  license (Apache-2.0) still applies to the base weights.
- Not an official Minecraft product. Not approved by or associated with Mojang or Microsoft.
"""


def resolve_repo(repo: str, preset: dict[str, Any], whoami: Any = None) -> str:
    """``auto`` -> ``<your HF name>/<edition repo>``; ``name`` -> ``<your HF name>/name``; ``user/name`` as is."""
    if "/" in repo:
        return repo
    if whoami is None:
        raise SystemExit("--hub-repo needs your Hugging Face name (user/repo) or a token to look it up")
    user = whoami().get("name")
    name = (preset.get("hub") or "pymc-minecraft-GGUF") if repo == "auto" else repo
    return f"{user}/{name}"


def ollama_params(preset: dict[str, Any]) -> dict[str, Any]:
    """The ``params`` file Ollama reads from a Hugging Face repo (``ollama pull hf.co/...``)."""
    params: dict[str, Any] = dict(preset.get("run_parameters") or RUN_PARAMETERS)
    params["stop"] = list(STOP_TOKENS.get(preset.get("family", "chatml"), []))
    return params


def write_ollama_params(out: Path, preset: dict[str, Any]) -> Path:
    target = Path(out) / "params"
    target.write_text(json.dumps(ollama_params(preset), indent=2) + "\n", encoding="utf-8")
    return target


def publish(out: Path, repo: str, preset_name: str, token: str, private: bool = False, include_lora: bool = False) -> str:
    """Upload the finished folder (GGUF, mmproj, Modelfile, model card, report) to Hugging Face."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo = resolve_repo(repo, PRESETS[preset_name], api.whoami)
    write_ollama_params(out, PRESETS[preset_name])
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    ignore = ["checkpoints/*", "*.lock"] + ([] if include_lora else ["lora/*"])
    api.upload_folder(repo_id=repo, folder_path=str(out), repo_type="model", ignore_patterns=ignore,
                      commit_message=f"{display_name(preset_name, PRESETS[preset_name])} from PyMC Bot")
    return repo


def load_image_from(base: Path, side: int = IMAGE_SIDE) -> Any:
    """A loader for the dataset's pictures (PIL, RGB, longest side ``side``)."""
    from PIL import Image

    cache: dict[str, Any] = {}

    def load(name: str) -> Any:
        if name not in cache:
            with Image.open(base / name) as image:
                picture = image.convert("RGB")
            picture.thumbnail((side, side))
            cache[name] = picture
        return cache[name]

    return load


def train(args: argparse.Namespace, preset: dict[str, Any], rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> int:
    if not args.split:
        # one T4 is plenty up to 14B and avoids DataParallel, which Unsloth does not use
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if preset.get("vision"):
        return train_vision(args, preset, rows, eval_rows)
    from unsloth import FastLanguageModel  # noqa: I001  (unsloth must be imported before transformers)
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model_name = args.model or preset["model"]
    print(f"loading {model_name} ({'4-bit' if preset['load_in_4bit'] else '16-bit'}) on {torch.cuda.device_count()} GPU(s)")
    load: dict[str, Any] = {"model_name": model_name, "max_seq_length": args.max_seq, "load_in_4bit": preset["load_in_4bit"]}
    if args.split:
        load["device_map"] = "unsloth_balanced"
    model, tokenizer = FastLanguageModel.from_pretrained(**load)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=args.seed,
    )
    if preset["chat_template"]:
        tokenizer = get_chat_template(tokenizer, chat_template=preset["chat_template"])

    texts = [render(tokenizer, prepare_messages(row, preset), preset) for row in rows]
    dataset = Dataset.from_dict({"text": texts})
    config = SFTConfig(
        dataset_text_field="text", max_length=args.max_seq, per_device_train_batch_size=preset["batch"],
        gradient_accumulation_steps=preset["accum"], warmup_steps=10, num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1, learning_rate=args.lr, logging_steps=10,
        optim="adamw_8bit", weight_decay=0.001, lr_scheduler_type="linear", seed=args.seed, output_dir=str(out / "checkpoints"),
        save_strategy="no", report_to="none", dataset_num_proc=2,
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=config)
    user, assistant = TURN_MARKERS[preset["family"]]
    trainer = train_on_responses_only(trainer, instruction_part=user, response_part=assistant)
    stats = trainer.train()
    print(f"trained in {(time.time() - started) / 60:.0f} min, loss {stats.training_loss:.3f}")

    def generate(row: dict[str, Any]) -> str:
        prompt = render(tokenizer, row["messages"][:-1], preset, generation=True)
        inputs = tokenizer(text=prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        output = model.generate(**inputs, max_new_tokens=preset.get("max_new_tokens", 256), temperature=0.3, top_p=0.8,
                                top_k=20, do_sample=True)
        return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return finish(args, preset, model, tokenizer, rows, eval_rows, stats.training_loss, started, generate,
                  lambda: FastLanguageModel.for_inference(model))


def train_vision(args: argparse.Namespace, preset: dict[str, Any], rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> int:
    """The VL editions: Qwen3-VL with FastVisionModel, pictures + text rows in one run."""
    from unsloth import FastVisionModel  # noqa: I001
    from unsloth.trainer import UnslothVisionDataCollator
    import torch
    from trl import SFTConfig, SFTTrainer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model_name = args.model or preset["model"]
    data_dir = Path(args.data_dir)
    load_image = load_image_from(data_dir, args.image_side)
    pictures = sum(len(row.get("images") or []) for row in rows)
    print(f"loading {model_name} (vision, 4-bit) on {torch.cuda.device_count()} GPU(s); {pictures} pictures in {len(rows)} rows")
    if not pictures:
        print("no pictures in the data - it still trains (text only), but record with /pymcbot screenshots on and "
              "build with --screenshots so the vision part learns Minecraft too")
    model, processor = FastVisionModel.from_pretrained(model_name, load_in_4bit=preset["load_in_4bit"],
                                                       use_gradient_checkpointing="unsloth")
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=bool(args.vision_layers), finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True, r=args.rank, lora_alpha=args.rank,
        lora_dropout=0, bias="none", random_state=args.seed,
    )
    dataset = [{"messages": vision_messages(row, preset, load_image)} for row in rows]
    FastVisionModel.for_training(model)
    user, assistant = TURN_MARKERS[preset["family"]]
    try:  # recent unsloth-zoo: response-only loss inside the collator (also for text-only rows)
        collator = UnslothVisionDataCollator(model, processor, train_on_responses_only=True,
                                             instruction_part=user, response_part=assistant)
    except TypeError:
        collator = UnslothVisionDataCollator(model, processor)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    config = SFTConfig(
        per_device_train_batch_size=preset["batch"], gradient_accumulation_steps=preset["accum"], warmup_steps=10,
        num_train_epochs=args.epochs, max_steps=args.max_steps if args.max_steps > 0 else -1, learning_rate=args.lr,
        logging_steps=10, optim="adamw_8bit", weight_decay=0.001, lr_scheduler_type="linear", seed=args.seed,
        output_dir=str(out / "checkpoints"), save_strategy="no", report_to="none", fp16=not bf16, bf16=bf16,
        remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_seq,
    )
    trainer = SFTTrainer(model=model, tokenizer=processor, data_collator=collator, train_dataset=dataset, args=config)
    stats = trainer.train()
    print(f"trained in {(time.time() - started) / 60:.0f} min, loss {stats.training_loss:.3f}")

    def generate(row: dict[str, Any]) -> str:
        messages = vision_messages({**row, "reasoning": ""}, {**preset, "reasoning": False}, load_image)[:-1]
        images = [part["image"] for m in messages for part in m["content"] if part.get("type") == "image"]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=images or None, return_tensors="pt", add_special_tokens=False).to(model.device)
        output = model.generate(**inputs, max_new_tokens=preset.get("max_new_tokens", 256), temperature=0.3, top_p=0.8,
                                top_k=20, do_sample=True)
        return processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return finish(args, preset, model, processor, rows, eval_rows, stats.training_loss, started, generate,
                  lambda: FastVisionModel.for_inference(model))


def finish(args: argparse.Namespace, preset: dict[str, Any], model: Any, tokenizer: Any, rows: list[dict[str, Any]],
           eval_rows: list[dict[str, Any]], loss: float, started: float, generate: Any, for_inference: Any) -> int:
    """Shared by both trainers: save the adapter, evaluate, export GGUF, write the card, upload."""
    out = Path(args.out)
    model_name = args.model or preset["model"]
    adapter = out / "lora"
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))

    report: dict[str, Any] = {"preset": args.preset, "model_name": display_name(args.preset, preset), "base_model": model_name,
                              "edition": preset.get("edition"), "vision": bool(preset.get("vision")),
                              "reasoning": bool(preset.get("reasoning")), "train_rows": len(rows), "rank": args.rank,
                              "images": sum(len(row.get("images") or []) for row in rows), "loss": round(loss, 4)}
    if eval_rows and args.eval_limit:
        for_inference()
        results = []
        for row in eval_rows[: args.eval_limit]:
            results.append(score(row, generate(row)))
        report["eval"] = summarize(results)
        print("eval:", json.dumps(report["eval"]))

    if args.gguf != "none":
        print(f"exporting GGUF ({args.gguf}) - this takes 10-20 minutes")
        model.save_pretrained_gguf(str(out / "gguf"), tokenizer, quantization_method=args.gguf)
        found = sorted(set(glob.glob(str(out / "gguf*" / "*.gguf")) + glob.glob(str(out / "*.gguf")) + glob.glob("*.gguf")))
        projectors = [f for f in found if "mmproj" in Path(f).name.lower()]
        found = [f for f in found if "mmproj" not in Path(f).name.lower()]
        if not found:
            print("GGUF export produced no file - the LoRA adapter is in", adapter)
            return 1
        gguf = out / output_name(args.preset, preset, args.gguf)
        shutil.move(found[0], gguf)
        mmproj = None
        if preset.get("vision"):
            if projectors:
                mmproj = out / f"mmproj-{gguf.name.split('.')[0]}.F16.gguf"
                shutil.move(projectors[0], mmproj)
            else:
                print("warning: no mmproj (vision projector) was exported - the model will only read text")
        unsloth_modelfile = next(iter(glob.glob(str(out / "gguf*" / "Modelfile"))), None)
        template = template_from_modelfile(Path(unsloth_modelfile).read_text()) if unsloth_modelfile else None
        (out / "Modelfile").write_text(modelfile_text(gguf.name, preset["family"], template, mmproj.name if mmproj else None,
                                                      preset.get("run_parameters")), encoding="utf-8")
        report["gguf"] = gguf.name
        report["quant"] = args.gguf.upper()
        report["gguf_gb"] = round(gguf.stat().st_size / 1e9, 2)
        if mmproj:
            report["mmproj"] = mmproj.name
        shutil.rmtree(out / "gguf", ignore_errors=True)
    shutil.rmtree(out / "checkpoints", ignore_errors=True)
    report.update({"family": preset["family"], "compact_tag": COMPACT_TAG, "compact_prompt": True,
                   "run_parameters": preset.get("run_parameters") or RUN_PARAMETERS,
                   "minutes": round((time.time() - started) / 60, 1), "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    repo = args.hub_repo if args.hub_repo and "/" in args.hub_repo else f"<your-hf-name>/{preset.get('hub') or 'pymc-minecraft-GGUF'}"
    (out / "pymc_model.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(model_card(args.preset, preset, report, repo), encoding="utf-8")
    if args.hub_repo:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("--hub-repo needs HF_TOKEN (Kaggle: Add-ons > Secrets > HF_TOKEN, a 'write' token)")
        else:
            try:
                from huggingface_hub import HfApi

                full = resolve_repo(args.hub_repo, preset, HfApi(token=token).whoami)
                report["hub_repo"] = full
                (out / "pymc_model.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                (out / "README.md").write_text(model_card(args.preset, preset, report, full), encoding="utf-8")
                publish(out, full, args.preset, token, private=args.hub_private, include_lora=args.hub_lora)
                print(f"uploaded -> https://huggingface.co/{full}")
            except Exception as exc:  # never lose a finished model to an upload problem
                print(f"Hugging Face upload failed ({type(exc).__name__}: {exc}) - the files are still in {out}")
    print(f"done -> {out}  (download it, then: python -m pymc_bot llm import <folder>)")
    return 0


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune the PyMC Bot Minecraft LLM (Unsloth, Kaggle T4).")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--model", help="override the base model of the preset (any Unsloth/HF model id)")
    parser.add_argument("--data", help="train.jsonl or its folder (default: search /kaggle/input)")
    parser.add_argument("--eval-data", help="eval.jsonl (default: next to train.jsonl)")
    parser.add_argument("--out", default=None, help="output folder (default: /kaggle/working/<preset> on Kaggle, ./<preset> elsewhere)")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = full epochs)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--max-seq", type=int, default=3072, help="longest example in tokens")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval-limit", type=int, default=150, help="eval rows to try after training (0 = skip)")
    parser.add_argument("--gguf", default="q4_k_m", help="GGUF quantisation (q4_k_m, q5_k_m, q8_0) or 'none'")
    parser.add_argument("--split", action="store_true", help="spread the model over both GPUs (only needed for 14B)")
    parser.add_argument("--hub-repo", help="upload to Hugging Face (needs HF_TOKEN): 'auto' = <you>/<edition repo>, "
                                           "a repo name, or user/repo")
    parser.add_argument("--hub-private", action="store_true", help="create the Hugging Face repo as private")
    parser.add_argument("--hub-lora", action="store_true", help="also upload the LoRA adapter folder")
    parser.add_argument("--image-side", type=int, default=IMAGE_SIDE, help="vision editions: longest side of a picture")
    parser.add_argument("--vision-layers", type=int, default=0, choices=[0, 1],
                        help="vision editions: 1 = also fine-tune the image encoder (slower, more memory)")
    parser.add_argument("--dry-run", action="store_true", help="check the data and show the plan without a GPU")
    parser.add_argument("--list-presets", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_presets:
        print(f"{MODEL_NAME} editions:")
        for name, preset in PRESETS.items():
            if name == EDITIONS[0] or (name not in EDITIONS and list(PRESETS).index(name) == len(EDITIONS)):
                print("" if name == EDITIONS[0] else "\nother sizes (generic PyMC Minecraft model):")
            mark = " (default)" if name == DEFAULT_PRESET else ""
            what = (f" [{'vision' if preset.get('vision') else 'text'}, {'thinks' if preset.get('reasoning') else 'fast'}]"
                    if preset.get("edition") else "")
            print(f"{name}{mark}{what}: {preset['model']} - train ~{preset['train_gb']} GB, run ~{preset['run_gb']} GB, "
                  f"file ~{preset['gguf_gb']} GB\n    {preset['why']}")
        return 0
    preset = PRESETS[args.preset]
    if not args.out:
        args.out = str(Path("/kaggle/working") / args.preset) if Path("/kaggle/working").is_dir() else args.preset
    data = find_data(args.data)
    args.data_dir = str(data.parent)
    rows = select_rows(read_jsonl(data), preset)
    eval_path = Path(args.eval_data) if args.eval_data else data.with_name("eval.jsonl")
    eval_rows = select_rows(read_jsonl(eval_path), preset) if eval_path.is_file() else []
    missing = [name for row in rows if preset.get("vision") for name in row.get("images") or []
               if not (data.parent / name).is_file()]
    if missing:
        print(f"{len(missing)} picture(s) listed in the data are missing next to train.jsonl (first: {missing[0]}) - "
              "upload the images/ folder with the dataset")
        return 2
    if preset.get("vision") and not any(row.get("images") for row in rows):
        print(f"{args.preset} is a vision edition but the data has no pictures. Record playtime with the mod's "
              "screenshots on (/pymcbot screenshots on), then: python -m pymc_bot llm build --playtime <folder> "
              "--screenshots - or train a text edition (minecraftism, minecraftism-thinking).")
        return 2
    problems = check_rows(rows) + check_rows(eval_rows)
    if problems:
        print("\n".join(problems[:20]))
        return 2
    plan = estimate(rows, preset, args.epochs, args.max_steps)
    sources: dict[str, int] = {}
    for row in rows:
        sources[row.get("source", "?")] = sources.get(row.get("source", "?"), 0) + 1
    print(f"data: {data} ({len(rows)} train, {len(eval_rows)} eval) sources={sources}")
    print(f"preset: {args.preset} ({display_name(args.preset, preset)}) -> {args.model or preset['model']}; "
          f"plan: {json.dumps(plan)}")
    if preset.get("reasoning"):
        with_trace = sum(1 for row in rows if row.get("reasoning"))
        print(f"thinking edition: {with_trace}/{len(rows)} rows have a reasoning trace")
    if args.dry_run:
        sample = prepare_messages(rows[0], preset)[-1]["content"]
        print("dry run: data is fine. Sample reply:", sample[:400])
        return 0
    return train(args, preset, rows, eval_rows)


if __name__ == "__main__":
    sys.exit(main())
