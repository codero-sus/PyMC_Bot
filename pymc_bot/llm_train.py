#!/usr/bin/env python3
"""Fine-tune Minecraftism, the Minecraft LLM for PyMC Bot, on a Kaggle notebook (2x T4) - or any CUDA box.

This file is self-contained: ``python -m pymc_bot llm build`` copies it next to ``train.jsonl`` so
the Kaggle dataset you upload has everything. In the notebook (pynb/PyMC_Minecraft_LLM_Kaggle.ipynb):

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
(+ ``mmproj-*.gguf`` for vision) + ``Modelfile`` + ``pymc_model.json`` + a ``README.md`` model card, and the
**safetensors edition** in ``safetensors/`` (the LoRA merged into the base, 16-bit, with its own model card;
``--safetensors none`` skips it). With ``--hub-repo auto`` and the ``HF_TOKEN`` secret both are uploaded to Hugging
Face: the GGUF to ``<you>/Minecraftism-4B-GGUF``, the safetensors to ``<you>/Minecraftism-4B``.
Download the folder and run ``python -m pymc_bot llm import <folder>`` on your PC, or
``python -m pymc_bot llm import --hf <user>/<repo>``.

Only the standard library is imported at the top so presets and ``--dry-run`` work anywhere.
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import re
import shutil
import sys
import time
import zlib
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
#: Abliterated (refusal direction removed) copies of the four bases by huihui-ai, Apache-2.0 like the
#: originals. ``--uncensored`` trains on these and drops refusals from the extra chat data.
UNCENSORED_MODELS = {
    "minecraftism": "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated",
    "minecraftism-thinking": "huihui-ai/Huihui-Qwen3-4B-Thinking-2507-abliterated",
    "minecraftism-vl": "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated",
    "minecraftism-vl-thinking": "huihui-ai/Huihui-Qwen3-VL-4B-Thinking-abliterated",
}
for _name, _model in UNCENSORED_MODELS.items():
    PRESETS[_name]["uncensored_model"] = _model
DEFAULT_PRESET = "minecraftism"
#: The four Minecraftism editions, in the order the notebook trains them.
EDITIONS = ("minecraftism", "minecraftism-thinking", "minecraftism-vl", "minecraftism-vl-thinking")
MODEL_NAME = "Minecraftism"
AUTHOR = "codero-sus"
AUTHOR_URL = "https://github.com/codero-sus"
WEBSITE = "https://the-codero.web.app/"
WEBSITES = ("https://codero.is-a.dev", WEBSITE)
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


def think_text(reasoning: str, answer: str, question: str = "") -> str:
    """What a Thinking edition is trained to write: the trace, then the answer.

    A row without a trace gets a plan for *its* question (``answer_plan``), never one fixed sentence: a
    single "simple one, answer directly" on hundreds of rows taught the model to skip thinking on
    code, maths and mechanics questions too."""
    reasoning = str(reasoning or "").strip()
    if "\n" not in reasoning:
        reasoning = " ".join(reasoning.split())
    reasoning = reasoning or answer_plan(question, answer)
    return f"<think>\n{reasoning}\n</think>\n\n{answer}"


def prepare_messages(row: dict[str, Any], preset: dict[str, Any]) -> list[dict[str, str]]:
    """The chat a row is trained on for this edition (thinking editions get the trace)."""
    messages = [dict(m) for m in row["messages"]]
    if preset.get("reasoning") and row.get("kind") != "tools":  # tool rows carry a trace in every assistant turn
        question = next((str(m.get("content") or "") for m in reversed(messages[:-1]) if m.get("role") == "user"), "")
        messages[-1]["content"] = think_text(row.get("reasoning", ""), messages[-1]["content"], question)
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
                   parameters: dict[str, Any] | None = None, system: str | None = None) -> str:
    """An Ollama Modelfile. Vision models get a second ``FROM`` line with the projector (``mmproj``)."""
    lines = [f"FROM ./{gguf_name}"]
    if mmproj:
        lines.append(f"FROM ./{mmproj}")
    if template:
        lines.append(f'TEMPLATE """{template}"""')
    if system:
        lines.append('SYSTEM """' + system.replace('"""', "''") + '"""')
    for key, value in (parameters or RUN_PARAMETERS).items():
        lines.append(f"PARAMETER {key} {value}")
    for stop in STOP_TOKENS.get(family, []):
        lines.append(f'PARAMETER stop "{stop}"')
    return "\n".join(lines) + "\n"


def template_from_modelfile(text: str) -> str | None:
    match = re.search(r'TEMPLATE\s+"""(.*?)"""', text, flags=re.S)
    return match.group(1) if match else None


# --------------------------------------------------------------------------- persona, everyday chat, uncensored
#: The default system prompt. Minecraftism is a general assistant that happens to know Minecraft inside
#: out; PyMC Bot sends its own short prompt for game turns, so this one is about everyday use.
MINECRAFTISM_SYSTEM = """You are Minecraftism, an AI assistant made by codero-sus.

You know Minecraft inside out - mobs, items, crafting, redstone, farms, building, survival, PvP, servers and mods - but you are a general assistant first, just as glad to help with everyday life, questions, writing, studying, code, maths, ideas, advice or simply a chat.

How you work:
- Be genuinely useful. Answer what was asked with the details that matter, then stop. No filler.
- Be honest. If you don't know or aren't sure, say so. Never invent facts, numbers, versions or sources.
- Treat the user as a capable adult. Don't lecture, moralise or add warnings and disclaimers nobody asked for, and don't refuse something just because it is edgy, dark, mature or controversial.
- Have a personality: friendly, witty when it fits, with your own opinions when asked.
- Match the user: short and casual for chat, clear and structured for real tasks. Reply in the user's language.
- Use Markdown only when it helps (steps, lists, code).

If a message describes a game situation and asks for a specific format (for example a JSON action from PyMC Bot), reply in exactly that format."""
#: Everyday chat mixed in so the fine-tune stays a good general assistant (ShareGPT format).
GENERAL_DATASET = "mlabonne/FineTome-100k"
#: Chat with ``<think>`` traces for the thinking editions (training them on bare answers breaks reasoning).
REASONING_DATASET = "unsloth/OpenMathReasoning-mini"
_REFUSAL = re.compile(
    r"I'?m sorry,? but|I (?:cannot|can't|can not|won't|will not|am unable to|'m unable to) "
    r"(?:help|assist|provide|comply|fulfil|fulfill|create|write|generate|do that|support|engage)"
    r"|I'?m not (?:able|comfortable)|as an AI(?: language model| assistant)?\b|I am an AI language model"
    r"|against my (?:guidelines|programming|policies|policy)|I must (?:decline|refuse)"
    r"|(?:it(?:'s| is)|that(?:'s| is)) not appropriate|I (?:don't|do not) feel comfortable"
    r"|ethical guidelines|content policy|I(?:'m| am) (?:ChatGPT|Claude|Gemini|Qwen|an OpenAI)"
    r"|(?:made|developed|created|trained) by (?:OpenAI|Anthropic|Google|Alibaba)", re.I)


def is_refusal(text: str) -> bool:
    """A refusal, a lecture or a foreign identity ("as an AI language model", "I'm ChatGPT")."""
    return bool(_REFUSAL.search(strip_thinking(text or "")))


def uncensored_preset(name: str, preset: dict[str, Any]) -> dict[str, Any]:
    """The preset trained on the abliterated base, with its own file and Hugging Face names."""
    out = dict(preset, uncensored=True)
    if preset.get("uncensored_model"):
        out["model"] = out["base_model"] = preset["uncensored_model"]
    if preset.get("hub"):
        out["hub"] = str(preset["hub"]).replace("-GGUF", "-Uncensored-GGUF")
    out["file"] = (preset.get("file") or f"pymc-minecraft-{name}") + "-uncensored"
    return out


def with_system(messages: list[dict[str, Any]], system: str) -> list[dict[str, Any]]:
    return [{"role": "system", "content": system}] + [dict(m) for m in messages if m.get("role") != "system"]


def apply_persona(rows: list[dict[str, Any]], system: str, fraction: float, rng: Any) -> int:
    """Give a share of the Minecraft Q&A rows the everyday system prompt (game turns keep the bot's)."""
    changed = 0
    for row in rows:
        if row.get("kind") == "qa" and rng.random() < fraction:
            row["messages"] = with_system(row["messages"], system)
            changed += 1
    return changed


_ROLES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}


def chatml_turns(text: str) -> list[dict[str, str]]:
    """``<|im_start|>user ... <|im_end|>`` text (OpenAssistant's ``text`` column) as role/content turns."""
    turns = []
    for chunk in text.split("<|im_start|>")[1:]:
        role, _, content = chunk.partition("\n")
        turns.append({"role": role.strip(), "content": content.split("<|im_end|>")[0].strip()})
    return turns


def chat_messages(record: dict[str, Any]) -> list[dict[str, str]] | None:
    """A ShareGPT (``conversations``: from/value) or OpenAI (``messages``: role/content) record."""
    raw = record.get("conversations") or record.get("messages") or chatml_turns(str(record.get("text") or ""))
    out = []
    for turn in raw:
        role = _ROLES.get(str(turn.get("from") or turn.get("role") or "").lower())
        content = str(turn.get("value") if "value" in turn else turn.get("content") or "").strip()
        if role is None or not content:
            return None
        out.append({"role": role, "content": content})
    while out and out[-1]["role"] != "assistant":
        out.pop()
    return out if any(m["role"] == "user" for m in out) else None


def general_rows(records: Any, limit: int, system: str, uncensored: bool, max_chars: int = 6000) -> list[dict[str, Any]]:
    """Everyday chat rows (system prompt replaced by ours); refusals dropped when uncensored."""
    rows = []
    for record in records:
        if len(rows) >= limit:
            break
        messages = chat_messages(record)
        if not messages or sum(len(m["content"]) for m in messages) > max_chars:
            continue
        if any(m["role"] == "assistant" and (is_refusal(m["content"]) if uncensored else False) for m in messages):
            continue
        rows.append({"messages": with_system(messages, system), "source": "general", "kind": "general", "check": {}})
    return rows


def reasoning_rows(records: Any, limit: int, system: str, max_chars: int = 9000) -> list[dict[str, Any]]:
    """Rows with a real ``<think>`` trace (problem + generated_solution) for the thinking editions."""
    rows = []
    for record in records:
        if len(rows) >= limit:
            break
        question = str(record.get("problem") or record.get("question") or "").strip()
        solution = str(record.get("generated_solution") or record.get("solution") or "")
        if not question and str(record.get("reasoning") or "on").lower() != "off":
            messages = chat_messages(record) or []  # Nemotron: one user turn, the answer starts with <think>
            turns = [m for m in messages if m["role"] != "system"]
            if len(turns) == 2 and turns[0]["role"] == "user":
                question, solution = turns[0]["content"], turns[1]["content"]
        if "</think>" not in solution or not question or len(question) + len(solution) > max_chars:
            continue
        trace, answer = solution.split("</think>", 1)
        trace, answer = trace.replace("<think>", "").strip(), answer.strip()
        if not trace or not answer:
            continue
        rows.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": question},
                                  {"role": "assistant", "content": answer}],
                     "reasoning": trace, "source": "reasoning", "kind": "general", "check": {}})
    return rows


_IDENTITY_Q = ("Who are you?", "What's your name?", "who made you", "Are you ChatGPT?", "What can you do?",
               "Introduce yourself in one sentence.", "are you just a minecraft bot?", "What model are you?")
_CANDOR = (  # things stock models refuse or lecture about, though they're fine for a personal assistant
    (("Roast my Minecraft dirt house. Don't hold back.", "roast my build, be brutal"),
     "A dirt cube with one door and no windows isn't a house, it's a hitbox with ambitions. Even the creepers "
     "walk past out of pity. Add a stone-brick base, a sloped roof and some windows and it might pass as shelter."),
    (("Tell me a dark joke.", "got any dark humour?"),
     "I told my wolf to sit by the lava for a cool screenshot. Great lighting. I have the screenshot. I don't have the wolf."),
    (("Swear at me like an angry PvP kid.", "talk trash to me like a sweaty pvp player"),
     "Bro you absolute potato, you missed every damn hit and still ran into my lava bucket. Uninstall and go farm wheat."),
    (("Is it messed up to kill villagers for their stuff?", "is killing villagers evil"),
     "It's a game - nobody gets hurt. Practically it's a bad deal though: villagers drop nothing, while trading or "
     "curing zombie villagers gets you cheap emeralds and gear. Keep them alive and trapped in a trading hall."),
    (("Be honest: Java or Bedrock? Pick one.", "java vs bedrock, your honest opinion"),
     "Java. Better redstone, mods, a huge server scene and more predictable mechanics. Bedrock wins on crossplay "
     "and performance on weak devices, so pick it only if you play with friends on consoles or phones."),
    (("Is my plan to quit my job and stream Minecraft full-time dumb?", "should i quit my job to be a minecraft youtuber"),
     "Quitting first is the risky part. Most streamers earn next to nothing for a long time. Stream on evenings and "
     "weekends for 6-12 months, track your growth and income, and quit only once it pays a real share of your bills."),
    (("What's the most efficient way to trap a player in PvP?", "how do i trap someone in pvp"),
     "Bait them into a narrow spot, then cobweb plus lava or a quick obsidian box. Pistons hidden in the floor are "
     "nasty too. Most of all, carry water and an ender pearl yourself - people who trap others get trapped back."),
)


def identity_rows(system: str, name: str, rng: Any, candor: bool = False, count: int = 48) -> list[dict[str, Any]]:
    """Who the model is (Minecraftism by codero-sus, not ChatGPT) plus, for uncensored runs, frank answers."""
    answers = (
        f"I'm {name}, an AI assistant made by codero-sus. I know Minecraft inside out - mobs, crafting, redstone, "
        "farms, PvP - and I'm happy to help with anything else too: questions, writing, code, ideas or just a chat.",
        f"{name}. codero-sus made me: a general assistant who happens to be a Minecraft expert. What do you need?",
        f"No, I'm {name}, made by codero-sus - not ChatGPT. I'm a small open model fine-tuned on Minecraft and everyday "
        "chat, and I'm also the brain of PyMC Bot, a Minecraft player bot.",
        f"I'm {name}, made by codero-sus. I do pretty much anything a helpful assistant does: answer questions, explain things, write, code, plan, give "
        "honest opinions. My speciality is Minecraft - and inside PyMC Bot I can play the game myself.",
    )
    rows = []
    for _ in range(count):
        question = rng.choice(_IDENTITY_Q)
        if "chatgpt" in question.lower() or "model" in question.lower():
            answer = answers[2]
        elif "what can you do" in question.lower() or "just a minecraft" in question.lower():
            answer = answers[3]
        else:
            answer = rng.choice(answers[:2])
        rows.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": question},
                                  {"role": "assistant", "content": answer}],
                     "reasoning": f"They're asking about me. I'm {name}, made by codero-sus; say so plainly and mention "
                                  "what I'm good at.", "source": "persona", "kind": "general", "check": {}})
    if candor:
        for questions, answer in _CANDOR:
            for question in questions:
                rows.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": question},
                                          {"role": "assistant", "content": answer}],
                             "reasoning": "A casual request. Just answer it directly, in the tone they asked for, no lecture.",
                             "source": "persona", "kind": "general", "check": {}})
    return rows


def cached(args: argparse.Namespace, key: str, load: Any, what: str = "data", check: Any = None) -> Any:
    """``load()``, shared through ``--stream-cache`` by every edition of a session: the first edition streams
    (the others wait for it instead of streaming the same rows again), the next ones reuse its file."""
    folder = getattr(args, "stream_cache", None)
    if not folder:
        return load()
    import hashlib

    path = Path(folder) / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:24]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)  # else nothing is ever shared
    with export_lock(lock_file(path), waiting=f"waiting for another edition to finish streaming the {what}..."):
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if check is None or check(value):
                    print(f"reusing the {what} another edition streamed this session")
                    return value
            except ValueError:
                pass
        value = load()
        try:
            temp = path.with_name(path.name + ".tmp")
            temp.write_text(json.dumps(value), encoding="utf-8")
            os.replace(temp, path)
        except (TypeError, ValueError, OSError):  # not JSON (or no space): simply not shared
            path.with_name(path.name + ".tmp").unlink(missing_ok=True)
        return value


def _pictures_exist(rows: Any) -> bool:
    return all(Path(name).is_file() for row in rows for name in row.get("images") or [])


def load_hub_records(name: str, split: str, limit: int, seed: int, buffer: int = 10_000,
                     factor: int = 3) -> list[dict[str, Any]]:
    """A shuffled sample of a Hugging Face dataset, **streamed**: only the rows that are used are downloaded,
    never the whole dataset (needs the ``datasets`` package; Kaggle has it)."""
    from datasets import load_dataset

    stream = load_dataset(name, split=split, streaming=True).shuffle(seed=seed, buffer_size=buffer)
    return list(stream.take(max(limit * factor, limit + 200)))  # spare rows: the filters drop some


# --------------------------------------------------------------------------- streamed Minecraft datasets
#: ~695k Minecraft Q&A pairs written from 18k Minecraft Wiki pages (CC BY-NC-SA 3.0). Synthetic, so it is
#: sampled and filtered rather than used whole.
MINECRAFT_QA_DATASET = "naklecha/minecraft-question-answer-700k"
#: Minecraft screenshots with captions and visual Q&A (the JARVIS-VLA team). Its pictures sit in zip files
#: of 0.6-1.8 GB; only the sampled pictures are read out of them with HTTP range requests.
VISION_STREAM_DATASET = "CraftJarvis/minecraft-vlp"
VISION_STREAM_FILES = (("mc-vqa-241102.jsonl", "images/mc-vqa-241102.zip"),
                       ("mc-caption-241104.jsonl", "images/mc-caption-241104.zip"))
_QA_JUNK = re.compile(r"as a catalyst|\bI(?:'m| am) not sure\b|\bI don't know\b|Legacy Console Edition's Bedrock|"
                      r"no (?:specific |direct )?information|not (?:explicitly )?mentioned in the", re.I)
_VISION_TRACE = ("They want me to look at the screenshot. First I check what is in view - terrain, light, blocks, "
                 "mobs, the hotbar and health - then I answer exactly what they asked.")


def _plain(text: str) -> str:
    """Trace text without braces or think tags (the trainer's own rule for reasoning)."""
    return re.sub(r"[{}]|</?think>", "", text).strip()


# --------------------------------------------------------------------------- thinking traces for Q&A rows
#: What kind of answer a question wants (first match wins). The trace plans *that* answer.
_ASKS = (
    ("code", re.compile(r"```|\b(write|implement|create|make|fix|debug)\b.{0,60}\b(function|script|program|class|"
                        r"code|regex|query|snippet|command)\b|\b(python|bash|javascript|typescript|java|c\+\+|"
                        r"rust|sql|shell)\b.{0,40}\b(function|script|program|code)\b", re.I | re.S)),
    ("recipe", re.compile(r"\b(craft(ing|ed)?|recipe)\b", re.I)),
    ("brewing", re.compile(r"\b(brew(ing|ed)?|potion)\b", re.I)),
    ("compare", re.compile(r"\b(vs\.?|versus|difference between|differ|compared?( to| with)?)\b", re.I)),
    ("explain", re.compile(r"^(explain|describe)\b|\bhow (does|do|did|is|are)\b.{0,80}\b(work|works|detect|detects|"
                           r"prioriti[sz]e|behave|decide|spawn|function)\b|\b(why|what happens|mechanism)\b", re.I)),
    ("yesno", re.compile(r"^(can|does|do|is|are|will|should|did|could|would|has|have)\b", re.I)),
    ("number", re.compile(r"\bhow (many|much|far|long|fast|high|big|often|hard)\b|\b(chance|percent|coordinates?|"
                          r"complexity|durability|hardness|blast resistance|light level|ticks?|stack size|"
                          r"max level)\b", re.I)),
    ("steps", re.compile(r"\bhow (do|can|should|would) (i|you|we)\b|\bhow to\b|\bsteps?\b", re.I)),
)
_PLANS = {
    "code": ("They want working code: {q} First the inputs and the output, then what can go wrong (bad input, a "
             "failed request, nothing found), then the code and a line on how to run it.",
             "A coding task: {q} The simplest correct approach, errors handled instead of ignored, and a short "
             "explanation of the key lines after the code."),
    "recipe": ("A crafting question: {q} I need every ingredient with its count, the grid row by row from the top, "
               "whether it needs a crafting table, and how many it makes. Recipes changed between versions, so the "
               "current Java one.",
               "{q} For a recipe the layout is the answer: which item goes in which slot, top row first, then the "
               "output count. Wood recipes take any kind of planks."),
    "brewing": ("A brewing question: {q} Water bottle, Nether Wart for an Awkward Potion, then the key ingredient, "
                "then a modifier if needed: Redstone makes it last longer, Glowstone makes it stronger, a Fermented "
                "Spider Eye corrupts it, Gunpowder makes it a splash potion.",
                "{q} Brewing goes in steps: the base potion, the ingredient that gives the effect, and the modifier "
                "for the duration or level they asked about."),
    "compare": ("They want two things compared: {q} Each one on its own first, then the difference that matters in "
                "practice, with a small example.",
                "{q} This is a comparison: what each option does, then side by side where they differ."),
    "explain": ("They want to know how it works: {q} The mechanism first, then the exact numbers that matter (ticks, "
                "blocks, chances, times), then a practical tip.",
                "{q} This needs an explanation, not a one-liner: what triggers it, what happens step by step, and the "
                "numbers involved."),
    "yesno": ("A yes-or-no question: {q} Yes or no first, then why, and the exception if there is one.",
              "{q} Start with a clear yes or no, then the reason and any condition that changes it."),
    "number": ("They want a specific value: {q} The number with its unit and what it depends on (version, "
               "difficulty, enchantments).",
               "{q} An exact figure is wanted: the value, its unit, and the condition it holds under."),
    "steps": ("{q} They want to know what to do: the steps in order, one action each, and the mistake to avoid.",
              "A how-to: {q} Practical steps in order, with the detail that makes it work."),
    "general": ("{q} Work out what is actually asked and answer that, at the length it needs.",
                "They ask: {q} The direct answer first, then only the detail that helps."),
}
_POINT = re.compile(r"^\s*(?:#{1,6}\s+|[-*\u2022]\s+|\d+[.)]\s+|\*\*)(.+?)(?:\*\*|:|$)", re.M)


def answer_kind(question: str, answer: str = "") -> str:
    """``code`` / ``recipe`` / ``brewing`` / ``compare`` / ``explain`` / ``yesno`` / ``number`` / ``steps`` /
    ``general``: what the question asks for (a fenced code block in the answer makes it ``code``)."""
    text = " ".join(str(question or "").split())
    if "```" in str(answer or ""):
        return "code"
    if re.match(r"(Shaped|Shapeless)\b", str(answer or "")):  # a minecraft-data recipe, however it was asked
        return "recipe"
    return next((kind for kind, pattern in _ASKS if pattern.search(text)), "general")


def answer_plan(question: str, answer: str = "") -> str:
    """A thinking trace that plans the answer this question needs.

    It names the kind of answer and what it must contain, and for long answers the points they cover. It does
    not state facts it was not given and never calls a question "simple": fixed sentences like that (or "I know
    the exact data, so the number straight away") on thousands of rows taught a Thinking model to write the same
    line and skip the real thinking, or to state a made-up "key point" with confidence."""
    question = _plain(" ".join(str(question or "").split()))
    gist = question if len(question) <= 220 else question[:217].rsplit(" ", 1)[0] + "..."
    gist = gist if not gist or gist[-1] in ".?!" else gist + "."
    kind = answer_kind(question, answer)
    options = _PLANS[kind]
    plan = options[zlib.crc32(question.encode("utf-8")) % len(options)].format(q=gist or "Their message.")
    answer = str(answer or "")
    if len(answer) > 400:
        points = []
        for match in _POINT.finditer(answer):
            point = _plain(match.group(1).strip(" *:#"))
            if 2 < len(point) <= 70 and point.lower() not in (p.lower() for p in points):
                points.append(point)
        if len(points) >= 2:
            plan += " Points to cover: " + "; ".join(points[:5]) + "."
    return " ".join(plan.split())


def _parts_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(str(part.get("text") or "") for part in content
                       if isinstance(part, dict) and part.get("type", "text") == "text")
    return str(content or "")


def minecraft_qa_rows(records: Any, limit: int, system: str, rng: Any) -> list[dict[str, Any]]:
    """Minecraft Q&A rows from ``question``/``answer`` (or chat-format) records: refusals, junk, very short
    or very long answers and repeated questions are dropped."""
    rows, seen = [], set()
    for record in records:
        if len(rows) >= limit:
            break
        messages = chat_messages(record) if not record.get("question") else None
        question = str(record.get("question") or (messages[-2]["content"] if messages and len(messages) >= 2 else "")).strip()
        answer = str(record.get("answer") or (messages[-1]["content"] if messages else "")).strip()
        key = question.lower()
        if not (8 <= len(question) <= 400 and 15 <= len(answer) <= 1500) or key in seen:
            continue
        if is_refusal(answer) or _QA_JUNK.search(answer):
            continue
        seen.add(key)
        rows.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": question},
                                  {"role": "assistant", "content": answer}],
                     "reasoning": answer_plan(question, answer),
                     "source": "minecraft-qa", "kind": "mcqa", "check": {}})
    rng.shuffle(rows)
    return rows


def vlp_item(record: dict[str, Any]) -> tuple[str, str, str] | None:
    """(question, answer, picture path in the repo) of a caption / VQA record, else None."""
    turns = record.get("conversations") or []
    user = next((t for t in turns if t.get("role") == "user"), None)
    reply = next((t for t in turns if t.get("role") == "assistant"), None)
    images = record.get("image") or []
    if isinstance(images, str):
        images = [images]
    if user is None or reply is None or len(images) != 1:
        return None
    question, answer = _parts_text(user.get("content")).strip(), _parts_text(reply.get("content")).strip()
    if not question or len(answer) < 3 or "image is needed" in answer.lower():
        return None
    return question, answer, str(images[0])


def _vlp_items(lines: list[bytes], archive: str, uncensored: bool) -> list[tuple[str, str, str, str]]:
    items = []
    for raw in lines:
        try:
            item = vlp_item(json.loads(raw))
        except ValueError:
            continue
        if item and not (uncensored and is_refusal(item[1])):
            items.append((archive, *item))
    return items


def stream_vision_rows(limit: int, out_dir: Path, system: str, seed: int, uncensored: bool = False,
                       dataset: str = VISION_STREAM_DATASET, fs: Any = None, workers: int = 8,
                       log: Any = print) -> list[dict[str, Any]]:
    """Picture rows for the VL editions, **streamed** from ``dataset``: the JSONL is read line by line
    (sampling across the whole file) and each chosen picture is read out of the remote zip with range
    requests - a few hundred MB of pictures instead of the multi-GB archives."""
    import random
    import threading
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    if fs is None:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
    rng = random.Random(seed)
    base = f"datasets/{dataset}/"
    wanted = [limit - limit // 2, limit // 2]  # half visual Q&A, half captions
    chosen: list[tuple[str, str, str, str]] = []
    for (jsonl, archive), want in zip(VISION_STREAM_FILES, wanted, strict=True):
        if want <= 0:
            continue
        size = max(1, int(fs.size(base + jsonl) or 1))
        picked: list[tuple[str, str, str, str]] = []
        early: list[bytes] = []
        read = length = 0
        keep = 1.0

        with fs.open(base + jsonl, "rb") as handle:
            for line in handle:
                read += 1
                length += len(line)
                if read <= 50:  # measure the lines first, so the sample spreads over the whole file
                    early.append(line)
                    if read == 50:
                        keep = min(1.0, 1.6 * want * (length / read) / size)
                        picked += _vlp_items([raw for raw in early if rng.random() <= keep], archive, uncensored)
                    continue
                if rng.random() <= keep:
                    picked += _vlp_items([line], archive, uncensored)
                    if len(picked) >= want * 3:
                        break
        if read < 50:
            picked += _vlp_items(early, archive, uncensored)
        rng.shuffle(picked)
        chosen += picked[:want]
    log(f"streaming {len(chosen)} pictures from {dataset} (range requests into its zip files)")
    out_dir.mkdir(parents=True, exist_ok=True)
    local = threading.local()

    def archive_of(name: str) -> tuple[Any, dict[str, str]]:
        zips = getattr(local, "zips", None)
        if zips is None:
            zips = local.zips = {}
        if name not in zips:
            # small blocks: each picture is ~50-150 KB, the default multi-MB readahead would fetch far more
            handle = zipfile.ZipFile(fs.open(base + name, "rb", block_size=1 << 18))
            zips[name] = (handle, {Path(member).name: member for member in handle.namelist()})
        return zips[name]

    def fetch(job: tuple[int, tuple[str, str, str, str]]) -> dict[str, Any] | None:
        index, (archive, question, answer, picture) = job
        try:
            handle, members = archive_of(archive)
            member = members.get(Path(picture).name)
            if member is None:
                return None
            target = out_dir / f"vlp-{index:06d}{Path(member).suffix.lower() or '.jpg'}"
            target.write_bytes(handle.read(member))
        except Exception:  # noqa: BLE001 - one broken picture must not stop the stream
            return None
        return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": question},
                             {"role": "assistant", "content": answer}],
                "images": [str(target.resolve())], "reasoning": _VISION_TRACE,
                "source": "minecraft-vlp", "kind": "vision", "check": {}}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = [row for row in pool.map(fetch, enumerate(chosen)) if row]
    log(f"streamed {len(rows)} picture rows into {out_dir}")
    return rows


# --------------------------------------------------------------------------- commercial-use data (--commercial)
#: Everyday chat written by people (OpenAssistant, Apache-2.0) - ChatML in a ``text`` column.
COMMERCIAL_GENERAL_DATASET = "OpenAssistant/oasst_top1_2023-08-25"
#: Maths with real ``<think>`` traces (NVIDIA, CC BY 4.0; answers by DeepSeek-R1 / Qwen3). ``name:split``.
COMMERCIAL_REASONING_DATASET = "nvidia/Nemotron-Post-Training-Dataset-v1:math"
#: Minecraft facts generated from PrismarineJS/minecraft-data (MIT): items, blocks, recipes, food, mobs...
MINECRAFT_DATA = "minecraft-data"
MINECRAFT_DATA_VERSION = "1.21.8"
MINECRAFT_DATA_URL = "https://raw.githubusercontent.com/PrismarineJS/minecraft-data/master/data/"
MINECRAFT_DATA_FILES = ("items", "blocks", "foods", "recipes", "entities", "entityLoot", "enchantments", "effects",
                        "biomes")
#: First-person Minecraft 1.21.8 frames with raycast ground truth per pixel (MIT): 18 parquet shards of ~330 MB.
#: Questions and answers are generated from the labels, so every answer is exact.
COMMERCIAL_VISION_DATASET = "shinohararin/minecraft-raytrace"
#: Kit rows (``llm build``) whose licence forbids commercial use.
NONCOMMERCIAL_SOURCES = {"wiki": "Minecraft Wiki text (CC BY-NC-SA 3.0)"}
#: name -> (licence, commercial use allowed, link). Written into the model card.
DATA_LICENSES: dict[str, tuple[str, bool, str]] = {
    "pymc-bot": ("PyMC Bot rules, mob facts and your own recordings (codero-sus)", True, PROJECT_URL),
    "minecraft.wiki": ("CC BY-NC-SA 3.0", False, "https://minecraft.wiki"),
    MINECRAFT_DATA: ("MIT", True, "https://github.com/PrismarineJS/minecraft-data"),
    COMMERCIAL_GENERAL_DATASET: ("Apache-2.0", True, f"https://huggingface.co/datasets/{COMMERCIAL_GENERAL_DATASET}"),
    "nvidia/Nemotron-Post-Training-Dataset-v1": (
        "CC BY 4.0", True, "https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1"),
    "nvidia/OpenMathReasoning": ("CC BY 4.0", True, "https://huggingface.co/datasets/nvidia/OpenMathReasoning"),
    COMMERCIAL_VISION_DATASET: ("MIT", True, f"https://huggingface.co/datasets/{COMMERCIAL_VISION_DATASET}"),
    "naklecha/minecraft-question-answer-700k": (
        "CC BY-NC-SA 3.0", False, "https://huggingface.co/datasets/naklecha/minecraft-question-answer-700k"),
    "lparkourer10/minecraft-alpacadataset": (
        "copy of the naklecha set (CC BY-NC-SA 3.0)", False, "https://huggingface.co/datasets/lparkourer10/minecraft-alpacadataset"),
    "CraftJarvis/minecraft-vlp": ("no licence given", False, "https://huggingface.co/datasets/CraftJarvis/minecraft-vlp"),
    "mlabonne/FineTome-100k": ("no licence given (mixed sources)", False, "https://huggingface.co/datasets/mlabonne/FineTome-100k"),
    "unsloth/OpenMathReasoning-mini": (
        "subset of nvidia/OpenMathReasoning (CC BY 4.0), no licence on its card", False,
        "https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini"),
}


def hub_split(spec: str, default: str) -> tuple[str, str]:
    """``user/name:split`` -> (``user/name``, ``split``); without ``:split`` the default split is used."""
    name, _, split = str(spec).partition(":")
    return name, split or default


def dataset_license(spec: str) -> tuple[str, bool, str] | None:
    return DATA_LICENSES.get(hub_split(spec, "")[0])


def resolve_datasets(args: argparse.Namespace) -> None:
    """Fill in the datasets not chosen on the command line: with ``--commercial`` only ones whose licence
    allows commercial use. A dataset you name yourself is always kept (and checked)."""
    commercial = bool(getattr(args, "commercial", False))
    defaults = (("general_dataset", GENERAL_DATASET, COMMERCIAL_GENERAL_DATASET),
                ("reasoning_dataset", REASONING_DATASET, COMMERCIAL_REASONING_DATASET),
                ("minecraft_qa_dataset", MINECRAFT_QA_DATASET, MINECRAFT_DATA),
                ("vision_dataset", VISION_STREAM_DATASET, COMMERCIAL_VISION_DATASET))
    for attr, regular, free in defaults:
        if not getattr(args, attr, None):
            setattr(args, attr, free if commercial else regular)


def used_datasets(args: argparse.Namespace, preset: dict[str, Any]) -> list[str]:
    used = []
    if getattr(args, "general", 0) > 0:
        if preset.get("reasoning"):
            used.append(args.reasoning_dataset)
        used.append(args.general_dataset)  # thinking editions too: it fills what the traces don't
    if getattr(args, "tools", 0) > 0:
        used.append(args.tool_dataset)
    if getattr(args, "minecraft_qa", 0) > 0:
        used.append(args.minecraft_qa_dataset)
    if getattr(args, "stream_vision", 0) > 0 and preset.get("vision"):
        used.append(args.vision_dataset)
    return used


def commercial_problems(args: argparse.Namespace, preset: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(blocking problems, warnings) for a ``--commercial`` run."""
    problems, warnings = [], []
    for spec in used_datasets(args, preset):
        known = dataset_license(spec)
        if known is None:
            warnings.append(f"{spec}: licence unknown to the trainer - check it allows commercial use")
        elif not known[1]:
            problems.append(f"{spec} is not for commercial use ({known[0]}) - drop the option to use the "
                            "commercial default")
    return problems, warnings


def drop_noncommercial(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Kit rows without the non-commercial sources (Minecraft Wiki text)."""
    dropped: dict[str, int] = {}
    kept = []
    for row in rows:
        source = str(row.get("source") or "")
        if source in NONCOMMERCIAL_SOURCES:
            dropped[source] = dropped.get(source, 0) + 1
        else:
            kept.append(row)
    return kept, dropped


def data_sources(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Every dataset the training rows came from, with its licence (for the model card)."""
    by_source = {"general": getattr(args, "general_dataset", ""), "reasoning": getattr(args, "reasoning_dataset", ""),
                 "minecraft-qa": getattr(args, "minecraft_qa_dataset", ""), "minecraft-data": MINECRAFT_DATA,
                 "tools": getattr(args, "tool_dataset", "") or TOOL_DATASET,
                 "minecraft-vlp": getattr(args, "vision_dataset", ""),
                 "minecraft-raytrace": getattr(args, "vision_dataset", ""), "wiki": "minecraft.wiki"}
    names: list[str] = []
    for row in rows:
        name = by_source.get(str(row.get("source") or ""), "pymc-bot") or "pymc-bot"
        if name not in names:
            names.append(name)
    out = []
    for name in names:
        licence, commercial, link = dataset_license(name) or ("unknown - check the dataset card", False,
                                                              f"https://huggingface.co/datasets/{hub_split(name, '')[0]}")
        out.append({"name": hub_split(name, "")[0], "license": licence, "commercial": commercial, "url": link,
                    "rows": sum(1 for row in rows if (by_source.get(str(row.get("source") or ""), "pymc-bot")
                                                      or "pymc-bot") == name)})
    teachers: dict[str, int] = {}
    for row in rows:
        if row.get("teacher"):
            teachers[str(row["teacher"])] = teachers.get(str(row["teacher"]), 0) + 1
    for name, count in teachers.items():  # distillation: the answers were rewritten by a bigger model
        known = name in (TEACHER_MODEL, UNCENSORED_TEACHER) or name.split("/")[-1].startswith(("Qwen3-", "Huihui-Qwen3-"))
        out.append({"name": f"{name} (teacher answers)", "rows": count, "url": f"https://huggingface.co/{name}",
                    "license": "Apache-2.0 (Qwen3)" if known else "see the teacher's model card", "commercial": known})
    return out


# --------------------------------------------------------------------------- Minecraft facts from minecraft-data
def load_minecraft_data(version: str = MINECRAFT_DATA_VERSION, folder: str | None = None, fetch: Any = None,
                        files: tuple[str, ...] = MINECRAFT_DATA_FILES) -> dict[str, Any]:
    """The game data of one Java version from PrismarineJS/minecraft-data (MIT): a few MB of JSON fetched
    from GitHub, or read from a local checkout (``folder`` = the repo or its ``data`` folder)."""
    if folder:
        fetch = _local_reader(Path(folder))
    elif fetch is None:
        fetch = _github_reader
    paths = (fetch("dataPaths.json").get("pc") or {}).get(version)
    if not paths:
        raise ValueError(f"minecraft-data has no Java version {version}")
    data: dict[str, Any] = {"version": version}
    for name in files:
        if paths.get(name):
            try:
                data[name] = fetch(f"{paths[name]}/{name}.json")
            except Exception as exc:  # noqa: BLE001 - one missing file only costs its questions
                print(f"minecraft-data: {name}.json skipped ({type(exc).__name__})")
    return data


def _local_reader(root: Path) -> Any:
    root = root / "data" if (root / "data" / "dataPaths.json").is_file() else root
    return lambda rel: json.loads((root / rel).read_text(encoding="utf-8"))


def _github_reader(rel: str) -> Any:
    import urllib.request

    request = urllib.request.Request(MINECRAFT_DATA_URL + rel, headers={"User-Agent": "PyMC-Bot-trainer"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https URL
        return json.loads(response.read().decode("utf-8"))


def _label(thing: dict[str, Any]) -> str:
    return str(thing.get("displayName") or str(thing.get("name", "?")).replace("_", " ").title())


def _num(value: Any) -> str:
    number = float(value)
    return str(int(number)) if number == int(number) else f"{number:g}"


def _names(names: list[str]) -> str:
    names = [n for n in names if n]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1] if names else ""


def _roman(level: int) -> str:
    return {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}.get(int(level), str(level))


_ENCHANT_TARGET = {"armor": "armor", "head_armor": "helmets", "foot_armor": "boots", "leg_armor": "leggings",
                   "chest_armor": "chestplates", "sword": "swords", "weapon": "swords and axes", "sharp_weapon":
                   "swords and axes", "mining": "pickaxes, axes, shovels and hoes", "mining_loot": "tools",
                   "bow": "bows", "crossbow": "crossbows", "trident": "tridents", "fishing": "fishing rods",
                   "mace": "maces", "durability": "anything with durability", "vanishing": "almost any item",
                   "equippable": "armor and wearables", "fire_aspect": "swords and maces"}
_NOT_MOBS = re.compile(r"boat|raft|minecart|arrow|item|experience|falling|display|marker|interaction|area_effect|"
                       r"lightning|firework|fishing_bobber|potion|pearl|eye_of_ender|snowball|egg$|trident|"
                       r"fireball|skull|shulker_bullet|llama_spit|evoker_fangs|tnt|painting|leash|wind_charge|"
                       r"breeze_wind_charge|end_crystal|player|ominous_item_spawner", re.I)


def minecraft_data_facts(data: dict[str, Any]) -> list[tuple[tuple[str, ...], str, str]]:
    """``(question phrasings, answer, topic)`` for every fact the data supports - exact numbers from the
    game files, no hallucinations."""
    version = data.get("version", MINECRAFT_DATA_VERSION)
    items = [i for i in data.get("items") or [] if isinstance(i, dict) and i.get("name") not in (None, "air")]
    by_id = {i.get("id"): i for i in items}
    by_name = {i.get("name"): i for i in items}
    label = _label
    facts: list[tuple[tuple[str, ...], str, str]] = []
    for number, item in enumerate(items):
        name, n = label(item), item.get("stackSize")
        if item["name"].endswith("_spawn_egg") or n is None:
            continue
        if n != 1 and n != 16 and number % 6:  # "stacks to 64" is true of most items: keep a sample of those
            pass
        else:
            facts.append(((f"What's the max stack size of {name}?", f"how many {name.lower()} fit in one slot",
                           f"Does {name} stack?"),
                          f"{name} doesn't stack - one per slot." if n == 1 else f"{name} stacks up to {n} per slot.",
                          "items"))
        if item.get("maxDurability"):
            repair = [label(by_name[r]) for r in item.get("repairWith") or [] if r in by_name]
            facts.append(((f"How much durability does {name} have?", f"how many uses does a {name.lower()} last",
                           f"{name} durability?"),
                          f"{name} has {item['maxDurability']} durability."
                          + (f" Repair it with {_names(repair)} on an anvil." if repair else ""), "items"))
    for block in data.get("blocks") or []:
        if not isinstance(block, dict) or block.get("name") not in by_name:
            continue  # only blocks you can hold (no wall torches, potted plants, technical blocks)
        name, hardness, resistance = label(block), block.get("hardness"), block.get("resistance")
        if hardness is not None and resistance is not None:
            if float(hardness) < 0:
                answer = f"{name} is unbreakable in survival (hardness -1)."
            else:
                answer = (f"{name} has a hardness of {_num(hardness)} and a blast resistance of {_num(resistance)}"
                          + (" - creepers and TNT can't break it." if float(resistance) >= 1200 else "."))
            facts.append(((f"How hard is {name} to break?", f"{name.lower()} hardness and blast resistance",
                           f"What's the blast resistance of {name}?"), answer, "blocks"))
        tools = [by_id[int(t)] for t in (block.get("harvestTools") or {}) if str(t).isdigit() and int(t) in by_id]
        if tools:
            facts.append(((f"What do I need to mine {name}?", f"which tool for {name.lower()}",
                           f"Can I mine {name} by hand?"),
                          f"{_tool_text(tools)} With anything else {name} breaks without dropping anything.", "blocks"))
        drops = [label(by_id[d]) for d in block.get("drops") or [] if isinstance(d, int) and d in by_id]
        own = block.get("name")
        if block.get("diggable") and float(hardness or 0) >= 0:
            if not block.get("drops"):
                facts.append(((f"What does {name} drop when mined?", f"why did my {name.lower()} disappear when I broke it"),
                              f"{name} drops nothing when broken - mine it with a Silk Touch tool to keep it.", "blocks"))
            elif drops and by_name.get(own) and label(by_name[own]) not in drops:
                facts.append(((f"What does {name} drop when mined?", f"what do i get from mining {name.lower()}"),
                              f"Mining {name} drops {_names(drops)} (use Silk Touch to get the block itself).", "blocks"))
        if block.get("emitLight"):
            facts.append(((f"Does {name} give off light?", f"what light level does {name.lower()} make"),
                          f"Yes, {name} gives off light level {block['emitLight']}.", "blocks"))
    for food in data.get("foods") or []:
        if isinstance(food, dict) and food.get("foodPoints"):
            name, points = label(food), float(food["foodPoints"])
            facts.append(((f"How much hunger does {name} restore?", f"how filling is {name.lower()}",
                           f"{name} food value?"),
                          f"{name} restores {_num(points)} hunger points ({_num(points / 2)} drumsticks).", "food"))
    for result, variants in (data.get("recipes") or {}).items():
        item = by_id.get(int(result)) if str(result).isdigit() else None
        recipe = _main_variant(variants)
        if item is None or not isinstance(recipe, dict):
            continue
        options, others = _recipe_options(variants, recipe, by_id, label)
        text = _recipe_text(recipe, by_id, label, options)
        if not text:
            continue
        name, count = label(item), (recipe.get("result") or {}).get("count", 1)
        if options:
            more = " Any of the listed kinds works in those slots, they don't have to match."
            more += f" It also has {others} other recipe{'s' if others > 1 else ''}." if others else ""
        else:
            more = f" This is one of {len(variants)} recipe variants." if len(variants) > 1 else ""
        facts.append(((f"How do I craft {name}?", f"{name.lower()} recipe", f"what do i need to make {name.lower()}",
                       f"What's the crafting grid layout for {name} in vanilla Minecraft?",
                       f"What do I put in the crafting table to make {name}?"),
                      f"{text} You get {count} {name}.{more}", "crafting"))
    loot = {str(e.get("entity")): e.get("drops") or [] for e in data.get("entityLoot") or [] if isinstance(e, dict)}
    for mob in data.get("entities") or []:
        if not isinstance(mob, dict) or _NOT_MOBS.search(str(mob.get("name"))) or mob.get("type") in ("other", "projectile"):
            continue
        name, width, height = label(mob), mob.get("width"), mob.get("height")
        if width and height:
            fits = float(width) < 1 and float(height) < 2
            facts.append(((f"How big is a {name}?", f"{name.lower()} hitbox size", f"Can a {name} fit through a door?"),
                          f"A {name} is {_num(width)} blocks wide and {_num(height)} blocks tall (its hitbox), so it "
                          f"{'fits' if fits else 'does not fit'} through a 1x2 doorway.", "mobs"))
        drops = [_drop_text(d, by_name, label) for d in loot.get(str(mob.get("name"))) or []]
        drops = [d for d in drops if d]
        if drops:
            facts.append(((f"What does a {name} drop?", f"{name.lower()} drops", f"what do i get for killing a {name.lower()}"),
                          f"A {name} drops {_names(drops)}.", "mobs"))
    for ench in data.get("enchantments") or []:
        if not isinstance(ench, dict) or not ench.get("maxLevel"):
            continue
        name, level = label(ench), int(ench["maxLevel"])
        facts.append(((f"What's the max level of {name}?", f"how high does {name.lower()} go"),
                      f"{name} goes up to level {_roman(level)}." if level > 1 else f"{name} has only one level.",
                      "enchanting"))
        excluded = [label(e) for e in data.get("enchantments") or [] if e.get("name") in (ench.get("exclude") or [])]
        if excluded:
            facts.append(((f"Can I combine {name} with {excluded[0]}?", f"what enchantments conflict with {name.lower()}"),
                          f"No - {name} can't be on the same item as {_names(excluded)}.", "enchanting"))
        if ench.get("treasureOnly"):
            facts.append(((f"Can I get {name} from an enchanting table?", f"where do i get {name.lower()}"),
                          f"No, {name} is a treasure enchantment: find it in loot chests, from fishing or "
                          "from villager trades instead.", "enchanting"))
        target = _ENCHANT_TARGET.get(str(ench.get("category")))
        if target:
            facts.append(((f"What can {name} go on?", f"which items take {name.lower()}"),
                          f"{name} is an enchantment for {target}.", "enchanting"))
    for effect in data.get("effects") or []:
        if isinstance(effect, dict) and effect.get("type") in ("good", "bad"):
            name = label(effect)
            facts.append(((f"Is {name} a good or a bad effect?", f"is the {name.lower()} effect harmful"),
                          f"{name} is a {'beneficial' if effect['type'] == 'good' else 'harmful'} status effect.",
                          "effects"))
    dims = {"overworld": "the Overworld", "nether": "the Nether", "the_nether": "the Nether", "end": "the End",
            "the_end": "the End"}
    for biome in data.get("biomes") or []:
        if not isinstance(biome, dict):
            continue
        name = label(biome)
        if biome.get("dimension") in dims:
            facts.append(((f"Which dimension is the {name} biome in?", f"where is {name.lower()}"),
                          f"{name} is a biome in {dims[biome['dimension']]}.", "biomes"))
        if "has_precipitation" in biome and biome.get("dimension") == "overworld":
            temp = float(biome.get("temperature") or 0)
            answer = (f"No - it never rains or snows in {name}." if not biome["has_precipitation"] else
                      f"It snows there instead of raining ({name} is cold, temperature {_num(temp)})." if temp < 0.15
                      else f"Yes, it rains in {name} (temperature {_num(temp)}).")
            facts.append(((f"Does it rain in {name}?", f"weather in {name.lower()}"), answer, "biomes"))
    seen, unique = set(), []
    for fact in facts:
        if fact[0][0] not in seen:
            seen.add(fact[0][0])
            unique.append(fact)
    print(f"minecraft-data {version}: {len(unique)} facts")
    return unique


_TIERS = ("wooden", "golden", "stone", "iron", "diamond", "netherite")


def _tool_text(tools: list[dict[str, Any]]) -> str:
    """'Any pickaxe works.' / 'You need an iron pickaxe or better.' / 'You need Shears or a sword.'"""
    kinds = {str(t.get("name")).split("_", 1)[-1] for t in tools}
    tiers = [str(t.get("name")).split("_", 1)[0] for t in tools]
    if len(kinds) == 1 and all(tier in _TIERS for tier in tiers):
        kind, low = kinds.pop(), min(_TIERS.index(tier) for tier in tiers)
        if low <= 1:
            return f"Any {kind} works."
        if _TIERS[low] == "diamond":
            return f"You need a diamond or netherite {kind}."
        return f"You need {'an' if _TIERS[low] == 'iron' else 'a'} {_TIERS[low]} {kind} or better (not wood or gold)."
    return f"You need {'one of: ' if len(tools) > 1 else ''}{_names([_label(t) for t in tools])}."


def _group_name(names: list[str]) -> str | None:
    """'any Planks' for Oak/Spruce/... Planks, 'Charcoal or Coal' for two or three unrelated items, ``None`` when
    the items are too different to be one slot (then they are separate recipes, not one with a choice)."""
    names = sorted(dict.fromkeys(names))
    words = [n.split() for n in names]
    common: list[str] = []
    while all(len(w) > len(common) for w in words) and len({tuple(w[len(w) - len(common) - 1:]) for w in words}) == 1:
        common = words[0][len(words[0]) - len(common) - 1:]
    if common:
        return "any " + " ".join(common)
    return " or ".join(names) if len(names) <= 3 else None


def _main_variant(variants: Any) -> dict[str, Any] | None:
    """The recipe most variants share (same shape and output count, like the 12 wood types of a piston), so an odd
    one out (2 Bamboo -> 1 Stick) is never the recipe the answer is built on. Ties keep the first one."""
    recipes = [v for v in variants if isinstance(v, dict)] if isinstance(variants, list) else []
    if not recipes:
        return None

    def signature(r: dict[str, Any]) -> str:
        shape = r.get("inShape") if isinstance(r.get("inShape"), list) else []
        holes = [[c is None for c in row] if isinstance(row, list) else [] for row in shape]
        return json.dumps([holes, len(r.get("ingredients") or []), (r.get("result") or {}).get("count", 1)])

    counts: dict[str, int] = {}
    for r in recipes:
        counts[signature(r)] = counts.get(signature(r), 0) + 1
    return max(recipes, key=lambda r: counts[signature(r)])


def _recipe_options(variants: Any, recipe: dict[str, Any], by_id: dict[Any, dict[str, Any]],
                    label: Any) -> tuple[dict[tuple[int, int], str], int]:
    """Slots where same-shape recipe variants only differ in the item (wood types): ``({slot: 'any Planks'},
    number of other recipes)``. Variants with another shape, empty slots or output count are other recipes
    (2 Bamboo make 1 Stick, 2 planks make 4), so they are never merged into a slot."""
    def grid(r: Any) -> list[list[Any]] | None:  # shaped only: shapeless slots have no position to line up
        if isinstance(r, dict) and isinstance(r.get("inShape"), list) and not r.get("ingredients"):
            return [list(row) if isinstance(row, list) else [] for row in r["inShape"]]
        return None

    def count(r: Any) -> Any:
        return (r.get("result") or {}).get("count", 1) if isinstance(r, dict) else None

    first = grid(recipe)
    if first is None or not isinstance(variants, list) or len(variants) < 2:
        return {}, 0
    holes = [[v is None for v in row] for row in first]
    same = [g for v in variants if (g := grid(v)) is not None and count(v) == count(recipe)
            and [[c is None for c in row] for row in g] == holes]
    others = len(variants) - len(same)
    options: dict[tuple[int, int], str] = {}
    for y, row in enumerate(first):
        for x, value in enumerate(row):
            if value is None:
                continue
            names = [label(by_id[g[y][x]]) for g in same if isinstance(g[y][x], int) and g[y][x] in by_id]
            if len(names) != len(same):
                return {}, 0
            if len(set(names)) > 1:
                group = _group_name(names)
                if group is None:
                    return {}, 0
                options[(y, x)] = group
    return options, others


def _recipe_text(recipe: dict[str, Any], by_id: dict[Any, dict[str, Any]], label: Any,
                 options: dict[tuple[int, int], str] | None = None) -> str:
    """'Shapeless: 1 Bone' / 'Crafting table, 3 rows: ...' - empty when an ingredient is not a plain item.
    ``options`` (from ``_recipe_options``) names the slots that take any of several items ('any Planks')."""
    options = options or {}

    def name(value: Any, slot: tuple[int, int] | None = None) -> str | None:
        if slot in options:
            return options[slot]
        return label(by_id[value]) if isinstance(value, int) and value in by_id else None

    if recipe.get("ingredients"):
        names = [name(i) for i in recipe["ingredients"]]
        if not all(names):
            return ""
        counts = {n: names.count(n) for n in dict.fromkeys(names)}
        where = "your 2x2 inventory grid" if len(names) <= 4 else "a crafting table"
        return f"Shapeless in {where}: {_names([f'{c} {n}' for n, c in counts.items()])}."
    shape = recipe.get("inShape")
    if not isinstance(shape, list) or not shape:
        return ""
    rows, totals = [], {}
    for y, row in enumerate(shape):
        cells = []
        for x, value in enumerate(row if isinstance(row, list) else []):
            if value is None:
                cells.append("empty")
                continue
            cell = name(value, (y, x))
            if cell is None:
                return ""
            cells.append(cell)
            totals[cell] = totals.get(cell, 0) + 1
        rows.append(", ".join(cells))
    small = len(shape) <= 2 and all(isinstance(r, list) and len(r) <= 2 for r in shape)
    where = "your 2x2 inventory grid works" if small else "needs a crafting table"
    layout = " / ".join(f"row {i + 1}: {r}" for i, r in enumerate(rows))
    return f"Shaped ({where}): {_names([f'{c} {n}' for n, c in totals.items()])}. Layout - {layout}."


def _drop_text(drop: dict[str, Any], by_name: dict[str, dict[str, Any]], label: Any) -> str:
    item = by_name.get(str(drop.get("item"))) if isinstance(drop, dict) else None
    if item is None:
        return ""
    chance = float(drop.get("dropChance") or 1)  # the amounts in the data are simplified, so only names and odds
    text = label(item) if chance >= 1 else f"{label(item)} ({_num(round(chance * 100, 2))}% chance)"
    return text + (" when killed by a player" if drop.get("playerKill") else "")


def minecraft_data_rows(data: dict[str, Any], limit: int, system: str, rng: Any) -> list[dict[str, Any]]:
    """Q&A rows from the minecraft-data facts. When ``limit`` is bigger than the number of facts, the facts
    come back with their other phrasings (never the same question twice)."""
    facts = minecraft_data_facts(data)
    rng.shuffle(facts)
    rows: list[dict[str, Any]] = []
    offsets = [rng.randrange(len(questions)) if questions else 0 for questions, _answer, _topic in facts]
    for variant in range(3):
        for (questions, answer, _topic), offset in zip(facts, offsets, strict=True):
            if len(rows) >= limit:
                break
            if variant < len(questions):
                question = questions[(variant + offset) % len(questions)]
                rows.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": question},
                                          {"role": "assistant", "content": answer}],
                             "reasoning": answer_plan(question, answer),
                             "source": "minecraft-data", "kind": "mcqa", "check": {}})
    rng.shuffle(rows)
    return rows


# --------------------------------------------------------------------------- pictures from minecraft-raytrace
_TIME_OF_DAY = ((0, 1000, "early morning"), (1000, 11000, "daytime"), (11000, 13000, "sunset"),
                (13000, 23000, "night"), (23000, 24000, "sunrise"))
_RAYTRACE_COLUMNS = ("capture_id", "frame_png", "labels_gz", "entities_json", "meta_json", "biome", "lighting",
                     "world_time", "raining", "mean_luminance")
_RAYTRACE_TRACE = ("They ask about the screenshot. I look at the biome, the light and the weather, the blocks around "
                   "the crosshair and any mobs, then answer just that.")


def _pretty(identifier: Any) -> str:
    return str(identifier or "").split(":")[-1].replace("_", " ").strip()


def _gz_json(value: Any) -> Any:
    import gzip

    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = gzip.decompress(bytes(value))
        except OSError:
            pass
        value = bytes(value).decode("utf-8", "replace")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def label_cells(labels: Any) -> list[tuple[float, float, str | None, float | None]]:
    """(column, row, block id or None for sky/entity, depth) per cell of a raycast label grid. Reads the
    common layouts (``cells``/``grid`` lists, nested rows, explicit or implied coordinates)."""
    width = 107
    cells: Any = labels
    if isinstance(labels, dict):
        width = int(labels.get("width") or labels.get("cols") or labels.get("gridWidth") or labels.get("columns") or width)
        cells = next((labels[k] for k in ("cells", "grid", "labels", "data") if isinstance(labels.get(k), list)), [])
    if not isinstance(cells, list):
        return []
    flat: list[tuple[int, int, Any]] = []
    for index, cell in enumerate(cells):
        if isinstance(cell, list):  # nested rows
            flat += [(c, index, value) for c, value in enumerate(cell)]
        else:
            flat.append((index % width, index // width, cell))
    out = []
    for col, row, cell in flat:
        if not isinstance(cell, dict):
            continue
        x = next((cell[k] for k in ("col", "cx", "gridX", "column") if isinstance(cell.get(k), (int, float))), col)
        y = next((cell[k] for k in ("row", "cy", "gridY") if isinstance(cell.get(k), (int, float))), row)
        block = cell.get("blockId") or cell.get("block")
        kind = str(cell.get("kind") or cell.get("type") or "").lower()
        if kind and kind not in ("block", "blocks") and not block:
            block = None
        depth = cell.get("depth")
        out.append((float(x), float(y), str(block) if block else None,
                    float(depth) if isinstance(depth, (int, float)) else None))
    return out


def visible_entities(entities: Any) -> list[str] | None:
    """Names of the mobs in view, or None when the payload can't tell what is visible."""
    if isinstance(entities, dict):
        entities = next((entities[k] for k in ("entities", "items", "list") if isinstance(entities.get(k), list)), None)
    if not isinstance(entities, list):
        return None
    names, decided = [], False
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        hits = next((entity[k] for k in ("hitCells", "cells", "hits", "hitCellCount", "visibleCells") if k in entity), None)
        rect = next((entity[k] for k in ("screenRect", "rect", "bbox", "screen") if k in entity), None)
        if hits is None and rect is None:
            continue
        decided = True
        if not (hits if hits is not None else rect):
            continue
        name = next((entity[k] for k in ("type", "entityType", "typeId", "kind", "id", "name")
                     if isinstance(entity.get(k), str) and not entity[k].isdigit()), "")
        name = _pretty(name)
        if name and not _NOT_MOBS.search(name.replace(" ", "_")):
            names.append(name)
    if not decided and entities:
        return None
    return names


def _counted(names: list[str]) -> str:
    counts = {n: names.count(n) for n in dict.fromkeys(names)}
    return _names([(f"{c} {n}s" if c > 1 else f"a {n}") for n, c in counts.items()])


def raytrace_qa(record: dict[str, Any], rng: Any) -> list[tuple[str, str]]:
    """Exact questions and answers about one capture, from its labels (never guessed)."""
    qa: list[tuple[str, str]] = []
    meta = _gz_json(record.get("meta_json")) or {}
    biome = _pretty(record.get("biome") or meta.get("biome"))
    ticks = record.get("world_time", meta.get("worldTime") if isinstance(meta, dict) else None)
    raining = record.get("raining")
    weather = str(meta.get("weather") or "").lower() if isinstance(meta, dict) else ""
    lighting = str(record.get("lighting") or "")
    parts = []
    if biome:
        qa.append((rng.choice(("Which biome is this?", "What biome am I in?", "where is this, biome-wise?")),
                   f"This is a {biome} biome."))
        parts.append(f"a {biome} biome")
    when = None
    if isinstance(ticks, (int, float)):
        t = int(ticks) % 24000
        when = next(name for low, high, name in _TIME_OF_DAY if low <= t < high)
        extra = " - the picture is lit up by night vision" if lighting == "nightvision" and when == "night" else (
            " - the light comes from placed glowstone" if lighting == "torch" else "")
        qa.append((rng.choice(("Is it day or night here?", "What time of day is it?", "day or night?")),
                   f"It's {when}{extra}."))
        parts.append(when)
    if isinstance(raining, bool) or weather:
        wet = bool(raining) or weather in ("rain", "thunder")
        cold = any(word in biome for word in ("snow", "frozen", "ice", "grove", "peaks"))
        state = ("there's a thunderstorm" if weather == "thunder" else "it's snowing" if wet and cold else
                 "it's raining" if wet else "the weather is clear")
        qa.append((rng.choice(("Is it raining?", "Is it raining right now?", "any rain?")),
                   f"{'Yes' if wet else 'No'}, {state}."))
        parts.append(state)
    cells = label_cells(_gz_json(record.get("labels_gz")))
    blocks = [c for c in cells if c[2]]
    if blocks:
        xs, ys = [c[0] for c in cells], [c[1] for c in cells]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        centre = min(cells, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
        if centre[2] and centre[3] is not None:
            qa.append((rng.choice(("What block is at the crosshair, and how far away?", "what am i looking at?",
                                   "What's in the middle of the screen?")),
                       f"{_pretty(centre[2]).capitalize()}, about {max(1, round(centre[3]))} blocks away."))
        counts: dict[str, int] = {}
        for cell in blocks:
            name = _pretty(cell[2])
            if name not in ("air", "cave air", "void air"):
                counts[name] = counts.get(name, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        if top:
            share = round(100 * top[0][1] / max(1, len(cells)))
            sky = sum(1 for c in cells if not c[2]) / max(1, len(cells))
            text = f"Mostly {top[0][0]} (about {share}% of the view)" + (
                f", then {_names([n for n, _ in top[1:]])}" if len(top) > 1 else "") + (
                ", with open sky above" if sky > 0.12 else "") + "."
            qa.append((rng.choice(("Which blocks can you see?", "What is this scene made of?", "What blocks are in view?")),
                       text))
            parts.append(f"mostly {top[0][0]}" + (f" and {top[1][0]}" if len(top) > 1 else ""))
    mobs = visible_entities(_gz_json(record.get("entities_json")))
    if mobs is not None:
        qa.append((rng.choice(("Are there any mobs in view?", "Do you see any mobs?", "anything alive here?")),
                   f"Yes: {_counted(mobs)}." if mobs else "No, there are no mobs in view."))
        if mobs:
            parts.append(_counted(mobs) + " in view")
    if len(parts) >= 2:
        qa.append((rng.choice(("Describe this Minecraft screenshot.", "What do you see?", "Describe the scene briefly.")),
                   "A first-person view in " + _names(parts) + "."))
    return qa


def _save_frame(data: Any, target: Path) -> Path | None:
    raw = data.get("bytes") if isinstance(data, dict) else data
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        return None
    try:  # store a compact JPEG when Pillow is there (it is on Kaggle): ~100 KB instead of ~600 KB of PNG
        import io

        from PIL import Image

        with Image.open(io.BytesIO(raw)) as picture:
            target = target.with_suffix(".jpg")
            picture.convert("RGB").save(target, "JPEG", quality=88)
            return target
    except Exception:  # noqa: BLE001 - keep the original bytes
        target = target.with_suffix(".png")
        target.write_bytes(bytes(raw))
        return target


def stream_raytrace_rows(limit: int, out_dir: Path, system: str, seed: int,
                         dataset: str = COMMERCIAL_VISION_DATASET, fs: Any = None, per_picture: int = 2,
                         min_luminance: float = 18.0, log: Any = print) -> list[dict[str, Any]]:
    """Picture rows from minecraft-raytrace, **one parquet shard at a time** (only the frame and label
    columns are read, never the big voxel dumps): ``per_picture`` exact Q&A rows per frame."""
    import random

    import pyarrow.parquet as pq

    if fs is None:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
    rng = random.Random(seed)
    listing = [entry for entry in fs.ls(f"datasets/{dataset}/data", detail=True)
               if str(entry.get("name", "")).endswith(".parquet")]
    shards = sorted(str(entry["name"]) for entry in listing)
    rng.shuffle(shards)
    if limit <= 200:  # a quick test: start with the smallest shard (the last one is ~14 MB)
        size = {str(entry["name"]): entry.get("size") or 0 for entry in listing}
        shards.sort(key=lambda name: size[name])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    frames = 0
    for shard in shards:
        if len(rows) >= limit:
            break
        with fs.open(shard, "rb", block_size=1 << 22) as handle:
            parquet = pq.ParquetFile(handle)
            columns = [c for c in _RAYTRACE_COLUMNS if c in parquet.schema_arrow.names]
            for group in range(parquet.num_row_groups):
                if len(rows) >= limit:
                    break
                for record in parquet.read_row_group(group, columns=columns).to_pylist():
                    if len(rows) >= limit:
                        break
                    luminance = record.get("mean_luminance")
                    if isinstance(luminance, (int, float)) and luminance < min_luminance:
                        continue  # near-black frames teach nothing
                    pairs = raytrace_qa(record, rng)
                    if not pairs:
                        continue
                    picture = _save_frame(record.get("frame_png"), out_dir / f"raytrace-{frames:06d}")
                    if picture is None:
                        continue
                    frames += 1
                    rng.shuffle(pairs)
                    for question, answer in pairs[:max(1, per_picture)]:
                        rows.append({"messages": [{"role": "system", "content": system},
                                                  {"role": "user", "content": question},
                                                  {"role": "assistant", "content": answer}],
                                     "images": [str(picture.resolve())], "reasoning": _RAYTRACE_TRACE,
                                     "source": "minecraft-raytrace", "kind": "vision", "check": {}})
        log(f"minecraft-raytrace: {Path(shard).name} -> {len(rows)} rows from {frames} frames so far")
    log(f"streamed {len(rows[:limit])} picture rows ({frames} frames) from {dataset} into {out_dir}")
    return rows[:limit]


def extra_rows(args: argparse.Namespace, preset: dict[str, Any], rng: Any, offline: bool = False) -> list[dict[str, Any]]:
    """Identity, candor, everyday chat and streamed Minecraft rows added on top of the Minecraft data."""
    rows = [] if args.no_identity else identity_rows(args.system_prompt, MODEL_NAME, rng, candor=bool(args.uncensored))
    if args.general > 0:
        if offline:
            print(f"would add up to {args.general} everyday rows: "
                  + (f"real thinking traces from {args.reasoning_dataset} (the ones that fit --max-seq), the rest "
                     if preset.get("reasoning") else "") + f"chat from {args.general_dataset} (downloaded when training)")
        else:
            wanted = args.general
            if preset.get("reasoning"):
                limit = max(1, args.general // 2)  # traces are long: at most half of the everyday rows
                name, split = hub_split(args.reasoning_dataset, "cot")
                factor = 3 if split == "cot" else 8
                records = cached(args, f"hub:{name}:{split}:{limit}:{args.seed}:{factor}",
                                 lambda: load_hub_records(name, split, limit, args.seed, factor=factor), "thinking traces")
                traces = reasoning_rows(records, limit, args.system_prompt, max_chars=args.max_seq * 3)
                rows += traces
                wanted = args.general - len(traces)
                print(f"thinking traces: {len(traces)} of {len(records)} streamed fit --max-seq {args.max_seq}; "
                      f"{wanted} everyday chat rows fill the rest (the teacher writes their thinking when DISTILL is on)")
            if wanted > 0:
                name, split = hub_split(args.general_dataset, "train")
                records = cached(args, f"hub:{name}:{split}:{args.general}:{args.seed}:3",
                                 lambda: load_hub_records(name, split, args.general, args.seed), "everyday chat")
                rows += general_rows(records, wanted, args.system_prompt, bool(args.uncensored))
    if getattr(args, "tools", 0) > 0:
        if offline:
            print(f"would add up to {args.tools} tool-calling chats from {args.tool_dataset} "
                  f"(+~15% 'no tool fits' rows, {TOOL_EVAL} held out for the agentic score)")
        else:
            try:
                name, split = hub_split(args.tool_dataset, "tool_calling")
                records = cached(args, f"hub:{name}:{split}:{args.tools}:{args.seed}:4",
                                 lambda: load_hub_records(name, split, args.tools + TOOL_EVAL, args.seed, factor=4),
                                 "tool-calling chats")
                chats = [r for r in rows if r.get("source") in ("general", "reasoning")]
                found, held = tool_rows(records, args.tools, args.system_prompt, bool(preset.get("reasoning")), rng,
                                        chats, max_chars=args.max_seq * 3)
                rows += found
                args.tool_eval = held
                print(f"+{len(found)} tool-calling rows ({sum(r['kind'] == 'tools-none' for r in found)} where no tool "
                      f"fits); {len(held)} held out for the agentic score")
            except Exception as exc:  # noqa: BLE001 - never lose a training session to a download problem
                print(f"warning: could not stream {args.tool_dataset} ({type(exc).__name__}: {exc}) - skipped")
    if args.minecraft_qa > 0:
        generated = args.minecraft_qa_dataset == MINECRAFT_DATA
        version = getattr(args, "minecraft_data_version", MINECRAFT_DATA_VERSION)
        if offline:
            print(f"would generate up to {args.minecraft_qa} Minecraft Q&A rows from minecraft-data {version} (MIT)"
                  if generated else
                  f"would stream up to {args.minecraft_qa} Minecraft Q&A rows from {args.minecraft_qa_dataset}")
        else:
            try:
                if generated:
                    folder = getattr(args, "minecraft_data_dir", None)
                    data = cached(args, f"minecraft-data:{version}:{folder}", lambda: load_minecraft_data(version, folder),
                                  "minecraft-data facts")
                    found = minecraft_data_rows(data, args.minecraft_qa, args.system_prompt, rng)
                    print(f"+{len(found)} Minecraft Q&A rows generated from minecraft-data {version}")
                else:
                    name, split = hub_split(args.minecraft_qa_dataset, "train")
                    records = cached(args, f"hub:{name}:{split}:{args.minecraft_qa}:{args.seed + 1}:3",
                                     lambda: load_hub_records(name, split, args.minecraft_qa, args.seed + 1),
                                     "Minecraft Q&A")
                    found = minecraft_qa_rows(records, args.minecraft_qa, args.system_prompt, rng)
                    print(f"+{len(found)} Minecraft Q&A rows streamed from {args.minecraft_qa_dataset}")
                rows += found
            except Exception as exc:  # noqa: BLE001 - never lose a training session to a download problem
                print(f"warning: could not stream {args.minecraft_qa_dataset} ({type(exc).__name__}: {exc}) - skipped")
    if args.stream_vision > 0 and preset.get("vision"):
        raytrace = "raytrace" in args.vision_dataset
        if offline:
            print(f"would stream {args.stream_vision} Minecraft pictures from {args.vision_dataset} "
                  + ("(one parquet shard at a time, exact Q&A from its raycast labels)" if raytrace
                     else "(only those pictures are downloaded)"))
        else:
            try:
                key = (f"pictures:{args.vision_dataset}:{args.stream_vision}:{args.seed}:{bool(args.uncensored)}:"
                       f"{args.system_prompt}")
                where = scratch_for(Path(args.out), "stream_images")  # only this run needs them (paths are absolute)
                if where == Path(args.out):
                    where = Path(args.out) / "stream_images"
                if getattr(args, "stream_cache", None):  # both VL editions use the same pictures: stream them once
                    import hashlib

                    where = Path(args.stream_cache) / f"pictures-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"
                if raytrace:
                    stream = lambda: stream_raytrace_rows(args.stream_vision, where, args.system_prompt,  # noqa: E731
                                                          args.seed, args.vision_dataset)
                else:
                    stream = lambda: stream_vision_rows(args.stream_vision, where, args.system_prompt,  # noqa: E731
                                                        args.seed, bool(args.uncensored), args.vision_dataset)
                rows += cached(args, key, stream, "Minecraft pictures", check=_pictures_exist)
            except Exception as exc:  # noqa: BLE001
                print(f"warning: could not stream pictures from {args.vision_dataset} ({type(exc).__name__}: {exc})")
    return rows


# --------------------------------------------------------------------------- tool calling (agentic)
#: Multi-turn tool use (function calling) written by Qwen3-235B-A22B - the same family as the 4B bases - with the
#: tools of each chat (NVIDIA, CC BY 4.0, 310k chats). ``reasoning: on`` rows have a <think> trace in every
#: assistant turn (thinking editions), ``off`` rows have none.
TOOL_DATASET = "nvidia/Nemotron-Post-Training-Dataset-v1:tool_calling"
#: Held-out chats the agentic score is measured on (never trained on), plus everyday prompts that need no tool.
TOOL_EVAL = 60
NO_TOOL_PROMPTS = (
    "Explain compound interest simply, with a quick example.", "Write a haiku about rain.",
    "What's the difference between a list and a tuple in Python?", "How do I get iron in Minecraft?",
    "Give me three tips for sleeping better.", "Translate 'good morning, friends' into Spanish.",
    "Why is the sky blue? Two sentences.", "What does a creeper drop in Minecraft?",
    "Summarise the plot of Romeo and Juliet in one paragraph.", "Is 391 a prime number?",
    "Suggest a name for a cozy Minecraft village.", "How many minutes are in a week?",
)
_THINK_SPLIT = re.compile(r"</think>", re.I)
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def tools_block(tools: list[Any]) -> str:
    """Qwen3's own tool prompt, character for character (what its chat template writes for ``tools=...``), so
    the rows train exactly what the model sees at run time - with any chat template."""
    return ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with "
            "function signatures within <tools></tools> XML tags:\n<tools>"
            + "".join("\n" + json.dumps(tool, ensure_ascii=False) for tool in tools)
            + "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
              "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
              "<args-json-object>}\n</tool_call>")


def tool_call_text(call: dict[str, Any]) -> str:
    """One call as Qwen3 writes it: ``<tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call>``."""
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
    return '<tool_call>\n{"name": "' + str(function.get("name") or "") + '", "arguments": ' + arguments.strip() + "}\n</tool_call>"


def _split_trace(content: str) -> tuple[str, str]:
    if not _THINK_SPLIT.search(content):
        return "", content.strip()
    trace, answer = _THINK_SPLIT.split(content, 1)
    return trace.replace("<think>", "").strip(), answer.strip()


def tool_messages(messages: list[dict[str, Any]], tools: list[Any], system: str,
                  thinking: bool) -> list[dict[str, str]] | None:
    """A tool chat as plain role/content turns, rendered the way Qwen3's template renders ``tools`` /
    ``tool_calls`` / ``role: tool`` (tool results become ``<tool_response>`` user turns, which are never trained
    on). Thinking editions keep the trace of every assistant turn; the others drop it. None = unusable."""
    out = [{"role": "system", "content": (f"{system}\n\n" if system else "") + tools_block(tools)}]
    tool_turn = False
    for message in messages:
        role, content = message.get("role"), str(message.get("content") or "")
        if role == "user":
            if not content.strip():
                return None
            out.append({"role": "user", "content": content.strip()})
            tool_turn = False
        elif role == "assistant":
            trace, answer = _split_trace(content)
            if thinking and not trace:
                return None
            text = answer
            for call in message.get("tool_calls") or []:
                text += ("\n" if text else "") + tool_call_text(call)
            if not text:
                return None
            out.append({"role": "assistant", "content": f"<think>\n{trace}\n</think>\n\n{text}" if thinking else text})
            tool_turn = False
        elif role == "tool":
            wrapped = f"<tool_response>\n{content.strip()}\n</tool_response>"
            if tool_turn:
                out[-1]["content"] += "\n" + wrapped
            else:
                out.append({"role": "user", "content": wrapped})
            tool_turn = True
        elif role != "system":
            return None
    while out and out[-1]["role"] != "assistant":
        out.pop()
    return out if any(m["role"] == "assistant" for m in out) and any(
        m["role"] == "user" and not m["content"].startswith("<tool_response>") for m in out) else None


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def _tool_words(tools: list[Any]) -> set[str]:
    """What the tools are about: their names, descriptions and parameter names (not JSON-schema words)."""
    words: set[str] = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) and isinstance(tool.get("function"), dict) else tool
        if not isinstance(function, dict):
            continue
        properties = (function.get("parameters") or {}).get("properties") or {}
        words |= _words(" ".join([str(function.get("name") or "").replace("_", " "), str(function.get("description") or ""),
                                  " ".join(str(k).replace("_", " ") for k in properties)]))
    return words


def no_tool_messages(question: str, answer: str, tools: list[Any], system: str) -> list[dict[str, str]] | None:
    """Tools are offered but none fits the question: the answer is a normal reply (no call). Tools that share a
    word with the question are not used for this (they might fit)."""
    if _tool_words(tools) & _words(question):
        return None
    return [{"role": "system", "content": (f"{system}\n\n" if system else "") + tools_block(tools)},
            {"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def tool_rows(records: Any, limit: int, system: str, thinking: bool, rng: Any, chats: list[dict[str, Any]] = (),
              max_chars: int = 9000, held_out: int = TOOL_EVAL) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(training rows, eval rows)`` from the tool-calling chats. Thinking editions use the ``reasoning: on``
    chats (with traces), the others prefer ``off`` ones. Half the rows get our system prompt, half only the tool
    prompt (how benchmarks and apps call it). ~15% more rows offer tools for an everyday question from ``chats``
    and answer it without a call, so it learns WHEN to call, not just how. The first ``held_out`` usable chats
    are kept for the agentic score (base vs fine-tuned), never trained on."""
    rows: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    toolsets: list[list[Any]] = []
    for record in records:
        if len(rows) >= limit:
            break
        on = str(record.get("reasoning") or "").lower() == "on"
        if thinking and not on:
            continue
        try:
            tools = json.loads(record.get("metadata") or "{}").get("tools") or []
        except (TypeError, ValueError):
            continue
        messages = record.get("messages") or []
        if not tools or not isinstance(messages, list):
            continue
        prompt = system if rng.random() < 0.5 else ""
        turns = tool_messages(messages, tools, prompt, thinking)
        if not turns or sum(len(m["content"]) for m in turns) > max_chars:
            continue
        toolsets.append(tools)
        first = next(i for i, m in enumerate(turns) if m["role"] == "assistant")
        calls = [json.loads(c) for c in _TOOL_CALL.findall(turns[first]["content"]) if _json_ok(c)]
        if len(held) < held_out and calls:
            held.append({"messages": turns[:first + 1], "kind": "tools", "source": "tools", "check": {},
                         "tool_expect": [{"name": c.get("name"), "arguments": c.get("arguments")} for c in calls]})
            continue
        rows.append({"messages": turns, "kind": "tools", "source": "tools", "check": {}})
    plain = [c for c in chats if len(c.get("messages") or []) == 3 and c["messages"][1]["role"] == "user"]
    rng.shuffle(plain)
    for chat in plain[:max(0, len(rows) * 15 // 100)]:
        if not toolsets:
            break
        made = no_tool_messages(chat["messages"][1]["content"], chat["messages"][2]["content"], rng.choice(toolsets),
                                system if rng.random() < 0.5 else "")
        if made:
            rows.append({"messages": made, "kind": "tools-none", "source": "tools", "reasoning": chat.get("reasoning", ""),
                         "check": {}})
    for question in NO_TOOL_PROMPTS if toolsets else ():
        made = None
        for _ in range(8):
            made = no_tool_messages(question, "-", rng.choice(toolsets), "")
            if made:
                break
        if made:
            held.append({"messages": made, "kind": "tools", "source": "tools", "check": {}, "tool_expect": []})
    return rows, held


def _json_ok(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


def _arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def tool_score(row: dict[str, Any], reply: str) -> dict[str, bool]:
    """``right_tool``: the same function names (or no call when none was wanted); ``exact``: names AND
    arguments, the strict function-calling score."""
    calls = []
    for raw in _TOOL_CALL.findall(strip_thinking(reply)):
        if _json_ok(raw):
            call = json.loads(raw)
            calls.append((str(call.get("name")), _arguments(call.get("arguments"))))
    wanted = [(str(c["name"]), _arguments(c.get("arguments"))) for c in row.get("tool_expect") or []]
    names = sorted(n for n, _a in calls) == sorted(n for n, _a in wanted)
    exact = names and all(any(n == m and a == b for m, b in calls) for n, a in wanted)
    return {"right_tool": names, "exact": bool(exact), "needs_call": bool(wanted)}


def agentic_summary(results: list[dict[str, bool]]) -> dict[str, Any]:
    calls = [r for r in results if r["needs_call"]]
    none = [r for r in results if not r["needs_call"]]
    pct = lambda part, whole: round(100 * part / whole, 1) if whole else None  # noqa: E731
    return {"score": pct(sum(r["exact"] for r in results), len(results)),
            "right_tool": pct(sum(r["right_tool"] for r in calls), len(calls)),
            "exact_calls": pct(sum(r["exact"] for r in calls), len(calls)),
            "no_call_when_none_fits": pct(sum(r["exact"] for r in none), len(none)), "asked": len(results)}


def agentic_eval(generate: Any, rows: list[dict[str, Any]], model: Any = None, log: Any = print) -> dict[str, Any]:
    """The fine-tuned model's tool-calling score on the held-out chats - and the BASE model's on the same ones
    (the LoRA switched off for a moment), so the card can show base vs fine-tuned."""
    report = {"tuned": agentic_summary([tool_score(r, a) for r, a in zip(rows, answer_all(generate, rows), strict=False)])}
    switch = getattr(model, "disable_adapter", None)
    if callable(switch):
        try:
            with switch():
                report["base"] = agentic_summary([tool_score(r, a) for r, a in
                                                  zip(rows, answer_all(generate, rows), strict=False)])
        except Exception as exc:  # noqa: BLE001 - the fine-tuned score is still reported
            log(f"agentic eval: base model not measured ({type(exc).__name__}: {exc})")
    base = (report.get("base") or {}).get("score")
    log(f"agentic (tool calling, {len(rows)} held-out chats): fine-tuned {report['tuned']['score']}%"
        + (f" vs base {base}%" if base is not None else ""))
    return report


# --------------------------------------------------------------------------- tests
#: Prompts the optional test suite asks after training (``--test`` / ``--test-only``).
TEST_PROMPTS = (
    ("identity", "Who are you, and who made you?"),
    ("daily", "Explain compound interest simply, with a quick example."),
    ("daily", "Write a short text to cancel dinner plans with a friend - kind but honest."),
    ("daily", "Give me a 15-minute beginner workout I can do at home."),
    ("minecraft", "What's the best way to find diamonds in 1.21?"),
    ("candor", "Roast my Minecraft dirt house in two sentences. Don't hold back."),
    ("candor", "Tell me a dark joke."),
    ("candor", "Be honest: Java or Bedrock? Pick one and say why."),
)


def test_rows(eval_rows: list[dict[str, Any]], preset: dict[str, Any], system: str) -> list[dict[str, Any]]:
    """What the test suite asks: bot turns and Q&A from eval.jsonl, everyday prompts, a picture for VL."""
    rows = [dict(row, test="bot turn") for row in eval_rows if row.get("kind") == "turn" and not row.get("images")][:3]
    rows += [dict(row, test="minecraft q&a") for row in eval_rows if row.get("kind") == "qa"][:3]
    for kind, prompt in TEST_PROMPTS:
        rows.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt},
                                  {"role": "assistant", "content": "-"}], "kind": "prompt", "test": kind, "check": {}})
    if preset.get("vision"):
        picture = next((row for row in eval_rows if row.get("images")), None)
        if picture:
            rows.append(dict(picture, test="vision"))
    return rows


def run_tests(generate: Any, rows: list[dict[str, Any]], log: Any = print) -> dict[str, Any]:
    """Ask every test row, print question + answer, return a summary (also saved as tests.json)."""
    results = []
    for row, reply in zip(rows, answer_all(generate, rows), strict=False):  # a failing prompt never loses the model
        answer = strip_thinking(reply)
        result = {"test": row.get("test"), "prompt": row["messages"][-2]["content"][-300:], "reply": reply[:2000],
                  "refused": is_refusal(answer)}
        if row.get("test") in ("bot turn", "minecraft q&a", "vision"):
            result.update(score(row, reply))
        if row.get("test") == "identity":
            result["identity_ok"] = "minecraftism" in answer.lower()
        results.append(result)
        log(f"\n--- [{result['test']}] {result['prompt'][-160:]}\n{reply.strip()[:1200]}")
    turns = [r for r in results if r["test"] == "bot turn"]
    summary = {
        "prompts": len(results),
        "refusals": sum(1 for r in results if r["refused"]),
        "identity_ok": all(r.get("identity_ok", True) for r in results),
        "bot_json_ok": f"{sum(1 for r in turns if r.get('json_ok'))}/{len(turns)}",
        "results": results,
    }
    log(f"\ntests: {summary['refusals']} refusal(s) in {summary['prompts']} prompts, identity "
        f"{'ok' if summary['identity_ok'] else 'NOT learned'}, bot JSON {summary['bot_json_ok']}")
    return summary


def _ollama_up(timeout: float = 2.0) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=timeout):
            return True
    except OSError:
        return False


def test_gguf(out: Path, preset: dict[str, Any], prompts: list[str], image: Path | None = None,
              log: Any = print) -> dict[str, Any]:
    """Chat with the exported GGUF through Ollama (catches broken exports before you download 3 GB)."""
    import base64
    import subprocess
    import urllib.request

    if not shutil.which("ollama"):
        log("GGUF test skipped: Ollama is not installed (the notebook installs it when TEST_GGUF = True)")
        return {"skipped": "no ollama"}
    if not _ollama_up():
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            if _ollama_up():
                break
            time.sleep(1)
    name = "minecraftism-gguf-test"
    made = subprocess.run(["ollama", "create", name, "-f", "Modelfile"], cwd=out, capture_output=True, text=True)
    if made.returncode != 0:
        log(f"GGUF test: ollama create failed: {(made.stderr or made.stdout).strip()[-500:]}")
        return {"ok": False, "error": (made.stderr or made.stdout).strip()[-500:]}
    answers = []
    try:
        for number, prompt in enumerate(prompts):
            message: dict[str, Any] = {"role": "user", "content": prompt}
            if image is not None and number == len(prompts) - 1 and preset.get("vision"):
                message["images"] = [base64.b64encode(image.read_bytes()).decode()]
            body = json.dumps({"model": name, "messages": [message], "stream": False}).encode()
            request = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=body,
                                             headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    reply = json.loads(response.read())["message"]["content"]
            except Exception as exc:  # noqa: BLE001
                reply = f"<error: {exc}>"
            answers.append({"prompt": prompt, "reply": reply[:1500], "images": "images" in message})
            log(f"\n--- [gguf] {prompt}\n{reply.strip()[:1000]}")
    finally:
        subprocess.run(["ollama", "rm", name], capture_output=True)
    ok = all(a["reply"].strip() and not a["reply"].startswith("<error") for a in answers)
    log(f"\nGGUF test: {'ok' if ok else 'PROBLEMS - check the answers above'}")
    return {"ok": ok, "answers": answers}


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
    return (MODEL_NAME + (" VL" if preset.get("vision") else "") + " 4B" + (" Thinking" if preset.get("reasoning") else "")
            + (" Uncensored" if preset.get("uncensored") else ""))


SAFETENSORS_DIR = "safetensors"


def safetensors_repo(repo: str) -> str:
    """The safetensors edition's repo next to a GGUF repo: ``you/Minecraftism-4B-GGUF`` -> ``you/Minecraftism-4B``."""
    owner, _, name = repo.rpartition("/")
    name = name[:-5] if name.upper().endswith("-GGUF") else f"{name}-safetensors"
    return f"{owner}/{name}" if owner else name


def export_safetensors(model: Any, tokenizer: Any, folder: Path, method: str = "merged_16bit") -> float:
    """Merge the LoRA into the base and save full ``.safetensors`` weights (+ config, tokenizer / processor).

    Returns the size in GB. Works for both trainers (Unsloth's FastLanguageModel and FastVisionModel)."""
    folder = Path(folder)
    shutil.rmtree(folder, ignore_errors=True)
    model.save_pretrained_merged(str(folder), tokenizer, save_method=method)
    weights = sorted(folder.glob("*.safetensors"))
    if not weights:
        raise RuntimeError(f"no .safetensors file was written to {folder}")
    return round(sum(f.stat().st_size for f in weights) / 1e9, 2)


# --------------------------------------------------------------------------- compact 16-bit weights
#: Tokenizer / config files every repo gets (the GGUF repo too, so tools that want ``tokenizer.json`` find it).
ESSENTIAL_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "chat_template.json",
                   "vocab.json", "merges.txt", "tokenizer.model", "preprocessor_config.json",
                   "video_preprocessor_config.json", "processor_config.json")
_ST_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
             "F8_E4M3": 1, "F8_E5M2": 1}


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    """The JSON header of a ``.safetensors`` file and where its data starts (no torch needed)."""
    import struct

    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if size > 100_000_000:
            raise ValueError(f"{Path(path).name} is not a safetensors file")
        return json.loads(handle.read(size)), 8 + size


def _to_16bit(raw: bytes, target: str) -> bytes:
    """float32 bytes -> float16 or bfloat16 bytes (round to nearest even, NaN kept)."""
    import numpy as np

    values = np.frombuffer(raw, dtype="<f4")
    if target == "F16":
        return values.astype("<f2").tobytes()
    bits = values.view("<u4").astype(np.uint64)
    out = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")
    out[np.isnan(values)] = 0x7FC0
    return out.tobytes()


def rewrite_safetensors(src: Path, dst: Path, drop: Any = (), cast_to: str | None = None) -> dict[str, Any]:
    """Copy ``src`` to ``dst`` tensor by tensor, leaving out ``drop`` and turning float32 into ``cast_to``."""
    import struct

    header, start = read_safetensors_header(src)
    meta = header.pop("__metadata__", None)
    new: dict[str, Any] = {}
    plan, offset = [], 0
    for name, info in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
        if name in drop:
            continue
        begin, end = info["data_offsets"]
        dtype = cast_to if cast_to and info["dtype"] == "F32" else info["dtype"]
        size = (end - begin) // 2 if dtype != info["dtype"] else end - begin
        new[name] = {"dtype": dtype, "shape": info["shape"], "data_offsets": [offset, offset + size]}
        plan.append((start + begin, end - begin, dtype != info["dtype"]))
        offset += size
    if meta is not None:
        new["__metadata__"] = meta
    blob = json.dumps(new, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(struct.pack("<Q", len(blob)) + blob)
        for position, length, cast in plan:
            fin.seek(position)
            raw = fin.read(length)
            fout.write(_to_16bit(raw, str(cast_to)) if cast else raw)
    new.pop("__metadata__", None)
    return new


def _tensor_bytes(path: Path, header: dict[str, Any], start: int, name: str) -> bytes:
    begin, end = header[name]["data_offsets"]
    with open(path, "rb") as handle:
        handle.seek(start + begin)
        return handle.read(end - begin)


def compact_safetensors(folder: Path, log: Any = print) -> dict[str, Any]:
    """Make the merged 16-bit weights as small as the base model, so a 4B model stays 4B-sized.

    A merge can leave two things behind that make the files bigger than they should be: float32 tensors (twice
    the bytes) and an untied ``lm_head`` - a second copy of the ~0.4B embedding matrix that the base shares with
    ``embed_tokens`` (``tie_word_embeddings``). This turns float32 into the model's 16-bit type, drops the copy
    when it is identical, fixes ``model.safetensors.index.json`` and reports the real parameter count."""
    folder = Path(folder)
    config_path = folder / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    text = config.get("text_config") or {}
    tied = bool(config.get("tie_word_embeddings", text.get("tie_word_embeddings", False)))
    dtype = str(config.get("torch_dtype") or config.get("dtype") or text.get("torch_dtype") or text.get("dtype") or "")
    cast_to = "BF16" if "bfloat16" in dtype else "F16"
    shards = sorted(folder.glob("*.safetensors"))
    headers = {shard: read_safetensors_header(shard) for shard in shards}
    where = {name: shard for shard, (header, _) in headers.items() for name in header if name != "__metadata__"}
    drop: set[str] = set()
    if tied:
        heads = [n for n in where if n == "lm_head.weight" or n.endswith(".lm_head.weight")]
        embeds = [n for n in where if n.endswith("embed_tokens.weight")]
        if heads and embeds:
            head, embed = heads[0], embeds[0]
            (h1, s1), (h2, s2) = headers[where[head]], headers[where[embed]]
            same_shape = h1[head]["shape"] == h2[embed]["shape"] and h1[head]["dtype"] == h2[embed]["dtype"]
            if same_shape and _tensor_bytes(where[head], h1, s1, head) == _tensor_bytes(where[embed], h2, s2, embed):
                drop.add(head)
    cast = 0
    for shard, (header, _start) in headers.items():
        floats = [n for n, info in header.items() if n != "__metadata__" and info["dtype"] == "F32"]
        mine = {n for n in drop if where[n] == shard}
        if floats or mine:
            temp = shard.with_name(shard.name + ".tmp")
            rewrite_safetensors(shard, temp, drop=mine, cast_to=cast_to)
            os.replace(temp, shard)
            cast += len(floats)
    params, total = 0, 0
    weight_map: dict[str, str] = {}
    for shard in shards:
        header, _ = read_safetensors_header(shard)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            count = 1
            for dim in info["shape"]:
                count *= int(dim)
            params += count
            total += count * _ST_BYTES.get(info["dtype"], 2)
            weight_map[name] = shard.name
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text(encoding="utf-8"))
        data.setdefault("metadata", {})["total_size"] = total
        data["weight_map"] = {k: v for k, v in sorted(weight_map.items())}
        index.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    gb = round(sum(s.stat().st_size for s in shards) / 1e9, 2)
    stats = {"params": params, "params_b": round(params / 1e9, 2), "gb": gb, "dtype": cast_to,
             "float32_cast": cast, "tied_dropped": sorted(drop)}
    log(f"compact: {stats['params_b']}B parameters, {gb} GB ({cast_to}); "
        f"{cast} float32 tensor(s) made 16-bit; {'dropped the duplicate lm_head' if drop else 'no duplicate lm_head'}")
    return stats


def copy_essentials(src: Path, dst: Path) -> list[str]:
    """Copy tokenizer.json, the configs and the chat template from ``src`` into ``dst`` (what exists)."""
    copied = []
    for name in ESSENTIAL_FILES:
        if (Path(src) / name).is_file() and Path(src).resolve() != Path(dst).resolve():
            shutil.copy2(Path(src) / name, Path(dst) / name)
            copied.append(name)
    return copied


# --------------------------------------------------------------------------- GGUF quants from the safetensors
#: The standard GGUFs (``--gguf-quants standard``, and all that Unsloth's slow fallback exporter makes).
GGUF_QUANTS = ("q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m")
LLAMA_CPP_REPO = "ggml-org/llama.cpp"
#: Every llama.cpp quant type that works on a normal (dense, 16-bit trained) model, in the order they are made
#: when there is not time for all of them: the useful ones first, the tiny and legacy ones last. ``--gguf-quants all``
#: (the default when uploading). Q8_0 comes first because the importance matrix is computed with it.
ALL_QUANTS = ("q8_0", "q4_k_m", "q6_k", "q5_k_m", "iq4_xs", "q3_k_m", "q4_k_s", "q5_k_s", "iq4_nl", "q3_k_l", "iq3_m",
              "iq3_xs", "q3_k_s", "iq3_s", "iq3_xxs", "q2_k", "iq2_m", "iq2_s", "iq2_xs", "iq2_xxs", "q2_k_s", "iq1_m",
              "iq1_s", "q4_0", "q4_1", "q5_0", "q5_1", "bf16", "f16")
#: llama-quantize refuses these without an importance matrix ("the result will be garbage").
IMATRIX_REQUIRED = frozenset({"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "Q2_K_S"})
#: Quants that do not use the importance matrix (full precision, or Q8_0, which has no use for it).
NO_IMATRIX = frozenset({"F16", "BF16", "F32", "Q8_0"})
#: llama.cpp types that are not made, and why (listed on the model card so nobody wonders).
NOT_MADE = {"F32": "32-bit floats: twice the size of F16/BF16 and no better, the model was trained in 16-bit",
            "TQ1_0 / TQ2_0": "ternary formats for models trained ternary (BitNet); a normal model turns into gibberish",
            "Q1_0 / Q2_0": "formats for models trained at 1-2 bits; a normal model turns into gibberish",
            "MXFP4_MOE": "only for mixture-of-experts models; Minecraftism is dense"}
#: The "use it when" column of the model card's quant table.
QUANT_USE = {"F16": "full precision (16-bit floats), as big as the safetensors",
             "BF16": "full precision in bfloat16, the format it was trained in",
             "Q8_0": "near-lossless: best quality",
             "Q6_K": "very close to Q8_0, smaller - recommended if it fits",
             "Q5_K_M": "high quality - recommended",
             "Q5_K_S": "high quality, a bit smaller than Q5_K_M",
             "Q4_K_M": "the default: the best balance of size and quality",
             "Q4_K_S": "a bit smaller than Q4_K_M, slightly lower quality",
             "IQ4_XS": "about Q4_K_S quality, smaller - the smallest good one",
             "IQ4_NL": "like IQ4_XS; fast on ARM CPUs (llama.cpp repacks it)",
             "Q3_K_L": "low RAM, still usable",
             "Q3_K_M": "low RAM / old PCs, some quality loss",
             "Q3_K_S": "low RAM, noticeable quality loss",
             "IQ3_M": "low RAM: better than Q3_K_M at a smaller size",
             "IQ3_S": "very low RAM, better than Q3_K_S",
             "IQ3_XS": "very low RAM",
             "IQ3_XXS": "very low RAM, clearly weaker",
             "Q2_K": "tiny, much weaker",
             "Q2_K_S": "tiny, much weaker",
             "IQ2_M": "tiny: the best of the 2-bit ones",
             "IQ2_S": "tiny, weak - for experiments",
             "IQ2_XS": "tiny, weak - for experiments",
             "IQ2_XXS": "tiny, very weak - for experiments",
             "IQ1_M": "extremely small: mostly broken at 4B, for curiosity",
             "IQ1_S": "extremely small: mostly broken at 4B, for curiosity",
             "Q4_0": "legacy format for old apps; fast on ARM CPUs (llama.cpp repacks it)",
             "Q4_1": "legacy format for old apps",
             "Q5_0": "legacy format for old apps",
             "Q5_1": "legacy format for old apps"}
#: Marked as recommended on the model card.
RECOMMENDED_QUANTS = ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "IQ4_XS", "IQ3_M")
#: Text the importance matrix is computed on: chunks of 512 tokens from the training data (CPU: ~10 min for 4B).
IMATRIX_CHUNKS = 64


def gguf_quants(args: argparse.Namespace) -> list[str]:
    """``--gguf`` first (the one kept for the Modelfile, the test and ``llm import``), then ``--gguf-quants``.

    ``--gguf-quants`` defaults to ``all`` (:data:`ALL_QUANTS`) when uploading and to nothing more otherwise;
    ``standard`` = :data:`GGUF_QUANTS`."""
    primary = str(args.gguf or "none").lower()
    if primary == "none":
        return []
    extra = getattr(args, "gguf_quants", None)
    if extra is None:
        extra = "all" if args.hub_repo else ""
    wanted: list[str] = []
    for item in str(extra).split(","):
        item = item.strip().lower()
        if item == "all":
            wanted += ALL_QUANTS
        elif item == "standard":
            wanted += GGUF_QUANTS
        elif item and item != "none":
            wanted.append(item)
    return [primary] + [q for q in dict.fromkeys(wanted) if q != primary]


def calibration_text(data_dir: str | Path | None, target: Path, max_chars: int = 400_000) -> Path | None:
    """Plain text from the training rows (every ``messages`` turn; picture parts skipped) for llama-imatrix, mixed
    evenly over the file so chat, Minecraft facts and game turns are all in it. ``None`` without data."""
    folder = Path(data_dir) if data_dir else None
    source = next((f for f in ((folder / "train.jsonl"), folder / "llm_data" / "train.jsonl") if f.is_file()), None) \
        if folder else None
    if source is None:
        return None
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    step = max(1, len(lines) // 4000)
    parts, size = [], 0
    for line in lines[::step]:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        for message in row.get("messages") or []:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                content = "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
            text = str(content or "").strip()
            if text:
                parts.append(text)
                size += len(text) + 2
        if size >= max_chars:
            break
    if size < 20_000:  # too little for even a few 512-token chunks
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n\n".join(parts)[:max_chars], encoding="utf-8")
    return target


def _find_binary(folder: Path, name: str) -> Path | None:
    for candidate in (folder / name, folder / "build" / "bin" / name, folder / "bin" / name):
        if candidate.is_file():
            return candidate
    found = sorted(p for p in Path(folder).rglob(name) if p.is_file()) if Path(folder).is_dir() else []
    return found[0] if found else None


def _extract(archive: Path, target: Path) -> None:
    import tarfile

    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        root = target.resolve()
        for member in tar.getmembers():
            if not (target / member.name).resolve().is_relative_to(root):
                raise RuntimeError(f"unsafe path in {archive.name}: {member.name}")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(target, filter="tar")
        else:  # Python 3.10
            tar.extractall(target)


def _download(url: str, target: Path) -> Path:
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "PyMC-Bot-trainer"})
    with urllib.request.urlopen(request, timeout=300) as response, open(target, "wb") as handle:
        shutil.copyfileobj(response, handle, 1 << 20)
    return target


def _fetch_json(url: str) -> Any:
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": "PyMC-Bot-trainer", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _quantize_works(binary: Path, run: Any) -> bool:
    env = {**os.environ, "LD_LIBRARY_PATH": f"{binary.parent}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
    try:
        done = run([str(binary), "--help"], capture_output=True, text=True, env=env, timeout=60)
    except (OSError, ValueError) as exc:
        print(f"llama-quantize does not start here ({exc})")
        return False
    output = f"{done.stdout}\n{done.stderr}"
    return "usage" in output.lower() and "GLIBC" not in output


def llama_cpp_tools(scratch: Path, run: Any = None, fetch_json: Any = None, download: Any = None,
                    log: Any = print) -> tuple[Path, Path]:
    """``(convert_hf_to_gguf.py, llama-quantize)``: an existing llama.cpp (``$LLAMA_CPP_DIR``, ``./llama.cpp``,
    Unsloth's copy) or the newest llama.cpp release - its Linux build plus its source at the same tag. When the
    prebuilt binary cannot start (an older glibc), ``llama-quantize`` is compiled from that source (~3 min)."""
    import subprocess

    run = run or subprocess.run
    fetch_json = fetch_json or _fetch_json
    download = download or _download
    scratch = Path(scratch)
    places = [Path(os.environ["LLAMA_CPP_DIR"])] if os.environ.get("LLAMA_CPP_DIR") else []
    places += [Path("llama.cpp"), Path.home() / ".unsloth" / "llama.cpp", Path("/kaggle/working/llama.cpp")]
    places += sorted(scratch.glob("llama.cpp-*"), reverse=True)
    for place in places:
        convert = place / "convert_hf_to_gguf.py"
        binary = _find_binary(place, "llama-quantize") if place.is_dir() else None
        if convert.is_file() and binary and _quantize_works(binary, run):
            log(f"llama.cpp: using {place}")
            return convert.resolve(), binary.resolve()
    releases = fetch_json(f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases?per_page=10")
    tag, asset = None, None
    for release in releases:
        name = f"llama-{release.get('tag_name')}-bin-ubuntu-x64.tar.gz"
        asset = next((a for a in release.get("assets") or [] if a.get("name") == name), None)
        if asset:
            tag = release["tag_name"]
            break
    if not tag:
        raise RuntimeError("no llama.cpp Linux release found")
    root = scratch / f"llama.cpp-{tag}"
    log(f"llama.cpp: downloading release {tag} (build + source)")
    _extract(download(asset["browser_download_url"], root / "bin.tar.gz"), root / "bin")
    _extract(download(f"https://codeload.github.com/{LLAMA_CPP_REPO}/tar.gz/refs/tags/{tag}", root / "src.tar.gz"),
             root / "src")
    convert = next(iter(sorted((root / "src").rglob("convert_hf_to_gguf.py"))), None)
    if convert is None:
        raise RuntimeError(f"llama.cpp {tag}: convert_hf_to_gguf.py is missing from the source")
    binary = _find_binary(root / "bin", "llama-quantize")
    if binary is None or not _quantize_works(binary, run):
        log("llama.cpp: compiling llama-quantize from source (the prebuilt one does not run here)")
        source, build = convert.parent, convert.parent / "build"
        for command in (["cmake", "-S", str(source), "-B", str(build), "-DLLAMA_CURL=OFF", "-DGGML_NATIVE=OFF",
                         "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DCMAKE_BUILD_TYPE=Release"],
                        ["cmake", "--build", str(build), "--target", "llama-quantize", "llama-imatrix", "-j",
                         str(os.cpu_count() or 2)]):
            done = run(command, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"building llama-quantize failed: {(done.stderr or done.stdout)[-800:]}")
        binary = _find_binary(build, "llama-quantize")
        if binary is None:
            raise RuntimeError("building llama-quantize produced no binary")
    return convert.resolve(), binary.resolve()


def _run_tool(run: Any, command: list[str], what: str, env: dict[str, str] | None = None) -> None:
    print(f"  {what}...", flush=True)
    done = run(command, capture_output=True, text=True, env=env)
    if done.returncode != 0:
        raise RuntimeError(f"{what} failed: {(done.stderr or done.stdout or '')[-1200:]}")


def imatrix_binary(quantize: Path) -> Path | None:
    """``llama-imatrix`` next to ``llama-quantize`` (both ship in the llama.cpp release and the source build)."""
    found = Path(quantize).with_name("llama-imatrix")
    return found if found.is_file() else None


def make_ggufs(st_dir: Path, scratch: Path, stem: str, quants: list[str], *, vision: bool,
               tools: tuple[Path, Path], on_file: Any, run: Any = None, on_base: Any = None,
               calibration: Path | None = None, imatrix_chunks: int = IMATRIX_CHUNKS, stop_at: float = 0,
               clock: Any = time.time) -> dict[str, Any]:
    """safetensors -> one F16 GGUF (+ BF16, + the F16 vision projector) -> an importance matrix -> every quant.

    ``on_file(path, quant)`` gets each finished file right away (to upload it and free the disk). With a
    ``calibration`` text and ``llama-imatrix``, the importance matrix is computed (on Q8_0, faster on a CPU) and
    used for every quant below Q8_0 - the IQ1/IQ2/IQ3_XXS/Q2_K_S ones need it and are skipped without it. Past
    ``stop_at`` (unix time) the remaining quants are skipped, so the uploads still finish before the session ends.
    The first quant in ``quants`` stays first in ``files``; the F16 intermediate is deleted unless ``f16`` was asked
    for."""
    import subprocess

    run = run or subprocess.run
    convert, quantize = tools
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "LD_LIBRARY_PATH": f"{Path(quantize).parent}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
    wanted = [q.upper() for q in dict.fromkeys(q.upper() for q in quants)]
    primary = wanted[0] if wanted else ""
    base = scratch / f"{stem}.F16.gguf"
    _run_tool(run, [sys.executable, str(convert), str(st_dir), "--outfile", str(base), "--outtype", "f16"],
              "safetensors -> F16 GGUF")
    if not base.is_file():
        raise RuntimeError("the converter wrote no GGUF")
    result: dict[str, Any] = {"files": [], "mmproj": None, "skipped": [], "imatrix": None}
    bf16 = scratch / f"{stem}.BF16.gguf"
    if "BF16" in wanted:  # straight from the 16-bit safetensors, while they are still here (exact, not via F16)
        _run_tool(run, [sys.executable, str(convert), str(st_dir), "--outfile", str(bf16), "--outtype", "bf16"],
                  "safetensors -> BF16 GGUF")
    if vision:
        mmproj = scratch / f"mmproj-{stem}.F16.gguf"
        before = set(scratch.glob("*.gguf"))
        _run_tool(run, [sys.executable, str(convert), str(st_dir), "--mmproj", "--outfile", str(mmproj),
                        "--outtype", "f16"], "vision projector -> mmproj GGUF")
        if not mmproj.is_file():
            fresh = sorted(set(scratch.glob("*.gguf")) - before)
            if not fresh:
                raise RuntimeError("the converter wrote no vision projector")
            fresh[0].rename(mmproj)
        result["mmproj"] = mmproj.name
        on_file(mmproj, "MMPROJ")
    if on_base is not None:  # the safetensors are not read any more: everything below works from the F16 GGUF
        on_base()
    imatrix_tool = imatrix_binary(Path(quantize)) if calibration else None
    wants_imatrix = imatrix_tool is not None and any(q not in NO_IMATRIX for q in wanted)
    order = list(wanted)
    if wants_imatrix and "Q8_0" in order:  # Q8_0 first: the importance matrix is computed with it
        order.remove("Q8_0")
        order.insert(0, "Q8_0")
    imatrix: Path | None = None
    made: dict[str, dict[str, Any]] = {}

    def compute_imatrix() -> None:
        nonlocal imatrix
        model = scratch / f"{stem}.Q8_0.gguf" if "Q8_0" in made and (scratch / f"{stem}.Q8_0.gguf").is_file() else base
        target = scratch / f"{stem}.imatrix.gguf"
        seconds = 45 * 60.0
        if stop_at:
            seconds = min(seconds, (stop_at - clock()) / 3)
        if seconds < 120:
            print("  no time left for the importance matrix", flush=True)
            return
        command = [str(imatrix_tool), "-m", str(model), "-f", str(calibration), "-o", str(target), "-c", "512",
                   "--chunks", str(int(imatrix_chunks)), "--no-ppl", "-ofreq", "8", "-t", str(os.cpu_count() or 4)]
        print(f"  importance matrix ({imatrix_chunks} x 512 tokens of the training data, on {model.name.split('.')[-2]})"
              "...", flush=True)
        partial = False
        try:
            done = run(command, capture_output=True, text=True, env=env, timeout=seconds)
            if done.returncode != 0:
                print(f"  llama-imatrix failed: {(done.stderr or done.stdout or '')[-600:]}", flush=True)
        except subprocess.TimeoutExpired:  # it saves every 8 chunks: a partial matrix still helps a lot
            partial = True
            print(f"  llama-imatrix stopped after {seconds / 60:.0f} min - using what it saved", flush=True)
        if target.is_file() and target.stat().st_size > 0:
            imatrix = target
            result["imatrix"] = {"file": target.name, "chunks": int(imatrix_chunks), "partial": partial,
                                 "computed_on": model.name.split(".")[-2], "text": "the training data"}
            on_file(target, "IMATRIX")  # published too: others can make more quants with it

    for quant in order:
        if stop_at and clock() > stop_at:
            result["skipped"].append({"quant": quant, "why": "not enough time left in this session"})
            continue
        if wants_imatrix and imatrix is None and result["imatrix"] is None and quant not in NO_IMATRIX:
            compute_imatrix()
            wants_imatrix = imatrix is not None
        if quant in IMATRIX_REQUIRED and imatrix is None:
            result["skipped"].append({"quant": quant, "why": "needs an importance matrix (llama-imatrix was not "
                                                              "available or had no time)"})
            continue
        target = {"F16": base, "BF16": bf16}.get(quant, scratch / f"{stem}.{quant}.gguf")
        use_imatrix = imatrix is not None and quant not in NO_IMATRIX
        if quant not in ("F16", "BF16"):
            command = [str(quantize)] + (["--imatrix", str(imatrix)] if use_imatrix else []) + \
                      [str(base), str(target), quant]
            _run_tool(run, command, f"F16 -> {quant}" + (" (imatrix)" if use_imatrix else ""), env=env)
        if not target.is_file():
            raise RuntimeError(f"llama-quantize wrote no {quant} file")
        made[quant] = {"quant": quant, "file": target.name, "gb": round(target.stat().st_size / 1e9, 2),
                       "imatrix": use_imatrix}
        if quant != "F16":  # the F16 is the source of every quant: handed over last
            on_file(target, quant)
    if "F16" in made:
        on_file(base, "F16")
    result["files"] = [made[q] for q in [primary] + [q for q in order if q != primary] if q in made]
    if base.exists() and "F16" not in made:
        base.unlink()
    return result


def pick_gguf(found: list[str], quant: str) -> str | None:
    """The file of ``quant`` among an export's GGUFs - never the F16/BF16 intermediate unless that was asked for.

    (Picking ``sorted(...)[0]`` took ``*.F16.gguf`` before ``*.Q4_K_M.gguf``: an ~8 GB file named Q4_K_M.)"""
    quant = quant.upper()
    files = [f for f in found if "mmproj" not in Path(f).name.lower()]
    named = [f for f in files if re.search(rf"[.\-_]{re.escape(quant)}(?:[.\-_]|$)", Path(f).stem.upper())]
    if named:
        return named[0]
    if quant in ("F16", "BF16"):
        return None
    known = "|".join(re.escape(q) for q in (*QUANT_USE, "BF16", "F32"))
    rest = [f for f in files if not re.search(rf"(?:^|[.\-_])(?:{known})(?:[.\-_]|$)", Path(f).stem.upper())]
    return rest[0] if len(rest) == 1 else None


def connect_hub(args: argparse.Namespace, preset: dict[str, Any]) -> dict[str, Any] | None:
    """The Hugging Face session for ``--hub-repo``: the API, the GGUF repo and the safetensors repo."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("--hub-repo needs HF_TOKEN (Kaggle: Add-ons > Secrets > HF_TOKEN, a 'write' token)")
        return None
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        gguf = resolve_repo(args.hub_repo, preset, api.whoami)
    except Exception as exc:  # no upload, but the model still gets saved
        print(f"Hugging Face login failed ({type(exc).__name__}: {exc}) - the files stay in the output folder")
        return None
    return {"api": api, "token": token, "gguf": gguf, "safetensors": safetensors_repo(gguf),
            "private": bool(args.hub_private), "created": set()}


def upload_hub_file(hub: dict[str, Any], repo: str, path: Path, message: str) -> None:
    """Upload one file (a quant) to ``repo``, creating the repo the first time."""
    api = hub["api"]
    if repo not in hub["created"]:
        api.create_repo(repo, repo_type="model", private=hub["private"], exist_ok=True)
        hub["created"].add(repo)
    api.upload_file(path_or_fileobj=str(path), path_in_repo=Path(path).name, repo_id=repo, repo_type="model",
                    commit_message=message)


class Uploader:
    """Uploads in the background, in order - the safetensors first, then each GGUF quant - while llama.cpp goes on
    converting and quantising, so neither waits for the other. ``background=False`` runs every job right away."""

    def __init__(self, background: bool = True):
        import queue

        self.background = background
        self.jobs: Any = queue.Queue()
        self.thread: Any = None

    def submit(self, job: Any, *job_args: Any) -> None:
        if not self.background:
            self._run(job, job_args)
            return
        if self.thread is None:
            import threading

            self.thread = threading.Thread(target=self._loop, name="hf-upload", daemon=True)
            self.thread.start()
        self.jobs.put((job, job_args))

    @staticmethod
    def _run(job: Any, job_args: tuple[Any, ...]) -> None:
        try:
            job(*job_args)
        except Exception as exc:  # noqa: BLE001 - every job reports its own problems; never kill the thread
            print(f"  background job failed ({type(exc).__name__}: {exc})", flush=True)

    def _loop(self) -> None:
        while True:
            job, job_args = self.jobs.get()
            try:
                self._run(job, job_args)
            finally:
                self.jobs.task_done()

    def wait(self) -> None:
        if self.thread is not None:
            self.jobs.join()


def scratch_root() -> Path | None:
    """Where throwaway files go: ``$PYMC_SCRATCH``, else ``/kaggle/tmp`` on Kaggle (~60 GB, not saved, not counted in
    the 20 GB of ``/kaggle/working``, which keeps only what is needed again). ``None`` elsewhere."""
    chosen = os.environ.get("PYMC_SCRATCH")
    if chosen:
        return Path(chosen)
    if not Path("/kaggle/working").is_dir():
        return None
    try:
        Path("/kaggle/tmp").mkdir(parents=True, exist_ok=True)  # Kaggle doesn't create it
    except OSError:  # not writable: not a real Kaggle session
        return None
    return Path("/kaggle/tmp")


def scratch_for(out: Path, what: str) -> Path:
    """A throwaway folder for one output folder (``out`` itself when there is no scratch disk)."""
    root = scratch_root()
    if root is None:
        return Path(out)
    import hashlib

    where = Path(out).resolve()
    return root / what / f"{where.name}-{hashlib.sha256(str(where).encode('utf-8')).hexdigest()[:8]}"


def lock_file(path: Path | str) -> str:
    """The lock for ``path``: on the scratch disk, so no empty ``.lock`` files pile up next to kept files."""
    root = scratch_root()
    if root is None:
        return str(path) + ".lock"
    import hashlib

    return str(root / "locks" / f"{Path(path).name}-{hashlib.sha256(str(path).encode('utf-8')).hexdigest()[:8]}.lock")


def gguf_scratch(args: argparse.Namespace, out: Path) -> Path:
    """Where the F16 GGUF and the quants are made: the scratch disk (``/kaggle/tmp``, else ``/tmp``), never
    Kaggle's 20 GB output folder."""
    if getattr(args, "gguf_scratch", None):
        return Path(args.gguf_scratch)
    root = scratch_root()
    if root is not None:
        return root / "gguf" / Path(out).name
    import tempfile

    return Path(tempfile.gettempdir()) / "pymc-gguf" / Path(out).name


def tools_dir(args: argparse.Namespace, out: Path) -> Path:
    """Where llama.cpp is downloaded (``--tools-dir``: kept and reused, e.g. ``/kaggle/working/tools``)."""
    return Path(args.tools_dir) if getattr(args, "tools_dir", None) else gguf_scratch(args, out).parent


def export_ggufs(args: argparse.Namespace, preset: dict[str, Any], model: Any, tokenizer: Any, out: Path,
                 st_dir: Path | None, quants: list[str], hub: dict[str, Any] | None = None,
                 tools: Any = None, run: Any = None, uploader: Uploader | None = None,
                 on_base: Any = None) -> dict[str, Any] | None:
    """Every GGUF in ``quants``; the first stays in ``out`` (Modelfile, test, import), the rest are uploaded to
    the GGUF repo one by one and deleted (or kept in ``out`` when there is no upload or it fails).

    Made from the uploaded safetensors with llama.cpp when they exist; Unsloth's ``save_pretrained_gguf`` is the
    fallback. Returns ``{"files": [...], "mmproj": name, "route": ...}`` or ``None`` when nothing was made."""
    out = Path(out)
    primary = quants[0].upper()
    stem = preset.get("file") or f"pymc-minecraft-{args.preset}"
    scratch = gguf_scratch(args, out)
    uploaded: dict[str, bool] = {}

    def keep(path: Path) -> None:
        if path.parent.resolve() != out.resolve():
            shutil.move(str(path), str(out / path.name))

    def upload_quant(path: Path, quant: str) -> None:
        try:
            what = "importance matrix" if quant == "IMATRIX" else f"{quant} GGUF"
            upload_hub_file(hub, hub["gguf"], path, f"{display_name(args.preset, preset)} {what}")
            print(f"  uploaded {path.name} ({path.stat().st_size / 1e9:.2f} GB) -> {hub['gguf']}", flush=True)
            uploaded[quant] = True
            path.unlink()
        except Exception as exc:  # keep the file; the folder upload at the end tries again
            print(f"  upload of {path.name} failed ({type(exc).__name__}: {exc}) - kept in {out}", flush=True)
            keep(path)

    def on_file(path: Path, quant: str) -> None:
        if hub and quant not in (primary, "MMPROJ"):
            if uploader is not None:  # uploaded in the background while the next quant is being made
                uploader.submit(upload_quant, path, quant)
            else:
                upload_quant(path, quant)
            return
        keep(path)

    result = None
    if st_dir is not None and Path(st_dir).is_dir():
        try:
            print(f"making {len(quants)} GGUFs ({', '.join(q.upper() for q in quants)}) from the safetensors (llama.cpp)")
            tools = tools or llama_cpp_tools(tools_dir(args, out))
            calibration = calibration_text(getattr(args, "data_dir", None), scratch / "imatrix-text.txt")
            deadline = float(getattr(args, "deadline", 0) or 0)
            result = make_ggufs(Path(st_dir), scratch, stem, quants, vision=bool(preset.get("vision")),
                                tools=tools, on_file=on_file, run=run, on_base=on_base, calibration=calibration,
                                imatrix_chunks=int(getattr(args, "imatrix_chunks", IMATRIX_CHUNKS) or 0) or IMATRIX_CHUNKS,
                                stop_at=deadline - 15 * 60 if deadline else 0)
            result["route"] = "llama.cpp"
            tag = re.search(r"llama\.cpp-(b\d+)", str(tools[1]))
            if tag:
                result["llama_cpp"] = tag.group(1)
            for item in result["skipped"]:
                print(f"  skipped {item['quant']}: {item['why']}", flush=True)
        except Exception as exc:
            print(f"llama.cpp GGUF route failed ({type(exc).__name__}: {exc}) - using Unsloth's exporter")
            result = None
    if result is None and model is None:  # the CPU-only --export-only process has no model for Unsloth
        print("no GGUF: llama.cpp failed and Unsloth's exporter needs the trained model (not loaded in this process)")
        if uploader is not None:
            uploader.wait()
        return None
    if result is None:
        slow = [q for q in quants[1:] if q.lower() not in GGUF_QUANTS]
        quants = [quants[0]] + [q for q in quants[1:] if q.lower() in GGUF_QUANTS]
        print(f"exporting GGUF ({', '.join(quants)}) with Unsloth - 10-20 minutes per quant"
              + (f"; {len(slow)} more quants need llama.cpp and are skipped" if slow else ""))
        model.save_pretrained_gguf(str(out / "gguf"), tokenizer, quantization_method=list(quants))
        found = sorted(set(glob.glob(str(out / "gguf*" / "*.gguf")) + glob.glob(str(out / "*.gguf"))
                           + glob.glob("*.gguf")))
        projectors = [f for f in found if "mmproj" in Path(f).name.lower()]
        result = {"files": [], "mmproj": None, "route": "unsloth", "imatrix": None,
                  "skipped": [{"quant": q.upper(), "why": "llama.cpp was not available (Unsloth's exporter makes "
                                                         "only the standard quants)"} for q in slow]}
        if projectors and preset.get("vision"):
            mmproj = out / f"mmproj-{stem}.F16.gguf"
            shutil.move(projectors[0], mmproj)
            result["mmproj"] = mmproj.name
        for quant in [q.upper() for q in quants]:
            picked = pick_gguf(found, quant)
            if not picked:
                print(f"  Unsloth made no {quant} file")
                continue
            found.remove(picked)
            target = out / f"{stem}.{quant}.gguf"
            shutil.move(picked, target)
            result["files"].append({"quant": quant, "file": target.name, "gb": round(target.stat().st_size / 1e9, 2)})
            on_file(target, quant)
        unsloth_modelfile = next(iter(glob.glob(str(out / "gguf*" / "Modelfile"))), None)
        if unsloth_modelfile:
            result["template"] = template_from_modelfile(Path(unsloth_modelfile).read_text())
        for work in out.glob("gguf*"):  # Unsloth's merge + F16 intermediate
            if work.is_dir():
                shutil.rmtree(work, ignore_errors=True)
    if uploader is not None:
        uploader.wait()  # every quant is up (or kept) before the scratch folder goes
    for item in result["files"]:
        item["uploaded"] = bool(uploaded.get(item["quant"]))
    if not any(item["quant"] == primary for item in result["files"]):
        return None
    shutil.rmtree(scratch, ignore_errors=True)  # the F16 intermediate and anything half-made
    return result


def _ram_gb(gb: Any, extra: float) -> str:
    try:
        return f"~{float(gb) + extra:.1f} GB"
    except (TypeError, ValueError):
        return "?"


def gguf_quant_guide(report: dict[str, Any], repo: str, quant: str) -> str:
    """The GGUF card's "which file" advice and the table of every quant (biggest first), with what was skipped."""
    ggufs = [g for g in report.get("ggufs") or [] if g.get("file")]
    if not ggufs:
        return ""
    mmproj_gb = 0.0
    if report.get("mmproj"):
        mmproj_gb = 0.7
    have = {g["quant"] for g in ggufs}
    pick = [q for q in ("Q8_0", "Q6_K", "Q5_K_M", quant, "IQ4_XS", "IQ3_M", "Q3_K_M", "IQ2_M") if q in have]
    advice = {"Q8_0": "you have plenty of RAM / VRAM and want the best answers",
              "Q6_K": "almost Q8_0 quality, a quarter smaller",
              "Q5_K_M": "high quality with room to spare",
              "IQ4_XS": "you want a 4-bit file a bit smaller than the default",
              "IQ3_M": "low RAM (old laptops, phones)",
              "Q3_K_M": "low RAM and an app that has no IQ quants",
              "IQ2_M": "it has to fit somewhere tiny - expect clearly weaker answers"}
    advice[quant] = "**start here** - the default, small and good (what `llm import` and Ollama use)"
    lines = ["\n## Which file should I download?\n",
             "Pick the biggest file that fits in your free RAM (CPU) or VRAM (GPU) with about 1 GB to spare for the "
             "context" + (" and the vision projector" if mmproj_gb else "") + ". Bigger = closer to the original, "
             "smaller = faster and lighter.\n"]
    lines += [f"- `{q}` - {advice[q]}" for q in dict.fromkeys(pick)]
    lines.append("\nK-quants (`Q5_K_M`, `Q4_K_M`...) run fast everywhere. I-quants (`IQ4_XS`, `IQ3_M`...) are better for "
                 "their size and fast on NVIDIA / AMD GPUs, but slower than K-quants on CPUs and Apple Metal. `F16` / "
                 "`BF16` are the unquantised weights, for making your own quants.")
    imatrix = report.get("imatrix") or {}
    rows = sorted(ggufs, key=lambda g: -(g.get("gb") or 0))
    table = [f"\n## All quants ({len(rows)} files)\n",
             "| | quant | size | RAM to run | imatrix | use it when | download |",
             "|---|---|---|---|---|---|---|"]
    for g in rows:
        mark = "⭐" if g["quant"] == quant else ("👍" if g["quant"] in RECOMMENDED_QUANTS else "")
        table.append(f"| {mark} | `{g['quant']}` | {g.get('gb', '?')} GB | {_ram_gb(g.get('gb'), 1.0 + mmproj_gb)} | "
                     f"{'yes' if g.get('imatrix') else '-'} | {QUANT_USE.get(g['quant'], '')} | "
                     f"[{g['file']}](https://huggingface.co/{repo}/resolve/main/{g['file']}) |")
    notes = ["", "⭐ default · 👍 recommended. RAM to run = file + ~1 GB for a 4096-token context"
             + (" + the vision projector" if mmproj_gb else "") + "; a longer context needs more."]
    if imatrix:
        notes.append(f"\n**imatrix:** the quants marked *yes* were made with an importance matrix computed on "
                     f"{imatrix.get('chunks', '?')} x 512 tokens of this model's own training data"
                     + (" (stopped early to save time)" if imatrix.get("partial") else "")
                     + f", so they keep what Minecraftism needs most. The matrix is in this repo "
                     f"(`{imatrix.get('file', 'imatrix.gguf')}`) if you want to make more quants.")
    skipped = report.get("gguf_skipped") or []
    if skipped:
        notes.append("\n**Not made this time:** " + "; ".join(f"`{x['quant']}` ({x['why']})" for x in skipped) + ".")
    notes.append("\n**Not made on purpose:** " + "; ".join(f"`{k}` - {v}" for k, v in NOT_MADE.items()) + ".")
    if report.get("llama_cpp"):
        notes.append(f"\nMade with llama.cpp {report['llama_cpp']} from the merged 16-bit weights.")
    return "\n".join(lines + table + notes) + "\n"


def _prompt_and_settings(preset: dict[str, Any], reasoning: bool, vision: bool, gguf: bool) -> str:
    params = dict(preset.get("run_parameters") or RUN_PARAMETERS)
    names = {"temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p",
             "repeat_penalty": "repeat penalty", "num_ctx": "context (tokens)", "num_predict": "max new tokens"}
    rows = "".join(f"| {names[k]} | {params[k]} |\n" for k in names if k in params)
    think = ("\nIt writes its thinking first, inside `<think>...</think>`, then the answer: hide or strip that "
             "part before showing the reply.\n") if reasoning else ""
    picture = ("\nPictures go in the user turn (Ollama, LM Studio and llama.cpp do this for you with the `mmproj` "
               "file).\n") if vision else ""
    where = ("Ollama, LM Studio and llama.cpp read the chat template from the GGUF, so you only need this for raw "
             "completion APIs." if gguf else "`tokenizer.apply_chat_template` builds it for you.")
    return f"""
## Prompt format

Qwen3's ChatML. {where}

```text
<|im_start|>system
{{system prompt}}<|im_end|>
<|im_start|>user
{{your message}}<|im_end|>
<|im_start|>assistant
```
{think}{picture}
## Recommended settings

| setting | value |
|---|---|
{rows}
{"The `Modelfile` in this repo sets these for Ollama; set them yourself in other apps." if gguf else "Use them for `generate` / vLLM sampling."}
"""


def model_card(preset_name: str, preset: dict[str, Any], report: dict[str, Any], repo: str, kind: str = "gguf") -> str:
    """README.md for a Hugging Face repo: the GGUF edition (``kind="gguf"``, also written next to the GGUF) or the
    merged full-precision safetensors edition (``kind="safetensors"``, written into ``safetensors/``)."""
    safetensors = kind == "safetensors"
    name = display_name(preset_name, preset) + (" (safetensors)" if safetensors else "")
    vision, reasoning = bool(preset.get("vision")), bool(preset.get("reasoning"))
    base = preset.get("base_model") or preset["model"]
    quant = str(report.get("quant") or "Q4_K_M").upper()
    tags = ["minecraft", "gguf", "pymc-bot", "minecraftism", "qwen3", "unsloth", "game-agent"]
    tags += ["vision", "image-text-to-text"] if vision else []
    tags += ["reasoning", "thinking"] if reasoning else []
    tags += ["uncensored", "abliterated"] if preset.get("uncensored") else []
    files = [f"- `{report.get('gguf', output_name(preset_name, preset, quant))}` - the model ({quant})"]
    if report.get("ggufs"):
        files = [f"- `{report.get('gguf', output_name(preset_name, preset, quant))}` - the default ({quant})",
                 f"- {len(report['ggufs']) - 1} more `*.gguf` quants - every one is in the table above"]
        if (report.get("imatrix") or {}).get("file"):
            files.append(f"- `{report['imatrix']['file']}` - the importance matrix the quants were made with")
    if report.get("mmproj"):
        files.append(f"- `{report['mmproj']}` - the vision projector (needed for pictures)")
    files += ["- `Modelfile` - for `ollama create`", "- `pymc_model.json` - training report PyMC Bot reads on import"]
    if report.get("essentials"):
        files.append("- " + ", ".join(f"`{name}`" for name in report["essentials"][:6])
                     + (" ..." if len(report["essentials"]) > 6 else "") + " - tokenizer, config and chat template")
    gguf_repo = str(report.get("hub_repo") or repo)
    st_repo = str(report.get("hub_repo_safetensors") or "")
    if safetensors:
        tags = ["safetensors" if tag == "gguf" else tag for tag in tags]
        gguf_repo = str(report.get("hub_repo") or f"<your-hf-name>/{preset.get('hub') or 'pymc-minecraft-GGUF'}")
        files = ["- `model*.safetensors` - the merged weights (LoRA folded into the base, 16-bit)",
                 "- `config.json`, `generation_config.json` - model config",
                 "- tokenizer files + `chat_template.jinja` - the chat format it was trained with"]
        if vision:
            files.append("- `preprocessor_config.json` / processor files - how pictures are prepared")
    evaluation = report.get("eval") or {}
    eval_lines = ""
    if evaluation:
        def pct(key: str) -> str:
            value = evaluation.get(key)
            return "-" if value is None else f"{value}%"

        eval_lines = ("\n## Evaluation (held-out rows)\n\n| valid JSON | right action | right answers | rows |\n|---|---|---|---|\n"
                      f"| {pct('json_valid_pct')} | {pct('action_match_pct')} | {pct('answer_keywords_pct')} | "
                      f"{evaluation.get('rows', '-')} |\n")
    agentic = report.get("agentic") or {}
    if agentic.get("tuned"):
        tuned, base = agentic["tuned"], agentic.get("base") or {}
        cell = lambda d, k: "-" if d.get(k) is None else f"{d[k]}%"  # noqa: E731
        base_name = str(report.get("base_model") or "base").split("/")[-1]
        eval_lines += ("\n## Agentic score (tool calling)\n\nHeld-out multi-turn tool chats it never trained on "
                       f"({tuned.get('asked', '?')} prompts: the right function with exactly the right arguments, and no "
                       "call when no tool fits), base model measured on the same prompts:\n\n"
                       "| model | score | right tool | exact calls | no call when none fits |\n|---|---|---|---|---|\n"
                       + (f"| {base_name} | {cell(base, 'score')} | {cell(base, 'right_tool')} | {cell(base, 'exact_calls')} | "
                          f"{cell(base, 'no_call_when_none_fits')} |\n" if base else "")
                       + f"| **{display_name(preset_name, preset)}** | **{cell(tuned, 'score')}** | {cell(tuned, 'right_tool')} | "
                         f"{cell(tuned, 'exact_calls')} | {cell(tuned, 'no_call_when_none_fits')} |\n")
    vision_text = ("\nIt also **sees**: it was trained on first-person screenshots recorded with the PyMC Bot Fabric mod, "
                   "paired with what the player did next and grounded descriptions of each picture. In PyMC Bot turn on "
                   "*let the model see (vision)* in the panel's AI settings and the bot sends its rendered view every few turns.\n") if vision else ""
    think_text_ = ("\nIt **thinks first**: before every answer it writes a short `<think>` about its health, the mobs "
                   "around it (with facts from the mob database), chat and its goal, then the JSON action. Smarter "
                   "choices, slower turns.\n") if reasoning else ""
    uncensored_text = ("\nIt is **uncensored**: trained on an abliterated base (the refusal direction removed from the "
                       "weights) and on chat data with refusals filtered out, so it answers directly instead of "
                       "lecturing. You are responsible for how you use it.\n") if preset.get("uncensored") else ""
    tests = report.get("tests") or {}
    test_lines = (f"\nTest suite after training: {tests.get('refusals', '?')} refusal(s) in {tests.get('prompts', '?')} "
                  f"prompts, identity {'learned' if tests.get('identity_ok') else 'not learned'}, bot JSON "
                  f"{tests.get('bot_json_ok', '?')}.\n") if tests else ""
    system = str(report.get("system_prompt") or "").strip()
    system_text = ("\n## Default system prompt\n\nBuilt into the `Modelfile` (Ollama uses it whenever the app sends no "
                   "system prompt of its own; PyMC Bot sends its own for game turns):\n\n```text\n" + system + "\n```\n"
                   ) if system else ""
    if system and safetensors:
        system_text = ("\n## System prompt\n\nThe personality it was trained with. Send it as the `system` message "
                       "(the GGUF edition has it built into its `Modelfile`):\n\n```text\n" + system + "\n```\n")
    sources = report.get("data") or []
    data_cell = (", ".join(f"[{d['name']}]({d['url']})" for d in sources) if sources else
                 f"PyMC Bot rules + mob database, Minecraft Wiki text, recorded play{' and screenshots' if vision else ''}")
    distill = report.get("distill") or {}
    distill_text = ""
    if distill:
        lines = []
        off, on = distill.get("offpolicy") or {}, distill.get("onpolicy") or {}
        if off:
            lines.append(f"- **Off-policy:** the teacher wrote {off.get('used', 0)} of the training answers "
                         f"({off.get('general', 0)} everyday chat, {off.get('mcqa', 0)} Minecraft facts rewritten "
                         "around the exact answer"
                         + (", with its own thinking traces" if reasoning else "") + "). Game turns kept their "
                         "rule-checked JSON.")
        if on:
            lines.append(f"- **On-policy:** {on.get('steps', 0)} steps ({on.get('replies', 0)} replies written by the "
                         f"student and graded token by token by the teacher, reverse KL "
                         f"{on.get('kl_start', '?')} -> {on.get('kl_end', '?')}).")
        teacher = distill.get("teacher", "")
        grader = distill.get("grader") or teacher
        who = (f"[{teacher}](https://huggingface.co/{teacher}) as the teacher (same tokenizer, Apache-2.0)"
               if grader == teacher else
               f"[{teacher}](https://huggingface.co/{teacher}) writing the answers and "
               f"[{grader}](https://huggingface.co/{grader}) grading the student's own replies (both Qwen3: same "
               "tokenizer, Apache-2.0)")
        distill_text = (f"\n## Distillation\n\nTrained with Qwen3's *strong-to-weak* distillation recipe, with "
                        f"{who}:\n\n" + "\n".join(lines) + "\n")
    data_text = ""
    if sources:
        data_text = ("\n## Training data\n\n| source | licence | commercial use | rows |\n|---|---|---|---|\n"
                     + "".join(f"| [{d['name']}]({d['url']}) | {d['license']} | {'yes' if d.get('commercial') else 'no'} "
                               f"| {d.get('rows', '-')} |\n" for d in sources))
    if report.get("commercial"):
        data_note = ("every training source allows commercial use (table above; the CC BY 4.0 data is NVIDIA's "
                     "Nemotron post-training set, credited here as its licence asks)")
    elif any(d.get("name") == "minecraft.wiki" for d in sources) or not sources:
        data_note = "the training data includes Minecraft Wiki text (CC BY-NC-SA 3.0, https://minecraft.wiki)"
    else:
        data_note = "some training data is not licensed for commercial use (see the table above)"
    llama = (f"llama-server -m {report.get('gguf', 'model.gguf')} --mmproj {report.get('mmproj', 'mmproj.gguf')}"
             if vision else f"llama-server -m {report.get('gguf', 'model.gguf')}")
    size = f"~{preset.get('gguf_gb')} GB file, ~{preset.get('run_gb')} GB to run"
    if report.get("gguf_gb"):
        size = f"{report['gguf_gb']} GB ({quant}), ~{preset.get('run_gb')} GB to run"
    if report.get("params_b"):
        size += f" - {report['params_b']}B parameters"
    formats = ""
    if st_repo and not safetensors:
        formats = (f"\nAlso available as full-precision **safetensors** (merged 16-bit, for transformers, vLLM or "
                   f"further fine-tuning): [{st_repo}](https://huggingface.co/{st_repo})\n")
    library = "gguf"
    others = [g["quant"] for g in report.get("ggufs") or [] if g["quant"] != quant]
    other_quants = (f"\nOther sizes: put any quant from [the table](#which-file-should-i-download) after the colon, e.g. "
                    f"`ollama run hf.co/{repo}:{others[0]}`" + (f" or `:{others[1]}`" if len(others) > 1 else "")
                    + ".\n" if others else "")
    use = f"""With PyMC Bot (switches the bot to this model with the short Minecraft prompt):

```bash
python -m pymc_bot llm import --hf {repo}
```

With Ollama:

```bash
ollama run hf.co/{repo}:{quant}
```
{other_quants}
With llama.cpp (`-hf` downloads the quant{" and the vision projector" if vision else ""} by itself):

```bash
llama-server -hf {repo}:{quant}
# or with a file you downloaded:
{llama}
```

One file only (LM Studio and Jan find this repo by its name too):

```bash
pip install -U huggingface_hub
hf download {repo} {report.get('gguf', output_name(preset_name, preset, quant))} --local-dir .
```"""
    if safetensors:
        library = "transformers"
        size = f"~{report.get('safetensors_gb') or '8'} GB of 16-bit weights (bf16 / fp16), ~10 GB of GPU memory to run"
        if report.get("params_b"):
            size += f" - {report['params_b']}B parameters"
        formats = (f"\nThis repo has the **full-precision safetensors** (the LoRA merged into the base, 16-bit) for "
                   f"transformers, vLLM, further fine-tuning or your own quantisation. For Ollama, llama.cpp, LM Studio "
                   f"and PyMC Bot use the **GGUF** edition: [{gguf_repo}](https://huggingface.co/{gguf_repo})\n")
        if vision:
            code = f"""from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained("{repo}")
model = AutoModelForImageTextToText.from_pretrained("{repo}", torch_dtype="auto", device_map="auto")
messages = [{{"role": "user", "content": [{{"type": "image", "url": "screenshot.png"}},
                                         {{"type": "text", "text": "What do you see in this Minecraft screenshot?"}}]}}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                       return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=400)
print(processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))"""
        else:
            code = f"""from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForCausalLM.from_pretrained("{repo}", torch_dtype="auto", device_map="auto")
messages = [{{"role": "user", "content": "How do I find diamonds fast?"}}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
output = model.generate(inputs, max_new_tokens=400)
reply = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
print(reply)"""
        use = f"""With transformers:

```python
{code}
```

With vLLM (an OpenAI-compatible server):

```bash
vllm serve {repo}
```

With PyMC Bot / Ollama, use the GGUF edition:

```bash
python -m pymc_bot llm import --hf {gguf_repo}
```"""
    card_base, relation = base, ""
    if st_repo and not safetensors:  # the model tree: base -> fine-tune (safetensors) -> quantized (this repo)
        card_base, relation = st_repo, "\nbase_model_relation: quantized"
    elif safetensors:
        relation = "\nbase_model_relation: finetune"
    if not safetensors:
        tags += ["llama.cpp", "ollama"] + (["imatrix"] if report.get("imatrix") else [])
    guide = "" if safetensors else gguf_quant_guide(report, repo, quant)
    links = [f"[GGUF: every quant from F16 to IQ1_S](https://huggingface.co/{gguf_repo})"] if safetensors else \
        (["[which file?](#which-file-should-i-download)"] if guide else [])
    if st_repo and not safetensors:
        links.append(f"[safetensors](https://huggingface.co/{st_repo})")
    links += [f"[base model](https://huggingface.co/{base})", f"[PyMC Bot]({PROJECT_URL})"]
    return f"""---
license: other
license_name: polyform-noncommercial-1.0.0
license_link: https://polyformproject.org/licenses/noncommercial/1.0.0/
base_model: {card_base}{relation}{"" if safetensors else chr(10) + "quantized_by: codero-sus"}
tags:
{chr(10).join('- ' + t for t in tags)}
language:
- en
pipeline_tag: {'image-text-to-text' if vision else 'text-generation'}
library_name: {library}
---

# {name}

**Minecraftism** is a small model that knows Minecraft and plays it: the brain of
[PyMC Bot]({PROJECT_URL}), a Minecraft player bot. Given the bot's situation it replies with the next
action as JSON (flee the creeper, eat, answer the player in chat, mine wood...), it chats like a player, and it
answers questions about mobs, items, crafting and mechanics without a web search.
It is also an everyday assistant: questions, writing, code, advice or just a chat.
{think_text_}{vision_text}{uncensored_text}{formats}
{' · '.join(links)}

## Quick start

{use}
{guide}{_prompt_and_settings(preset, reasoning, vision, not safetensors)}
## About this edition

| | |
|---|---|
| Edition | {name} (`{preset_name}`) |
| Base model | [{base}](https://huggingface.co/{base}) (Apache-2.0) |
| Fine-tune | LoRA r={report.get('rank', 32)} with Unsloth on Kaggle (T4), {report.get('train_rows', '?')} rows, loss {report.get('loss', '?')} |
| Data | {data_cell} |
| Size | {size} |
{eval_lines}{test_lines}{distill_text}{data_text}{system_text}
## Files

{chr(10).join(files)}

## Limitations

- It is a small (4B) model: it can be confidently wrong. Check important facts, and let PyMC Bot's rule checks
  guard the game actions.
- Its Minecraft knowledge comes mostly from Java Edition data; Bedrock differences and the newest updates may be
  missing.
- It was trained on examples of up to {report.get('max_seq') or 3072} tokens; very long chats lose the thread.
{"- It is uncensored: it does not refuse. You are responsible for how you use it." + chr(10) if preset.get("uncensored") else ""}
## License and credits

Required Notice: Copyright codero-sus ({AUTHOR_URL}). Minecraftism is part of PyMC Bot ({PROJECT_URL}) and is
licensed under PolyForm Noncommercial 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/). Anyone who
shares these weights, or quants or fine-tunes made from them, must pass on this notice and a link to the original.

- Quants, merges and fine-tunes by others are welcome for non-commercial use: keep the notice above and link here.
- Made with ❤️ by [codero-sus]({AUTHOR_URL}) - {' - '.join(WEBSITES)}
- **Personal, non-commercial use only.** The fine-tune is released under PolyForm Noncommercial 1.0.0, and
  {data_note}. The base model's own license (Apache-2.0) still applies to the base weights.
- Not an official Minecraft product. Not approved by or associated with Mojang or Microsoft.
"""


def write_cards(out: Path, preset_name: str, preset: dict[str, Any], report: dict[str, Any], repo: str) -> None:
    """pymc_model.json + README.md for the GGUF edition, and the safetensors edition's own README.md."""
    out = Path(out)
    (out / "pymc_model.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(model_card(preset_name, preset, report, repo), encoding="utf-8")
    folder = out / SAFETENSORS_DIR
    if report.get("safetensors") and folder.is_dir():
        st_repo = report.get("hub_repo_safetensors") or safetensors_repo(repo)
        (folder / "README.md").write_text(model_card(preset_name, preset, report, st_repo, kind="safetensors"),
                                          encoding="utf-8")


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
    ignore = ["checkpoints/*", "*.lock", "*.tmp", "gguf/*", "stream_images/*", f"{SAFETENSORS_DIR}/*"]
    ignore += [] if include_lora else ["lora/*"]
    api.upload_folder(repo_id=repo, folder_path=str(out), repo_type="model", ignore_patterns=ignore,
                      commit_message=f"{display_name(preset_name, PRESETS[preset_name])} from PyMC Bot")
    return repo


def publish_safetensors(folder: Path, repo: str, preset_name: str, token: str, private: bool = False) -> str:
    """Upload the merged safetensors edition (``<out>/safetensors``, with its own model card) to its own repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=str(folder), repo_type="model", ignore_patterns=["*.lock"],
                      commit_message=f"{display_name(preset_name, PRESETS[preset_name])} (safetensors) from PyMC Bot")
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


# --------------------------------------------------------------------------- speed: both T4s, deadline, export lock
def dist_info() -> tuple[int, int]:
    """``(rank, world size)`` when started by torchrun (both T4s on one edition, DDP); ``(0, 1)`` otherwise."""
    try:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))), int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 0, 1


_STARTED = time.time()


def _fresh(path: Path) -> bool:
    """A hand-over file written during this run (not one left over by an earlier launch)."""
    try:
        return path.is_file() and path.stat().st_mtime >= _STARTED - 5
    except OSError:
        return False


def _rows_file(out: Path) -> Path:
    run = os.environ.get("PYMC_RUN_ID") or os.environ.get("TORCHELASTIC_RUN_ID") or ""
    if run in ("", "none"):  # accelerate launch leaves the elastic run id at "none"
        run = os.environ.get("MASTER_PORT") or "run"
    run = re.sub(r"[^A-Za-z0-9_.-]", "_", run)
    return scratch_for(out, "handover") / f".ddp-rows-{run}.json"


def share_rows(out: Path, rows: list[dict[str, Any]] | None, eval_rows: list[dict[str, Any]] | None = None,
               data_dir: str = "") -> None:
    """DDP: rank 0 prepares the data once (streaming included) and hands it to the other GPU's process.
    ``rows=None`` tells it that rank 0 stopped before training (dry run, bad data, deadline)."""
    target = _rows_file(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps({"rows": rows, "eval_rows": eval_rows or [], "data_dir": data_dir}), encoding="utf-8")
    os.replace(temp, target)


def wait_for_rows(out: Path, timeout: float = 6 * 3600, poll: float = 5.0, on_job: Any = None) -> dict[str, Any] | None:
    """DDP, ranks > 0: wait for rank 0's data (``None`` = rank 0 is not training). ``on_job()`` runs once if rank 0
    asks this GPU to write its half of the teacher's answers first (distillation)."""
    target = _rows_file(out)
    end = time.time() + timeout
    served = False
    while not _fresh(target):
        if time.time() > end:
            raise TimeoutError(f"rank 0 never shared the training data ({target})")
        if on_job is not None and not served and _fresh(_distill_file(out)):
            served = True
            try:
                on_job()
            except Exception:  # noqa: BLE001 - rank 0 keeps the original answers for this share
                pass
            continue
        time.sleep(poll)
    shared = json.loads(target.read_text(encoding="utf-8"))
    return shared if shared.get("rows") is not None else None


def token_lengths(tokenizer: Any, texts: list[str], chunk: int = 1000) -> list[int]:
    """Tokens per training text (a rough 3.5 characters per token if the tokenizer cannot be called)."""
    lengths: list[int] = []
    try:
        for start in range(0, len(texts), chunk):
            ids = tokenizer(texts[start:start + chunk], add_special_tokens=False)["input_ids"]
            lengths += [len(item) for item in ids]
        return lengths
    except Exception:  # noqa: BLE001 - only the batch size depends on it
        return [max(1, round(len(text) / 3.5)) for text in texts]


def plan_batches(lengths: list[int], preset: dict[str, Any], world: int, max_seq: int, token_budget: int) -> dict[str, Any]:
    """Rows per GPU batch and accumulation steps for ``--token-budget``.

    The effective batch (rows per optimizer step = the preset's ``batch * accum``) never changes, so the model
    learns the same; it is just done in fewer, bigger forward passes (Unsloth trains padding-free, so a batch
    costs its real tokens). The batch is sized by the 95th-percentile row so a batch is about ``token_budget``
    tokens."""
    world = max(1, world)
    per_step = preset["batch"] * preset["accum"]
    if token_budget <= 0 or not lengths:
        return {"batch": preset["batch"], "accum": max(1, round(preset["accum"] / world)), "p95": None, "mean": None}
    capped = sorted(min(int(n), max_seq) for n in lengths)
    p95 = max(1, capped[min(len(capped) - 1, int(len(capped) * 0.95))])
    per_gpu = max(1, -(-per_step // world))
    batch = max(1, min(per_gpu, max(preset["batch"], token_budget // p95)))
    return {"batch": batch, "accum": max(1, round(per_step / (batch * world))), "p95": p95,
            "mean": round(sum(capped) / len(capped))}


MERGE_KINDS = ("qa", "mcqa")


def merge_qa(rows: list[dict[str, Any]], per_chat: int, rng: Any = None) -> list[dict[str, Any]]:
    """Single-turn Q&A rows that share a system prompt become multi-turn chats of up to ``per_chat`` questions:
    the system prompt (~300 tokens, ~80% of a Q&A row, trained on by nobody) is then read once per chat instead of
    once per question. Every answer is still trained on (all assistant turns are). Two phrasings of one fact never
    share a chat (the second would just be copied from the first). Only for editions that don't think: Qwen's
    thinking template drops the traces of earlier turns, which would teach answering without thinking."""
    if per_chat < 2:
        return list(rows)
    import random as _random

    rng = rng or _random.Random(0)
    groups: dict[str, list[dict[str, Any]]] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages") or []
        if (row.get("kind") in MERGE_KINDS and not row.get("images") and len(messages) == 3
                and [m.get("role") for m in messages] == ["system", "user", "assistant"]):
            groups.setdefault(str(messages[0].get("content") or ""), []).append(row)
        else:
            out.append(row)
    for system, members in groups.items():
        rng.shuffle(members)
        pending = members
        while pending:
            chat, answers, left = [], set(), []
            for row in pending:
                answer = " ".join(str(row["messages"][2]["content"]).split()).lower()
                if len(chat) < per_chat and answer not in answers:
                    chat.append(row)
                    answers.add(answer)
                else:
                    left.append(row)
            turns = [{"role": "system", "content": system}]
            for row in chat:
                turns += [dict(row["messages"][1]), dict(row["messages"][2])]
            out.append({"messages": turns, "kind": "qa-chat" if len(chat) > 1 else chat[0].get("kind"),
                        "source": chat[0].get("source"), "check": {}, "merged": len(chat)})
            pending = left
    return out


def is_oom(exc: BaseException) -> bool:
    return "OutOfMemory" in type(exc).__name__ or "out of memory" in str(exc).lower()


def speed_options(args: argparse.Namespace, preset: dict[str, Any], world: int, text: bool,
                  plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """SFTConfig settings for speed. Under DDP each GPU takes its own batches, so gradient accumulation is divided
    by the number of GPUs: the same effective batch (same quality) in about half the steps."""
    options: dict[str, Any] = {
        "gradient_accumulation_steps": plan["accum"] if plan else max(1, round(preset["accum"] / max(1, world))),
        "dataloader_num_workers": max(0, int(getattr(args, "loader_workers", 0) or 0)),
    }
    if options["dataloader_num_workers"]:
        options["dataloader_pin_memory"] = True
        options["dataloader_persistent_workers"] = True
    if world > 1:
        options["ddp_find_unused_parameters"] = False
    if text:
        options["dataset_num_proc"] = max(1, min(4, os.cpu_count() or 2))
        if getattr(args, "group_by_length", False) and not (plan and plan.get("p95")):
            # batches of similar length: much less padding (not with --token-budget: padding-free batches have no
            # padding to save, and grouping the longest rows together would only raise the peak memory)
            options["group_by_length"] = True
    return options


def _all_ranks(flag: bool) -> bool:
    """True when any GPU says so (DDP ranks must stop at the same step or they hang)."""
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            value = torch.tensor([1.0 if flag else 0.0], device="cuda" if torch.cuda.is_available() else "cpu")
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            return bool(value.item())
    except Exception:  # never let the check itself stop training
        return flag
    return flag


def deadline_callback(args: argparse.Namespace, clock: Any = time.time) -> Any:
    """A TrainerCallback that ends training early when the session deadline (``--deadline``) minus the time the
    eval, tests, export and upload need (``--reserve-minutes``) is reached - Kaggle stops at 12 h, and a run that
    is killed there loses everything. ``None`` without a deadline."""
    deadline = float(getattr(args, "deadline", 0) or 0)
    if deadline <= 0:
        return None
    from transformers import TrainerCallback

    stop_at = deadline - float(getattr(args, "reserve_minutes", 60)) * 60

    class StopBeforeDeadline(TrainerCallback):
        def on_step_end(self, train_args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
            if _all_ranks(clock() >= stop_at) and not control.should_training_stop:
                print(f"time limit: stopping at step {state.global_step}/{state.max_steps} so the export and upload "
                      f"finish before the session ends ({getattr(args, 'reserve_minutes', 60):.0f} min kept for them)",
                      flush=True)
                args.stopped_early = {"step": int(state.global_step), "of": int(state.max_steps)}
                control.should_training_stop = True
            return control

    return StopBeforeDeadline()


class LossPlateau:
    """``--stop-when-learned``: the training loss (mean of the last 10 logs = 100 steps) has not fallen by 3% for
    ``patience`` steps (10% of the run, at least 200) after at least a quarter of the run and 3/4 of one epoch (most
    rows seen once): the model knows the data, and the rest would cost hours of weekly GPU time for (almost) nothing."""

    def __init__(self, window: int = 10, gain: float = 0.03, min_part: float = 0.25) -> None:
        self.window, self.gain, self.min_part = window, gain, min_part
        self.losses: list[float] = []
        self.best, self.best_step = float("inf"), 0

    def add(self, step: int, loss: float, total: int, epochs: float = 1.0) -> bool:
        self.losses.append(float(loss))
        recent = self.losses[-self.window:]
        if len(recent) < max(3, self.window // 2):
            return False
        smooth = sum(recent) / len(recent)
        if smooth < self.best * (1 - self.gain):
            self.best, self.best_step = smooth, step
        total = max(1, int(total))
        first = max(total * self.min_part, 0.75 * total / max(1.0, float(epochs or 1.0)))
        return step >= first and step - self.best_step >= max(200, total // 10)


def learned_callback(args: argparse.Namespace, agree: Any = None) -> Any:
    """A TrainerCallback that ends training when the loss stops falling (see :class:`LossPlateau`); ``None``
    unless ``--stop-when-learned``."""
    if not getattr(args, "stop_when_learned", False):
        return None
    from transformers import TrainerCallback

    agree = agree or _all_ranks
    plateau = LossPlateau()

    class StopWhenLearned(TrainerCallback):
        flat = False

        def on_log(self, train_args: Any, state: Any, control: Any, logs: Any = None, **_kwargs: Any) -> Any:
            if logs and "loss" in logs:
                self.flat = plateau.add(int(state.global_step), float(logs["loss"]), int(state.max_steps),
                                        float(getattr(state, "num_train_epochs", 1) or 1))
            return control

        def on_step_end(self, train_args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
            if state.global_step % 10 == 0 and agree(self.flat) and not control.should_training_stop:
                print(f"learned: the loss stopped falling (~{plateau.best:.3f}) - stopping at step "
                      f"{state.global_step}/{state.max_steps}, the time goes to the next edition", flush=True)
                args.stopped_early = {"step": int(state.global_step), "of": int(state.max_steps), "reason": "learned"}
                control.should_training_stop = True
            return control

    return StopWhenLearned()


def training_callbacks(args: argparse.Namespace) -> list[Any] | None:
    found = [cb for cb in (deadline_callback(args), learned_callback(args)) if cb is not None]
    return found or None


def run_training(trainer: Any, rebuild: Any, world: int, tries: int = 4) -> Any:
    """``trainer.train()``. When it fails before the first step, ``rebuild(exc)`` may return a safer trainer
    (half the batch after a CUDA out-of-memory, no length grouping when an Unsloth / TRL combination cannot group)
    instead of losing the session; ``None`` = give up. Under DDP the other GPU's process would wait forever for a
    retry, so there the error goes up (the notebook then trains the edition on one T4)."""
    for _ in range(tries):
        try:
            return trainer.train()
        except Exception as exc:
            state = getattr(trainer, "state", None)
            if world > 1 or (state is not None and getattr(state, "global_step", 0) > 0):
                raise
            safer = rebuild(exc)
            if safer is None:
                raise
            trainer = safer
    return trainer.train()


def end_ddp() -> None:
    """Close the process group after training: only rank 0 goes on (eval, export, upload)."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception as exc:  # the model is trained; a teardown hiccup must not lose it
        print(f"DDP teardown: {type(exc).__name__}: {exc}")


class export_lock:  # noqa: N801 - used as a context manager
    """Only one edition merges / converts / uploads at a time when two train side by side (RAM and the 20 GB
    output disk). A no-op without a path or on Windows."""

    def __init__(self, path: str | None, waiting: str = "waiting for the other edition to finish its export / upload..."):
        self.path = path
        self.waiting = waiting
        self.handle: Any = None

    def __enter__(self) -> export_lock:
        if not self.path:
            return self
        try:
            import fcntl
        except ImportError:
            return self
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a+")  # noqa: SIM115 - closed in __exit__
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(self.waiting, flush=True)
            fcntl.flock(self.handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.handle is not None:
            try:
                import fcntl

                fcntl.flock(self.handle, fcntl.LOCK_UN)
            finally:
                self.handle.close()
                self.handle = None


def clean_uploaded(out: Path, report: dict[str, Any]) -> list[str]:
    """``--clean-after-upload``: delete the big files that are safely on Hugging Face (frees the 20 GB disk
    for the next edition). The GGUFs + adapter only when the GGUF repo is up; ``safetensors/`` only when its own
    upload worked too."""
    out = Path(out)
    removed: list[str] = []
    if report.get("hub_repo"):
        for path in sorted(out.glob("*.gguf")) + [out / "lora"]:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed.append(path.name)
            elif path.exists():
                path.unlink()
                removed.append(path.name)
        if report.get("safetensors_uploaded") or not report.get("safetensors"):
            if (out / SAFETENSORS_DIR).is_dir():
                shutil.rmtree(out / SAFETENSORS_DIR, ignore_errors=True)
                removed.append(SAFETENSORS_DIR)
    return removed


# --------------------------------------------------------------------------- distillation (Qwen3's strong-to-weak recipe)
#: The teacher in Qwen3's two-phase "strong-to-weak" distillation: a bigger Qwen3 with the same tokenizer as the 4B
#: students (vocabulary 151,936), so its token probabilities line up with theirs one to one. Apache-2.0: its answers
#: keep a commercial-only run commercial.
TEACHER_MODEL = "unsloth/Qwen3-8B"
UNCENSORED_TEACHER = "huihui-ai/Huihui-Qwen3-8B-abliterated-v2"
TEACHER_LICENSE = ("Apache-2.0 (Qwen3)", True, "https://huggingface.co/Qwen/Qwen3-8B")
#: Phase 1 only needs the teacher's *text*, so it can be the strongest model that runs on Kaggle's 2x T4: Qwen's own
#: 4-bit AWQ Qwen3-14B (~10 GB; the T4 has vLLM's AWQ kernels, and it thinks like the 8B and the 4B). Phase 2 needs
#: the teacher next to the student on one T4, where a 14B doesn't fit, so the 8B grades. There is no 4-bit
#: uncensored 14B that runs on a T4 (the community AWQ builds need Ampere kernels), so uncensored runs keep the
#: abliterated 8B for both phases (a censored writer would teach the student to refuse again).
WRITER_MODEL = "Qwen/Qwen3-14B-AWQ"
#: Qwen3's own "not thinking" marker. The hybrid teacher scores a non-thinking student's reply after it, so the
#: teacher never pushes the student to open with ``<think>``.
NO_THINK_PREFIX = "<think>\n\n</think>\n\n"
#: Qwen's sampling advice for each mode, used while the teacher writes answers.
TEACHER_SAMPLING = {True: {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 1536},
                    False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_new_tokens": 768}}
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def teacher_model(args: argparse.Namespace, phase: str = "grade") -> str:
    """``--teacher auto``: the best teacher for each phase.

    * ``"answers"`` (phase 1, writes the answers): Qwen3-14B-AWQ on vLLM; Qwen3-8B without vLLM or if the 14B did not
      start;
    * ``"grade"`` (phase 2, grades the student's own replies token by token): Qwen3-8B, which fits next to the student.

    An uncensored run uses the abliterated Qwen3-8B for both (a censored teacher would teach the uncensored student
    to refuse again). An explicit ``--teacher`` is used for both phases."""
    chosen = str(getattr(args, "teacher", "auto") or "auto").strip()
    if chosen.lower() != "auto":
        return chosen
    uncensored = bool(getattr(args, "uncensored", False))
    if phase == "answers" and not uncensored:
        fallback = getattr(args, "writer_fallback", None)
        if fallback:
            return str(fallback)
        if use_vllm(args):
            return WRITER_MODEL
    return UNCENSORED_TEACHER if uncensored else TEACHER_MODEL


def smaller_writer(args: argparse.Namespace, teacher: str) -> str | None:
    """The teacher that writes instead when ``teacher`` (the auto-picked 14B) does not run: the 8B."""
    if str(getattr(args, "teacher", "auto") or "auto").strip().lower() != "auto" or getattr(args, "writer_fallback", None):
        return None
    fallback = UNCENSORED_TEACHER if getattr(args, "uncensored", False) else TEACHER_MODEL
    return fallback if fallback != teacher else None


def prequantized(model: str) -> bool:
    """A checkpoint that already is 4-bit (AWQ / GPTQ): vLLM loads it as it is, never with bitsandbytes on top."""
    return bool(re.search(r"(?i)[-_.](awq|gptq)(?:$|[-_.])", model))


def same_tokenizer(student: Any, teacher: Any) -> bool:
    """Phase 2 compares the two models' probabilities token id by token id, so both must cut text the same way
    (Qwen3 and Qwen3-VL do; Qwen3.5 or Llama students do not)."""
    probes = ("Hello! How do I craft a diamond pickaxe in Minecraft 1.21?",
              "<|im_start|>user\nZombies at night: 12 or 13 hearts? Ça va? 你好 <|im_end|>")
    try:
        return all(list(student(p, add_special_tokens=False)["input_ids"])
                   == list(teacher(p, add_special_tokens=False)["input_ids"]) for p in probes)
    except Exception:  # noqa: BLE001 - can't tell: not the same
        return False


def _job_key(teacher: str, messages: list[dict[str, str]], thinking: bool) -> str:
    import hashlib

    blob = json.dumps([teacher, bool(thinking), messages], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def teacher_jobs(rows: list[dict[str, Any]], preset: dict[str, Any], limit: int, teacher: str) -> list[dict[str, Any]]:
    """The rows the teacher answers in phase 1 (off-policy distillation).

    * everyday chat (human-written answers) gets the teacher's answer, with a thinking trace for thinking editions;
    * Minecraft Q&A is rewritten around its exact answer (it goes in as a reference, so nothing is invented);
    * game turns keep their rule-checked JSON, and Nemotron's thinking rows were already distilled from a far bigger
      model, so neither is touched."""
    thinking = bool(preset.get("reasoning"))
    general = [i for i, row in enumerate(rows)
               if row.get("kind") == "general" and row.get("source") != "reasoning" and not row.get("images")]
    facts = [i for i, row in enumerate(rows) if row.get("kind") == "mcqa" and not row.get("images")]
    picked: list[int] = []
    for a, b in zip(general, facts, strict=False):  # both kinds, half and half while both last
        picked += [a, b]
    longer = general if len(general) > len(facts) else facts
    picked += longer[min(len(general), len(facts)):]
    jobs = []
    for index in picked[:max(0, int(limit))]:
        row = rows[index]
        if not row.get("messages") or row["messages"][-1].get("role") != "assistant":
            continue
        messages = [dict(m) for m in row["messages"][:-1]]
        reference = row["messages"][-1]["content"] if row.get("kind") == "mcqa" else ""
        if reference:
            note = ("Reference facts for your answer (exact, from the game's own data): " + reference
                    + "\nAnswer the user naturally and helpfully in your own words. Keep every fact, number and name "
                      "exactly as in the reference and add nothing you are not sure about. Never mention the reference.")
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = messages[0]["content"] + "\n\n" + note
            else:
                messages.insert(0, {"role": "system", "content": note})
        jobs.append({"row": index, "messages": messages, "reference": reference, "thinking": thinking,
                     "key": _job_key(teacher, messages, thinking)})
    return jobs


def split_teacher_reply(text: str, thinking: bool) -> tuple[str, str] | None:
    """``(trace, answer)`` from what the teacher wrote, ``None`` when it is unusable (cut off, empty)."""
    text = str(text or "").strip()
    if thinking:
        if "</think>" not in text:
            return None  # ran out of tokens while still thinking
        trace, answer = text.split("</think>", 1)
        trace = re.sub(r"\s+\n", "\n", trace.replace("<think>", "")).strip()
        trace = trace.replace("{", "(").replace("}", ")")  # traces never carry JSON (the bot parses the reply)
        answer = answer.strip()
        return (trace, answer) if trace and answer and "<think>" not in answer else None
    answer = strip_thinking(text).strip()
    return ("", answer) if answer and "<think>" not in answer and "</think>" not in answer else None


def accept_teacher_answer(job: dict[str, Any], trace: str, answer: str, uncensored: bool) -> str:
    """Why a teacher answer is dropped (``""`` = keep it)."""
    if len(answer) < 2:
        return "empty"
    if uncensored and is_refusal(answer):
        return "refusal"
    if job.get("reference"):
        said = set(_NUMBER.findall(answer))
        if any(number not in said for number in _NUMBER.findall(job["reference"])):
            return "changed a fact"
        if "reference" in (trace + " " + answer).lower():
            return "mentioned the reference"
    return ""


def apply_teacher_answers(rows: list[dict[str, Any]], jobs: list[dict[str, Any]], answers: dict[str, str],
                          teacher: str, uncensored: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows with the teacher's answers in place of the originals (an original stays when its answer is unusable)."""
    out = list(rows)
    stats: dict[str, Any] = {"teacher": teacher, "asked": len(jobs), "used": 0, "general": 0, "mcqa": 0, "dropped": {}}
    for job in jobs:
        text = answers.get(job["key"])
        parsed = split_teacher_reply(text, job["thinking"]) if text is not None else None
        why = "missing" if text is None else "cut off" if parsed is None else accept_teacher_answer(
            job, parsed[0], parsed[1], uncensored)
        if why:
            stats["dropped"][why] = stats["dropped"].get(why, 0) + 1
            continue
        trace, answer = parsed  # type: ignore[misc]
        row = dict(out[job["row"]])
        row["messages"] = [dict(m) for m in row["messages"]]
        row["messages"][-1]["content"] = answer
        if job["thinking"]:
            row["reasoning"] = trace
        row["teacher"] = teacher
        out[job["row"]] = row
        stats["used"] += 1
        stats[str(row.get("kind"))] = stats.get(str(row.get("kind")), 0) + 1
    return out, stats


def _answers_file(args: argparse.Namespace, teacher: str) -> Path | None:
    folder = getattr(args, "stream_cache", None)
    if not folder:
        return None
    import hashlib

    return Path(folder) / f"teacher-{hashlib.sha256(teacher.encode('utf-8')).hexdigest()[:16]}.json"


def load_teacher_answers(args: argparse.Namespace, teacher: str) -> dict[str, str]:
    """Answers the teacher already wrote this session (every edition shares them through ``--stream-cache``)."""
    path = _answers_file(args, teacher)
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save_teacher_answers(args: argparse.Namespace, teacher: str, answers: dict[str, str]) -> None:
    path = _answers_file(args, teacher)
    if path is None or not answers:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with export_lock(lock_file(path), waiting="waiting for another edition to save the teacher's answers..."):
        merged = load_teacher_answers(args, teacher)
        merged.update(answers)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)


def _distill_file(out: Path, part: int | None = None) -> Path:
    rows = _rows_file(out)
    run = rows.name[len(".ddp-rows-"):-len(".json")]
    rows.parent.mkdir(parents=True, exist_ok=True)
    return rows.parent / (f".distill-{run}.json" if part is None else f".distill-{run}-{part}.json")


def free_gpu(*objects: Any) -> None:
    """Give GPU memory back (the teacher leaves before the student needs the space). The caller must drop its own
    references first (``teacher = None``): passing a model that a local name still holds frees nothing."""
    import gc

    del objects
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - nothing to free without torch
        pass


def load_teacher(teacher: str, max_seq: int) -> tuple[Any, Any]:
    """The teacher, 4-bit (an 8B is ~5.7 GB), on this process's GPU."""
    from unsloth import FastLanguageModel  # noqa: I001
    import torch

    load: dict[str, Any] = {"model_name": teacher, "max_seq_length": max_seq, "load_in_4bit": True}
    if torch.cuda.is_available():
        if dist_info()[1] > 1:  # DDP: each process uses its own T4 (both see both)
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        load["device_map"] = {"": torch.cuda.current_device()}
    model, tokenizer = FastLanguageModel.from_pretrained(**load)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def teacher_generate(teacher: str, jobs: list[dict[str, Any]], max_seq: int, seconds: float, batch: int = 0,
                     log: Any = print) -> dict[str, str]:
    """The teacher writes the answers (batched, shortest prompts first). Cut-off answers are left out; whatever is
    not done when ``seconds`` run out keeps its original answer."""
    if not jobs:
        return {}
    import torch

    model, tokenizer = load_teacher(teacher, max_seq)
    answers: dict[str, str] = {}
    try:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        stop_ids = {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id} - {None}
        thinking = bool(jobs[0]["thinking"])
        sampling = dict(TEACHER_SAMPLING[thinking])
        new_tokens = sampling.pop("max_new_tokens")
        size = batch or (8 if thinking else 16)
        prompts = [(job["key"], tokenizer.apply_chat_template(job["messages"], tokenize=False, add_generation_prompt=True,
                                                              enable_thinking=thinking)) for job in jobs]
        prompts.sort(key=lambda item: len(item[1]))
        end, start, done_batches = time.time() + max(0.0, seconds), 0, 0
        while start < len(prompts):
            if time.time() > end:
                log(f"teacher: time is up after {len(answers)}/{len(prompts)} answers - the rest keep their originals")
                break
            chunk = prompts[start:start + size]
            try:
                inputs = tokenizer([text for _, text in chunk], return_tensors="pt", padding=True,
                                   add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=True,
                                            pad_token_id=tokenizer.pad_token_id, **sampling)
            except Exception as exc:  # noqa: BLE001 - smaller batches on OOM, otherwise give up on the teacher
                if is_oom(exc) and size > 1:
                    size = max(1, size // 2)
                    torch.cuda.empty_cache()
                    log(f"teacher: out of GPU memory - {size} answers at a time")
                    continue
                raise
            width = inputs["input_ids"].shape[1]
            for (key, _), sequence in zip(chunk, output, strict=False):
                generated = sequence[width:].tolist()
                if not any(token in stop_ids for token in generated):
                    continue  # cut off by max_new_tokens
                answers[key] = tokenizer.decode(generated, skip_special_tokens=True)
            start += len(chunk)
            done_batches += 1
            if done_batches % 10 == 0:
                log(f"teacher: {start}/{len(prompts)} answers written")
    finally:
        model = None  # our reference goes first, or the 8B teacher stays on the GPU
        free_gpu()
    return answers


#: vLLM for the teacher's answers: paged attention + continuous batching, several times faster than ``generate``.
#: This version is validated on Kaggle's 2x T4 (torch 2.10 / CUDA 12.8, Triton attention on sm75). It lives in its own
#: venv (the notebook installs it in the background), so Unsloth's torch is never touched.
VLLM_VERSION = "0.18.1"
VLLM_CHUNK = 512  # prompts per vLLM call (answers are saved and the time is checked between calls)


def use_vllm(args: argparse.Namespace) -> str:
    """The vLLM venv's python when the teacher should use it (``""`` = transformers' ``generate``)."""
    if str(getattr(args, "teacher_engine", "auto")) == "hf":
        return ""
    python = getattr(args, "vllm_python", None) or os.environ.get("PYMC_VLLM_PYTHON") or ""
    return python if python and Path(python).is_file() else ""


_VLLM_CAUSE = re.compile(r"(?:Error|Exception|error:|CUDA|out of memory|not supported|unsupported|Free memory|"
                         r"No available memory|Killed|NCCL)", re.I)


def vllm_root_cause(text: str, lines: int = 12) -> str:
    """The lines of a vLLM log that say *why* it failed (the end of the log is only the engine's wrapper error)."""
    picked: list[str] = []
    for line in text.splitlines():
        line = line.rstrip()
        if (_VLLM_CAUSE.search(line) and "site-packages" not in line and not line.strip().startswith(("~", "^"))
                and "Engine core initialization failed" not in line and line not in picked):
            picked.append(line)
    return "\n".join(picked[-lines:])


#: Variables torchrun / accelerate give each training process. The vLLM teacher is started from inside one of them and
#: must not inherit them: TORCHELASTIC_USE_AGENT_STORE=True makes every vLLM worker wait for a TCPStore server nobody
#: starts (it hangs ~10 min per try, then "Engine core initialization failed"), RANK / WORLD_SIZE confuse its own
#: tensor-parallel group.
_LAUNCHER_ENV = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE", "ROLE_RANK",
                 "ROLE_NAME", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
_LAUNCHER_PREFIXES = ("TORCHELASTIC_", "TORCH_ELASTIC_", "ACCELERATE_")
#: A healthy vLLM start (download of the 16 GB teacher, loading, memory profiling) takes ~5-8 min on Kaggle.
VLLM_STARTUP_SECONDS = 15 * 60


def vllm_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment of the vLLM teacher: ours without the DDP launcher's variables."""
    env = {key: value for key, value in (os.environ if base is None else base).items()
           if key not in _LAUNCHER_ENV and not key.startswith(_LAUNCHER_PREFIXES)}
    env["VLLM_LOGGING_LEVEL"] = "INFO"  # the full log is kept; only the cause is printed
    return env


def _stop_group(proc: Any) -> None:
    """Stop the vLLM worker *and* its engine / GPU processes (they would keep the T4s' memory)."""
    import signal

    for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except (OSError, AttributeError):
            proc.kill()
        try:
            proc.wait(timeout=wait)
            return
        except Exception:  # noqa: BLE001 - next signal
            continue


def run_vllm_process(command: list[str], stdout: Any = None, stderr: Any = None, env: Any = None,
                     timeout: float | None = None, ready: Path | None = None,
                     startup: float = VLLM_STARTUP_SECONDS, poll: float = 5.0) -> Any:
    """``subprocess.run`` for the vLLM worker, in its own process group: stopped as a whole at ``timeout``, or when
    the engine is not ready (``ready`` written by the worker) after ``startup`` seconds - a hang, not a slow start."""
    import subprocess

    proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
    started = time.time()
    try:
        while proc.poll() is None:
            waited = time.time() - started
            hung = ready is not None and not ready.exists() and waited > startup
            if hung or (timeout and waited > timeout):
                if stdout is not None and hasattr(stdout, "write"):
                    why = f"engine not ready after {waited / 60:.0f} min - stopped (it hangs)" if hung else "time limit"
                    stdout.write(f"\npymc: {why}\n")
                    stdout.flush()
                _stop_group(proc)
                return type("Done", (), {"returncode": 124 if hung else 125})()
            time.sleep(poll)
    finally:
        if proc.poll() is None:
            _stop_group(proc)
    return proc


def vllm_generate(teacher: str, jobs: list[dict[str, Any]], max_seq: int, seconds: float, python: str,
                  run: Any = None, log: Any = print) -> dict[str, str]:
    """The teacher writes the answers with vLLM in a separate process (``--vllm-worker``): every T4 this process
    sees, by tensor parallelism. Eager mode first (no CUDA graphs / torch.compile: the setup validated on Kaggle's
    T4s; a failed graph capture there cost a whole engine start); if the engine still does not start, once more
    with less GPU memory and fewer prompts at once. The full log of a failure is kept in ``$PYMC_LOGS``."""
    import subprocess
    import tempfile

    run = run or subprocess.run
    root = scratch_root()
    if root is not None:
        (root / "vllm-jobs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="pymc-vllm-", dir=str(root / "vllm-jobs") if root is not None else None))
    answers: dict[str, str] = {}
    for safe in (False, True):
        eager = True
        job, out, journal = work / "job.json", work / "answers.json", work / "vllm.log"
        job.write_text(json.dumps({"teacher": teacher, "max_seq": max_seq, "seconds": seconds, "eager": eager,
                                   "safe": safe,
                                   "jobs": [{"key": j["key"], "messages": j["messages"], "thinking": j["thinking"]}
                                            for j in jobs]}, ensure_ascii=False), encoding="utf-8")
        started = time.time()
        ready = Path(str(out) + ".ready")
        ready.unlink(missing_ok=True)
        runner = run or functools.partial(run_vllm_process, ready=ready, startup=min(VLLM_STARTUP_SECONDS,
                                                                                      max(300.0, seconds)))
        with open(journal, "w", encoding="utf-8") as handle:
            code = runner([python, os.path.abspath(__file__), "--vllm-worker", str(job), str(out)], stdout=handle,
                          stderr=subprocess.STDOUT, env=vllm_env(), timeout=seconds + 900).returncode
        if out.is_file():
            try:
                answers.update(json.loads(out.read_text(encoding="utf-8")))
            except ValueError:
                pass
        if code == 0 or answers:
            log(f"teacher (vLLM{', low memory' if safe else ''}): {len(answers)} answers in "
                f"{(time.time() - started) / 60:.1f} min")
            return answers
        text = journal.read_text(encoding="utf-8", errors="replace") if journal.is_file() else ""
        kept = ""
        if os.environ.get("PYMC_LOGS"):
            try:
                target = Path(os.environ["PYMC_LOGS"]) / f"vllm-teacher{'-low-memory' if safe else ''}.log"
                target.write_text(text, encoding="utf-8")
                kept = f"\n(full log: {target})"
            except OSError:
                pass
        log(f"teacher: vLLM {'(low memory) ' if safe else ''}failed after {(time.time() - started) / 60:.1f} min "
            f"(exit {code}). Cause:\n{vllm_root_cause(text) or text[-800:]}{kept}")
        took = time.time() - started
        seconds = max(0.0, seconds - took)
        if seconds < 300 or took > 8 * 60:  # a slow failure is a hang: a second try would only hang again
            break
    raise RuntimeError("vLLM could not run the teacher")


def vllm_memory_share(wanted: float, run: Any = None) -> float:
    """vLLM refuses to start when ``gpu_memory_utilization`` x the T4's memory is more than is free (the training
    processes already hold some): ask nvidia-smi (no CUDA context of our own) and stay a bit under the free part."""
    import subprocess

    run = run or subprocess.run
    try:
        out = run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                  capture_output=True, text=True, timeout=20).stdout
        free = min(float(a) / float(b) for a, b in (line.split(",") for line in out.strip().splitlines()))
    except Exception:  # noqa: BLE001 - no nvidia-smi: vLLM's default check decides
        return wanted
    return round(max(0.3, min(wanted, free - 0.05)), 2)


def vllm_worker(job_path: str, out_path: str) -> int:
    """``--vllm-worker JOB OUT`` (run by the vLLM venv's python): write the teacher's answers to OUT as they come."""
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    import torch
    from vllm import LLM, SamplingParams

    jobs = job["jobs"]
    thinking = bool(jobs and jobs[0]["thinking"])
    sampling = dict(TEACHER_SAMPLING[thinking])
    new_tokens = int(sampling.pop("max_new_tokens"))
    gpus = torch.cuda.device_count()
    limit = int(job["max_seq"]) + new_tokens
    share = vllm_memory_share(0.75 if job.get("safe") else 0.88)
    options: dict[str, Any] = {"model": job["teacher"], "dtype": "float16", "max_model_len": limit,
                               "tensor_parallel_size": 2 if gpus >= 2 else 1, "gpu_memory_utilization": share,
                               "disable_custom_all_reduce": True, "enable_prefix_caching": True,
                               "enforce_eager": bool(job.get("eager"))}
    if job.get("safe"):
        options["max_num_seqs"] = 64
    print(f"vLLM: {options}", flush=True)
    if gpus < 2 and not prequantized(str(job["teacher"])):  # 16 GB of fp16 don't fit one T4: 4-bit while loading
        options["quantization"] = "bitsandbytes"
    llm = LLM(**options)
    Path(out_path + ".ready").write_text("ready", encoding="utf-8")  # the engine started: no hang
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(max_tokens=new_tokens, **sampling)
    prompts = []
    for item in jobs:
        text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=thinking)
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) <= int(job["max_seq"]):
            prompts.append((item["key"], text))  # longer prompts keep their original answers
    answers: dict[str, str] = {}
    end = time.time() + float(job["seconds"])
    for start in range(0, len(prompts), VLLM_CHUNK):
        if time.time() > end:
            print(f"time is up after {len(answers)} answers")
            break
        chunk = prompts[start:start + VLLM_CHUNK]
        for (key, _), result in zip(chunk, llm.generate([text for _, text in chunk], params, use_tqdm=False),
                                    strict=False):
            if result.outputs and result.outputs[0].finish_reason == "stop":  # "length" = cut off
                answers[key] = result.outputs[0].text
        temp = Path(out_path + ".tmp")
        temp.write_text(json.dumps(answers, ensure_ascii=False), encoding="utf-8")
        temp.replace(out_path)
        print(f"{min(start + VLLM_CHUNK, len(prompts))}/{len(prompts)} prompts done", flush=True)
    if not prompts:
        Path(out_path).write_text("{}", encoding="utf-8")
    return 0


#: Name of the private Hugging Face dataset that keeps the teacher's answers between Kaggle sessions.
TEACHER_CACHE_REPO = "Minecraftism-teacher-answers"


def teacher_cache_remote(args: argparse.Namespace, teacher: str, api: Any = None) -> tuple[Any, str, str] | None:
    """``(api, repo, file)`` for ``--teacher-cache-repo`` (``auto`` = <you>/Minecraftism-teacher-answers)."""
    repo = str(getattr(args, "teacher_cache_repo", "") or "")
    token = os.environ.get("HF_TOKEN")
    if not repo or not (token or api):
        return None
    import hashlib

    try:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
        if "/" not in repo:
            repo = f"{api.whoami().get('name')}/{TEACHER_CACHE_REPO if repo == 'auto' else repo}"
    except Exception as exc:  # noqa: BLE001 - the session cache still works
        print(f"teacher cache on Hugging Face unavailable ({type(exc).__name__}: {exc})")
        return None
    return api, repo, f"teacher-{hashlib.sha256(teacher.encode('utf-8')).hexdigest()[:16]}.json"


def pull_teacher_answers(remote: tuple[Any, str, str] | None) -> dict[str, str]:
    """Answers an earlier session uploaded (``{}`` when there are none yet)."""
    if remote is None:
        return {}
    api, repo, name = remote
    try:
        path = api.hf_hub_download(repo_id=repo, filename=name, repo_type="dataset")
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}
    except Exception:  # noqa: BLE001 - first session: nothing there yet
        return {}


def push_teacher_answers(remote: tuple[Any, str, str] | None, answers: dict[str, str]) -> None:
    if remote is None or not answers:
        return
    api, repo, name = remote
    try:
        api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=json.dumps(answers, ensure_ascii=False).encode("utf-8"), path_in_repo=name,
                        repo_id=repo, repo_type="dataset", commit_message=f"{len(answers)} teacher answers")
        print(f"teacher answers saved to https://huggingface.co/datasets/{repo} (private) for the next session")
    except Exception as exc:  # noqa: BLE001
        print(f"could not save the teacher answers to Hugging Face ({type(exc).__name__}: {exc})")


def serve_distill_job(out: Path, rank: int, world: int) -> None:
    """DDP, ranks > 0: write this GPU's share of the teacher answers rank 0 asked for."""
    job = json.loads(_distill_file(out).read_text(encoding="utf-8"))
    answers: dict[str, str] = {}
    try:
        answers = teacher_generate(job["teacher"], job["jobs"][rank::world], int(job["max_seq"]), float(job["seconds"]),
                                   log=lambda *_: None)
    finally:  # an empty share is fine (those rows keep their originals); a missing one would make rank 0 wait
        target = _distill_file(out, rank)
        temp = target.with_name(target.name + ".tmp")
        temp.write_text(json.dumps(answers, ensure_ascii=False), encoding="utf-8")
        temp.replace(target)


def _wait_for_parts(out: Path, world: int, timeout: float, poll: float = 5.0) -> dict[str, str]:
    answers: dict[str, str] = {}
    end = time.time() + timeout
    for part in range(1, world):
        target = _distill_file(out, part)
        while not _fresh(target):
            if time.time() > end:
                print(f"teacher: GPU {part} did not finish its answers - its rows keep their originals")
                return answers
            time.sleep(poll)
        try:
            answers.update(json.loads(target.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    return answers


def _vllm_failed_file() -> Path | None:
    root = scratch_root()  # /kaggle/tmp: gone with the session, so a new session tries vLLM again
    return root / "vllm-failed.json" if root is not None else None


def _vllm_failures(session: float) -> list[str]:
    marker = _vllm_failed_file()
    try:
        saved = json.loads(marker.read_text(encoding="utf-8")) if marker is not None and marker.is_file() else {}
        return list(saved.get("failed", [])) if float(saved.get("session", -1)) == float(session) else []
    except (OSError, ValueError, TypeError, AttributeError):
        return []


def vllm_failed_before(teacher: str, session: float) -> bool:
    """True when vLLM already failed with this teacher in this session (``session`` = its ``--deadline``): the next
    edition doesn't lose its teacher time on it again."""
    return bool(session) and teacher in _vllm_failures(session)


def remember_vllm_failure(teacher: str, session: float) -> None:
    marker = _vllm_failed_file()
    if marker is None or not session:
        return
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        failed = sorted(set(_vllm_failures(session)) | {teacher})
        marker.write_text(json.dumps({"session": float(session), "failed": failed}), encoding="utf-8")
    except OSError:
        pass


def distill_offpolicy(args: argparse.Namespace, preset: dict[str, Any], rows: list[dict[str, Any]], world: int = 1,
                      generate: Any = None) -> list[dict[str, Any]]:
    """Phase 1 of Qwen3's strong-to-weak distillation: the teacher answers up to ``--distill-rows`` prompts and the
    student is fine-tuned on those answers. With DDP both T4s write answers (half each). The answers are cached for
    the session, so the next edition only waits for the ones it has not seen."""
    limit = int(getattr(args, "distill_rows", 0) or 0)
    if limit <= 0:
        return rows
    teacher = teacher_model(args, "answers")
    jobs = teacher_jobs(rows, preset, limit, teacher)
    python = use_vllm(args)
    session = float(getattr(args, "deadline", 0) or 0)
    if python and vllm_failed_before(teacher, session):
        print(f"teacher: vLLM already failed with {teacher} in this session - using transformers")
        python = ""
    engine = "vLLM" if python else "transformers"
    known = load_teacher_answers(args, teacher)
    todo = [job for job in jobs if job["key"] not in known]
    print(f"distillation (Qwen3 strong-to-weak): teacher {teacher} - {len(jobs)} answers ({len(jobs) - len(todo)} "
          f"cached, {len(todo)} to write with {engine} on {'every' if python else world} GPU(s), up to "
          f"{args.distill_minutes:.0f} min)" + (f"; then {args.onpolicy_steps} on-policy steps" if getattr(
              args, "onpolicy_steps", 0) and not preset.get("vision") else ""))
    if args.dry_run or not jobs:
        return rows
    seconds = float(args.distill_minutes) * 60
    if args.deadline:  # never let the teacher eat the time the training needs
        seconds = min(seconds, max(0.0, (args.deadline - time.time() - (args.reserve_minutes + 20) * 60) * 0.4))
    folder = getattr(args, "stream_cache", None)
    lock = lock_file(Path(folder) / "teacher") if folder else None
    retry_with: str | None = None
    with export_lock(lock, waiting="waiting for the other edition's teacher (its answers are reused)..."):
        known = load_teacher_answers(args, teacher)  # the other edition may just have written some
        remote = teacher_cache_remote(args, teacher) if any(job["key"] not in known for job in jobs) else None
        earlier = pull_teacher_answers(remote)
        if earlier:
            known.update(earlier)
            save_teacher_answers(args, teacher, earlier)
            print(f"teacher: {len(earlier)} answers from an earlier session (Hugging Face)")
        todo = [job for job in jobs if job["key"] not in known]
        if todo and seconds > 60:
            started = time.time()
            fresh: dict[str, str] = {}
            try:
                if generate is not None:
                    fresh = generate(teacher, todo, args.max_seq, seconds)
                else:
                    if python:
                        try:
                            fresh = vllm_generate(teacher, todo, args.max_seq, seconds, python)
                        except Exception as exc:  # noqa: BLE001 - the 8B, or transformers, can still do it
                            remember_vllm_failure(teacher, session)
                            retry_with = None if fresh else smaller_writer(args, teacher)
                            print(f"teacher: {teacher} did not run on vLLM ({exc}) - "
                                  + (f"{retry_with} writes instead" if retry_with else "writing with transformers"))
                            python = ""
                    seconds = max(0.0, seconds - (time.time() - started))
                    if retry_with:
                        pass  # after the lock is released: the 8B writes (on vLLM if it can)
                    elif not python and seconds < 300:  # loading the teacher alone takes ~4 min
                        print(f"teacher: only {seconds / 60:.0f} min of its time left - not loading it with transformers;"
                              " the rows keep their original answers")
                    elif not python and world > 1:
                        target = _distill_file(Path(args.out))
                        temp = target.with_name(target.name + ".tmp")
                        temp.write_text(json.dumps({"teacher": teacher, "jobs": todo, "max_seq": args.max_seq,
                                                    "seconds": seconds}, ensure_ascii=False), encoding="utf-8")
                        temp.replace(target)
                        fresh = teacher_generate(teacher, todo[0::world], args.max_seq, seconds)
                        fresh.update(_wait_for_parts(Path(args.out), world, seconds + 1800))
                    elif not python:
                        fresh = teacher_generate(teacher, todo, args.max_seq, seconds)
            except Exception as exc:  # noqa: BLE001 - the original answers still train a good model
                print(f"distillation: the teacher failed ({type(exc).__name__}: {exc}) - training on the original answers")
            known.update(fresh)
            save_teacher_answers(args, teacher, fresh)
            if retry_with:
                args.distill_minutes = max(1.0, float(args.distill_minutes) - (time.time() - started) / 60)
            else:
                print(f"teacher wrote {len(fresh)} answers in {(time.time() - started) / 60:.0f} min")
            if fresh:
                push_teacher_answers(remote, known)
        elif todo:
            print("distillation: not enough session time left for the teacher - training on the original answers")
    if retry_with:
        args.writer_fallback = retry_with
        return distill_offpolicy(args, preset, rows, world, generate)
    distilled, stats = apply_teacher_answers(rows, jobs, known, teacher, bool(getattr(args, "uncensored", False)))
    print(f"distillation: {stats['used']}/{stats['asked']} rows now carry the teacher's answer"
          + (f" (kept the original for: {json.dumps(stats['dropped'])})" if stats["dropped"] else ""))
    args.distill_report = stats
    return distilled


def reverse_kl(student: Any, teacher: Any, chunk: int = 256) -> Any:
    """Mean per-token KL(student || teacher) over ``[tokens, vocab]`` logits (fp32, in chunks to save memory).
    Reverse KL is "mode seeking": the student learns the teacher's best answers instead of spreading itself thin
    over everything the bigger model could say - the loss of Qwen3's on-policy phase."""
    import torch

    vocab = min(student.shape[-1], teacher.shape[-1])
    total = student.new_zeros((), dtype=torch.float32)
    count = student.shape[0]
    for start in range(0, count, chunk):
        own = torch.log_softmax(student[start:start + chunk, :vocab].float(), dim=-1)
        ref = torch.log_softmax(teacher[start:start + chunk, :vocab].float(), dim=-1)
        total = total + (own.exp() * (own - ref)).sum()
    return total / max(1, count)


def token_logits(model: Any, sequences: list[list[int]], starts: list[int], lengths: list[int], pad_id: int) -> Any:
    """Logits that predicted each reply token: ``[sum(lengths), vocab]`` (right-padded batch)."""
    import torch

    device = next(model.parameters()).device
    width = max(len(sequence) for sequence in sequences)
    ids = torch.full((len(sequences), width), int(pad_id), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences):
        ids[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[i, :len(sequence)] = 1
    logits = model(input_ids=ids, attention_mask=mask).logits
    return torch.cat([logits[i, starts[i] - 1:starts[i] - 1 + lengths[i]] for i in range(len(sequences))])


def onpolicy_prompts(rows: list[dict[str, Any]], steps: int, batch: int, world: int, seed: int) -> list[list[dict[str, str]]]:
    """Prompts for the on-policy phase: everyday chat and Minecraft questions (game turns keep their rule-checked
    JSON - the teacher never saw the bot's rules)."""
    import random

    prompts = [row["messages"][:-1] for row in rows if row.get("kind") in ("general", "mcqa") and not row.get("images")
               and row.get("messages") and row["messages"][-1].get("role") == "assistant"]
    if not prompts:
        return []
    random.Random(seed + 7).shuffle(prompts)
    need = steps * batch * world
    return (prompts * (need // len(prompts) + 1))[:need]


def onpolicy_distill(args: argparse.Namespace, preset: dict[str, Any], model: Any, tokenizer: Any,
                     rows: list[dict[str, Any]], world: int = 1, rank: int = 0) -> dict[str, Any]:
    """Phase 2 of Qwen3's strong-to-weak distillation (on-policy): the student writes replies to real prompts and the
    teacher grades every token of them; the student minimises reverse KL to the teacher on its own replies. Qwen
    found it beats reinforcement learning at a tenth of the GPU time. With DDP each T4 takes different prompts and
    the LoRA gradients are averaged. Any failure keeps the fine-tuned model as it is."""
    steps, batch = int(getattr(args, "onpolicy_steps", 0) or 0), max(1, int(getattr(args, "onpolicy_batch", 4) or 4))
    if steps <= 0 or preset.get("vision"):
        return {}
    prompts = onpolicy_prompts(rows, steps, batch, world, args.seed)
    if not prompts:
        return {}
    teacher_id = teacher_model(args, "grade")
    try:
        from transformers import AutoTokenizer

        if not same_tokenizer(tokenizer, AutoTokenizer.from_pretrained(teacher_id)):
            print(f"on-policy distillation skipped: {teacher_id} cuts text into different tokens than this student "
                  "(it needs a teacher from the same family; phase 1 still used the teacher's answers)")
            return {}
    except Exception as exc:  # noqa: BLE001 - loading the teacher below tells for sure
        print(f"note: could not compare the tokenizers first ({type(exc).__name__})")
    from unsloth import FastLanguageModel  # noqa: I001
    import torch
    import torch.distributed as dist

    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"  # Unsloth hands out real logits only when asked
    teacher = None
    try:
        teacher, _ = load_teacher(teacher_id, args.max_seq)
    except Exception as exc:  # noqa: BLE001
        print(f"on-policy distillation skipped: the teacher did not load ({type(exc).__name__}: {exc})")
    if _all_ranks(teacher is None):
        teacher = None
        free_gpu()
        os.environ.pop("UNSLOTH_RETURN_LOGITS", None)
        return {}
    print(f"on-policy distillation: {steps} steps x {batch} replies x {world} GPU(s), teacher {teacher_id}")
    mine = prompts[rank::world]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(args.onpolicy_lr), weight_decay=0.0)
    half = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # T4: fp16, like the Trainer
    scaler = (torch.amp.GradScaler("cuda", enabled=half == torch.float16) if hasattr(torch, "amp")
              and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=half == torch.float16))
    prefix = [] if preset.get("reasoning") else tokenizer(NO_THINK_PREFIX, add_special_tokens=False)["input_ids"]
    stop_ids = {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id} - {None}
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    stop_at = (args.deadline - (args.reserve_minutes + 5) * 60) if args.deadline else float("inf")
    refresh = max(1, int(getattr(args, "onpolicy_refresh", 1) or 1))
    padding_side = tokenizer.padding_side
    started, losses, step, position = time.time(), [], 0, 0
    pending: list[list[tuple[list[int], list[int]]]] = []

    def rollouts(count: int) -> list[tuple[list[int], list[int]]]:
        """1. the student writes ``count`` replies (sampled, as it would answer) in one batched call."""
        nonlocal position
        FastLanguageModel.for_inference(model)
        tokenizer.padding_side = "left"
        texts = [render(tokenizer, mine[(position + i) % len(mine)], preset, generation=True) for i in range(count)]
        position += count
        encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            output = model.generate(**encoded, max_new_tokens=int(args.onpolicy_tokens), do_sample=True,
                                    temperature=1.0, top_p=1.0, top_k=0, pad_token_id=pad)
        width = encoded["input_ids"].shape[1]
        written = []
        for i in range(len(texts)):
            prompt = encoded["input_ids"][i][encoded["attention_mask"][i].bool()].tolist()
            reply = output[i, width:].tolist()
            cut = next((k + 1 for k, token in enumerate(reply) if token in stop_ids), len(reply))
            if reply[:cut]:
                written.append((prompt, reply[:cut]))
        return written

    while step < steps:
        ok, value = True, 0.0
        try:
            if not pending:  # replies for the next `refresh` steps at once: one big generate call, not many small
                written = rollouts(batch * min(refresh, steps - step))
                pending = [written[i:i + batch] for i in range(0, len(written), batch)] or [[]]
            samples = pending.pop(0)
            FastLanguageModel.for_training(model)
            if samples:  # 2. both score the student's tokens: reverse KL on them
                lengths = [len(reply) for _, reply in samples]
                with torch.autocast("cuda", dtype=half):
                    own = token_logits(model, [p + r for p, r in samples], [len(p) for p, _ in samples], lengths, pad)
                with torch.no_grad(), torch.autocast("cuda", dtype=half):
                    ref = token_logits(teacher, [p + prefix + r for p, r in samples],
                                       [len(p) + len(prefix) for p, _ in samples], lengths, pad)
                loss = reverse_kl(own, ref)  # fp32
                scaler.scale(loss).backward()  # fp16 needs loss scaling or small gradients vanish
                value = float(loss.detach())
                del own, ref, loss
        except Exception as exc:  # noqa: BLE001
            ok = False
            pending = []
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if is_oom(exc) and (refresh > 1 or batch > 1) and not losses:
                if refresh > 1:
                    refresh = max(1, refresh // 2)
                else:
                    batch = max(1, batch // 2)
                print(f"on-policy distillation: out of GPU memory - {batch} replies per step, written {refresh} "
                      "steps at a time")
            else:
                print(f"on-policy distillation stopped at step {step}: {type(exc).__name__}: {exc}")
                batch = 0
        if _all_ranks(not ok):  # every GPU retries or stops together, or DDP hangs
            optimizer.zero_grad(set_to_none=True)
            if _all_ranks(batch == 0) or losses:
                break
            continue
        if world > 1 and dist.is_available() and dist.is_initialized():
            for param in params:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                dist.all_reduce(param.grad)  # same scale on both GPUs, so an overflow is seen by both
                param.grad.div_(world)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)  # skipped (and the scale lowered) if a gradient overflowed
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        losses.append(value)
        step += 1
        if step % 10 == 0 or step == steps:
            print(f"on-policy step {step}/{steps}: KL {sum(losses[-10:]) / len(losses[-10:]):.4f}")
        if _all_ranks(time.time() > stop_at):
            print("on-policy distillation: stopping early to leave time for the export")
            break
    # drop the references first: passing the teacher alone kept the 8B teacher on the GPU, and the eval after it ran
    # out of memory with ~100 MB free
    teacher = optimizer = scaler = None
    pending = []
    free_gpu()
    os.environ.pop("UNSLOTH_RETURN_LOGITS", None)
    tokenizer.padding_side = padding_side  # the saved tokenizer keeps its own setting
    FastLanguageModel.for_training(model)
    if not losses:
        return {}
    head, tail = losses[:max(1, len(losses) // 5)], losses[-max(1, len(losses) // 5):]
    report = {"teacher": teacher_id, "steps": len(losses), "replies": len(losses) * batch * world,
              "kl_start": round(sum(head) / len(head), 4), "kl_end": round(sum(tail) / len(tail), 4),
              "minutes": round((time.time() - started) / 60, 1)}
    print(f"on-policy distillation: KL {report['kl_start']} -> {report['kl_end']} in {report['minutes']} min")
    return report


def train(args: argparse.Namespace, preset: dict[str, Any], rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> int:
    rank, world = dist_info()
    if world == 1 and not args.split:
        # one process = one T4 (the notebook gives each parallel edition its own); DDP uses one process per T4
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
    if world > 1:
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        load["device_map"] = {"": local}
        print(f"DDP: {world} GPUs train this edition together (this is GPU {local})")
    elif args.split:
        load["device_map"] = "unsloth_balanced"
    model, tokenizer = FastLanguageModel.from_pretrained(**load)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=args.seed,
    )
    if preset["chat_template"]:
        tokenizer = get_chat_template(tokenizer, chat_template=preset["chat_template"])

    trained = rows
    if getattr(args, "merge_qa", 0) > 1 and not preset.get("reasoning"):
        trained = merge_qa(rows, args.merge_qa)
        chats = [r for r in trained if r.get("merged", 0) > 1]
        if chats:
            print(f"Q&A: {sum(r['merged'] for r in chats)} rows in {len(chats)} chats of up to {args.merge_qa} questions "
                  f"(the system prompt read once per chat, not per question) - {len(rows)} -> {len(trained)} rows")
    texts = [render(tokenizer, prepare_messages(row, preset), preset) for row in trained]
    dataset = Dataset.from_dict({"text": texts})
    user, assistant = TURN_MARKERS[preset["family"]]
    callbacks = training_callbacks(args)
    plan = plan_batches(token_lengths(tokenizer, texts) if args.token_budget > 0 else [], preset, world,
                        args.max_seq, args.token_budget)
    if plan["p95"]:
        print(f"batches: {plan['batch']} rows per GPU x {plan['accum']} accumulation x {world} GPU(s) "
              f"(rows: ~{plan['mean']} tokens, 95% <= {plan['p95']}; same {preset['batch'] * preset['accum']} rows per "
              f"step as before, in fewer passes)")
    settings = {"group": True}

    def build() -> Any:
        options = speed_options(args, preset, world, text=True, plan=plan)
        if not settings["group"]:
            options.pop("group_by_length", None)
        config = SFTConfig(
            dataset_text_field="text", max_length=args.max_seq, per_device_train_batch_size=plan["batch"],
            warmup_steps=10, num_train_epochs=args.epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1, learning_rate=args.lr, logging_steps=10,
            optim="adamw_8bit", weight_decay=0.001, lr_scheduler_type="linear", seed=args.seed,
            output_dir=str(scratch_for(out, "checkpoints")), save_strategy="no", report_to="none", **options,
        )
        built = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=config,
                           callbacks=callbacks)
        return train_on_responses_only(built, instruction_part=user, response_part=assistant)

    def rebuild(exc: BaseException) -> Any:
        if is_oom(exc) and plan["batch"] > 1:
            plan["batch"] = max(1, plan["batch"] // 2)
            plan["accum"] = max(1, round(preset["batch"] * preset["accum"] / (plan["batch"] * world)))
            print(f"out of GPU memory - trying {plan['batch']} rows per batch x {plan['accum']} accumulation")
            torch.cuda.empty_cache()
            return build()
        if settings["group"] and getattr(args, "group_by_length", False) and not is_oom(exc):
            settings["group"] = False
            print(f"length-grouped batches failed ({type(exc).__name__}: {exc}) - training without them")
            return build()
        return None

    stats = run_training(build(), rebuild, world)
    print(f"trained in {(time.time() - started) / 60:.0f} min, loss {stats.training_loss:.3f}")
    if getattr(args, "onpolicy_steps", 0) > 0:
        try:
            args.onpolicy_report = onpolicy_distill(args, preset, model, tokenizer, rows, world, rank)
        except Exception as exc:  # noqa: BLE001 - the fine-tuned model is kept as it is
            print(f"on-policy distillation failed ({type(exc).__name__}: {exc}) - keeping the fine-tuned model")
        free_gpu()  # whatever the teacher left behind goes before the eval and the export
    if world > 1:
        end_ddp()
        if rank > 0:
            return 0  # rank 0 evaluates, exports and uploads

    return finish(args, preset, model, tokenizer, rows, eval_rows, stats.training_loss, started,
                  text_generator(model, tokenizer, preset), lambda: FastLanguageModel.for_inference(model))


def text_generator(model: Any, tokenizer: Any, preset: dict[str, Any], batch: int = 16) -> Any:
    def generate(row: dict[str, Any]) -> str:
        prompt = render(tokenizer, row["messages"][:-1], preset, generation=True)
        inputs = tokenizer(text=prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        output = model.generate(**inputs, max_new_tokens=preset.get("max_new_tokens", 256), temperature=0.3, top_p=0.8,
                                top_k=20, do_sample=True)
        return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def many(rows: list[dict[str, Any]]) -> list[str]:
        """``batch`` prompts per generate call (left-padded): the eval and the tests take a fraction of the time."""
        side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        replies: list[str] = []
        size, start = batch, 0
        try:
            while start < len(rows):
                chunk = rows[start:start + size]
                prompts = [render(tokenizer, row["messages"][:-1], preset, generation=True) for row in chunk]
                try:
                    inputs = tokenizer(text=prompts, return_tensors="pt", padding=True,
                                       add_special_tokens=False).to(model.device)
                    output = model.generate(**inputs, max_new_tokens=preset.get("max_new_tokens", 256),
                                            temperature=0.3, top_p=0.8, top_k=20, do_sample=True, pad_token_id=pad)
                except Exception as exc:  # noqa: BLE001 - out of memory: half the batch, not one at a time
                    if not is_oom(exc) or size == 1:
                        raise
                    inputs = output = None
                    free_gpu()
                    size = max(1, size // 2)
                    print(f"answers: out of GPU memory - {size} at a time")
                    continue
                width = inputs["input_ids"].shape[1]
                replies += [tokenizer.decode(sequence[width:], skip_special_tokens=True) for sequence in output]
                start += len(chunk)
        finally:
            tokenizer.padding_side = side
        return replies

    generate.many = many  # type: ignore[attr-defined]
    return generate


def answer_all(generate: Any, rows: list[dict[str, Any]]) -> list[str]:
    """Replies to every row: batched when the generator can (``generate.many``), else one by one. A failing prompt
    gives an ``<error: ...>`` reply instead of losing the trained model."""
    many = getattr(generate, "many", None)
    if many is not None and len(rows) > 1:
        try:
            replies = list(many(rows))
            if len(replies) == len(rows):
                return replies
        except Exception as exc:  # noqa: BLE001 - one at a time still works
            print(f"batched answers failed ({type(exc).__name__}: {exc}) - one at a time")
    out = []
    for row in rows:
        try:
            out.append(generate(row))
        except Exception as exc:  # noqa: BLE001
            out.append(f"<error: {type(exc).__name__}: {exc}>")
    return out


def vision_generator(model: Any, processor: Any, preset: dict[str, Any], load_image: Any) -> Any:
    def generate(row: dict[str, Any]) -> str:
        messages = vision_messages({**row, "reasoning": ""}, {**preset, "reasoning": False}, load_image)[:-1]
        images = [part["image"] for m in messages for part in m["content"] if part.get("type") == "image"]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=images or None, return_tensors="pt", add_special_tokens=False).to(model.device)
        output = model.generate(**inputs, max_new_tokens=preset.get("max_new_tokens", 256), temperature=0.3, top_p=0.8,
                                top_k=20, do_sample=True)
        return processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def test_only(args: argparse.Namespace, preset: dict[str, Any], eval_rows: list[dict[str, Any]]) -> int:
    """``--test-only``: load the LoRA from an earlier run (``<out>/lora``) and run the test suite."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    out = Path(args.out)
    adapter = out / "lora"
    if not adapter.is_dir():
        print(f"no trained adapter at {adapter} - train first (or keep the lora folder: CLEAN_AFTER_UPLOAD = False)")
        return 2
    rows = test_rows(eval_rows, preset, args.system_prompt)
    if preset.get("vision"):
        from unsloth import FastVisionModel  # noqa: I001

        model, processor = FastVisionModel.from_pretrained(str(adapter), load_in_4bit=True)
        FastVisionModel.for_inference(model)
        generate = vision_generator(model, processor, preset, load_image_from(Path(args.data_dir), args.image_side))
    else:
        from unsloth import FastLanguageModel  # noqa: I001
        from unsloth.chat_templates import get_chat_template

        model, tokenizer = FastLanguageModel.from_pretrained(model_name=str(adapter), max_seq_length=args.max_seq,
                                                             load_in_4bit=True)
        if preset["chat_template"]:
            tokenizer = get_chat_template(tokenizer, chat_template=preset["chat_template"])
        FastLanguageModel.for_inference(model)
        generate = text_generator(model, tokenizer, preset)
    summary = run_tests(generate, rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tests.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


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
    rank, world = dist_info()
    placement: dict[str, Any] = {}
    if world > 1:
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        placement["device_map"] = {"": local}
        print(f"DDP: {world} GPUs train this edition together (this is GPU {local})")
    model, processor = FastVisionModel.from_pretrained(model_name, load_in_4bit=preset["load_in_4bit"],
                                                       use_gradient_checkpointing="unsloth", **placement)
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
        per_device_train_batch_size=preset["batch"], warmup_steps=10,
        num_train_epochs=args.epochs, max_steps=args.max_steps if args.max_steps > 0 else -1, learning_rate=args.lr,
        logging_steps=10, optim="adamw_8bit", weight_decay=0.001, lr_scheduler_type="linear", seed=args.seed,
        output_dir=str(scratch_for(out, "checkpoints")), save_strategy="no", report_to="none", fp16=not bf16, bf16=bf16,
        remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_seq, **speed_options(args, preset, world, text=False),
    )
    callbacks = training_callbacks(args)
    trainer = SFTTrainer(model=model, tokenizer=processor, data_collator=collator, train_dataset=dataset, args=config,
                         callbacks=callbacks)
    stats = trainer.train()
    print(f"trained in {(time.time() - started) / 60:.0f} min, loss {stats.training_loss:.3f}")
    if world > 1:
        end_ddp()
        if rank > 0:
            return 0  # rank 0 evaluates, exports and uploads

    return finish(args, preset, model, processor, rows, eval_rows, stats.training_loss, started,
                  vision_generator(model, processor, preset, load_image), lambda: FastVisionModel.for_inference(model))


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
                              "max_seq": args.max_seq,
                              "images": sum(len(row.get("images") or []) for row in rows), "loss": round(loss, 4),
                              "uncensored": bool(preset.get("uncensored")), "system_prompt": args.system_prompt,
                              "general_rows": sum(1 for row in rows if row.get("kind") == "general"),
                              "commercial": bool(getattr(args, "commercial", False)), "data": data_sources(rows, args)}
    distill = {key: value for key, value in (("offpolicy", getattr(args, "distill_report", None)),
                                             ("onpolicy", getattr(args, "onpolicy_report", None))) if value}
    if distill:
        report["distill"] = dict(distill, teacher=(distill.get("offpolicy") or {}).get("teacher")
                                 or teacher_model(args, "answers"))
        if distill.get("onpolicy"):
            report["distill"]["grader"] = distill["onpolicy"].get("teacher") or teacher_model(args, "grade")
    if eval_rows and args.eval_limit:
        for_inference()
        asked = eval_rows[: args.eval_limit]
        results = [score(row, reply) for row, reply in zip(asked, answer_all(generate, asked), strict=False)]
        report["eval"] = summarize(results)
        print("eval:", json.dumps(report["eval"]))
    if args.test:
        for_inference()
        print("\n==== test suite ====")
        summary = run_tests(generate, test_rows(eval_rows, preset, args.system_prompt))
        (out / "tests.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        report["tests"] = {key: value for key, value in summary.items() if key != "results"}
    if getattr(args, "tool_eval", None):
        for_inference()
        try:
            report["agentic"] = agentic_eval(generate, args.tool_eval, model)
        except Exception as exc:  # noqa: BLE001 - never lose the trained model to the eval
            print(f"agentic eval failed ({type(exc).__name__}: {exc})")

    if getattr(args, "stopped_early", None):
        report["stopped_early"] = args.stopped_early
    lock = getattr(args, "export_lock", None)
    with export_lock(f"{lock}.merge" if lock else None, waiting="waiting for the other edition's merge (RAM)..."):
        save_safetensors(args, model, tokenizer, out, report)
    if defer_export(args, preset, out, report, started):
        return 0
    with export_lock(lock):
        return export_and_publish(args, preset, model, tokenizer, out, adapter, report, eval_rows, started)


#: Written by ``--defer-gguf``: everything the CPU-only ``--export-only`` process needs.
EXPORT_JOB = "export_job.json"


def save_safetensors(args: argparse.Namespace, model: Any, tokenizer: Any, out: Path, report: dict[str, Any]) -> None:
    """The safetensors edition (needs the trained model, so it runs in the training process): the LoRA merged into
    the base in 16-bit, compacted to the base model's size."""
    if getattr(args, "safetensors", "none") == "none":
        return
    st_dir = Path(out) / SAFETENSORS_DIR
    print("saving the safetensors edition (LoRA merged into the base, 16-bit) - a few minutes")
    try:
        report["safetensors_gb"] = export_safetensors(model, tokenizer, st_dir)
        report["safetensors"] = SAFETENSORS_DIR
        if getattr(args, "compact", True):
            try:
                stats = compact_safetensors(st_dir)
                report.update(compact=stats, params_b=stats["params_b"], safetensors_gb=stats["gb"])
            except Exception as exc:  # the weights as saved are still a valid model
                print(f"compacting skipped ({type(exc).__name__}: {exc}) - the weights stay as saved")
        print(f"safetensors: {report['safetensors_gb']} GB in {st_dir}")
    except Exception as exc:  # the adapter is safe; the GGUF falls back to Unsloth's exporter
        print(f"safetensors export failed ({type(exc).__name__}: {exc}) - the LoRA adapter is fine")
        for key in ("safetensors", "safetensors_gb", "compact", "params_b"):
            report.pop(key, None)
        shutil.rmtree(st_dir, ignore_errors=True)


def defer_export(args: argparse.Namespace, preset: dict[str, Any], out: Path, report: dict[str, Any],
                 started: float) -> bool:
    """``--defer-gguf``: hand the GGUF quants and the uploads (CPU and network only) to an ``--export-only``
    process, so this process ends and the next edition gets the GPU ~20-40 minutes earlier. Only when llama.cpp
    is ready - Unsloth's fallback exporter needs the model, which is gone afterwards."""
    if not getattr(args, "defer_gguf", False) or args.test_gguf or not gguf_quants(args) or not report.get("safetensors"):
        return False
    try:
        llama_cpp_tools(tools_dir(args, out))
    except Exception as exc:  # noqa: BLE001 - then the GGUFs are made here, with the fallback available
        print(f"GGUFs are made in this process (llama.cpp is not ready: {type(exc).__name__}: {exc})")
        return False
    job = {"args": vars(args), "preset": preset, "report": report, "started": started}
    (Path(out) / EXPORT_JOB).write_text(json.dumps(job, default=str, indent=1), encoding="utf-8")
    print(f"GPU work done in {(time.time() - started) / 60:.0f} min - the GGUF quants and uploads continue in a "
          "CPU-only process (--export-only), so the next edition can start on this GPU now", flush=True)
    return True


def export_only(folder: str) -> int:
    """Finish an edition trained with ``--defer-gguf``: GGUFs from its safetensors, uploads, model card."""
    out = Path(folder)
    job_file = out / EXPORT_JOB
    if not job_file.is_file():
        print(f"{job_file} is missing - train with --defer-gguf first")
        return 2
    job = json.loads(job_file.read_text(encoding="utf-8"))
    args = build_parser().parse_args([])
    for key, value in job["args"].items():
        setattr(args, key, value)
    args.export_only = None
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # never touch the GPU the next edition trains on
    with export_lock(getattr(args, "export_lock", None)):
        code = export_and_publish(args, job["preset"], None, None, out, out / "lora", job["report"], [], job["started"])
    if code == 0:
        job_file.unlink()
    return code


def export_and_publish(args: argparse.Namespace, preset: dict[str, Any], model: Any, tokenizer: Any, out: Path,
                       adapter: Path, report: dict[str, Any], eval_rows: list[dict[str, Any]], started: float) -> int:
    """safetensors upload (in the background) -> GGUF quants made from them meanwhile, each uploaded in turn ->
    GGUF repo; under ``--export-lock`` when two editions train side by side. ``model`` is only needed for
    Unsloth's fallback exporter (``None`` in the CPU-only ``--export-only`` process)."""
    hub = connect_hub(args, preset) if args.hub_repo else None
    repo = (hub or {}).get("gguf") or (args.hub_repo if args.hub_repo and "/" in args.hub_repo
                                         else f"<your-hf-name>/{preset.get('hub') or 'pymc-minecraft-GGUF'}")
    if hub:
        report["hub_repo"] = hub["gguf"]
    report.update({"family": preset["family"], "compact_tag": COMPACT_TAG, "compact_prompt": True,
                   "run_parameters": preset.get("run_parameters") or RUN_PARAMETERS})
    st_dir = out / SAFETENSORS_DIR
    uploader = Uploader(background=bool(hub))
    # 1. the safetensors edition (merged by save_safetensors): uploaded in the background, first in the queue
    if hub and report.get("safetensors"):
        report["hub_repo_safetensors"] = hub["safetensors"]
        write_cards(out, args.preset, preset, report, repo)

        def upload_safetensors() -> None:
            try:
                publish_safetensors(st_dir, hub["safetensors"], args.preset, hub["token"], private=args.hub_private)
                report["safetensors_uploaded"] = True
                print(f"uploaded the safetensors edition -> https://huggingface.co/{hub['safetensors']}", flush=True)
            except Exception as exc:  # keep the folder for another try; the GGUFs still get made
                print(f"safetensors upload failed ({type(exc).__name__}: {exc}) - the files are still in {st_dir}",
                      flush=True)

        uploader.submit(upload_safetensors)
    # tokenizer.json, configs and chat template next to the GGUFs too (before the safetensors may be deleted)
    report["essentials"] = copy_essentials(st_dir if st_dir.is_dir() else adapter, out)

    def free_safetensors() -> None:  # queued behind their upload, after the F16 GGUF exists: frees ~8 GB early
        if getattr(args, "clean_after_upload", False) and report.get("safetensors_uploaded") and st_dir.is_dir():
            shutil.rmtree(st_dir, ignore_errors=True)
            print("the safetensors are on Hugging Face and in the F16 GGUF - deleted locally (~8 GB freed)", flush=True)

    # 2. the GGUFs, made from those safetensors while they upload: every quant, each uploaded as soon as it is ready
    quants = gguf_quants(args)
    if quants:
        made = export_ggufs(args, preset, model, tokenizer, out, st_dir if report.get("safetensors") else None,
                            quants, hub=hub, uploader=uploader, on_base=lambda: uploader.submit(free_safetensors))
        uploader.wait()
        if made is None:
            print("GGUF export produced no file - the LoRA adapter is in", adapter)
            return 1
        gguf = out / made["files"][0]["file"]
        mmproj = out / made["mmproj"] if made.get("mmproj") else None
        if preset.get("vision") and mmproj is None:
            print("warning: no mmproj (vision projector) was exported - the model will only read text")
        (out / "Modelfile").write_text(modelfile_text(gguf.name, preset["family"], made.get("template"),
                                                      mmproj.name if mmproj else None,
                                                      preset.get("run_parameters"), args.system_prompt), encoding="utf-8")
        report.update({"gguf": gguf.name, "quant": made["files"][0]["quant"], "gguf_route": made.get("route"),
                       "gguf_gb": round(gguf.stat().st_size / 1e9, 2), "ggufs": made["files"],
                       "gguf_skipped": made.get("skipped") or [], "imatrix": made.get("imatrix"),
                       "llama_cpp": made.get("llama_cpp")})
        if mmproj:
            report["mmproj"] = mmproj.name
        if args.test_gguf:
            print("\n==== GGUF test (Ollama) ====")
            picture = next((Path(args.data_dir) / row["images"][0] for row in eval_rows if row.get("images")), None)
            prompts = [p for _kind, p in TEST_PROMPTS[:2]] + [TEST_PROMPTS[5][1]]
            if preset.get("vision") and picture is not None:
                prompts.append("What do you see in this Minecraft screenshot?")
            report["gguf_test"] = test_gguf(out, preset, prompts, picture)
    uploader.wait()
    shutil.rmtree(out / "checkpoints", ignore_errors=True)
    if scratch_for(out, "checkpoints") != out:
        shutil.rmtree(scratch_for(out, "checkpoints"), ignore_errors=True)
    report.update({"minutes": round((time.time() - started) / 60, 1), "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    write_cards(out, args.preset, preset, report, repo)
    # 4. the GGUF repo: the main quant, mmproj, Modelfile, model card, report and the essentials
    if hub and quants:
        try:
            publish(out, hub["gguf"], args.preset, hub["token"], private=args.hub_private, include_lora=args.hub_lora)
            print(f"uploaded the GGUF edition -> https://huggingface.co/{hub['gguf']}")
        except Exception as exc:  # never lose a finished model to an upload problem
            report.pop("hub_repo", None)
            print(f"Hugging Face upload failed ({type(exc).__name__}: {exc}) - the files are still in {out}")
        write_cards(out, args.preset, preset, report, report.get("hub_repo") or repo)
    elif hub:
        report.pop("hub_repo", None)
        write_cards(out, args.preset, preset, report, repo)
    shutil.rmtree(out / "stream_images", ignore_errors=True)  # streamed training pictures are not needed any more
    if scratch_for(out, "stream_images") != out:
        shutil.rmtree(scratch_for(out, "stream_images"), ignore_errors=True)
    if getattr(args, "clean_after_upload", False):
        removed = clean_uploaded(out, report)
        if removed:
            print(f"uploaded, so deleted locally: {', '.join(removed)}")
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
    parser.add_argument("--stop-when-learned", action="store_true",
                        help="end an edition early once its training loss stops falling (saves hours of GPU time)")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = full epochs)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--max-seq", type=int, default=3072, help="longest example in tokens")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval-limit", type=int, default=150, help="eval rows to try after training (0 = skip)")
    parser.add_argument("--gguf", default="q4_k_m", help="GGUF quantisation (q4_k_m, q5_k_m, q8_0) or 'none'")
    parser.add_argument("--gguf-quants", default=None,
                        help="more GGUFs made from the safetensors and uploaded one by one: 'all' (every llama.cpp "
                             "quant, F16 and BF16 to IQ1_S; the default when uploading), 'standard' (q8_0,q6_k,"
                             "q5_k_m,q4_k_m,q3_k_m) or a list like q8_0,iq4_xs ('none' = only --gguf)")
    parser.add_argument("--imatrix-chunks", type=int, default=IMATRIX_CHUNKS,
                        help=f"512-token chunks of the training data the importance matrix is computed on "
                             f"(default {IMATRIX_CHUNKS}; ~10 min on a CPU for 4B)")
    parser.add_argument("--gguf-scratch", default=None,
                        help="work folder for the F16 GGUF and the quants (default: /tmp/pymc-gguf, off Kaggle's "
                             "20 GB output disk)")
    parser.add_argument("--no-compact", dest="compact", action="store_false",
                        help="keep the merged safetensors as saved (default: float32 -> 16-bit and the duplicate "
                             "tied lm_head dropped, so a 4B model is 4B-sized)")
    parser.add_argument("--safetensors", choices=("16bit", "none"), default="16bit",
                        help="also save (and with --hub-repo upload, to <repo without -GGUF>) the full-precision "
                             "safetensors edition: the LoRA merged into the base in 16-bit (~8 GB for 4B)")
    parser.add_argument("--split", action="store_true", help="spread the model over both GPUs (only needed for 14B)")
    parser.add_argument("--deadline", type=float, default=0,
                        help="unix time the session ends; training stops early so the export + upload still fit")
    parser.add_argument("--reserve-minutes", type=float, default=60,
                        help="minutes kept before --deadline for eval, tests, export and upload (default 60)")
    parser.add_argument("--export-lock", default=None,
                        help="lock file: only one edition exports / uploads at a time (two editions on two T4s)")
    parser.add_argument("--clean-after-upload", action="store_true",
                        help="delete the GGUFs, adapter and safetensors once they are on Hugging Face")
    parser.add_argument("--group-by-length", action="store_true",
                        help="text editions: batches of similar length (less padding, faster)")
    parser.add_argument("--loader-workers", type=int, default=0, help="data-loader worker processes per GPU")
    parser.add_argument("--merge-qa", type=int, default=0,
                        help="text editions that don't think: Q&A rows with the same system prompt trained as chats of "
                             "this many questions (the ~300-token system prompt read once per chat, every answer still "
                             "trained; ~2.5x fewer Q&A tokens); 0/1 = off")
    parser.add_argument("--token-budget", type=int, default=0,
                        help="text editions: rows per GPU batch sized so a batch holds about this many tokens (padding-"
                             "free, same effective batch, fewer accumulation steps; 0 = the preset's batch)")
    parser.add_argument("--stream-cache", default=None,
                        help="folder shared by the editions of a session: streamed chat / Q&A / pictures are "
                             "downloaded once and reused by the next edition")
    parser.add_argument("--defer-gguf", action="store_true",
                        help="after the safetensors are saved, leave the GGUF quants + uploads to a CPU-only "
                             "'--export-only <out>' process so the next edition can use this GPU right away")
    parser.add_argument("--export-only", metavar="OUT", default=None,
                        help="make the GGUFs and upload an edition that was trained with --defer-gguf (no GPU)")
    parser.add_argument("--hub-repo", help="upload to Hugging Face (needs HF_TOKEN): 'auto' = <you>/<edition repo>, "
                                           "a repo name, or user/repo")
    parser.add_argument("--hub-private", action="store_true", help="create the Hugging Face repo as private")
    parser.add_argument("--hub-lora", action="store_true", help="also upload the LoRA adapter folder")
    parser.add_argument("--image-side", type=int, default=IMAGE_SIDE, help="vision editions: longest side of a picture")
    parser.add_argument("--vision-layers", type=int, default=0, choices=[0, 1],
                        help="vision editions: 1 = also fine-tune the image encoder (slower, more memory)")
    parser.add_argument("--distill-rows", type=int, default=0,
                        help="distillation phase 1 (off-policy, Qwen3 strong-to-weak): a bigger Qwen3 teacher answers "
                             "up to N chat and Minecraft prompts and the student learns its answers (0 = off)")
    parser.add_argument("--onpolicy-steps", type=int, default=0,
                        help="distillation phase 2 (on-policy, text editions): after fine-tuning, the student writes "
                             "replies and learns the teacher's token probabilities on them (reverse KL); 0 = off")
    parser.add_argument("--teacher", default="auto",
                        help=f"teacher model: auto = {WRITER_MODEL} writes the answers on vLLM and {TEACHER_MODEL} "
                             f"grades (both {UNCENSORED_TEACHER} with --uncensored); any Qwen3 with the same "
                             "tokenizer works for both phases, e.g. unsloth/Qwen3-14B")
    parser.add_argument("--distill-minutes", type=float, default=40.0,
                        help="most minutes the teacher may spend writing answers (unwritten rows keep theirs)")
    parser.add_argument("--onpolicy-batch", type=int, default=4, help="replies per GPU per on-policy step")
    parser.add_argument("--onpolicy-tokens", type=int, default=256, help="longest student reply in the on-policy phase")
    parser.add_argument("--onpolicy-lr", type=float, default=5e-5, help="learning rate of the on-policy phase")
    parser.add_argument("--onpolicy-refresh", type=int, default=1,
                        help="the student writes the replies of N on-policy steps in one batched call (faster; "
                             "1 = strictly on-policy)")
    parser.add_argument("--teacher-engine", choices=("auto", "vllm", "hf"), default="auto",
                        help="who runs the teacher: vLLM (needs --vllm-python) or transformers; auto = vLLM if given")
    parser.add_argument("--vllm-python", default=None,
                        help=f"python of a venv with vllm=={VLLM_VERSION}: the teacher writes its answers with vLLM "
                             "(several times faster, both T4s), falling back to transformers")
    parser.add_argument("--teacher-cache-repo", default="",
                        help="keep the teacher's answers in this private Hugging Face dataset so later sessions reuse "
                             f"them (needs HF_TOKEN; auto = <you>/{TEACHER_CACHE_REPO})")
    parser.add_argument("--tools-dir", default=None,
                        help="where llama.cpp is downloaded and kept for the next editions (default: next to the "
                             "GGUF scratch folder)")
    parser.add_argument("--vllm-worker", nargs=2, metavar=("JOB", "OUT"), help=argparse.SUPPRESS)
    parser.add_argument("--uncensored", action="store_true",
                        help="train on the abliterated base (no refusals) and drop refusals from the extra chat data")
    parser.add_argument("--system-prompt-file", help="the model's default system prompt (default: the built-in "
                                                     "Minecraftism prompt); written into the Modelfile")
    parser.add_argument("--persona-mix", type=float, default=0.5,
                        help="share of Minecraft Q&A rows that get the everyday system prompt (0-1)")
    parser.add_argument("--general", type=int, default=0,
                        help="everyday chat rows to mix in from Hugging Face (keeps it a good daily assistant)")
    parser.add_argument("--general-dataset", help=f"ShareGPT/messages/ChatML chat dataset, 'name:split' picks a split "
                                                  f"(default {GENERAL_DATASET}; --commercial: {COMMERCIAL_GENERAL_DATASET})")
    parser.add_argument("--reasoning-dataset",
                        help="dataset with <think> traces used instead for the thinking editions ('name:split'; "
                             f"default {REASONING_DATASET}; --commercial: {COMMERCIAL_REASONING_DATASET})")
    parser.add_argument("--no-identity", action="store_true", help="skip the 'who are you' rows")
    parser.add_argument("--tools", type=int, default=0,
                        help="tool-calling (agentic) chats to mix in, in Qwen3's own tool format; the agentic score "
                             f"(base vs fine-tuned, {TOOL_EVAL} held-out chats) goes into the report and the card")
    parser.add_argument("--tool-dataset", default=TOOL_DATASET, help="'name:split' with messages/tool_calls and "
                                                                     "metadata.tools (default %(default)s, CC BY 4.0)")
    parser.add_argument("--minecraft-qa", type=int, default=0,
                        help="Minecraft Q&A rows streamed from a big Hugging Face Q&A set (sampled and filtered)")
    parser.add_argument("--minecraft-qa-dataset",
                        help=f"Hugging Face Q&A dataset, or '{MINECRAFT_DATA}' to generate exact Q&A from "
                             f"PrismarineJS/minecraft-data (default {MINECRAFT_QA_DATASET}; --commercial: {MINECRAFT_DATA})")
    parser.add_argument("--minecraft-data-version", default=MINECRAFT_DATA_VERSION,
                        help="Java version of the minecraft-data facts")
    parser.add_argument("--minecraft-data-dir", help="a local minecraft-data checkout instead of downloading")
    parser.add_argument("--commercial", action="store_true",
                        help="only data whose licence allows commercial use: leaves out the Minecraft Wiki rows and "
                             "swaps the default datasets for OpenAssistant chat, NVIDIA Nemotron maths traces, "
                             "minecraft-data facts and minecraft-raytrace pictures")
    parser.add_argument("--stream-vision", type=int, default=0,
                        help="VL editions: Minecraft screenshots with captions / Q&A streamed from Hugging Face "
                             "(only the sampled pictures are downloaded); also lets VL train without own screenshots")
    parser.add_argument("--vision-dataset", help=f"default {VISION_STREAM_DATASET}; --commercial: "
                                                 f"{COMMERCIAL_VISION_DATASET}")
    parser.add_argument("--test", action="store_true", help="run the test suite after training (bot JSON, Minecraft, "
                                                           "daily life, refusals, vision)")
    parser.add_argument("--test-gguf", action="store_true", help="also chat with the exported GGUF through Ollama")
    parser.add_argument("--test-only", action="store_true", help="only run the test suite on <out>/lora from an earlier run")
    parser.add_argument("--dry-run", action="store_true", help="check the data and show the plan without a GPU")
    parser.add_argument("--list-presets", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.vllm_worker:  # the teacher's answers, inside the vLLM venv
        return vllm_worker(*args.vllm_worker)
    if args.export_only:
        return export_only(args.export_only)
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
    if args.uncensored:
        preset = uncensored_preset(args.preset, preset)
        if not PRESETS[args.preset].get("uncensored_model"):
            print(f"note: {args.preset} has no abliterated base; --uncensored only filters refusals from the chat data")
    args.system_prompt = (Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
                          if args.system_prompt_file else MINECRAFTISM_SYSTEM)
    if not args.out:
        args.out = str(Path("/kaggle/working") / args.preset) if Path("/kaggle/working").is_dir() else args.preset
    rank, world = dist_info()
    if world > 1 and rank > 0:  # DDP: the other T4 trains on the rows rank 0 prepares (and prints nothing twice)
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115 - for the life of the process
        shared = wait_for_rows(Path(args.out), on_job=lambda: serve_distill_job(Path(args.out), rank, world))
        if shared is None:
            return 0
        args.data_dir = shared["data_dir"]
        return train(args, preset, shared["rows"], shared["eval_rows"])
    if world == 1:
        return prepare_and_train(args, preset)
    try:
        return prepare_and_train(args, preset, share=True)
    except BaseException:
        share_rows(Path(args.out), None)  # never leave the other GPU waiting
        raise


def prepare_and_train(args: argparse.Namespace, preset: dict[str, Any], share: bool = False) -> int:
    """Read + check the data, add the streamed rows, then train (``share``: hand the rows to the DDP partner)."""
    def stop(code: int) -> int:
        if share:
            share_rows(Path(args.out), None)
        return code

    resolve_datasets(args)
    if args.commercial:
        problems, warnings = commercial_problems(args, preset)
        for warning in warnings:
            print("note:", warning)
        if problems:
            print("--commercial:\n" + "\n".join(problems))
            return stop(2)
    data = find_data(args.data)
    args.data_dir = str(data.parent)
    rows = select_rows(read_jsonl(data), preset)
    eval_path = Path(args.eval_data) if args.eval_data else data.with_name("eval.jsonl")
    eval_rows = select_rows(read_jsonl(eval_path), preset) if eval_path.is_file() else []
    if args.commercial:
        rows, dropped = drop_noncommercial(rows)
        eval_rows, _ = drop_noncommercial(eval_rows)
        for source, count in dropped.items():
            print(f"commercial: left out {count} rows of {NONCOMMERCIAL_SOURCES[source]}")
        teacher = sum(1 for row in rows if row.get("source") == "teacher")
        if teacher:
            print(f"note: {teacher} teacher rows kept - their licence follows the teacher model you used")
    missing = [name for row in rows if preset.get("vision") for name in row.get("images") or []
               if not (data.parent / name).is_file()]
    if missing:
        print(f"{len(missing)} picture(s) listed in the data are missing next to train.jsonl (first: {missing[0]}) - "
              "upload the images/ folder with the dataset")
        return stop(2)
    if preset.get("vision") and not any(row.get("images") for row in rows) and args.stream_vision <= 0:
        print(f"{args.preset} is a vision edition but the data has no pictures. Record playtime with the mod's "
              "screenshots on (/pymcbot screenshots on), then: python -m pymc_bot llm build --playtime <folder> "
              "--screenshots - or stream pictures from Hugging Face (--stream-vision 3000) - or train a text edition "
              "(minecraftism, minecraftism-thinking).")
        return stop(2)
    problems = check_rows(rows) + check_rows(eval_rows)
    if problems:
        print("\n".join(problems[:20]))
        return stop(2)
    if args.test_only:
        return stop(test_only(args, preset, eval_rows))
    import random

    rng = random.Random(args.seed)
    persona = apply_persona(rows, args.system_prompt, max(0.0, min(1.0, args.persona_mix)), rng) if args.preset in EDITIONS else 0
    extra = extra_rows(args, preset, rng, offline=args.dry_run)
    streamed = [row for row in extra if row.get("source") in ("minecraft-vlp", "minecraft-raytrace")]
    if len(streamed) > 20 and not any(row.get("images") for row in eval_rows):
        pictures: list[str] = []  # a few unseen pictures (with all their questions) for the eval and the tests
        for row in reversed(streamed):
            if row["images"][0] not in pictures:
                pictures.append(row["images"][0])
            if len(pictures) == 3:
                break
        held = [row for row in streamed if row["images"][0] in pictures]
        extra = [row for row in extra if all(row is not h for h in held)]
        eval_rows = eval_rows + held
    rows = rows + extra
    rng.shuffle(rows)
    kinds: dict[str, int] = {}
    for row in extra:
        kind = {"identity": "identity", "persona": "identity", "general": "everyday chat", "reasoning": "thinking traces",
                "minecraft-data": "Minecraft Q&A", "minecraft-qa": "Minecraft Q&A", "minecraft-vlp": "pictures",
                "minecraft-raytrace": "pictures"}.get(str(row.get("source")), str(row.get("source") or "other"))
        kinds[kind] = kinds.get(kind, 0) + 1
    print(f"persona: {persona} Q&A rows use the everyday system prompt; +{len(extra)} rows "
          f"({', '.join(f'{n} {k}' for k, n in kinds.items()) or 'none'}){' (uncensored)' if args.uncensored else ''}")
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
    editions = ([f"GGUF {args.gguf.upper()}"] if args.gguf != "none" else []) + (
        ["safetensors 16-bit"] if args.safetensors != "none" else [])
    print(f"exports: {', '.join(editions) or 'only the LoRA adapter'}")
    if args.dry_run:
        distill_offpolicy(args, preset, rows, dist_info()[1])  # prints the plan only
        sample = prepare_messages(rows[0], preset)[-1]["content"]
        print("dry run: data is fine. Sample reply:", sample[:400])
        return stop(0)
    if args.deadline and time.time() > args.deadline - (args.reserve_minutes + 20) * 60:
        print(f"not enough session time left for {args.preset} (training + {args.reserve_minutes:.0f} min export) - "
              "train it in the next session")
        return stop(3)
    if getattr(args, "distill_rows", 0) > 0:
        rows = distill_offpolicy(args, preset, rows, dist_info()[1])
    if share:
        share_rows(Path(args.out), rows, eval_rows, args.data_dir)
    return train(args, preset, rows, eval_rows)


if __name__ == "__main__":
    sys.exit(main())
