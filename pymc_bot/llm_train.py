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
    reasoning = str(reasoning or "").strip()
    if "\n" not in reasoning:
        reasoning = " ".join(reasoning.split())
    reasoning = reasoning or "Simple one, answer directly."
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
                     "reasoning": _plain(f"A Minecraft question: {question[:240]} I recall what the wiki says about "
                                         "it and answer directly, with the numbers that matter."),
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
        used.append(args.reasoning_dataset if preset.get("reasoning") else args.general_dataset)
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
        recipe = variants[0] if isinstance(variants, list) and variants else None
        if item is None or not isinstance(recipe, dict):
            continue
        text = _recipe_text(recipe, by_id, label)
        if not text:
            continue
        name, count = label(item), (recipe.get("result") or {}).get("count", 1)
        more = f" This is one of {len(variants)} recipe variants." if len(variants) > 1 else ""
        facts.append(((f"How do I craft {name}?", f"{name.lower()} recipe", f"what do i need to make {name.lower()}"),
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


def _recipe_text(recipe: dict[str, Any], by_id: dict[Any, dict[str, Any]], label: Any) -> str:
    """'Shapeless: 1 Bone' / 'Crafting table, 3 rows: ...' - empty when an ingredient is not a plain item."""
    def name(value: Any) -> str | None:
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
    for row in shape:
        cells = []
        for value in row if isinstance(row, list) else []:
            if value is None:
                cells.append("empty")
                continue
            cell = name(value)
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
    for variant in range(3):
        for questions, answer, topic in facts:
            if len(rows) >= limit:
                break
            if variant < len(questions):
                question = questions[variant]
                rows.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": question},
                                          {"role": "assistant", "content": answer}],
                             "reasoning": _plain(f"A Minecraft {topic} question: {question[:200]} I know the exact "
                                                 "game data for this, so I give the number straight away."),
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
        source = args.reasoning_dataset if preset.get("reasoning") else args.general_dataset
        if offline:
            print(f"would add up to {args.general} everyday rows from {source} (downloaded when training)")
        elif preset.get("reasoning"):
            limit = max(1, args.general // 2)  # traces are long: half as many rows
            name, split = hub_split(args.reasoning_dataset, "cot")
            records = load_hub_records(name, split, limit, args.seed, factor=3 if split == "cot" else 8)
            rows += reasoning_rows(records, limit, args.system_prompt, max_chars=args.max_seq * 3)
        else:
            name, split = hub_split(args.general_dataset, "train")
            rows += general_rows(load_hub_records(name, split, args.general, args.seed), args.general,
                                 args.system_prompt, bool(args.uncensored))
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
                    data = load_minecraft_data(version, getattr(args, "minecraft_data_dir", None))
                    found = minecraft_data_rows(data, args.minecraft_qa, args.system_prompt, rng)
                    print(f"+{len(found)} Minecraft Q&A rows generated from minecraft-data {version}")
                else:
                    name, split = hub_split(args.minecraft_qa_dataset, "train")
                    records = load_hub_records(name, split, args.minecraft_qa, args.seed + 1)
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
                where = Path(args.out) / "stream_images"
                if raytrace:
                    rows += stream_raytrace_rows(args.stream_vision, where, args.system_prompt, args.seed,
                                                 args.vision_dataset)
                else:
                    rows += stream_vision_rows(args.stream_vision, where, args.system_prompt, args.seed,
                                               bool(args.uncensored), args.vision_dataset)
            except Exception as exc:  # noqa: BLE001
                print(f"warning: could not stream pictures from {args.vision_dataset} ({type(exc).__name__}: {exc})")
    return rows


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
    for row in rows:
        try:
            reply = generate(row)
        except Exception as exc:  # a failing prompt must not lose the trained model
            reply = f"<error: {type(exc).__name__}: {exc}>"
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
    if report.get("mmproj"):
        files.append(f"- `{report['mmproj']}` - the vision projector (needed for pictures)")
    files += ["- `Modelfile` - for `ollama create`", "- `pymc_model.json` - training report PyMC Bot reads on import"]
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
    formats = ""
    if st_repo and not safetensors:
        formats = (f"\nAlso available as full-precision **safetensors** (merged 16-bit, for transformers, vLLM or "
                   f"further fine-tuning): [{st_repo}](https://huggingface.co/{st_repo})\n")
    library = "gguf"
    use = f"""With PyMC Bot (switches the bot to this model with the short Minecraft prompt):

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
```"""
    if safetensors:
        library = "transformers"
        size = f"~{report.get('safetensors_gb') or '8'} GB of 16-bit weights (bf16 / fp16), ~10 GB of GPU memory to run"
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
library_name: {library}
---

# {name}

**Minecraftism** is a small model that knows Minecraft and plays it: the brain of
[PyMC Bot]({PROJECT_URL}), a Minecraft player bot. Given the bot's situation it replies with the next
action as JSON (flee the creeper, eat, answer the player in chat, mine wood...), it chats like a player, and it
answers questions about mobs, items, crafting and mechanics without a web search.
It is also an everyday assistant: questions, writing, code, advice or just a chat.
{think_text_}{vision_text}{uncensored_text}{formats}
| | |
|---|---|
| Edition | {name} (`{preset_name}`) |
| Base model | [{base}](https://huggingface.co/{base}) (Apache-2.0) |
| Fine-tune | LoRA r={report.get('rank', 32)} with Unsloth on Kaggle (T4), {report.get('train_rows', '?')} rows, loss {report.get('loss', '?')} |
| Data | {data_cell} |
| Size | {size} |
{eval_lines}{test_lines}{data_text}{system_text}
## Files

{chr(10).join(files)}

## Use it

{use}

## License and credits

- Made with ❤️ by [codero-sus]({AUTHOR_URL}) - {WEBSITE}
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
    ignore = ["checkpoints/*", "*.lock", f"{SAFETENSORS_DIR}/*"] + ([] if include_lora else ["lora/*"])
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

    return finish(args, preset, model, tokenizer, rows, eval_rows, stats.training_loss, started,
                  text_generator(model, tokenizer, preset), lambda: FastLanguageModel.for_inference(model))


def text_generator(model: Any, tokenizer: Any, preset: dict[str, Any]) -> Any:
    def generate(row: dict[str, Any]) -> str:
        prompt = render(tokenizer, row["messages"][:-1], preset, generation=True)
        inputs = tokenizer(text=prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        output = model.generate(**inputs, max_new_tokens=preset.get("max_new_tokens", 256), temperature=0.3, top_p=0.8,
                                top_k=20, do_sample=True)
        return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


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
                              "images": sum(len(row.get("images") or []) for row in rows), "loss": round(loss, 4),
                              "uncensored": bool(preset.get("uncensored")), "system_prompt": args.system_prompt,
                              "general_rows": sum(1 for row in rows if row.get("kind") == "general"),
                              "commercial": bool(getattr(args, "commercial", False)), "data": data_sources(rows, args)}
    if eval_rows and args.eval_limit:
        for_inference()
        results = []
        for row in eval_rows[: args.eval_limit]:
            results.append(score(row, generate(row)))
        report["eval"] = summarize(results)
        print("eval:", json.dumps(report["eval"]))
    if args.test:
        for_inference()
        print("\n==== test suite ====")
        summary = run_tests(generate, test_rows(eval_rows, preset, args.system_prompt))
        (out / "tests.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        report["tests"] = {key: value for key, value in summary.items() if key != "results"}

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
                                                      preset.get("run_parameters"), args.system_prompt), encoding="utf-8")
        report["gguf"] = gguf.name
        report["quant"] = args.gguf.upper()
        report["gguf_gb"] = round(gguf.stat().st_size / 1e9, 2)
        if mmproj:
            report["mmproj"] = mmproj.name
        shutil.rmtree(out / "gguf", ignore_errors=True)
        if args.test_gguf:
            print("\n==== GGUF test (Ollama) ====")
            picture = next((Path(args.data_dir) / row["images"][0] for row in eval_rows if row.get("images")), None)
            prompts = [p for _kind, p in TEST_PROMPTS[:2]] + [TEST_PROMPTS[5][1]]
            if preset.get("vision") and picture is not None:
                prompts.append("What do you see in this Minecraft screenshot?")
            report["gguf_test"] = test_gguf(out, preset, prompts, picture)
    if getattr(args, "safetensors", "none") != "none":
        print("saving the safetensors edition (LoRA merged into the base, 16-bit) - a few minutes")
        try:
            report["safetensors_gb"] = export_safetensors(model, tokenizer, out / SAFETENSORS_DIR)
            report["safetensors"] = SAFETENSORS_DIR
            print(f"safetensors: {report['safetensors_gb']} GB in {out / SAFETENSORS_DIR}")
        except Exception as exc:  # the GGUF and the adapter are already safe
            print(f"safetensors export failed ({type(exc).__name__}: {exc}) - the GGUF and the LoRA adapter are fine")
            shutil.rmtree(out / SAFETENSORS_DIR, ignore_errors=True)
    shutil.rmtree(out / "checkpoints", ignore_errors=True)
    report.update({"family": preset["family"], "compact_tag": COMPACT_TAG, "compact_prompt": True,
                   "run_parameters": preset.get("run_parameters") or RUN_PARAMETERS,
                   "minutes": round((time.time() - started) / 60, 1), "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    repo = args.hub_repo if args.hub_repo and "/" in args.hub_repo else f"<your-hf-name>/{preset.get('hub') or 'pymc-minecraft-GGUF'}"
    write_cards(out, args.preset, preset, report, repo)
    if args.hub_repo:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("--hub-repo needs HF_TOKEN (Kaggle: Add-ons > Secrets > HF_TOKEN, a 'write' token)")
        else:
            try:
                from huggingface_hub import HfApi

                full = resolve_repo(args.hub_repo, preset, HfApi(token=token).whoami)
                report["hub_repo"] = full
                if report.get("safetensors"):
                    report["hub_repo_safetensors"] = safetensors_repo(full)
                write_cards(out, args.preset, preset, report, full)
                publish(out, full, args.preset, token, private=args.hub_private, include_lora=args.hub_lora)
                print(f"uploaded the GGUF edition -> https://huggingface.co/{full}")
            except Exception as exc:  # never lose a finished model to an upload problem
                report.pop("hub_repo", None)
                print(f"Hugging Face upload failed ({type(exc).__name__}: {exc}) - the files are still in {out}")
            if report.get("hub_repo") and report.get("hub_repo_safetensors"):
                st_repo = report["hub_repo_safetensors"]
                try:
                    publish_safetensors(out / SAFETENSORS_DIR, st_repo, args.preset, token, private=args.hub_private)
                    report["safetensors_uploaded"] = True
                    print(f"uploaded the safetensors edition -> https://huggingface.co/{st_repo}")
                except Exception as exc:  # the GGUF repo is already up; keep the folder for another try
                    print(f"safetensors upload failed ({type(exc).__name__}: {exc}) - the files are still in "
                          f"{out / SAFETENSORS_DIR}")
            write_cards(out, args.preset, preset, report, report.get("hub_repo") or repo)
    shutil.rmtree(out / "stream_images", ignore_errors=True)  # streamed training pictures are not needed any more
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
    parser.add_argument("--safetensors", choices=("16bit", "none"), default="16bit",
                        help="also save (and with --hub-repo upload, to <repo without -GGUF>) the full-precision "
                             "safetensors edition: the LoRA merged into the base in 16-bit (~8 GB for 4B)")
    parser.add_argument("--split", action="store_true", help="spread the model over both GPUs (only needed for 14B)")
    parser.add_argument("--hub-repo", help="upload to Hugging Face (needs HF_TOKEN): 'auto' = <you>/<edition repo>, "
                                           "a repo name, or user/repo")
    parser.add_argument("--hub-private", action="store_true", help="create the Hugging Face repo as private")
    parser.add_argument("--hub-lora", action="store_true", help="also upload the LoRA adapter folder")
    parser.add_argument("--image-side", type=int, default=IMAGE_SIDE, help="vision editions: longest side of a picture")
    parser.add_argument("--vision-layers", type=int, default=0, choices=[0, 1],
                        help="vision editions: 1 = also fine-tune the image encoder (slower, more memory)")
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
    resolve_datasets(args)
    if args.commercial:
        problems, warnings = commercial_problems(args, preset)
        for warning in warnings:
            print("note:", warning)
        if problems:
            print("--commercial:\n" + "\n".join(problems))
            return 2
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
        return 2
    if preset.get("vision") and not any(row.get("images") for row in rows) and args.stream_vision <= 0:
        print(f"{args.preset} is a vision edition but the data has no pictures. Record playtime with the mod's "
              "screenshots on (/pymcbot screenshots on), then: python -m pymc_bot llm build --playtime <folder> "
              "--screenshots - or stream pictures from Hugging Face (--stream-vision 3000) - or train a text edition "
              "(minecraftism, minecraftism-thinking).")
        return 2
    problems = check_rows(rows) + check_rows(eval_rows)
    if problems:
        print("\n".join(problems[:20]))
        return 2
    if args.test_only:
        return test_only(args, preset, eval_rows)
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
    print(f"persona: {persona} Q&A rows use the everyday system prompt; +{len(extra)} identity/everyday rows"
          f"{' (uncensored)' if args.uncensored else ''}")
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
        sample = prepare_messages(rows[0], preset)[-1]["content"]
        print("dry run: data is fine. Sample reply:", sample[:400])
        return 0
    return train(args, preset, rows, eval_rows)


if __name__ == "__main__":
    sys.exit(main())
