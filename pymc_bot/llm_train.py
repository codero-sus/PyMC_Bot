#!/usr/bin/env python3
"""Fine-tune a Minecraft-only LLM for PyMC Bot on a Kaggle notebook (2x T4) - or any CUDA box.

This file is self-contained: ``python -m pymc_bot llm build`` copies it next to ``train.jsonl`` so
the Kaggle dataset you upload has everything. In the notebook (training/llm/*.ipynb):

    python train_minecraft_llm.py --preset qwen3-4b            # ~1-2 h on one T4
    python train_minecraft_llm.py --dry-run                    # check the data, no GPU needed

It trains a LoRA adapter with Unsloth (4-bit QLoRA, loss on the replies only), checks the result on
``eval.jsonl``, and writes a ready-to-run Ollama model to ``/kaggle/working/pymc-minecraft``:
``*.gguf`` + ``Modelfile`` + ``pymc_model.json``. Download that folder and run
``python -m pymc_bot llm import <folder>`` on your PC.

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
PRESETS: dict[str, dict[str, Any]] = {
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
        "why": "Recommended. Strong for its size, no 'thinking' so replies are fast JSON, trains in ~1-2 h on "
               "one T4 and runs on most PCs (4 GB VRAM or CPU).",
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
DEFAULT_PRESET = "qwen3-4b"
#: How each family marks a user turn / an assistant reply (loss is computed on replies only).
TURN_MARKERS = {
    "chatml": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "llama3": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
}
STOP_TOKENS = {"chatml": ["<|im_end|>", "<|im_start|>"], "llama3": ["<|eot_id|>", "<|start_header_id|>"]}
#: Sampling for the finished model (Qwen's advice for non-thinking mode, a bit cooler for JSON).
RUN_PARAMETERS = {"temperature": 0.6, "top_p": 0.8, "top_k": 20, "num_ctx": 4096, "repeat_penalty": 1.05}
DATA_GLOBS = ("train.jsonl", "llm_data/train.jsonl", "/kaggle/input/**/train.jsonl")


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


def first_json(text: str) -> dict[str, Any] | None:
    """The first JSON object in a reply (``<think>`` blocks and code fences are ignored)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()
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
    tokens = chars / 3.6
    per_step = preset["batch"] * preset["accum"]
    steps = max_steps if max_steps > 0 else int(len(rows) * epochs / per_step) + 1
    # ~1.6k tokens/s for a 4B QLoRA on one T4, scaled by the preset's relative cost
    hours = (tokens / max(1, len(rows))) * per_step * steps / 1600 * preset["speed"] / 3600
    return {"rows": len(rows), "tokens": int(tokens), "avg_tokens": int(tokens / max(1, len(rows))), "steps": steps,
            "est_hours_t4": round(hours, 1)}


# --------------------------------------------------------------------------- Ollama files
def modelfile_text(gguf_name: str, family: str, template: str | None = None) -> str:
    lines = [f"FROM ./{gguf_name}"]
    if template:
        lines.append(f'TEMPLATE """{template}"""')
    for key, value in RUN_PARAMETERS.items():
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


def train(args: argparse.Namespace, preset: dict[str, Any], rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> int:
    if not args.split:
        # one T4 is plenty up to 14B and avoids DataParallel, which Unsloth does not use
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
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

    texts = [render(tokenizer, row["messages"], preset) for row in rows]
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

    adapter = out / "lora"
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))

    report = {"preset": args.preset, "base_model": model_name, "train_rows": len(rows), "loss": round(stats.training_loss, 4)}
    if eval_rows and args.eval_limit:
        FastLanguageModel.for_inference(model)
        results = []
        for row in eval_rows[: args.eval_limit]:
            prompt = render(tokenizer, row["messages"][:-1], preset, generation=True)
            inputs = tokenizer(text=prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            output = model.generate(**inputs, max_new_tokens=256, temperature=0.3, top_p=0.8, top_k=20, do_sample=True)
            reply = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append(score(row, reply))
        report["eval"] = summarize(results)
        print("eval:", json.dumps(report["eval"]))

    if args.gguf != "none":
        print(f"exporting GGUF ({args.gguf}) - this takes 10-20 minutes")
        model.save_pretrained_gguf(str(out / "gguf"), tokenizer, quantization_method=args.gguf)
        found = sorted(set(glob.glob(str(out / "gguf*" / "*.gguf")) + glob.glob(str(out / "*.gguf")) + glob.glob("*.gguf")))
        found = [f for f in found if "mmproj" not in f.lower()]
        if not found:
            print("GGUF export produced no file - the LoRA adapter is in", adapter)
            return 1
        gguf = out / f"pymc-minecraft-{args.preset}.{args.gguf.upper()}.gguf"
        shutil.move(found[0], gguf)
        unsloth_modelfile = next(iter(glob.glob(str(out / "gguf*" / "Modelfile"))), None)
        template = template_from_modelfile(Path(unsloth_modelfile).read_text()) if unsloth_modelfile else None
        (out / "Modelfile").write_text(modelfile_text(gguf.name, preset["family"], template), encoding="utf-8")
        report["gguf"] = gguf.name
        report["gguf_gb"] = round(gguf.stat().st_size / 1e9, 2)
        shutil.rmtree(out / "gguf", ignore_errors=True)
    shutil.rmtree(out / "checkpoints", ignore_errors=True)
    report.update({"family": preset["family"], "compact_tag": COMPACT_TAG, "compact_prompt": True,
                   "run_parameters": RUN_PARAMETERS, "minutes": round((time.time() - started) / 60, 1),
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    (out / "pymc_model.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.hub_repo:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("--hub-repo needs HF_TOKEN (Kaggle: Add-ons > Secrets)")
        else:
            model.push_to_hub_gguf(args.hub_repo, tokenizer, quantization_method=[args.gguf], token=token)
    print(f"done -> {out}  (download it, then: python -m pymc_bot llm import <folder>)")
    return 0


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune the PyMC Bot Minecraft LLM (Unsloth, Kaggle T4).")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--model", help="override the base model of the preset (any Unsloth/HF model id)")
    parser.add_argument("--data", help="train.jsonl or its folder (default: search /kaggle/input)")
    parser.add_argument("--eval-data", help="eval.jsonl (default: next to train.jsonl)")
    parser.add_argument("--out", default="/kaggle/working/pymc-minecraft" if Path("/kaggle/working").is_dir() else "pymc-minecraft")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = full epochs)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--max-seq", type=int, default=3072, help="longest example in tokens")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval-limit", type=int, default=150, help="eval rows to try after training (0 = skip)")
    parser.add_argument("--gguf", default="q4_k_m", help="GGUF quantisation (q4_k_m, q5_k_m, q8_0) or 'none'")
    parser.add_argument("--split", action="store_true", help="spread the model over both GPUs (only needed for 14B)")
    parser.add_argument("--hub-repo", help="also upload the GGUF to this Hugging Face repo (needs HF_TOKEN)")
    parser.add_argument("--dry-run", action="store_true", help="check the data and show the plan without a GPU")
    parser.add_argument("--list-presets", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_presets:
        for name, preset in PRESETS.items():
            mark = " (default)" if name == DEFAULT_PRESET else ""
            print(f"{name}{mark}: {preset['model']} - train ~{preset['train_gb']} GB, run ~{preset['run_gb']} GB, "
                  f"file ~{preset['gguf_gb']} GB\n    {preset['why']}")
        return 0
    preset = PRESETS[args.preset]
    data = find_data(args.data)
    rows = read_jsonl(data)
    eval_path = Path(args.eval_data) if args.eval_data else data.with_name("eval.jsonl")
    eval_rows = read_jsonl(eval_path) if eval_path.is_file() else []
    problems = check_rows(rows) + check_rows(eval_rows)
    if problems:
        print("\n".join(problems[:20]))
        return 2
    plan = estimate(rows, preset, args.epochs, args.max_steps)
    sources: dict[str, int] = {}
    for row in rows:
        sources[row.get("source", "?")] = sources.get(row.get("source", "?"), 0) + 1
    print(f"data: {data} ({len(rows)} train, {len(eval_rows)} eval) sources={sources}")
    print(f"preset: {args.preset} -> {args.model or preset['model']}; plan: {json.dumps(plan)}")
    if args.dry_run:
        print("dry run: data is fine. Sample reply:", rows[0]["messages"][-1]["content"][:200])
        return 0
    return train(args, preset, rows, eval_rows)


if __name__ == "__main__":
    sys.exit(main())
