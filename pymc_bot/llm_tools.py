"""Bring a Minecraft model trained on Kaggle into Ollama, and measure how well a model plays.

* :func:`import_model` - ``python -m pymc_bot llm import <folder|file.gguf>``: writes a Modelfile,
  runs ``ollama create`` and switches the bot to the new model with the short prompt.
* :func:`evaluate` - ``python -m pymc_bot llm eval``: runs ``eval.jsonl`` against any Ollama model
  and reports valid JSON %, right action % and right answers % (compare before/after training).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pymc_bot.config import ConfigStore
from pymc_bot.llm_train import PRESETS, modelfile_text, score, summarize, template_from_modelfile

DEFAULT_MODEL_NAME = "pymc-minecraft"


@dataclass
class ImportResult:
    name: str
    gguf: Path
    modelfile: Path
    created: bool
    activated: bool
    message: str
    info: dict[str, Any] = field(default_factory=dict)


def find_gguf(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_file() and path.suffix.lower() == ".gguf":
        return path.resolve()
    if path.is_dir():
        files = sorted(p for p in path.glob("*.gguf") if "mmproj" not in p.name.lower())
        if files:
            return max(files, key=lambda p: p.stat().st_size).resolve()
    raise FileNotFoundError(f"no .gguf file at {path} - download the pymc-minecraft folder from Kaggle's Output tab")


def write_modelfile(gguf: Path, info: dict[str, Any] | None = None) -> Path:
    """Modelfile next to the GGUF, pointing at it by absolute path (keeps a TEMPLATE written by training)."""
    info = info or {}
    target = gguf.parent / "Modelfile"
    template = template_from_modelfile(target.read_text(encoding="utf-8")) if target.is_file() else None
    family = info.get("family") or PRESETS.get(str(info.get("preset")), {}).get("family") or (
        "llama3" if "llama" in gguf.name.lower() else "chatml")
    text = modelfile_text(gguf.name, family, template).replace(f"FROM ./{gguf.name}", f"FROM {gguf.as_posix()}", 1)
    target.write_text(text, encoding="utf-8")
    return target


def import_model(path: Path, store: ConfigStore, name: str = DEFAULT_MODEL_NAME, activate: bool = True,
                 run: Callable[..., Any] = subprocess.run, which: Callable[[str], str | None] = shutil.which) -> ImportResult:
    gguf = find_gguf(path)
    info_path = gguf.parent / "pymc_model.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    modelfile = write_modelfile(gguf, info)
    created = False
    if which("ollama"):
        proc = run(["ollama", "create", name, "-f", str(modelfile)], capture_output=True, text=True)
        created = proc.returncode == 0
        message = (f"Ollama model '{name}' created from {gguf.name}" if created
                   else f"ollama create failed: {(proc.stderr or proc.stdout or '').strip()[-400:]}")
    else:
        message = (f"Ollama is not on PATH. Install it (https://ollama.com), then run:\n"
                   f"  ollama create {name} -f \"{modelfile}\"")
    activated = False
    if activate:
        store.update({"ollama": {"enabled": True, "model": name},
                      "agent": {"mode": "ollama", "compact_prompt": bool(info.get("compact_prompt", True))}})
        activated = True
    return ImportResult(name, gguf, modelfile, created, activated, message, info)


def evaluate(rows: list[dict[str, Any]], ask: Callable[[list[dict[str, str]], str], str], limit: int | None = None,
             log: Callable[[str], None] = print) -> dict[str, Any]:
    """Score a model on eval rows; ``ask(messages, system)`` returns the reply text."""
    results = []
    errors = 0
    for number, row in enumerate(rows[: limit or None], 1):
        messages = row["messages"]
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        prompt = [m for m in messages[:-1] if m["role"] != "system"]
        try:
            reply = ask(prompt, system)
        except Exception as exc:  # timeouts count as failures
            errors += 1
            reply = ""
            if errors <= 3:
                log(f"row {number}: {exc}")
        results.append(score(row, reply))
        if number % 25 == 0:
            log(f"{number} rows... {json.dumps(summarize(results))}")
    summary = summarize(results)
    summary["errors"] = errors
    by_source: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_source.setdefault(result["source"], []).append(result)
    summary["by_source"] = {source: summarize(items) for source, items in sorted(by_source.items())}
    return summary
