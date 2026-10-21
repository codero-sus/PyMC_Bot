"""Bring a Minecraft model trained on Kaggle into Ollama, and measure how well a model plays.

* :func:`import_model` - ``python -m pymc_bot llm import <folder|file.gguf>``: writes a Modelfile
  (with the vision projector for the VL editions), runs ``ollama create`` and switches the bot to
  the new model with the short prompt.
* :func:`import_model_hf` - ``python -m pymc_bot llm import --hf user/Minecraftism-4B-GGUF``:
  ``ollama pull hf.co/...`` straight from Hugging Face, then the same switch.
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
    """Modelfile next to the GGUF, pointing at it (and at the vision projector, if any) by absolute
    path. Keeps a TEMPLATE written by training and uses the edition's sampling parameters."""
    info = info or {}
    target = gguf.parent / "Modelfile"
    template = template_from_modelfile(target.read_text(encoding="utf-8")) if target.is_file() else None
    preset = PRESETS.get(str(info.get("preset")), {})
    family = info.get("family") or preset.get("family") or ("llama3" if "llama" in gguf.name.lower() else "chatml")
    mmproj = find_mmproj(gguf, info)
    parameters = info.get("run_parameters") or preset.get("run_parameters")
    text = modelfile_text(gguf.name, family, template, mmproj.name if mmproj else None, parameters)
    text = text.replace(f"FROM ./{gguf.name}", f"FROM {gguf.as_posix()}", 1)
    if mmproj:
        text = text.replace(f"FROM ./{mmproj.name}", f"FROM {mmproj.resolve().as_posix()}", 1)
    target.write_text(text, encoding="utf-8")
    return target


def find_mmproj(gguf: Path, info: dict[str, Any] | None = None) -> Path | None:
    """The vision projector that belongs to ``gguf`` (named in pymc_model.json, else any mmproj*.gguf)."""
    named = (info or {}).get("mmproj")
    if named and (gguf.parent / str(named)).is_file():
        return gguf.parent / str(named)
    found = sorted(p for p in gguf.parent.glob("*.gguf") if "mmproj" in p.name.lower())
    return found[0] if found else None


def default_name(info: dict[str, Any]) -> str:
    """Ollama name for an imported model: the Minecraftism edition, else the old default."""
    return str(info.get("edition") or DEFAULT_MODEL_NAME)


def _activate(store: ConfigStore, name: str, info: dict[str, Any]) -> None:
    agent: dict[str, Any] = {"mode": "ollama", "compact_prompt": bool(info.get("compact_prompt", True))}
    if info.get("vision"):
        agent["vision"] = True
    store.update({"ollama": {"enabled": True, "model": name}, "agent": agent})


def import_model(path: Path, store: ConfigStore, name: str | None = None, activate: bool = True,
                 run: Callable[..., Any] = subprocess.run, which: Callable[[str], str | None] = shutil.which) -> ImportResult:
    gguf = find_gguf(path)
    info_path = gguf.parent / "pymc_model.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    name = name or default_name(info)
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
        _activate(store, name, info)
        activated = True
    return ImportResult(name, gguf, modelfile, created, activated, message, info)


def hf_reference(repo: str, quant: str = "Q4_K_M") -> str:
    """``user/repo`` (or a huggingface.co URL) -> ``hf.co/user/repo:QUANT`` for ``ollama pull``."""
    repo = repo.strip().removeprefix("https://").removeprefix("http://")
    for prefix in ("huggingface.co/", "hf.co/"):
        repo = repo.removeprefix(prefix)
    repo = repo.strip("/")
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(f"expected a Hugging Face repo like user/Minecraftism-4B-GGUF, got {repo!r}")
    return f"hf.co/{repo}:{quant.upper()}" if quant else f"hf.co/{repo}"


def edition_from_repo(repo: str) -> dict[str, Any]:
    """Guess the Minecraftism edition from a repo name (for the Ollama name and vision switch)."""
    low = repo.lower()
    for edition in sorted((n for n in PRESETS if n.startswith("minecraftism")), key=len, reverse=True):
        hub = str(PRESETS[edition].get("hub") or "").lower()
        if hub and low.endswith("/" + hub):
            return {"edition": edition, "vision": bool(PRESETS[edition].get("vision")), "compact_prompt": True}
    vision = "-vl" in low or "vision" in low
    return {"vision": vision, "compact_prompt": "minecraftism" in low or "pymc" in low}


def import_model_hf(repo: str, store: ConfigStore, name: str | None = None, quant: str = "Q4_K_M", activate: bool = True,
                    run: Callable[..., Any] = subprocess.run,
                    which: Callable[[str], str | None] = shutil.which) -> ImportResult:
    """``ollama pull hf.co/<repo>:<quant>``, copy it to a short name, switch the bot to it."""
    reference = hf_reference(repo, quant)
    info = edition_from_repo(reference)
    name = name or str(info.get("edition") or reference.split("/")[-1].split(":")[0].lower())
    created = False
    if which("ollama"):
        proc = run(["ollama", "pull", reference], capture_output=True, text=True)
        if proc.returncode == 0:
            copy = run(["ollama", "cp", reference, name], capture_output=True, text=True)
            created = copy.returncode == 0
            message = (f"Ollama model '{name}' pulled from {reference}" if created
                       else f"pulled {reference}, but ollama cp failed: {(copy.stderr or copy.stdout or '').strip()[-300:]}")
        else:
            message = f"ollama pull {reference} failed: {(proc.stderr or proc.stdout or '').strip()[-400:]}"
    else:
        message = (f"Ollama is not on PATH. Install it (https://ollama.com), then run:\n"
                   f"  ollama pull {reference}\n  ollama cp {reference} {name}")
    activated = False
    if activate and (created or not which("ollama")):
        _activate(store, name, info)
        activated = True
    return ImportResult(name, Path(reference), Path(""), created, activated, message, info)


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
