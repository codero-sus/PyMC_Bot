#!/usr/bin/env python3
"""Everything except the mod jars that CI attaches to the GitHub release.

    python scripts/release_kit.py --out kit                    # all of it (wiki pages need internet)
    python scripts/release_kit.py --out kit --no-wiki --no-wheel   # offline / quick

Writes to ``--out``:

* ``updater.py``, ``updater.sh``, ``updater.cmd`` - the updater on its own (download, double-click)
* ``PyMC_Bot-source.zip``      - the whole project, ready to run (``python -m pymc_bot``)
* ``pymc_bot-<v>-py3-none-any.whl`` / ``pymc_bot-<v>.tar.gz`` - ``pip install`` packages
* ``pymc-minecraft-llm-kit.zip`` - the ready-made training data + Kaggle script (upload it to Kaggle)
* ``PyMC_Minecraft_LLM_Kaggle.ipynb``, ``train_minecraft_llm.py`` - the notebook and trainer on their own

``SHA256SUMS.txt`` is written by :func:`checksums` over the final folder (jars included).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPDATER_FILES = ("updater.py", "updater.sh", "updater.cmd")
NOTEBOOK = ROOT / "training" / "llm" / "PyMC_Minecraft_LLM_Kaggle.ipynb"
KIT_README = """PyMC Bot - Minecraft LLM kit
============================

Ready-made training data for Minecraftism, your own Minecraft model (the bot's brain + a Minecraft
expert) in four editions: minecraftism, minecraftism-thinking, minecraftism-vl, minecraftism-vl-thinking.

1. kaggle.com > Datasets > New Dataset > upload everything in the llm_data folder.
2. Code > New Notebook > File > Import Notebook > PyMC_Minecraft_LLM_Kaggle.ipynb
   Add Input = your dataset, Settings > Accelerator = GPU T4 x2, Internet on.
   Optional: Add-ons > Secrets > HF_TOKEN (a Hugging Face write token) to publish the model.
   Pick the EDITIONS, Run All.
3. python -m pymc_bot llm import --hf <your-hf-name>/Minecraftism-4B-GGUF
   or download /kaggle/working/minecraftism and run:  python -m pymc_bot llm import minecraftism
The VL editions need pictures: record with /pymcbot screenshots on (mod 0.10+), then
python -m pymc_bot llm build --playtime <folder> --screenshots

Want your own recorded turns / playtime / a teacher model in it? Build it yourself:
python -m pymc_bot llm build  (guide: docs/MINECRAFT_LLM.md)

Minecraft Wiki text (if included, see dataset_card.json): CC BY-NC-SA 3.0, https://minecraft.wiki
PyMC Bot: PolyForm Noncommercial 1.0.0 - personal, non-commercial use.
Made with love by codero-sus - https://github.com/codero-sus - https://the-codero.web.app/
"""


def run(cmd: list[str], **kwargs) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=kwargs.pop("cwd", ROOT), **kwargs)


def updater_files(out: Path) -> list[Path]:
    copied = []
    for name in UPDATER_FILES:
        target = out / name
        shutil.copyfile(ROOT / name, target)
        if name.endswith(".sh"):
            target.chmod(0o755)
        copied.append(target)
    return copied


def source_zip(out: Path, ref: str = "HEAD") -> Path:
    target = out / "PyMC_Bot-source.zip"
    run(["git", "archive", "--format=zip", "--prefix=PyMC_Bot/", "-o", str(target), ref])
    return target


def wheel(out: Path) -> list[Path]:
    before = set(out.iterdir())
    run([sys.executable, "-m", "build", "--outdir", str(out), str(ROOT)])
    return sorted(set(out.iterdir()) - before)


def llm_kit(out: Path, work: Path, wiki_pages: int, turns: int) -> list[Path]:
    data = work / "llm_data"
    cmd = [sys.executable, "-m", "pymc_bot", "llm", "build", "--out", str(data), "--turns", str(turns),
           "--config", str(work / "kit-config.json")]
    cmd += ["--wiki-pages", str(wiki_pages)] if wiki_pages > 0 else ["--no-wiki"]
    run(cmd)
    kit = out / "pymc-minecraft-llm-kit.zip"
    with zipfile.ZipFile(kit, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(data.iterdir()):
            if path.is_file():
                archive.write(path, f"llm_data/{path.name}")
        archive.write(NOTEBOOK, NOTEBOOK.name)
        archive.writestr("README.txt", KIT_README)
    shutil.copyfile(NOTEBOOK, out / NOTEBOOK.name)
    shutil.copyfile(data / "train_minecraft_llm.py", out / "train_minecraft_llm.py")
    return [kit, out / NOTEBOOK.name, out / "train_minecraft_llm.py"]


def checksums(folder: Path) -> Path:
    lines = []
    for path in sorted(p for p in folder.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt"):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    target = folder / "SHA256SUMS.txt"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="release-kit")
    parser.add_argument("--work", default=None, help="scratch folder (default: <out>/../release-kit-work)")
    parser.add_argument("--wiki-pages", type=int, default=600)
    parser.add_argument("--no-wiki", action="store_true")
    parser.add_argument("--turns", type=int, default=6000)
    parser.add_argument("--no-wheel", action="store_true", help="skip the pip packages (needs the 'build' package)")
    parser.add_argument("--checksums", metavar="FOLDER", help="only write SHA256SUMS.txt for FOLDER")
    args = parser.parse_args(argv)
    if args.checksums:
        print(checksums(Path(args.checksums)).read_text(encoding="utf-8"), end="")
        return 0
    out = Path(args.out).resolve()
    work = Path(args.work).resolve() if args.work else out.parent / "release-kit-work"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    made = updater_files(out)
    made.append(source_zip(out))
    if not args.no_wheel:
        made += wheel(out)
    made += llm_kit(out, work, 0 if args.no_wiki else args.wiki_pages, args.turns)
    for path in made:
        print(f"{path.stat().st_size:>12,}  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
