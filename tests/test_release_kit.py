"""scripts/release_kit.py: the non-jar files of the GitHub release."""

from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import release_kit  # noqa: E402
import release_notes  # noqa: E402


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs the git checkout")
def test_offline_kit_has_every_file(tmp_path):
    out = tmp_path / "kit"
    assert release_kit.main(["--out", str(out), "--work", str(tmp_path / "work"), "--no-wiki", "--no-wheel",
                             "--no-node-bundle", "--turns", "40"]) == 0
    names = {p.name for p in out.iterdir()}
    assert {"updater.py", "updater.sh", "updater.cmd", "PyMC_Bot-source.zip", "pymc-minecraft-llm-kit.zip",
            "PyMC_Minecraft_LLM_Kaggle.ipynb", "train_minecraft_llm.py"} <= names
    for name in ("updater.py", "updater.sh", "updater.cmd"):
        assert (out / name).read_bytes() == (ROOT / name).read_bytes()
    with zipfile.ZipFile(out / "PyMC_Bot-source.zip") as source:
        assert {"PyMC_Bot/pymc_bot/__main__.py", "PyMC_Bot/updater.py", "PyMC_Bot/requirements.txt"} <= set(source.namelist())
    with zipfile.ZipFile(out / "pymc-minecraft-llm-kit.zip") as kit:
        listed = set(kit.namelist())
        assert {"llm_data/train.jsonl", "llm_data/eval.jsonl", "llm_data/train_minecraft_llm.py", "README.txt",
                "PyMC_Minecraft_LLM_Kaggle.ipynb"} <= listed
        assert not any("wiki_cache" in name for name in listed)
    sums = release_kit.checksums(out).read_text().splitlines()
    assert len(sums) == len(names)
    digest, name = sums[0].split("  ")
    assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    # every file the kit makes is explained in the release notes
    notes = release_notes.notes()
    assert all(f"`{name}`" in notes for name in names)
