"""The release notes CI attaches to the mod's GitHub release."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import release_notes  # noqa: E402


def test_tag_follows_mod_version():
    version = release_notes.mod_version()
    assert version.count(".") == 2
    assert release_notes.tag_name() == f"pymcbot-v{version}"


def test_notes_list_every_jar_and_how_it_was_tested():
    table = json.loads((ROOT / "mod/versions.json").read_text())
    text = release_notes.notes()
    version = release_notes.mod_version()
    for entry in table["versions"]:
        row = next(line for line in text.splitlines() if line.startswith(f"| {entry['minecraft']} |"))
        assert f"`pymcbot-{version}+{entry['minecraft']}.jar`" in row
        assert ("launched in CI" if entry["gametest"] else "compiled in CI") in row
    assert "Made with \u2764\ufe0f by [codero-sus](https://github.com/codero-sus)" in text
    assert "Cortex" not in text


def test_cli_writes_a_file(tmp_path, capsys):
    out = tmp_path / "notes.md"
    assert release_notes.main(["-o", str(out)]) == 0
    assert out.read_text().startswith("## PyMC Bot ")
    assert release_notes.main(["--tag"]) == 0
    assert capsys.readouterr().out.strip() == release_notes.tag_name()


def test_notes_list_every_release_file():
    text = release_notes.notes()
    version = release_notes.mod_version()
    for name in ("PyMC_Bot-source.zip", "updater.py", "updater.cmd", "updater.sh", f"pymcbot-{version}-all-versions.zip",
                 "pymc-minecraft-llm-kit.zip", "PyMC_Minecraft_LLM_Kaggle.ipynb", "train_minecraft_llm.py",
                 "SHA256SUMS.txt", f"pymc_bot-{release_notes.wheel_version()}-py3-none-any.whl"):
        assert f"`{name}`" in text, name


def test_notes_say_the_notebook_uploads_safetensors_then_gguf_quants():
    text = release_notes.notes()
    assert "`PyMC_Minecraft_LLM_Kaggle.ipynb`" in text
    assert "Both T4s train" in text and "MAX_HOURS" in text and "AUTO_EXIT" in text
    assert "strong-to-weak distillation" in text and 'GPU_MODE = "parallel"' in text
    assert "editions_to_be_trained = {txt, txt_think, vl, vl_think}" in text
    assert "**safetensors**" in text and "**GGUF**" in text and "SAFETENSORS = True" in text
    assert text.index("**safetensors**") < text.index("**GGUF**")
    assert "Q8_0" in text and "Q4_K_M" in text and "`GGUF_QUANTS`" in text and "`tokenizer.json`" in text
