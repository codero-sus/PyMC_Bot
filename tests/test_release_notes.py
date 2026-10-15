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
    assert release_notes.tag_name() == f"pymcplaytime-v{version}"


def test_notes_list_every_jar_and_how_it_was_tested():
    table = json.loads((ROOT / "mod/versions.json").read_text())
    text = release_notes.notes()
    version = release_notes.mod_version()
    for entry in table["versions"]:
        row = next(line for line in text.splitlines() if line.startswith(f"| {entry['minecraft']} |"))
        assert f"`pymcplaytime-{version}+{entry['minecraft']}.jar`" in row
        assert ("launched in CI" if entry["gametest"] else "compiled in CI") in row
    assert "Made with \u2764\ufe0f by [codero-sus](https://github.com/codero-sus)" in text
    assert "Cortex" not in text


def test_cli_writes_a_file(tmp_path, capsys):
    out = tmp_path / "notes.md"
    assert release_notes.main(["-o", str(out)]) == 0
    assert out.read_text().startswith("## PyMC Playtime Recorder ")
    assert release_notes.main(["--tag"]) == 0
    assert capsys.readouterr().out.strip() == release_notes.tag_name()
