"""Static checks for the multi-version Fabric mod (mod/).

The mod itself only compiles in CI (one Gradle build per Minecraft version), so these tests
guard the parts that can be checked without Java: the version table, the per-era compat
classes and the rule that version-specific calls stay inside them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "mod"
TABLE = json.loads((MOD / "versions.json").read_text())

# Every stable Minecraft release from 1.21.1 to 26.3 (meta.fabricmc.net/v2/versions/game).
EXPECTED = [
    "1.21.1", "1.21.2", "1.21.3", "1.21.4", "1.21.5", "1.21.6", "1.21.7", "1.21.8",
    "1.21.9", "1.21.10", "1.21.11", "26.1", "26.1.1", "26.1.2", "26.2", "26.3",
]

# Only src/compat/<era>/.../Mc.java may use these: they differ between Minecraft versions.
VERSION_SPECIFIC = [
    r"net\.minecraft\.resources\.Identifier",
    r"net\.minecraft\.resources\.ResourceLocation",
    r"\bClientCommandManager\b",
    r"\bClientCommands\b",
    r"\bKeyBindingHelper\b",
    r"\bKeyMappingHelper\b",
    r"getSelectedSlot\(",
    r"\.identifier\(\)",
    r"dimension\(\)\.location\(\)",
    r"displayClientMessage\(",
    r"sendSystemMessage\(",
    r"InputConstants",
    r"\.attended\(\)",
    r"client\.screen\b",
    r"gui\.screen\(\)",
]


def versions() -> list[dict]:
    return TABLE["versions"]


def compat_files() -> dict[str, Path]:
    return {p.parents[6].name: p for p in MOD.glob("src/compat/*/java/dev/pymc/pymcb/client/compat/Mc.java")}


def test_every_release_from_1_21_1_to_26_3_is_built():
    assert [v["minecraft"] for v in versions()] == EXPECTED
    assert TABLE["default"] in EXPECTED


@pytest.mark.parametrize("entry", versions(), ids=lambda v: v["minecraft"])
def test_version_entry_is_consistent(entry):
    mc = entry["minecraft"]
    assert entry["fabric_api"].endswith("+" + mc), "Fabric API build must be for the same game version"
    legacy = mc.startswith("1.")
    assert entry["obfuscated"] is legacy, "1.21.x is obfuscated (remap Loom), 26.x is not"
    assert entry["java"] == (21 if legacy else 25)
    assert entry["compat"] in compat_files(), f"src/compat/{entry['compat']} is missing"


def test_every_compat_era_is_used_and_shares_one_api():
    files = compat_files()
    used = {v["compat"] for v in versions()}
    assert used == set(files), "unused or missing compat folders"

    def api(path: Path) -> list[str]:
        text = path.read_text()
        return sorted(re.findall(r"public static [^=(]+?\b(\w+)\s*\(", text) + re.findall(r"public static final \w+ (\w+)", text))

    reference = None
    for era, path in sorted(files.items()):
        names = api(path)
        assert names, era
        reference = reference or names
        assert names == reference, f"{era} exposes {names}, expected {reference}"


def test_version_specific_calls_stay_in_the_compat_layer():
    offenders = []
    for path in MOD.glob("src/**/*.java"):
        if "compat" in path.parts:
            continue
        text = path.read_text()
        for pattern in VERSION_SPECIFIC:
            if re.search(pattern, text):
                offenders.append(f"{path.relative_to(MOD)}: {pattern}")
    assert not offenders, offenders


def test_compat_eras_match_the_api_changes():
    files = {era: path.read_text() for era, path in compat_files().items()}
    assert "getInventory().selected" in files["v1_21_1"]
    for era in ("v1_21_5", "v1_21_9", "v1_21_11", "v26_1", "v26_2"):
        assert "getSelectedSlot()" in files[era]
    for era in ("v1_21_1", "v1_21_5"):
        assert 'String CATEGORY = "key.categories.pymcplaytime"' in files[era]
    assert "ResourceLocation.fromNamespaceAndPath" in files["v1_21_9"]
    for era in ("v1_21_11", "v26_1", "v26_2"):
        assert "Identifier.fromNamespaceAndPath" in files[era]
        assert ".identifier()" in files[era]
    for era in ("v26_1", "v26_2"):
        assert "ClientCommands." in files[era] and "KeyMappingHelper.registerKeyMapping" in files[era]
    assert "source.attended()" in files["v26_2"]
    assert "client.gui.screen() != null" in files["v26_2"]
    assert all("client.screen != null" in files[e] for e in ("v1_21_1", "v1_21_5", "v1_21_9", "v1_21_11", "v26_1"))
    assert all("return true;" in files[e] for e in ("v1_21_1", "v1_21_5", "v1_21_9", "v1_21_11", "v26_1"))


def test_fabric_mod_json_is_filled_per_version():
    text = (MOD / "src/main/resources/fabric.mod.json").read_text()
    for placeholder in ("${version}", "${minecraft}", "${java}", "${loader}"):
        assert placeholder in text
    expanded = json.loads(re.sub(r"\$\{(\w+)\}", "1", text))
    assert expanded["depends"]["fabric-api"] == "*"


def test_build_script_reads_the_version_table():
    gradle = (MOD / "build.gradle").read_text()
    assert "versions.json" in gradle and "-Pmc" in gradle
    assert "net.fabricmc.fabric-loom-remap" in gradle and "'net.fabricmc.fabric-loom'" in gradle
    assert "src/compat/" in gradle
    props = (MOD / "gradle.properties").read_text()
    assert "minecraft_version" not in props, "game pins belong in versions.json"


def test_ci_builds_every_version_from_the_table():
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "mod/versions.json" in ci
    assert "matrix.mc" in ci and "-Pmc=${{ matrix.mc }}" in ci
    assert "pymcplaytime-all-versions" in ci


def test_hotkeys_have_translations():
    lang = json.loads((MOD / "src/main/resources/assets/pymcplaytime/lang/en_us.json").read_text())
    client = (MOD / "src/client/java/dev/pymc/pymcb/client/PymcPlaytimeClient.java").read_text()
    for key in re.findall(r'registerKey\("([^"]+)"\)', client):
        assert key in lang
    assert "key.categories.pymcplaytime" in lang and "key.category.pymcplaytime.recorder" in lang


def test_build_all_scripts_cover_both_shells():
    sh = (MOD / "build-all.sh").read_text()
    ps = (MOD / "build-all.ps1").read_text()
    assert "versions.json" in sh and "versions.json" in ps
    assert "-Pmc=" in sh and "-Pmc=" in ps
