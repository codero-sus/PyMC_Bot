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
    # GUI drawing: GuiGraphics.drawString (1.21.x) became GuiGraphicsExtractor.text (26.x).
    r"\bGuiGraphics\b",
    r"\bGuiGraphicsExtractor\b",
    r"drawString\(",
    r"extractRenderState",
    r"\bHudRenderCallback\b",
    r"\bHudElementRegistry\b",
    # Opening a screen: Minecraft.setScreen until 26.1, Minecraft.gui.setScreen from 26.2.
    r"(?:client|mc|minecraft|getInstance\(\))\.setScreen\(",
    r"gui\.setScreen\(",
    # F1 state: Options.hideGui until 26.1, Minecraft.gui.hud.isHidden() from 26.2.
    r"\bhideGui\b",
    r"hud\.isHidden\(\)",
]

COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)


def code_only(text: str) -> str:
    """Java source without comments, so Javadoc may name the APIs it explains."""
    return COMMENTS.sub("", text)


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
        text = code_only(path.read_text())
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
        assert 'String CATEGORY = "key.categories.pymcbot"' in files[era]
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
    release = json.loads(re.sub(r"\$\{(\w+)\}", "1", text.replace("${gametest_entrypoint}", "")))
    assert release["depends"]["fabric-api"] == "*"
    assert list(release["entrypoints"]) == ["client", "modmenu"], "release jars carry no test entrypoint"
    assert release["id"] == "pymcbot" and release["name"] == "PyMC Bot"
    assert release["provides"] == ["pymcplaytime"], "the pre-0.6 id must keep resolving"
    assert release["entrypoints"]["modmenu"] == ["dev.pymc.pymcb.client.PymcModMenu"]
    assert release["icon"] == "assets/pymcbot/icon.png" and (MOD / "src/main/resources" / release["icon"]).is_file()


def gametest_expansion() -> str:
    """The value build.gradle substitutes for ${gametest_entrypoint} under -PgameTest."""
    gradle = (MOD / "build.gradle").read_text()
    match = re.search(r"gametest_entrypoint: gameTest\s*\?\s*'([^']*)'\s*:\s*''", gradle)
    assert match, "build.gradle must expand ${gametest_entrypoint} only for -PgameTest"
    return match.group(1).encode().decode("unicode_escape")


def test_game_test_builds_add_the_client_gametest_entrypoint():
    text = (MOD / "src/main/resources/fabric.mod.json").read_text()
    expanded = json.loads(re.sub(r"\$\{(\w+)\}", "1", text.replace("${gametest_entrypoint}", gametest_expansion())))
    (entry,) = expanded["entrypoints"]["fabric-client-gametest"]
    source = MOD / "src/gametest/java" / (entry.replace(".", "/") + ".java")
    assert source.is_file(), f"{entry} has no source file"
    body = source.read_text()
    assert "implements FabricClientGameTest" in body and "runTest(ClientGameTestContext" in body
    assert "takeScreenshot(" in body and "recorder().start()" in body and "recorder().export()" in body


def test_game_test_code_never_reaches_release_jars():
    gradle = (MOD / "build.gradle").read_text()
    assert "providers.gradleProperty('gameTest')" in gradle
    assert "archivesName = gameTest ? 'pymcbot-gametest' : 'pymcbot'" in gradle
    block = gradle[gradle.index("if (gameTest) {"):]
    assert block.index("srcDir 'src/gametest/java'") < block.index("}"), "test sources only under -PgameTest"
    assert "ClientProductionRunTask" in gradle and "runProductionClientGameTest" in gradle
    assert "-Dfabric.client.gametest" in gradle and "productionRuntimeMods" in gradle
    assert not list((MOD / "src/client").rglob("*GameTest*.java")), "tests belong in src/gametest"


@pytest.mark.parametrize("entry", versions(), ids=lambda v: v["minecraft"])
def test_in_game_tests_run_where_fabric_has_the_api(entry):
    parts = [int(x) for x in entry["minecraft"].split(".")]
    assert entry["gametest"] is (parts >= [1, 21, 4]), "Fabric's client game test API starts at 1.21.4"


def test_gui_bridge_matches_each_era():
    files = {era: code_only(path.read_text()) for era, path in compat_files().items()}
    for era in ("v1_21_1", "v1_21_5"):
        assert "HudRenderCallback.EVENT.register" in files[era], era
    for era in ("v1_21_9", "v1_21_11", "v26_1", "v26_2"):
        assert "HudElementRegistry.addLast" in files[era], era
    for era in ("v1_21_1", "v1_21_5", "v1_21_9", "v1_21_11"):
        assert "GuiGraphics graphics" in files[era] and "graphics.drawString(font" in files[era]
        assert "public void render(GuiGraphics graphics" in files[era]
    for era in ("v26_1", "v26_2"):
        assert "GuiGraphicsExtractor graphics" in files[era] and "graphics.text(font" in files[era]
        assert "public void extractRenderState(GuiGraphicsExtractor graphics" in files[era]
        assert "HudRenderCallback" not in files[era], "removed in 26.1"
    assert "client.gui.setScreen(screen)" in files["v26_2"]
    assert "client.gui.hud.isHidden()" in files["v26_2"]
    assert all("client.options.hideGui" in files[e] for e in ("v1_21_1", "v1_21_5", "v1_21_9", "v1_21_11", "v26_1"))
    assert all("client.setScreen(screen)" in files[e] for e in ("v1_21_1", "v1_21_5", "v1_21_9", "v1_21_11", "v26_1"))
    assert "ResourceLocation.fromNamespaceAndPath(\"pymcbot\", \"recorder_badge\")" in files["v1_21_9"]
    for era, text in files.items():
        assert "abstract static class PaintedScreen extends Screen" in text, era
        # the settings page gets the mouse position for its hover help
        assert "paint(canvas(graphics), mouseX, mouseY);" in text, era
        assert "protected abstract void paint(HudCanvas canvas, int mouseX, int mouseY);" in text, era
        assert "pymcplaytime" not in text, era


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
    assert "pymcbot-all-versions" in ci and "pymcplaytime" not in ci


def test_ci_runs_the_in_game_test_and_publishes_a_release():
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "runProductionClientGameTest -Pmc=${{ matrix.mc }} -PgameTest" in ci
    assert "steps.gt.outputs.enabled == 'true'" in ci and ".gametest" in ci
    assert "xvfb" in ci and "mod/build/run/gametest/screenshots/" in ci
    # SDL3 (26.3+) needs a 24-bit screen; the task must use CI's display instead of xvfb-run's 8-bit one.
    assert "-screen 0 1920x1080x24" in ci and 'echo "DISPLAY=:99" >> "$GITHUB_ENV"' in ci
    assert "useXVFB = providers.environmentVariable('DISPLAY').map { false }" in (MOD / "build.gradle").read_text()
    release = ci[ci.index("\n  release:"):]
    assert "needs: [mod-bundle, release-kit]" in release and "contents: write" in release
    # the release carries everything: jars + all-versions zip + updater, bot and LLM kit + checksums
    assert 'gh release create "$tag" dist/*' in release and "all-versions.zip" in release
    assert "release_kit.py --checksums dist" in release and "name: release-kit" in release
    assert "scripts/release_kit.py --out release-kit" in ci
    assert "gh release create" in release and "scripts/release_notes.py" in release
    assert "--prerelease" in release and "refs/heads/main" in release
    assert "contains game test code" in ci, "CI must check that release jars carry no test code"
    assert "./gradlew downloadAssets" in ci and "for attempt in 1 2 3" in ci, "flaky asset downloads are retried"


def test_hotkeys_have_translations():
    lang = json.loads((MOD / "src/main/resources/assets/pymcbot/lang/en_us.json").read_text())
    client = (MOD / "src/client/java/dev/pymc/pymcb/client/PymcPlaytimeClient.java").read_text()
    for key in re.findall(r'registerKey\("([^"]+)"\)', client):
        assert key in lang
    assert "key.categories.pymcbot" in lang and "key.category.pymcbot.recorder" in lang
    assert lang["key.categories.pymcbot"] == "PyMC Bot" == lang["key.category.pymcbot.recorder"]
    assert lang["key.pymcbot.settings"] == "Open PyMC Bot settings"
    assert not (MOD / "src/main/resources/assets/pymcplaytime").exists()


def test_settings_screen_and_hud_are_wired_up():
    client_dir = MOD / "src/client/java/dev/pymc/pymcb/client"
    client = (client_dir / "PymcPlaytimeClient.java").read_text()
    assert "Mc.registerHud(RecorderHud::paint)" in client
    assert "new PymcBotScreen(null)" in client and "openSettingsNextTick" in client
    commands = (client_dir / "PymcCommands.java").read_text()
    assert 'guarded("settings")' in commands and 'guarded("config")' in commands and 'literal("hud")' in commands
    assert 'ROOTS = {"pymcbot", "pymc"}' in commands and "PymcBotScreen.CHAT_PREFIX" in commands
    assert not (client_dir / "PymcSettingsScreen.java").exists()
    screen = (client_dir / "PymcBotScreen.java").read_text()
    assert "extends Mc.PaintedScreen" in screen and "Mc.openScreen(" in screen
    assert "Made with \\u2764 by codero-sus" in screen
    config = (MOD / "src/main/java/dev/pymc/pymcb/PlaytimeConfig.java").read_text()
    for key in ("showHud", "hudCorner"):
        assert f'props.setProperty("{key}"' in config and f'"{key}"' in config
    hud = (client_dir / "RecorderHud.java").read_text()
    # 1.21.6+ draws 0x00RRGGBB as fully transparent: every colour constant needs its alpha byte.
    colours = re.findall(r"static final int [A-Z_]+ = 0x([0-9A-Fa-f]{8});", hud)
    assert len(colours) >= 5
    for colour in colours:
        assert not colour.startswith("00"), colour


def test_build_all_scripts_cover_both_shells():
    sh = (MOD / "build-all.sh").read_text()
    ps = (MOD / "build-all.ps1").read_text()
    assert "versions.json" in sh and "versions.json" in ps
    assert "-Pmc=" in sh and "-Pmc=" in ps


def test_pymc_bot_settings_page_covers_every_option():
    """The PyMC Bot config & settings page has a button for every saved option."""
    client_dir = MOD / "src/client/java/dev/pymc/pymcb/client"
    screen = (client_dir / "PymcBotScreen.java").read_text()
    config = (MOD / "src/main/java/dev/pymc/pymcb/PlaytimeConfig.java").read_text()
    saved = re.findall(r'props\.setProperty\("([a-zA-Z]+)"', config)
    saved = [key for key in saved if not key.startswith("_")]
    assert len(saved) >= 15
    missing = [key for key in saved if f"cfg().{key}" not in screen]
    assert not missing, f"options without a button on the settings page: {missing}"
    for page in ('RECORDER("Recorder")', 'RECORDING("Recording")', 'ADVANCED("Advanced")',
                 'POWER("Power")', 'HUD("HUD & chat")', 'ABOUT("About")'):
        assert page in screen
    for reset in ("resetRecording", "resetAdvanced", "resetPower", "resetHud"):
        assert f"public void {reset}()" in config and f"cfg().{reset}()" in screen
    assert 'PymcPlaytimeMod.NAME + " - Settings"' in screen
    assert "option.hovered(mouseX, mouseY)" in screen, "hover help"
    assert "RecorderHud.draw(offset(" in screen, "live badge preview"
    assert "slider.hovered(mouseX, mouseY)" in screen, "hover help for the Power sliders"
    # only widgets that exist unchanged on every version (no edit boxes / tooltips); the vanilla
    # slider is used solely through PowerSlider, so a version shim would only touch that file
    for risky in ("AbstractSliderButton", "EditBox", "CycleButton", "Tooltip", "getWidth()"):
        assert risky not in screen, risky


def test_power_sliders_stay_inside_one_wrapper():
    """Power tab: one wrapper class owns the vanilla slider and only uses its shared API."""
    client_dir = MOD / "src/client/java/dev/pymc/pymcb/client"
    users = [path.name for path in client_dir.glob("*.java") if "AbstractSliderButton" in path.read_text()]
    assert users == ["PowerSlider.java"]
    slider = (client_dir / "PowerSlider.java").read_text()
    assert "extends AbstractSliderButton" in slider
    assert "protected void updateMessage()" in slider and "protected void applyValue()" in slider
    assert "Math.abs(next - stored) > 1e-9" in slider, "the config file is only written when the value changes"
    for risky in ("renderWidget", "onClick", "onDrag", "keyPressed", "mouseClicked", "GuiGraphics"):
        assert risky not in slider, risky
    screen = (client_dir / "PymcBotScreen.java").read_text()
    config = (MOD / "src/main/java/dev/pymc/pymcb/PlaytimeConfig.java").read_text()
    for key in ("sampleTicks", "turnDegrees", "moveEpsilon", "idleSkipSeconds", "maxNearby", "maxEntities",
                "entityRange", "maxGroundItems", "minEpisodeSeconds", "idleKeepPercent"):
        assert f"() -> cfg().{key}" in screen, f"no Power slider for {key}"
        assert f"        {key} = d.{key};" in config.split("public void resetPower()")[1].split("save();")[0], key
    recorder = (client_dir / "PlaytimeRecorder.java").read_text()
    assert "new IdleThinner(cfg.idleKeepPercent)" in recorder and "lastTick(file) < minTicks" in recorder


def test_mod_menu_integration_ships_no_stubs():
    client_root = MOD / "src/client/java"
    stubs = client_root / "com/terraformersmc/modmenu/api"
    assert (stubs / "ModMenuApi.java").is_file() and (stubs / "ConfigScreenFactory.java").is_file()
    factory = (stubs / "ConfigScreenFactory.java").read_text()
    assert "S create(Screen parent);" in factory, "must match Mod Menu's real signature"
    hook = (client_root / "dev/pymc/pymcb/client/PymcModMenu.java").read_text()
    assert "implements ModMenuApi" in hook and "new PymcBotScreen(parent)" in hook
    gradle = (MOD / "build.gradle").read_text()
    assert "def modMenuStubs = 'com/terraformersmc/**'" in gradle
    assert gradle.count("exclude modMenuStubs") == 2, "excluded from the jar and the sources jar"
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "ships the Mod Menu API stubs" in ci


def test_mod_is_named_pymc_bot_and_migrates_old_config():
    config = (MOD / "src/main/java/dev/pymc/pymcb/PlaytimeConfig.java").read_text()
    assert 'FILE_NAME = "pymcbot.properties"' in config
    assert 'LEGACY_FILE_NAME = "pymcplaytime.properties"' in config and "migrate" in config
    mod = (MOD / "src/main/java/dev/pymc/pymcb/PymcPlaytimeMod.java").read_text()
    assert 'MOD_ID = "pymcbot"' in mod and 'NAME = "PyMC Bot"' in mod
    assert "rootProject.name = 'pymcbot'" in (MOD / "settings.gradle").read_text()
    # Nothing a player reads still says "PyMC Playtime" or prefixes chat with "[pymc]".
    for path in (MOD / "src").rglob("*.java"):
        text = code_only(path.read_text())
        assert '"[pymc] "' not in text, path
        assert "PyMC Playtime Recorder\"" not in text and "\"PyMC Playtime" not in text, path
