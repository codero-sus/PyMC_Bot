package dev.pymc.pymcb.gametest;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import dev.pymc.pymcb.Json;
import dev.pymc.pymcb.PlaytimeConfig;
import dev.pymc.pymcb.PymcPlaytimeMod;
import dev.pymc.pymcb.client.BotLink;
import dev.pymc.pymcb.client.PlaytimeRecorder;
import dev.pymc.pymcb.client.PymcPlaytimeClient;
import dev.pymc.pymcb.client.PymcBotScreen;
import dev.pymc.pymcb.client.compat.Mc;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.fabricmc.fabric.api.client.gametest.v1.context.TestSingleplayerContext;
import net.fabricmc.loader.api.FabricLoader;

/**
 * In-game smoke test, run by CI on every Minecraft version that has Fabric's client game test
 * API (1.21.4 and newer): it launches the real game with the mod, creates a singleplayer world,
 * records a few seconds of scripted play and checks the dataset that comes out.
 *
 * <p>Only compiled into the jar when building with {@code -PgameTest} (release jars never contain
 * it). Run it locally with {@code ./gradlew runProductionClientGameTest -Pmc=<version> -PgameTest}.
 * Screenshots land in {@code build/run/gametest/screenshots/}, the recorded dataset in
 * {@code build/run/gametest/pymcbot/}.
 *
 * <p>Only API calls that are identical from 1.21.4 to 26.3 are used, so this one file serves every
 * version (no chunk-wait helper: it was renamed in 26.1 and removed in 26.3).
 */
@SuppressWarnings("UnstableApiUsage")
public class PymcClientGameTest implements FabricClientGameTest {
    private static PlaytimeRecorder recorder() {
        return PymcPlaytimeClient.recorder();
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError("[pymc gametest] " + message);
        }
    }

    @Override
    public void runTest(ClientGameTestContext context) {
        StringBuilder report = new StringBuilder();
        report.append(PymcPlaytimeMod.NAME).append(' ').append(PymcPlaytimeMod.modVersion())
              .append(" on Minecraft ").append(PymcPlaytimeMod.minecraftVersion())
              .append(" (compat ").append(Mc.ERA).append(")\n");

        // The mod is PyMC Bot (id pymcbot); the pre-0.6 id still resolves through "provides".
        check(FabricLoader.getInstance().isModLoaded("pymcbot"), "mod id pymcbot is not loaded");
        check(FabricLoader.getInstance().isModLoaded("pymcplaytime"), "the old id pymcplaytime is not provided");
        check(PlaytimeConfig.file().getFileName().toString().equals("pymcbot.properties"),
                "config file should be pymcbot.properties, is " + PlaytimeConfig.file());

        // /pymcbot bot ...: the panel link's JSON reader, option parser and URL check (no panel needed).
        Object doc = Json.parse("{\"ok\": true, \"detail\": \"chopped a tree: 6 logs\", \"n\": [1, 2.5, null, \"\\u00e9\"]}");
        check(Boolean.TRUE.equals(Json.get(doc, "ok")), "Json.parse lost a boolean: " + doc);
        check("chopped a tree: 6 logs".equals(Json.get(doc, "detail")), "Json.parse lost a string: " + doc);
        check(String.valueOf(Json.get(doc, "n")).equals("[1.0, 2.5, null, \u00e9]"), "Json.parse lost an array: " + doc);
        check(BotLink.options("count=8 oak_log").equals("{\"count\":8,\"name\":\"oak_log\"}"),
                "skill options: " + BotLink.options("count=8 oak_log"));
        check(PlaytimeConfig.panelProp("http://192.168.1.20:8000/", "x").equals("http://192.168.1.20:8000"), "panel URL");
        check(PlaytimeConfig.panelProp("file:///etc/passwd", "x").equals("x"), "panel URL must be http(s)");
        check(PlaytimeConfig.tokenProp("Bearer pymc_Ab-9_xyz", "x").equals("pymc_Ab-9_xyz"), "panel token");
        check(PlaytimeConfig.tokenProp("", "x").isEmpty(), "an empty panel token clears it");
        check(PlaytimeConfig.tokenProp("pymc bad token\n", "x").equals("x"), "panel token must be URL-safe");
        report.append("bot link: ok (").append(BotLink.panel()).append(")\n");

        try (TestSingleplayerContext singleplayer = context.worldBuilder().create()) {
            context.waitFor(client -> client.player != null && client.level != null);
            context.waitTicks(60); // let the first chunks render so the screenshots show a world

            check(context.computeOnClient(client -> recorder() != null), "client entrypoint did not create the recorder");
            // Minecraftism VL: a small screenshot every second while this test records.
            context.runOnClient(client -> {
                PlaytimeConfig.get().setScreenshots(true);
                PlaytimeConfig.get().setScreenshotSeconds(1);
            });
            String started = context.computeOnClient(client -> recorder().start());
            check(context.computeOnClient(client -> recorder().isActive()), "recording did not start: " + started);
            report.append("start: ").append(started.replace('\n', ' ')).append('\n');

            // Scripted play: walk forward, jump, strafe, turn the camera.
            context.getInput().holdKeyFor(options -> options.keyUp, 40);
            context.getInput().pressKey(options -> options.keyJump);
            context.waitTicks(10);
            context.getInput().holdKeyFor(options -> options.keyLeft, 20);
            context.getInput().moveCursor(200.0, 0.0);
            context.getInput().holdKeyFor(options -> options.keyUp, 20);
            context.waitTicks(5);

            PlaytimeRecorder.HudState playing = context.computeOnClient(client -> recorder().hudState());
            report.append("after play: ").append(playing.label()).append(' ').append(playing.samples()).append(" samples\n");
            check(playing.samples() > 0, "no samples were written while walking around");
            check("REC".equals(playing.label()), "HUD badge state should be REC, was " + playing.label());
            context.takeScreenshot("pymc-1-recording-hud");
            context.waitTicks(10); // the PNGs are written on a background thread
            String shotStatus = context.computeOnClient(client -> recorder().screenshotStatus());
            long shotsWritten = context.computeOnClient(client -> recorder().screenshotsWritten());
            report.append("vision: ").append(shotStatus).append('\n');
            check(shotsWritten > 0, "no Minecraftism VL screenshot was saved: " + shotStatus);
            context.runOnClient(client -> PlaytimeConfig.get().setScreenshots(false));

            // The PyMC Bot settings page opens, and sampling holds while it is open (pauseInMenus=true).
            context.setScreen(() -> new PymcBotScreen(null, PymcBotScreen.Page.RECORDER));
            context.waitTicks(5);
            long before = context.computeOnClient(client -> recorder().totalSamples());
            context.waitTicks(20);
            long after = context.computeOnClient(client -> recorder().totalSamples());
            check(context.computeOnClient(client -> Mc.screenOpen(client)), "settings page did not open");
            check(before == after, "samples were written while a menu was open (" + before + " -> " + after + ")");

            // Every tab renders (one screenshot each), and a page reset restores the defaults.
            int shot = 2;
            for (PymcBotScreen.Page page : PymcBotScreen.Page.values()) {
                context.setScreen(() -> new PymcBotScreen(null, page));
                context.waitTicks(3);
                context.takeScreenshot("pymc-" + shot++ + "-settings-" + page.name().toLowerCase(java.util.Locale.ROOT));
            }
            boolean blocksRestored = context.computeOnClient(client -> {
                PlaytimeConfig.get().setIncludeBlocks(false);
                PlaytimeConfig.get().setMaxNearby(64);
                PlaytimeConfig.get().resetRecording();
                return PlaytimeConfig.get().includeBlocks && PlaytimeConfig.get().maxNearby == 12;
            });
            check(blocksRestored, "Reset page did not restore the Recording defaults");
            boolean powerRestored = context.computeOnClient(client -> {
                PlaytimeConfig cfg = PlaytimeConfig.get();
                cfg.setSampleTicks(7);
                cfg.setTurnDegrees(12.5);
                cfg.setMinEpisodeSeconds(90);
                cfg.setIdleKeepPercent(25);
                cfg.resetPower();
                return cfg.sampleTicks == 2 && cfg.turnDegrees == 1.0f && cfg.minEpisodeSeconds == 0
                        && cfg.idleKeepPercent == 100;
            });
            check(powerRestored, "Reset page did not restore the Power (slider) defaults");
            // Export filter maths: 25% keeps the first idle sample, then every fourth one.
            PlaytimeRecorder.IdleThinner thinner = new PlaytimeRecorder.IdleThinner(25);
            StringBuilder kept = new StringBuilder();
            for (String action : new String[]{"forward", "none", "none", "none", "none", "none", "jump", "none", "none"}) {
                kept.append(thinner.keep(action) ? '1' : '0');
            }
            check("110001110".equals(kept.toString()), "idle thinning kept " + kept + " instead of 110001110");
            check(PlaytimeRecorder.tickOf("{\"tick\": 1234, \"activity\": \"none\"}") == 1234, "tick parsing");
            report.append("settings page: ").append(PymcBotScreen.Page.values().length).append(" tabs rendered\n");
            context.setScreen(() -> null);
            context.waitTicks(5);

            // Pause / resume through the same calls the hotkey uses.
            context.computeOnClient(client -> recorder().pause());
            check(context.computeOnClient(client -> recorder().isPaused()), "pause did not pause");
            context.takeScreenshot("pymc-" + shot + "-paused-hud");
            context.computeOnClient(client -> recorder().resume());
            context.getInput().holdKeyFor(options -> options.keyDown, 10);

            String stopped = context.computeOnClient(client -> recorder().stop());
            String exported = context.computeOnClient(client -> recorder().export());
            Path dataset = context.computeOnClient(client -> recorder().datasetPath());
            report.append("stop: ").append(stopped.replace('\n', ' ')).append('\n');
            report.append("export: ").append(exported.replace('\n', ' ')).append('\n');

            List<String> lines;
            try {
                lines = Files.readAllLines(dataset, StandardCharsets.UTF_8);
            } catch (IOException e) {
                throw new AssertionError("[pymc gametest] could not read " + dataset + ": " + e, e);
            }
            Map<String, Integer> actions = new TreeMap<>();
            for (String line : lines) {
                int at = line.indexOf("\\\"action\\\": \\\"");
                if (at >= 0) {
                    int from = at + "\\\"action\\\": \\\"".length();
                    int to = line.indexOf("\\\"", from);
                    if (to > from) {
                        actions.merge(line.substring(from, to), 1, Integer::sum);
                    }
                }
            }
            report.append("dataset: ").append(lines.size()).append(" line(s) at ").append(dataset).append('\n');
            report.append("actions: ").append(actions).append('\n');
            check(!lines.isEmpty(), "export wrote an empty dataset");
            check(lines.get(0).contains("\"instruction\""), "dataset lines are not in the trainer's format");
            check(actions.containsKey("forward"), "walking forward was not recorded: " + actions);
            boolean shotField = lines.stream().anyMatch(line -> line.contains("\"shot\": \"shots/"));
            check(shotField, "no sample references a screenshot (\"shot\" field)");
            Path firstShot = null;
            for (String line : lines) {
                int at = line.indexOf("\"shot\": \"");
                if (at >= 0) {
                    int from = at + "\"shot\": \"".length();
                    firstShot = dataset.getParent().resolve(line.substring(from, line.indexOf('"', from)));
                    break;
                }
            }
            check(firstShot != null && Files.isRegularFile(firstShot), "screenshot file missing: " + firstShot);
            report.append("first screenshot: ").append(firstShot).append('\n');
        } finally {
            Path out = FabricLoader.getInstance().getGameDir().resolve("pymc-gametest.txt");
            try {
                Files.writeString(out, report.toString(), StandardCharsets.UTF_8);
            } catch (IOException ignored) {
                // The report is a convenience for the CI artifact; the assertions are what count.
            }
            PymcPlaytimeMod.LOGGER.info("[pymc gametest]\n{}", report);
        }
    }
}
