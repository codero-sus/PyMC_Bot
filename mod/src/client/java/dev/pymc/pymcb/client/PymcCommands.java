package dev.pymc.pymcb.client;

import java.util.Locale;

import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import dev.pymc.pymcb.Json;
import dev.pymc.pymcb.PlaytimeConfig;
import dev.pymc.pymcb.PymcPlaytimeMod;
import dev.pymc.pymcb.client.compat.Mc;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;

/**
 * {@code /pymcbot ...} (short alias {@code /pymc ...}) - client side commands of the PyMC Bot mod.
 *
 * <pre>
 * /pymcbot record start|stop|toggle|status
 * /pymcbot record episode          close the current episode and start a new one
 * /pymcbot pause | resume          stop sampling for a moment without closing the episode
 * /pymcbot export                  write dataset.jsonl for the Python trainer
 * /pymcbot sample &lt;ticks&gt;          change the sampling rate (2 ticks = 100 ms)
 * /pymcbot advanced on|off|status   record entities, items, armour and drops too
 * /pymcbot entities &lt;n&gt;            how many entities to record per sample (advanced)
 * /pymcbot drops &lt;n&gt;               how many dropped item stacks to record (advanced)
 * /pymcbot autorecord on|off        start recording automatically when joining a world
 * /pymcbot menus on|off             skip samples while a screen is open (default on)
 * /pymcbot idle &lt;seconds&gt;          stop writing after this long without input (0 = never)
 * /pymcbot notify on|off            chat lines for hotkey / auto-record changes
 * /pymcbot hud on|off|status        the REC / PAUSED badge in a corner of the screen
 * /pymcbot hud corner &lt;corner&gt;      top_left, top_right, bottom_left or bottom_right
 * /pymcbot settings | config       open the PyMC Bot settings page (every option above, in tabs)
 * /pymcbot status                  where the dataset lives, how big it is
 * /pymcbot train                   print the exact command to train on this dataset
 * /pymcbot version                 mod, Minecraft and compatibility layer versions
 * /pymcbot about                   author, links and credits
 *
 * /pymcbot bot status | list        the PyMC Bot panel's bots (python -m pymc_bot must be running)
 * /pymcbot bot skills               the 44 skills
 * /pymcbot bot do &lt;skill&gt; [k=v ..]  run a skill, e.g. /pymcbot bot do chop_tree count=12
 * /pymcbot bot come | follow | stop | home
 * /pymcbot bot say &lt;text&gt;
 * /pymcbot bot auto on|off          the autopilot (auto-sleep, eat, fight back, torches...)
 * /pymcbot bot ai start|stop
 * /pymcbot bot panel [url]          which panel to talk to (default http://127.0.0.1:8000)
 * </pre>
 *
 * <p>Every command that changes what gets recorded requires an <em>attended</em> run on
 * Minecraft 26.2+ (see {@link Mc#attended}), so a server cannot start a recording through a
 * clickable chat message. Read-only commands are always available.
 *
 * <p>Made with ❤ by codero-sus - https://github.com/codero-sus - https://the-codero.web.app/</p>
 */
public final class PymcCommands {
    /** Both spellings work; {@code /pymcbot} is the documented one, {@code /pymcbot} the short alias. */
    static final String[] ROOTS = {"pymcbot", "pymc"};

    /** Shown by {@code /pymcbot about}. */
    static final String ABOUT = PymcPlaytimeMod.NAME + " mod - part of PyMC_Bot\n"
            + "Made with \u2764 by codero-sus\n"
            + "GitHub:  https://github.com/codero-sus\n"
            + "Website: https://the-codero.web.app/\n"
            + "Source:  https://github.com/codero-sus/PyMC_Bot (PolyForm Noncommercial 1.0.0 - free for personal, non-commercial use)";

    private final PlaytimeRecorder recorder;

    public PymcCommands(PlaytimeRecorder recorder) {
        this.recorder = recorder;
    }

    /** A literal that changes recorder state: only runs when the player typed it themselves. */
    private static LiteralArgumentBuilder<FabricClientCommandSource> guarded(String name) {
        return Mc.literal(name).requires(Mc::attended);
    }

    private static LiteralArgumentBuilder<FabricClientCommandSource> literal(String name) {
        return Mc.literal(name);
    }

    public void register() {
        ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) -> {
            LiteralArgumentBuilder<FabricClientCommandSource> record = literal("record")
                    .then(guarded("start").executes(ctx -> feedback(ctx.getSource(), recorder.start())))
                    .then(guarded("stop").executes(ctx -> feedback(ctx.getSource(), recorder.stop())))
                    .then(guarded("toggle").executes(ctx -> feedback(ctx.getSource(), recorder.toggle())))
                    .then(guarded("episode").executes(ctx -> feedback(ctx.getSource(), recorder.endEpisode())))
                    .then(literal("status").executes(ctx -> feedback(ctx.getSource(), recorder.status())));

            LiteralArgumentBuilder<FabricClientCommandSource> sample = guarded("sample")
                    .then(Mc.argument("ticks", IntegerArgumentType.integer(1, 200))
                            .executes(ctx -> {
                                int ticks = IntegerArgumentType.getInteger(ctx, "ticks");
                                PlaytimeConfig.get().setSampleTicks(ticks);
                                return feedback(ctx.getSource(),
                                        "sampling every " + ticks + " tick(s) (~" + ticks * 50 + " ms)");
                            }));

            LiteralArgumentBuilder<FabricClientCommandSource> advanced = literal("advanced")
                    .then(guarded("on").executes(ctx -> {
                        PlaytimeConfig.get().setAdvanced(true);
                        return feedback(ctx.getSource(),
                                "advanced recording ON - entities, items, held item, armour and drops "
                                + "are written from now on; train with --advanced");
                    }))
                    .then(guarded("off").executes(ctx -> {
                        PlaytimeConfig.get().setAdvanced(false);
                        return feedback(ctx.getSource(), "advanced recording off - back to the basic schema");
                    }))
                    .then(literal("status").executes(ctx -> feedback(ctx.getSource(), PlaytimeConfig.get().describe())));

            LiteralArgumentBuilder<FabricClientCommandSource> screenshots = literal("screenshots")
                    .then(guarded("on").executes(ctx -> {
                        PlaytimeConfig.get().setScreenshots(true);
                        PlaytimeConfig cfg = PlaytimeConfig.get();
                        return feedback(ctx.getSource(), "screenshots ON - one every " + cfg.screenshotSeconds
                                + "s at " + cfg.screenshotWidth + " px wide while recording (for the Minecraftism VL model)");
                    }))
                    .then(guarded("off").executes(ctx -> {
                        PlaytimeConfig.get().setScreenshots(false);
                        return feedback(ctx.getSource(), "screenshots off");
                    }))
                    .then(guarded("every")
                            .then(Mc.argument("seconds", IntegerArgumentType.integer(1, 600))
                                    .executes(ctx -> {
                                        int seconds = IntegerArgumentType.getInteger(ctx, "seconds");
                                        PlaytimeConfig.get().setScreenshotSeconds(seconds);
                                        return feedback(ctx.getSource(), "one screenshot every " + seconds + "s while recording");
                                    })))
                    .then(guarded("width")
                            .then(Mc.argument("pixels", IntegerArgumentType.integer(160, 1920))
                                    .executes(ctx -> {
                                        int pixels = IntegerArgumentType.getInteger(ctx, "pixels");
                                        PlaytimeConfig.get().setScreenshotWidth(pixels);
                                        return feedback(ctx.getSource(), "screenshots are scaled to " + pixels + " px wide");
                                    })))
                    .then(literal("status").executes(ctx -> feedback(ctx.getSource(),
                            PymcPlaytimeClient.recorder() == null ? "recorder not ready"
                                    : PymcPlaytimeClient.recorder().screenshotStatus())));

            LiteralArgumentBuilder<FabricClientCommandSource> entities = guarded("entities")
                    .then(Mc.argument("count", IntegerArgumentType.integer(0, 128))
                            .executes(ctx -> {
                                int count = IntegerArgumentType.getInteger(ctx, "count");
                                PlaytimeConfig.get().setMaxEntities(count);
                                return feedback(ctx.getSource(), "recording up to " + count + " entities per sample");
                            }));

            LiteralArgumentBuilder<FabricClientCommandSource> drops = guarded("drops")
                    .then(Mc.argument("count", IntegerArgumentType.integer(0, 64))
                            .executes(ctx -> {
                                int count = IntegerArgumentType.getInteger(ctx, "count");
                                PlaytimeConfig.get().setMaxGroundItems(count);
                                return feedback(ctx.getSource(), "recording up to " + count + " dropped stack(s) per sample");
                            }));

            LiteralArgumentBuilder<FabricClientCommandSource> autorecord = guarded("autorecord")
                    .then(literal("on").executes(ctx -> {
                        PlaytimeConfig.get().setAutoRecord(true);
                        return feedback(ctx.getSource(), "auto-record ON - recording starts whenever you join a world or server");
                    }))
                    .then(literal("off").executes(ctx -> {
                        PlaytimeConfig.get().setAutoRecord(false);
                        return feedback(ctx.getSource(), "auto-record off - start with /pymcbot record start or the hotkey");
                    }));

            LiteralArgumentBuilder<FabricClientCommandSource> menus = guarded("menus")
                    .then(literal("on").executes(ctx -> {
                        PlaytimeConfig.get().setPauseInMenus(true);
                        return feedback(ctx.getSource(), "samples are skipped while a screen (inventory, chat, menu) is open");
                    }))
                    .then(literal("off").executes(ctx -> {
                        PlaytimeConfig.get().setPauseInMenus(false);
                        return feedback(ctx.getSource(), "samples are recorded even while a screen is open");
                    }));

            LiteralArgumentBuilder<FabricClientCommandSource> idle = guarded("idle")
                    .then(Mc.argument("seconds", IntegerArgumentType.integer(0, 3600))
                            .executes(ctx -> {
                                int seconds = IntegerArgumentType.getInteger(ctx, "seconds");
                                PlaytimeConfig.get().setIdleSkipSeconds(seconds);
                                return feedback(ctx.getSource(), seconds == 0
                                        ? "idle skipping off - every sample is written"
                                        : "samples stop after " + seconds + " s without input and resume on the next action");
                            }));

            LiteralArgumentBuilder<FabricClientCommandSource> notify = guarded("notify")
                    .then(literal("on").executes(ctx -> {
                        PlaytimeConfig.get().setNotifications(true);
                        return feedback(ctx.getSource(), "chat notifications on");
                    }))
                    .then(literal("off").executes(ctx -> {
                        PlaytimeConfig.get().setNotifications(false);
                        return feedback(ctx.getSource(), "chat notifications off");
                    }));

            LiteralArgumentBuilder<FabricClientCommandSource> corner = guarded("corner");
            for (String name : PlaytimeConfig.HUD_CORNERS) {
                corner.then(literal(name).executes(ctx -> {
                    PlaytimeConfig.get().setHudCorner(name);
                    return feedback(ctx.getSource(), "HUD badge moved to the " + PlaytimeConfig.cornerLabel(name).toLowerCase(Locale.ROOT) + " corner");
                }));
            }
            LiteralArgumentBuilder<FabricClientCommandSource> hud = literal("hud")
                    .then(guarded("on").executes(ctx -> {
                        PlaytimeConfig.get().setShowHud(true);
                        return feedback(ctx.getSource(), "HUD badge on - shows REC / PAUSED / IDLE / MENU while recording");
                    }))
                    .then(guarded("off").executes(ctx -> {
                        PlaytimeConfig.get().setShowHud(false);
                        return feedback(ctx.getSource(), "HUD badge off");
                    }))
                    .then(corner)
                    .then(literal("status").executes(ctx -> feedback(ctx.getSource(),
                            "HUD badge " + (PlaytimeConfig.get().showHud ? "on" : "off") + ", "
                                    + PlaytimeConfig.cornerLabel(PlaytimeConfig.get().hudCorner).toLowerCase(Locale.ROOT) + " corner")));

            LiteralArgumentBuilder<FabricClientCommandSource> bot = botCommands();

            for (String root : ROOTS) {
                dispatcher.register(literal(root)
                        .executes(ctx -> feedback(ctx.getSource(), help()))
                        .then(literal("help").executes(ctx -> feedback(ctx.getSource(), help())))
                        .then(record)
                        .then(guarded("pause").executes(ctx -> feedback(ctx.getSource(), recorder.pause())))
                        .then(guarded("resume").executes(ctx -> feedback(ctx.getSource(), recorder.resume())))
                        .then(sample)
                        .then(advanced)
                        .then(screenshots)
                        .then(entities)
                        .then(drops)
                        .then(autorecord)
                        .then(menus)
                        .then(idle)
                        .then(notify)
                        .then(hud)
                        .then(bot)
                        .then(guarded("settings").executes(ctx -> {
                            PymcPlaytimeClient.openSettings();
                            return 1;
                        }))
                        .then(guarded("config").executes(ctx -> {
                            PymcPlaytimeClient.openSettings();
                            return 1;
                        }))
                        .then(literal("param").executes(ctx -> feedback(ctx.getSource(), PlaytimeConfig.get().describe())))
                        .then(literal("status").executes(ctx -> feedback(ctx.getSource(), recorder.status())))
                        .then(guarded("export").executes(ctx -> feedback(ctx.getSource(), recorder.export())))
                        .then(literal("train").executes(ctx -> feedback(ctx.getSource(), trainHint(recorder))))
                        .then(literal("version").executes(ctx -> feedback(ctx.getSource(), version())))
                        .then(literal("about").executes(ctx -> feedback(ctx.getSource(), ABOUT))));
            }
        });
    }

    /** {@code /pymcbot bot ...}: drive the PyMC Bot panel's bots from inside the game (see {@link BotLink}). */
    private static LiteralArgumentBuilder<FabricClientCommandSource> botCommands() {
        LiteralArgumentBuilder<FabricClientCommandSource> doSkill = guarded("do")
                .then(Mc.argument("skill", StringArgumentType.word())
                        .executes(ctx -> skill(StringArgumentType.getString(ctx, "skill"), "{}"))
                        .then(Mc.argument("options", StringArgumentType.greedyString())
                                .executes(ctx -> skill(StringArgumentType.getString(ctx, "skill"),
                                        BotLink.options(StringArgumentType.getString(ctx, "options"))))));

        LiteralArgumentBuilder<FabricClientCommandSource> auto = literal("auto")
                .executes(ctx -> autopilotStatus())
                .then(literal("status").executes(ctx -> autopilotStatus()))
                .then(guarded("on").executes(ctx -> autopilot(true)))
                .then(guarded("off").executes(ctx -> autopilot(false)));

        LiteralArgumentBuilder<FabricClientCommandSource> ai = literal("ai")
                .then(guarded("start").executes(ctx -> {
                    BotLink.post("/api/ai/start", "{}", doc -> "AI started");
                    return 1;
                }))
                .then(guarded("stop").executes(ctx -> {
                    BotLink.post("/api/ai/stop", "{}", doc -> "AI stopped");
                    return 1;
                }));

        LiteralArgumentBuilder<FabricClientCommandSource> panel = literal("panel")
                .executes(ctx -> feedback(ctx.getSource(), "talking to the PyMC Bot panel at " + BotLink.panel()
                        + "\nchange it with /pymcbot bot panel http://<pc>:8000 (start the panel with --host 0.0.0.0)"))
                .then(guarded("reset").executes(ctx -> {
                    PlaytimeConfig.get().setPanelUrl(PlaytimeConfig.DEFAULT_PANEL_URL);
                    return feedback(ctx.getSource(), "panel: " + BotLink.panel());
                }))
                .then(Mc.argument("url", StringArgumentType.greedyString()).requires(Mc::attended)
                        .executes(ctx -> {
                            String url = StringArgumentType.getString(ctx, "url");
                            return feedback(ctx.getSource(), PlaytimeConfig.get().setPanelUrl(url)
                                    ? "panel: " + BotLink.panel() + " - try /pymcbot bot status"
                                    : "not a panel address: " + url + " (example: http://192.168.1.20:8000)");
                        }));

        return literal("bot")
                .executes(ctx -> feedback(ctx.getSource(), botHelp()))
                .then(literal("help").executes(ctx -> feedback(ctx.getSource(), botHelp())))
                .then(literal("status").executes(ctx -> {
                    BotLink.get("/api/status", BotLink::describeStatus);
                    return 1;
                }))
                .then(literal("list").executes(ctx -> {
                    BotLink.get("/api/fleet/status", BotLink::describeFleet);
                    return 1;
                }))
                .then(literal("skills").executes(ctx -> {
                    BotLink.get("/api/skills", BotLink::describeSkills);
                    return 1;
                }))
                .then(doSkill)
                .then(guarded("come").executes(ctx -> skill("come", "{\"player\":" + BotLink.quote(me()) + "}")))
                .then(guarded("home").executes(ctx -> skill("go_home", "{}")))
                .then(guarded("follow").executes(ctx -> action("{\"action\":\"follow\",\"player\":" + BotLink.quote(me()) + "}", "follow")))
                .then(guarded("stop").executes(ctx -> action("{\"action\":\"stop\"}", "stop")))
                .then(guarded("say").then(Mc.argument("text", StringArgumentType.greedyString())
                        .executes(ctx -> {
                            String text = StringArgumentType.getString(ctx, "text");
                            BotLink.post("/api/bot/chat", "{\"message\":" + BotLink.quote(text) + "}", doc -> "said: " + text);
                            return 1;
                        })))
                .then(auto)
                .then(ai)
                .then(panel);
    }

    private static int skill(String name, String optionsJson) {
        BotLink.post("/api/bot/skill", BotLink.skillBody(name, optionsJson), BotLink.describeResult(name));
        return 1;
    }

    private static int action(String body, String what) {
        BotLink.post("/api/bot/action", body, BotLink.describeResult(what));
        return 1;
    }

    private static int autopilot(boolean on) {
        BotLink.put("/api/config", "{\"autopilot\":{\"enabled\":" + on + "}}",
                doc -> on ? "autopilot ON - the bots eat, sleep at night, fight back, swim up, place torches... on their own"
                        : "autopilot off - the bots only do what the AI or you tell them");
        return 1;
    }

    private static int autopilotStatus() {
        BotLink.get("/api/autopilot/status", doc -> {
            StringBuilder sb = new StringBuilder("autopilot " + (Boolean.TRUE.equals(Json.get(doc, "enabled")) ? "on" : "off")
                    + " · " + Json.text(Json.get(doc, "actions"), "0") + " reflexes so far · idle job "
                    + Json.text(Json.get(doc, "idle_job"), "none"));
            Object next = Json.get(doc, "next");
            if (next != null) {
                sb.append("\nnext: ").append(Json.text(Json.get(next, "rule"), "?"))
                        .append(" (").append(Json.text(Json.get(next, "reason"), "")).append(')');
            }
            return sb.toString();
        });
        return 1;
    }

    /** The local player's name (for come / follow). */
    private static String me() {
        Minecraft client = Minecraft.getInstance();
        return client.player == null ? "" : client.player.getName().getString();
    }

    static String botHelp() {
        return "control the PyMC Bot panel's bots (" + BotLink.panel() + "):\n"
                + "/pymcbot bot status | list | skills\n"
                + "/pymcbot bot do <skill> [key=value ...]   e.g. do chop_tree count=12, do mine_ore iron\n"
                + "/pymcbot bot come | follow | stop | home   /pymcbot bot say <text>\n"
                + "/pymcbot bot auto on|off   (auto-sleep at night, eat, fight back, torches...)\n"
                + "/pymcbot bot ai start|stop   /pymcbot bot panel [url]";
    }

    /** Shown by {@code /pymcbot} on its own. */
    static String help() {
        return PymcPlaytimeMod.NAME + " " + PymcPlaytimeMod.modVersion() + " - records your playtime for PyMC Bot\n"
                + "/pymcbot settings   open the settings page (also: hotkey, Mod Menu)\n"
                + "/pymcbot record start|stop   /pymcbot pause|resume   /pymcbot export\n"
                + "/pymcbot screenshots on|off   pictures for the Minecraftism VL model\n"
                + "/pymcbot bot ...   drive your PyMC Bot bots from the game (/pymcbot bot help)\n"
                + "/pymcbot status   /pymcbot train   /pymcbot about   (/pymc works too)";
    }

    static String version() {
        return PymcPlaytimeMod.NAME + " " + PymcPlaytimeMod.modVersion() + "\n"
                + "Minecraft " + PymcPlaytimeMod.minecraftVersion() + " (compat layer: " + Mc.ERA + ")\n"
                + "one jar per Minecraft version, 1.21.1 to 26.3 - pick the one matching your game";
    }

    /** The command that trains a model on the current dataset (also used by the settings page). */
    static String trainHint(PlaytimeRecorder recorder) {
        String dataset = recorder.datasetPath() == null ? "<gameDir>/pymcbot/dataset.jsonl"
                : recorder.datasetPath().toString();
        String advancedFlag = PlaytimeConfig.get().advanced ? " --advanced" : "";
        String mode = PlaytimeConfig.get().advanced
                ? "recording mode: advanced (entities + items)"
                : "recording mode: basic - run /pymcbot advanced on to record entities and items";
        return "dataset: " + dataset + "\n"
                + mode + "\n"
                + "train it with:  python -m pymc_bot train --dataset \"" + dataset + "\" --steps 5000"
                + advancedFlag + "\n"
                + "then either run the native policy in the panel (Agent -> trained) or\n"
                + "python -m pymc_bot train --dataset \"" + dataset + "\" --ollama-model pymc-bot\n"
                + "to import it into Ollama and select it in the config page."
                + (PlaytimeConfig.get().screenshots
                ? "\nscreenshots are on - build Minecraftism VL data with:\n"
                        + "python -m pymc_bot llm build --playtime \"" + dataset + "\" --screenshots"
                : "");
    }

    private static int feedback(FabricClientCommandSource source, String message) {
        for (String line : message.split("\n")) {
            source.sendFeedback(Component.literal(PymcBotScreen.CHAT_PREFIX + line));
        }
        return 1;
    }
}
