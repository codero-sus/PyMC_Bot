package dev.pymc.pymcb.client;

import java.util.Locale;

import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import dev.pymc.pymcb.PlaytimeConfig;
import dev.pymc.pymcb.PymcPlaytimeMod;
import dev.pymc.pymcb.client.compat.Mc;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
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

            for (String root : ROOTS) {
                dispatcher.register(literal(root)
                        .executes(ctx -> feedback(ctx.getSource(), help()))
                        .then(literal("help").executes(ctx -> feedback(ctx.getSource(), help())))
                        .then(record)
                        .then(guarded("pause").executes(ctx -> feedback(ctx.getSource(), recorder.pause())))
                        .then(guarded("resume").executes(ctx -> feedback(ctx.getSource(), recorder.resume())))
                        .then(sample)
                        .then(advanced)
                        .then(entities)
                        .then(drops)
                        .then(autorecord)
                        .then(menus)
                        .then(idle)
                        .then(notify)
                        .then(hud)
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

    /** Shown by {@code /pymcbot} on its own. */
    static String help() {
        return PymcPlaytimeMod.NAME + " " + PymcPlaytimeMod.modVersion() + " - records your playtime for PyMC Bot\n"
                + "/pymcbot settings   open the settings page (also: hotkey, Mod Menu)\n"
                + "/pymcbot record start|stop   /pymcbot pause|resume   /pymcbot export\n"
                + "/pymcbot status   /pymcbot train   /pymcbot about   (/pymcbot works too)";
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
                + "to import it into Ollama and select it in the config page.";
    }

    private static int feedback(FabricClientCommandSource source, String message) {
        for (String line : message.split("\n")) {
            source.sendFeedback(Component.literal(PymcBotScreen.CHAT_PREFIX + line));
        }
        return 1;
    }
}
