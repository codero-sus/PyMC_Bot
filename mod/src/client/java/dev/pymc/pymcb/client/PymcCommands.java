package dev.pymc.pymcb.client;

import static net.fabricmc.fabric.api.client.command.v2.ClientCommandManager.argument;
import static net.fabricmc.fabric.api.client.command.v2.ClientCommandManager.literal;

import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import dev.pymc.pymcb.PlaytimeConfig;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.minecraft.network.chat.Component;

/**
 * {@code /pymc ...} - client side commands for the playtime recorder.
 *
 * <pre>
 * /pymc record start|stop|toggle|status
 * /pymc record episode          close the current episode and start a new one
 * /pymc export                  write dataset.jsonl for the Python trainer
 * /pymc sample &lt;ticks&gt;          change the sampling rate (2 ticks = 100 ms)
 * /pymc status                  where the dataset lives, how big it is
 * /pymc train                   print the exact command to train on this dataset
 * </pre>
 */
public final class PymcCommands {
    private final PlaytimeRecorder recorder;

    public PymcCommands(PlaytimeRecorder recorder) {
        this.recorder = recorder;
    }

    public void register() {
        ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) -> {
            LiteralArgumentBuilder<FabricClientCommandSource> record = literal("record")
                    .then(literal("start").executes(ctx -> feedback(ctx.getSource(), recorder.start())))
                    .then(literal("stop").executes(ctx -> feedback(ctx.getSource(), recorder.stop())))
                    .then(literal("toggle").executes(ctx -> feedback(ctx.getSource(), recorder.toggle())))
                    .then(literal("episode").executes(ctx -> feedback(ctx.getSource(), recorder.endEpisode())))
                    .then(literal("status").executes(ctx -> feedback(ctx.getSource(), recorder.status())));

            LiteralArgumentBuilder<FabricClientCommandSource> sample = literal("sample")
                    .then(argument("ticks", IntegerArgumentType.integer(1, 200))
                            .executes(ctx -> {
                                int ticks = IntegerArgumentType.getInteger(ctx, "ticks");
                                PlaytimeConfig.get().setSampleTicks(ticks);
                                return feedback(ctx.getSource(),
                                        "sampling every " + ticks + " tick(s) (~" + ticks * 50 + " ms)");
                            }));

            dispatcher.register(literal("pymc")
                    .then(record)
                    .then(sample)
                    .then(literal("param").executes(ctx -> feedback(ctx.getSource(), PlaytimeConfig.get().describe())))
                    .then(literal("status").executes(ctx -> feedback(ctx.getSource(), recorder.status())))
                    .then(literal("export").executes(ctx -> feedback(ctx.getSource(), recorder.export())))
                    .then(literal("train").executes(ctx -> feedback(ctx.getSource(), trainHint()))));
        });
    }

    private String trainHint() {
        String dataset = recorder.datasetPath() == null ? "<gameDir>/pymc-playtime/dataset.jsonl"
                : recorder.datasetPath().toString();
        return "dataset: " + dataset + "\n"
                + "train it with:  python -m pymc_bot train --dataset \"" + dataset + "\" --steps 5000\n"
                + "then either run the native policy in the panel (Agent -> trained) or\n"
                + "python -m pymc_bot train --dataset \"" + dataset + "\" --ollama-model pymc-playtime\n"
                + "to import it into Ollama and select it in the config page.";
    }

    private static int feedback(FabricClientCommandSource source, String message) {
        for (String line : message.split("\n")) {
            source.sendFeedback(Component.literal("[pymc] " + line));
        }
        return 1;
    }
}
