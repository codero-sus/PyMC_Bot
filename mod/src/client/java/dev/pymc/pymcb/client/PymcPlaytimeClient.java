package dev.pymc.pymcb.client;

import dev.pymc.pymcb.PymcPlaytimeMod;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.Minecraft;

/**
 * Client entrypoint: wires the recorder into the client tick loop and registers {@code /pymc}.
 *
 * <p>The recorder samples the local player only - it never talks to the server, so it works on
 * the same offline-mode (cracked) servers PyMC Bot plays on.
 */
public class PymcPlaytimeClient implements ClientModInitializer {
    private static PlaytimeRecorder recorder;

    @Override
    public void onInitializeClient() {
        Minecraft client = Minecraft.getInstance();
        recorder = new PlaytimeRecorder(client);

        ClientTickEvents.END_CLIENT_TICK.register(ignored -> {
            if (recorder != null) {
                recorder.onClientTick();
            }
        });
        ClientLifecycleEvents.CLIENT_STOPPING.register(ignored -> {
            if (recorder != null) {
                recorder.shutdown();
            }
        });

        new PymcCommands(recorder).register();
        PymcPlaytimeMod.LOGGER.info("PyMC Playtime Recorder ready - /pymc record start begins a dataset");
    }

    public static PlaytimeRecorder recorder() {
        return recorder;
    }
}
