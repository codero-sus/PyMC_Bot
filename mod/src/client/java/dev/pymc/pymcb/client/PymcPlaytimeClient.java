package dev.pymc.pymcb.client;

import dev.pymc.pymcb.PymcPlaytimeMod;
import dev.pymc.pymcb.client.compat.Mc;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayConnectionEvents;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;

/**
 * Client entrypoint: wires the recorder into the client tick loop, registers {@code /pymcbot}, the
 * three hotkeys, the HUD recording badge and the join/disconnect hooks.
 *
 * <p>The recorder samples the local player only - it never talks to the server, so it works on
 * the same offline-mode (cracked) servers PyMC Bot plays on.
 */
public class PymcPlaytimeClient implements ClientModInitializer {
    private static PlaytimeRecorder recorder;
    private static KeyMapping toggleKey;
    private static KeyMapping pauseKey;
    private static KeyMapping settingsKey;
    /** Set by /pymcbot settings; the screen opens on the next tick, after the chat screen has closed. */
    private static volatile boolean openSettingsNextTick;

    @Override
    public void onInitializeClient() {
        Minecraft client = Minecraft.getInstance();
        recorder = new PlaytimeRecorder(client);

        // Unbound by default (Options -> Controls -> Key Binds -> PyMC Bot), so they can
        // never steal a key from the game or another mod.
        toggleKey = Mc.registerKey("key.pymcbot.toggle");
        pauseKey = Mc.registerKey("key.pymcbot.pause");
        settingsKey = Mc.registerKey("key.pymcbot.settings");

        // REC / PAUSED badge in a corner of the HUD while recording (/pymcbot hud off hides it).
        Mc.registerHud(RecorderHud::paint);

        ClientTickEvents.END_CLIENT_TICK.register(mc -> {
            if (recorder == null) {
                return;
            }
            handleKeys(mc);
            if (openSettingsNextTick) {
                openSettingsNextTick = false;
                Mc.openScreen(mc, new PymcBotScreen(null));
            }
            recorder.onClientTick();
        });
        ClientPlayConnectionEvents.JOIN.register((handler, sender, mc) -> {
            if (recorder != null) {
                recorder.onJoin();
            }
        });
        ClientPlayConnectionEvents.DISCONNECT.register((handler, mc) -> {
            if (recorder != null) {
                recorder.onDisconnect();
            }
        });
        ClientLifecycleEvents.CLIENT_STOPPING.register(ignored -> {
            if (recorder != null) {
                recorder.shutdown();
            }
        });

        new PymcCommands(recorder).register();
        PymcPlaytimeMod.LOGGER.info("{} {} ready on Minecraft {} (compat {}) - /pymcbot settings opens the settings page",
                PymcPlaytimeMod.NAME, PymcPlaytimeMod.modVersion(), PymcPlaytimeMod.minecraftVersion(), Mc.ERA);
    }

    private static void handleKeys(Minecraft mc) {
        while (toggleKey.consumeClick()) {
            if (mc.player != null) {
                PlaytimeRecorder.tell(mc.player, recorder.toggle());
            }
        }
        while (pauseKey.consumeClick()) {
            if (mc.player != null) {
                PlaytimeRecorder.tell(mc.player, recorder.togglePause());
            }
        }
        while (settingsKey.consumeClick()) {
            openSettingsNextTick = true;
        }
    }

    /** Opens the PyMC Bot settings page on the next client tick (safe to call from a chat command). */
    public static void openSettings() {
        openSettingsNextTick = true;
    }

    public static PlaytimeRecorder recorder() {
        return recorder;
    }
}
