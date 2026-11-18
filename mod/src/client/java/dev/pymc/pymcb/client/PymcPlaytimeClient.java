package dev.pymc.pymcb.client;

import dev.pymc.pymcb.PymcPlaytimeMod;
import java.nio.file.Files;
import java.nio.file.Path;
import net.fabricmc.loader.api.FabricLoader;
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
    private static final RemoteRecorder remoteRecorder = new RemoteRecorder();
    private static final PlayerTakeover takeover = new PlayerTakeover();
    private static KeyMapping releaseKey;
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
        releaseKey = Mc.registerKey("key.pymcbot.release_control");

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
            takeover.tick(mc);
            recorder.onClientTick();
            remoteRecorder.tick(mc);
        });
        ClientPlayConnectionEvents.JOIN.register((handler, sender, mc) -> {
            if (recorder != null) {
                recorder.onJoin();
            }
        });
        ClientPlayConnectionEvents.DISCONNECT.register((handler, mc) -> {
            takeover.disconnect(mc);
            remoteRecorder.stop();
            if (recorder != null) {
                recorder.onDisconnect();
            }
        });
        ClientLifecycleEvents.CLIENT_STOPPING.register(ignored -> {
            takeover.disconnect(Minecraft.getInstance());
            launchUpdateHelper();
            remoteRecorder.stop();
            if (recorder != null) {
                recorder.shutdown();
            }
        });

        new PymcCommands(recorder).register();
        PymcPlaytimeMod.LOGGER.info("{} {} ready on Minecraft {} (compat {}) - /pymcbot settings opens the settings page",
                PymcPlaytimeMod.NAME, PymcPlaytimeMod.modVersion(), PymcPlaytimeMod.minecraftVersion(), Mc.ERA);
    }

    private static void launchUpdateHelper() {
        try {
            var loader = FabricLoader.getInstance();
            var container = loader.getModContainer(PymcPlaytimeMod.MOD_ID).orElseThrow();
            var paths = container.getOrigin().getPaths();
            if (paths.size() != 1) return;
            Path jar = paths.get(0).toAbsolutePath().normalize();
            Path mods = loader.getGameDir().resolve("mods").toAbsolutePath().normalize();
            if (!mods.equals(jar.getParent()) || !Files.isRegularFile(jar, java.nio.file.LinkOption.NOFOLLOW_LINKS)) return;
            Path tmp = Files.createTempDirectory("pymcbot-update-");
            for (String name : new String[] {"ModUpdater.class", "Json.class", "Json$Reader.class"}) {
                String resource = "dev/pymc/pymcb/" + name;
                Path dest = tmp.resolve(resource);
                Files.createDirectories(dest.getParent());
                try (var in = PymcPlaytimeClient.class.getClassLoader().getResourceAsStream(resource)) {
                    if (in == null) return;
                    Files.copy(in, dest);
                }
            }
            String executable = System.getProperty("os.name").toLowerCase(java.util.Locale.ROOT).contains("win") ? "java.exe" : "java";
            Path java = Path.of(System.getProperty("java.home"), "bin", executable);
            new ProcessBuilder(java.toString(), "-cp", tmp.toString(), "dev.pymc.pymcb.ModUpdater",
                    Long.toString(ProcessHandle.current().pid()), jar.toString(),
                    PymcPlaytimeMod.minecraftVersion(), PymcPlaytimeMod.modVersion())
                    .redirectOutput(ProcessBuilder.Redirect.DISCARD).redirectError(ProcessBuilder.Redirect.DISCARD).start();
        } catch (Exception e) {
            PymcPlaytimeMod.LOGGER.debug("Mod update check deferred: {}", e.toString());
        }
    }

    private static void handleKeys(Minecraft mc) {
        while (releaseKey.consumeClick()) {
            if (takeover.active()) takeover.stop(mc, "Takeover released by emergency hotkey.");
        }
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

    public static PlayerTakeover takeover() { return takeover; }

    public static RemoteRecorder remoteRecorder() {
        return remoteRecorder;
    }

    public static PlaytimeRecorder recorder() {
        return recorder;
    }
}
