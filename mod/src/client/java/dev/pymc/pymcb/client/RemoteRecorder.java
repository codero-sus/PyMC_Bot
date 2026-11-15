package dev.pymc.pymcb.client;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;
import java.util.UUID;

import dev.pymc.pymcb.Json;
import dev.pymc.pymcb.PymcPlaytimeMod;
import dev.pymc.pymcb.client.compat.Mc;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.Minecraft;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.phys.Vec3;

/** Records only visible motion; remote key presses and private inventory are not observable. */
public final class RemoteRecorder {
    private BufferedWriter writer;
    private UUID target;
    private String name;
    private String episode;
    private Path file;
    private Vec3 previous;
    private float yaw;
    private float pitch;
    private long tick;
    private int missing;
    private int samples;

    public synchronized String start(Minecraft mc, String requested) {
        if (writer != null) return "remote observation already active; stop it first";
        if (mc.player == null || mc.level == null) return "join a world first";
        if (!requested.matches("[A-Za-z0-9_]{1,16}")) return "invalid player name";
        Player found = null;
        for (Player player : mc.level.players()) {
            if (player != mc.player && player.getName().getString().equalsIgnoreCase(requested)) {
                found = player;
                break;
            }
        }
        if (found == null) return "player is not visible nearby";
        try {
            Path dir = FabricLoader.getInstance().getGameDir().resolve("pymcbot").resolve("remote-observations");
            Files.createDirectories(dir);
            episode = "remote-" + UUID.randomUUID();
            file = dir.resolve(episode + ".ndjsonl");
            writer = Files.newBufferedWriter(file, StandardCharsets.UTF_8, StandardOpenOption.CREATE_NEW);
        } catch (IOException e) {
            return "could not start observation: " + e.getMessage();
        }
        target = found.getUUID(); name = found.getName().getString(); previous = null;
        tick = 0; missing = 0; samples = 0;
        return "observing " + name + " -> " + file + " (visible movement/look only). Stop: /pymcbot observe stop";
    }

    public synchronized void tick(Minecraft mc) {
        if (writer == null || mc.level == null || mc.player == null) return;
        if (++tick > 72000) { stop(); return; } // at most one hour
        if (tick % 4 != 0) return;
        Player found = null;
        for (Player p : mc.level.players()) if (p.getUUID().equals(target)) { found = p; break; }
        if (found == null) {
            previous = null;
            if (++missing >= 50) stop(); // stop after ten seconds out of view
            return;
        }
        missing = 0;
        Vec3 pos = found.position();
        float nextYaw = found.getYRot(), nextPitch = found.getXRot();
        if (previous != null) {
            double distance = pos.distanceTo(previous);
            double turn = Math.abs(((nextYaw - yaw + 540.0) % 360.0) - 180.0) + Math.abs(nextPitch - pitch);
            // Teleports/packet gaps are not training actions. Stationary or ambiguous samples are skipped.
            if (distance < 2.0 && (distance > 0.02 || turn > 2.0)) {
                Vec3 velocity = found.getDeltaMovement();
                String action = turn > 2.0 ? "look" : "move";
                String line = String.format(Locale.ROOT,
                        "{\"episode\":\"%s\",\"source\":\"remote_observation_inferred\",\"tick\":%d,\"activity\":\"%s\",\"dimension\":\"%s\",\"x\":%s,\"y\":%s,\"z\":%s,\"vx\":%s,\"vy\":%s,\"vz\":%s,\"yaw\":%s,\"pitch\":%s,\"moved\":%s,\"on_ground\":%s}\n",
                        Json.escape(episode), tick, action, Json.escape(Mc.dimensionId(mc.player)), Json.number(pos.x, 4), Json.number(pos.y, 4), Json.number(pos.z, 4),
                        Json.number(velocity.x, 4), Json.number(velocity.y, 4), Json.number(velocity.z, 4),
                        Json.number(nextYaw, 3), Json.number(nextPitch, 3), Json.number(distance, 4), found.onGround());
                try { writer.write(line); if (++samples % 20 == 0) writer.flush(); }
                catch (IOException e) { PymcPlaytimeMod.LOGGER.warn("Remote observation stopped: {}", e.toString()); stop(); }
            }
        }
        previous = pos; yaw = nextYaw; pitch = nextPitch;
    }

    public synchronized String stop() {
        if (writer == null) return "remote observation is not running";
        try { writer.close(); } catch (IOException e) { PymcPlaytimeMod.LOGGER.warn("Remote observation close failed: {}", e.toString()); }
        writer = null; target = null; previous = null;
        return "remote observation stopped: " + samples + " inferred samples in " + file;
    }

    public synchronized String status() {
        return writer == null ? "remote observation is off" : "observing " + name + " (" + samples + " inferred samples)";
    }

    public synchronized boolean active() { return writer != null; }
}
