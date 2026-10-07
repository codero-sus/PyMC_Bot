package dev.pymc.pymcb.client;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import dev.pymc.pymcb.Json;
import dev.pymc.pymcb.PlaytimeConfig;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.monster.Monster;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.Vec3;

/**
 * Records the local player's playtime as a training dataset for PyMC Bot.
 *
 * <p>Every {@code sampleTicks} client ticks (default 2 = 100 ms = 20 Hz) one JSON object is
 * appended to {@code <gameDir>/pymc-playtime/episodes/<run>.ndjsonl}. A record holds the
 * player's local state (position, velocity, orientation, hotbar, health, the 3x3x3 block
 * neighbourhood and the nearby entities) plus the action that was pressed since the previous
 * sample. {@code /pymc export} flattens every episode into {@code pymc-playtime/dataset.jsonl},
 * one instruction/completion pair per line, which is what
 * {@code python -m pymc_bot train --dataset ...} consumes.
 *
 * <p>Everything is written under the Minecraft game directory, which is also where PyMC Bot
 * looks by default, so the Python side never has to talk to the game.
 */
public final class PlaytimeRecorder {
    public static final String DIR_NAME = "pymc-playtime";
    public static final String EPISODES_DIR = "episodes";
    public static final String DATASET_FILE = "dataset.jsonl";
    public static final String META_FILE = "meta.json";

    /** Entities further away than this are never written to a sample. */
    private static final double NEARBY_RANGE = 16.0;
    private static final DateTimeFormatter RUN_STAMP = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss", Locale.ROOT);

    private final Minecraft client;

    private BufferedWriter writer;
    private Path root;
    private Path episodesDir;
    private Path metaPath;
    private Path datasetPath;

    private String runName = "";
    private boolean active;
    private long startedAtMillis;
    private long tick;
    private long episodeStartTick;
    private int episodeSamples;
    private long totalSamples;
    private int episodeNumber = 1;
    private long lastSampleTick;

    private Vec3 lastPos;
    private float prevYaw;
    private float prevPitch;
    private boolean hasPrevView;
    private int lastSlot;
    private boolean hasPrevSlot;
    private double movedSinceSample;
    private float turnedSinceSample;
    // Sticky input flags: true if the key/button was pressed at any point since the last sample.
    private boolean forward, back, left, right, jump, sneak, attack, use;

    public PlaytimeRecorder(Minecraft client) {
        this.client = client;
    }

    // ------------------------------------------------------------------ lifecycle

    /** Starts a new recording run (no-op when already recording). */
    public synchronized String start() {
        if (active) {
            return "already recording as '" + runName + "' - use /pymc record stop";
        }
        resolvePaths();
        runName = "run-" + LocalDateTime.now().format(RUN_STAMP) + "-" + episodeNumber;
        try {
            Files.createDirectories(episodesDir);
            writer = Files.newBufferedWriter(
                    episodesDir.resolve(runName + ".ndjsonl"),
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
        } catch (IOException e) {
            writer = null;
            return "failed to open the episode file: " + e.getMessage();
        }
        active = true;
        startedAtMillis = System.currentTimeMillis();
        tick = 0;
        episodeStartTick = 0;
        episodeSamples = 0;
        totalSamples = 0;
        lastSampleTick = 0;
        lastPos = null;
        hasPrevView = false;
        hasPrevSlot = false;
        movedSinceSample = 0.0;
        turnedSinceSample = 0.0f;
        resetInputs();
        writeMeta();
        return "recording -> " + episodesDir.resolve(runName + ".ndjsonl")
                + "\nplay normally, then run /pymc record stop and /pymc export";
    }

    /** Stops the active recording and updates meta.json. */
    public synchronized String stop() {
        if (!active) {
            return "not recording - start one with /pymc record start";
        }
        closeWriter();
        active = false;
        writeMeta();
        return String.format(Locale.ROOT,
                "stopped '%s': %d sample(s), about %.1f minute(s) of playtime",
                runName, totalSamples, totalSamples * PlaytimeConfig.get().sampleTicks * 50.0 / 60000.0);
    }

    /** Called when the game shuts down so the file handle is not leaked. */
    public synchronized void shutdown() {
        if (active) {
            closeWriter();
            active = false;
            writeMeta();
        }
    }

    private void closeWriter() {
        if (writer != null) {
            try {
                writer.flush();
                writer.close();
            } catch (IOException ignored) {
                // Nothing sensible to do while the game is going away.
            }
            writer = null;
        }
    }

    /** Closes the current episode file and starts a fresh one inside the same run. */
    public synchronized String endEpisode() {
        if (!active) {
            return "not recording - start one with /pymc record start";
        }
        if (episodeSamples == 0) {
            return "the current episode has no samples yet";
        }
        int closed = episodeSamples;
        closeWriter();
        writeMeta();
        episodeNumber++;
        episodeSamples = 0;
        episodeStartTick = tick;
        startedAtMillis = System.currentTimeMillis();
        try {
            writer = Files.newBufferedWriter(
                    episodesDir.resolve(runName + "-ep" + episodeNumber + ".ndjsonl"),
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
        } catch (IOException e) {
            writer = null;
            active = false;
            return "episode closed (" + closed + " samples) but the next file could not be opened: " + e.getMessage();
        }
        return "episode closed with " + closed + " samples - recording continues in episode " + episodeNumber;
    }

    private void resolvePaths() {
        root = FabricLoader.getInstance().getGameDir().resolve(DIR_NAME);
        episodesDir = root.resolve(EPISODES_DIR);
        metaPath = root.resolve(META_FILE);
        datasetPath = root.resolve(DATASET_FILE);
    }

    // ------------------------------------------------------------------ ticking

    /** Called from the client tick hook (every client tick, recording or not). */
    public synchronized void onClientTick() {
        LocalPlayer player = client.player;
        if (player == null || client.level == null) {
            return;
        }
        tick++;
        sampleInputs();
        trackView(player);
        trackMovement(player);

        if (!active) {
            prime(player);
            return;
        }
        if (tick - lastSampleTick >= Math.max(1, PlaytimeConfig.get().sampleTicks)) {
            writeSample(player);
            lastSampleTick = tick;
            prime(player);
        }
    }

    /** Reads the vanilla key mappings (mouse buttons drive keyAttack/keyUse) into sticky flags. */
    private void sampleInputs() {
        forward |= client.options.keyUp.isDown();
        back |= client.options.keyDown.isDown();
        left |= client.options.keyLeft.isDown();
        right |= client.options.keyRight.isDown();
        jump |= client.options.keyJump.isDown();
        sneak |= client.options.keyShift.isDown();
        attack |= client.options.keyAttack.isDown();
        use |= client.options.keyUse.isDown();
    }

    /** Accumulates the degrees the view turned, no matter what turned it. */
    private void trackView(LocalPlayer player) {
        float yaw = player.getYRot();
        float pitch = player.getXRot();
        if (hasPrevView) {
            float dyaw = Math.abs(yaw - prevYaw);
            if (dyaw > 180.0f) {
                dyaw = 360.0f - dyaw; // shortest way round
            }
            turnedSinceSample += dyaw + Math.abs(pitch - prevPitch) * 0.5f;
        }
        prevYaw = yaw;
        prevPitch = pitch;
        hasPrevView = true;
    }

    /** Accumulates how far the player walked since the previous sample. */
    private void trackMovement(LocalPlayer player) {
        Vec3 pos = player.position();
        if (lastPos != null) {
            movedSinceSample += pos.distanceTo(lastPos);
        }
        lastPos = pos;
    }

    /** Remembers where the player is now, and clears the per-sample accumulators. */
    private void prime(LocalPlayer player) {
        lastPos = player.position();
        lastSlot = selectedSlot(player);
        hasPrevSlot = true;
        movedSinceSample = 0.0;
        turnedSinceSample = 0.0f;
        resetInputs();
    }

    private void resetInputs() {
        forward = back = left = right = jump = sneak = attack = use = false;
    }

    /** Works out the action label for the sample that is about to be written. */
    private String action(LocalPlayer player) {
        PlaytimeConfig cfg = PlaytimeConfig.get();
        if (turnedSinceSample >= cfg.turnDegrees) {
            return "look";
        }
        if (attack) {
            return "attack";
        }
        if (use) {
            return "use";
        }
        if (jump) {
            return "jump";
        }
        if (sneak) {
            return "sneak";
        }
        if (forward) {
            return "forward";
        }
        if (back) {
            return "back";
        }
        if (left) {
            return "left";
        }
        if (right) {
            return "right";
        }
        if (hasPrevSlot && lastSlot != selectedSlot(player)) {
            return "hold";
        }
        if (movedSinceSample >= cfg.moveEpsilon) {
            return "move";
        }
        return "none";
    }

    private static int selectedSlot(LocalPlayer player) {
        return player.getInventory().getSelectedSlot();
    }

    private Sample writeSample(LocalPlayer player) {
        Sample sample = new Sample();
        sample.tick = tick - episodeStartTick;
        sample.episode = runName;
        sample.activity = action(player);

        Vec3 pos = player.position();
        Vec3 vel = player.getDeltaMovement();
        sample.x = pos.x;
        sample.y = pos.y;
        sample.z = pos.z;
        sample.vx = vel.x;
        sample.vy = vel.y;
        sample.vz = vel.z;
        sample.yaw = player.getYRot();
        sample.pitch = player.getXRot();
        sample.onGround = player.onGround();
        sample.sprinting = player.isSprinting();
        sample.sneaking = player.isShiftKeyDown();
        sample.selectedSlot = selectedSlot(player);
        sample.health = player.getHealth();
        sample.food = player.getFoodData().getFoodLevel();
        sample.dimension = dimensionName(player);
        if (PlaytimeConfig.get().includeBlocks) {
            sample.blocks = blockNeighbourhood(player);
        }
        sample.nearby = nearbyEntities(player, pos);
        sample.moved = movedSinceSample;

        writeLine(toJson(sample));
        totalSamples++;
        episodeSamples++;
        if (totalSamples % 20 == 0) {
            writeMeta();
        }
        return sample;
    }

    private static String dimensionName(LocalPlayer player) {
        var id = player.level().dimension().identifier();
        return id.getNamespace().equals("minecraft") ? id.getPath() : id.toString();
    }

    private static List<BlockHit> blockNeighbourhood(LocalPlayer player) {
        BlockPos origin = player.blockPosition();
        List<BlockHit> blocks = new ArrayList<>(27);
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                for (int dx = -1; dx <= 1; dx++) {
                    BlockPos at = origin.offset(dx, dy, dz);
                    BlockState state = player.level().getBlockState(at);
                    blocks.add(new BlockHit(dx, dy, dz, BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString()));
                }
            }
        }
        return blocks;
    }

    private static List<EntityHit> nearbyEntities(LocalPlayer player, Vec3 pos) {
        List<EntityHit> hits = new ArrayList<>();
        for (Entity entity : player.level().getEntities(player, player.getBoundingBox().inflate(NEARBY_RANGE))) {
            if (entity == player) {
                continue;
            }
            Vec3 diff = entity.position().subtract(pos);
            EntityHit hit = new EntityHit();
            hit.type = BuiltInRegistries.ENTITY_TYPE.getKey(entity.getType()).toString();
            hit.dx = diff.x;
            hit.dy = diff.y;
            hit.dz = diff.z;
            hit.dist = diff.length();
            hit.hostile = entity instanceof Monster;
            hit.health = entity instanceof LivingEntity living ? living.getHealth() : 0.0f;
            hits.add(hit);
        }
        hits.sort((a, b) -> Double.compare(a.dist, b.dist));
        int limit = PlaytimeConfig.get().maxNearby;
        return hits.size() > limit ? new ArrayList<>(hits.subList(0, limit)) : hits;
    }

    // ------------------------------------------------------------------ writing

    private void writeLine(String line) {
        if (writer == null) {
            return;
        }
        try {
            writer.write(line);
            writer.newLine();
            if (totalSamples % 20 == 0) {
                writer.flush();
            }
        } catch (IOException ignored) {
            // A full disk must never crash the game; the samples simply stop.
        }
    }

    static String toJson(Sample s) {
        StringBuilder sb = new StringBuilder(1024);
        sb.append("{\"tick\": ").append(s.tick)
          .append(", \"episode\": \"").append(Json.escape(s.episode)).append('"')
          .append(", \"activity\": \"").append(Json.escape(s.activity)).append('"')
          .append(", \"x\": ").append(Json.number(s.x, 3))
          .append(", \"y\": ").append(Json.number(s.y, 3))
          .append(", \"z\": ").append(Json.number(s.z, 3))
          .append(", \"vx\": ").append(Json.number(s.vx, 3))
          .append(", \"vy\": ").append(Json.number(s.vy, 3))
          .append(", \"vz\": ").append(Json.number(s.vz, 3))
          .append(", \"moved\": ").append(Json.number(s.moved, 3))
          .append(", \"yaw\": ").append(Json.number(s.yaw, 2))
          .append(", \"pitch\": ").append(Json.number(s.pitch, 2))
          .append(", \"on_ground\": ").append(s.onGround)
          .append(", \"sprinting\": ").append(s.sprinting)
          .append(", \"sneaking\": ").append(s.sneaking)
          .append(", \"selected_slot\": ").append(s.selectedSlot)
          .append(", \"health\": ").append(Json.number(s.health, 1))
          .append(", \"food\": ").append(s.food)
          .append(", \"dimension\": \"").append(Json.escape(s.dimension)).append('"');
        if (s.blocks != null) {
            sb.append(", \"blocks\": [");
            for (int i = 0; i < s.blocks.size(); i++) {
                BlockHit b = s.blocks.get(i);
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append('[').append(b.dx).append(", ").append(b.dy).append(", ").append(b.dz)
                  .append(", \"").append(Json.escape(b.block)).append("\"]");
            }
            sb.append(']');
        }
        sb.append(", \"nearby\": [");
        for (int i = 0; i < s.nearby.size(); i++) {
            EntityHit e = s.nearby.get(i);
            if (i > 0) {
                sb.append(", ");
            }
            sb.append("{\"type\": \"").append(Json.escape(e.type)).append('"')
              .append(", \"dx\": ").append(Json.number(e.dx, 2))
              .append(", \"dy\": ").append(Json.number(e.dy, 2))
              .append(", \"dz\": ").append(Json.number(e.dz, 2))
              .append(", \"dist\": ").append(Json.number(e.dist, 2))
              .append(", \"hostile\": ").append(e.hostile)
              .append(", \"health\": ").append(Json.number(e.health, 1))
              .append('}');
        }
        sb.append("]}");
        return sb.toString();
    }

    private void writeMeta() {
        if (metaPath == null) {
            return;
        }
        StringBuilder sb = new StringBuilder("{\n");
        sb.append("  \"mod\": \"pymcplaytime\",\n");
        sb.append("  \"mode\": \"live\",\n");
        sb.append("  \"active\": ").append(active).append(",\n");
        sb.append("  \"run\": \"").append(Json.escape(runName)).append("\",\n");
        sb.append("  \"started_at\": ").append(startedAtMillis).append(",\n");
        sb.append("  \"updated_at\": ").append(System.currentTimeMillis()).append(",\n");
        sb.append("  \"sample_ticks\": ").append(PlaytimeConfig.get().sampleTicks).append(",\n");
        sb.append("  \"samples\": ").append(totalSamples).append(",\n");
        sb.append("  \"episode\": ").append(episodeNumber).append(",\n");
        sb.append("  \"episode_samples\": ").append(episodeSamples).append(",\n");
        sb.append("  \"dataset\": \"").append(datasetPath == null ? "" : Json.escape(datasetPath.toString())).append("\"\n");
        sb.append("}\n");
        try {
            Files.writeString(metaPath, sb.toString(), StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
        } catch (IOException ignored) {
            // Best effort - the episodes themselves are what matter.
        }
    }

    // ------------------------------------------------------------------ export

    /**
     * Flattens every episode file in the directory into dataset.jsonl, one training example per
     * line: {@code {"instruction": ..., "input": <observation>, "output": "{\"action\": ...}",
     * "episode": ...}}.
     */
    public synchronized String export() {
        resolvePaths();
        if (!Files.isDirectory(episodesDir)) {
            return "nothing recorded yet - run /pymc record start first";
        }
        String system = "You are a Minecraft player bot. Given the observation, reply with the action to take next.";
        long lines = 0;
        int files = 0;
        if (active) {
            // Make sure the tail of the current episode is on disk before reading it back.
            try {
                if (writer != null) {
                    writer.flush();
                }
            } catch (IOException ignored) {
                // Keep going, we export what is readable.
            }
        }
        try (BufferedWriter out = Files.newBufferedWriter(
                datasetPath, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            List<Path> episodeFiles;
            try (var stream = Files.list(episodesDir)) {
                episodeFiles = stream
                        .filter(p -> p.getFileName().toString().endsWith(".ndjsonl"))
                        .sorted()
                        .toList();
            }
            for (Path file : episodeFiles) {
                String episode = file.getFileName().toString().replace(".ndjsonl", "");
                try (var reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        if (line.isBlank() || !line.startsWith("{")) {
                            continue;
                        }
                        out.write("{\"instruction\": \"");
                        out.write(Json.escape(system));
                        out.write("\", \"input\": ");
                        out.write(line);
                        out.write(", \"output\": \"{\\\"action\\\": \\\"");
                        out.write(Json.escape(actionOf(line)));
                        out.write("\\\"}\", \"episode\": \"");
                        out.write(Json.escape(episode));
                        out.write("\"}");
                        out.newLine();
                        lines++;
                    }
                }
                files++;
            }
        } catch (IOException e) {
            return "export failed: " + e.getMessage();
        }
        if (lines == 0) {
            return "no samples found in " + episodesDir + " - record some playtime first";
        }
        return String.format(Locale.ROOT,
                "exported %d sample(s) from %d episode file(s) -> %s (%.1f MB)",
                lines, files, datasetPath, Files.exists(datasetPath) ? datasetPath.toFile().length() / 1048576.0 : 0.0);
    }

    /** Pulls the "activity" value out of a raw episode line without a full JSON parser. */
    static String actionOf(String rawLine) {
        String needle = "\"activity\": \"";
        int start = rawLine.indexOf(needle);
        if (start < 0) {
            return "none";
        }
        start += needle.length();
        int end = rawLine.indexOf('"', start);
        return end < 0 ? "none" : rawLine.substring(start, end);
    }

    // ------------------------------------------------------------------ status

    public synchronized String status() {
        resolvePaths();
        StringBuilder sb = new StringBuilder();
        if (active) {
            sb.append("RECORDING '").append(runName).append("' - ").append(totalSamples)
              .append(" sample(s), ").append(episodeSamples).append(" in episode ").append(episodeNumber)
              .append(" (~").append(String.format(Locale.ROOT, "%.1f", samplesToMinutes(totalSamples)))
              .append(" min)\n");
        } else {
            sb.append("not recording\n");
        }
        sb.append("config: ").append(PlaytimeConfig.get().describe()).append('\n');
        sb.append("episodes: ").append(episodesDir).append('\n');
        long datasetLines = 0;
        if (datasetPath != null && Files.isRegularFile(datasetPath)) {
            try (var stream = Files.lines(datasetPath)) {
                datasetLines = stream.count();
            } catch (IOException ignored) {
                datasetLines = -1;
            }
        }
        sb.append("dataset.jsonl: ").append(datasetLines < 0 ? "unreadable" : datasetLines + " sample(s)")
          .append(datasetPath == null ? "" : " (" + datasetPath + ")");
        return sb.toString();
    }

    private static double samplesToMinutes(long samples) {
        return samples * PlaytimeConfig.get().sampleTicks * 50.0 / 60000.0;
    }

    public synchronized boolean isActive() {
        return active;
    }

    public synchronized long totalSamples() {
        return totalSamples;
    }

    public synchronized String runName() {
        return runName;
    }

    public synchronized Path datasetPath() {
        resolvePaths();
        return datasetPath;
    }

    public String toggle() {
        return isActive() ? stop() : start();
    }

    // ------------------------------------------------------------------ records

    static final class Sample {
        long tick;
        String episode;
        String activity;
        double x, y, z, vx, vy, vz, moved;
        float yaw, pitch;
        boolean onGround, sprinting, sneaking;
        int selectedSlot;
        float health;
        int food;
        String dimension;
        List<BlockHit> blocks;
        List<EntityHit> nearby;
    }

    record BlockHit(int dx, int dy, int dz, String block) {
    }

    static final class EntityHit {
        String type;
        double dx, dy, dz, dist;
        boolean hostile;
        float health;
    }
}
