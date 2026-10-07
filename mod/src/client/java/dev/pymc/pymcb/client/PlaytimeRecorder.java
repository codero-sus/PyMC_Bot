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
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.entity.monster.Monster;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.ItemStack;
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
    private long advancedSamples;
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
        advancedSamples = 0;
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
        if (PlaytimeConfig.get().advanced) {
            advancedSamples++;
        }
        sample.nearby = nearbyEntities(player, pos);
        if (PlaytimeConfig.get().advanced) {
            // Advanced recording: every entity in range (id, where, health, held item,
            // whether it is a player), the inventory with what is in hand, the armour and
            // the dropped stacks - the raw material an advanced model learns from.
            sample.entities = visibleEntities(player, pos);
            sample.groundItems = groundItems(player, pos);
            sample.items = inventory(player);
            sample.heldItem = itemName(player.getMainHandItem());
            sample.armor = armor(player);
        }
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

    /** Every entity the player can see, described well enough for the item/entity vocabulary. */
    private static List<VisualEntity> visibleEntities(LocalPlayer player, Vec3 pos) {
        PlaytimeConfig cfg = PlaytimeConfig.get();
        List<VisualEntity> out = new ArrayList<>();
        for (Entity entity : player.level().getEntities(player, player.getBoundingBox().inflate(cfg.entityRange))) {
            if (entity == player) {
                continue;
            }
            Vec3 diff = entity.position().subtract(pos);
            VisualEntity hit = new VisualEntity();
            hit.type = BuiltInRegistries.ENTITY_TYPE.getKey(entity.getType()).toString();
            hit.dx = diff.x;
            hit.dy = diff.y;
            hit.dz = diff.z;
            hit.dist = diff.length();
            hit.hostile = entity instanceof Monster;
            hit.player = entity instanceof Player;
            hit.health = entity instanceof LivingEntity living ? living.getHealth() : 0.0f;
            hit.onGround = entity.onGround();
            hit.yaw = entity.getYRot();
            hit.heldItem = entity instanceof LivingEntity living ? itemName(living.getMainHandItem()) : "";
            hit.count = entity instanceof ItemEntity drop ? drop.getItem().getCount() : 0;
            out.add(hit);
        }
        out.sort((a, b) -> Double.compare(a.dist, b.dist));
        return out.size() > cfg.maxEntities ? new ArrayList<>(out.subList(0, cfg.maxEntities)) : out;
    }

    /** Dropped item stacks lying around, so the model learns what is worth walking over. */
    private static List<GroundItem> groundItems(LocalPlayer player, Vec3 pos) {
        PlaytimeConfig cfg = PlaytimeConfig.get();
        List<GroundItem> out = new ArrayList<>();
        for (Entity entity : player.level().getEntities(player, player.getBoundingBox().inflate(12.0))) {
            if (!(entity instanceof ItemEntity drop)) {
                continue;
            }
            Vec3 diff = entity.position().subtract(pos);
            GroundItem hit = new GroundItem();
            hit.item = itemName(drop.getItem());
            hit.count = drop.getItem().getCount();
            hit.dist = diff.length();
            out.add(hit);
        }
        out.sort((a, b) -> Double.compare(a.dist, b.dist));
        return out.size() > cfg.maxGroundItems ? new ArrayList<>(out.subList(0, cfg.maxGroundItems)) : out;
    }

    /** The whole inventory as [slot, item, count] - hotbar first, then the main rows. */
    private static List<InvItem> inventory(LocalPlayer player) {
        var inventory = player.getInventory();
        List<InvItem> out = new ArrayList<>();
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = inventory.getItem(slot);
            if (stack.isEmpty()) {
                continue;
            }
            out.add(new InvItem(slot, itemName(stack), stack.getCount()));
        }
        return out;
    }

    private static List<String> armor(LocalPlayer player) {
        List<String> out = new ArrayList<>(4);
        for (ItemStack stack : player.getArmorSlots()) {
            if (!stack.isEmpty()) {
                out.add(itemName(stack));
            }
        }
        return out;
    }

    private static String itemName(ItemStack stack) {
        if (stack == null || stack.isEmpty()) {
            return "";
        }
        return BuiltInRegistries.ITEM.getKey(stack.getItem()).toString();
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
        if (s.entities != null) {
            sb.append(", \"entities\": [");
            for (int i = 0; i < s.entities.size(); i++) {
                VisualEntity e = s.entities.get(i);
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append("{\"type\": \"").append(Json.escape(e.type)).append('"')
                  .append(", \"dx\": ").append(Json.number(e.dx, 2))
                  .append(", \"dy\": ").append(Json.number(e.dy, 2))
                  .append(", \"dz\": ").append(Json.number(e.dz, 2))
                  .append(", \"dist\": ").append(Json.number(e.dist, 2))
                  .append(", \"hostile\": ").append(e.hostile)
                  .append(", \"player\": ").append(e.player)
                  .append(", \"health\": ").append(Json.number(e.health, 1))
                  .append(", \"on_ground\": ").append(e.onGround)
                  .append(", \"yaw\": ").append(Json.number(e.yaw, 1))
                  .append(", \"count\": ").append(e.count)
                  .append(", \"held_item\": \"").append(Json.escape(e.heldItem)).append("\"}");
            }
            sb.append(']');
        }
        if (s.items != null) {
            sb.append(", \"items\": [");
            for (int i = 0; i < s.items.size(); i++) {
                InvItem item = s.items.get(i);
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append('[').append(item.slot).append(", \"").append(Json.escape(item.item))
                  .append("\", ").append(item.count).append(']');
            }
            sb.append(']');
            sb.append(", \"held_item\": \"").append(Json.escape(s.heldItem == null ? "" : s.heldItem)).append('"');
            sb.append(", \"armor\": [");
            for (int i = 0; i < s.armor.size(); i++) {
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append('"').append(Json.escape(s.armor.get(i))).append('"');
            }
            sb.append(']');
        }
        if (s.groundItems != null) {
            sb.append(", \"ground_items\": [");
            for (int i = 0; i < s.groundItems.size(); i++) {
                GroundItem g = s.groundItems.get(i);
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append("{\"item\": \"").append(Json.escape(g.item)).append('"')
                  .append(", \"count\": ").append(g.count)
                  .append(", \"dist\": ").append(Json.number(g.dist, 2))
                  .append('}');
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
        sb.append("  \"advanced\": ").append(PlaytimeConfig.get().advanced).append(",\n");
        sb.append("  \"entities_offered\": ").append(advancedSamples).append(",\n");
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
        boolean advanced = PlaytimeConfig.get().advanced;
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
                lines, files, datasetPath, Files.exists(datasetPath) ? datasetPath.toFile().length() / 1048576.0 : 0.0)
                + (advanced
                ? "\nadvanced data included (entities, items, held item, armour, drops)"
                : "\nbasic recording - run /pymc advanced on before recording to also export entities and items");
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
        sb.append("advanced: ").append(PlaytimeConfig.get().advanced ? "ON (entities + items)" : "off")
          .append(" - ").append(advancedSamples).append(" advanced sample(s) this run\n");
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
        // Advanced mode (only written when PlaytimeConfig.advanced is on).
        List<VisualEntity> entities;
        List<InvItem> items;
        List<GroundItem> groundItems;
        List<String> armor;
        String heldItem;
    }

    record BlockHit(int dx, int dy, int dz, String block) {
    }

    static final class EntityHit {
        String type;
        double dx, dy, dz, dist;
        boolean hostile;
        float health;
    }

    /** One entity, described for the advanced entity vocabulary. */
    static final class VisualEntity {
        String type;
        double dx, dy, dz, dist;
        float yaw;
        float health;
        int count;
        boolean hostile;
        boolean player;
        boolean onGround;
        String heldItem = "";
    }

    /** One inventory slot: [slot, item, count]. */
    record InvItem(int slot, String item, int count) {
    }

    /** One dropped stack on the ground. */
    static final class GroundItem {
        String item;
        int count;
        double dist;
    }
}
