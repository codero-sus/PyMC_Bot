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
import dev.pymc.pymcb.PymcPlaytimeMod;
import dev.pymc.pymcb.client.compat.Mc;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.network.chat.Component;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.MobCategory;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.Vec3;

/**
 * Records the local player's playtime as a training dataset for PyMC Bot.
 *
 * <p>Every {@code sampleTicks} client ticks (default 2 = 100 ms = 20 Hz) one JSON object is
 * appended to {@code <gameDir>/pymcbot/episodes/<run>.ndjsonl}. A record holds the
 * player's local state (position, velocity, orientation, hotbar, health, the 3x3x3 block
 * neighbourhood and the nearby entities) plus the action that was pressed since the previous
 * sample. {@code /pymcbot export} flattens every episode into {@code pymcbot/dataset.jsonl},
 * one instruction/completion pair per line, which is what
 * {@code python -m pymc_bot train --dataset ...} consumes.
 *
 * <p>Everything is written under the Minecraft game directory, which is also where PyMC Bot
 * looks by default, so the Python side never has to talk to the game.
 *
 * <p>Dataset quality: samples are skipped while a screen is open ({@code pauseInMenus}), while
 * the recorder is paused, and after {@code idleSkipSeconds} without any action, so AFK time and
 * inventory juggling do not drown out real play. Every Minecraft-version-specific call goes
 * through {@link Mc}.
 */
public final class PlaytimeRecorder {
    public static final String DIR_NAME = "pymcbot";
    /** Folder name used before 0.7.0; moved to {@link #DIR_NAME} the first time the recorder needs it. */
    public static final String LEGACY_DIR_NAME = "pymc-playtime";
    private static boolean legacyNoticeLogged;
    public static final String EPISODES_DIR = "episodes";
    public static final String DATASET_FILE = "dataset.jsonl";
    public static final String META_FILE = "meta.json";

    /** Entities further away than this are never written to a sample. */
    private static final double NEARBY_RANGE = 16.0;
    /** Armour slots, in the order the advanced recording writes them. */
    private static final EquipmentSlot[] ARMOR_SLOTS = {
        EquipmentSlot.HEAD, EquipmentSlot.CHEST, EquipmentSlot.LEGS, EquipmentSlot.FEET,
    };
    private static final DateTimeFormatter RUN_STAMP = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss", Locale.ROOT);

    private final Minecraft client;
    /** Minecraftism VL screenshots (only when {@code screenshots=true}). */
    private final ScreenshotTaker shots = new ScreenshotTaker();

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

    /** Manually paused with /pymcbot pause or the pause hotkey (the episode file stays open). */
    private boolean paused;
    /** Set when a world was joined with autoRecord on; the start happens on the first tick with a player. */
    private boolean pendingAutoStart;
    /** Consecutive samples labelled "none". */
    private int idleRun;
    /** Ticks skipped because a screen was open / the recorder was paused, this run. */
    private long menuTicks;
    private long pausedTicks;
    /** Samples not written because the player was idle, this run. */
    private long idleSkipped;
    /** Wall-clock start of the run (unlike startedAtMillis it survives /pymcbot record episode). */
    private long runStartedMillis;
    /** True while samples are being skipped because the player is AFK (shown on the HUD badge). */
    private boolean idleSkipping;

    public PlaytimeRecorder(Minecraft client) {
        this.client = client;
    }

    // ------------------------------------------------------------------ lifecycle

    /** Starts a new recording run (no-op when already recording). */
    public synchronized String start() {
        if (active) {
            return "already recording as '" + runName + "' - use /pymcbot record stop";
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
        runStartedMillis = startedAtMillis;
        idleSkipping = false;
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
        paused = false;
        idleRun = 0;
        menuTicks = 0;
        pausedTicks = 0;
        idleSkipped = 0;
        shots.reset();
        resetInputs();
        writeMeta();
        return "recording -> " + episodesDir.resolve(runName + ".ndjsonl")
                + "\nplay normally, then run /pymcbot record stop and /pymcbot export";
    }

    /** Stops the active recording and updates meta.json. */
    public synchronized String stop() {
        if (!active) {
            return "not recording - start one with /pymcbot record start";
        }
        closeWriter();
        active = false;
        paused = false;
        writeMeta();
        return String.format(Locale.ROOT,
                "stopped '%s': %d sample(s), about %.1f minute(s) of playtime",
                runName, totalSamples, totalSamples * PlaytimeConfig.get().sampleTicks * 50.0 / 60000.0)
                + skippedSummary();
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
            return "not recording - start one with /pymcbot record start";
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
        Path gameDir = FabricLoader.getInstance().getGameDir();
        root = migrateLegacyDir(gameDir);
        episodesDir = root.resolve(EPISODES_DIR);
        metaPath = root.resolve(META_FILE);
        datasetPath = root.resolve(DATASET_FILE);
    }

    /**
     * Returns {@code <gameDir>/pymcbot}. Recordings made by older versions live in
     * {@code <gameDir>/pymc-playtime}: that folder is renamed when the new one does not exist yet,
     * otherwise both are kept (nothing is ever merged or deleted) and the old one is mentioned in the log.
     */
    static Path migrateLegacyDir(Path gameDir) {
        Path target = gameDir.resolve(DIR_NAME);
        Path legacy = gameDir.resolve(LEGACY_DIR_NAME);
        if (!Files.isDirectory(legacy)) {
            return target;
        }
        if (Files.exists(target)) {
            if (legacyNoticeLogged) {
                return target;
            }
            legacyNoticeLogged = true;
            PymcPlaytimeMod.LOGGER.info("[pymcbot] old recordings are still in {} (new ones go to {})", legacy, target);
            return target;
        }
        try {
            Files.move(legacy, target);
            PymcPlaytimeMod.LOGGER.info("[pymcbot] moved recordings from {} to {}", legacy, target);
        } catch (IOException e) {
            PymcPlaytimeMod.LOGGER.warn("[pymcbot] could not move {} to {}: {}", legacy, target, e.getMessage());
        }
        return target;
    }

    // ------------------------------------------------------------------ ticking

    /** Called from the client tick hook (every client tick, recording or not). */
    public synchronized void onClientTick() {
        LocalPlayer player = client.player;
        if (player == null || client.level == null) {
            return;
        }
        if (pendingAutoStart) {
            pendingAutoStart = false;
            if (!active) {
                String result = start();
                tell(player, active ? "auto-record started (/pymcbot autorecord off to disable)" : result);
            }
        }
        tick++;
        sampleInputs();
        trackView(player);
        trackMovement(player);

        if (!active) {
            prime(player);
            return;
        }
        PlaytimeConfig cfg = PlaytimeConfig.get();
        boolean inMenu = cfg.pauseInMenus && Mc.screenOpen(client);
        if (paused || inMenu) {
            // Nothing the player does in a menu (or while paused) is a game action the bot can
            // copy; drop it so the next sample starts clean.
            if (paused) {
                pausedTicks++;
            } else {
                menuTicks++;
            }
            lastSampleTick = tick;
            prime(player);
            return;
        }
        if (tick - lastSampleTick >= Math.max(1, cfg.sampleTicks)) {
            String activity = action(player);
            idleRun = "none".equals(activity) ? idleRun + 1 : 0;
            idleSkipping = idleLimitReached(cfg);
            if (idleSkipping) {
                idleSkipped++;
            } else {
                writeSample(player, activity);
            }
            lastSampleTick = tick;
            prime(player);
        }
    }

    /** True once the player has been idle for longer than {@code idleSkipSeconds}. */
    private boolean idleLimitReached(PlaytimeConfig cfg) {
        if (cfg.idleSkipSeconds <= 0) {
            return false;
        }
        long limit = Math.max(1L, cfg.idleSkipSeconds * 20L / Math.max(1, cfg.sampleTicks));
        return idleRun > limit;
    }

    // ------------------------------------------------------------------ pause / connection

    /** Pauses sampling without closing the episode (no-op when not recording). */
    public synchronized String pause() {
        if (!active) {
            return "not recording - start one with /pymcbot record start";
        }
        if (paused) {
            return "already paused - /pymcbot resume continues";
        }
        paused = true;
        flushQuietly();
        writeMeta();
        return "paused '" + runName + "' at " + totalSamples + " sample(s) - /pymcbot resume continues";
    }

    public synchronized String resume() {
        if (!active) {
            return "not recording - start one with /pymcbot record start";
        }
        if (!paused) {
            return "not paused";
        }
        paused = false;
        writeMeta();
        return "resumed '" + runName + "'";
    }

    public synchronized String togglePause() {
        return paused ? resume() : pause();
    }

    /** A world or server was joined: arm auto-record if it is enabled. */
    public synchronized void onJoin() {
        pendingAutoStart = PlaytimeConfig.get().autoRecord;
    }

    /**
     * The connection closed: stop and flush so an episode never spans two worlds and nothing is
     * lost if the game is closed from the title screen.
     */
    public synchronized void onDisconnect() {
        pendingAutoStart = false;
        if (active) {
            String result = stop();
            PymcPlaytimeMod.LOGGER.info("[pymc] disconnected - {}", result.replace('\n', ' '));
        }
        lastPos = null;
        hasPrevView = false;
        hasPrevSlot = false;
    }

    private void flushQuietly() {
        if (writer != null) {
            try {
                writer.flush();
            } catch (IOException ignored) {
                // Best effort.
            }
        }
    }

    /** Sends a chat line to the local player when notifications are on. */
    static void tell(LocalPlayer player, String message) {
        if (player != null && PlaytimeConfig.get().notifications) {
            Mc.message(player, Component.literal(PymcBotScreen.CHAT_PREFIX + message.split("\n", 2)[0]));
        }
    }

    private String skippedSummary() {
        if (menuTicks == 0 && pausedTicks == 0 && idleSkipped == 0) {
            return "";
        }
        return String.format(Locale.ROOT,
                "\nskipped: %.0fs in menus, %.0fs paused, %d idle sample(s)",
                menuTicks / 20.0, pausedTicks / 20.0, idleSkipped);
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
        return Mc.selectedSlot(player);
    }

    private Sample writeSample(LocalPlayer player, String activity) {
        Sample sample = new Sample();
        sample.tick = tick - episodeStartTick;
        sample.episode = runName;
        sample.activity = activity;

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
        sample.shot = shots.maybeCapture(client, root, runName, tick, sample.tick);

        writeLine(toJson(sample));
        totalSamples++;
        episodeSamples++;
        if (totalSamples % 20 == 0) {
            writeMeta();
        }
        return sample;
    }

    private static String dimensionName(LocalPlayer player) {
        String id = Mc.dimensionId(player);
        return id.startsWith("minecraft:") ? id.substring("minecraft:".length()) : id;
    }

    /**
     * Hostile = the entity type spawns in the MONSTER category. Unlike checking for the Monster
     * base class this also covers slimes, magma cubes, ghasts and phantoms, and it does not depend
     * on which package a Minecraft version keeps its monster classes in.
     */
    private static boolean hostile(Entity entity) {
        return entity.getType().getCategory() == MobCategory.MONSTER;
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
            hit.hostile = hostile(entity);
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
            hit.hostile = hostile(entity);
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

    /** The four armour slots, read one by one (stable vanilla API). */
    private static List<String> armor(LocalPlayer player) {
        List<String> out = new ArrayList<>(4);
        for (EquipmentSlot slot : ARMOR_SLOTS) {
            ItemStack stack = player.getItemBySlot(slot);
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
        if (s.shot != null) {
            sb.append(", \"shot\": \"").append(Json.escape(s.shot)).append('"');
        }
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
        sb.append("  \"mod\": \"pymcbot\",\n");
        sb.append("  \"mode\": \"live\",\n");
        sb.append("  \"mod_version\": \"").append(Json.escape(PymcPlaytimeMod.modVersion())).append("\",\n");
        sb.append("  \"minecraft_version\": \"").append(Json.escape(PymcPlaytimeMod.minecraftVersion())).append("\",\n");
        sb.append("  \"compat\": \"").append(Json.escape(Mc.ERA)).append("\",\n");
        sb.append("  \"paused\": ").append(paused).append(",\n");
        sb.append("  \"skipped_menu_ticks\": ").append(menuTicks).append(",\n");
        sb.append("  \"skipped_paused_ticks\": ").append(pausedTicks).append(",\n");
        sb.append("  \"skipped_idle_samples\": ").append(idleSkipped).append(",\n");
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
     *
     * <p>The two export filters from the settings page's Power tab are applied here (the episode
     * files themselves are never changed, so a different filter can be tried with another export):
     * episodes shorter than {@code minEpisodeSeconds} are left out, and long standing-still
     * stretches keep only {@code idleKeepPercent} of their {@code none} samples ({@link IdleThinner}).
     */
    public synchronized String export() {
        resolvePaths();
        if (!Files.isDirectory(episodesDir)) {
            return "nothing recorded yet - run /pymcbot record start first";
        }
        String system = "You are a Minecraft player bot. Given the observation, reply with the action to take next.";
        PlaytimeConfig cfg = PlaytimeConfig.get();
        boolean advanced = cfg.advanced;
        long minTicks = cfg.minEpisodeSeconds * 20L;
        long lines = 0;
        int files = 0;
        int shortEpisodes = 0;
        long idleDropped = 0;
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
                if (minTicks > 0 && lastTick(file) < minTicks) {
                    shortEpisodes++;
                    continue;
                }
                IdleThinner thinner = new IdleThinner(cfg.idleKeepPercent);
                try (var reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        if (line.isBlank() || !line.startsWith("{")) {
                            continue;
                        }
                        String action = actionOf(line);
                        if (!thinner.keep(action)) {
                            idleDropped++;
                            continue;
                        }
                        out.write("{\"instruction\": \"");
                        out.write(Json.escape(system));
                        out.write("\", \"input\": ");
                        out.write(line);
                        out.write(", \"output\": \"{\\\"action\\\": \\\"");
                        out.write(Json.escape(action));
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
        String filtered = shortEpisodes == 0 && idleDropped == 0 ? "" : String.format(Locale.ROOT,
                "\nfilters: %d episode(s) shorter than %ds left out, %d idle sample(s) thinned out (keeping %d%%)",
                shortEpisodes, cfg.minEpisodeSeconds, idleDropped, cfg.idleKeepPercent);
        if (lines == 0) {
            return shortEpisodes > 0
                    ? "every episode is shorter than " + cfg.minEpisodeSeconds
                            + "s - lower \"Min episode\" on the Power tab (/pymcbot settings)" + filtered
                    : "no samples found in " + episodesDir + " - record some playtime first";
        }
        return String.format(Locale.ROOT,
                "exported %d sample(s) from %d episode file(s) -> %s (%.1f MB)",
                lines, files, datasetPath, Files.exists(datasetPath) ? datasetPath.toFile().length() / 1048576.0 : 0.0)
                + filtered
                + (advanced
                ? "\nadvanced data included (entities, items, held item, armour, drops)"
                : "\nbasic recording - run /pymcbot advanced on before recording to also export entities and items");
    }

    /**
     * Decides which {@code none} (standing still, no input) samples an export keeps.
     *
     * <p>Every other action is kept, and so is the first {@code none} after it - that sample is
     * where the player stopped, which the model needs to learn. Inside a longer idle stretch a
     * budget grows by {@code keepPercent} per sample and a sample is kept each time it reaches
     * 100, so 25 keeps exactly every fourth one: evenly spread and the same on every export.
     */
    public static final class IdleThinner {
        private final int keepPercent;
        private int budget;
        private boolean idle;

        public IdleThinner(int keepPercent) {
            this.keepPercent = Math.max(0, Math.min(100, keepPercent));
        }

        public boolean keep(String action) {
            if (!"none".equals(action)) {
                idle = false;
                return true;
            }
            if (!idle) {
                idle = true;
                budget = 0;
                return true;
            }
            budget += keepPercent;
            if (budget >= 100) {
                budget -= 100;
                return true;
            }
            return false;
        }
    }

    /** The largest {@code "tick"} in an episode file (its length in client ticks), 0 if unreadable. */
    static long lastTick(Path file) {
        long last = 0;
        try (var reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                last = Math.max(last, tickOf(line));
            }
        } catch (IOException ignored) {
            // An unreadable file counts as empty (and is skipped by the length filter).
        }
        return last;
    }

    /** Pulls the {@code "tick"} number out of a raw episode line (0 when missing). */
    public static long tickOf(String rawLine) {
        String needle = "\"tick\": ";
        int start = rawLine.indexOf(needle);
        if (start < 0) {
            return 0;
        }
        start += needle.length();
        int end = start;
        while (end < rawLine.length() && Character.isDigit(rawLine.charAt(end))) {
            end++;
        }
        try {
            return end == start ? 0 : Long.parseLong(rawLine.substring(start, end));
        } catch (NumberFormatException e) {
            return 0;
        }
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
            sb.append(paused ? "PAUSED '" : "RECORDING '").append(runName).append("' - ").append(totalSamples)
              .append(" sample(s), ").append(episodeSamples).append(" in episode ").append(episodeNumber)
              .append(" (~").append(String.format(Locale.ROOT, "%.1f", samplesToMinutes(totalSamples)))
              .append(" min)\n");
        } else {
            sb.append("not recording\n");
        }
        if (active && !skippedSummary().isEmpty()) {
            sb.append(skippedSummary().substring(1)).append('\n');
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

    /** One line about the Minecraftism VL screenshots ({@code /pymcbot screenshots status}). */
    public String screenshotStatus() {
        return shots.status();
    }

    /** Screenshots written to disk this session. */
    public long screenshotsWritten() {
        return shots.written();
    }

    public synchronized boolean isActive() {
        return active;
    }

    public synchronized boolean isPaused() {
        return paused;
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

    /** Everything the on-screen badge needs, read under one lock so a frame never sees a half-updated state. */
    public synchronized HudState hudState() {
        boolean inMenu = active && PlaytimeConfig.get().pauseInMenus && Mc.screenOpen(client);
        return new HudState(active, paused, active && idleSkipping, inMenu, totalSamples,
                active ? System.currentTimeMillis() - runStartedMillis : 0L, runName);
    }

    /** Snapshot of the recorder for the HUD badge and the PyMC Bot settings page. */
    public record HudState(boolean active, boolean paused, boolean idle, boolean inMenu, long samples,
                           long elapsedMillis, String runName) {
        /** Short state label: REC, PAUSED, IDLE (AFK samples skipped), MENU (screen open) or OFF. */
        public String label() {
            if (!active) {
                return "OFF";
            }
            if (paused) {
                return "PAUSED";
            }
            if (inMenu) {
                return "MENU";
            }
            return idle ? "IDLE" : "REC";
        }

        /** Elapsed run time as m:ss or h:mm:ss. */
        public String clock() {
            long secs = elapsedMillis / 1000L;
            long h = secs / 3600L;
            long m = (secs % 3600L) / 60L;
            long sec = secs % 60L;
            return h > 0 ? String.format(Locale.ROOT, "%d:%02d:%02d", h, m, sec)
                    : String.format(Locale.ROOT, "%d:%02d", m, sec);
        }
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
        /** Screenshot path relative to the pymcbot folder (Minecraftism VL), or null. */
        String shot;
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
