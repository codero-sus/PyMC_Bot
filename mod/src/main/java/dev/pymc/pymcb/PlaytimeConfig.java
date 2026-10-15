package dev.pymc.pymcb;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Properties;

import net.fabricmc.loader.api.FabricLoader;

/**
 * Recorder configuration, stored as {@code config/pymcplaytime.properties} so the sampling
 * rate can be changed without rebuilding the mod.
 */
public final class PlaytimeConfig {
    private static final String FILE_NAME = "pymcplaytime.properties";
    private static final PlaytimeConfig INSTANCE = load();

    /** How many client ticks between two dataset samples. 2 ticks = 100 ms = 20 Hz. */
    public int sampleTicks = 2;
    /** Minimum distance (blocks) moved before an unattributed move counts as "move". */
    public double moveEpsilon = 0.02;
    /** Minimum accumulated view change (degrees) before the sample is labelled "look". */
    public float turnDegrees = 1.0f;
    /** Maximum number of nearby entities written per sample. */
    public int maxNearby = 12;
    /** Whether to write the 3x3x3 block neighbourhood (adds ~600 bytes per sample). */
    public boolean includeBlocks = true;

    // ---------------------------------------------------------------- advanced mode
    /** Write the advanced half of the schema (entities, items, armour, drops). */
    public boolean advanced = false;
    /** Maximum entities written per sample in advanced mode. */
    public int maxEntities = 24;
    /** How far (blocks) an entity may be and still be written in advanced mode. */
    public double entityRange = 32.0;
    /** Maximum dropped item stacks written per sample in advanced mode. */
    public int maxGroundItems = 8;

    // ---------------------------------------------------------------- dataset quality
    /** Skip samples while a screen (inventory, chat, pause menu...) is open. */
    public boolean pauseInMenus = true;
    /**
     * Stop writing samples after this many seconds without any input or movement, so AFK time
     * does not flood the dataset with "none". Recording resumes on the next action. 0 = never.
     */
    public int idleSkipSeconds = 15;
    /** Start recording automatically when joining a world or server. */
    public boolean autoRecord = false;
    /** Show a chat line when the hotkey or auto-record changes the recorder state. */
    public boolean notifications = true;

    // ---------------------------------------------------------------- on-screen badge
    /** Corners the HUD badge can sit in (cycled by the settings screen and /pymc hud corner). */
    public static final String[] HUD_CORNERS = {"top_left", "top_right", "bottom_left", "bottom_right"};
    /** Show the REC / PAUSED badge while a recording is running. */
    public boolean showHud = true;
    /** Where the badge is drawn: one of {@link #HUD_CORNERS}. */
    public String hudCorner = "top_left";

    public static PlaytimeConfig get() {
        return INSTANCE;
    }

    private static PlaytimeConfig load() {
        PlaytimeConfig cfg = new PlaytimeConfig();
        Path path = path();
        if (Files.isRegularFile(path)) {
            Properties props = new Properties();
            try (InputStream in = Files.newInputStream(path)) {
                props.load(in);
                cfg.sampleTicks = intProp(props, "sampleTicks", cfg.sampleTicks, 1, 200);
                cfg.moveEpsilon = doubleProp(props, "moveEpsilon", cfg.moveEpsilon, 0.0, 10.0);
                cfg.turnDegrees = (float) doubleProp(props, "turnDegrees", cfg.turnDegrees, 0.0, 180.0);
                cfg.maxNearby = intProp(props, "maxNearby", cfg.maxNearby, 0, 64);
                cfg.includeBlocks = boolProp(props, "includeBlocks", cfg.includeBlocks);
                cfg.advanced = boolProp(props, "advanced", cfg.advanced);
                cfg.maxEntities = intProp(props, "maxEntities", cfg.maxEntities, 0, 128);
                cfg.entityRange = doubleProp(props, "entityRange", cfg.entityRange, 4.0, 128.0);
                cfg.maxGroundItems = intProp(props, "maxGroundItems", cfg.maxGroundItems, 0, 64);
                cfg.pauseInMenus = boolProp(props, "pauseInMenus", cfg.pauseInMenus);
                cfg.idleSkipSeconds = intProp(props, "idleSkipSeconds", cfg.idleSkipSeconds, 0, 3600);
                cfg.autoRecord = boolProp(props, "autoRecord", cfg.autoRecord);
                cfg.notifications = boolProp(props, "notifications", cfg.notifications);
                cfg.showHud = boolProp(props, "showHud", cfg.showHud);
                cfg.hudCorner = cornerProp(props.getProperty("hudCorner"), cfg.hudCorner);
            } catch (IOException ignored) {
                // Fall back to defaults; the recorder still works.
            }
        } else {
            cfg.save();
        }
        return cfg;
    }

    /** Writes the current configuration back to disk. */
    public void save() {
        Path path = path();
        Properties props = new Properties();
        props.setProperty("sampleTicks", Integer.toString(sampleTicks));
        props.setProperty("moveEpsilon", Double.toString(moveEpsilon));
        props.setProperty("turnDegrees", Double.toString(turnDegrees));
        props.setProperty("maxNearby", Integer.toString(maxNearby));
        props.setProperty("includeBlocks", Boolean.toString(includeBlocks));
        props.setProperty("advanced", Boolean.toString(advanced));
        props.setProperty("maxEntities", Integer.toString(maxEntities));
        props.setProperty("entityRange", Double.toString(entityRange));
        props.setProperty("maxGroundItems", Integer.toString(maxGroundItems));
        props.setProperty("pauseInMenus", Boolean.toString(pauseInMenus));
        props.setProperty("idleSkipSeconds", Integer.toString(idleSkipSeconds));
        props.setProperty("autoRecord", Boolean.toString(autoRecord));
        props.setProperty("notifications", Boolean.toString(notifications));
        props.setProperty("showHud", Boolean.toString(showHud));
        props.setProperty("hudCorner", hudCorner);
        props.setProperty("_comment", "PyMC Playtime Recorder - sampleTicks=2 means one dataset sample every 100 ms");
        try {
            Files.createDirectories(path.getParent());
            try (OutputStream out = Files.newOutputStream(path)) {
                props.store(out, "PyMC Playtime Recorder configuration");
            }
        } catch (IOException ignored) {
            // Best effort.
        }
    }

    public void setSampleTicks(int ticks) {
        sampleTicks = Math.max(1, Math.min(200, ticks));
        save();
    }

    /** Turn advanced (entity + item) recording on or off and persist it. */
    public void setAdvanced(boolean enabled) {
        advanced = enabled;
        save();
    }

    public void setMaxEntities(int limit) {
        maxEntities = Math.max(0, Math.min(128, limit));
        save();
    }

    public void setMaxGroundItems(int limit) {
        maxGroundItems = Math.max(0, Math.min(64, limit));
        save();
    }

    public void setPauseInMenus(boolean enabled) {
        pauseInMenus = enabled;
        save();
    }

    public void setIdleSkipSeconds(int seconds) {
        idleSkipSeconds = Math.max(0, Math.min(3600, seconds));
        save();
    }

    public void setAutoRecord(boolean enabled) {
        autoRecord = enabled;
        save();
    }

    public void setNotifications(boolean enabled) {
        notifications = enabled;
        save();
    }

    public void setShowHud(boolean enabled) {
        showHud = enabled;
        save();
    }

    /** Sets the badge corner; unknown names are ignored. Returns whether it was accepted. */
    public boolean setHudCorner(String corner) {
        String normalized = cornerProp(corner, null);
        if (normalized == null) {
            return false;
        }
        hudCorner = normalized;
        save();
        return true;
    }

    /** Moves the badge to the next corner (clockwise order of {@link #HUD_CORNERS}). */
    public String cycleHudCorner() {
        int index = 0;
        for (int i = 0; i < HUD_CORNERS.length; i++) {
            if (HUD_CORNERS[i].equals(hudCorner)) {
                index = i;
            }
        }
        hudCorner = HUD_CORNERS[(index + 1) % HUD_CORNERS.length];
        save();
        return hudCorner;
    }

    /** Human label for a corner id, e.g. top_left -> "Top left". */
    public static String cornerLabel(String corner) {
        String spaced = corner.replace('_', ' ');
        return spaced.isEmpty() ? spaced : Character.toUpperCase(spaced.charAt(0)) + spaced.substring(1);
    }

    public String describe() {
        return String.format(Locale.ROOT,
                "sample every %d tick(s) (~%d ms), moveEpsilon=%.3f, turnDegrees=%.1f, maxNearby=%d, includeBlocks=%s",
                sampleTicks, sampleTicks * 50, moveEpsilon, turnDegrees, maxNearby, includeBlocks)
                + String.format(Locale.ROOT,
                "\nadvanced=%s (entities<=%d within %.0f blocks, ground items<=%d)",
                advanced, maxEntities, entityRange, maxGroundItems)
                + String.format(Locale.ROOT,
                "\npauseInMenus=%s, idleSkip=%s, autoRecord=%s, notifications=%s, hud=%s (%s)",
                pauseInMenus, idleSkipSeconds == 0 ? "off" : idleSkipSeconds + "s", autoRecord, notifications,
                showHud ? "on" : "off", hudCorner);
    }

    private static Path path() {
        return FabricLoader.getInstance().getConfigDir().resolve(FILE_NAME);
    }

    private static int intProp(Properties props, String key, int fallback, int min, int max) {
        try {
            int value = Integer.parseInt(props.getProperty(key, Integer.toString(fallback)).trim());
            return Math.max(min, Math.min(max, value));
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private static double doubleProp(Properties props, String key, double fallback, double min, double max) {
        try {
            double value = Double.parseDouble(props.getProperty(key, Double.toString(fallback)).trim());
            return Math.max(min, Math.min(max, value));
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private static String cornerProp(String raw, String fallback) {
        if (raw == null) {
            return fallback;
        }
        String wanted = raw.trim().toLowerCase(Locale.ROOT).replace('-', '_').replace(' ', '_');
        for (String corner : HUD_CORNERS) {
            if (corner.equals(wanted)) {
                return corner;
            }
        }
        return fallback;
    }

    private static boolean boolProp(Properties props, String key, boolean fallback) {
        String raw = props.getProperty(key);
        return raw == null ? fallback : Boolean.parseBoolean(raw.trim());
    }
}
