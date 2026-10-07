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

    public String describe() {
        return String.format(Locale.ROOT,
                "sample every %d tick(s) (~%d ms), moveEpsilon=%.3f, turnDegrees=%.1f, maxNearby=%d, includeBlocks=%s",
                sampleTicks, sampleTicks * 50, moveEpsilon, turnDegrees, maxNearby, includeBlocks);
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

    private static boolean boolProp(Properties props, String key, boolean fallback) {
        String raw = props.getProperty(key);
        return raw == null ? fallback : Boolean.parseBoolean(raw.trim());
    }
}
