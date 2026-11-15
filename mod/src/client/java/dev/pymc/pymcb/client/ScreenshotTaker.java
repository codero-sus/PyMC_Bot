package dev.pymc.pymcb.client;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import com.mojang.blaze3d.platform.NativeImage;
import dev.pymc.pymcb.PlaytimeConfig;
import dev.pymc.pymcb.PymcPlaytimeMod;
import dev.pymc.pymcb.client.compat.Mc;
import net.minecraft.client.Minecraft;

/**
 * Saves the small screenshots the vision (VL) editions of the Minecraftism model learn from.
 *
 * <p>While recording with {@code screenshots=true}, one frame every {@code screenshotSeconds} is
 * read back from the game's main framebuffer, scaled down to {@code screenshotWidth} pixels on the
 * client thread (cheap: a few hundred thousand pixel copies) and written as PNG on a background
 * thread, so the game never waits for the disk. The sample recorded at that tick gets a
 * {@code "shot"} field with the path relative to the {@code pymcbot} folder, e.g.
 * {@code shots/run-20261021-101500-1/1200.png}.
 *
 * <p>Reading the framebuffer differs between Minecraft versions (synchronous before 1.21.5,
 * a GPU callback after), so that part lives in the version bridge, {@link Mc#screenshot}.
 */
public final class ScreenshotTaker {
    static final String SHOTS_DIR = "shots";
    /** Never queue more than this many unwritten screenshots (slow disks skip frames instead). */
    private static final int MAX_PENDING = 3;
    private static final ExecutorService WRITER = Executors.newSingleThreadExecutor(runnable -> {
        Thread thread = new Thread(runnable, "pymcbot-screenshots");
        thread.setDaemon(true);
        return thread;
    });

    private final AtomicInteger pending = new AtomicInteger();
    private final AtomicLong written = new AtomicLong();
    private final AtomicLong failed = new AtomicLong();
    private long lastTick = Long.MIN_VALUE / 2;
    private volatile String lastError = "";
    /** Set when the framebuffer could not be read at all: no more attempts this session. */
    private volatile boolean broken;

    /** Forget the timing of the previous run (called when a recording starts). */
    void reset() {
        lastTick = Long.MIN_VALUE / 2;
    }

    /**
     * Takes a screenshot when one is due and returns its path relative to {@code root}, or null.
     * The file appears a moment later; the dataset builder skips samples whose picture is missing.
     */
    String maybeCapture(Minecraft client, Path root, String run, long tick, long sampleTick) {
        PlaytimeConfig cfg = PlaytimeConfig.get();
        if (!cfg.screenshots || broken || root == null) {
            return null;
        }
        if (tick - lastTick < Math.max(1, cfg.screenshotSeconds) * 20L || pending.get() >= MAX_PENDING) {
            return null;
        }
        lastTick = tick;
        String relative = SHOTS_DIR + "/" + run + "/" + sampleTick + ".png";
        Path target = root.resolve(relative);
        int width = cfg.screenshotWidth;
        pending.incrementAndGet();
        try {
            Mc.screenshot(client, width, image -> save(image, target));
        } catch (Throwable error) {
            pending.decrementAndGet();
            broken = true;
            lastError = error.toString();
            PymcPlaytimeMod.LOGGER.warn("[pymcbot] screenshots disabled for this session: {}", error.toString());
            return null;
        }
        return relative;
    }

    private void save(NativeImage image, Path target) {
        if (image == null) {
            pending.decrementAndGet();
            failed.incrementAndGet();
            return;
        }
        try {
            WRITER.execute(() -> {
                try (image) {
                    Files.createDirectories(target.getParent());
                    image.writeToFile(target);
                    written.incrementAndGet();
                } catch (IOException | RuntimeException error) {
                    failed.incrementAndGet();
                    lastError = error.toString();
                } finally {
                    pending.decrementAndGet();
                }
            });
        } catch (RuntimeException rejected) {
            image.close();
            pending.decrementAndGet();
            failed.incrementAndGet();
        }
    }

    long written() {
        return written.get();
    }

    String status() {
        PlaytimeConfig cfg = PlaytimeConfig.get();
        if (!cfg.screenshots) {
            return "screenshots off - /pymcbot screenshots on (for the Minecraftism VL model)";
        }
        String state = broken ? "NOT WORKING on this version: " + lastError
                : "on, one every " + cfg.screenshotSeconds + "s at " + cfg.screenshotWidth + " px wide";
        return "screenshots " + state + " - " + written.get() + " saved, " + failed.get() + " failed"
                + (lastError.isEmpty() || broken ? "" : " (last error: " + lastError + ")");
    }

    /**
     * Scales {@code source} down to {@code maxWidth} pixels wide with a 2x2 box filter per output
     * pixel. Pixels are read and written through the same accessors, so the channel order the
     * version uses (RGBA or ARGB) does not matter: every byte lane is averaged on its own.
     */
    public static NativeImage downscale(NativeImage source, int maxWidth, PixelReader reader, PixelWriter writer) {
        int sw = source.getWidth();
        int sh = source.getHeight();
        if (sw <= maxWidth) {
            NativeImage copy = new NativeImage(sw, sh, false);
            for (int y = 0; y < sh; y++) {
                for (int x = 0; x < sw; x++) {
                    writer.set(copy, x, y, reader.get(source, x, y));
                }
            }
            return copy;
        }
        int dw = Math.max(1, maxWidth);
        int dh = Math.max(1, (int) Math.round(sh * (dw / (double) sw)));
        NativeImage out = new NativeImage(dw, dh, false);
        double fx = sw / (double) dw;
        double fy = sh / (double) dh;
        for (int y = 0; y < dh; y++) {
            int y0 = Math.min(sh - 1, (int) (y * fy));
            int y1 = Math.min(sh - 1, (int) ((y + 0.5) * fy));
            for (int x = 0; x < dw; x++) {
                int x0 = Math.min(sw - 1, (int) (x * fx));
                int x1 = Math.min(sw - 1, (int) ((x + 0.5) * fx));
                writer.set(out, x, y, average(reader.get(source, x0, y0), reader.get(source, x1, y0),
                        reader.get(source, x0, y1), reader.get(source, x1, y1)));
            }
        }
        return out;
    }

    private static int average(int a, int b, int c, int d) {
        int result = 0;
        for (int shift = 0; shift < 32; shift += 8) {
            int sum = ((a >>> shift) & 0xFF) + ((b >>> shift) & 0xFF) + ((c >>> shift) & 0xFF) + ((d >>> shift) & 0xFF);
            result |= ((sum + 2) / 4) << shift;
        }
        return result;
    }

    /** Reads one packed pixel (the version bridge supplies the accessor for its era). */
    @FunctionalInterface
    public interface PixelReader {
        int get(NativeImage image, int x, int y);
    }

    /** Writes one packed pixel. */
    @FunctionalInterface
    public interface PixelWriter {
        void set(NativeImage image, int x, int y, int value);
    }
}
