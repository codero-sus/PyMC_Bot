package dev.pymc.pymcb.client;

import java.util.Locale;

import dev.pymc.pymcb.PlaytimeConfig;
import dev.pymc.pymcb.client.compat.Mc;
import net.minecraft.client.Minecraft;

/**
 * The on-screen recording badge: a small box in a corner of the HUD showing
 * <b>REC</b> (blinking red dot), <b>PAUSED</b>, <b>IDLE</b> (AFK samples are being skipped) or
 * <b>MENU</b> (a screen is open, sampling holds), plus the run clock and the sample count.
 *
 * <p>It is only drawn while a recording is running, disappears with F1, and can be turned off
 * or moved with {@code /pymcbot hud} or the PyMC Bot settings page ({@code /pymcbot settings}).
 */
public final class RecorderHud {
    static final int RED = 0xFFFF4545;
    static final int AMBER = 0xFFFFC53D;
    static final int GREY = 0xFFA0A6AE;
    static final int BLUE = 0xFF5CB8FF;
    static final int WHITE = 0xFFFFFFFF;
    static final int BACKGROUND = 0xB0000000;

    private static final int MARGIN = 4;
    private static final int PAD = 4;
    private static final int DOT = 6;

    private RecorderHud() {
    }

    /** HUD callback, registered through {@code Mc.registerHud}. */
    public static void paint(HudCanvas canvas) {
        PlaytimeRecorder recorder = PymcPlaytimeClient.recorder();
        PlaytimeConfig cfg = PlaytimeConfig.get();
        Minecraft mc = Minecraft.getInstance();
        if (recorder == null || !cfg.showHud || mc.player == null || Mc.hudHidden(mc)) {
            return;
        }
        PlaytimeRecorder.HudState state = recorder.hudState();
        if (state.active()) {
            draw(canvas, state, cfg.hudCorner, System.currentTimeMillis());
        }
    }

    /** Colour for a state label. */
    static int accent(String label) {
        return switch (label) {
            case "REC" -> RED;
            case "PAUSED" -> AMBER;
            case "MENU" -> BLUE;
            case "IDLE" -> GREY;
            default -> WHITE;
        };
    }

    /** "12,345 samples". */
    static String samples(long count) {
        return String.format(Locale.ROOT, "%,d sample%s", count, count == 1 ? "" : "s");
    }

    /** Draws the badge for {@code state} into {@code corner} (one of PlaytimeConfig.HUD_CORNERS). */
    static void draw(HudCanvas canvas, PlaytimeRecorder.HudState state, String corner, long nowMillis) {
        String label = state.label();
        String detail = "  " + state.clock() + "  " + samples(state.samples());
        int accent = accent(label);

        int textW = canvas.textWidth(label) + canvas.textWidth(detail);
        int w = PAD + DOT + 4 + textW + PAD;
        int h = Math.max(canvas.lineHeight(), DOT) + PAD * 2;
        int x = corner.endsWith("right") ? canvas.width() - w - MARGIN : MARGIN;
        int y = corner.startsWith("bottom") ? canvas.height() - h - MARGIN : MARGIN;

        canvas.fill(x, y, x + w, y + h, BACKGROUND);
        canvas.fill(x, y, x + 1, y + h, accent);

        // The REC dot blinks once a second; the other states show a steady dot.
        boolean lit = !"REC".equals(label) || (nowMillis / 500L) % 2L == 0L;
        int dotColor = lit ? accent : (accent & 0x00FFFFFF) | 0x50000000;
        int dotY = y + (h - DOT) / 2;
        canvas.fill(x + PAD, dotY, x + PAD + DOT, dotY + DOT, dotColor);

        int textX = x + PAD + DOT + 4;
        int textY = y + (h - canvas.lineHeight()) / 2 + 1;
        canvas.text(label, textX, textY, accent);
        canvas.text(detail, textX + canvas.textWidth(label), textY, WHITE);
    }
}
