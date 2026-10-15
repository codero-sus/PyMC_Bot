package dev.pymc.pymcb.client;

/**
 * The few 2D drawing calls the badge and the settings screen need.
 *
 * <p>Minecraft renamed its GUI drawing class between versions ({@code GuiGraphics} in 1.21.x,
 * {@code GuiGraphicsExtractor} with {@code text(...)} instead of {@code drawString(...)} in 26.x),
 * so the version bridge ({@code compat/<era>/Mc.java}) wraps whichever one the game has in this
 * interface and the drawing code itself stays identical on every version.
 *
 * <p>Colours are ARGB; always include the alpha byte (1.21.6+ treats {@code 0x00RRGGBB} as invisible).
 */
public interface HudCanvas {
    /** Scaled GUI width. */
    int width();

    /** Scaled GUI height. */
    int height();

    /** Fills the rectangle from (x1, y1) inclusive to (x2, y2) exclusive. */
    void fill(int x1, int y1, int x2, int y2, int argb);

    /** Draws a line of text with a drop shadow, top-left at (x, y). */
    void text(String text, int x, int y, int argb);

    /** Width of {@code text} in GUI pixels. */
    int textWidth(String text);

    /** Height of one line of text. */
    int lineHeight();

    /** Draws {@code text} horizontally centred on {@code centerX}. */
    default void centeredText(String text, int centerX, int y, int argb) {
        text(text, centerX - textWidth(text) / 2, y, argb);
    }

    /** Cuts {@code text} with an ellipsis so it fits in {@code maxWidth} pixels. */
    default String fit(String text, int maxWidth) {
        if (textWidth(text) <= maxWidth) {
            return text;
        }
        String cut = text;
        while (!cut.isEmpty() && textWidth(cut + "...") > maxWidth) {
            cut = cut.substring(0, cut.length() - 1);
        }
        return cut + "...";
    }
}
