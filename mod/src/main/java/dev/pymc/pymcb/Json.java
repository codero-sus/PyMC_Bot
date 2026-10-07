package dev.pymc.pymcb;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * Tiny JSON helpers. The mod has no runtime JSON library (Minecraft shadows Gson but
 * depending on it from a client mod is fragile), and the dataset only needs a handful of
 * primitives, so escaping and number rounding are done here.
 */
public final class Json {
    private Json() {
    }

    /** Escapes a string for embedding between double quotes. */
    public static String escape(String raw) {
        StringBuilder sb = new StringBuilder(raw.length() + 8);
        for (int i = 0; i < raw.length(); i++) {
            char c = raw.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                case '\b' -> sb.append("\\b");
                case '\f' -> sb.append("\\f");
                default -> {
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        return sb.toString();
    }

    /** Rounds a double to the given number of decimals, never emitting scientific notation. */
    public static String number(double value, int decimals) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return "0";
        }
        return BigDecimal.valueOf(value)
                .setScale(decimals, RoundingMode.HALF_UP)
                .toPlainString();
    }
}
