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

    /**
     * Reads a JSON document (the PyMC Bot panel's answers) into plain Java values:
     * {@code Map<String, Object>}, {@code List<Object>}, {@code String}, {@code Double},
     * {@code Boolean} or {@code null}. Throws {@link IllegalArgumentException} on bad input.
     */
    public static Object parse(String text) {
        Reader reader = new Reader(text == null ? "" : text);
        reader.space();
        Object value = reader.value();
        reader.space();
        if (reader.pos != reader.text.length()) {
            throw new IllegalArgumentException("trailing characters at " + reader.pos);
        }
        return value;
    }

    /** {@code get(doc, "bot", "state")}: walks nested objects; {@code null} when a step is missing. */
    public static Object get(Object doc, String... keys) {
        Object current = doc;
        for (String key : keys) {
            if (!(current instanceof java.util.Map<?, ?> map)) {
                return null;
            }
            current = map.get(key);
        }
        return current;
    }

    /** A value as short text for chat: whole numbers without ".0", {@code fallback} for null. */
    public static String text(Object value, String fallback) {
        if (value == null) {
            return fallback;
        }
        if (value instanceof Double d) {
            return d == Math.rint(d) && Math.abs(d) < 1e15 ? Long.toString(d.longValue()) : number(d, 1);
        }
        return String.valueOf(value);
    }

    private static final class Reader {
        final String text;
        int pos;

        Reader(String text) {
            this.text = text;
        }

        void space() {
            while (pos < text.length() && Character.isWhitespace(text.charAt(pos))) {
                pos++;
            }
        }

        Object value() {
            if (pos >= text.length()) {
                throw new IllegalArgumentException("unexpected end of JSON");
            }
            char c = text.charAt(pos);
            switch (c) {
                case '{':
                    return object();
                case '[':
                    return array();
                case '"':
                    return string();
                case 't':
                    return word("true", Boolean.TRUE);
                case 'f':
                    return word("false", Boolean.FALSE);
                case 'n':
                    return word("null", null);
                default:
                    return number();
            }
        }

        Object word(String word, Object value) {
            if (!text.startsWith(word, pos)) {
                throw new IllegalArgumentException("bad JSON at " + pos);
            }
            pos += word.length();
            return value;
        }

        java.util.Map<String, Object> object() {
            java.util.Map<String, Object> out = new java.util.LinkedHashMap<>();
            pos++;
            space();
            if (pos < text.length() && text.charAt(pos) == '}') {
                pos++;
                return out;
            }
            while (true) {
                space();
                String key = string();
                space();
                expect(':');
                space();
                out.put(key, value());
                space();
                if (pos < text.length() && text.charAt(pos) == ',') {
                    pos++;
                    continue;
                }
                expect('}');
                return out;
            }
        }

        java.util.List<Object> array() {
            java.util.List<Object> out = new java.util.ArrayList<>();
            pos++;
            space();
            if (pos < text.length() && text.charAt(pos) == ']') {
                pos++;
                return out;
            }
            while (true) {
                space();
                out.add(value());
                space();
                if (pos < text.length() && text.charAt(pos) == ',') {
                    pos++;
                    continue;
                }
                expect(']');
                return out;
            }
        }

        String string() {
            expect('"');
            StringBuilder sb = new StringBuilder();
            while (pos < text.length()) {
                char c = text.charAt(pos++);
                if (c == '"') {
                    return sb.toString();
                }
                if (c != '\\') {
                    sb.append(c);
                    continue;
                }
                if (pos >= text.length()) {
                    break;
                }
                char e = text.charAt(pos++);
                switch (e) {
                    case 'n' -> sb.append('\n');
                    case 't' -> sb.append('\t');
                    case 'r' -> sb.append('\r');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'u' -> {
                        if (pos + 4 > text.length()) {
                            throw new IllegalArgumentException("bad unicode escape");
                        }
                        sb.append((char) Integer.parseInt(text.substring(pos, pos + 4), 16));
                        pos += 4;
                    }
                    default -> sb.append(e);
                }
            }
            throw new IllegalArgumentException("unterminated string");
        }

        Double number() {
            int start = pos;
            while (pos < text.length() && "+-0123456789.eE".indexOf(text.charAt(pos)) >= 0) {
                pos++;
            }
            if (start == pos) {
                throw new IllegalArgumentException("bad JSON at " + pos);
            }
            return Double.valueOf(text.substring(start, pos));
        }

        void expect(char c) {
            if (pos >= text.length() || text.charAt(pos) != c) {
                throw new IllegalArgumentException("expected '" + c + "' at " + pos);
            }
            pos++;
        }
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
