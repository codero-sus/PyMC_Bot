package dev.pymc.pymcb.client;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ConnectException;
import java.net.HttpURLConnection;
import java.net.SocketTimeoutException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Function;

import dev.pymc.pymcb.Json;
import dev.pymc.pymcb.PlaytimeConfig;
import dev.pymc.pymcb.client.compat.Mc;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;

/**
 * Talks to the PyMC Bot control panel (the Python app, {@code python -m pymc_bot}) over its HTTP
 * API, so the bots can be driven from inside the game: {@code /pymcbot bot status}, {@code do},
 * {@code come}, {@code auto on}... Requests run on a background thread and the answer is printed
 * in chat when it arrives, so the game never freezes while a bot chops a tree for a minute.
 *
 * <p>Only plain {@link HttpURLConnection} from {@code java.base} is used - it exists in every Java
 * runtime Minecraft ships with.</p>
 */
public final class BotLink {
    private static final ExecutorService POOL = Executors.newFixedThreadPool(2, runnable -> {
        Thread thread = new Thread(runnable, "pymcbot-botlink");
        thread.setDaemon(true);
        return thread;
    });

    private BotLink() {
    }

    public static String panel() {
        return PlaytimeConfig.get().panelUrl;
    }

    /** GET {@code path}; {@code describe} turns the JSON answer into the chat line(s). */
    public static void get(String path, Function<Object, String> describe) {
        send("GET", path, null, describe);
    }

    public static void post(String path, String body, Function<Object, String> describe) {
        send("POST", path, body, describe);
    }

    public static void put(String path, String body, Function<Object, String> describe) {
        send("PUT", path, body, describe);
    }

    static void send(String method, String path, String body, Function<Object, String> describe) {
        String base = panel();
        POOL.execute(() -> {
            String message;
            try {
                message = describe.apply(request(base, method, path, body));
            } catch (ConnectException exc) {
                message = "the PyMC Bot panel is not running at " + base
                        + " - start it with: python -m pymc_bot   (other PC? /pymcbot bot panel <url>)";
            } catch (SocketTimeoutException exc) {
                message = "the panel at " + base + " did not answer in time";
            } catch (IOException | RuntimeException exc) {
                message = "panel: " + exc.getMessage();
            }
            tell(message);
        });
    }

    /** Synchronous request (called on the worker thread). Errors carry the panel's {@code detail}. */
    static Object request(String base, String method, String path, String body) throws IOException {
        HttpURLConnection connection = (HttpURLConnection) URI.create(base + path).toURL().openConnection();
        connection.setRequestMethod(method);
        connection.setConnectTimeout(3000);
        connection.setReadTimeout(180_000); // skills such as chop_tree or fish take a while
        connection.setRequestProperty("Accept", "application/json");
        connection.setRequestProperty("User-Agent", "PyMC-Bot-mod");
        if (body != null) {
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            connection.setDoOutput(true);
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setFixedLengthStreamingMode(bytes.length);
            try (OutputStream out = connection.getOutputStream()) {
                out.write(bytes);
            }
        }
        int code = connection.getResponseCode();
        InputStream stream = code >= 400 ? connection.getErrorStream() : connection.getInputStream();
        String text = "";
        if (stream != null) {
            try (InputStream in = stream) {
                ByteArrayOutputStream buffer = new ByteArrayOutputStream();
                byte[] chunk = new byte[8192];
                int read;
                while ((read = in.read(chunk)) > 0 && buffer.size() < 4_000_000) {
                    buffer.write(chunk, 0, read);
                }
                text = buffer.toString(StandardCharsets.UTF_8);
            }
        }
        Object doc;
        try {
            doc = text.isBlank() ? null : Json.parse(text);
        } catch (IllegalArgumentException exc) {
            doc = text;
        }
        if (code >= 400) {
            Object detail = Json.get(doc, "detail");
            throw new IOException("HTTP " + code + (detail != null ? " - " + Json.text(detail, "") : ""));
        }
        return doc;
    }

    /** Prints lines in chat on the game thread. */
    static void tell(String message) {
        Minecraft client = Minecraft.getInstance();
        client.execute(() -> {
            if (client.player == null) {
                return;
            }
            for (String line : message.split("\n")) {
                Mc.message(client.player, Component.literal(PymcBotScreen.CHAT_PREFIX + line));
            }
        });
    }

    // ------------------------------------------------------------------ request bodies

    /** {@code count=8 block=oak_log} or a bare {@code oak_log} -> JSON options for a skill. */
    public static String options(String raw) {
        StringBuilder sb = new StringBuilder("{");
        if (raw != null) {
            int bare = 0;
            for (String token : raw.trim().split("\\s+")) {
                if (token.isEmpty()) {
                    continue;
                }
                String key;
                String value;
                int eq = token.indexOf('=');
                if (eq > 0) {
                    key = token.substring(0, eq);
                    value = token.substring(eq + 1);
                } else {
                    key = bare == 0 ? "name" : bare == 1 ? "count" : "extra" + bare;
                    value = token;
                    bare++;
                }
                if (sb.length() > 1) {
                    sb.append(',');
                }
                sb.append('"').append(Json.escape(key)).append("\":");
                if (value.matches("-?\\d+(\\.\\d+)?") || value.equals("true") || value.equals("false")) {
                    sb.append(value);
                } else {
                    sb.append('"').append(Json.escape(value)).append('"');
                }
            }
        }
        return sb.append('}').toString();
    }

    public static String skillBody(String name, String optionsJson) {
        return "{\"name\":\"" + Json.escape(name) + "\",\"params\":" + optionsJson + "}";
    }

    public static String quote(String text) {
        return "\"" + Json.escape(text) + "\"";
    }

    // ------------------------------------------------------------------ answers -> chat

    /** {@code /api/status} -> one or two lines about the selected bot. */
    public static String describeStatus(Object doc) {
        Object bot = Json.get(doc, "bot");
        String name = Json.text(Json.get(bot, "username"), "bot");
        String state = Json.text(Json.get(bot, "state"), "?");
        StringBuilder sb = new StringBuilder(name + ": " + state);
        if ("connected".equals(state)) {
            sb.append(" · hp ").append(Json.text(Json.get(bot, "health"), "?"))
                    .append(" · food ").append(Json.text(Json.get(bot, "food"), "?"));
            Object pos = Json.get(bot, "position");
            if (pos != null) {
                sb.append(" · at ").append(round(Json.get(pos, "x"))).append(' ')
                        .append(round(Json.get(pos, "y"))).append(' ').append(round(Json.get(pos, "z")));
            }
        }
        Object agent = Json.get(doc, "agent");
        sb.append("\nAI ").append(Boolean.TRUE.equals(Json.get(agent, "running")) ? "running" : "off")
                .append(" · autopilot ").append(Boolean.TRUE.equals(Json.get(doc, "autopilot", "enabled")) ? "on" : "off");
        Object lastRule = Json.get(doc, "autopilot", "bot", "last_rule");
        if (lastRule != null && !Json.text(lastRule, "").isEmpty()) {
            sb.append(" (last: ").append(Json.text(lastRule, "")).append(" - ")
                    .append(Json.text(Json.get(doc, "autopilot", "bot", "last_message"), "")).append(')');
        }
        Object fleet = Json.get(doc, "fleet", "connected");
        if (fleet != null) {
            sb.append(" · ").append(Json.text(fleet, "0")).append(" bot(s) online");
        }
        return sb.toString();
    }

    /** {@code /api/fleet/status} -> one line per bot. */
    public static String describeFleet(Object doc) {
        Object bots = Json.get(doc, "bots");
        if (!(bots instanceof List<?> list) || list.isEmpty()) {
            return "no bots yet - add some in the panel (Players)";
        }
        List<String> lines = new ArrayList<>();
        for (Object entry : list) {
            String line = Json.text(Json.get(entry, "username"), "?") + " - " + Json.text(Json.get(entry, "state"), "?");
            Object health = Json.get(entry, "health");
            if (health != null) {
                line += " · hp " + Json.text(health, "?");
            }
            if (Boolean.TRUE.equals(Json.get(entry, "selected"))) {
                line += "  (selected)";
            }
            lines.add(line);
        }
        return String.join("\n", lines);
    }

    /** {@code /api/skills} -> the names by group. */
    public static String describeSkills(Object doc) {
        Object skills = Json.get(doc, "skills");
        if (!(skills instanceof List<?> list)) {
            return "the panel did not list any skills (update PyMC Bot with the updater)";
        }
        Map<String, List<String>> groups = new java.util.LinkedHashMap<>();
        for (Object skill : list) {
            groups.computeIfAbsent(Json.text(Json.get(skill, "group"), "other"), key -> new ArrayList<>())
                    .add(Json.text(Json.get(skill, "name"), "?"));
        }
        StringBuilder sb = new StringBuilder(list.size() + " skills - /pymcbot bot do <skill> [key=value ...]");
        for (Map.Entry<String, List<String>> group : groups.entrySet()) {
            sb.append('\n').append(group.getKey().toUpperCase(Locale.ROOT)).append(": ").append(String.join(", ", group.getValue()));
        }
        return sb.toString();
    }

    /** Most write endpoints answer {@code {"ok": ..., "detail": ...}}. */
    public static Function<Object, String> describeResult(String what) {
        return doc -> {
            boolean ok = Boolean.TRUE.equals(Json.get(doc, "ok"));
            String detail = Json.text(Json.get(doc, "detail"), ok ? "done" : "nothing happened");
            return what + (ok ? ": " : " failed: ") + detail;
        };
    }

    private static String round(Object value) {
        return value instanceof Double d ? Long.toString(Math.round(d)) : Json.text(value, "?");
    }
}
