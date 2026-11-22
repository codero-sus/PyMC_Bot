package dev.pymc.pymcb;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.time.Duration;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/** Standalone helper, launched outside Minecraft after shutdown to avoid locked mod jars. */
public final class ModUpdater {
    private static final String REPO = "https://github.com/codero-sus/PyMC_Bot/releases/download/";
    private static final String API = "https://api.github.com/repos/codero-sus/PyMC_Bot/releases?per_page=30";
    private ModUpdater() { }

    public static void main(String[] args) {
        if (args.length != 5 || !(args[4].equals("check") || args[4].equals("install"))) return;
        try {
            long parent = Long.parseLong(args[0]);
            if (args[4].equals("install")) {
                ProcessHandle.of(parent).ifPresent(p -> {
                    try { p.onExit().get(5, java.util.concurrent.TimeUnit.MINUTES); }
                    catch (Exception e) { throw new IllegalStateException(e); }
                });
            }
            Path jar = Path.of(args[1]).toAbsolutePath().normalize();
            String mc = args[2], current = args[3].split("\\+", 2)[0];
            if (!mc.matches("[0-9]+(?:\\.[0-9]+)+") || !current.matches("[0-9]+(?:\\.[0-9]+)+")
                    || !Files.isRegularFile(jar, LinkOption.NOFOLLOW_LINKS)
                    || !jar.getFileName().toString().equals("pymcbot-" + current + "+" + mc + ".jar")) return;
            HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10))
                    .followRedirects(HttpClient.Redirect.NORMAL).build();
            Object parsed = Json.parse(new String(fetch(client, API, 2_000_000), java.nio.charset.StandardCharsets.UTF_8));
            if (!(parsed instanceof List<?> releases)) return;
            Map<?, ?> best = null; String version = "";
            for (Object entry : releases) {
                if (!(entry instanceof Map<?, ?> release) || Boolean.TRUE.equals(release.get("draft"))
                        || Boolean.TRUE.equals(release.get("prerelease"))) continue;
                String tag = String.valueOf(release.get("tag_name"));
                if (!tag.matches("pymcbot-v[0-9]+(?:\\.[0-9]+)+")) continue;
                String candidate = tag.substring("pymcbot-v".length());
                if (compare(candidate, current) > 0 && (best == null || compare(candidate, version) > 0)) {
                    best = release; version = candidate;
                }
            }
            // Startup probes never mutate the live Minecraft installation. Recheck on shutdown.
            if (args[4].equals("check") || best == null || !(best.get("assets") instanceof List<?> assets)) return;
            String wanted = "pymcbot-" + version + "+" + mc + ".jar";
            for (Object item : assets) {
                if (!(item instanceof Map<?, ?> asset) || !wanted.equals(asset.get("name"))) continue;
                String digest = String.valueOf(asset.get("digest"));
                String url = String.valueOf(asset.get("browser_download_url"));
                Object size = asset.get("size");
                if (!digest.matches("sha256:[0-9a-fA-F]{64}") || !url.startsWith(REPO)
                        || !(size instanceof Number) || ((Number) size).longValue() <= 0
                        || ((Number) size).longValue() > 50_000_000) return;
                byte[] bytes = fetch(client, url, 50_000_000);
                if (bytes.length != ((Number) size).longValue()
                        || !HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes))
                                .equalsIgnoreCase(digest.substring(7)) || !validJar(bytes, mc, version)) return;
                Path folder = jar.getParent();
                Path dest = folder.resolve(wanted);
                if (Files.exists(dest, LinkOption.NOFOLLOW_LINKS)) return;
                Path tmp = Files.createTempFile(folder, ".pymcbot-update-", ".part");
                Path backup = folder.resolve(jar.getFileName() + ".old-" + UUID.randomUUID());
                try {
                    Files.write(tmp, bytes);
                    Files.move(jar, backup);
                    try { Files.move(tmp, dest, StandardCopyOption.ATOMIC_MOVE); }
                    catch (java.nio.file.AtomicMoveNotSupportedException e) { Files.move(tmp, dest); }
                } catch (Exception e) {
                    if (!Files.exists(jar) && Files.exists(backup)) Files.move(backup, jar);
                    throw e;
                } finally { Files.deleteIfExists(tmp); }
                return;
            }
        } catch (Exception ignored) {
            // Retry after a later game shutdown; keep the installed jar on any error.
        } finally {
            try {
                Path root = Path.of(ModUpdater.class.getProtectionDomain().getCodeSource().getLocation().toURI());
                if (Files.isDirectory(root) && root.getFileName().toString().startsWith("pymcbot-update-")) {
                    try (var files = Files.walk(root)) {
                        files.sorted(java.util.Comparator.reverseOrder()).forEach(path -> {
                            try { Files.deleteIfExists(path); } catch (Exception ignored) { }
                        });
                    }
                }
            } catch (Exception ignored) { }
        }
    }

    static int compare(String a, String b) {
        String[] x = a.split("\\."), y = b.split("\\.");
        for (int i = 0; i < Math.max(x.length, y.length); i++) {
            long left = i < x.length ? Long.parseLong(x[i]) : 0;
            long right = i < y.length ? Long.parseLong(y[i]) : 0;
            if (left != right) return Long.compare(left, right);
        }
        return 0;
    }

    static byte[] fetch(HttpClient client, String url, int limit) throws Exception {
        HttpRequest request = HttpRequest.newBuilder(URI.create(url)).timeout(Duration.ofSeconds(60))
                .header("User-Agent", "PyMC-Bot-mod-updater").GET().build();
        HttpResponse<InputStream> response = client.send(request, HttpResponse.BodyHandlers.ofInputStream());
        try (InputStream in = response.body()) {
            if (response.statusCode() != 200) throw new java.io.IOException("HTTP " + response.statusCode());
            byte[] data = in.readNBytes(limit + 1);
            if (data.length > limit) throw new java.io.IOException("response too large");
            return data;
        }
    }

    static boolean validJar(byte[] bytes, String minecraft, String version) throws Exception {
        try (ZipInputStream zip = new ZipInputStream(new ByteArrayInputStream(bytes))) {
            ZipEntry entry;
            while ((entry = zip.getNextEntry()) != null) {
                if (!"fabric.mod.json".equals(entry.getName())) continue;
                byte[] data = zip.readNBytes(1_000_001);
                if (data.length > 1_000_000) return false;
                Object meta = Json.parse(new String(data, java.nio.charset.StandardCharsets.UTF_8));
                return "pymcbot".equals(Json.get(meta, "id")) && minecraft.equals(Json.get(meta, "depends", "minecraft"))
                        && (version + "+" + minecraft).equals(Json.get(meta, "version"));
            }
        }
        return false;
    }
}
