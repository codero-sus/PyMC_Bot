package dev.pymc.pymcb.client;

import com.sun.net.httpserver.HttpServer;
import dev.pymc.pymcb.Json;
import dev.pymc.pymcb.PlaytimeConfig;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public class BotLinkHarness {
    private static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
    }
    public static void main(String[] args) throws Exception {
        if (args.length == 1) {
            String backend = args[0];
            BotLink.request(backend, "PUT", "/api/config", "{\"minecraft\":{\"backend\":\"simulated\",\"username\":\"JavaBot\"}}", "");
            BotLink.request(backend, "POST", "/api/bot/start", "{}", "");
            long deadline = System.currentTimeMillis() + 8000;
            Object status;
            do {
                status = BotLink.request(backend, "GET", "/api/status", null, "");
                if ("connected".equals(Json.get(status, "bot", "state"))) break;
                Thread.sleep(100);
            } while (System.currentTimeMillis() < deadline);
            check("connected".equals(Json.get(status, "bot", "state")), "simulated bot not connected");
            String options = BotLink.options("count=2");
            Object result = BotLink.request(backend, "POST", "/api/bot/skill", BotLink.skillBody("chop_tree", options), "");
            check(Boolean.TRUE.equals(Json.get(result, "ok")), "skill failed: " + Json.text(Json.get(result, "detail"), ""));
            Object stop = BotLink.request(backend, "POST", "/api/bot/action", "{\"action\":\"stop\"}", "");
            check(Boolean.TRUE.equals(Json.get(stop, "ok")), "stop failed");
            BotLink.request(backend, "POST", "/api/bot/stop", "{}", "");
            System.out.println("Java client -> Python API -> simulated bot passed");
            return;
        }
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.setExecutor(Executors.newCachedThreadPool(r -> { Thread t = new Thread(r); t.setDaemon(true); return t; }));
        CountDownLatch started = new CountDownLatch(2), release = new CountDownLatch(1), stopped = new CountDownLatch(1);
        AtomicInteger skills = new AtomicInteger(), redirects = new AtomicInteger();
        server.createContext("/api/bot/skill", ex -> {
            skills.incrementAndGet(); started.countDown();
            try { release.await(5, TimeUnit.SECONDS); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            byte[] body = "{\"ok\":true,\"detail\":\"done\"}".getBytes(StandardCharsets.UTF_8);
            ex.sendResponseHeaders(200, body.length); ex.getResponseBody().write(body); ex.close();
        });
        server.createContext("/api/ai/stop", ex -> {
            byte[] body = "{\"ok\":true}".getBytes(StandardCharsets.UTF_8);
            ex.sendResponseHeaders(200, body.length); ex.getResponseBody().write(body); ex.close(); stopped.countDown();
        });
        server.createContext("/redirect", ex -> {
            ex.getResponseHeaders().add("Location", "/stolen"); ex.sendResponseHeaders(302, -1); ex.close();
        });
        server.createContext("/stolen", ex -> { redirects.incrementAndGet(); ex.sendResponseHeaders(200, -1); ex.close(); });
        server.createContext("/auth", ex -> { ex.sendResponseHeaders(401, -1); ex.close(); });
        server.createContext("/missing", ex -> {
            byte[] body = "{\"detail\":\"Not Found\"}".getBytes(StandardCharsets.UTF_8);
            ex.sendResponseHeaders(404, body.length); ex.getResponseBody().write(body); ex.close();
        });
        server.start();
        String base = "http://127.0.0.1:" + server.getAddress().getPort();
        PlaytimeConfig.get().panelUrl = base;
        try {
            try { BotLink.request(base, "GET", "/redirect", null, "secret"); throw new AssertionError("redirect accepted"); }
            catch (IOException expected) { check(expected.getMessage().contains("redirect"), "redirect error"); }
            check(redirects.get() == 0, "redirect forwarded credentials");
            try { BotLink.request(base, "GET", "/auth", null, "bad-token"); throw new AssertionError("auth accepted"); }
            catch (IOException expected) { check(expected.getMessage().contains("token"), "auth hint"); }
            try { BotLink.request(base, "GET", "/missing", null, ""); throw new AssertionError("404 accepted"); }
            catch (IOException expected) { check(expected.getMessage().contains("versions match"), "version hint"); }
            try { BotLink.request("http://user:password@127.0.0.1", "GET", "/", null, ""); throw new AssertionError("userinfo accepted"); }
            catch (IOException expected) { check(!expected.getMessage().contains("password"), "userinfo leaked"); }
            Object options = Json.parse(BotLink.options("direction=forward+left yaw=90 confirm=true"));
            check("forward+left".equals(Json.get(options, "direction")) && Boolean.TRUE.equals(Json.get(options, "confirm")), "options");
            BotLink.post("/api/bot/skill", "{}", BotLink.describeResult("one"));
            BotLink.post("/api/bot/skill", "{}", BotLink.describeResult("two"));
            check(started.await(3, TimeUnit.SECONDS), "skills did not start");
            BotLink.post("/api/bot/skill", "{}", BotLink.describeResult("queued"));
            BotLink.postPriority("/api/ai/stop", "{}", doc -> "stop requested");
            check(stopped.await(3, TimeUnit.SECONDS), "stop starved");
            release.countDown();
            Thread.sleep(300);
            check(skills.get() == 2, "queued skill was sent after stop");
            System.out.println("Java BotLink HTTP and queue tests passed");
        } finally {
            release.countDown(); server.stop(0);
        }
    }
}
