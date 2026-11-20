package dev.pymc.pymcb.client;

import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import dev.pymc.pymcb.Json;
import dev.pymc.pymcb.client.compat.Mc;
import net.minecraft.client.Minecraft;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.phys.Vec3;

/** Explicit, short-burst control of the local player, not a separate Mineflayer account. */
public final class PlayerTakeover {
    private static final Set<String> GROUND = Set.of(
            "minecraft:stone", "minecraft:cobblestone", "minecraft:deepslate", "minecraft:cobbled_deepslate",
            "minecraft:dirt", "minecraft:grass_block", "minecraft:coarse_dirt", "minecraft:rooted_dirt",
            "minecraft:andesite", "minecraft:granite", "minecraft:diorite", "minecraft:tuff");
    private final ExecutorService worker = Executors.newSingleThreadExecutor(r -> {
        Thread thread = new Thread(r, "pymcbot-local-control");
        thread.setDaemon(true);
        return thread;
    });
    private boolean hooked, connecting, active, waitingForGame, pending, pressing;
    private String panel = "", token = "", detail = "Not hooked";
    private Object world;
    private UUID playerId;
    private long generation, hookTicket, requestStarted, nextRequest, sessionStarted;
    private int remaining;
    private float lastYaw, lastHealth;

    public boolean active() { return active; }
    public boolean hooked() { return hooked; }
    public String status() { return active ? "TAKEOVER ACTIVE - open a menu or use release key" : detail; }

    /** Called only from the client's game thread. Never starts control by itself. */
    public String hook(Minecraft mc) {
        if (active) return "Release control before reconnecting.";
        if (connecting || pending) return "A backend request is still in progress.";
        String base = BotLink.panel(), secret = BotLink.token();
        long request = ++generation;
        hookTicket = request;
        connecting = true; hooked = false; detail = "Connecting to the PyMC Bot panel...";
        worker.execute(() -> {
            String outcome;
            boolean ok;
            try {
                Object result = BotLink.request(base, "GET", "/api/takeover/status", null, secret);
                ok = Boolean.TRUE.equals(Json.get(result, "ok")) && Double.valueOf(1).equals(Json.get(result, "protocol"));
                outcome = ok ? "Hooked: " + Json.text(Json.get(result, "detail"), "ready")
                        : "Hook refused: " + Json.text(Json.get(result, "detail"), "not ready");
            } catch (Exception e) {
                ok = false; outcome = "Hook failed. Check the panel address, login and AI movement settings.";
            }
            final boolean ready = ok;
            final String message = outcome;
            mc.execute(() -> {
                if (request == hookTicket) connecting = false;
                if (request != generation) return;
                if (!base.equals(BotLink.panel()) || !secret.equals(BotLink.token())) {
                    hooked = false; detail = "Panel configuration changed; hook again."; return;
                }
                hooked = ready; panel = base; token = secret; detail = message;
                BotLink.tell(message);
            });
        });
        return detail;
    }

    /** Called by a local UI click after explicit confirmation. */
    public String start(Minecraft mc) {
        if (active) return "Takeover is already active.";
        if (!hooked || connecting || pending || !panel.equals(BotLink.panel()) || !token.equals(BotLink.token())) {
            return "Hook to the configured panel first.";
        }
        if (mc.player == null || mc.level == null) return "Join a world first.";
        if (!mc.isWindowActive()) return "Focus the game window first.";
        if (mc.options.keyUp.isDown() || mc.options.keyDown.isDown() || mc.options.keyLeft.isDown()
                || mc.options.keyRight.isDown() || mc.options.keyJump.isDown()) return "Release movement keys first.";
        active = true; waitingForGame = true; world = mc.level; playerId = mc.player.getUUID();
        lastYaw = mc.player.getYRot(); lastHealth = mc.player.getHealth(); nextRequest = 0; remaining = 0;
        sessionStarted = System.nanoTime(); generation++;
        detail = "Takeover armed. Close settings; open a menu or press the release hotkey to stop.";
        return detail;
    }

    /** Releasing never sends a server command and never executes a late response. */
    public String stop(Minecraft mc, String reason) {
        boolean wasActive = active;
        generation++;
        if (pressing) mc.options.keyUp.setDown(false);
        pressing = false; remaining = 0; active = false; waitingForGame = false;
        world = null; playerId = null; detail = reason;
        if (wasActive) BotLink.tell(reason);
        return reason;
    }

    public void disconnect(Minecraft mc) {
        stop(mc, "Takeover released on disconnect.");
        hooked = false; token = "";
    }

    /** Tick on the Minecraft game thread. Actions last at most four ticks. */
    public void tick(Minecraft mc) {
        if (!active) return;
        if (System.nanoTime() - sessionStarted > 300_000_000_000L) {
            stop(mc, "Takeover released: five-minute session limit."); return;
        }
        if (mc.player == null || mc.level == null || mc.level != world || !mc.player.getUUID().equals(playerId)
                || !panel.equals(BotLink.panel()) || !token.equals(BotLink.token())) {
            disconnect(mc); return;
        }
        if (!mc.isWindowActive()) { stop(mc, "Takeover released: game lost focus."); return; }
        if (Mc.screenOpen(mc)) {
            if (!waitingForGame) stop(mc, "Takeover released: a screen opened.");
            return;
        }
        waitingForGame = false;
        if (mc.options.keyDown.isDown() || mc.options.keyLeft.isDown() || mc.options.keyRight.isDown()
                || mc.options.keyJump.isDown() || mc.options.keyAttack.isDown() || mc.options.keyUse.isDown()
                || (!pressing && mc.options.keyUp.isDown())) {
            stop(mc, "Takeover released: manual input."); return;
        }
        float yaw = mc.player.getYRot();
        if (Math.abs(Math.IEEEremainder((double) yaw - lastYaw, 360.0)) > 2) {
            stop(mc, "Takeover released: manual camera movement."); return;
        }
        lastYaw = yaw;
        float health = mc.player.getHealth();
        if (health < lastHealth || health <= 8 || mc.player.getFoodData().getFoodLevel() <= 6 || !mc.player.onGround()) {
            stop(mc, "Takeover released: damage, low health/food or player not on ground."); return;
        }
        lastHealth = health;
        if (pressing && (!safeAhead(mc) || --remaining <= 0)) {
            mc.options.keyUp.setDown(false); pressing = false; remaining = 0;
        }
        long now = System.nanoTime();
        if (pending && now - requestStarted > 14_000_000_000L) {
            stop(mc, "Takeover released: backend timed out."); return;
        }
        if (pending || pressing || now < nextRequest) return;
        boolean safe = safeAhead(mc);
        boolean safeLeft = safeFacing(mc, yaw - 45);
        boolean safeRight = safeFacing(mc, yaw + 45);
        String body = "{\"health\":" + Json.number(mc.player.getHealth(), 2)
                + ",\"food\":" + mc.player.getFoodData().getFoodLevel()
                + ",\"yaw\":" + Json.number(yaw, 2) + ",\"on_ground\":true,\"safe_ahead\":" + safe
                + ",\"safe_left\":" + safeLeft + ",\"safe_right\":" + safeRight + "}";
        Vec3 origin = mc.player.position();
        long ticket = generation;
        String base = panel, secret = token;
        pending = true; requestStarted = now; nextRequest = now + 2_000_000_000L;
        worker.execute(() -> {
            Object response = null;
            try { response = BotLink.request(base, "POST", "/api/takeover/decide", body, secret); }
            catch (Exception ignored) { /* fail closed on the game thread */ }
            Object answer = response;
            mc.execute(() -> {
                pending = false;
                if (!active || ticket != generation) return;
                if (mc.player == null || mc.level != world || !mc.isWindowActive() || Mc.screenOpen(mc)
                        || !mc.player.getUUID().equals(playerId) || mc.player.getHealth() < lastHealth || mc.player.getHealth() <= 8
                        || mc.player.getFoodData().getFoodLevel() <= 6 || !mc.player.onGround()
                        || mc.player.position().distanceTo(origin) > 0.3
                        || Math.abs(Math.IEEEremainder((double) mc.player.getYRot() - yaw, 360.0)) > 2
                        || System.nanoTime() - requestStarted > 14_000_000_000L) {
                    stop(mc, "Takeover released: state changed while waiting."); return;
                }
                Object protocol = Json.get(answer, "protocol"), move = Json.get(answer, "move"), turn = Json.get(answer, "turn");
                if (!Double.valueOf(1).equals(protocol) || !(move instanceof String text)
                        || !(text.equals("forward") || text.equals("stop")) || !(turn instanceof Number)) {
                    stop(mc, "Takeover released: backend response unavailable or invalid."); return;
                }
                double degrees = ((Number) turn).doubleValue();
                if (!Double.isFinite(degrees) || Math.abs(degrees) > 10) {
                    stop(mc, "Takeover released: invalid turn."); return;
                }
                mc.player.setYRot(mc.player.getYRot() + (float) degrees);
                lastYaw = mc.player.getYRot();
                if (text.equals("forward") && safeAhead(mc)) {
                    remaining = 4; pressing = true; mc.options.keyUp.setDown(true);
                }
            });
        });
    }

    private boolean safeAhead(Minecraft mc) {
        return mc.player != null && safeFacing(mc, mc.player.getYRot());
    }

    /** Conservative three-lane sample, not a collision proof or a hidden-trap detector. */
    private boolean safeFacing(Minecraft mc, float yaw) {
        if (mc.player == null || mc.level == null || !mc.player.onGround()) return false;
        Vec3 p = mc.player.position();
        double radians = Math.toRadians(yaw);
        double forwardX = -Math.sin(radians), forwardZ = Math.cos(radians);
        for (double distance : new double[]{0, 0.4, 0.8, 1.2, 1.6}) {
            for (double lateral : new double[]{-0.3, 0, 0.3}) {
                BlockPos feet = BlockPos.containing(p.x + forwardX * distance + forwardZ * lateral, p.y,
                        p.z + forwardZ * distance - forwardX * lateral);
                String floor = BuiltInRegistries.BLOCK.getKey(mc.level.getBlockState(feet.below()).getBlock()).toString();
                String body = BuiltInRegistries.BLOCK.getKey(mc.level.getBlockState(feet).getBlock()).toString();
                String head = BuiltInRegistries.BLOCK.getKey(mc.level.getBlockState(feet.above()).getBlock()).toString();
                if (!GROUND.contains(floor) || !isAir(body) || !isAir(head)) return false;
            }
        }
        return true;
    }

    private static boolean isAir(String id) {
        return id.equals("minecraft:air") || id.equals("minecraft:cave_air");
    }
}
