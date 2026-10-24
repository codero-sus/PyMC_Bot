package dev.pymc.pymcb.client.compat;

import net.minecraft.client.Screenshot;
import dev.pymc.pymcb.client.ScreenshotTaker;
import com.mojang.blaze3d.platform.NativeImage;
import com.mojang.blaze3d.pipeline.RenderTarget;
import java.util.function.Consumer;

import com.mojang.blaze3d.platform.InputConstants;
import com.mojang.brigadier.arguments.ArgumentType;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import com.mojang.brigadier.builder.RequiredArgumentBuilder;
import dev.pymc.pymcb.client.HudCanvas;
import net.fabricmc.fabric.api.client.command.v2.ClientCommands;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.fabricmc.fabric.api.client.keymapping.v1.KeyMappingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.hud.HudElementRegistry;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.Font;
import net.minecraft.client.gui.GuiGraphicsExtractor;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;

/**
 * Version bridge for Minecraft 26.2 - 26.3.
 *
 * <p>Everything in the recorder that differs between Minecraft versions goes through this
 * class; the rest of the mod is identical for every target. There is one copy of this file
 * per API era under {@code mod/src/compat/<era>/}, and the build picks the right one from
 * {@code mod/versions.json} ({@code ./gradlew build -Pmc=<version>}).
 *
 * <p>What changed in this era: the open screen moved to {@code Minecraft.gui.screen()}. Fabric API 0.152 added attended client commands. 26.3 replaced GLFW with SDL3; the hotkey is registered unbound through {@code InputConstants.UNKNOWN}, so no raw key codes are involved.
 */
public final class Mc {
    /** The Minecraft versions this copy is compiled for (shown by {@code /pymcbot version}). */
    public static final String ERA = "26.2 - 26.3";

    /** Key mapping category (a registered object since 1.21.9). */
    private static final KeyMapping.Category CATEGORY =
            KeyMapping.Category.register(Identifier.fromNamespaceAndPath("pymcbot", "recorder"));

    private Mc() {
    }

    /** The hotbar slot (0-8) the player has selected. */
    public static int selectedSlot(LocalPlayer player) {
        return player.getInventory().getSelectedSlot();
    }

    /** The dimension id, e.g. {@code minecraft:overworld}. */
    public static String dimensionId(LocalPlayer player) {
        return player.level().dimension().identifier().toString();
    }

    /** A literal client command node. */
    public static LiteralArgumentBuilder<FabricClientCommandSource> literal(String name) {
        return ClientCommands.literal(name);
    }

    /** A client command argument node. */
    public static <T> RequiredArgumentBuilder<FabricClientCommandSource, T> argument(String name, ArgumentType<T> type) {
        return ClientCommands.argument(name, type);
    }

    /**
     * Whether the player really ran the command themselves.
     *
     * <p>Since 26.2 a server can no longer make the client run a command behind the player's back (for example from a clickable chat message): commands that change what is recorded require an attended run.
     */
    public static boolean attended(FabricClientCommandSource source) {
        return source.attended();
    }

    /**
     * Registers a key mapping under the "PyMC Bot" category, unbound by default so it never
     * clashes with another key (bind it in Options - Controls - Key Binds).
     */
    public static KeyMapping registerKey(String translationKey) {
        return KeyMappingHelper.registerKeyMapping(new KeyMapping(translationKey, InputConstants.UNKNOWN.getValue(), CATEGORY));
    }

    /** Whether a screen (inventory, chat, pause menu...) is open. Since 26.2 the open screen lives on {@code Minecraft.gui}. */
    public static boolean screenOpen(Minecraft client) {
        return client.gui.screen() != null;
    }

    /** Shows a line in the chat for the local player only. */
    public static void message(LocalPlayer player, Component text) {
        player.sendSystemMessage(text);
    }

    // ------------------------------------------------------------------ screenshots

    /**
     * Reads the current frame, scales it to {@code maxWidth} pixels and hands the small copy to
     * {@code done} (which owns and closes it). Since 1.21.5 the GPU read-back is asynchronous: the
     * callback runs a frame or two later, on the render thread.
     */
    public static void screenshot(Minecraft client, int maxWidth, Consumer<NativeImage> done) {
        Screenshot.takeScreenshot(mainRenderTarget(client), full -> {
            try (full) {
                done.accept(ScreenshotTaker.downscale(full, maxWidth, NativeImage::getPixel, NativeImage::setPixel));
            }
        });
    }

    /**
     * The game's main framebuffer. 26.2 removed {@code Minecraft.getMainRenderTarget()}; since 26.x
     * runs unobfuscated, the new home is found once by type (a {@code RenderTarget} reachable from
     * the client or one of its public renderer objects, preferring a member named "main") and the
     * route is remembered.
     */
    private static RenderTarget mainRenderTarget(Minecraft client) {
        if (targetRoute != null) {
            try {
                Object holder = targetRoute[0] == null ? client : ((java.lang.reflect.Field) targetRoute[0]).get(client);
                Object target = ((java.lang.reflect.Method) targetRoute[1]).invoke(holder);
                if (target instanceof RenderTarget renderTarget) {
                    return renderTarget;
                }
            } catch (ReflectiveOperationException | RuntimeException ignored) {
                targetRoute = null;
            }
        }
        java.util.List<java.lang.reflect.Field> holders = new java.util.ArrayList<>();
        holders.add(null);
        for (java.lang.reflect.Field field : Minecraft.class.getFields()) {
            if (!java.lang.reflect.Modifier.isStatic(field.getModifiers())
                    && field.getType().getName().startsWith("net.minecraft.client")) {
                holders.add(field);
            }
        }
        java.lang.reflect.Method best = null;
        java.lang.reflect.Field bestHolder = null;
        for (java.lang.reflect.Field holderField : holders) {
            Object holder;
            try {
                holder = holderField == null ? client : holderField.get(client);
            } catch (ReflectiveOperationException | RuntimeException e) {
                continue;
            }
            if (holder == null) {
                continue;
            }
            for (java.lang.reflect.Method method : holder.getClass().getMethods()) {
                if (method.getParameterCount() != 0 || java.lang.reflect.Modifier.isStatic(method.getModifiers())
                        || !RenderTarget.class.isAssignableFrom(method.getReturnType())) {
                    continue;
                }
                boolean main = method.getName().toLowerCase(java.util.Locale.ROOT).contains("main");
                if (best == null || main) {
                    best = method;
                    bestHolder = holderField;
                }
                if (main) {
                    break;
                }
            }
            if (best != null && best.getName().toLowerCase(java.util.Locale.ROOT).contains("main")) {
                break;
            }
        }
        if (best == null) {
            throw new IllegalStateException("no RenderTarget getter found on Minecraft or its renderers");
        }
        targetRoute = new Object[]{bestHolder, best};
        return mainRenderTarget(client);
    }

    /** {holder field or null for the client itself, getter} found by {@link #mainRenderTarget}. */
    private static Object[] targetRoute;

    // ------------------------------------------------------------------ GUI

    /** Opens {@code screen}, or returns to the game when it is null. Screens are managed by {@code Minecraft.gui} in this era. */
    public static void openScreen(Minecraft client, Screen screen) {
        client.gui.setScreen(screen);
    }

    /** Whether the HUD is hidden (F1). Since 26.2 that state lives on {@code Minecraft.gui.hud}. */
    public static boolean hudHidden(Minecraft client) {
        return client.gui.hud.isHidden();
    }

    /** Draws {@code painter} on the HUD every frame, through Fabric API's {@code HudElementRegistry}, drawn after every vanilla HUD element. */
    public static void registerHud(Consumer<HudCanvas> painter) {
        HudElementRegistry.addLast(Identifier.fromNamespaceAndPath("pymcbot", "recorder_badge"),
                (graphics, deltaTracker) -> painter.accept(canvas(graphics)));
    }

    /** Wraps this era's GUI drawing class ({@code GuiGraphicsExtractor}) for the shared drawing code. */
    private static HudCanvas canvas(GuiGraphicsExtractor graphics) {
        Font font = Minecraft.getInstance().font;
        return new HudCanvas() {
            @Override
            public int width() {
                return graphics.guiWidth();
            }

            @Override
            public int height() {
                return graphics.guiHeight();
            }

            @Override
            public void fill(int x1, int y1, int x2, int y2, int argb) {
                graphics.fill(x1, y1, x2, y2, argb);
            }

            @Override
            public void text(String text, int x, int y, int argb) {
                graphics.text(font, text, x, y, argb);
            }

            @Override
            public int textWidth(String text) {
                return font.width(text);
            }

            @Override
            public int lineHeight() {
                return font.lineHeight;
            }
        };
    }

    /**
     * A screen that draws extra text through {@link HudCanvas} on top of its widgets. In this era
     * screens draw in {@code extractRenderState(GuiGraphicsExtractor, int, int, float)}.
     */
    public abstract static class PaintedScreen extends Screen {
        protected PaintedScreen(Component title) {
            super(title);
        }

        @Override
        public void extractRenderState(GuiGraphicsExtractor graphics, int mouseX, int mouseY, float partialTick) {
            super.extractRenderState(graphics, mouseX, mouseY, partialTick);
            paint(canvas(graphics), mouseX, mouseY);
        }

        /** Draws the screen's text; called every frame after the widgets, with the mouse position (for hover help). */
        protected abstract void paint(HudCanvas canvas, int mouseX, int mouseY);
    }
}
