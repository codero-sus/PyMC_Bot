package dev.pymc.pymcb.client.compat;

import java.util.function.Consumer;

import com.mojang.blaze3d.platform.InputConstants;
import com.mojang.brigadier.arguments.ArgumentType;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import com.mojang.brigadier.builder.RequiredArgumentBuilder;
import dev.pymc.pymcb.client.HudCanvas;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandManager;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.hud.HudElementRegistry;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.Font;
import net.minecraft.client.gui.GuiGraphics;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;

/**
 * Version bridge for Minecraft 1.21.11.
 *
 * <p>Everything in the recorder that differs between Minecraft versions goes through this
 * class; the rest of the mod is identical for every target. There is one copy of this file
 * per API era under {@code mod/src/compat/<era>/}, and the build picks the right one from
 * {@code mod/versions.json} ({@code ./gradlew build -Pmc=<version>}).
 *
 * <p>What changed in this era: Mojang renamed {@code ResourceLocation} to {@code Identifier} and {@code ResourceKey.location()} to {@code identifier()}.
 */
public final class Mc {
    /** The Minecraft versions this copy is compiled for (shown by {@code /pymc version}). */
    public static final String ERA = "1.21.11";

    /** Key mapping category (a registered object since 1.21.9). */
    private static final KeyMapping.Category CATEGORY =
            KeyMapping.Category.register(Identifier.fromNamespaceAndPath("pymcplaytime", "recorder"));

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
        return ClientCommandManager.literal(name);
    }

    /** A client command argument node. */
    public static <T> RequiredArgumentBuilder<FabricClientCommandSource, T> argument(String name, ArgumentType<T> type) {
        return ClientCommandManager.argument(name, type);
    }

    /**
     * Whether the player really ran the command themselves.
     *
     * <p>Fabric API only tells attended and server-triggered runs apart from 26.2 on, so on this version every run counts as attended.
     */
    public static boolean attended(FabricClientCommandSource source) {
        return true;
    }

    /**
     * Registers a key mapping under the "PyMC Playtime" category, unbound by default so it never
     * clashes with another key (bind it in Options - Controls - Key Binds).
     */
    public static KeyMapping registerKey(String translationKey) {
        return KeyBindingHelper.registerKeyBinding(new KeyMapping(translationKey, InputConstants.UNKNOWN.getValue(), CATEGORY));
    }

    /** Whether a screen (inventory, chat, pause menu...) is open. */
    public static boolean screenOpen(Minecraft client) {
        return client.screen != null;
    }

    /** Shows a line in the chat for the local player only. */
    public static void message(LocalPlayer player, Component text) {
        player.displayClientMessage(text, false);
    }

    // ------------------------------------------------------------------ GUI

    /** Opens {@code screen}, or returns to the game when it is null. Screens are managed by {@code Minecraft} in this era. */
    public static void openScreen(Minecraft client, Screen screen) {
        client.setScreen(screen);
    }

    /** Whether the HUD is hidden (F1). */
    public static boolean hudHidden(Minecraft client) {
        return client.options.hideGui;
    }

    /** Draws {@code painter} on the HUD every frame, through Fabric API's {@code HudElementRegistry}, drawn after every vanilla HUD element. */
    public static void registerHud(Consumer<HudCanvas> painter) {
        HudElementRegistry.addLast(Identifier.fromNamespaceAndPath("pymcplaytime", "recorder_badge"),
                (graphics, deltaTracker) -> painter.accept(canvas(graphics)));
    }

    /** Wraps this era's GUI drawing class ({@code GuiGraphics}) for the shared drawing code. */
    private static HudCanvas canvas(GuiGraphics graphics) {
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
                graphics.drawString(font, text, x, y, argb);
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
     * screens draw in {@code render(GuiGraphics, int, int, float)}.
     */
    public abstract static class PaintedScreen extends Screen {
        protected PaintedScreen(Component title) {
            super(title);
        }

        @Override
        public void render(GuiGraphics graphics, int mouseX, int mouseY, float partialTick) {
            super.render(graphics, mouseX, mouseY, partialTick);
            paint(canvas(graphics));
        }

        /** Draws the screen's text; called every frame after the widgets. */
        protected abstract void paint(HudCanvas canvas);
    }
}
