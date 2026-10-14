package dev.pymc.pymcb.client.compat;

import com.mojang.blaze3d.platform.InputConstants;
import com.mojang.brigadier.arguments.ArgumentType;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import com.mojang.brigadier.builder.RequiredArgumentBuilder;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandManager;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.network.chat.Component;

/**
 * Version bridge for Minecraft 1.21.5 - 1.21.8.
 *
 * <p>Everything in the recorder that differs between Minecraft versions goes through this
 * class; the rest of the mod is identical for every target. There is one copy of this file
 * per API era under {@code mod/src/compat/<era>/}, and the build picks the right one from
 * {@code mod/versions.json} ({@code ./gradlew build -Pmc=<version>}).
 *
 * <p>What changed in this era: {@code Inventory.selected} became private behind {@code getSelectedSlot()}.
 */
public final class Mc {
    /** The Minecraft versions this copy is compiled for (shown by {@code /pymc version}). */
    public static final String ERA = "1.21.5 - 1.21.8";

    /** Key mapping category (a translation key before 1.21.9). */
    private static final String CATEGORY = "key.categories.pymcplaytime";

    private Mc() {
    }

    /** The hotbar slot (0-8) the player has selected. */
    public static int selectedSlot(LocalPlayer player) {
        return player.getInventory().getSelectedSlot();
    }

    /** The dimension id, e.g. {@code minecraft:overworld}. */
    public static String dimensionId(LocalPlayer player) {
        return player.level().dimension().location().toString();
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

    /** Shows a line in the chat for the local player only. */
    public static void message(LocalPlayer player, Component text) {
        player.displayClientMessage(text, false);
    }
}
