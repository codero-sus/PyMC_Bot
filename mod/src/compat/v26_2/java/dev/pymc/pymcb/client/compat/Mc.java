package dev.pymc.pymcb.client.compat;

import com.mojang.blaze3d.platform.InputConstants;
import com.mojang.brigadier.arguments.ArgumentType;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import com.mojang.brigadier.builder.RequiredArgumentBuilder;
import net.fabricmc.fabric.api.client.command.v2.ClientCommands;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.fabricmc.fabric.api.client.keymapping.v1.KeyMappingHelper;
import net.minecraft.client.KeyMapping;
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
 * <p>What changed in this era: Fabric API 0.152 added attended client commands. 26.3 replaced GLFW with SDL3; the hotkey is registered unbound through {@code InputConstants.UNKNOWN}, so no raw key codes are involved.
 */
public final class Mc {
    /** The Minecraft versions this copy is compiled for (shown by {@code /pymc version}). */
    public static final String ERA = "26.2 - 26.3";

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
     * Registers a key mapping under the "PyMC Playtime" category, unbound by default so it never
     * clashes with another key (bind it in Options - Controls - Key Binds).
     */
    public static KeyMapping registerKey(String translationKey) {
        return KeyMappingHelper.registerKeyMapping(new KeyMapping(translationKey, InputConstants.UNKNOWN.getValue(), CATEGORY));
    }

    /** Shows a line in the chat for the local player only. */
    public static void message(LocalPlayer player, Component text) {
        player.sendSystemMessage(text);
    }
}
