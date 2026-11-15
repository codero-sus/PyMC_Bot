package dev.pymc.pymcb;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.loader.api.FabricLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Mod entrypoint of the PyMC Bot mod. This mod is client only (it records the local player's own playtime), so the
 * real work happens in {@code dev.pymc.pymcb.client.PymcPlaytimeClient}; this class exists to own
 * the mod id and logger the same way the official Fabric template does.
 *
 * <p>Only the Fabric Loader API is used here (never Minecraft classes), so this file is the same
 * for every supported Minecraft version.
 */
public class PymcPlaytimeMod implements ModInitializer {
    public static final String MOD_ID = "pymcbot";
    /** The name players see: mod list, settings page, chat prefix. */
    public static final String NAME = "PyMC Bot";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    @Override
    public void onInitialize() {
        LOGGER.info(NAME + " mod loaded (client entrypoint does the recording) - made by codero-sus, https://github.com/codero-sus, https://codero.is-a.dev, https://the-codero.web.app/");
    }

    /** This mod's version, e.g. {@code 0.4.0+26.3}. */
    public static String modVersion() {
        return versionOf(MOD_ID);
    }

    /** The running Minecraft version, e.g. {@code 1.21.4} or {@code 26.3}. */
    public static String minecraftVersion() {
        return versionOf("minecraft");
    }

    private static String versionOf(String modId) {
        return FabricLoader.getInstance().getModContainer(modId)
                .map(container -> container.getMetadata().getVersion().getFriendlyString())
                .orElse("unknown");
    }
}
