package dev.pymc.pymcb;

import net.fabricmc.api.ModInitializer;
import net.minecraft.resources.Identifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Mod entrypoint. This mod is client only (it records the local player's own playtime), so the
 * real work happens in {@code dev.pymc.pymcb.client.PymcPlaytimeClient}; this class exists to own
 * the mod id and logger the same way the official Fabric template does.
 */
public class PymcPlaytimeMod implements ModInitializer {
    public static final String MOD_ID = "pymcplaytime";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    @Override
    public void onInitialize() {
        LOGGER.info("PyMC Playtime Recorder loaded (client entrypoint does the recording)");
    }

    public static Identifier id(String path) {
        return Identifier.fromNamespaceAndPath(MOD_ID, path);
    }
}
