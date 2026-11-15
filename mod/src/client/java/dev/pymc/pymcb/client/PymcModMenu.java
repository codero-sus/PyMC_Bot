package dev.pymc.pymcb.client;

import com.terraformersmc.modmenu.api.ConfigScreenFactory;
import com.terraformersmc.modmenu.api.ModMenuApi;

/**
 * Mod Menu integration: the "Configure" button next to PyMC Bot in the mods list opens the
 * PyMC Bot settings page. Registered as the {@code modmenu} entrypoint in fabric.mod.json; Fabric
 * only loads this class when Mod Menu asks for it, so Mod Menu stays optional.
 */
public final class PymcModMenu implements ModMenuApi {
    @Override
    public ConfigScreenFactory<?> getModConfigScreenFactory() {
        return parent -> new PymcBotScreen(parent);
    }
}
