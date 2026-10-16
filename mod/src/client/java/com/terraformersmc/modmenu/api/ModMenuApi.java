package com.terraformersmc.modmenu.api;

/**
 * COMPILE-ONLY STUB of Mod Menu's public API (https://github.com/TerraformersMC/ModMenu, MIT).
 *
 * <p>Only the method PyMC Bot uses is declared. build.gradle excludes {@code com/terraformersmc/**}
 * from every jar, so at runtime the real interface from the installed Mod Menu is used - and
 * without Mod Menu the {@code modmenu} entrypoint is simply never loaded.
 */
public interface ModMenuApi {
    default ConfigScreenFactory<?> getModConfigScreenFactory() {
        return null;
    }
}
