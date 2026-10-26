package com.terraformersmc.modmenu.api;

import net.minecraft.client.gui.screens.Screen;

/**
 * COMPILE-ONLY STUB of Mod Menu's {@code ConfigScreenFactory} (same signature as the real one, which
 * has been unchanged from Minecraft 1.21 to 26.3). Never shipped - see {@link ModMenuApi}.
 */
@FunctionalInterface
public interface ConfigScreenFactory<S extends Screen> {
    S create(Screen parent);
}
