package dev.pymc.pymcb.client;

import java.util.function.DoubleConsumer;
import java.util.function.DoubleFunction;
import java.util.function.DoubleSupplier;

import net.minecraft.client.gui.components.AbstractSliderButton;
import net.minecraft.network.chat.Component;

/**
 * A vanilla slider bound to one number in {@link dev.pymc.pymcb.PlaytimeConfig}, used by the
 * "Power" tab of {@link PymcBotScreen}.
 *
 * <p>The value snaps to {@code step}, and the config setter (which saves the file) only runs when
 * the snapped value actually changes - dragging does not rewrite {@code pymcbot.properties} on
 * every mouse event. {@code min} may be larger than {@code max} to flip the direction (the sample
 * rate slider goes from 1 sample/s on the left to 20/s on the right while it stores ticks).
 *
 * <p>This is the only class that touches {@code AbstractSliderButton}; it uses just the
 * constructor, {@code value}, {@code updateMessage()} and {@code applyValue()}, which every
 * supported Minecraft version shares. If an era ever changes them, only this file needs a shim.
 */
final class PowerSlider extends AbstractSliderButton {
    private final String name;
    private final double min;
    private final double max;
    private final double step;
    private final DoubleSupplier getter;
    private final DoubleConsumer setter;
    private final DoubleFunction<String> format;
    /** The last value written to (or read from) the config. */
    private double stored;

    PowerSlider(int x, int y, int width, String name, double min, double max, double step,
                DoubleSupplier getter, DoubleConsumer setter, DoubleFunction<String> format) {
        super(x, y, width, 20, Component.literal(name), 0.0);
        this.name = name;
        this.min = min;
        this.max = max;
        this.step = step;
        this.getter = getter;
        this.setter = setter;
        this.format = format;
        sync();
    }

    /** {@code raw} rounded to the nearest step counted from {@code min}, kept inside the range. */
    static double snap(double raw, double min, double max, double step) {
        double snapped = step > 0 ? min + Math.round((raw - min) / step) * step : raw;
        double low = Math.min(min, max);
        double high = Math.max(min, max);
        snapped = Math.max(low, Math.min(high, snapped));
        // Kill float noise such as 0.30000000000000004 so labels and the saved file stay clean.
        return Math.round(snapped * 1e6) / 1e6;
    }

    /** Slider position (0..1) for a config value. */
    static double position(double current, double min, double max) {
        double span = max - min;
        if (span == 0) {
            return 0.0;
        }
        return Math.max(0.0, Math.min(1.0, (current - min) / span));
    }

    /** The snapped value under the knob. */
    double current() {
        return snap(min + value * (max - min), min, max, step);
    }

    /** Moves the knob to the config's value (after a reset, or when another tab changed it). */
    void sync() {
        stored = snap(getter.getAsDouble(), min, max, step);
        value = position(stored, min, max);
        updateMessage();
    }

    @Override
    protected void updateMessage() {
        if (format == null) {
            return; // called from the super constructor on some versions, before the fields are set
        }
        setMessage(Component.literal(name + ": " + format.apply(current())));
    }

    @Override
    protected void applyValue() {
        double next = current();
        if (Math.abs(next - stored) > 1e-9) {
            stored = next;
            setter.accept(next);
        }
    }
}
