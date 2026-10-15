package dev.pymc.pymcb.client;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import dev.pymc.pymcb.PlaytimeConfig;
import dev.pymc.pymcb.PymcPlaytimeMod;
import dev.pymc.pymcb.client.compat.Mc;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.network.chat.Component;

/**
 * In-game settings for the recorder: start / stop / pause, the HUD badge, and every option that
 * {@code /pymc} can change. Open it with {@code /pymc settings} or the (unbound by default)
 * "Open settings" hotkey.
 *
 * <p>Only vanilla buttons are used and the text is drawn through {@link HudCanvas}, so this one
 * class works on every supported Minecraft version; the per-version drawing hook lives in
 * {@code Mc.PaintedScreen}.
 */
public class PymcSettingsScreen extends Mc.PaintedScreen {
    /** Steps the "Skip AFK after" button cycles through, in seconds (0 = never skip). */
    static final int[] IDLE_STEPS = {0, 5, 15, 30, 60, 120, 300};
    /** Steps the "Sample rate" button cycles through, in client ticks between samples. */
    static final int[] SAMPLE_STEPS = {1, 2, 4, 10, 20};

    private static final int COLUMN = 150;
    private static final int ROW = 24;

    /** Screen to return to when closed (null = back to the game). */
    private final Screen parent;
    private final List<Runnable> labels = new ArrayList<>();
    private String lastMessage = "";
    private int top;

    public PymcSettingsScreen(Screen parent) {
        super(Component.literal("PyMC Playtime Recorder"));
        this.parent = parent;
    }

    private static PlaytimeRecorder recorder() {
        return PymcPlaytimeClient.recorder();
    }

    private static PlaytimeConfig cfg() {
        return PlaytimeConfig.get();
    }

    private static String onOff(boolean value) {
        return value ? "ON" : "OFF";
    }

    /** The value after {@code current} in {@code steps}, wrapping around. */
    static int next(int[] steps, int current) {
        for (int step : steps) {
            if (step > current) {
                return step;
            }
        }
        return steps[0];
    }

    @Override
    protected void init() {
        labels.clear();
        int left = this.width / 2 - COLUMN - 5;
        int right = this.width / 2 + 5;
        top = Math.max(44, this.height / 2 - 78);

        Button recordButton = button(left, top, () -> act(recorder().toggle()));
        Button pauseButton = button(right, top, () -> act(recorder().togglePause()));
        label(() -> {
            PlaytimeRecorder.HudState state = recorder().hudState();
            recordButton.setMessage(Component.literal(state.active() ? "Stop recording" : "Start recording"));
            pauseButton.setMessage(Component.literal(state.paused() ? "Resume" : "Pause"));
            pauseButton.active = state.active();
        });

        Button hud = button(left, top + ROW, () -> cfg().setShowHud(!cfg().showHud));
        Button corner = button(right, top + ROW, cfg()::cycleHudCorner);
        label(() -> {
            hud.setMessage(Component.literal("HUD badge: " + onOff(cfg().showHud)));
            corner.setMessage(Component.literal("Badge corner: " + PlaytimeConfig.cornerLabel(cfg().hudCorner)));
            corner.active = cfg().showHud;
        });

        Button menus = button(left, top + ROW * 2, () -> cfg().setPauseInMenus(!cfg().pauseInMenus));
        Button idle = button(right, top + ROW * 2, () -> cfg().setIdleSkipSeconds(next(IDLE_STEPS, cfg().idleSkipSeconds)));
        label(() -> {
            menus.setMessage(Component.literal("Pause in menus: " + onOff(cfg().pauseInMenus)));
            idle.setMessage(Component.literal("Skip AFK after: "
                    + (cfg().idleSkipSeconds == 0 ? "never" : cfg().idleSkipSeconds + "s")));
        });

        Button advanced = button(left, top + ROW * 3, () -> cfg().setAdvanced(!cfg().advanced));
        Button sample = button(right, top + ROW * 3, () -> cfg().setSampleTicks(next(SAMPLE_STEPS, cfg().sampleTicks)));
        label(() -> {
            advanced.setMessage(Component.literal("Entities + items: " + onOff(cfg().advanced)));
            double perSecond = 20.0 / Math.max(1, cfg().sampleTicks);
            sample.setMessage(Component.literal(String.format(Locale.ROOT, "Sample rate: %s/s",
                    perSecond == Math.rint(perSecond) ? Integer.toString((int) perSecond) : String.format(Locale.ROOT, "%.1f", perSecond))));
        });

        Button auto = button(left, top + ROW * 4, () -> cfg().setAutoRecord(!cfg().autoRecord));
        Button notify = button(right, top + ROW * 4, () -> cfg().setNotifications(!cfg().notifications));
        label(() -> {
            auto.setMessage(Component.literal("Auto-record on join: " + onOff(cfg().autoRecord)));
            notify.setMessage(Component.literal("Chat notices: " + onOff(cfg().notifications)));
        });

        Button export = button(left, top + ROW * 5, () -> act(recorder().export()));
        export.setMessage(Component.literal("Export dataset"));
        Button done = button(right, top + ROW * 5, this::onClose);
        done.setMessage(Component.translatable("gui.done"));

        refresh();
    }

    /** Adds a 150x20 button; every click re-labels all buttons afterwards. */
    private Button button(int x, int y, Runnable action) {
        Button button = Button.builder(Component.empty(), pressed -> {
            action.run();
            refresh();
        }).bounds(x, y, COLUMN, 20).build();
        this.addRenderableWidget(button);
        return button;
    }

    private void label(Runnable updater) {
        labels.add(updater);
    }

    private void refresh() {
        for (Runnable updater : labels) {
            updater.run();
        }
    }

    /** Runs a recorder action and keeps its first line on screen (and in chat). */
    private void act(String result) {
        String[] lines = result.split("\n");
        lastMessage = lines.length == 0 ? "" : lines[0];
        Minecraft mc = Minecraft.getInstance();
        if (mc.player != null) {
            PlaytimeRecorder.tell(mc.player, result);
        }
    }

    @Override
    protected void paint(HudCanvas canvas) {
        int center = this.width / 2;
        canvas.centeredText("PyMC Playtime Recorder", center, top - 34, RecorderHud.WHITE);

        PlaytimeRecorder.HudState state = recorder().hudState();
        String status = state.active()
                ? state.label() + "  " + state.clock() + "  " + RecorderHud.samples(state.samples()) + "  " + state.runName()
                : "Not recording - press Start, or bind the hotkey in Controls";
        int color = state.active() ? RecorderHud.accent(state.label()) : RecorderHud.GREY;
        canvas.centeredText(canvas.fit(status, this.width - 20), center, top - 20, color);

        if (!lastMessage.isEmpty()) {
            canvas.centeredText(canvas.fit(lastMessage, this.width - 20), center, top + ROW * 6 + 2, RecorderHud.GREY);
        }
        String footer = "v" + PymcPlaytimeMod.modVersion() + " for Minecraft " + PymcPlaytimeMod.minecraftVersion()
                + "  -  Made with \u2764 by codero-sus";
        canvas.centeredText(canvas.fit(footer, this.width - 10), center, this.height - 12, 0xFF7F8590);
    }

    @Override
    public void onClose() {
        Mc.openScreen(Minecraft.getInstance(), parent);
    }
}
