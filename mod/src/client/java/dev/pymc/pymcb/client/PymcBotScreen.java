package dev.pymc.pymcb.client;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.BooleanSupplier;
import java.util.function.DoubleConsumer;
import java.util.function.DoubleFunction;
import java.util.function.DoubleSupplier;
import java.util.function.Supplier;

import dev.pymc.pymcb.PlaytimeConfig;
import dev.pymc.pymcb.PymcPlaytimeMod;
import dev.pymc.pymcb.client.compat.Mc;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.network.chat.Component;

/**
 * The PyMC Bot config and settings page.
 *
 * <p>Six tabs cover everything in {@code config/pymcbot.properties} plus the recorder controls:
 * <ul>
 *   <li><b>Recorder</b> - start / stop, pause, new episode, export, status and the training command;</li>
 *   <li><b>Recording</b> - sample rate, menus, AFK skipping, auto-record, blocks, nearby entities, thresholds;</li>
 *   <li><b>Advanced</b> - entity + item recording and its limits;</li>
 *   <li><b>Power</b> - sliders for fine control over every number (sample rate, thresholds, limits)
 *       plus the export filters that drop short episodes and thin out standing-still samples;</li>
 *   <li><b>HUD &amp; chat</b> - the REC badge (with a live preview), its corner and chat notices;</li>
 *   <li><b>About</b> - versions, file locations and credits.</li>
 * </ul>
 * Every change is saved immediately. Hovering a button explains it; each settings tab has a
 * "Reset page" button (click twice).
 *
 * <p>Open it with {@code /pymcbot settings}, the "Open PyMC Bot settings" hotkey, or Mod Menu's
 * Configure button ({@link PymcModMenu}). Only vanilla buttons and vanilla sliders (wrapped in
 * {@link PowerSlider}) are used and all text goes through {@link HudCanvas}, so this one class works
 * on every supported Minecraft version.
 */
public class PymcBotScreen extends Mc.PaintedScreen {
    /** The tabs, in display order. */
    public enum Page {
        RECORDER("Recorder"),
        RECORDING("Recording"),
        ADVANCED("Advanced"),
        POWER("Power"),
        HUD("HUD & chat"),
        ABOUT("About");

        final String title;

        Page(String title) {
            this.title = title;
        }

        /** Whether the page has settings that "Reset page" can restore. */
        boolean resettable() {
            return this == RECORDING || this == ADVANCED || this == POWER || this == HUD;
        }
    }

    /** "Skip AFK after" steps in seconds (0 = never skip). */
    static final int[] IDLE_STEPS = {0, 5, 15, 30, 60, 120, 300};
    /** "Sample rate" steps in client ticks between samples. */
    static final int[] SAMPLE_STEPS = {1, 2, 4, 10, 20};
    /** "Nearby entities" steps (basic mode). */
    static final int[] NEARBY_STEPS = {0, 4, 8, 12, 16, 24, 32, 64};
    /** "Move threshold" steps in blocks. */
    static final double[] MOVE_STEPS = {0.0, 0.01, 0.02, 0.05, 0.1, 0.25};
    /** "Turn threshold" steps in degrees. */
    static final double[] TURN_STEPS = {0.5, 1.0, 2.0, 5.0, 10.0};
    /** "Max entities" steps (advanced mode). */
    static final int[] ENTITY_STEPS = {0, 8, 16, 24, 32, 48, 64, 128};
    /** "Screenshot every" steps in seconds (Minecraftism VL screenshots). */
    static final int[] SHOT_STEPS = {2, 5, 10, 20, 30, 60};
    /** "Screenshot width" steps in pixels. */
    static final int[] SHOT_WIDTHS = {320, 480, 640, 854, 1280};
    /** "Entity range" steps in blocks (advanced mode). */
    static final double[] RANGE_STEPS = {8, 16, 24, 32, 48, 64, 96, 128};
    /** "Ground items" steps (advanced mode). */
    static final int[] DROP_STEPS = {0, 4, 8, 16, 32, 64};

    static final String TITLE = PymcPlaytimeMod.NAME + " - Settings";
    static final String CHAT_PREFIX = "[" + PymcPlaytimeMod.NAME + "] ";

    private static final int COLUMN = 150;
    private static final int ROW = 24;
    private static final int TAB_Y = 24;
    private static final int TAB_GAP = 4;
    private static final int CONTENT_Y = 64;
    private static final int DIM = 0xFF7F8590;

    /** The tab shown when the page is opened again (remembered until the game closes). */
    private static Page lastPage = Page.RECORDER;

    /** A button whose label / enabled state follows the config, plus its hover help. */
    private record Option(Button button, int x, int y, Supplier<String> label, BooleanSupplier enabled, String hint) {
        boolean hovered(int mouseX, int mouseY) {
            return mouseX >= x && mouseX < x + COLUMN && mouseY >= y && mouseY < y + 20;
        }
    }

    /** A Power-tab slider, its enabled state and its hover help. */
    private record SliderOption(PowerSlider slider, int x, int y, BooleanSupplier enabled, String hint) {
        boolean hovered(int mouseX, int mouseY) {
            return mouseX >= x && mouseX < x + COLUMN && mouseY >= y && mouseY < y + 20;
        }
    }

    /** Screen to return to when closed (null = back to the game). */
    private final Screen parent;
    private final Page page;
    private final List<Option> options = new ArrayList<>();
    private final List<SliderOption> sliders = new ArrayList<>();
    private String lastMessage;
    private boolean confirmReset;
    private Button resetButton;

    public PymcBotScreen(Screen parent) {
        this(parent, lastPage);
    }

    public PymcBotScreen(Screen parent, Page page) {
        this(parent, page, "");
    }

    private PymcBotScreen(Screen parent, Page page, String message) {
        super(Component.literal(TITLE));
        this.parent = parent;
        this.page = page;
        this.lastMessage = message;
        lastPage = page;
    }

    /** The tab this screen shows. */
    public Page page() {
        return page;
    }

    private static PlaytimeRecorder recorder() {
        return PymcPlaytimeClient.recorder();
    }

    private static PlaytimeConfig cfg() {
        return PlaytimeConfig.get();
    }

    // ------------------------------------------------------------------ cycling helpers

    /** The value after {@code current} in {@code steps}, wrapping around. */
    static int next(int[] steps, int current) {
        for (int step : steps) {
            if (step > current) {
                return step;
            }
        }
        return steps[0];
    }

    /** The value after {@code current} in {@code steps}, wrapping around. */
    static double next(double[] steps, double current) {
        for (double step : steps) {
            if (step > current + 1e-9) {
                return step;
            }
        }
        return steps[0];
    }

    static String onOff(boolean value) {
        return value ? "ON" : "OFF";
    }

    /** "20/s", "2.5/s"... */
    static String perSecond(int sampleTicks) {
        double perSecond = 20.0 / Math.max(1, sampleTicks);
        return (perSecond == Math.rint(perSecond) ? Integer.toString((int) perSecond)
                : String.format(Locale.ROOT, "%.1f", perSecond)) + "/s";
    }

    /** "never" / "45s" / "2m 30s". */
    static String seconds(int value) {
        if (value <= 0) {
            return "never";
        }
        return value < 60 ? value + "s" : value / 60 + "m" + (value % 60 == 0 ? "" : " " + value % 60 + "s");
    }

    static String number(double value) {
        return value == Math.rint(value) ? Integer.toString((int) value) : String.format(Locale.ROOT, "%.2f", value)
                .replaceAll("0+$", "");
    }

    // ------------------------------------------------------------------ layout

    @Override
    protected void init() {
        options.clear();
        sliders.clear();
        int center = this.width / 2;

        // Tab row: the open tab is shown as a disabled (pressed) button.
        Page[] pages = Page.values();
        int tabWidth = Math.min(76, (this.width - 20 - TAB_GAP * (pages.length - 1)) / pages.length);
        int tabsLeft = center - (tabWidth * pages.length + TAB_GAP * (pages.length - 1)) / 2;
        for (int i = 0; i < pages.length; i++) {
            Page target = pages[i];
            Button tab = Button.builder(Component.literal(target.title), pressed -> showPage(target))
                    .bounds(tabsLeft + i * (tabWidth + TAB_GAP), TAB_Y, tabWidth, 20).build();
            tab.active = target != page;
            this.addRenderableWidget(tab);
        }

        switch (page) {
            case RECORDER -> initRecorder();
            case RECORDING -> initRecording();
            case ADVANCED -> initAdvanced();
            case POWER -> initPower();
            case HUD -> initHud();
            case ABOUT -> initAbout();
        }

        // Bottom row: Reset page (settings tabs only) + Done.
        int bottom = this.height - 32;
        if (page.resettable()) {
            resetButton = Button.builder(Component.literal("Reset page"), pressed -> reset())
                    .bounds(center - COLUMN - 5, bottom, COLUMN, 20).build();
            this.addRenderableWidget(resetButton);
            this.addRenderableWidget(Button.builder(Component.translatable("gui.done"), pressed -> onClose())
                    .bounds(center + 5, bottom, COLUMN, 20).build());
        } else {
            resetButton = null;
            this.addRenderableWidget(Button.builder(Component.translatable("gui.done"), pressed -> onClose())
                    .bounds(center - COLUMN / 2, bottom, COLUMN, 20).build());
        }
        refresh();
    }

    private int columnX(int column) {
        return column == 0 ? this.width / 2 - COLUMN - 5 : this.width / 2 + 5;
    }

    private static int rowY(int row) {
        return CONTENT_Y + row * ROW;
    }

    /** Adds a config button at (column, row); its label is recomputed after every click. */
    private void option(int column, int row, Supplier<String> label, BooleanSupplier enabled, String hint, Runnable action) {
        int x = columnX(column);
        int y = rowY(row);
        Button button = Button.builder(Component.literal(label.get()), pressed -> {
            confirmReset = false;
            action.run();
            refresh();
        }).bounds(x, y, COLUMN, 20).build();
        this.addRenderableWidget(button);
        options.add(new Option(button, x, y, label, enabled, hint));
    }

    private void option(int column, int row, Supplier<String> label, String hint, Runnable action) {
        option(column, row, label, () -> true, hint, action);
    }

    private void initRecorder() {
        option(0, 0, () -> recorder().isActive() ? "Stop recording" : "Start recording",
                "Starts a new recording run, or stops and saves the current one.",
                () -> act(recorder().toggle()));
        option(1, 0, () -> recorder().isPaused() ? "Resume" : "Pause", () -> recorder().isActive(),
                "Stops sampling for a moment without ending the run.",
                () -> act(recorder().togglePause()));
        option(0, 1, () -> "New episode", () -> recorder().isActive(),
                "Closes the current episode and starts the next one in the same run.",
                () -> act(recorder().endEpisode()));
        option(1, 1, () -> "Export dataset",
                "Writes dataset.jsonl for the PyMC Bot trainer (python -m pymc_bot train).",
                () -> act(recorder().export()));
        option(0, 2, () -> "Status in chat",
                "Prints where the dataset lives, how big it is and the current settings.",
                () -> sayAll(recorder().status()));
        option(1, 2, () -> "Training command",
                "Prints the exact command that trains a PyMC Bot model on this dataset.",
                () -> sayAll(PymcCommands.trainHint(recorder())));
    }

    private void initRecording() {
        option(0, 0, () -> "Sample rate: " + perSecond(cfg().sampleTicks),
                "How often a sample is written. 10/s (every 2 ticks) is a good default.",
                () -> cfg().setSampleTicks(next(SAMPLE_STEPS, cfg().sampleTicks)));
        option(1, 0, () -> "Pause in menus: " + onOff(cfg().pauseInMenus),
                "Skips samples while the inventory, chat or another screen is open.",
                () -> cfg().setPauseInMenus(!cfg().pauseInMenus));
        option(0, 1, () -> "Skip AFK after: " + (cfg().idleSkipSeconds == 0 ? "never" : cfg().idleSkipSeconds + "s"),
                "Stops writing after this long without input, so AFK time does not flood the dataset.",
                () -> cfg().setIdleSkipSeconds(next(IDLE_STEPS, cfg().idleSkipSeconds)));
        option(1, 1, () -> "Auto-record on join: " + onOff(cfg().autoRecord),
                "Starts recording automatically when you join a world or server.",
                () -> cfg().setAutoRecord(!cfg().autoRecord));
        option(0, 2, () -> "Block area: " + onOff(cfg().includeBlocks),
                "Records the 3x3x3 blocks around you (about 600 bytes per sample).",
                () -> cfg().setIncludeBlocks(!cfg().includeBlocks));
        option(1, 2, () -> "Nearby entities: " + cfg().maxNearby,
                "How many nearby mobs / players are written per sample.",
                () -> cfg().setMaxNearby(next(NEARBY_STEPS, cfg().maxNearby)));
        option(0, 3, () -> "Move threshold: " + number(cfg().moveEpsilon) + " blocks",
                "Smallest movement that counts as \"move\" when no key explains it.",
                () -> cfg().setMoveEpsilon(next(MOVE_STEPS, cfg().moveEpsilon)));
        option(1, 3, () -> "Turn threshold: " + number(cfg().turnDegrees) + "\u00b0",
                "Smallest camera turn that labels a sample \"look\".",
                () -> cfg().setTurnDegrees(next(TURN_STEPS, cfg().turnDegrees)));
    }

    private void initAdvanced() {
        option(0, 0, () -> "Entities + items: " + onOff(cfg().advanced),
                "Also records mobs, held items, armour and drops (train with --advanced).",
                () -> cfg().setAdvanced(!cfg().advanced));
        option(1, 0, () -> "Max entities: " + cfg().maxEntities, () -> cfg().advanced,
                "How many entities are written per sample in advanced mode.",
                () -> cfg().setMaxEntities(next(ENTITY_STEPS, cfg().maxEntities)));
        option(0, 1, () -> "Entity range: " + number(cfg().entityRange) + " blocks", () -> cfg().advanced,
                "How far away an entity may be and still be written.",
                () -> cfg().setEntityRange(next(RANGE_STEPS, cfg().entityRange)));
        option(1, 1, () -> "Ground items: " + cfg().maxGroundItems, () -> cfg().advanced,
                "How many dropped item stacks are written per sample.",
                () -> cfg().setMaxGroundItems(next(DROP_STEPS, cfg().maxGroundItems)));
        option(0, 2, () -> "Screenshots: " + onOff(cfg().screenshots),
                "Saves a small screenshot while recording, for the Minecraftism VL (vision) model.",
                () -> cfg().setScreenshots(!cfg().screenshots));
        option(1, 2, () -> "Every " + cfg().screenshotSeconds + "s, " + cfg().screenshotWidth + " px", () -> cfg().screenshots,
                "Click: time between screenshots. Width: /pymcbot screenshots width <px>.",
                () -> cfg().setScreenshotSeconds(next(SHOT_STEPS, cfg().screenshotSeconds)));
    }

    /** Adds a config slider at (column, row). */
    private void slider(int column, int row, String name, double min, double max, double step,
                        DoubleSupplier getter, DoubleConsumer setter, DoubleFunction<String> format,
                        BooleanSupplier enabled, String hint) {
        int x = columnX(column);
        int y = rowY(row);
        PowerSlider slider = new PowerSlider(x, y, COLUMN, name, min, max, step, getter, value -> {
            confirmReset = false;
            setter.accept(value);
        }, format);
        this.addRenderableWidget(slider);
        sliders.add(new SliderOption(slider, x, y, enabled, hint));
    }

    /**
     * Power users: every number as a slider (finer than the Recording / Advanced buttons, which
     * cycle through a few presets) plus the two export filters. Drag, or click and use the arrow keys.
     */
    private void initPower() {
        BooleanSupplier always = () -> true;
        BooleanSupplier advanced = () -> cfg().advanced;
        // min > max: 1 sample/s on the left, 20/s on the right, while the config stores ticks.
        slider(0, 0, "Sample rate", 20, 1, 1, () -> cfg().sampleTicks,
                v -> cfg().setSampleTicks((int) Math.round(v)),
                v -> perSecond((int) Math.round(v)) + " (" + (int) Math.round(v) + "t)", always,
                "Samples per second. More = smoother motion targets but bigger files; 10/s suits most play.");
        slider(1, 0, "Turn threshold", 0.1, 20.0, 0.1, () -> cfg().turnDegrees,
                cfg()::setTurnDegrees, v -> number(v) + "\u00b0", always,
                "Camera turn that labels a sample \"look\". Lower = catches small aim corrections.");
        slider(0, 1, "Move threshold", 0.0, 0.5, 0.01, () -> cfg().moveEpsilon,
                cfg()::setMoveEpsilon, v -> number(v) + " blk", always,
                "Drift (knockback, ice, water) that counts as \"move\" when no key explains it.");
        slider(1, 1, "Skip AFK after", 0, 600, 5, () -> cfg().idleSkipSeconds,
                v -> cfg().setIdleSkipSeconds((int) Math.round(v)), v -> seconds((int) Math.round(v)), always,
                "Stops writing after this long without input. Shorter = less AFK noise in the dataset.");
        slider(0, 2, "Nearby entities", 0, 64, 1, () -> cfg().maxNearby,
                v -> cfg().setMaxNearby((int) Math.round(v)), v -> Integer.toString((int) Math.round(v)), always,
                "Closest mobs / players per sample. Basic training reads the nearest 6; more only grows files.");
        slider(1, 2, "Min episode", 0, 300, 5, () -> cfg().minEpisodeSeconds,
                v -> cfg().setMinEpisodeSeconds((int) Math.round(v)),
                v -> v < 0.5 ? "off" : seconds((int) Math.round(v)), always,
                "Export filter: leaves out accidental short recordings. Episode files stay on disk.");
        slider(0, 3, "Keep idle samples", 0, 100, 5, () -> cfg().idleKeepPercent,
                v -> cfg().setIdleKeepPercent((int) Math.round(v)), v -> (int) Math.round(v) + "%", always,
                "Export filter: share of standing-still samples kept. Less idle = the bot stands around less.");
        slider(1, 3, "Max entities", 0, 128, 1, () -> cfg().maxEntities,
                v -> cfg().setMaxEntities((int) Math.round(v)), v -> Integer.toString((int) Math.round(v)), advanced,
                "Advanced mode: entities written per sample (turn it on in the Advanced tab).");
        slider(0, 4, "Entity range", 4, 128, 1, () -> cfg().entityRange,
                cfg()::setEntityRange, v -> number(v) + " blk", advanced,
                "Advanced mode: how far away an entity may be and still be written.");
        slider(1, 4, "Ground items", 0, 64, 1, () -> cfg().maxGroundItems,
                v -> cfg().setMaxGroundItems((int) Math.round(v)), v -> Integer.toString((int) Math.round(v)), advanced,
                "Advanced mode: dropped item stacks written per sample.");
    }

    private void initHud() {
        option(0, 0, () -> "REC badge: " + onOff(cfg().showHud),
                "Shows REC / PAUSED / IDLE / MENU in a corner of the screen while recording.",
                () -> cfg().setShowHud(!cfg().showHud));
        option(1, 0, () -> "Badge corner: " + PlaytimeConfig.cornerLabel(cfg().hudCorner), () -> cfg().showHud,
                "Moves the badge to the next corner of the screen.",
                cfg()::cycleHudCorner);
        option(0, 1, () -> "Chat notices: " + onOff(cfg().notifications),
                "Chat lines when the hotkeys or auto-record change the recorder.",
                () -> cfg().setNotifications(!cfg().notifications));
        option(1, 1, () -> "Hotkeys...",
                "Bind the hotkeys in Options > Controls > Key Binds > PyMC Bot.",
                () -> sayAll("Hotkeys: Options > Controls > Key Binds > " + PymcPlaytimeMod.NAME
                        + " (start / stop, pause / resume, open settings - unbound by default)"));
    }

    private void initAbout() {
        option(0, 3, () -> "Links in chat",
                "Prints the author, website and source links in chat (clickable there).",
                () -> sayAll(PymcCommands.ABOUT));
        option(1, 3, () -> "Version in chat",
                "Prints the mod, Minecraft and compatibility layer versions.",
                () -> sayAll(PymcCommands.version()));
        option(0, 4, () -> "Bot panel: " + cfg().panelUrl.replaceFirst("^https?://", ""),
                "The PyMC Bot app that /pymcbot bot talks to. Change it: /pymcbot bot panel <url>.",
                () -> sayAll(PymcCommands.botHelp()));
        option(1, 4, () -> "Bot status in chat",
                "Asks the PyMC Bot app how your bot is doing (the app must be running).",
                () -> BotLink.get("/api/status", BotLink::describeStatus));
    }

    private void refresh() {
        for (Option option : options) {
            option.button().setMessage(Component.literal(option.label().get()));
            option.button().active = option.enabled().getAsBoolean();
        }
        for (SliderOption slider : sliders) {
            slider.slider().sync();
            slider.slider().active = slider.enabled().getAsBoolean();
        }
        if (resetButton != null) {
            resetButton.setMessage(Component.literal(confirmReset ? "Click again to reset" : "Reset page"));
        }
    }

    // ------------------------------------------------------------------ actions

    /** Switches tab by opening a fresh screen (works the same on every Minecraft version). */
    public void showPage(Page target) {
        if (target != page) {
            Mc.openScreen(Minecraft.getInstance(), new PymcBotScreen(parent, target, lastMessage));
        }
    }

    private void reset() {
        if (!confirmReset) {
            confirmReset = true;
            refresh();
            return;
        }
        confirmReset = false;
        switch (page) {
            case RECORDING -> cfg().resetRecording();
            case ADVANCED -> cfg().resetAdvanced();
            case POWER -> cfg().resetPower();
            case HUD -> cfg().resetHud();
            default -> {
            }
        }
        lastMessage = page.title + " settings were reset to their defaults";
        refresh();
    }

    /** Runs a recorder action: its first line stays on screen and goes to chat. */
    private void act(String result) {
        String[] lines = result.split("\n");
        lastMessage = lines.length == 0 ? "" : lines[0];
        Minecraft mc = Minecraft.getInstance();
        if (mc.player != null) {
            PlaytimeRecorder.tell(mc.player, result);
        }
    }

    /** Prints every line of {@code text} in chat (even with chat notices off - the player asked). */
    private void sayAll(String text) {
        Minecraft mc = Minecraft.getInstance();
        String[] lines = text.split("\n");
        lastMessage = mc.player == null ? lines[0] : "Sent to chat - close this page to read it";
        if (mc.player != null) {
            for (String line : lines) {
                Mc.message(mc.player, Component.literal(CHAT_PREFIX + line));
            }
        }
    }

    // ------------------------------------------------------------------ drawing

    @Override
    protected void paint(HudCanvas canvas, int mouseX, int mouseY) {
        int center = this.width / 2;
        canvas.centeredText(TITLE, center, 10, RecorderHud.WHITE);

        PlaytimeRecorder.HudState state = recorder().hudState();
        String status = state.active()
                ? state.label() + "  " + state.clock() + "  " + RecorderHud.samples(state.samples()) + "  " + state.runName()
                : "Not recording";
        int color = state.active() ? RecorderHud.accent(state.label()) : RecorderHud.GREY;
        canvas.centeredText(canvas.fit(status, this.width - 20), center, TAB_Y + 26, color);

        switch (page) {
            case RECORDER -> paintRecorder(canvas, center);
            case ADVANCED -> paintLines(canvas, center, rowY(3) + 4, cfg().advanced
                    ? new String[]{"Advanced recording is ON: train with  python -m pymc_bot train --advanced",
                            "Larger limits = bigger datasets; 24 entities within 32 blocks is the default."}
                    : new String[]{"Turn on \"Entities + items\" to record mobs, held items, armour and drops.",
                            "The PyMC Bot trainer learns them with --advanced."});
            case POWER -> paintPower(canvas, center);
            case HUD -> paintBadgePreview(canvas, center, state);
            case ABOUT -> paintAbout(canvas, center);
            default -> {
            }
        }

        // Hover help for the button under the mouse, otherwise the last result.
        String help = lastMessage;
        for (Option option : options) {
            if (option.hovered(mouseX, mouseY)) {
                help = option.hint();
            }
        }
        for (SliderOption slider : sliders) {
            if (slider.hovered(mouseX, mouseY)) {
                help = slider.hint();
            }
        }
        if (help != null && !help.isEmpty()) {
            canvas.centeredText(canvas.fit(help, this.width - 20), center, this.height - 46, RecorderHud.GREY);
        }

        String footer = PymcPlaytimeMod.NAME + " " + PymcPlaytimeMod.modVersion() + "  -  Made with \u2764 by codero-sus";
        canvas.centeredText(canvas.fit(footer, this.width - 10), center, this.height - 10, DIM);
    }

    private void paintLines(HudCanvas canvas, int center, int y, String[] lines) {
        for (int i = 0; i < lines.length; i++) {
            canvas.centeredText(canvas.fit(lines[i], this.width - 20), center, y + i * (canvas.lineHeight() + 3), DIM);
        }
    }

    /** What the sliders add up to: samples per minute and the active export filters. */
    private void paintPower(HudCanvas canvas, int center) {
        int y = rowY(5) + 2;
        if (y + 2 * (canvas.lineHeight() + 3) > this.height - 50) {
            return; // window too small - the hover help still explains each slider
        }
        int perMinute = 1200 / Math.max(1, cfg().sampleTicks);
        List<String> active = new ArrayList<>();
        if (cfg().minEpisodeSeconds > 0) {
            active.add("episodes under " + seconds(cfg().minEpisodeSeconds) + " dropped");
        }
        if (cfg().idleKeepPercent < 100) {
            active.add(cfg().idleKeepPercent + "% of idle kept");
        }
        String filters = active.isEmpty() ? "export filters off" : "export: " + String.join(", ", active);
        paintLines(canvas, center, y, new String[]{
                "Up to " + perMinute + " samples per minute of play  -  " + filters,
                cfg().advanced ? "Advanced is ON: train with  python -m pymc_bot train --advanced"
                        : "Advanced is OFF: the last three sliders apply once it is on (Advanced tab)"});
    }

    private void paintRecorder(HudCanvas canvas, int center) {
        String dataset = recorder().datasetPath() == null ? "<game folder>/pymcbot/" : recorder().datasetPath().toString();
        paintLines(canvas, center, rowY(3) + 2, new String[]{
                "Dataset: " + dataset,
                "Mode: " + (cfg().advanced ? "advanced (entities + items)" : "basic") + "   Sample rate: "
                        + perSecond(cfg().sampleTicks)});
    }

    private void paintAbout(HudCanvas canvas, int center) {
        int y = CONTENT_Y - 2;
        int step = canvas.lineHeight() + 3;
        String[] lines = {
                PymcPlaytimeMod.NAME + " " + PymcPlaytimeMod.modVersion() + " for Minecraft " + PymcPlaytimeMod.minecraftVersion()
                        + " (compat " + Mc.ERA + ")",
                "Records your playtime so PyMC Bot can learn to play like you.",
                "Config file: " + PlaytimeConfig.file(),
                "Commands: /pymcbot (or /pymc) - try /pymcbot status",
                "Made with \u2764 by codero-sus  -  github.com/codero-sus",
                "codero.is-a.dev  -  the-codero.web.app  -  github.com/codero-sus/PyMC_Bot (PolyForm Noncommercial)",
        };
        for (int i = 0; i < lines.length; i++) {
            canvas.centeredText(canvas.fit(lines[i], this.width - 20), center, y + i * step,
                    i == 0 || i == 4 ? RecorderHud.WHITE : DIM);
        }
    }

    /** A small frame standing in for the game screen, with the badge drawn in the chosen corner. */
    private void paintBadgePreview(HudCanvas canvas, int center, PlaytimeRecorder.HudState state) {
        int w = Math.min(260, this.width - 40);
        int h = 52;
        int x = center - w / 2;
        int y = rowY(2) + 2;
        if (y + h > this.height - 50) {
            return; // window too small for the preview
        }
        canvas.fill(x - 1, y - 1, x + w + 1, y + h + 1, 0xFF3A3F47);
        canvas.fill(x, y, x + w, y + h, 0xFF15181D);
        canvas.centeredText(cfg().showHud ? "preview" : "badge hidden", center, y + h / 2 - canvas.lineHeight() / 2, DIM);
        if (!cfg().showHud) {
            return;
        }
        PlaytimeRecorder.HudState shown = state.active() ? state
                : new PlaytimeRecorder.HudState(true, false, false, false, 1234, 83_000L, "preview");
        RecorderHud.draw(offset(canvas, x, y, w, h), shown, cfg().hudCorner, System.currentTimeMillis());
    }

    /** {@code canvas} restricted to a w x h box at (x, y), so corner maths lands inside the box. */
    private static HudCanvas offset(HudCanvas canvas, int x, int y, int w, int h) {
        return new HudCanvas() {
            @Override
            public int width() {
                return w;
            }

            @Override
            public int height() {
                return h;
            }

            @Override
            public void fill(int x1, int y1, int x2, int y2, int argb) {
                canvas.fill(x + x1, y + y1, x + x2, y + y2, argb);
            }

            @Override
            public void text(String text, int tx, int ty, int argb) {
                canvas.text(text, x + tx, y + ty, argb);
            }

            @Override
            public int textWidth(String text) {
                return canvas.textWidth(text);
            }

            @Override
            public int lineHeight() {
                return canvas.lineHeight();
            }
        };
    }

    @Override
    public void onClose() {
        Mc.openScreen(Minecraft.getInstance(), parent);
    }
}
