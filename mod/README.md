# PyMC Bot mod (Fabric)

The **PyMC Bot** companion mod: a client-side Fabric mod that records **your own playtime** as a training dataset for
PyMC_Bot's player model: where you were, where you looked, what was around you (3×3×3
blocks + nearby entities) and **which key you pressed** at that moment.

**Works on every Fabric release from Minecraft 1.21.1 to 26.3** – one small jar per version,
all built from the same source (see [Supported versions](#supported-versions)).

**Download:** every jar is attached to the GitHub release
[`pymcbot-v0.8.0`](https://github.com/codero-sus/PyMC_Bot/releases/tag/pymcbot-v0.8.0) (all releases: [Releases](https://github.com/codero-sus/PyMC_Bot/releases)) – no
account needed. Or [compile it yourself](#compiling-the-mod).

Everything is configured on the **[PyMC Bot settings page](#pymc-bot-settings-page)** – open it
with `/pymcbot settings`, the *Open PyMC Bot settings* hotkey, or the **Configure** button next to
PyMC Bot in [Mod Menu](https://modrinth.com/mod/modmenu). While recording, a small **REC** badge in
the corner of the screen shows the run clock and the sample count.

> Until 0.5.0 this mod was called "PyMC Playtime Recorder" (`pymcplaytime-*.jar`). From 0.6.0 it is
> **PyMC Bot** (`pymcbot-*.jar`, mod id `pymcbot`, command `/pymcbot`). Delete the old jar (or run
> `updater --mod-only`, which swaps it for you); your settings move from
> `config/pymcplaytime.properties` to `config/pymcbot.properties` automatically and `/pymc` keeps
> working as a short alias.
>
> From 0.7.0 recordings go to `<gameDir>/pymcbot/` instead of `<gameDir>/pymc-playtime/`. The old
> folder is renamed the first time the recorder needs it (if both exist, nothing is merged or deleted);
> the PyMC Bot app finds datasets in either folder. Changes per version: [CHANGELOG.md](CHANGELOG.md).

```
/pymcbot record start     # or press your "Start / stop recording" hotkey; play normally
/pymcbot record stop
/pymcbot export           # writes pymcbot/dataset.jsonl
```

Then, from the repository root:

```powershell
python -m pymc_bot train --dataset "<gameDir>/pymcbot/dataset.jsonl"

# advanced: learn the entity and item words the recording contains
# (/pymcbot advanced on before recording)
python -m pymc_bot train --dataset "<gameDir>/pymcbot/dataset.jsonl" --advanced
python -m pymc_bot run --think trained --run-name playtime-dataset
```

## How it was made (the Fabric way)

This project follows the [official Fabric workflow](https://docs.fabricmc.net/develop/getting-started/creating-a-project):
the mod started from the canonical **`fabric-example-mod` template zip**, downloaded and
extracted, and then edited into the recorder:

```bash
# what was run to bootstrap this folder
curl -L -o fem.zip https://codeload.github.com/FabricMC/fabric-example-mod/zip/refs/heads/1.21.11
unzip fem.zip && cp -r fabric-example-mod-1.21.11/{gradle,gradlew,gradlew.bat} ./mod/
```

Everything else in this folder is the edited result: the Gradle wrapper jar (which you
cannot hand-roll), `settings.gradle`/`build.gradle`/`gradle.properties` (mod id
`pymcplaytime` (renamed `pymcbot` in 0.6.0), plugin `net.fabricmc.fabric-loom-remap`, `splitEnvironmentSourceSets()`),
`fabric.mod.json` and the Java sources under `src/`. The template's example mixins were
deleted — this mod needs no mixins at all: it reads the vanilla key mappings and the
client tick event, so it cannot break on other mods' code.

## Supported versions

| Minecraft | Jar to install | Fabric API it was built with | Java to play | CI |
| --- | --- | --- | --- | --- |
| 1.21.1 | `pymcbot-0.6.0+1.21.1.jar` | 0.116.17+1.21.1 or newer | 21 | compiled |
| 1.21.2 | `pymcbot-0.6.0+1.21.2.jar` | 0.106.1+1.21.2 or newer | 21 | compiled |
| 1.21.3 | `pymcbot-0.6.0+1.21.3.jar` | 0.114.1+1.21.3 or newer | 21 | compiled |
| 1.21.4 | `pymcbot-0.6.0+1.21.4.jar` | 0.119.4+1.21.4 or newer | 21 | compiled + launched in game |
| 1.21.5 | `pymcbot-0.6.0+1.21.5.jar` | 0.128.2+1.21.5 or newer | 21 | compiled + launched in game |
| 1.21.6 | `pymcbot-0.6.0+1.21.6.jar` | 0.128.2+1.21.6 or newer | 21 | compiled + launched in game |
| 1.21.7 | `pymcbot-0.6.0+1.21.7.jar` | 0.129.0+1.21.7 or newer | 21 | compiled + launched in game |
| 1.21.8 | `pymcbot-0.6.0+1.21.8.jar` | 0.136.1+1.21.8 or newer | 21 | compiled + launched in game |
| 1.21.9 | `pymcbot-0.6.0+1.21.9.jar` | 0.134.1+1.21.9 or newer | 21 | compiled + launched in game |
| 1.21.10 | `pymcbot-0.6.0+1.21.10.jar` | 0.138.4+1.21.10 or newer | 21 | compiled + launched in game |
| 1.21.11 | `pymcbot-0.6.0+1.21.11.jar` | 0.141.6+1.21.11 or newer | 21 | compiled + launched in game |
| 26.1 | `pymcbot-0.6.0+26.1.jar` | 0.145.1+26.1 or newer | 25 | compiled + launched in game |
| 26.1.1 | `pymcbot-0.6.0+26.1.1.jar` | 0.145.4+26.1.1 or newer | 25 | compiled + launched in game |
| 26.1.2 | `pymcbot-0.6.0+26.1.2.jar` | 0.155.3+26.1.2 or newer | 25 | compiled + launched in game |
| 26.2 | `pymcbot-0.6.0+26.2.jar` | 0.161.0+26.2 or newer | 25 | compiled + launched in game |
| 26.3 | `pymcbot-0.6.0+26.3.jar` | 0.162.0+26.3 or newer | 25 | compiled + launched in game |

That is every stable release from 1.21.1 to 26.3, 16 jars in total. *compiled + launched in game*
means CI also starts that Minecraft version with the jar and runs the [in-game test](#6-run-the-in-game-test-optional)
(Fabric's client game test API exists from 1.21.4; 1.21.1 – 1.21.3 are compile-checked). Each jar declares exactly the
Minecraft version it was compiled for, so Fabric Loader refuses a mismatched one instead of
crashing mid-game – **pick the jar whose name ends in your game version**. The exact pins
(Fabric API build, Java level, loom flavour) are in [`versions.json`](versions.json) and come
from [fabricmc.net/develop](https://fabricmc.net/develop) and the official
[fabric-example-mod](https://github.com/FabricMC/fabric-example-mod) branches for each version.

Why one jar per version? 26.1 is the first unobfuscated Minecraft: mods for 1.21.11 and older
are remapped (intermediary names) while 26.x mods use the game's real names, so no single jar
can load on both. Inside each line a few APIs also moved (see [How multi-version works](#how-multi-version-works)),
and building every version separately means CI has really compiled each one.

## Compiling the mod

Everything below works the same on Windows, macOS and Linux. You need **only a JDK** – Gradle,
Minecraft, Fabric Loom and Fabric API are downloaded automatically on the first build.

### 1. Install JDK 25

Loom 1.18 (the Fabric build plugin every target uses) only runs on **Java 25**, even when the
jar is for 1.21.x (those are still compiled down to Java 21 bytecode).

| OS | Command |
| --- | --- |
| Windows | `winget install EclipseAdoptium.Temurin.25.JDK` |
| macOS | `brew install --cask temurin` |
| Linux | your package manager (`sudo apt install openjdk-25-jdk`), or [SDKMAN](https://sdkman.io/): `sdk install java 25-tem` |

Or download Temurin 25 from [adoptium.net](https://adoptium.net/temurin/releases/?version=25).
Check it in a **new** terminal:

```bash
java -version        # must say 25 (or newer)
```

If an older Java still shows up, point `JAVA_HOME` at the JDK 25 folder (Windows:
*System Properties → Environment Variables*; macOS/Linux: `export JAVA_HOME=...` in your shell
profile).

### 2. Get the source

```bash
git clone https://github.com/codero-sus/PyMC_Bot.git
cd PyMC_Bot/mod
```

(or download the repository ZIP from GitHub, extract it and open a terminal in its `mod` folder).

### 3. Build the jar for your Minecraft version

```bash
# macOS / Linux
./gradlew build -Pmc=1.21.4

# Windows (PowerShell or cmd)
.\gradlew.bat build -Pmc=1.21.4
```

The jar is **`build/libs/pymcbot-0.6.0+1.21.4.jar`** (ignore the `-sources.jar` next to it).
Leave out `-Pmc` to build the newest version (26.3). `./gradlew printTargets` lists every
version you can pass. The first build of a version takes a few minutes (it downloads and
prepares that Minecraft); later builds take seconds.

### 4. Or build every version at once

```bash
./build-all.sh                    # macOS / Linux  -> dist/ (16 jars)
./build-all.sh 1.21.1 26.3        # just these
```

```powershell
.\build-all.ps1                   # Windows -> dist\ (16 jars)
.\build-all.ps1 1.21.1, 26.3
```

(If PowerShell refuses to run scripts: `powershell -ExecutionPolicy Bypass -File .\build-all.ps1`.)
A version that fails does not stop the others; the summary at the end names it.

### 5. Try it in a development client (optional)

```bash
./gradlew runClient -Pmc=26.3     # starts Minecraft 26.3 with the mod loaded
```

### 6. Run the in-game test (optional)

```bash
./gradlew runProductionClientGameTest -Pmc=1.21.4 -PgameTest     # any version from 1.21.4 up
```

This launches the real game (a production Fabric launch with Fabric API) and runs
[`PymcClientGameTest`](src/gametest/java/dev/pymc/pymcb/gametest/PymcClientGameTest.java): it
creates a singleplayer world, starts recording, walks, jumps, strafes and turns, checks the HUD
badge state, opens the PyMC Bot settings page (and checks that sampling holds while it is open),
renders all five of its tabs, checks *Reset page*, pauses,
stops, exports and asserts that the dataset contains the walk. Results land in
`build/run/gametest/`: `screenshots/` (badge, one per settings tab, paused badge), `pymc-gametest.txt`
and the recorded `pymcbot/`. The game runs on Java 21 for 1.21.x and Java 25 for 26.x
(Gradle needs both JDKs installed; override with `-PgameTestJava=25`). The game opens on your
`DISPLAY`. On a Linux machine without one, start a 24-bit virtual display first – 26.3+ (SDL3)
finds no OpenGL visual on Xvfb's default 8-bit screen:

```bash
Xvfb :99 -screen 0 1920x1080x24 &  export DISPLAY=:99
```

`-PgameTest` builds a separate `pymcbot-gametest-…` jar – the normal jar never contains test
code. CI runs this for every version from 1.21.4 on and uploads the screenshots as the
`gametest-<version>` artifact.

### No JDK? Download the release

Every jar is on the [GitHub release](https://github.com/codero-sus/PyMC_Bot/releases) – CI publishes it automatically once all 16 versions
have compiled and every in-game test has passed. Bleeding-edge builds of any push are under the
repository's **Actions** tab → the latest green **ci** run → *Artifacts* →
**`pymcbot-all-versions`** (or `pymcbot-<version>` for a single jar).

### Install

1. Install [Fabric Loader](https://fabricmc.net/use/installer/) **0.19.5 or newer** for your
   Minecraft version.
2. Put the matching **[Fabric API](https://modrinth.com/mod/fabric-api)** jar and
   `pymcbot-0.6.0+<your version>.jar` into `.minecraft/mods/`.
3. Start the Fabric profile and type `/pymcbot version` – it prints the mod, game and compat
   layer versions.

### Update

From the PyMC_Bot folder:

```bash
./updater.sh --mod-only                 # macOS / Linux  (updater.cmd --mod-only on Windows)
./updater.sh --mod-only --mc 1.21.4     # first install for 1.21.4
./updater.sh --mod-only --check         # just show whether a newer jar exists
```

It reads the `+<mc>` part of the installed jar's name, downloads the same Minecraft version from the
newest release, checks the SHA-256 and `fabric.mod.json`, and renames the old jar to `*.jar.old`
(Fabric ignores it). Use `--mods-dir <folder>` for launchers such as Prism or MultiMC, and close
Minecraft first on Windows. See [Updating](../README.md#updating) for the rest.

### Troubleshooting the build

| Problem | Fix |
| --- | --- |
| `Unknown Minecraft version 'x'` | Use one of the versions from `./gradlew printTargets` (spelled exactly, e.g. `1.21.10`, `26.1.2`). |
| `requires at least JVM runtime version 25` / `Unsupported class file major version` | Gradle is running on an older Java – install JDK 25 and set `JAVA_HOME` (step 1). `./gradlew --version` shows which JVM it uses. |
| `Permission denied: ./gradlew` | `chmod +x gradlew build-all.sh` |
| Odd errors after switching `-Pmc` | `./gradlew clean build -Pmc=<version>`; if that does not help, delete `mod/.gradle/` and `mod/build/`. |
| `OutOfMemoryError` | Raise `org.gradle.jvmargs` in `gradle.properties` (e.g. `-Xmx3G`). |
| Download timeouts behind a proxy | Set `systemProp.https.proxyHost` / `proxyPort` in `~/.gradle/gradle.properties`. |
| The game says the mod needs a different Minecraft | You installed the jar for another version – use the one ending in your exact version. |
| The game says it needs Java 25 | 26.x requires Java 25; the official launcher ships it, custom launchers may need updating. |

## How multi-version works

```
mod/
├── versions.json                 # one entry per Minecraft version: Fabric API, Java, loom flavour, compat era
├── build.gradle                  # reads versions.json; -Pmc picks the entry
└── src/
    ├── main/, client/            # identical for every version
    └── compat/<era>/.../Mc.java  # the only version-specific code
```

* **1.21.x** builds use the `net.fabricmc.fabric-loom-remap` plugin with Mojang's mappings and
  `modImplementation`; **26.x** builds use `net.fabricmc.fabric-loom` without remapping and plain
  `implementation`, exactly like the official template for each version. One Loom release
  (1.18) provides both, and Gradle applies one of them per build.
* Everything that differs between versions goes through `dev.pymc.pymcb.client.compat.Mc`.
  There is one copy per API era and `build.gradle` adds the right folder to the source set:

| Era folder | Minecraft | What changed |
| --- | --- | --- |
| `v1_21_1` | 1.21.1 – 1.21.4 | baseline: `Inventory.selected` field, string key categories, `ResourceLocation`; HUD through `HudRenderCallback`, drawing with `GuiGraphics.drawString` |
| `v1_21_5` | 1.21.5 – 1.21.8 | `Inventory.getSelectedSlot()` |
| `v1_21_9` | 1.21.9 – 1.21.10 | `KeyMapping.Category` objects; HUD through Fabric's `HudElementRegistry` (added in 1.21.6) |
| `v1_21_11` | 1.21.11 | `ResourceLocation` → `Identifier`, `ResourceKey.location()` → `identifier()` |
| `v26_1` | 26.1 – 26.1.2 | unobfuscated; Fabric API renames `ClientCommandManager` → `ClientCommands`, `KeyBindingHelper` → `KeyMappingHelper`; `sendSystemMessage` for chat lines; `GuiGraphics` → `GuiGraphicsExtractor` (`drawString` → `text`, `Screen.render` → `extractRenderState`), `HudRenderCallback` removed |
| `v26_2` | 26.2 – 26.3 | the open screen moved to `Minecraft.gui` (`gui.screen()`, `gui.setScreen(...)`); attended client commands; 26.3 switched GLFW → SDL3 (the hotkeys are registered unbound, so no raw key codes are involved) |

* `tests/test_mod_versions.py` (run with the Python test suite) checks that the table covers
  every release, that each entry is consistent, that all `Mc.java` copies expose the same API and
  that no version-specific call leaks outside `src/compat/`. CI then compiles every version and
  launches every version from 1.21.4 on.
* The badge and the settings page draw through a tiny `HudCanvas` interface (`fill`, `text`,
  `textWidth`) that each `Mc.java` implements on top of that era's drawing class, so the drawing
  code itself is shared.

**Adding a new Minecraft version:** add an entry to `versions.json` with the pins from
[fabricmc.net/develop](https://fabricmc.net/develop) (the `fabric-example-mod` branch for that
version shows the same values) and run `./gradlew build -Pmc=<new>`. If it compiles, you are
done; if Mojang or Fabric renamed something, copy the newest `src/compat/<era>` folder to a new
era, fix the calls there and point the new entry at it.

## Hotkeys

Options → Controls → Key Binds → **PyMC Bot**:

| Key | Default | Does |
| --- | --- | --- |
| Start / stop recording | unbound | same as `/pymcbot record toggle` |
| Pause / resume recording | unbound | same as `/pymcbot pause` / `/pymcbot resume` |
| Open PyMC Bot settings | unbound | same as `/pymcbot settings` |

They start unbound so they can never steal a key from the game or another mod – assign whatever
you like. With notifications on, a chat line confirms each change.

## On-screen badge

While a recording runs, a badge sits in the top-left corner of the screen:

| Badge | Meaning |
| --- | --- |
| 🔴 **REC** `3:42  4,410 samples` (blinking dot) | recording – clock since the run started, samples written |
| 🟡 **PAUSED** | paused with `/pymcbot pause` or the hotkey; nothing is written |
| 🔵 **MENU** | a screen is open and `/pymcbot menus` is on, so sampling holds |
| ⚪ **IDLE** | no input for longer than `/pymcbot idle` seconds – AFK samples are skipped until you move |

It hides with F1 and when nothing is being recorded. `/pymcbot hud off` turns it off,
`/pymcbot hud corner top_right` (or `top_left`, `bottom_left`, `bottom_right`) moves it.

## PyMC Bot settings page

Open it with **`/pymcbot settings`** (or `/pymcbot config`), the *Open PyMC Bot settings* hotkey,
or Mod Menu's **Configure** button. It has a button for every option in
`config/pymcbot.properties`, grouped in five tabs. Every click is saved at once, hovering a button
shows what it does, and each settings tab has **Reset page** (click twice) to restore its defaults.

| Tab | Buttons | Same as |
| --- | --- | --- |
| **Recorder** | Start / Stop recording, Pause / Resume, New episode, Export dataset, Status in chat, Training command | `/pymcbot record toggle`, `pause` / `resume`, `record episode`, `export`, `status`, `train` |
| **Recording** | Sample rate (20 / 10 / 5 / 2 / 1 per second), Pause in menus, Skip AFK after (never / 5 – 300 s), Auto-record on join, Block area (3×3×3), Nearby entities (0 – 64), Move threshold, Turn threshold | `/pymcbot sample`, `menus`, `idle`, `autorecord`; `includeBlocks`, `maxNearby`, `moveEpsilon`, `turnDegrees` |
| **Advanced** | Entities + items, Max entities (0 – 128), Entity range (8 – 128 blocks), Ground items (0 – 64) | `/pymcbot advanced`, `entities`, `drops`; `entityRange` |
| **HUD & chat** | REC badge on/off, Badge corner, Chat notices, Hotkeys… – with a **live preview** of the badge in the chosen corner | `/pymcbot hud on\|off`, `hud corner …`, `notify` |
| **About** | Mod / Minecraft / compat versions, config file location, credits; Links in chat, Version in chat | `/pymcbot about`, `version` |

The line under the tabs always shows the live recorder state (REC / PAUSED / MENU / IDLE, clock,
samples, run name). Mod Menu is optional – without it the page is still one command or hotkey away.

## Commands

| Command | What it does |
| --- | --- |
| `/pymcbot record start` | Start a recording run (`pymcbot/episodes/run-<timestamp>.ndjsonl`). |
| `/pymcbot record stop` | Stop recording and flush the file. |
| `/pymcbot record toggle` | Start/stop in one command. |
| `/pymcbot record episode` | Close the current episode and continue in a new file. |
| `/pymcbot pause` / `/pymcbot resume` | Stop sampling for a moment without closing the episode. |
| `/pymcbot export` | Flatten every episode into `pymcbot/dataset.jsonl` (the trainer input). |
| `/pymcbot status` | Recording state, dataset size, config. |
| `/pymcbot sample <ticks>` | Sampling rate: 2 ticks = one sample every 100 ms (20 Hz, the default). |
| `/pymcbot advanced on\|off` | Advanced recording: also write entities, items, the held item, armour and dropped stacks (what `train --advanced` learns from). |
| `/pymcbot advanced status` | Show the full recorder configuration, including the advanced limits. |
| `/pymcbot entities <n>` | How many entities to write per sample in advanced mode (default 24, 0 disables). |
| `/pymcbot drops <n>` | How many dropped item stacks to write per sample (default 8). |
| `/pymcbot autorecord on\|off` | Start recording automatically whenever you join a world or server. |
| `/pymcbot menus on\|off` | Skip samples while a screen (inventory, chat, pause menu) is open – default on. |
| `/pymcbot idle <seconds>` | Stop writing samples after this long without any input, resume on the next action (default 15, 0 = never). |
| `/pymcbot notify on\|off` | Chat lines when a hotkey or auto-record changes the state. |
| `/pymcbot hud on\|off\|status` | The REC / PAUSED badge in a corner of the screen (default on). |
| `/pymcbot hud corner <corner>` | Move the badge: `top_left` (default), `top_right`, `bottom_left`, `bottom_right`. |
| `/pymcbot settings` (or `config`) | Open the [PyMC Bot settings page](#pymc-bot-settings-page). |
| `/pymcbot` / `/pymcbot help` | Short help. `/pymc …` works as an alias for every command. |
| `/pymcbot param` | Show the current recorder configuration. |
| `/pymcbot train` | Print the exact Python command to train on this dataset. |
| `/pymcbot version` | Mod version, Minecraft version and compat layer. |
| `/pymcbot about` | Author, links and credits (made by codero-sus). |

On Minecraft 26.2+ every command that changes the recorder needs an *attended* run (you typed it),
so a server cannot start a recording through a clickable chat message; read-only commands
(`status`, `param`, `train`, `version`, `about`, `hud status`) always work.

All settings persist in `config/pymcbot.properties` (`sampleTicks`, `advanced`,
`maxEntities`, `pauseInMenus`, `idleSkipSeconds`, `autoRecord`, `notifications`, `showHud`,
`hudCorner`, ...).

**Dataset quality.** Time spent in menus, paused, or idle beyond `idleSkipSeconds` is not
written, so AFK minutes do not drown your real play in `none` samples; `/pymcbot status` and
`/pymcbot record stop` report how much was skipped. Disconnecting stops the recording and flushes
it, so an episode never spans two worlds. `meta.json` also records `mod_version`,
`minecraft_version`, the compat era and the skip counters.

## What is recorded

One NDJSON line per sample (default every 2 client ticks ≈ 100 ms):

```json
{"tick": 120, "episode": "run-20261007-181500", "activity": "forward",
 "x": 12.5, "y": 64.0, "z": -3.2, "vx": -0.04, "vy": 0.0, "vz": 0.21, "moved": 0.21,
 "yaw": 91.0, "pitch": 4.5, "on_ground": true, "sprinting": false, "sneaking": false,
 "selected_slot": 0, "health": 20.0, "food": 20, "dimension": "overworld",
 "blocks": [[-1, -1, -1, "minecraft:stone"], "... 27 cells ..."],
 "nearby": [{"type": "minecraft:zombie", "dx": 3.1, "dy": 0.0, "dz": -1.4,
             "dist": 3.4, "hostile": true, "health": 20.0}]}
```

`activity` is the action the player was performing during that sample:

`forward` · `back` · `left` · `right` · `jump` · `sneak` · `look` · `attack` · `use` ·
`hold` (hotbar switch) · `move` (moved without a key, e.g. pushed) · `none` (stood still)

`/pymcbot export` writes one training example per sample, which is what
`pymc_bot train` consumes:

```json
{"instruction": "You are a Minecraft player bot. Given the observation, reply with the action to take next.",
 "input": { ...the observation above... },
 "output": "{\"action\": \"forward\"}",
 "episode": "run-20261007-181500"}
```

With `/pymcbot advanced on` every sample also carries the entity and item half of the schema:

```json
 "entities": [{"type": "minecraft:zombie", "dx": 3.1, "dy": 0.0, "dz": -1.4, "dist": 3.4,
               "hostile": true, "player": false, "health": 20.0, "on_ground": true,
               "yaw": 87.5, "count": 0, "held_item": ""}],
 "items": [[0, "minecraft:iron_sword", 1], [3, "minecraft:cooked_beef", 5]],
 "held_item": "minecraft:iron_sword",
 "armor": ["minecraft:iron_helmet", "minecraft:iron_boots"],
 "ground_items": [{"item": "minecraft:oak_log", "count": 3, "dist": 2.4}]
```

`entities` is every entity within `entityRange` blocks (players, mobs, drops — with the held item and
whether it is hostile, i.e. spawns in the MONSTER category, which includes slimes, ghasts and phantoms), `items` is the whole inventory as `[slot, item, count]`, and `ground_items`
are the dropped stacks nearby. `train --advanced` learns which of those words matter and how close
they are, which is what makes the bot react to *what* is around it instead of just "something is".

Roughly **1.2 KB per sample ≈ 1.4 MB per minute of playtime**, about 2-4x that in advanced mode
(24 entity slots by default); a 10-minute session is plenty to see the model imitate your habits, an
hour makes it noticeably better.

## Privacy / safety

Everything is written to your own game directory (`<gameDir>/pymcbot/`) and never
uploaded anywhere. The mod is client-side only, talks to no server API and works on the
same offline-mode (cracked) servers PyMC_Bot plays on. `/pymcbot export` reads only that
folder.

## Publishing on Modrinth

Everything Modrinth needs is in the repo:

| File | Use |
| --- | --- |
| [`MODRINTH.md`](MODRINTH.md) | the project description – synced to Modrinth on every release |
| [`../docs/modrinth/`](../docs/modrinth/) | 512×512 icon, banner and a copy-paste [listing sheet](../docs/modrinth/LISTING.md) for every Modrinth field |
| [`modrinth.json`](modrinth.json) | title, summary, categories, links and licence – synced on every release |
| [`CHANGELOG.md`](CHANGELOG.md) | per-version changelog – the section for `mod_version` is used for every upload |
| [`../scripts/modrinth_publish.py`](../scripts/modrinth_publish.py) | syncs the project page and one Modrinth version per jar |

1. **Create the project once** on modrinth.com → *Create a project*: name **PyMC Bot**, type *Mod*,
   summary e.g. "Record your own playtime so a bot can learn to play like you". Then in its settings:
   environment **client-side only**, license **PolyForm Noncommercial 1.0.0** (`PolyForm-Noncommercial-1.0.0`), categories *Utility* (+ *Game Mechanics* if you like),
   source/issues links `https://github.com/codero-sus/PyMC_Bot`, upload the icon.
2. **Get the jars**: `./build-all.sh` (→ `mod/dist/`) or download them from the GitHub release.
3. **Connect GitHub to Modrinth once** – after that every push to `main` keeps Modrinth up to date:

   **a. Make a Modrinth token** at <https://modrinth.com/settings/pats> → *Create a PAT*: name `PyMC Bot CI`,
   expiry as long as you like, and tick these scopes:

   | Scope | Needed for |
   | --- | --- |
   | Read projects | reading the project while it is under review / unlisted |
   | Write projects | title, summary, description, links, licence |
   | Read versions | finding the versions that are already there |
   | Create versions | new versions |
   | Write versions | updating versions and replacing their jar |

   **b. Add it to GitHub** at
   [github.com/codero-sus/PyMC_Bot/settings/secrets/actions](https://github.com/codero-sus/PyMC_Bot/settings/secrets/actions)
   (*Settings → Secrets and variables → Actions*):

   | Tab | Button | Name | Value |
   | --- | --- | --- | --- |
   | **Secrets** | *New repository secret* | `MODRINTH_TOKEN` | the token (`mrp_…`) |
   | **Variables** | *New repository variable* | `MODRINTH_PROJECT` | `pymcbot` (the project slug) |

   Nothing else is needed. Without the secret the job only prints what it would publish; without the
   variable it does not run.

4. **What happens on every push to `main`**: CI builds all versions, runs the in-game tests, makes the
   GitHub release, then the **mod (Modrinth)** job
   - updates the project's **title, summary, description** (`MODRINTH.md`), categories, links and licence
     (`modrinth.json`; the summary's version range comes from `versions.json`);
   - updates **every version** of the current `mod_version`: name, changelog (`CHANGELOG.md`), game version,
     loader, Fabric API / Mod Menu dependencies and channel, and **replaces the jar when the code changed**
     (rebuilds of unchanged code are recognised and left alone);
   - creates versions that are missing (a new Minecraft version, or a new `mod_version`);
   - keeps only the newest Minecraft version of the current release *featured*.

   Bump `mod_version` in `gradle.properties` (+ a `CHANGELOG.md` section) when you want a new version that
   launchers offer as an update; without a bump the existing versions are refreshed in place. Branch and
   pull-request builds never publish – unless a branch commit message contains `[modrinth]`, which publishes
   that push without merging.
   To change **only the description** on the Modrinth page (no build, no versions), edit `MODRINTH.md`
   and put `[modrinth-doc]` in the commit message – it is updated within a minute. *Actions → Publish to Modrinth → Run workflow* does the same by hand.

   By hand from your PC:

   ```bash
   python scripts/modrinth_publish.py --jars mod/dist --update-project --dry-run    # check, no token needed
   MODRINTH_TOKEN=mrp_... python scripts/modrinth_publish.py --project pymcbot --jars mod/dist --update-project
   ```

   Limits: GitHub Actions minutes are free and unlimited for public repositories (20 jobs at once, 6 h per
   job). The Modrinth API allows 300 requests per minute per IP; a full sync needs at most about 70, and the
   script waits and retries when Modrinth answers 429 or a temporary 5xx.
5. Submit the project for review on Modrinth (new projects are checked by their moderators).

## Other modding platforms

Fabric is the right platform for this particular job (client-side, no mixins, no server
changes), but the recorder ports easily:

* **NeoForge / Forge** — start from the [NeoForge MDK](https://github.com/neoforged/ModDevGradle)
  (or the [Forge MDK zip](https://maven.minecraftforge.net/net/minecraftforge/forge/maven-metadata.json)),
  then port `PlaytimeRecorder` one-to-one: `ClientTickEvent.Post` instead of
  `ClientTickEvents.END_CLIENT_TICK`, `RegisterClientCommandsEvent` instead of
  `ClientCommandRegistrationCallback`, and the same `LocalPlayer`/`BlockState` calls.
  `Json` and `PlaytimeConfig` are plain Java and copy across unchanged.
* **Vanilla datapack** — cannot record a player's own inputs, so it is not an option.
* The dataset format is deliberately loader-agnostic: anything that writes those NDJSON
  lines feeds the same trainer.

## Layout

```
mod/
├── versions.json                                      # supported Minecraft versions + pins
├── build.gradle, settings.gradle, gradle.properties   # from the Fabric template, edited (-Pmc=<version>)
├── build-all.sh, build-all.ps1                        # build every version into dist/
├── gradle/wrapper/*, gradlew, gradlew.bat             # Gradle wrapper (from the template)
└── src/
    ├── main/java/dev/pymc/pymcb/                      # loader-agnostic helpers
    │   ├── PymcPlaytimeMod.java                       # mod id (pymcbot), name "PyMC Bot", logger, versions
    │   ├── Json.java                                  # tiny JSON writer
    │   └── PlaytimeConfig.java                        # config/pymcbot.properties (migrates the old file)
    ├── client/java/dev/pymc/pymcb/client/             # client-only code (same for every version)
    │   ├── PymcPlaytimeClient.java                    # tick hook, hotkeys, join/disconnect, entrypoint
    │   ├── PlaytimeRecorder.java                      # sampling + dataset export
    │   ├── PymcCommands.java                          # /pymcbot ... (alias /pymc)
    │   ├── RecorderHud.java                           # the REC / PAUSED / IDLE / MENU badge
    │   ├── PymcBotScreen.java                         # the PyMC Bot settings page (5 tabs)
    │   ├── PymcModMenu.java                           # Mod Menu "Configure" button -> settings page
    │   └── HudCanvas.java                             # drawing calls the badge and screen use
    ├── client/java/com/terraformersmc/modmenu/api/    # compile-only Mod Menu API stubs (never shipped)
    ├── compat/<era>/java/.../client/compat/Mc.java    # the version-specific calls, one copy per era
    ├── gametest/java/dev/pymc/pymcb/gametest/         # in-game test, only built with -PgameTest
    └── main/resources/fabric.mod.json, assets/...     # mod metadata (filled per version), icon, lang
```

## Licensing

Same license as PyMC_Bot: the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) – free for personal,
non-commercial use; no selling or commercial use without permission from codero-sus (see
[LICENSE](LICENSE) and the main [README](../README.md#license)). Releases up to 0.7.0 were published under
MIT. The Fabric template files the mod was bootstrapped from are CC0; the compile-only Mod Menu API stubs
are Mod Menu's own code (MIT, TerraformersMC) and are never shipped in the jars.

---

<sub>Made with ❤️ by [codero-sus](https://github.com/codero-sus) · [the-codero.web.app](https://the-codero.web.app/)</sub>
