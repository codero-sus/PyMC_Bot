# PyMC Playtime Recorder (Fabric mod)

A client-side Fabric mod that records **your own playtime** as a training dataset for
PyMC_Bot's player model: where you were, where you looked, what was around you (3×3×3
blocks + nearby entities) and **which key you pressed** at that moment.

```
/pymc record start     # play normally for a while
/pymc record stop
/pymc export           # writes pymc-playtime/dataset.jsonl
```

Then, from the repository root:

```powershell
python -m pymc_bot train --dataset "<gameDir>/pymc-playtime/dataset.jsonl"

# advanced: learn the entity and item words the recording contains
# (/pymc advanced on before recording)
python -m pymc_bot train --dataset "<gameDir>/pymc-playtime/dataset.jsonl" --advanced
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
`pymcplaytime`, plugin `net.fabricmc.fabric-loom-remap`, `splitEnvironmentSourceSets()`),
`fabric.mod.json` and the Java sources under `src/`. The template's example mixins were
deleted — this mod needs no mixins at all: it reads the vanilla key mappings and the
client tick event, so it cannot break on other mods' code.

## Building

```bash
cd mod
./gradlew build            # -> build/libs/pymcplaytime-0.3.0.jar
./gradlew runClient        # launch a dev client with the mod loaded
```

Requires **JDK 25** for Gradle (Loom 1.18 only resolves on a Java 25 runtime; the mod is
compiled down to Java 21 bytecode); Gradle downloads itself via the wrapper. Versions live in
`gradle.properties` (Minecraft 1.21.11, Fabric Loader 0.19.5, Fabric API 0.141.6+1.21.11);
[check them on fabricmc.net/develop](https://fabricmc.net/develop) and bump them there.
To play on another Minecraft version, update `minecraft_version`, `loader_version` and
`fabric_api_version`, then fix whatever the compiler complains about (the client API is
mostly stable across 1.21.x).

Install by dropping the jar plus [Fabric API](https://modrinth.com/mod/fabric-api) into
`.minecraft/mods/` for a Fabric-Loader profile.

## Commands

| Command | What it does |
| --- | --- |
| `/pymc record start` | Start a recording run (`pymc-playtime/episodes/run-<timestamp>.ndjsonl`). |
| `/pymc record stop` | Stop recording and flush the file. |
| `/pymc record toggle` | Start/stop in one command. |
| `/pymc record episode` | Close the current episode and continue in a new file. |
| `/pymc export` | Flatten every episode into `pymc-playtime/dataset.jsonl` (the trainer input). |
| `/pymc status` | Recording state, dataset size, config. |
| `/pymc sample <ticks>` | Sampling rate: 2 ticks = one sample every 100 ms (20 Hz, the default). |
| `/pymc advanced on\|off` | Advanced recording: also write entities, items, the held item, armour and dropped stacks (what `train --advanced` learns from). |
| `/pymc advanced status` | Show the full recorder configuration, including the advanced limits. |
| `/pymc entities <n>` | How many entities to write per sample in advanced mode (default 24, 0 disables). |
| `/pymc drops <n>` | How many dropped item stacks to write per sample (default 8). |
| `/pymc param` | Show the current recorder configuration. |
| `/pymc train` | Print the exact Python command to train on this dataset. |

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

`/pymc export` writes one training example per sample, which is what
`pymc_bot train` consumes:

```json
{"instruction": "You are a Minecraft player bot. Given the observation, reply with the action to take next.",
 "input": { ...the observation above... },
 "output": "{\"action\": \"forward\"}",
 "episode": "run-20261007-181500"}
```

With `/pymc advanced on` every sample also carries the entity and item half of the schema:

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
whether it is hostile), `items` is the whole inventory as `[slot, item, count]`, and `ground_items`
are the dropped stacks nearby. `train --advanced` learns which of those words matter and how close
they are, which is what makes the bot react to *what* is around it instead of just "something is".

Roughly **1.2 KB per sample ≈ 1.4 MB per minute of playtime**, about 2-4x that in advanced mode
(24 entity slots by default); a 10-minute session is plenty to see the model imitate your habits, an
hour makes it noticeably better.

## Privacy / safety

Everything is written to your own game directory (`<gameDir>/pymc-playtime/`) and never
uploaded anywhere. The mod is client-side only, talks to no server API and works on the
same offline-mode (cracked) servers PyMC_Bot plays on. `/pymc export` reads only that
folder.

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
├── build.gradle, settings.gradle, gradle.properties   # from the Fabric template, edited
├── gradle/wrapper/*, gradlew, gradlew.bat             # Gradle wrapper (from the template)
└── src/
    ├── main/java/dev/pymc/pymcb/                      # loader-agnostic helpers
    │   ├── PymcPlaytimeMod.java                       # mod id + logger
    │   ├── Json.java                                  # tiny JSON writer
    │   └── PlaytimeConfig.java                        # config/pymcplaytime.properties
    ├── client/java/dev/pymc/pymcb/client/             # client-only code
    │   ├── PymcPlaytimeClient.java                    # tick hook + entrypoint
    │   ├── PlaytimeRecorder.java                      # sampling + dataset export
    │   └── PymcCommands.java                          # /pymc ...
    └── main/resources/fabric.mod.json, assets/...     # mod metadata + icon
```

## Licensing

Same MIT license as PyMC_Bot; the Fabric template files it was bootstrapped from are CC0.
