# PyMC Bot mod – changelog

## 0.9.0

- New **Power** tab on the settings page with sliders for power users who want finer control over
  their datasets: sample rate (1 – 20/s), turn threshold (0.1 – 20°), move threshold (0 – 0.5 blocks),
  AFK skipping (0 – 600 s), nearby entities, and the advanced limits (max entities, entity range,
  ground items). Drag, or click and use the arrow keys; hover a slider for help.
- Two new **export filters** for cleaner datasets, also on the Power tab:
  - **Min episode** leaves accidental short recordings out of `/pymcbot export` (0 – 300 s).
  - **Keep idle samples** thins out long standing-still stretches (0 – 100 %). The moment you stop
    is always kept, so the bot still learns to stop, just not to stand around.
  Your episode files are never changed, so you can export again with other values.
  `/pymcbot export` reports what was filtered.
- *Reset page* on the Power tab restores every slider's default.

## 0.8.0

- New icon: the PyMC Bot logo (also shown in Mod Menu and on Modrinth).
- Now licensed under **PolyForm Noncommercial 1.0.0** – free for personal, non-commercial use
  (shown on the About tab and in `/pymcbot about`).
- Published on Modrinth: one version per Minecraft version, 1.21.1 to 26.3.

## 0.7.0

- Recordings now live in `<game folder>/pymcbot/` (was `pymc-playtime/`). The old folder is moved
  automatically the first time you record or export; the PyMC Bot app reads both.
- New mod icon.
- `/pymcbot train` suggests `pymc-bot` as the Ollama model name.
- Ready for Modrinth: client-only, Fabric API required, Mod Menu optional.

## 0.6.0

- Renamed from "PyMC Playtime Recorder" to **PyMC Bot** (`pymcbot-*.jar`, mod id `pymcbot`,
  command `/pymcbot`, `/pymc` still works). Settings move to `config/pymcbot.properties` automatically.
- In-game settings page with five tabs (also via Mod Menu's Configure button).
- REC / PAUSED / MENU / IDLE badge, hotkeys, auto-record, AFK skipping, advanced entity + item recording.
- One jar per Minecraft version from 1.21.1 to 26.3, all tested in CI.
