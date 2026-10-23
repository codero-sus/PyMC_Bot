# PyMC Bot mod – changelog

## 0.11.0

- **Control your bots from inside the game** with the new `/pymcbot bot …` commands. They talk to
  the PyMC Bot app's control panel (`python -m pymc_bot`, default `http://127.0.0.1:8000`):
  - `/pymcbot bot status` · `list` · `skills` – the selected bot, every bot, the 44 skills.
  - `/pymcbot bot do <skill> [key=value …]` – run any skill, e.g. `do chop_tree count=12`,
    `do mine_ore iron`, `do fish seconds=60`, `do craft name=torch count=8`.
  - `/pymcbot bot come` · `follow` · `stop` · `home` – the bot walks to you, follows you, stops, or
    goes back to its home point. `/pymcbot bot say <text>` chats through the bot.
  - `/pymcbot bot auto on|off|status` – the **autopilot**: the bots sleep at night, eat, fight back,
    swim up, put out fires, equip better armour, place torches and tidy their inventory on their own.
  - `/pymcbot bot ai start|stop` – the LLM brain.
  - `/pymcbot bot panel <url>` – the panel runs on another PC? Point the mod at it
    (start the panel with `--host 0.0.0.0`). `/pymcbot bot panel reset` goes back to the default.
- Replies arrive in chat when the bot is done; the game never waits for the panel.
- Recording is unchanged and still makes no network connections.

## 0.10.0

- **Screenshots for Minecraftism VL**, the vision edition of the PyMC Bot Minecraft model. Turn
  them on in *Settings - Advanced* or with `/pymcbot screenshots on`: while you record, a small
  picture of your screen (640 px wide by default) is saved every 5 seconds into
  `<game folder>/pymcbot/shots/`, and the sample recorded at that moment points to it. The model
  then learns what you did *and what you saw*.
  - `/pymcbot screenshots every <seconds>`, `/pymcbot screenshots width <pixels>`,
    `/pymcbot screenshots status`.
  - Pictures are scaled down on the client and written on a background thread, so the game does
    not stutter. Off by default: an hour of recording at the defaults is roughly 150 – 300 MB.
- `/pymcbot train` shows the command that builds vision data when screenshots are on.
- Screenshots work on every supported version, 1.21.1 – 26.3 (checked by the in-game tests on
  1.21.4 and newer).

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
