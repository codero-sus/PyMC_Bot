# PyMC Bot mod – changelog

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
