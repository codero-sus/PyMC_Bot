![PyMC Bot](https://raw.githubusercontent.com/codero-sus/PyMC_Bot/main/banner.jpg)

# PyMC Bot

**Teach a Minecraft bot to play like you.** PyMC Bot is a small, client-side Fabric mod that records
your own playtime — where you walk, where you look, what is around you and which keys you press — into
a dataset. The free, open-source [PyMC Bot](https://github.com/codero-sus/PyMC_Bot) Python app trains
a player model on it, and that model can then play a bot on your server the way you do.

> One jar per Minecraft version, **1.21.1 to 26.3** — download the file that matches your game.

## Features

- 🔴 **One-key recording** – start / stop / pause with a hotkey or `/pymcbot record toggle`; a small
  **REC** badge shows the run clock and the sample count (hides with F1).
- ⚙️ **Settings page** – every option has a button: `/pymcbot settings`, the hotkey, or
  **Configure** in [Mod Menu](https://modrinth.com/mod/modmenu). Changes save instantly.
- 🧠 **Smart sampling** – 1 to 20 samples per second, skips menus and AFK time, auto-records when
  you join a world if you want.
- 🎚️ **Power tab** – sliders for fine control over the dataset (sample rate, turn / move
  thresholds, AFK skipping, entity limits) plus export filters that drop short episodes and thin out
  standing-still time, so the bot learns your play and not your breaks.
- 🧭 **Rich observations** – position, velocity, look direction, a 3×3×3 block area, nearby
  entities, health and food. **Advanced mode** adds the inventory, held item, armour, dropped items and
  every entity in range.
- 📦 **One-command export** – `/pymcbot export` writes `pymcbot/dataset.jsonl` and
  `/pymcbot train` prints the exact command to train on it.
- 🎮 **Drive your bots from the game** – `/pymcbot bot come`, `follow`, `do chop_tree`,
  `do mine_ore iron`, `auto on`… talk to the PyMC Bot app on your PC, so you can play while your
  bots gather, farm, fish, guard and sleep at night on their own.
- 🪶 **Lightweight** – no mixins, no server-side part, nothing changes how the game plays.

## How to use it

1. Install [Fabric Loader](https://fabricmc.net/use/) and [Fabric API](https://modrinth.com/mod/fabric-api),
   then drop the PyMC Bot jar for your Minecraft version into `mods/`.
2. Assign the hotkeys under *Options → Controls → Key Binds → PyMC Bot* (they start unbound so they
   never clash with other mods).
3. Press **Start / stop recording** (or `/pymcbot record start`) and play normally – 10 minutes is
   enough to see the bot copy your habits, an hour makes it clearly better.
4. `/pymcbot export`, then train with the PyMC Bot app:

   ```
   python -m pymc_bot train --dataset "<game folder>/pymcbot/dataset.jsonl"
   python -m pymc_bot run --think trained
   ```

## The PyMC Bot app

The free Python app that goes with this mod runs bots on offline-mode, premium (Microsoft) and free-host
(Aternos & co.) servers, from a web control panel:

- 🤖 **Plays like you** – the model trained on your recordings drives the bot.
- 🧠 **Or thinks like a human** – with a local LLM (Ollama) it keeps goals, remembers what worked and chats
  casually. Pick how it behaves: 😊 Friendly helper, 😎 Chill survivor, 💬 Social butterfly,
  🤝 Loyal companion, 🏗️ Builder, 🧭 Explorer, ⛏️ Miner, 🐣 New player, 🏆 Experienced pro or 🤫 Quiet player.
- 👥 **Many bots** to fill a server, with anti-AFK movement so none get kicked.
- 🛠️ **44 skills** – chop trees, mine ores, farm, fish, breed, craft, smelt, use chests, build a
  shelter, light up caves, guard, hunt, waypoints…
- 🌙 **Autopilot** – sleeps when night falls, eats, fights back, flees, swims up, puts out fires,
  equips better gear and places torches, without any AI.
- 📦 **No npm needed** – `python -m pymc_bot setup` installs everything (even Node.js) in one step.

See the [GitHub page](https://github.com/codero-sus/PyMC_Bot).

## Commands

| Command | Does |
| --- | --- |
| `/pymcbot record start` · `stop` · `toggle` | start / stop a recording |
| `/pymcbot pause` · `resume` | hold sampling without closing the episode |
| `/pymcbot export` | build `pymcbot/dataset.jsonl` from every episode |
| `/pymcbot settings` | open the settings page |
| `/pymcbot status` · `train` · `about` | state, the training command, credits |
| `/pymcbot bot status` · `list` · `skills` | your PyMC Bot bots (the app must be running) |
| `/pymcbot bot do <skill> [key=value …]` | run a skill, e.g. `do chop_tree count=12` |
| `/pymcbot bot come` · `follow` · `stop` · `home` · `say <text>` | quick orders |
| `/pymcbot bot auto on` · `off` | the autopilot (auto-sleep at night, eat, fight back…) |
| `/pymcbot bot panel <url>` | the app runs on another PC |

`/pymc …` works as a short alias. `/pymcbot help` lists everything (sample rate, advanced mode, HUD
corner, auto-record, AFK skipping…).

## Privacy

Everything stays on your computer, in `<game folder>/pymcbot/`. Recording makes **no network
connections**, sends nothing anywhere and has no server component — it works in single-player and on
any server, including offline-mode ones. Recordings from versions before 0.7 (folder `pymc-playtime/`)
are moved to the new folder automatically. The only connection the mod ever opens is when you type
`/pymcbot bot …`: it talks to *your own* PyMC Bot app (`127.0.0.1:8000` unless you change it).

## Requirements

| | |
| --- | --- |
| Loader | Fabric |
| Required | [Fabric API](https://modrinth.com/mod/fabric-api) |
| Optional | [Mod Menu](https://modrinth.com/mod/modmenu) (adds the Configure button) |
| Side | client only (servers don't need it) |
| Java | 21 for 1.21.x, 25 for 26.x (the launcher handles this) |

## Links & credits

- Source, issues and the PyMC Bot app: <https://github.com/codero-sus/PyMC_Bot>
- Made with ❤️ by [codero-sus](https://github.com/codero-sus) · <https://the-codero.web.app/>
- License: [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) – free for personal, non-commercial use. Please don't re-upload or sell it.
