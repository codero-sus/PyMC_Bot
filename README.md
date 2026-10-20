# PyMC_Bot

[![Author: codero-sus](https://img.shields.io/badge/author-codero--sus-22c55e?logo=github)](https://github.com/codero-sus)
[![Website](https://img.shields.io/badge/website-the--codero.web.app-0ea5e9)](https://the-codero.web.app/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial-a78bfa)](LICENSE)

> **Made with ❤️ by [codero-sus](https://github.com/codero-sus)** · [the-codero.web.app](https://the-codero.web.app/)

A **Python-controlled Minecraft player bot** it joins a (cracked / offline-mode) server as a normal
player, you configure and steer it from a **web control panel served by Uvicorn**, and it can hand the
wheel to a **local LLM** – an Ollama model, or any model hosted by
[Cortex LLMHoster](docs/CORTEX.md) – that decides how it plays (walking, mining, following players, chatting).

**Premium accounts and whole crowds are first-class:** add a player with a Microsoft account
(device-code login, token cached per account), or fill the server with up to 200 bots at once.

Everything is Python: configuration, the movement engine, the AI loop, the REST/WebSocket API and the web UI.
The only JavaScript is a ~300 line adapter that speaks the Minecraft protocol through
[mineflayer](https://github.com/PrismarineJS/mineflayer) (there is no maintained Python client for modern
Minecraft versions), and it is driven purely by JSON from Python.

```
┌──────────────────────────────┐        ┌───────────────────────────────────────────┐
│  Browser (config page)       │        │  Python  (pymc_bot/)                      │
│  HTML/CSS/vanilla JS, no     │◄──────►│  FastAPI + Uvicorn      server.py         │
│  build step                  │  REST  │  MinecraftBot           bot.py            │
│  REST + WebSocket /ws        │   WS   │  AgentLoop (LLM/trained) agent.py         │
└──────────────────────────────┘        │  Backends: simulated    backends.py       │
                                        └───────────────┬───────────────────────────┘
                                                        │ newline-delimited JSON (stdin/stdout)
                                        ┌───────────────▼───────────────────────────┐
                                        │  Node  mineflayer  →  Minecraft server     │
                                        │  (cracked/offline auth supported)          │
                                        └───────────────────────────────────────────┘
```

* **Works without Minecraft** – a built-in simulated world (`backend: simulated`, reporting players, a mob, an animal, a dropped
  stack and a starter inventory, so trained models have entity and item inputs) lets you try the whole panel, movement and AI loop on any machine, and it is what the test suite runs against.
* **Works without an LLM** – if Ollama/Cortex is disabled or unreachable the bot keeps playing with a
  built-in heuristic policy, so it never just stands there.
* **Lives like a real player** – it handles its inventory the way people do: presses hotbar keys, pulls items
  from the backpack into the hotbar and back, sorts the hotbar, puts on the best armour, holds the right tool
  before mining and its best weapon before a fight, eats sensibly, picks up drops, hands items to players
  who ask, sleeps at night, flees when hurt and waves or crouch-greets people ([details](#lives-like-a-real-player)).
* **Knows Minecraft, and looks up what it doesn't** – many models barely know the game, so the LLM gets
  practical facts about the mobs around it (creepers explode, don't look at endermen, piglins want gold…)
  from a **JSON mob database** (hostility, health, damage to players and mobs, spawn time, the version each
  mob was added in – even 26.2's sulfur cube, which mineflayer doesn't know yet), and can **search the web**
  (Minecraft Wiki by default, no key) before it acts ([details](#minecraft-knowledge-and-web-search)).
* **Two local LLM hosts** – Ollama, or [Cortex LLMHoster](docs/CORTEX.md) (OpenAI-compatible, optional
  bearer key read from the environment and never saved).
* **Plays and talks like a human** – pick a behaviour (helper, companion, explorer, pro, …); the LLM gets a
  detailed, hidden player prompt for it, a goal it
  sticks to, short-term memory and plain-word perception; it chats in short casual lines with typing pauses
  and no spam ([details](#plays-and-talks-like-a-human)).
* **One bot or a hundred** – the *Players* panel spawns offline bots from a name pattern, adds premium
  players one email at a time, and can keep the crowd alive (rejoin on kick, optional idle chatter).
* **Anti-AFK built in** – every bot periodically walks a step, hops, crouches, swings its arm, scrolls its
  hotbar and turns its head in smooth, randomised bursts, so servers do not kick them for standing still. It never fights the AI
  or your manual commands, and it scales to the whole fleet from one thread.
* **A panel that is nice to live in** – four themes (midnight, daylight, nether, end), a command palette
  (<kbd>Ctrl</kbd>+<kbd>K</kbd>), an at-a-glance overview strip, a section bar that follows your scroll,
  collapsible cards, toasts for everything you do, an animated minimap with radar sweep, and a little
  confetti when the bot joins or a training run finishes. Every animation can be switched off (✨ / <kbd>Alt</kbd>+<kbd>M</kbd>,
  and the OS "reduce motion" setting is honoured).
* **Trains on *your* playtime** – a Fabric mod ([`./mod`](mod/README.md), every Minecraft from **1.21.1 to 26.3**) records how you play, `pymc_bot train`
  turns that into a checkpointed player model, and `think trained` runs the bot with it. No Ollama, no GPU,
  no cloud: the model learns your movements, your turns and your habits and reproduces them in game.

---

## Quickstart

```bash
git clone https://github.com/codero-sus/PyMC_Bot.git
cd PyMC_Bot

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

npm install            # mineflayer bridge (only for real servers, needs Node >= 22)

python -m pymc_bot serve          # or: python -m pymc_bot      (serve is the default)
# -> http://localhost:8000
```

In the panel:

1. **Minecraft server** – host, port, username (max 16 chars), version (`auto` sniffs it), auth
   `offline (cracked)`, backend `auto`, then **Save & connect**.
2. **AI brain** – tick *Let Ollama decide the bot's actions*, pick a model, **Start AI**.
3. **Manual control** – chat, server commands and single JSON actions that override the AI.

### Trying it without a Minecraft server

Set backend to `simulated` in the panel (or run `python -m pymc_bot run --backend simulated`).
The bot walks around a small pretend world, "mines" blocks, chats with two NPCs and answers you in the
event stream – a full dress rehearsal with no server needed.

### Connecting to a cracked server

Your server must allow offline-mode logins (`online-mode=false` in `server.properties`; this is what "cracked"
means). Then set host/port/username with auth `offline`.

### Premium players (Microsoft accounts)

Online-mode servers need a real account. Set **Auth → microsoft** and put the account's **email** in the
username field (the in-game name is whatever that account owns):

1. Press **Save & connect**. The event stream prints a device code:
   `Premium login: open https://www.microsoft.com/link and enter the code ABCD-1234 (account: you@example.com)`.
2. Enter it in the browser once. The refresh token is cached in `.pymc_profiles/<account>/`, so every later
   join is silent — including after a restart, a kick or a `--reconnect`.

Notes: each premium player needs its **own** Microsoft account; tokens are never shared between accounts.
Accounts must own Minecraft: Java Edition (Bedrock/console-only accounts fail with
*"does the account own minecraft?"*, which the panel shows verbatim).

### Fill the server with players (fleet)

The **Players** card adds more bots on top of the main one — all against the same server:

| Control | What it does |
| --- | --- |
| **Players to add** + **Name pattern** | spawn N bots named like `PyMC_Bot_1 … PyMC_Bot_N` (`{n}` = index, `{name}` = main bot name) |
| **Fleet auth** | `offline` (any name) or `microsoft` (premium, one email per bot) |
| **Fleet AI** | `heuristic` (cheap, default), `off` (just stand there) or `ollama` (uses the model) |
| **Gap between joins** | stagger connections (1–2s) so the server does not rate-limit the stampede |
| **Add premium player** | adds a single premium player by email |
| **Broadcast** | make every connected bot say the same thing |
| **idle chatter** | the bots occasionally chat (off by default) |
| **restore these players on restart** | remember the fleet in the config file and bring it back on startup |

Every player is listed with its auth mode, state, position and per-bot **use / start / stop / drop**
buttons. The dropdown in the top bar (or **use**) decides which player the rest of the panel controls —
manual chat, commands and actions then apply to that player, and `bot: "all"` applies to everyone.

`fleet.max_bots` (default 25, max 200) caps the whole instance. A join batch runs in the background, so the
panel stays responsive while twenty players connect.

Headless equivalent:

```bash
python -m pymc_bot run --populate 20            # 20 offline players join the configured server
python -m pymc_bot run --premium you@example.com  # add a premium player (prints the device code)
```

### Anti-AFK (never get kicked for standing still)

Enabled by default for **every** bot — the main one and the whole fleet. When a bot has been idle for a
while (8–25 s by default, randomised), it performs a short burst of human-ish habits:

| Habit | What it looks like |
| --- | --- |
| `look` | head turns over several small steps (never a snap), occasionally glancing up or down |
| `look_at_player` | turns towards a nearby player for a moment, then looks away |
| `stroll` | walks one to three blocks somewhere else, then stops |
| `strafe` | a short sideways step while looking slightly off-axis |
| `hop` | one or two jumps in place |
| `crouch` | a brief sneak |
| `swing` | swings the arm a couple of times (some plugins only check animation) |
| `hotbar` | scrolls a slot or two along the hotbar, usually back to the item it was holding |

Two or three habits are often chained, the same habit never repeats back-to-back, and the intervals are
jittered — so the pattern does not look like a metronome. The keeper only touches bots that are **not** busy:
AI decisions, follows, mining and your manual commands always take priority.

```jsonc
"antiafk": {
  "enabled": true,
  "interval_min": 8.0,        // idle seconds before a burst (randomised)
  "interval_max": 25.0,
  "look": true, "look_at_players": true, "walk": true, "jump": true, "sneak": true, "swing": true, "hotbar": true,
  "max_walk_distance": 3.0,
  "combo_probability": 0.35,  // chance of chaining 2-3 habits in one burst
  "verbose": false            // log every burst to the event stream
}
```

Controls live in the **Players → Anti-AFK** row (toggle, idle window, which habits, **Poke all now**), and the
player table shows each bot's idle time and how many anti-AFK bursts it has done. From the API:
`GET /api/antiafk/status` and `POST /api/antiafk/poke {"bot": "all"}`.

> Only connect the bot to servers you own or are allowed to use it on. Many servers forbid client bots;
> check their rules first. Attacking other players is **disabled by default**.

---

## Updating

Run the updater from the project folder - it only needs Python, nothing has to be installed first:

```bash
./updater.sh              # macOS / Linux
updater                   # Windows (cmd or PowerShell: .\updater.cmd), or double-click updater.cmd
python updater.py         # anywhere
```

| Command | What it does |
| --- | --- |
| `updater --check` | shows the new commits / mod release, changes nothing |
| `updater` | updates PyMC_Bot; reinstalls Python (and npm) packages **only if** `requirements*.txt` / `package.json` changed |
| `updater --mod` | also updates the recorder mod in your `.minecraft/mods` folder |
| `updater --mod-only` | only the mod |
| `updater --mod-only --mc 1.21.4` | installs the mod for 1.21.4 when it is not there yet |
| `updater --help` | all options (`--mods-dir`, `--pre`, `--reinstall`, `--deps`, `--no-deps`, `--branch`) |

* **PyMC_Bot itself – always a `.zip`, never git:** the updater asks GitHub for the newest commit
  of this build's *update channel*, downloads exactly that commit as a source `.zip`, checks it
  (complete archive, commit id inside the zip matches) and extracts it over the folder.
  `pymc_bot_config.json`, `.venv`, `models/`, recordings, `.git` and every other file of yours are
  left alone; project files you changed are backed up to `.pymc_backup/<time>/` first, and files the
  project deleted are removed on later updates – unless you edited them. Works the same for a
  downloaded ZIP and a git clone (git is never run; a clone just shows the new files in `git status`).
  The channel is built into `updater.py` and never printed; `--branch NAME` follows another branch
  instead. Out of GitHub API calls? The updater downloads the zip and reads the commit id from it.
* **Mod:** finds `pymcbot-<version>+<mc>.jar` (or a pre-0.6 `pymcplaytime-*.jar`, which it replaces) in the mods folder (`%APPDATA%\.minecraft\mods`,
  `~/Library/Application Support/minecraft/mods`, `~/.minecraft/mods`, or `--mods-dir`),
  downloads the jar for the same Minecraft version from the newest
  [release](https://github.com/codero-sus/PyMC_Bot/releases), verifies its size, SHA-256 and
  `fabric.mod.json`, and keeps the old jar as `*.jar.old`. Close Minecraft first on Windows.
  Stable releases are preferred; `--pre` opts into pre-releases (used automatically while no stable
  release exists).
* Packages go into `.venv` when it exists; otherwise the updater prints the `pip` command to run.
  Set `GITHUB_TOKEN` if you hit GitHub's 60-requests-per-hour limit. Exit code `0` = OK, `1` = error.

Restart the panel / bot afterwards.

## Train a player model on your own playtime

The bot can learn **how you play** and then play like that itself. Three steps:

**1. Record** – download the **PyMC Bot mod** jar (`pymcbot-<version>+<mc>.jar`) for your Minecraft version (**1.21.1 – 26.3**) from the
[GitHub release](https://github.com/codero-sus/PyMC_Bot/releases) (or build it from [`./mod`](mod/README.md): with JDK 25 installed,
`cd mod && ./gradlew build -Pmc=<your version>` – full steps in
[Compiling the mod](mod/README.md#compiling-the-mod)), drop it in `.minecraft/mods/` with
[Fabric API](https://modrinth.com/mod/fabric-api), then in game:

```text
/pymcbot record start      # or your hotkey; play normally for 10-60 minutes
/pymcbot record stop
/pymcbot export            # -> <gameDir>/pymcbot/dataset.jsonl
```

A **REC** badge in the corner shows the run clock and sample count while you record. Everything
else is on the **PyMC Bot settings page** – `/pymcbot settings`, a hotkey, or Mod Menu's
*Configure* button: tabs for the recorder controls, recording options, advanced entity + item
limits, a **Power** tab with sliders and export filters for nicer datasets, the HUD badge (with a
live preview) and About; hover any button for help
([details](mod/README.md#pymc-bot-settings-page)). `/pymc` still works as a short alias.

The mod samples your playtime 20×/s — position, velocity, view, posture, hotbar, health/food, the 3×3×3
block neighbourhood, nearby entities, and the action you were performing
(`forward`, `back`, `left`, `right`, `jump`, `sneak`, `look`, `attack`, `use`, `hold`, `move`, `none`).
`/pymcbot advanced on` additionally records **entities, items, the held item, armour and dropped stacks**
for [advanced training](#advanced-training-entities-and-items).

**2. Train** – *checkpointing itself as it goes*:

```bash
python -m pymc_bot train --dataset "<gameDir>/pymcbot/dataset.jsonl" --steps 2000
python -m pymc_bot train --resume --run-name playtime-dataset        # continue where it left off
python -m pymc_bot models                                            # list checkpoints + metrics
```

```
models/playtime-dataset/
├── latest.npz            weights (+ optimiser state) - or latest.pt for --engine transformer
├── checkpoints/step-0000200.npz
├── model.json            the card the panel lists (metrics, dataset digest, action vocabulary)
├── trainer_state.json    step/epoch/RNG/best-val -> exactly resumable
└── metrics.jsonl, metrics.csv
```

Training straight from the episode folder instead of an exported `dataset.jsonl`? `--min-episode 30`
and `--idle-keep 30` apply the same filters as the mod's Power tab (drop episodes under 30 s, keep
30 % of standing-still samples).

No recording yet? `python -m pymc_bot train --simulate 10` fabricates a believable session so you can try
the whole pipeline (and the panel's **Demo: simulate 5 min** button) without launching Minecraft.

**3. Run it** – either from the shell or the panel:

```bash
python -m pymc_bot run --think trained --run-name playtime-dataset --backend simulated
```

```jsonc
{ "agent": { "mode": "trained" }, "training": { "active_run": "playtime-dataset" } }
```

The policy predicts a burst of play (default 0.4 s) and applies the same low-level controls the mod
recorded, including the predicted head turn — so the bot walks, strafes, hops and looks around the way you
did, and the anti-AFK keeper treats it as activity. It obeys the same permission gates as the other
brains (looking and idling always; moving, mining/using and attacking only when their switches are on),
so a model trained on your playtime can never do something you turned off. Keep Ollama for chat and
high-level planning; the trained brain is a separate, local, dependency-light model (`--engine mlp`
needs nothing but NumPy; `--engine transformer` adds a small attention model). Full details, metrics and troubleshooting:
[docs/TRAINING.md](docs/TRAINING.md).

### Advanced training (entities + items)

The basic encoder describes the world in coarse buckets ("hostile", "passive", "resource block").
Advanced training **learns the entity and item words your playtime actually contains** and feeds each one
to the model with its distance and where it sits relative to your view — so "a creeper four blocks away
while holding a sword" is a different situation from "a cow across the field while holding a pickaxe":

```text
/pymcbot advanced on        # in game, before recording: entities + items + drops are written too
/pymcbot record start  ...  /pymcbot record stop  ...  /pymcbot export
```

```bash
python -m pymc_bot train --dataset "<gameDir>/pymcbot/dataset.jsonl" --advanced --steps 2000
python -m pymc_bot train --dataset ... --advanced --entity-slots 32 --item-slots 48   # bigger vocabularies
python -m pymc_bot train --dataset ... --inspect      # vocabulary + coverage before you train
```

The learned vocabulary is stored in the checkpoint and in `model.json`, so the bot encodes the live world
into exactly the same slots (layout v2; `feature_version` guards the hand-off). In the panel it is the
**Advanced training** box in the *Player model* card: tick it, train, and the status line shows what the
dataset offers and which words the newest model learned. Details:
[docs/TRAINING.md](docs/TRAINING.md#advanced-training-entities-and-items).

---

## The Ollama brain

```bash
# real model
ollama pull llama3.2
ollama serve                # usually already running on http://127.0.0.1:11434
```

Then in the panel: enable **AI brain**, set the model name, **Start AI**. Every
`ollama.decision_interval` seconds (default 6s, ±25% so it doesn't act like a metronome) the bot describes
the situation and asks for one action:

```json
{"thought": "need wood before tools", "goal": "get wood for tools", "action": "mine", "block": "oak_log"}
```

### Plays and talks like a human

The LLM brain (Ollama or Cortex) gets a detailed system prompt ([`pymc_bot/mind.py`](pymc_bot/mind.py))
that makes it act like a real player at a keyboard, not an assistant:

* **Thinks like a player** – every turn it checks danger, health, food and the time of day, answers people
  first, then keeps working on a **goal** for several turns (wood → tools → food → shelter before night)
  instead of picking random actions. A short private `thought` is shown in the log (`💭 ...`).
* **Sees like a player** – the world is described in plain words: "zombie 6 m north-east", "Steve (4 m east)",
  "Food 5/20 (hungry - eat soon)", "Waiting for your answer: <Steve> wanna mine together?" – with the exact
  JSON appended for coordinates.
* **Remembers** – its goal, the last `agent.memory_turns` actions with whether they worked (so it stops
  retrying a failing path), its last thoughts and what it already said.
* **Chats like a person** – short casual lines in the other player's language, no markdown, no "as an AI"
  speeches, never the same line twice; a typing pause before sending and a `chat_cooldown` for messages
  nobody asked for (replies skip it). If someone sincerely asks whether it is a bot, it says so.
* **Behaves the way you pick** – in the panel's **Bot behaviour** box choose one card; the prompt behind it
  stays hidden:

  | Behaviour | What it does |
  | --- | --- |
  | 😊 Friendly helper (default) | plays survival, says hi, helps anyone who asks |
  | 😎 Chill survivor | does its own thing at its own pace, few words |
  | 💬 Social butterfly | loves to chat, hangs out with players, joins in |
  | 🤝 Loyal companion | sticks with you, follows you around, watches your back |
  | 🏗️ Builder | gathers logs and stone, talks about builds |
  | 🧭 Explorer | wanders far, checks out caves and new places |
  | ⛏️ Miner | mines ores, upgrades tools, shares coal and iron |
  | 🐣 New player | still learning, asks questions, a bit clumsy |
  | 🏆 Experienced pro | efficient and confident, gives short useful tips |
  | 🤫 Quiet player | rarely talks, just plays near others |

  Optional *About the character* notes ("from Delhi, mixes Hindi and English") are added on top.

In the config the choice is `agent.persona` (`friendly`, `chill`, `social`, `companion`, `builder`,
`explorer`, `miner`, `newbie`, `pro`, `quiet`) plus `agent.persona_notes`. Advanced: a non-empty
`ollama.system_prompt` / `cortex.system_prompt` in the config file replaces the built-in prompt entirely
(configs that still hold the old default switch to the new prompt automatically). Small models work; 7–8B
models (e.g. `llama3.1:8b`, `qwen2.5:7b`) follow the behaviour and goals noticeably better.

Accepted actions: `say`, `goto`, `follow`, `wander`, `mine`, `attack` (gated), `jump`, `eat`,
`look_at_player`, `stop`, `wait`, plus the inventory and life actions below. The full list, including which permissions gate them, is in
[docs/API.md](docs/API.md#actions). Malformed model output is caught, logged and retried with the
heuristic policy instead of crashing the loop.

### Lives like a real player

The bot sees its inventory the way you do: **hotbar slots 1-9** (one is selected, so that item is in its
hand), the **backpack** above it, **armour** and the **off-hand**. Every brain (LLM, heuristic and you in
the panel) can use these actions:

| Action | Example | What happens |
| --- | --- | --- |
| `select_slot` | `{"action":"select_slot","slot":3}` | presses hotbar key 3 |
| `hold_item` | `{"action":"hold_item","item":"pickaxe"}` | takes the best pickaxe in hand, pulling it into the hotbar if it is in the backpack |
| `to_hotbar` | `{"action":"to_hotbar","item":"cobblestone","slot":5}` | backpack → hotbar (slot optional: the first free one) |
| `to_inventory` | `{"action":"to_inventory","slot":7}` | hotbar → backpack (or `"item":"dirt"`; no slot = the selected one) |
| `organize` | `{"action":"organize"}` | sorts the hotbar like a tidy player: 1 weapon, 2 pickaxe, 3 axe, 4 shovel, 5 blocks, 6 torches, 7 bow, 9 food – one click at a time |
| `wear_armor` | `{"action":"wear_armor"}` | puts on every piece that beats what it is wearing |
| `equip` / `unequip` | `{"action":"equip","item":"shield","to":"offhand"}` | armour goes to its slot; `unequip` takes `head`/`chest`/`legs`/`feet`/`offhand` off |
| `swap_hands` | `{"action":"swap_hands"}` | the F key |
| `drop` / `give` | `{"action":"give","player":"Steve","item":"bread","count":2}` | `give` walks over, looks at the player and tosses it |
| `fight` / `flee` | `{"action":"fight","mob":"zombie"}` | takes out the best weapon and fights a hostile mob (never a warden/enderman); `flee` sprints away |
| `collect` | `{"action":"collect"}` | walks over dropped items nearby |
| `sleep` | `{"action":"sleep"}` | finds a bed within 32 blocks (night only) |
| `place` / `use_item` | `{"action":"place","block":"blocks"}` | places a block next to it / right-clicks the held item |
| `emote` | `{"action":"emote","kind":"crouch"}` | `wave`, `crouch` (the sneak greeting), `nod`, `shake`, `spin` |
| `look_around` | `{"action":"look_around"}` | a few natural glances |

`item` can be an exact id (`iron_pickaxe`) or a word: `sword`, `weapon`, `pickaxe`, `axe`, `shovel`,
`food`, `blocks`. On its own it also **switches to the right tool before mining** (pickaxe for stone, axe for
logs, shovel for dirt…) and **eats sensibly** (no steak on an almost full bar, rotten flesh only when
starving, golden apples only when badly hurt).

The built-in heuristic brain uses all of this in the order a player would: flee when badly hurt, eat, fight a
mob that comes close, answer people and hand over what they ask for ("can i have some bread?"), put on better
armour, tidy a messy hotbar, pick up drops, sleep at night, greet newcomers once with a wave or crouch – then
follow friends, mine, explore and fidget with the hotbar. The LLM sees a player's view of the inventory
(`Hotbar (* = in your hand): [1*] iron_sword  [2] iron_pickaxe …`, armour, backpack and notes like
"better armour in your inventory").

Permissions (`agent.*`, panel **AI permissions**): `allow_inventory` (on), `allow_mob_fighting` (on) and
`allow_building` (**off** – placing blocks and right-clicking items, so an AI does not litter a server).

### Minecraft knowledge and web search

Plenty of models don't really know Minecraft: they walk up to creepers, stare at endermen or attack piglins.
The Ollama / Cortex brain gets help in two ways:

* **Mob database.** [`pymc_bot/data/mobs.json`](pymc_bot/data/mobs.json) has every vanilla Java mob (89) with
  its stats. Every turn the prompt carries one line about each mob type nearby, e.g.
  *creeper (hostile, 20 HP, up to 43 dmg, night/dark): walks up silently, hisses and EXPLODES about 1.5 s
  later… hit it and step back*. Questions about one mob (*is the sulphur cube hostile?*, *how much damage
  does a warden do?*, *does the sulfur cube exist in 1.21.4?*) are answered from the database offline,
  before any web request. Details are [below](#the-mob-database).
* **Web search.** The model can use `{"action": "search", "query": "how to tame a wolf"}` whenever it is unsure
  about a mob, block, recipe or mechanic. The answer goes straight back into the prompt (*WHAT YOU LOOKED UP*)
  and the model picks its real action **in the same turn**. The answer stays in its memory for the next turns too.
  Mobs it has no entry for, like modded ones, are looked up automatically in the background.

| Provider (`search.provider`) | Key | Notes |
| --- | --- | --- |
| `minecraft_wiki` (default) | none | the [Minecraft Wiki](https://minecraft.wiki)'s public API: the page intro plus the section that fits the question (Taming, Drops, Behavior…) |
| `duckduckgo` | none | instant answers; also the fallback when the main provider finds nothing (`search.fallback`) |
| `searxng` | none | your own [SearXNG](https://docs.searxng.org) instance (`search.searxng_url`, JSON format enabled) |
| `brave` | `BRAVE_API_KEY` env var | Brave Search API; the key is read from the environment and never saved |

Answers are cached in `knowledge_cache.json` next to the config file for 30 days (`search.cache_days`), so
repeated questions are free and instant. The number of real web requests is capped at 30 per hour
(`search.max_per_hour`). `search.max_per_turn` (1) sets how many lookups can happen before acting; with 0,
the answer is used on the next turn instead. Turn it all off with **Allow web search** in the panel's
*Minecraft knowledge* box (`agent.allow_web_search` / `search.enabled`); the mob database stays.
Questions are sent to the provider you pick. They contain only the short question, never chat logs or player names.

Try it like the AI would:

```bash
python -m pymc_bot search how to tame a wolf
python -m pymc_bot search "what drops blaze rods" --provider duckduckgo --json
```

The panel has the same **Look up** box, and the API is `POST /api/knowledge/search {"query": "..."}`.

#### The mob database

One entry per mob in [`pymc_bot/data/mobs.json`](pymc_bot/data/mobs.json). The values are Java Edition
figures from the [Minecraft Wiki](https://minecraft.wiki) infoboxes:

| Field | Meaning |
| --- | --- |
| `category`, `hostile`, `hostile_when` | `hostile` / `neutral` / `passive` / `boss`; whether it attacks on sight, and what makes a neutral one attack |
| `health`, `health_note`, `mob_type` | health points (2 HP = 1 heart), sizes/variants, `undead` / `arthropod` / `illager` / `aquatic` (Smite, Bane of Arthropods…) |
| `attacks[]` | `type`, damage to a player without armour on `easy` / `normal` / `hard` (a `[min, max]` pair for ranges), `effect`, `note`. Difficulty only scales damage to **players**, so damage to **mobs** is the Normal value (`to_mobs` when it differs – a returned ghast fireball deals 1000 to a ghast) |
| `spawn` | `when` (`dark` = light 0: night on the surface or any time in caves, `night`, `day`, `any`, `structure`, `summoned`, `built`, `special`, `event`), `where`, `dimensions`, `light`, `group` |
| `added_in` | first Java release with the mob (`Classic` … `1.21.11`, `26.2`). Mobs newer than mineflayer's last version (26.1) are flagged `mineflayer_supported: false` in the API/CLI: a bot sees them as an unknown entity, so the prompt lists the newer mobs the server version can have |
| `drops`, `xp`, `tame`, `breed`, `weaknesses`, `notes`, `aliases`, `fact` | the rest; `aliases` covers old names (*zombie pigman*), British spelling (*sulphur cube*) and baby names (*gurgle*) |

```bash
python -m pymc_bot mob sulphur cube              # everything about one mob
python -m pymc_bot mob warden --json
python -m pymc_bot mob sulfur cube --version 1.21.11   # "In 1.21.11: no, it does not exist yet"
python -m pymc_bot mob --list --category hostile --dimension nether
python -m pymc_bot mob --list --since 1.21       # mobs added in 1.21 or later
```

The API is `GET /api/mobs` (filters) and `GET /api/mobs/{name}`. Untick **answer mob questions from the mob
database** (`search.use_mob_database`) to send every question to the web instead. You can also add modded mobs
to the JSON file: `python -m pytest tests/test_mobs.py` checks the format.
They gate the AI only; the **Inventory & life** buttons and the clickable hotbar in the panel always work.

### Demo brain (no GPU, no download)

For demos, tests and CI there is an **Ollama-compatible stub** – clearly *not* a language model, just a
rule-based responder that answers the same `/api/chat`, `/api/generate` and `/api/tags` endpoints (and the
OpenAI-style `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, so it can also stand in for Cortex):

```bash
python -m pymc_bot stub            # listens on http://127.0.0.1:11434
```

The panel shows a `demo brain (stub)` badge whenever the endpoint identifies itself as the stub, so it is
never mistaken for a real model.

### Your own Minecraft model (Kaggle T4)

Train a small model that is **the bot's brain and a Minecraft expert in one**. It plays with the same JSON
decisions, chats like a player and answers mob, crafting and mechanics questions without a web search. Data
comes from the mob database, checked crafting facts, thousands of game situations, Minecraft Wiki pages,
turns the bot recorded, your playtime and (optionally) a bigger teacher model you run:

```bash
python -m pymc_bot llm build          # llm_data/: train.jsonl, eval.jsonl, train_minecraft_llm.py
#   upload llm_data to Kaggle, run training/llm/PyMC_Minecraft_LLM_Kaggle.ipynb on GPU T4 x2 (~1-2 h)
python -m pymc_bot llm import pymc-minecraft     # the downloaded output -> Ollama, bot switched over
python -m pymc_bot llm eval --model pymc-minecraft
```

Sizes: `qwen3-4b` (default), `qwen3-8b`, `qwen3-14b`, `llama-3.2-3b`, `qwen3.5-4b` - see `python -m pymc_bot
llm presets`. Full guide: [docs/MINECRAFT_LLM.md](docs/MINECRAFT_LLM.md).

## Cortex LLMHoster brain

[Cortex LLMHoster](https://github.com/codero-sus/Cortex_LLMHoster) serves local models through an
OpenAI-compatible API on `http://127.0.0.1:8624`. Select **cortex** as the brain (or enable it so `auto`
uses it when Ollama is off), export the same `CORTEX_API_KEY` Cortex was started with, and the bot sends
its world state to `POST /v1/chat/completions` every `cortex.decision_interval` seconds:

```bash
export CORTEX_API_KEY=...            # read from the environment, never written to the config
python -m pymc_bot doctor            # cortex : http://127.0.0.1:8624/v1 -> OK (1 model(s): qwen-local; ...)
python -m pymc_bot run --think cortex --cortex-model qwen-local
```

In the panel: **AI brain → Cortex LLMHoster → Check Cortex → Use Cortex as brain**. Fleet bots can use
`cortex` too. Same action contract, same permission gates and the same heuristic fallback as Ollama.
Setup, a weight-free demo (the stub standing in for Cortex) and troubleshooting:
[docs/CORTEX.md](docs/CORTEX.md). Cortex is a separate program under its own personal, non-commercial
license; PyMC_Bot only talks to it over HTTP.

---

## The control panel

`python -m pymc_bot serve` → <http://127.0.0.1:8000>. Besides the cards for the server, players, brain,
player model, live view, manual control, events and raw config, the panel has:

| | |
| --- | --- |
| **Overview strip** | connection, brain, players online, decisions, distance walked and the active player model, with live counters |
| **Command palette** | <kbd>Ctrl</kbd>/<kbd>⌘</kbd>+<kbd>K</kbd> (or <kbd>/</kbd>): jump to any card, connect, start the AI, train, poke the fleet, switch theme, open the docs |
| **Themes** | 🎨 or <kbd>Alt</kbd>+<kbd>T</kbd>: midnight · daylight · nether · end, remembered per browser |
| **Motion** | ✨ or <kbd>Alt</kbd>+<kbd>M</kbd> turns every animation and the floating-block background off; "reduce motion" in the OS is respected by default |
| **Cards** | collapse with ▾ or a double-click on the title (remembered), cursor spotlight, staggered entrance |
| **Feedback** | toasts for every panel action and connection change, spinners on buttons while a request runs, a damage flash and low-health pulse on the meters, confirmation before destructive actions |
| **Live view** | the minimap eases between status frames, scrolls its grid with the bot and draws a radar sweep |
| **About** | ⓘ opens the credits (also `GET /api/about`) |

---

## Aternos & other free hosts

PyMC_Bot joins servers on **Aternos, exaroton, Minehut, FalixNodes**, and servers shared through
**playit.gg, ngrok or e4mc** tunnels, as well as servers on your own PC.

- **Address:** type `name.aternos.me` and leave the port at 25565. The address has an SRV record that
  the bot follows like the real game does. You can also paste `host:port` (a Dyn IP or a tunnel
  address) into **Host**, and the port fills itself in.
- **Check server** (in the panel) or `python -m pymc_bot probe` pings the server without joining. It
  shows online, asleep or starting, the version, the players and where the SRV record points.
- **Sleeping servers:** free hosts stop empty servers, and while asleep the address answers with a
  placeholder ("● Offline"). The bot now reports it plainly and tells you to start the server from the
  host's dashboard. With **Server asleep? → wait for it to start** (`wait_for_server`), it checks every
  15 s and joins once someone has started it. PyMC_Bot never starts a server itself.
- **Cracked bots** need the host's cracked / online-mode-off setting. On Aternos that's
  *Options → Cracked*. Otherwise the kick message says which switch to flip. Minehut only lets premium
  (Microsoft) accounts in.
- **Aternos rules:** Aternos does not allow bots to keep an empty server online. On Aternos the bots
  therefore leave once no real player has been online for `alone_minutes` (5 by default), and they
  don't re-join on their own; press Connect when people are back. The bots play while you and your
  friends are on, which is what Aternos allows. Set `leave_when_alone: "always"` to use the same rule
  on any server.
- **New Minecraft versions:** if the server runs a version the bot library doesn't support yet
  (currently anything newer than 26.1), see *Troubleshooting*. ViaVersion + ViaBackwards (Aternos
  offers both as plugins) let the bot in.

## Command line

| Command | What it does |
| --- | --- |
| `python -m pymc_bot serve [--host 0.0.0.0] [--port 8000] [--autostart] [--auto-agent]` | run the web config panel (Uvicorn) |
| `python -m pymc_bot run [--backend node\|simulated] [--server host:port] [--seconds N] [--no-ai] [--populate N] [--premium EMAIL] [--think auto\|heuristic\|ollama\|cortex\|trained] [--cortex-url URL] [--cortex-model ID]` | headless bot: connect and play (optionally populate the server), logs to stdout |
| `python -m pymc_bot train [--dataset FILE] [--steps N] [--engine mlp\|transformer] [--advanced [--entity-slots N] [--item-slots N]] [--min-episode S] [--idle-keep PCT] [--resume] [--simulate MINUTES] [--ollama-model NAME]` | train a player model on recorded playtime (self-checkpointing, resumable); `--advanced` learns entity + item vocabularies |
| `python -m pymc_bot models [--models-dir models] [--json]` | list trained checkpoints with their metrics |
| `python -m pymc_bot action '{"action":"wander"}' [--url http://127.0.0.1:8000]` | poke a running panel from the shell |
| `python -m pymc_bot doctor` | environment check (Node, mineflayer, Minecraft range, Ollama, Cortex, config, server ping, hosting tips) |
| `python -m pymc_bot probe [host[:port]] [--json]` | ping a server without joining: online / asleep / starting, version, players, SRV target, tips for Aternos & co (exit 0 online, 3 asleep, 1 unreachable) |
| `python -m pymc_bot mob <name> [--version V] [--json]` / `mob --list [--category C] [--dimension D] [--since V]` | the built-in mob database (offline) |
| `python -m pymc_bot llm build [--no-wiki] [--turns N] [--teacher ollama\|cortex] [--playtime PATH]` | training data for your own Minecraft model ([guide](docs/MINECRAFT_LLM.md)) |
| `python -m pymc_bot llm presets` / `llm import <folder\|gguf> [--name N]` / `llm eval [--model M]` | model sizes; add the Kaggle-trained model to Ollama; score a model |
| `python -m pymc_bot stub [--port 11434]` | the Ollama/OpenAI-compatible demo stub |
| `python -m pymc_bot about [--json]` / `--version` | author, links and credits |

Everything is also configurable through `pymc_bot_config.json` (created next to where you run the bot, or
`--config path.json` / `PYMC_BOT_CONFIG`). The entire file can be edited in the panel under **Advanced**.

---

## Configuration reference

```jsonc
{
  "minecraft": {
    "host": "127.0.0.1", "port": 25565, // "name.aternos.me" (SRV) or "host:port" also work
    "username": "PyMC_Bot",            // max 16 characters
    "version": "auto",                 // or "1.20.4", "1.12.2", ...
    "auth": "offline",                 // "offline" (cracked) | "microsoft"
    "backend": "auto",                 // "auto" (node, else simulated) | "node" | "simulated"
    "view_distance": "normal", "connect_timeout": 20.0,
    "profiles_folder": ".pymc_profiles", // Microsoft token cache, one sub-folder per account
    "auto_rejoin": true,               // keep re-joining after kicks/disconnects
    "rejoin_seconds": 10.0,            // base delay between rejoin attempts (backs off to 60s)
    "wait_for_server": false,          // free host asleep? wait until someone starts it, then join
    "wait_minutes": 10.0,
    "leave_when_alone": "auto",        // "auto" (on for Aternos, whose rules require it) | "always"
    "alone_minutes": 5.0               // ... leave after this long with no real player online
  },
  "ollama": {
    "enabled": false,
    "base_url": "http://127.0.0.1:11434",
    "model": "llama3.2",
    "temperature": 0.7,                // a little variety makes it more human
    "decision_interval": 6.0,          // seconds between AI decisions (±25% jitter)
    "request_timeout": 60.0,
    "system_prompt": ""                // "" = built-in human-player prompt (recommended)
  },
  "cortex": {                         // Cortex LLMHoster (OpenAI-compatible), see docs/CORTEX.md
    "enabled": false,
    "base_url": "http://127.0.0.1:8624",
    "api_path": "/v1",
    "model": "",                      // blank = Cortex's default model
    "api_key_env": "CORTEX_API_KEY",  // env var NAME; the key itself is never stored
    "temperature": 0.7,
    "max_tokens": 320,
    "json_mode": true,
    "decision_interval": 6.0,
    "request_timeout": 120.0,
    "system_prompt": ""
  },
  "training": {
    "models_dir": "models",             // where checkpoints + model.json cards live
    "dataset": "pymcbot/dataset.jsonl",  // written by the mod's /pymcbot export
    "active_run": "",                   // "" = newest trained run
    "engine": "mlp",                    // "mlp" (NumPy) | "transformer" (needs torch)
    "steps": 1000, "batch_size": 64, "lr": 0.003, "checkpoint_every": 200,
    "hidden": [128, 64],                // MLP layers
    "include_blocks": true,             // feed the 3x3x3 block neighbourhood
    "advanced": false,                  // advanced training: learn entity + item vocabularies
    "entity_slots": 24, "item_slots": 32,  // how many entity/item words the model learns
    "val_split": 0.1,
    "temperature": 0.2,                 // 0 = deterministic, >0 samples among similar actions
    "step_seconds": 0.4,                // how long one predicted action is held
    "auto_retrain_samples": 0           // reserved for automatic retraining
  },
  "agent": {
    "mode": "auto",                    // "auto" | "heuristic" | "ollama" | "cortex" | "trained"
    "allow_movement": true, "allow_chat": true, "allow_mining": true,
    "allow_attacking": false,          // keep this off unless you mean it
    "allow_inventory": true,           // hotbar, armour, moving / dropping / giving items
    "allow_building": false,           // placing blocks, right-clicking items
    "allow_mob_fighting": true,        // fight hostile mobs (never players - that is allow_attacking)
    "greet_players": true, "max_chat_length": 200,
    "persona": "friendly",             // friendly | chill | social | companion | builder | explorer | miner | newbie | pro | quiet
    "persona_notes": "",               // extra character notes for the LLM (max 600 chars)
    "human_chat": true,                // typing pauses, no repeats, cooldown for unprompted chat
    "chat_cooldown": 6.0,              // seconds between messages nobody asked for
    "memory_turns": 8,                 // recent actions the LLM remembers
    "compact_prompt": false,           // short system prompt for a Minecraft-trained model (docs/MINECRAFT_LLM.md)
    "record_turns": false,             // save LLM turns as training data ...
    "turns_file": "llm_turns.jsonl",   // ... in this file next to the config
    "wander_radius": 16, "action_timeout": 30.0
  },
  "fleet": {
    "enabled": true,
    "max_bots": 25,                    // cap for main bot + fleet
    "count": 5,                        // default for "Add players"
    "name_pattern": "PyMC_Bot_{n}",
    "auth": "offline",                 // "offline" | "microsoft"
    "ai_mode": "heuristic",            // "off" | "heuristic" | "ollama" | "cortex" | "trained"
    "stagger_seconds": 1.5,            // gap between joins
    "chatter": false, "chatter_interval": 45.0,
    "chatter_lines": ["hey everyone!", "..."],
    "restore_on_start": false,         // bring the fleet back when the panel restarts
    "roster": []                       // written automatically: who should come back
  },
  "server": { "host": "0.0.0.0", "port": 8000 }
}
```

Permissions gate **the AI's own choices**; buttons you press in the panel still work (except `attack`,
which stays gated). Invalid values are rejected with HTTP 422 and never overwrite a good config.

---

## HTTP API

The panel is a thin client over a small JSON API – script it from anything:

```bash
# status: bot, AI, stats, backends, Ollama health
curl -s localhost:8000/api/status | jq '.bot.state, .agent.running'

# fill the server with 10 offline players, 1.5s apart
curl -sX POST localhost:8000/api/fleet/populate -H 'content-type: application/json' -d '{"count":10}'

# add one premium player (Microsoft account email)
curl -sX POST localhost:8000/api/fleet/spawn -H 'content-type: application/json' \
     -d '{"username":"you@example.com","auth":"microsoft","ai":"heuristic"}'

# who is online, and make them all say something
curl -s localhost:8000/api/fleet/status | jq '.connected, [.bots[].username]'
curl -sX POST localhost:8000/api/fleet/broadcast -H 'content-type: application/json' -d '{"message":"hi!"}'

# switch to the simulated world and connect
curl -sX PUT localhost:8000/api/config -H 'content-type: application/json' \
     -d '{"minecraft":{"backend":"simulated"}}'
curl -sX POST localhost:8000/api/bot/start

# drive it by hand
curl -sX POST localhost:8000/api/bot/action -H 'content-type: application/json' -d '{"action":"mine","block":"stone"}'
curl -sX POST localhost:8000/api/bot/chat   -H 'content-type: application/json' -d '{"message":"hi everyone"}'
curl -sX POST localhost:8000/api/ai/start
```

`WS /ws` pushes `event` frames (the same coloured log you see in the panel) and a `status` frame at least
every 1.5s. Full endpoint list: [docs/API.md](docs/API.md). The Python↔Node bridge contract is documented in
[docs/BRIDGE.md](docs/BRIDGE.md), the playtime-recording mod in [mod/README.md](mod/README.md), and training
plus running the learned policy in [docs/TRAINING.md](docs/TRAINING.md).

---

## Development

```bash
pip install -r requirements-dev.txt

pytest -q                       # 435 tests, no Minecraft server, no LLM required (2 skip without torch)
ruff check .                    # lint
pytest --cov=pymc_bot -q        # coverage

# opt-in end-to-end test against a real (offline-mode) server
PYMC_TEST_SERVER=127.0.0.1:25565 pytest tests/test_node_live.py -v
```

The suite runs the bot against the simulated world, exercises the whole FastAPI surface (including the
WebSocket), tests the Node adapter against a fake bridge process (`tests/fixtures/fake_bridge.js`) and runs
the AI loop against the Ollama stub. The training stack is covered end to end as well: the encoder layout is
pinned in `tests/test_features.py`, `tests/test_train.py` covers checkpoints/resume/metrics, and
`tests/test_local_model.py` lets a real trained model drive a simulated bot.
CI (`.github/workflows/ci.yml`) runs lint + tests on Python 3.10–3.12, the transformer engine with PyTorch,
and **builds the Fabric mod for all 16 Minecraft versions from 1.21.1 to 26.3** on JDK 25 (one matrix
job per version from `mod/versions.json`; 1.21.x jars target Java 21, 26.x jars Java 25), **launches
the game with the mod on every version from 1.21.4 on** (Fabric's client game test in a virtual
display: it records a few seconds of scripted play, checks the dataset and uploads screenshots) and
uploads each jar plus a `pymcbot-all-versions` bundle. When everything is green, the `release`
job attaches all 16 jars to the GitHub release `pymcbot-v<mod_version>`.

```
pymc_bot/
├── __main__.py      CLI (serve / run / train / models / action / doctor / stub)
├── config.py        pydantic settings + atomic JSON persistence
├── backends.py      simulated world + Node/mineflayer adapter
├── bot.py           player controller: steering, pathfinding, mining, inventory, fighting, chat, stats
├── fleet.py         one or many players: spawn/populate, premium accounts, roster, chatter
├── antiafk.py       anti-AFK keeper: human-like walking, looking, jumping, crouching, swinging, hotbar
├── inventory.py     hotbar / backpack / armour model, best tool / weapon / food, hotbar layout
├── agent.py         decision loop, action validation, permissions, heuristics
├── features.py      observation -> 215-float feature vector (mod & live bot share it)
├── playtime.py      reads the mod's NDJSON, exports datasets, synthesises demo playtime
├── models.py        numpy MLP (default engine) + optional tiny transformer
├── train.py         self-checkpointing trainer, metrics, resumable state, service
├── local_model.py   runs a checkpoint: predictions -> real low-level bot controls
├── ollama.py        tiny Ollama HTTP client
├── cortex.py        Cortex LLMHoster client (OpenAI-compatible, bearer key from the environment)
├── ollama_stub.py   Ollama/OpenAI-compatible demo stub (not an LLM)
├── server.py        FastAPI app + WebSocket + static UI
├── web/             index.html, styles.css, app.js (no build step)
└── node/            minecraft_bridge.js  (the only JavaScript)
mod/                 Fabric mod (MC 1.21.1 – 26.3): records your playtime into a training dataset
tests/               pytest suite + fake bridge fixture
scripts/dev.sh       small helper for common tasks
updater.py           self-updater for the bot, its dependencies and the mod (+ updater.cmd / updater.sh)
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `mineflayer is not installed` | run `npm install` in the project root (Node ≥ 22) |
| Bot connects then immediately leaves | check the event stream: wrong version? try an explicit `version` instead of `auto` |
| `unsupported protocol version` | set the app's version to the server's (e.g. `1.20.4`) |
| `No data available for version 26.3` / "server runs Minecraft 26.3, but … only supports up to 26.1" | the server is newer than the installed mineflayer (it currently stops at 26.1; 26.2/26.3 support is still being added upstream). The bot now joins as the newest supported version automatically, so either install **ViaVersion + ViaBackwards** on the server, or run the server on 26.1. Once mineflayer releases support, run the updater (or `npm update`) and set Version back to `auto` |
| Kicked with `Outdated client` | same cause: the server only accepts its own version – add ViaVersion + ViaBackwards to it |
| "… (Aternos) is offline" / "is starting" | the free host's server is asleep: start it from the dashboard, or set **Server asleep?** to *wait for it to start* |
| "refused the offline-mode login" | the server is in online mode: turn on *Cracked* (Aternos: Options) / `online-mode=false`, or use a Microsoft account |
| Bots left with "No real player has been online" | the leave-when-alone rule (always on for Aternos, whose rules forbid idling bots); press Connect when people are on |
| Connection error `ECONNREFUSED` | nothing listens at that host/port – start the Minecraft server or fix the address |
| Microsoft login hangs | the device code is printed in the event stream – complete it in a browser |
| `does the account own minecraft?` | that Microsoft account has no Java Edition – use one that owns the game |
| Premium bot joins again after a restart with no prompt | that is the per-account token cache in `.pymc_profiles/` working as intended |
| `fleet is full` | raise `fleet.max_bots` or stop/remove players first |
| Bots get kicked while joining in bulk | increase **Gap between joins** (2–3s) or ask the server to raise its throttling |
| Still kicked for AFK | lower `antiafk.interval_min`/`interval_max` (some servers use 60 s windows), keep *walk* and *swing* enabled, or turn on `verbose` to confirm bursts happen |
| Bots wander off while you are not looking | set a smaller `antiafk.max_walk_distance`, or disable `walk` (look/swing/jump already defeat most AFK plugins) |
| Panel shows `demo brain (stub)` | you are pointing Ollama at `python -m pymc_bot stub`, not a real model |
| Cortex section says `not running (optional…)` | nothing to fix: Cortex is only needed when the brain is set to Cortex |
| Panel pill says `cortex offline` | hover it for the reason; usually `CORTEX_API_KEY` not exported where PyMC_Bot runs, or the Cortex model is not started – see [docs/CORTEX.md](docs/CORTEX.md#troubleshooting) |
| Bot stands still | AI off (press **Start AI**), or movement permission disabled, or the server is unreachable |
| `no playtime samples found` | record with the mod first (`/pymcbot record start`, `/pymcbot export`), or try `python -m pymc_bot train --simulate 5` |
| advanced training looks like basic training | `train --inspect` shows an empty vocabulary: the recording was made before `/pymcbot advanced on`, so there is nothing extra to learn. Record again with it on. |
| Trained model only walks forward | play/train more, or lower `training.temperature`; check `val_balanced_accuracy` in the model card |
| `the transformer engine needs PyTorch` | `pip install -r requirements-train.txt`, or use the default `--engine mlp` |
| Bot ignores the trained model | set `agent.mode` to `trained` **and** activate a run (`training.active_run`, or the panel's checkpoint table) |
| `./gradlew build` fails in `mod/` | needs **JDK 25** to run Gradle (Loom 1.18 rejects older runtimes with "requires at least JVM runtime version 25"); pick a version with `-Pmc=<version>` (`./gradlew printTargets` lists them). See [the build troubleshooting table](mod/README.md#troubleshooting-the-build); CI builds every version on each push, so compare with the `mod (fabric <version>)` job there |
| updater: `is in use - close Minecraft` | Windows locks loaded jars - quit the game, run `updater --mod-only` again |
| updater: `rate limit` | wait an hour, or set `GITHUB_TOKEN` to any GitHub token |
| Web panel unreachable from another machine | `server.host` must stay `0.0.0.0` and the port must be open |

## Credits

| | |
| --- | --- |
| **Author** | [codero-sus](https://github.com/codero-sus) |
| **Website** | [the-codero.web.app](https://the-codero.web.app/) |
| **Source** | [github.com/codero-sus/PyMC_Bot](https://github.com/codero-sus/PyMC_Bot) |
| **Also by codero-sus** | [Cortex LLMHoster](https://github.com/codero-sus/Cortex_LLMHoster) – the optional self-hosted LLM brain |
| **Built with** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/), [mineflayer](https://github.com/PrismarineJS/mineflayer), [Fabric](https://fabricmc.net/), [Ollama](https://ollama.com/), NumPy / PyTorch |

The same credits are in the panel (footer and ⓘ *About*), at `GET /api/about`, in `python -m pymc_bot about`,
in the OpenAPI docs (`/docs`) and in the Fabric mod (`/pymcbot about`, `fabric.mod.json`).

## License

© 2026 [codero-sus](https://github.com/codero-sus). Licensed under the
[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) – see [LICENSE](LICENSE).

**Free for personal, non-commercial use.** You may use, change and share PyMC_Bot and the PyMC Bot mod for
yourself, as a hobby, for study or research, or in a charity, school or public organisation. You may
**not** sell it, put it behind a paywall, or use it to make money (for example selling bots or "player
filling" as a service) without written permission from codero-sus. Anyone you share it with must get the
licence text (or its URL) and the `Required Notice:` line at the top of [LICENSE](LICENSE).

Versions published before this change (up to commit `271a602` and the mod releases up to 0.7.0 built
from it) were released under the MIT license; copies already obtained under MIT keep those terms.

Third-party parts keep their own licenses: the compile-only Mod Menu API stubs in
`mod/src/client/java/com/terraformersmc/` (MIT, TerraformersMC; never shipped in the jars), the Fabric
template files the mod was bootstrapped from (CC0) and all npm/PyPI dependencies.

<p align="center"><sub>Made with ❤️ by <a href="https://github.com/codero-sus">codero-sus</a> · <a href="https://the-codero.web.app/">the-codero.web.app</a></sub></p>
