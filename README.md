# PyMC_Bot

A **Python-controlled Minecraft player bot** it joins a (cracked / offline-mode) server as a normal
player, you configure and steer it from a **web control panel served by Uvicorn**, and it can hand the
wheel to a **local Ollama model** that decides how it plays (walking, mining, following players, chatting).

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
│  REST + WebSocket /ws        │   WS   │  AgentLoop (Ollama)     agent.py          │
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
* **Works without an LLM** – if Ollama is disabled or unreachable the bot keeps playing with a small
  built-in heuristic policy, so it never just stands there.
* **One bot or a hundred** – the *Players* panel spawns offline bots from a name pattern, adds premium
  players one email at a time, and can keep the crowd alive (rejoin on kick, optional idle chatter).
* **Anti-AFK built in** – every bot periodically walks a step, hops, crouches, swings its arm and turns its
  head in smooth, randomised bursts, so servers do not kick them for standing still. It never fights the AI
  or your manual commands, and it scales to the whole fleet from one thread.
* **Trains on *your* playtime** – a Fabric mod ([`./mod`](mod/README.md)) records how you play, `pymc_bot train`
  turns that into a checkpointed player model, and `think trained` runs the bot with it. No Ollama, no GPU,
  no cloud: the model learns your movements, your turns and your habits and reproduces them in game.

---

## Quickstart

```bash
git clone https://github.com/codero-sus/PyMC_Bot.git
cd PyMC_Bot

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

npm install            # mineflayer bridge (only for real servers, needs Node >= 18)

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

Two or three habits are often chained, the same habit never repeats back-to-back, and the intervals are
jittered — so the pattern does not look like a metronome. The keeper only touches bots that are **not** busy:
AI decisions, follows, mining and your manual commands always take priority.

```jsonc
"antiafk": {
  "enabled": true,
  "interval_min": 8.0,        // idle seconds before a burst (randomised)
  "interval_max": 25.0,
  "look": true, "look_at_players": true, "walk": true, "jump": true, "sneak": true, "swing": true,
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

## Train a player model on your own playtime

The bot can learn **how you play** and then play like that itself. Three steps:

**1. Record** – build the Fabric mod in [`./mod`](mod/README.md) (it started from the official
`fabric-example-mod` template: download the zip, extract, edit), drop it in `.minecraft/mods/` with
[Fabric API](https://modrinth.com/mod/fabric-api), then in game:

```text
/pymc record start      # play normally for 10-60 minutes
/pymc record stop
/pymc export            # -> <gameDir>/pymc-playtime/dataset.jsonl
```

The mod samples your playtime 20×/s — position, velocity, view, posture, hotbar, health/food, the 3×3×3
block neighbourhood, nearby entities, and the action you were performing
(`forward`, `back`, `left`, `right`, `jump`, `sneak`, `look`, `attack`, `use`, `hold`, `move`, `none`).
`/pymc advanced on` additionally records **entities, items, the held item, armour and dropped stacks**
for [advanced training](#advanced-training-entities-and-items).

**2. Train** – *checkpointing itself as it goes*:

```bash
python -m pymc_bot train --dataset "<gameDir>/pymc-playtime/dataset.jsonl" --steps 2000
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
/pymc advanced on        # in game, before recording: entities + items + drops are written too
/pymc record start  ...  /pymc record stop  ...  /pymc export
```

```bash
python -m pymc_bot train --dataset "<gameDir>/pymc-playtime/dataset.jsonl" --advanced --steps 2000
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
`ollama.decision_interval` seconds (default 6s) the bot sends the current world state and asks for one action:

```json
{"action": "mine", "block": "oak_log"}
```

Accepted actions: `say`, `goto`, `follow`, `wander`, `mine`, `attack` (gated), `jump`, `eat`,
`look_at_player`, `stop`, `wait`. The full list, including which permissions gate them, is in
[docs/API.md](docs/API.md#actions). Malformed model output is caught, logged and retried with the
heuristic policy instead of crashing the loop.

### Demo brain (no GPU, no download)

For demos, tests and CI there is an **Ollama-compatible stub** – clearly *not* a language model, just a
rule-based responder that answers the same `/api/chat`, `/api/generate` and `/api/tags` endpoints:

```bash
python -m pymc_bot stub            # listens on http://127.0.0.1:11434
```

The panel shows a `demo brain (stub)` badge whenever the endpoint identifies itself as the stub, so it is
never mistaken for a real model.

---

## Command line

| Command | What it does |
| --- | --- |
| `python -m pymc_bot serve [--host 0.0.0.0] [--port 8000] [--autostart] [--auto-agent]` | run the web config panel (Uvicorn) |
| `python -m pymc_bot run [--backend node\|simulated] [--server host:port] [--seconds N] [--no-ai] [--populate N] [--premium EMAIL] [--think auto\|heuristic\|ollama\|trained]` | headless bot: connect and play (optionally populate the server), logs to stdout |
| `python -m pymc_bot train [--dataset FILE] [--steps N] [--engine mlp\|transformer] [--advanced [--entity-slots N] [--item-slots N]] [--resume] [--simulate MINUTES] [--ollama-model NAME]` | train a player model on recorded playtime (self-checkpointing, resumable); `--advanced` learns entity + item vocabularies |
| `python -m pymc_bot models [--models-dir models] [--json]` | list trained checkpoints with their metrics |
| `python -m pymc_bot action '{"action":"wander"}' [--url http://127.0.0.1:8000]` | poke a running panel from the shell |
| `python -m pymc_bot doctor` | environment check (Node, mineflayer, Ollama, config, server settings) |
| `python -m pymc_bot stub [--port 11434]` | the Ollama-compatible demo stub |

Everything is also configurable through `pymc_bot_config.json` (created next to where you run the bot, or
`--config path.json` / `PYMC_BOT_CONFIG`). The entire file can be edited in the panel under **Advanced**.

---

## Configuration reference

```jsonc
{
  "minecraft": {
    "host": "127.0.0.1", "port": 25565,
    "username": "PyMC_Bot",            // max 16 characters
    "version": "auto",                 // or "1.20.4", "1.12.2", ...
    "auth": "offline",                 // "offline" (cracked) | "microsoft"
    "backend": "auto",                 // "auto" (node, else simulated) | "node" | "simulated"
    "view_distance": "normal", "connect_timeout": 20.0,
    "profiles_folder": ".pymc_profiles", // Microsoft token cache, one sub-folder per account
    "auto_rejoin": true,               // keep re-joining after kicks/disconnects
    "rejoin_seconds": 10.0             // base delay between rejoin attempts (backs off to 60s)
  },
  "ollama": {
    "enabled": false,
    "base_url": "http://127.0.0.1:11434",
    "model": "llama3.2",
    "temperature": 0.4,
    "decision_interval": 6.0,          // seconds between AI decisions
    "request_timeout": 60.0,
    "system_prompt": "You are PyMC_Bot, an AI player ..."
  },
  "training": {
    "models_dir": "models",             // where checkpoints + model.json cards live
    "dataset": "pymc-playtime/dataset.jsonl",  // written by the mod's /pymc export
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
    "mode": "auto",                    // "auto" | "heuristic" | "ollama" | "trained"
    "allow_movement": true, "allow_chat": true, "allow_mining": true,
    "allow_attacking": false,          // keep this off unless you mean it
    "greet_players": true, "max_chat_length": 200,
    "wander_radius": 16, "action_timeout": 30.0
  },
  "fleet": {
    "enabled": true,
    "max_bots": 25,                    // cap for main bot + fleet
    "count": 5,                        // default for "Add players"
    "name_pattern": "PyMC_Bot_{n}",
    "auth": "offline",                 // "offline" | "microsoft"
    "ai_mode": "heuristic",            // "off" | "heuristic" | "ollama"
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

pytest -q                       # 264 tests, no Minecraft server, no LLM required (2 skip without torch)
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
and **builds the Fabric mod on JDK 25** (Loom 1.18 needs a Java 25 Gradle runtime; the
mod targets Java 21) and uploads the jar as an artifact.

```
pymc_bot/
├── __main__.py      CLI (serve / run / train / models / action / doctor / stub)
├── config.py        pydantic settings + atomic JSON persistence
├── backends.py      simulated world + Node/mineflayer adapter
├── bot.py           player controller: steering, pathfinding, mining, chat, stats
├── fleet.py         one or many players: spawn/populate, premium accounts, roster, chatter
├── antiafk.py       anti-AFK keeper: human-like walking, looking, jumping, crouching, swinging
├── agent.py         decision loop, action validation, permissions, heuristics
├── features.py      observation -> 215-float feature vector (mod & live bot share it)
├── playtime.py      reads the mod's NDJSON, exports datasets, synthesises demo playtime
├── models.py        numpy MLP (default engine) + optional tiny transformer
├── train.py         self-checkpointing trainer, metrics, resumable state, service
├── local_model.py   runs a checkpoint: predictions -> real low-level bot controls
├── ollama.py        tiny Ollama HTTP client
├── ollama_stub.py   Ollama-compatible demo stub (not an LLM)
├── server.py        FastAPI app + WebSocket + static UI
├── web/             index.html, styles.css, app.js (no build step)
└── node/            minecraft_bridge.js  (the only JavaScript)
mod/                 Fabric mod: records your playtime into a training dataset
tests/               pytest suite + fake bridge fixture
scripts/dev.sh       small helper for common tasks
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `mineflayer is not installed` | run `npm install` in the project root (Node ≥ 18) |
| Bot connects then immediately leaves | check the event stream: wrong version? try an explicit `version` instead of `auto` |
| `unsupported protocol version` | set the app's version to the server's (e.g. `1.20.4`) |
| Microsoft login hangs | the device code is printed in the event stream – complete it in a browser |
| `does the account own minecraft?` | that Microsoft account has no Java Edition – use one that owns the game |
| Premium bot joins again after a restart with no prompt | that is the per-account token cache in `.pymc_profiles/` working as intended |
| `fleet is full` | raise `fleet.max_bots` or stop/remove players first |
| Bots get kicked while joining in bulk | increase **Gap between joins** (2–3s) or ask the server to raise its throttling |
| Still kicked for AFK | lower `antiafk.interval_min`/`interval_max` (some servers use 60 s windows), keep *walk* and *swing* enabled, or turn on `verbose` to confirm bursts happen |
| Bots wander off while you are not looking | set a smaller `antiafk.max_walk_distance`, or disable `walk` (look/swing/jump already defeat most AFK plugins) |
| Panel shows `demo brain (stub)` | you are pointing Ollama at `python -m pymc_bot stub`, not a real model |
| Bot stands still | AI off (press **Start AI**), or movement permission disabled, or the server is unreachable |
| `no playtime samples found` | record with the mod first (`/pymc record start`, `/pymc export`), or try `python -m pymc_bot train --simulate 5` |
| advanced training looks like basic training | `train --inspect` shows an empty vocabulary: the recording was made before `/pymc advanced on`, so there is nothing extra to learn. Record again with it on. |
| Trained model only walks forward | play/train more, or lower `training.temperature`; check `val_balanced_accuracy` in the model card |
| `the transformer engine needs PyTorch` | `pip install -r requirements-train.txt`, or use the default `--engine mlp` |
| Bot ignores the trained model | set `agent.mode` to `trained` **and** activate a run (`training.active_run`, or the panel's checkpoint table) |
| `./gradlew build` fails in `mod/` | needs **JDK 25** to run Gradle (Loom 1.18 rejects older runtimes with "requires at least JVM runtime version 25"); the mod builds in CI on every push, so compare with the `mod` job there |
| Web panel unreachable from another machine | `server.host` must stay `0.0.0.0` and the port must be open |

## License

MIT – see [LICENSE](LICENSE).
