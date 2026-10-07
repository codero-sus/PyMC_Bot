# PyMC_Bot HTTP / WebSocket API

Base URL: `http://<host>:<port>` (default `http://0.0.0.0:8000`, panel on `/`).
All bodies are JSON, all responses are JSON. Interactive docs are served at `/docs` (Swagger) and
`/redoc` by FastAPI.

Errors use the usual FastAPI shape and these status codes:

| Code | Meaning |
| --- | --- |
| `409` | the bot is not connected (or the AI loop cannot start yet) |
| `422` | validation failed (bad config value, unknown action, unparseable JSON) |
| `502` | the Minecraft bridge or Ollama could not be reached |

---

## Endpoints

### Panel & health

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | the control panel (HTML) |
| `GET` | `/healthz` | `{"ok": true, "version": "0.1.0"}` |
| `GET` | `/api/status` | everything the panel shows: selected bot snapshot, AI status, stats, **fleet**, backends, Ollama health |
| `GET` | `/api/contract` | action list, bridge command list, auth/AI modes, brains, training engines, recorded player actions, WebSocket frame types |

`GET /api/status` shape (truncated):

```json
{
  "version": "0.1.0", "time": 1767000000.0,
  "config_path": "/abs/path/pymc_bot_config.json", "config_load_error": null,
  "bot": {
    "state": "connected", "username": "MainBot", "status": "connected", "backend": "node",
    "position": {"x": 12.5, "y": 64.0, "z": -3.1}, "yaw": 0.0, "pitch": -0.2,
    "health": 20, "food": 18, "dimension": "overworld", "time_of_day": "day",
    "players": [{"name": "Steve", "x": 1, "y": 64, "z": 2, "distance": 4.2}],
    "inventory": [{"name": "oak_log", "count": 3}],
    "server": {"host": "127.0.0.1", "port": 25565, "version": "1.20.4", "username": "PyMC_Bot"},
    "uptime": 42.7, "last_error": null
  },
  "agent": {"running": true, "decisions": 7, "errors": 0, "last_error": null,
            "last_decision": {"action": "mine", "params": {"block": "oak_log"}, "source": "ollama:llama3.2"}},
  "stats": {"actions": 19, "chats_sent": 2, "blocks_mined": 4, "distance_walked": 133.2, "uptime": 88.0, "reconnects": 0},
  "backends": {"node": {"available": true, "reason": "ok"}, "simulated": {"available": true},
               "node_version": "v22.1.0", "python": "3.11.9"},
  "ollama": {"ok": true, "detail": "1 model(s): llama3.2:latest", "settings": {"...": "..."}}
}
```

### Configuration

| Method | Path | Body | Description |
| --- | --- | --- | --- |
| `GET` | `/api/config` | – | current config, its file path and any load error |
| `PUT` | `/api/config` | partial config object | deep-merge + validate + save (atomic write); 422 on invalid values |
| `POST` | `/api/config/reset` | – | write the defaults |

```bash
curl -X PUT localhost:8000/api/config -H 'content-type: application/json' \
  -d '{"minecraft":{"host":"mc.example.net","port":25565,"auth":"offline"},
       "ollama":{"enabled":true,"model":"llama3.2"},"agent":{"allow_attacking":false}}'
```

### Fleet & premium players

One instance controls the **main bot** (the `minecraft` config section) plus any number of **fleet**
players. Premium players use `auth="microsoft"` and their **email** as the username.

| Method | Path | Body | Description |
| --- | --- | --- | --- |
| `GET` | `/api/fleet/status` | – | every player with state/auth/ai/position + totals |
| `POST` | `/api/fleet/populate` | `{"count":10,"pattern":"PyMC_Bot_{n}","auth":"offline","ai":"heuristic","stagger":1.5}` | register and join many players (staggered, background) |
| `POST` | `/api/fleet/populate` | `{"usernames":["alpha","beta"]}` | explicit names instead of a pattern |
| `POST` | `/api/fleet/spawn` | `{"username":"you@example.com","auth":"microsoft","ai":"heuristic"}` | add **one** player (premium or offline) |
| `POST` | `/api/fleet/start` | `{"bot":"Pop_1"}` or `{}` | (re)connect one player, or all of them |
| `POST` | `/api/fleet/stop` | `{"bot":"Pop_1","remove":false}` or `{}` | disconnect one, or all |
| `POST` | `/api/fleet/remove` | `{"bot":"Pop_1"}` | disconnect **and forget** (no restore after restart) |
| `POST` | `/api/fleet/select` | `{"bot":"Pop_1"}` or `{"bot":null}` | which player the panel controls |
| `POST` | `/api/fleet/broadcast` | `{"message":"hi"}` | every connected bot says it |
| `GET` | `/api/fleet/roster` | – | players remembered for `fleet.restore_on_start` |
| `DELETE` | `/api/fleet/roster` | – | forget them (does not disconnect anyone) |

Validation is strict and friendly: invalid offline names (`"bad name!"`), emails used with
`auth="offline"`, non-emails with `auth="microsoft"`, duplicate names, names already used by the main bot,
and batches that would exceed `fleet.max_bots` all return `422` with the reason.

```bash
# 5 extra offline players, then a premium one
curl -sX POST localhost:8000/api/fleet/populate -H 'content-type: application/json' -d '{"count":5}'
curl -sX POST localhost:8000/api/fleet/spawn -H 'content-type: application/json' \
     -d '{"username":"you@example.com","auth":"microsoft"}'
curl -s localhost:8000/api/fleet/status | jq '.connected, [.bots[] | {username, state, auth}]'
```

`/api/fleet/status` shape (anti-AFK fields shown):

```json
{
  "enabled": true, "max_bots": 25, "size": 6, "extra_bots": 5, "connected": 6,
  "roster_size": 5, "selected": "PyMC_Bot_3",
  "antiafk": {"enabled": true, "running": true, "pokes": 17},
  "primary": {"username": "MainBot", "sponsor": "primary", "state": "connected", "selected": false},
  "bots": [
    {"username": "MainBot", "configured_username": "MainBot", "auth": "offline", "ai": "heuristic",
     "ai_running": true, "sponsor": "primary", "state": "connected", "position": {"x":1,"y":64,"z":2},
     "health": 20, "uptime": 42.0, "blocks_mined": 3, "chats_sent": 1, "reconnects": 0, "selected": false,
     "busy": false, "idle_seconds": 4.2, "antiafk_pokes": 3},
    {"username": "PremiumPlayer", "configured_username": "you@example.com", "auth": "microsoft",
     "ai": "off", "sponsor": "premium", "state": "connected", "error": null, "selected": false}
  ]
}
```

### Anti-AFK

Keeps every bot moving and looking around so servers do not kick it for being idle. On by default.

| Method | Path | Body | Description |
| --- | --- | --- | --- |
| `GET` | `/api/antiafk/status` | – | keeper settings, total bursts, and per bot: idle seconds, busy flag, poke count, last habits |
| `POST` | `/api/antiafk/poke` | `{"bot":"all"}` / `{"bot":"selected"}` / `{"bot":"PyMC_Bot_3"}` | force a burst now. One bot answers with its habits; a crowd is queued in the background (`{"queued": N}`) |

Configuration lives in the `antiafk` section of `PUT /api/config` (see the README). Toggling it off stops all
bursts immediately; the per-bot `idle_seconds` / `antiafk_pokes` numbers are also part of
`GET /api/fleet/status` and of every `status` WebSocket frame.

```bash
curl -s localhost:8000/api/antiafk/status | jq '{pokes, bots: [.bots[] | {username, idle_seconds, last_habits}]}'
curl -sX POST localhost:8000/api/antiafk/poke -H 'content-type: application/json' -d '{"bot":"all"}'
```

### Bot control

| Method | Path | Body | Description |
| --- | --- | --- | --- |
| `POST` | `/api/bot/start` | – | connect using the saved config (blocks until spawned, or 502) |
| `POST` | `/api/bot/stop` | – | stop the AI loop and disconnect |
| `POST` | `/api/bot/reconnect` | – | stop + start |
| `GET` | `/api/bot/snapshot` | – | world snapshot only |
| `GET` | `/api/bot/chat` | – | recent chat history (`{"messages":[{"time","username","message"}]}`) |
| `POST` | `/api/bot/chat` | `{"message": "hi", "bot": "Pop_1"}` | send chat (respects `agent.max_chat_length`); `bot` may be a username, `"selected"` (default) or `"all"` |
| `POST` | `/api/bot/command` | `{"command": "list", "bot": "all"}` | run a server command on the selected/all bots (`/` is added if missing, needs op) |
| `POST` | `/api/bot/action` | `{"action": "wander", "params": {}, "bot": "all"}` **or** `{"raw": "{...}"}` | run one JSON action on the selected bot, one by name, or every bot |

Which player "the bot" means is decided by `/api/fleet/select`: `/api/bot/*`, `/api/ai/*`, `/api/status`
and the whole panel follow the selected player. The default is the main bot.

### AI (Ollama) loop

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/ai/status` | running flag, decision/error counters, last decision |
| `POST` | `/api/ai/start` | start the decision loop (409 if the bot is not connected) |
| `POST` | `/api/ai/stop` | stop the loop (aborts the current action) |
| `POST` | `/api/ai/step` | decide and execute exactly one action now |

### Training a player model on playtime

The Fabric mod in [`../mod`](../mod/README.md) records the player's playtime into
`pymc-playtime/dataset.jsonl`; these endpoints train on it, list the checkpoints and switch the bot over to
the trained brain. Training runs in a background thread and checkpoint itself every
`training.checkpoint_every` steps.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/training/status` | training progress + settings + dataset statistics + current brain |
| `GET` | `/api/training/dataset?dataset=...` | samples/episodes/action histogram of a playtime dataset (cached by mtime) |
| `POST` | `/api/training/start` | start (or resume) a run — see the body below |
| `POST` | `/api/training/stop` | ask the run to stop; it writes a checkpoint first (`409` when idle) |
| `GET` | `/api/models` | every run in `training.models_dir` with its `model.json` card, newest first |
| `GET` | `/api/models/{run}` | one card |
| `POST` | `/api/models/activate` | `{"run":"playtime-dataset"}` → `agent.mode="trained"`, reloads the AI loop |
| `POST` | `/api/models/deactivate` | back to `agent.mode="auto"` |
| `POST` | `/api/models/export-ollama` | `{"run":"...","name":"pymc-playtime"}` → create an Ollama model from the learned habits |
| `GET` | `/api/policy/preview` | what the active model would do right now, given the live bot snapshot |

`POST /api/training/start` body (all fields optional; missing ones come from the config):

```json
{
  "dataset": "pymc-playtime/dataset.jsonl",
  "run_name": "playtime-dataset",
  "engine": "mlp",
  "steps": 2000, "batch_size": 64, "lr": 0.003,
  "checkpoint_every": 200, "max_seconds": 0,
  "hidden": [128, 64], "include_blocks": true,
  "advanced": false, "entity_slots": 24, "item_slots": 32,
  "resume": false,
  "simulate_minutes": 0
}
```

`simulate_minutes > 0` first generates that much synthetic playtime (handy for demos and CI). Errors:
`400` when the dataset does not exist, `409` when a run is already in progress, `502` when exporting to an
Ollama endpoint that is not reachable.

```bash
curl -X POST localhost:8000/api/training/start -H 'content-type: application/json' \
  -d '{"simulate_minutes":5,"steps":400,"run_name":"demo"}'
curl localhost:8000/api/training/status
curl -X POST localhost:8000/api/models/activate -H 'content-type: application/json' -d '{"run":"demo"}'
curl localhost:8000/api/policy/preview
```

`advanced: true` learns the entity and item vocabularies the recording contains (the mod writes them
with `/pymc advanced on`) and feeds each word to the model with its distance and bearing; `entity_slots`
and `item_slots` cap the vocabularies. The learned words are stored in the checkpoint and in
`model.json`, so the bot encodes the live world into exactly the same slots.

`/api/status` gains a `training` block (`running`, `step`, `run`, `active_run`, `brain`, `models`) and,
next to it, an `advanced` overview:

```json
{
  "available": true, "trained": true, "run": "playtime-dataset",
  "entities": ["player", "zombie", "creeper", "..."], "items": ["iron_sword", "cooked_beef", "..."],
  "entity_slots": 24, "item_slots": 32,
  "coverage": {"entity": 1.0, "item": 0.98}
}
```

`/api/training/dataset` reports the same budget for the dataset itself (`entities_and_items`,
`vocabulary`, `entity_coverage`, `item_coverage`, top entity/item types), and `/api/contract` lists the
brains, the training engines, the training modes (`basic`, `advanced`), the advanced signals
(`entities`, `items`, `held_item`, `armor`, `ground_items`) and the recorded player actions.

### Ollama

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/ollama/health?force=true` | cached (15s) reachability probe |
| `GET` | `/api/ollama/models` | models available on that endpoint |
| `POST` | `/api/ollama/pull` | `{"model":"llama3.2"}` – pull a model (long request, 600s timeout) |

### Logs & backends

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/logs?limit=200` | ring-buffer of events (max 500 kept) |
| `POST` | `/api/logs/clear` | empty the buffer |
| `GET` | `/api/backends` | which backends can run here and why (Node/mineflayer detection) |

---

## Actions

Actions are the contract between you, the LLM and the bot. They are plain JSON objects; unknown keys are
ignored and every parameter is validated before anything happens.

| Action | Parameters | Gated by | Effect |
| --- | --- | --- | --- |
| `say` | `message` (string, truncated to `max_chat_length`) | `agent.allow_chat` | send a chat message |
| `goto` | `x`, `z` (numbers, required), `y` (optional) | `agent.allow_movement` | walk to a point (pathfinder or Python steering) |
| `follow` | `player` | `agent.allow_movement` | keep within ~3 blocks of a player |
| `wander` | – | `agent.allow_movement` | explore a few random waypoints inside `wander_radius` |
| `mine` | `block` (e.g. `oak_log`, `stone`, `coal_ore`) | `agent.allow_mining` | dig the nearest matching block, exploring if none is close |
| `attack` | `player` | `agent.allow_attacking` (**always** gated, even for humans) | hit a player |
| `look_at_player` | `player` | – | turn the bot's head towards a player |
| `jump` | – | – | hop (useful over obstacles) |
| `eat` | – | – | eat food from the inventory if hungry |
| `stop` | – | – | release all movement keys |
| `wait` | – | – | do nothing this turn |

Aliases accepted from models: `walk`/`move` → `goto`, `follow_player` → `follow`, `explore` → `wander`,
`dig`/`gather` → `mine`, `hit` → `attack`, `hop` → `jump`, `idle` → `stop`, `noop`/`none`/`nothing` → `wait`.

The executor also tolerates real-world LLM noise: ```` ```json ```` fences, prose before/after the object,
`name`/`cmd`/`command` instead of `action`, a namespaced `minecraft:mine`, nested `params`, and non-finite
coordinates (clamped to ±30000).

Manual requests (`POST /api/bot/action`) bypass the movement/chat/mining permissions because a human asked
for it explicitly; `attack` stays gated.

---

## WebSocket `/ws`

Connect once and the panel stays live. Frames:

```json
{"type": "hello",  "data": {"version": "0.1.0", "config": "/abs/path/pymc_bot_config.json"}}
{"type": "status", "data": { /* same object as GET /api/status */ }}
{"type": "event",  "data": {"level": "ai", "source": "agent", "message": "AI chose: wander", "time": "12:04:51", "ts": 1767000000.0, "data": null}}
```

* a `status` frame is sent every 1.5s even when nothing happens, so the UI stays in sync;
* event levels: `debug`, `info`, `success`, `warn`, `error`, `chat` (someone spoke), `bot` (the bot spoke),
  `ai` (a decision was made);
* the server never requires input; slow clients are dropped from the subscriber list automatically.

```js
const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onmessage = (m) => { const frame = JSON.parse(m.data); if (frame.type === 'event') console.log(frame.data.message); };
```
