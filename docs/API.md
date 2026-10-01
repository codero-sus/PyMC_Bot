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
| `GET` | `/api/status` | everything the panel shows: bot snapshot, AI status, stats, backends, Ollama health |
| `GET` | `/api/contract` | action list, bridge command list, WebSocket frame types |

`GET /api/status` shape (truncated):

```json
{
  "version": "0.1.0", "time": 1767000000.0,
  "config_path": "/abs/path/pymc_bot_config.json", "config_load_error": null,
  "bot": {
    "state": "connected", "status": "connected", "backend": "node",
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

### Bot control

| Method | Path | Body | Description |
| --- | --- | --- | --- |
| `POST` | `/api/bot/start` | – | connect using the saved config (blocks until spawned, or 502) |
| `POST` | `/api/bot/stop` | – | stop the AI loop and disconnect |
| `POST` | `/api/bot/reconnect` | – | stop + start |
| `GET` | `/api/bot/snapshot` | – | world snapshot only |
| `GET` | `/api/bot/chat` | – | recent chat history (`{"messages":[{"time","username","message"}]}`) |
| `POST` | `/api/bot/chat` | `{"message": "hi"}` | send chat (respects `agent.max_chat_length`) |
| `POST` | `/api/bot/command` | `{"command": "list"}` | run a server command (`/` is added if missing, needs op) |
| `POST` | `/api/bot/action` | `{"action": "wander", "params": {}}` **or** `{"raw": "{...}"}` | run one JSON action immediately |

### AI (Ollama) loop

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/ai/status` | running flag, decision/error counters, last decision |
| `POST` | `/api/ai/start` | start the decision loop (409 if the bot is not connected) |
| `POST` | `/api/ai/stop` | stop the loop (aborts the current action) |
| `POST` | `/api/ai/step` | decide and execute exactly one action now |

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
