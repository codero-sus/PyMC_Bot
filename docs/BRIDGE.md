# Python ↔ Node bridge contract

`pymc_bot/node/minecraft_bridge.js` is the only JavaScript in PyMC_Bot. It wraps
[mineflayer](https://github.com/PrismarineJS/mineflayer) — the mature Minecraft client library, which only
exists for Node.js — into a tiny JSON-RPC-ish process that Python drives.

* The bridge is started by `NodeBridgeBackend` (`pymc_bot/backends.py`) as a child process
  (`node pymc_bot/node/minecraft_bridge.js`).
* Communication is **newline-delimited JSON**: one JSON object per line, requests on stdin, replies and
  events on stdout. Everything the bridge prints goes through the protocol — `console.log` is redirected so
  stray output can never corrupt the stream (Node warnings land as `{"event":"log"}` frames).
* All bot behaviour (path planning, goal selection, AI decisions, permissions) stays in Python. The bridge
  only translates primitives.

## Requests (Python → Node)

```json
{"id": 7, "cmd": "control", "params": {"name": "forward", "state": true}}
```

Every request must carry an `id`; the bridge answers exactly once with the same `id`:

```json
{"id": 7, "ok": true,  "result": {"control": "forward", "state": true}}
{"id": 7, "ok": false, "error": "unknown control: teleport"}
```

| `cmd` | `params` | `result` | Notes |
| --- | --- | --- | --- |
| `connect` | `host`, `port`, `username`, `version` (null = sniff), `auth` (`offline`/`microsoft`), `view_distance`, `profiles_folder` | `{"connecting": true}` | replies immediately; the bot spawns later as an event. For `microsoft`, `username` is the account **email** and `profiles_folder` is the token cache (Python gives every account its own sub-folder) |
| `say` | `text` | `{"sent": true}` | chat message |
| `command` | `text` | `{"sent": true}` | server command (`/` added if missing) |
| `control` | `name` ∈ `forward back left right jump sneak sprint`, `state` | `{"control": ...}` | held keys |
| `look` | `yaw`, `pitch` (radians), `force` | `{"looked": true}` | `yaw 0` = +Z, `pitch -π/2` = up |
| `stop_motion` | – | `{"stopped": true}` | releases every key and cancels pathfinding |
| `pathfind_to` | `x`, `y`, `z`, `range` | `{"goal": true}` | fails with `pathfinder_unavailable` when mineflayer-pathfinder is missing |
| `stop_path` | – | `{"stopped": true}` | clear the current A* goal |
| `dig` | `block`, `timeout` (ms) | `{"dug": true, "block": "oak_log"}` or `{"dug": false, "reason": "..."}` | walks to the nearest matching block, then digs |
| `eat` | – | `{"ate": true, "item": "bread"}` | equips and consumes the first food item |
| `attack` | `player` | `{"attacked": "Steve"}` | only called when the AI is allowed to attack |
| `jump` | – | `{"jumped": true}` | short hop |
| `disconnect` | – | `{"disconnected": true}` | graceful quit |
| `ping` | – | `{"pong": true, "connected": bool}` | liveness |
| `state` | – | world snapshot | on demand; the bridge also pushes it periodically |

## Events (Node → Python, no `id`)

| `event` | Payload | Meaning |
| --- | --- | --- |
| `ready` | `caps: {"pathfinder": bool, "mineflayer": bool, "auth": ["offline","microsoft"], "multiple_bots": "..."}`, `node` | sent once at startup; `start()` waits for it |
| `log` | `level`, `message` | bridge log line, forwarded into the event stream |
| `state` | `state` | world snapshot, pushed every 250 ms while connected |
| `chat` | `username`, `message` | someone talked; Python logs it, greets new players |
| `msa_code` | `user_code`, `verification_uri`, `expires_in`, `message` | **premium login**: the device code the user must enter (Python logs it at level `auth` and keeps it in the event data) |
| `login` | `username` | authenticated and logged in (for premium this is the real in-game name) |
| `spawn` | – | the bot is in the world → `connect()` returns |
| `end` | `reason` | clean disconnect / TCP close |
| `kicked` | `reason` | server kicked the bot (whitelist, ban, …) |
| `error` | `message` | connection error, uncaught exception, unhandled rejection |

The snapshot fields mirror the REST `bot` object: `status, username, position, yaw, pitch, health, food,
dimension, time_of_day, players[], inventory[]`. For premium accounts `username` is the **in-game name**
owned by the account, which is not the email used to log in. Python merges each frame into its cached snapshot, so `GET
/api/status` never blocks on the child process.

## Failure handling

* A command that gets no reply within its timeout raises `BackendError`; fire-and-forget calls (`say`,
  `control`, …) only log a warning so a hiccup cannot kill the AI loop.
* `connect` raises when the bridge reports an `error`/`kicked`/`end` before `spawn`, including the server's
  reason (`"Timed out ..."`, `"getaddrinfo ENOTFOUND ..."`, `"You are not whitelisted"`, …).
* A premium login that fails surfaces the server's own reason (`"Failed to obtain profile data ... does the
  account own minecraft?"`), which Python raises as `BackendError` and the panel shows next to the player.
* If the child process dies, `NodeBridgeBackend.connected` flips to `false`, the snapshot switches to
  `disconnected` and `MinecraftBot`'s monitor thread reconnects (3 attempts, 5s/10s/15s backoff) unless
  `auto_reconnect=False`.

## Testing without Minecraft

`tests/fixtures/fake_bridge.js` implements this whole protocol with canned data, so the Python adapter is
covered in CI without mineflayer or a server (`tests/test_node_bridge.py`). The real bridge is exercised by
`tests/test_node_live.py`, which is opt-in via `PYMC_TEST_SERVER=host:port`.
