# Using Cortex LLMHoster as the AI brain

[Cortex LLMHoster](https://github.com/codero-sus/Cortex_LLMHoster) hosts models **on your own machine**
(llama.cpp GGUF files, or any local engine through its `command` runtime) behind an
**OpenAI-compatible API**. PyMC_Bot can use it as the brain instead of, or alongside, Ollama:

| Brain (`agent.mode`) | Who decides |
| --- | --- |
| `auto` | Ollama when `ollama.enabled`, else **Cortex when `cortex.enabled`**, else the built-in heuristic |
| `cortex` | always the model hosted by Cortex (falls back to the heuristic whenever a request fails) |
| `ollama`, `heuristic`, `trained` | unchanged |

Fleet bots can use it as well: pick **cortex** as *Fleet AI* / premium AI, or send `"ai": "cortex"` to
`POST /api/fleet/spawn` and `/api/fleet/populate`.

> Cortex is a separate, source-available program under its own **personal, non-commercial** license.
> PyMC_Bot does not bundle, modify or redistribute it; it only talks to a copy you run, over HTTP.

## 1. Run Cortex

Follow Cortex's own installation guide. In short (Python 3.11+, plus `llama-server` and a GGUF model):

```bash
git clone https://github.com/codero-sus/Cortex_LLMHoster && cd Cortex_LLMHoster
python3.11 -m venv .venv && . .venv/bin/activate && pip install .
cp cortex.example.toml cortex.toml          # set model_path to your .gguf
export CORTEX_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m cortex_llmhoster                   # -> http://127.0.0.1:8624
```

A small instruct model (for example a 1.5B–8B Qwen or Llama Q4_K_M GGUF) is enough: the bot asks for one
short JSON action every few seconds. CPU-only works (`gpu_layers = 0`).

## 2. Point PyMC_Bot at it

The bearer key is **never stored** in `pymc_bot_config.json`. PyMC_Bot reads it from the environment
variable named in `cortex.api_key_env` (default `CORTEX_API_KEY`, the same name Cortex uses), so start the
bot with that variable set:

```bash
export CORTEX_API_KEY=...                    # the same value Cortex was started with
python -m pymc_bot doctor                    # cortex : http://127.0.0.1:8624/v1 -> OK (1 model(s): qwen-local; ...)
python -m pymc_bot serve                     # panel: AI brain -> Cortex LLMHoster -> Check Cortex / Use Cortex as brain
# or headless
python -m pymc_bot run --think cortex [--cortex-url http://127.0.0.1:8624] [--cortex-model qwen-local]
```

If Cortex runs without a key, leave the variable unset.

```jsonc
"cortex": {
  "enabled": false,                 // let `auto` use Cortex (Ollama keeps priority when both are on)
  "base_url": "http://127.0.0.1:8624",
  "api_path": "/v1",                // Cortex's api.json `base_path`
  "model": "",                      // blank = the model Cortex marks as default
  "api_key_env": "CORTEX_API_KEY",  // NAME of the env var holding the key - never the key itself
  "temperature": 0.4,
  "max_tokens": 256,
  "json_mode": true,                // send response_format {"type":"json_object"}; dropped automatically if refused
  "decision_interval": 6.0,
  "request_timeout": 120.0,
  "system_prompt": "..."            // same action contract as the Ollama brain
}
```

## How the brain talks to Cortex

* **Health** – `GET /v1/models` (with the key) checks the configured model exists and declares
  `text_generation`; `GET /ready` checks that the default local model is actually running. The panel's
  `cortex` pill and `GET /api/cortex/health` show the exact reason when something is missing
  (no key, unknown model, model not started, Cortex not running).
* **Decisions** – `POST /v1/chat/completions` with the system prompt and the compact world state, the
  same prompt the Ollama brain uses. The reply goes through the same parser and permission gates, so a
  hosted model can never attack or mine when the config forbids it.
* **Fallbacks** – a runtime that refuses `response_format` gets the request again without it; a runtime
  without chat (404/405/422/501) is asked through `POST /v1/completions`. Network and auth errors are not
  retried: that decision falls back to the heuristic, the error shows in the panel, and the next decision
  tries Cortex again.

## Try the chain without model weights

The demo stub (`python -m pymc_bot stub`, rule-based, **not** a language model) also speaks the OpenAI
dialect (`/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`), so Cortex can supervise it
through its `command` runtime. This is how the integration was tested end to end:

```toml
# cortex.toml
[server]
host = "127.0.0.1"
port = 8624
api_key_env = "CORTEX_API_KEY"
default_model = "pymc-demo"

[[models]]
id = "pymc-demo"
runtime = "command"
api_base_path = "/v1"
health_path = "/health"
capabilities = ["text_generation"]
default = true
runtime_command = ["/path/to/PyMC_Bot/.venv/bin/python", "-m", "pymc_bot", "stub",
                   "--host", "{host}", "--port", "{port}", "--quiet"]
```

```bash
CORTEX_CONFIG=cortex.toml CORTEX_API_KEY=demo python -m cortex_llmhoster &
CORTEX_API_KEY=demo python -m pymc_bot run --backend simulated --think cortex --seconds 15
# Thinking with Cortex LLMHoster at http://127.0.0.1:8624: 1 model(s): pymc-demo
# [ai] agent: AI chose: mine {'block': 'oak_log'}
# [info] agent: ✓ mined oak_log
```

Swap the `command` model for a real `llama.cpp` GGUF model and nothing on the PyMC_Bot side changes.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Cortex needs a bearer key: set CORTEX_API_KEY ...` | export the key in the shell that starts PyMC_Bot (or change `cortex.api_key_env`) |
| `the key in CORTEX_API_KEY was rejected` | the value differs from the one Cortex was started with |
| `not ready: default_local_model_not_running` | start the model in the Cortex dashboard, or check its runtime log (bad `model_path`, missing `llama-server`) |
| `model 'x' is not configured in Cortex` | use an id from `GET /api/cortex/models`, or leave `cortex.model` blank for Cortex's default |
| `does not declare text_generation` | that model is e.g. a transcription model; pick a text model |
| Decisions are slow | lower `max_tokens`, use a smaller/quantised GGUF, or raise `cortex.decision_interval` |
| `Could not reach Cortex` from another machine | Cortex binds to `127.0.0.1` by default; run PyMC_Bot on the same host or set Cortex's host deliberately |
