# Using Cortex or an OpenAI-compatible endpoint as the AI brain

The existing `cortex` brain accepts OpenAI-compatible **chat completions** endpoints and
Anthropic's native **Messages** API and Google's native **generateContent** API. The panel provides presets for OpenAI, Anthropic,
Google Gemini's native and compatibility APIs, OpenRouter, Hugging Face Inference Providers, Groq,
DeepSeek, Mistral, Together AI, xAI, Perplexity, Cerebras, Fireworks, DeepInfra, NVIDIA NIM,
Qwen/DashScope, Moonshot, Z.ai, Zhipu, SiliconFlow, Cortex, Ollama, LM Studio, vLLM,
llama.cpp, LocalAI, Unsloth Studio and LiteLLM, plus custom URLs. Some services offer free tiers or models,
but availability, limits and prices change; local servers require their own model setup.
These presets are configuration conveniences, not endorsements or verified integrations:
provider features, accounts, model IDs, pricing, regions and API behavior vary. No live
provider integration was tested here. Azure-specific headers/authentication, Responses-only APIs and other non-compatible protocols are not
implemented; an OpenAI-compatible gateway may be an alternative.

Select a preset, apply it to the form, set a model ID and the **name** of a key environment
variable, then save and select the `cortex` brain. Start PyMC Bot with that variable set on
its host, not in your browser. Presets do not save or enable the AI. Never put API keys in
URLs, config files or chat. PyMC Bot refuses to send an API key over HTTP except to localhost or
literal loopback IPs. Use HTTPS for remote services, including LAN servers; without a key, HTTP
still sends observations unencrypted. This guard cannot protect against a malicious HTTPS provider,
compromised host, stolen environment variables or a panel administrator changing endpoint settings.
A remote provider receives prompts, gameplay observations and,
if explicitly enabled, images; check its terms, privacy policy and costs. The panel exposes
endpoint settings, so protect it with authentication and only grant trusted users access.

Compatible-mode health checks call `/models` only and do not verify that a model can generate
or that you have access to it. A failed or missing `/models` endpoint doesn't necessarily mean
chat is unavailable. Some providers do not support all request fields, JSON mode, images, or
plain-completion fallback; set `json_mode` off if needed. Models requiring
`max_completion_tokens` can select that instead of `max_tokens` in the panel; disable sending
`temperature` for models that reject it. Native Anthropic Messages uses
`x-api-key`, the `anthropic-version` header and its own message format; it requires a model ID
and temperature between 0 and 1, and does not use OpenAI JSON mode or completions fallback.
Images are limited to base64 PNG/JPEG. Native Gemini uses `x-goog-api-key`, requires a model ID,
and calls `/v1beta/models/{model}:generateContent`; system instructions, text, PNG/JPEG images,
JSON MIME mode and output-token limits are supported. It does not implement Gemini tools,
streaming, native thinking controls or completions fallback. Set `GEMINI_API_KEY` in the backend
environment and choose **Google Gemini (native generateContent)**, or use the compatibility
preset separately. Model availability and JSON mode vary. Native Gemini is mock-tested, not
verified against Google's live service. API failures fall back to the heuristic; no model is
guaranteed to behave safely or play well.

## Unsloth Studio

Start the Unsloth API and load a model. In **AI brain → Cortex / OpenAI-compatible endpoint**,
choose **Unsloth Studio**, apply the preset and change the URL port to the one Unsloth reports
(commonly 8000 or 8888). Enter the model ID and select `cortex` as the brain. On the machine
running PyMC Bot, set the environment variable named `UNSLOTH_STUDIO_AUTH_TOKEN` before launch
if your Unsloth server uses authentication; leave it unset for older keyless local installs.
Newer Unsloth versions document creating a key under Settings → API. Do not paste the key into
the panel, URL, or chat. The preset uses OpenAI-compatible `/v1/chat/completions`; if Unsloth
is on another machine, use a trusted HTTPS connection. Consult [Unsloth's API documentation](https://unsloth.ai/docs/basics/api)
for its current authentication and port. The keyless and authenticated Unsloth modes have not
been tested against a live Unsloth server in this environment.

[Cortex LLMHoster](https://github.com/codero-sus/Cortex_LLMHoster) hosts models **on your own machine**
behind an **OpenAI-compatible API**. PyMC_Bot can use it as the brain instead of, or alongside, Ollama:

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

Install and start Cortex LLMHoster on the same machine as PyMC_Bot, following Cortex's own documentation,
and load a text model in it. PyMC_Bot only needs two things from it:

* its **URL** – `http://127.0.0.1:8624` by default, with the OpenAI-compatible API under `/v1`;
* the **bearer key** Cortex was started with, if it uses one.

A small instruct model (roughly 1.5B–8B parameters, quantised) is enough: the bot asks for one short JSON
action every few seconds, and CPU-only works.

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
  "api_path": "/v1",                // the base path of Cortex's OpenAI-compatible API
  "model": "",                      // blank = the model Cortex marks as default
  "api_key_env": "CORTEX_API_KEY",  // NAME of the env var holding the key - never the key itself
  "health_mode": "cortex",          // or "compatible" for the /models-only probe
  "protocol": "openai",            // or "anthropic" / "gemini" for native APIs
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
  `text_generation`, and a readiness check confirms the model is actually running. The panel's
  `cortex` pill and `GET /api/cortex/health` show the exact reason when something is missing
  (no key, unknown model, model not started, Cortex not running).
* **Decisions** – `POST /v1/chat/completions` with the system prompt and the compact world state, the
  same prompt the Ollama brain uses. The reply goes through the same parser and permission gates, so a
  hosted model can never attack or mine when the config forbids it.
* **Fallbacks** – a runtime that refuses `response_format` gets the request again without it; a runtime
  without chat (404/405/422/501) is asked through `POST /v1/completions`. Network and auth errors are not
  retried: that decision falls back to the heuristic, the error shows in the panel, and the next decision
  tries Cortex again.

## Try the brain without model weights

The demo stub (`python -m pymc_bot stub`, rule-based, **not** a language model) also speaks the OpenAI
dialect (`/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`), so it can stand in for Cortex
and exercise the whole `cortex` brain path on any machine:

```bash
python -m pymc_bot stub --port 8624 &
python -m pymc_bot run --backend simulated --think cortex --cortex-url http://127.0.0.1:8624 --seconds 15
# Thinking with Cortex LLMHoster at http://127.0.0.1:8624: 1 model(s): pymc-stub:latest
# [ai] agent: AI chose: mine {'block': 'oak_log'}
# [info] agent: ✓ mined oak_log
```

Stop the stub and start Cortex with a real model on the same address: nothing on the PyMC_Bot side changes.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Cortex needs a bearer key: set CORTEX_API_KEY ...` | export the key in the shell that starts PyMC_Bot (or change `cortex.api_key_env`) |
| `the key in CORTEX_API_KEY was rejected` | the value differs from the one Cortex was started with |
| `not ready: ...` | the model is not running in Cortex yet – start it there and check Cortex's own log |
| `model 'x' is not configured in Cortex` | use an id from `GET /api/cortex/models`, or leave `cortex.model` blank for Cortex's default |
| `does not declare text_generation` | that model is e.g. a transcription model; pick a text model |
| Decisions are slow | lower `max_tokens`, use a smaller/quantised model, or raise `cortex.decision_interval` |
| `Could not reach Cortex` from another machine | Cortex binds to `127.0.0.1` by default; run PyMC_Bot on the same host or set Cortex's host deliberately |

---

<sub>Made with ❤️ by [codero-sus](https://github.com/codero-sus) · [codero.is-a.dev](https://codero.is-a.dev) · [the-codero.web.app](https://the-codero.web.app/)</sub>
