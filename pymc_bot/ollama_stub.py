"""An Ollama- and OpenAI-compatible stub server.

This is **not** a language model.  It answers ``/api/tags``, ``/api/chat`` and
``/api/generate`` (Ollama) plus ``/health``, ``/v1/models``,
``/v1/chat/completions`` and ``/v1/completions`` (OpenAI wire format) with canned,
rule-based JSON decisions so that the full AI loop can be demonstrated and
unit-tested without installing Ollama or downloading model weights.

    python -m pymc_bot.ollama_stub --port 11434

Point ``ollama.base_url`` at it (that is the default port anyway) and the bot
will "think" with the built-in rules instead of the real model. The OpenAI routes
let Cortex LLMHoster supervise it through its ``command`` runtime, which is how
the Cortex integration is demonstrated end to end (see docs/CORTEX.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

STUB_MODEL = "pymc-stub:latest"


class StubBrain:
    """Deterministic rule-based 'model' used by the stub server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._step = 0
        self._last_replied: str | None = None

    def decide(self, prompt: str) -> str:
        """Rule-based reply: reacts to the state, then rotates through actions."""
        state = _extract_state(prompt)
        with self._lock:
            self._step += 1
            cycle = self._step % 6
        health = float(state.get("health") or 20)
        food = float(state.get("food") or 20)
        players = state.get("players") or []
        chat = [str(line) for line in (state.get("recent_chat") or [])]
        position = state.get("position") or {"x": 0, "y": 64, "z": 0}
        bot_name = str((state.get("server") or {}).get("username") or "PyMC_Bot")

        # Reactive rules first: stay alive, answer people who talk to us.
        if health <= 8 or food <= 6:
            return _json({"action": "eat"})
        others = [line for line in chat[-4:] if not line.startswith(f"<{bot_name}>")]
        if others:
            latest = others[-1]
            addressed = "?" in latest or bot_name.lower() in latest.lower()
            # Answer each message once - never reply to the same line twice.
            if addressed and latest != self._last_replied:
                self._last_replied = latest
                return _json({"action": "say", "message": "Good question - I'm just a stub brain, but hello!"})

        # Then a deterministic rotation so a demo shows every kind of action.
        if cycle == 0 and players and players[0].get("distance", 99) < 20:
            return _json({"action": "say", "message": f"Hey {players[0].get('name', 'there')}!"})
        if cycle == 1:
            return _json({"action": "mine", "block": "oak_log"})
        if cycle == 2:
            target = {
                "x": round(float(position.get("x", 0)) + 12, 1),
                "y": round(float(position.get("y", 64)), 1),
                "z": round(float(position.get("z", 0)) - 9, 1),
            }
            return _json({"action": "goto", **target})
        if cycle == 3 and players:
            return _json({"action": "follow", "player": players[0].get("name", "Steve")})
        if cycle == 4:
            return _json({"action": "mine", "block": "stone"})
        return _json({"action": "wander"})


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def _last_user_message(messages: Any) -> str:
    """Text of the newest user message (string or OpenAI content parts)."""
    for message in reversed(messages if isinstance(messages, list) else []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        return str(content or "")
    return ""


def _extract_state(prompt: str) -> dict[str, Any]:
    start, end = prompt.find("{"), prompt.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        data = json.loads(prompt[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


class Handler(BaseHTTPRequestHandler):
    server_version = "PyMCBotOllamaStub/0.1"
    brain = StubBrain()

    # ------------------------------------------------------------- utilities
    def _send(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib signature
        sys.stderr.write("[ollama-stub] " + (fmt % args) + "\n")

    # ---------------------------------------------------------------- routes
    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/api/tags", "/api/tags/"):
            models = [{"name": STUB_MODEL, "model": STUB_MODEL, "size": 1, "details": {"family": "stub"}}]
            for name in self.server.created_models:
                models.append(
                    {
                        "name": name,
                        "model": name,
                        "size": 12,
                        "details": {"family": "pymc-trained", "note": "created via /api/create"},
                    }
                )
            self._send({"models": models})
        elif path == "/health":
            self._send({"status": "ok", "stub": True})
        elif path == "/v1/models":
            self._send(
                {
                    "object": "list",
                    "data": [{"id": STUB_MODEL, "object": "model", "created": 0, "owned_by": "pymc-stub"}],
                }
            )
        elif path == "/api/version":
            self._send({"version": "0.0.0-pymc-stub"})
        elif path == "/":
            self._send({"stub": True, "hint": "Ollama-compatible stub. Not a language model."})
        else:
            self._send({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        path = self.path.split("?")[0].rstrip("/")
        payload = self._read_json()
        model = str(payload.get("model") or STUB_MODEL)
        if path == "/v1/chat/completions":
            content = self.brain.decide(_last_user_message(payload.get("messages")))
            self._send(
                {
                    "id": "chatcmpl-pymc-stub",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
            )
        elif path == "/v1/completions":
            content = self.brain.decide(str(payload.get("prompt") or ""))
            self._send(
                {
                    "id": "cmpl-pymc-stub",
                    "object": "text_completion",
                    "created": 0,
                    "model": model,
                    "choices": [{"index": 0, "text": content, "finish_reason": "stop"}],
                }
            )
        elif path == "/api/chat":
            content = self.brain.decide(_last_user_message(payload.get("messages")))
            self._send(
                {
                    "model": model,
                    "created_at": "",
                    "message": {"role": "assistant", "content": content},
                    "done": True,
                    "done_reason": "stop",
                }
            )
        elif path == "/api/generate":
            prompt = str(payload.get("prompt") or "")
            self._send({"model": model, "response": self.brain.decide(prompt), "done": True})
        elif path == "/api/pull":
            self._send({"status": "success"})
        elif path == "/api/create":
            # The demo stub "creates" the model: remember its name so /api/tags lists it.
            created = str(payload.get("model") or "pymc-trained")
            self.server.created_models[created] = {
                "modelfile": str(payload.get("modelfile") or ""),
                "from": payload.get("from"),
                "system": payload.get("system"),
            }
            self._send({"status": "success", "model": created, "created": created})
        elif path == "/api/delete":
            self.server.created_models.pop(str(payload.get("model") or ""), None)
            self._send({"status": "success"})
        else:
            self._send({"error": "not found"}, status=404)


class StubServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that remembers models "created" through /api/create."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.created_models: dict[str, dict[str, Any]] = {}


def serve(host: str = "127.0.0.1", port: int = 11434) -> ThreadingHTTPServer:
    httpd = StubServer((host, port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, name="ollama-stub", daemon=True)
    thread.start()
    return httpd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ollama/OpenAI-compatible stub server for PyMC_Bot demos/tests (NOT a real LLM)."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--quiet", action="store_true", help="silence request logging")
    args = parser.parse_args(argv)
    if args.quiet:
        Handler.log_message = lambda *a, **k: None  # type: ignore[assignment]
    httpd = StubServer((args.host, args.port), Handler)
    print(f"[ollama-stub] listening on http://{args.host}:{args.port} (model: {STUB_MODEL}) - NOT a real LLM")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[ollama-stub] stopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
