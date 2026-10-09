"""A small client for Cortex LLMHoster (https://github.com/codero-sus/Cortex_LLMHoster).

Cortex hosts local models (llama.cpp GGUF, or any local engine through its command
runtime) behind an **OpenAI-compatible** API, by default on ``http://127.0.0.1:8624``:

* ``GET  /health``              - liveness, never needs a key
* ``GET  /ready``               - 200 once the default local model is running, else 503
* ``GET  /v1/models``           - model ids + declared capabilities
* ``POST /v1/chat/completions`` - chat (the brain uses this)
* ``POST /v1/completions``      - plain completion (fallback for text-only runtimes)

When the Cortex install sets ``CORTEX_API_KEY`` every ``/v1`` call needs
``Authorization: Bearer <key>``. That key is a *local* access secret, so PyMC_Bot never
stores it: the config only names the environment variable to read it from, mirroring
Cortex's own "no secrets in config files" rule.

Cortex is a separate, source-available program (personal, non-commercial license);
PyMC_Bot only talks to a running copy over HTTP and does not bundle any of it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8624"
DEFAULT_API_PATH = "/v1"
DEFAULT_KEY_ENV = "CORTEX_API_KEY"
DEFAULT_TIMEOUT = 120.0

# Statuses where "try the plain completions route instead" can help: the runtime may
# not implement chat (404/405/501) or reject chat-only fields such as response_format.
_FALLBACK_STATUSES = {400, 404, 405, 422, 501}


class CortexError(RuntimeError):
    """Raised when Cortex cannot be reached or answers with an error."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _error_message(response: httpx.Response) -> str:
    """The human message of an OpenAI-style ``{"error": {"message": ...}}`` body."""
    try:
        data = response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text[:200]
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    if isinstance(error, str):
        return error
    return response.text[:200]


def _content_text(content: Any) -> str:
    """OpenAI content is either a string or a list of ``{"type": "text"}`` parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text", "output_text")
        ]
        return "".join(parts)
    return ""


class CortexClient:
    """Talks to a running Cortex LLMHoster over its OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "",
        *,
        api_path: str = DEFAULT_API_PATH,
        api_key: str | None = None,
        api_key_env: str = DEFAULT_KEY_ENV,
        temperature: float = 0.4,
        max_tokens: int = 256,
        json_mode: bool = True,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_path = "/" + api_path.strip("/") if api_path.strip("/") else ""
        self._api_key = api_key
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.timeout = timeout
        self.last_model: str | None = None
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    # ---------------------------------------------------------------- helpers
    @property
    def api_key(self) -> str | None:
        """An explicit key wins; otherwise read the named environment variable."""
        if self._api_key:
            return self._api_key
        if self.api_key_env:
            value = os.environ.get(self.api_key_env, "").strip()
            return value or None
        return None

    @property
    def api_base(self) -> str:
        return f"{self.base_url}{self.api_path}"

    def _headers(self) -> dict[str, str]:
        key = self.api_key
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _request(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
        auth: bool = True,
    ) -> httpx.Response:
        try:
            response = self._client.request(
                method,
                url,
                json=payload,
                headers=self._headers() if auth else None,
                timeout=timeout or self.timeout,
            )
        except httpx.HTTPError as exc:
            raise CortexError(f"Could not reach Cortex at {self.base_url}: {exc}") from exc
        if response.status_code == 401:
            hint = (
                f"set {self.api_key_env} to this Cortex install's key"
                if not self.api_key
                else f"the key in {self.api_key_env or 'the config'} was rejected"
            )
            raise CortexError(f"Cortex needs a bearer key: {hint}", status=401)
        if response.status_code >= 400:
            raise CortexError(
                f"Cortex returned HTTP {response.status_code}: {_error_message(response)}",
                status=response.status_code,
            )
        return response

    def _json(self, response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise CortexError(f"Cortex returned invalid JSON: {response.text[:200]}") from exc
        if not isinstance(data, dict):
            raise CortexError("Cortex returned an unexpected payload")
        return data

    # ------------------------------------------------------------------- API
    def health(self, timeout: float = 5.0) -> bool:
        """``GET /health`` - is the Cortex process up at all? Never raises."""
        try:
            self._request("GET", f"{self.base_url}/health", timeout=timeout, auth=False)
        except CortexError:
            return False
        return True

    def ready(self, timeout: float = 5.0) -> tuple[bool, dict[str, Any]]:
        """``GET /ready`` - is the default local model running? Never raises."""
        try:
            response = self._client.get(f"{self.base_url}/ready", timeout=timeout)
        except httpx.HTTPError as exc:
            return False, {"status": "unreachable", "reason": str(exc)}
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            body = {"status": "unknown", "reason": response.text[:120]}
        if not isinstance(body, dict):
            body = {"status": "unknown"}
        body["http_status"] = response.status_code
        # 404: a plain OpenAI-compatible server (not Cortex) without a readiness probe.
        return response.status_code < 400 or response.status_code == 404, body

    def list_models(self, timeout: float = 10.0) -> list[dict[str, Any]]:
        """``GET /v1/models`` -> ``[{"id", "runtime", "capabilities", ...}]``."""
        data = self._json(self._request("GET", f"{self.api_base}/models", timeout=timeout))
        return [entry for entry in data.get("data", []) if isinstance(entry, dict)]

    def ping(self, timeout: float = 5.0) -> tuple[bool, str]:
        """Returns ``(usable, detail)`` - never raises.

        Usable means: the API answers (with the key, when one is required), the model
        the bot will use exists and can generate text, and Cortex reports it ready.
        """
        try:
            models = self.list_models(timeout=timeout)
        except CortexError as exc:
            return False, str(exc)
        if not models:
            return False, "Cortex is up but has no models configured"
        ids = [str(entry.get("id", "?")) for entry in models]
        listing = f"{len(ids)} model(s): {', '.join(ids[:6])}"
        if self.model and self.model not in ids:
            return False, f"model '{self.model}' is not configured in Cortex ({listing})"
        if self.model:
            chosen = next(entry for entry in models if entry.get("id") == self.model)
            capabilities = chosen.get("capabilities") or []
            if capabilities and "text_generation" not in capabilities:
                return False, f"model '{self.model}' does not declare text_generation ({listing})"
        ok, body = self.ready(timeout=timeout)
        if not ok:
            reason = body.get("reason") or body.get("status") or "not ready"
            model = body.get("model")
            return False, f"Cortex is up but not ready: {reason}" + (f" ({model})" if model else "") + f"; {listing}"
        return True, listing

    def chat(
        self,
        messages: Iterable[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> str:
        """``POST /v1/chat/completions`` -> the assistant text.

        A blank model lets Cortex pick its default model. With ``json_mode`` the
        request asks for a JSON object; runtimes that reject that field get the same
        request again without it.
        """
        chat_messages = list(messages)
        if system:
            chat_messages = [{"role": "system", "content": system}, *chat_messages]
        payload: dict[str, Any] = {
            "messages": chat_messages,
            "stream": False,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }
        chosen = model if model is not None else self.model
        if chosen:
            payload["model"] = chosen
        url = f"{self.api_base}/chat/completions"
        if self.json_mode:
            try:
                response = self._request(
                    "POST", url, payload={**payload, "response_format": {"type": "json_object"}}, timeout=timeout
                )
            except CortexError as exc:
                if exc.status not in (400, 422):
                    raise
                response = self._request("POST", url, payload=payload, timeout=timeout)
        else:
            response = self._request("POST", url, payload=payload, timeout=timeout)
        data = self._json(response)
        self.last_model = str(data.get("model") or chosen or "") or None
        choices = data.get("choices") or []
        message = (choices[0] or {}).get("message") if choices else None
        content = _content_text((message or {}).get("content"))
        if not content.strip():
            raise CortexError("Cortex returned an empty response")
        return content

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> str:
        """``POST /v1/completions`` for runtimes that only do plain completion."""
        text = f"{system}\n\n{prompt}" if system else prompt
        payload: dict[str, Any] = {
            "prompt": text,
            "stream": False,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }
        chosen = model if model is not None else self.model
        if chosen:
            payload["model"] = chosen
        data = self._json(self._request("POST", f"{self.api_base}/completions", payload=payload, timeout=timeout))
        self.last_model = str(data.get("model") or chosen or "") or None
        choices = data.get("choices") or []
        content = str((choices[0] or {}).get("text") or "") if choices else ""
        if not content.strip():
            raise CortexError("Cortex returned an empty completion")
        return content

    def decide(self, prompt: str, system: str | None = None) -> str:
        """Chat, falling back to plain completion when the runtime cannot chat."""
        try:
            return self.chat([{"role": "user", "content": prompt}], system=system)
        except CortexError as exc:
            if exc.status not in _FALLBACK_STATUSES:
                raise
            return self.complete(prompt, system=system)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> CortexClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def client_from_settings(settings: Any, timeout: float | None = None) -> CortexClient:
    """Build a client from a :class:`pymc_bot.config.CortexSettings`."""
    return CortexClient(
        base_url=settings.base_url,
        model=settings.model,
        api_path=settings.api_path,
        api_key_env=settings.api_key_env,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        json_mode=settings.json_mode,
        timeout=timeout or settings.request_timeout,
    )
