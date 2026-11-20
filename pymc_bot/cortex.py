"""A small client for Cortex LLMHoster (https://github.com/codero-sus/Cortex_LLMHoster).

Cortex hosts local models behind an **OpenAI-compatible** API, by default on
``http://127.0.0.1:8624``:

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

import base64
import binascii
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



def image_data_uri(image: str) -> str:
    """base64 picture -> ``data:`` URI (JPEG unless the bytes say PNG); URIs pass through."""
    if image.startswith("data:"):
        return image
    mime = "image/png" if image.startswith("iVBOR") else "image/jpeg"
    return f"data:{mime};base64,{image}"

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
        send_temperature: bool = True,
        max_tokens: int = 256,
        json_mode: bool = True,
        health_mode: str = "cortex",
        protocol: str = "openai",
        token_parameter: str = "max_tokens",
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_path = "/" + api_path.strip("/") if api_path.strip("/") else ""
        self._api_key = api_key
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.send_temperature = send_temperature
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.health_mode = health_mode
        self.protocol = protocol
        self.token_parameter = token_parameter
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
        if self.protocol == "anthropic":
            headers = {"anthropic-version": "2023-06-01"}
            if key:
                headers["x-api-key"] = key
            return headers
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
                follow_redirects=False,
            )
        except httpx.ConnectError as exc:
            raise CortexError(
                f"Could not reach Cortex at {self.base_url}: nothing is listening there (is Cortex running?)"
            ) from exc
        except httpx.HTTPError as exc:
            raise CortexError(f"Could not reach Cortex at {self.base_url}: {exc}") from exc
        if response.status_code == 401:
            hint = (
                f"set the {self.api_key_env} environment variable"
                if not self.api_key
                else f"the key in {self.api_key_env or 'the client'} was rejected"
            )
            raise CortexError(f"AI endpoint rejected authentication: {hint}", status=401)
        if 300 <= response.status_code < 400:
            raise CortexError("AI endpoint returned a redirect; redirects are not followed", status=response.status_code)
        if response.status_code >= 400:
            detail = _error_message(response)
            if self.api_key:
                detail = detail.replace(self.api_key, "[redacted]")
            raise CortexError(
                f"AI endpoint returned HTTP {response.status_code}: {detail}",
                status=response.status_code,
            )
        return response

    def _json(self, response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            # Providers can echo request headers or secrets in error bodies; do not log them.
            raise CortexError("AI endpoint returned invalid JSON") from exc
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
            return False, "Endpoint answered but returned no models; chat access is not verified"
        ids = [str(entry.get("id", "?")) for entry in models]
        listing = f"{len(ids)} model(s): {', '.join(ids[:6])}"
        if self.health_mode == "compatible":
            return True, f"models endpoint reachable; chat/model access not verified ({listing})"
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

    def _anthropic_chat(
        self, messages: Iterable[dict[str, str]], system: str | None, model: str | None,
        temperature: float | None, timeout: float | None, images: list[str] | None,
    ) -> str:
        chosen = model if model is not None else self.model
        if not chosen:
            raise CortexError("Anthropic Messages requires a model ID")
        temp = self.temperature if temperature is None else temperature
        if self.send_temperature and not 0 <= temp <= 1:
            raise CortexError("Anthropic temperature must be between 0 and 1")
        system_parts = [system] if system else []
        chat_messages: list[dict[str, Any]] = []
        for message in messages:
            role, content = message.get("role"), message.get("content")
            if not isinstance(content, str):
                raise CortexError("Anthropic Messages requires text messages")
            if role == "system":
                system_parts.append(content)
            elif role in ("user", "assistant"):
                chat_messages.append({"role": role, "content": content})
            else:
                raise CortexError("Anthropic Messages supports user and assistant messages only")
        if images:
            target = next((m for m in reversed(chat_messages) if m["role"] == "user"), None)
            if target is None:
                raise CortexError("cannot attach images without a user message")
            parts: list[dict[str, Any]] = [{"type": "text", "text": target["content"]}]
            for image in images:
                uri = image_data_uri(image)
                prefix, sep, encoded = uri.partition(",")
                if not sep or prefix not in ("data:image/png;base64", "data:image/jpeg;base64"):
                    raise CortexError("Anthropic images must be base64 PNG or JPEG")
                try:
                    base64.b64decode(encoded, validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise CortexError("invalid base64 image") from exc
                parts.append({"type": "image", "source": {"type": "base64",
                              "media_type": prefix[5:].split(";", 1)[0], "data": encoded}})
            target["content"] = parts
        payload: dict[str, Any] = {"model": chosen, "messages": chat_messages, "stream": False,
                                   "max_tokens": self.max_tokens}
        if self.send_temperature:
            payload["temperature"] = temp
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        data = self._json(self._request("POST", f"{self.api_base}/messages", payload=payload, timeout=timeout))
        self.last_model = str(data.get("model") or chosen)
        content = _content_text(data.get("content"))
        if not content.strip():
            raise CortexError("Anthropic returned an empty response")
        return content

    def chat(
        self,
        messages: Iterable[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        images: list[str] | None = None,
    ) -> str:
        """``POST /v1/chat/completions`` -> the assistant text.

        ``images`` (base64 JPEG/PNG) are sent with the last user message as OpenAI-style
        ``image_url`` data URIs, which vision runtimes understand.

        A blank model lets Cortex pick its default model. With ``json_mode`` the
        request asks for a JSON object; runtimes that reject that field get the same
        request again without it.
        """
        if self.protocol == "anthropic":
            return self._anthropic_chat(messages, system, model, temperature, timeout, images)
        chat_messages = [dict(m) for m in messages]
        if images:
            for message in reversed(chat_messages):
                if message.get("role") == "user":
                    message["content"] = [
                        {"type": "text", "text": str(message.get("content") or "")},
                        *({"type": "image_url", "image_url": {"url": image_data_uri(image)}} for image in images),
                    ]
                    break
            else:
                raise CortexError("cannot attach images without a user message")
        if system:
            chat_messages = [{"role": "system", "content": system}, *chat_messages]
        payload: dict[str, Any] = {
            "messages": chat_messages,
            "stream": False,
            self.token_parameter: self.max_tokens,
        }
        if self.send_temperature:
            payload["temperature"] = self.temperature if temperature is None else temperature
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
            "max_tokens": self.max_tokens,
        }
        if self.send_temperature:
            payload["temperature"] = self.temperature if temperature is None else temperature
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

    def decide(self, prompt: str, system: str | None = None, images: list[str] | None = None) -> str:
        """Chat, falling back to plain completion when the runtime cannot chat."""
        try:
            return self.chat([{"role": "user", "content": prompt}], system=system, images=images)
        except CortexError as exc:
            if self.protocol == "anthropic" or images or exc.status not in _FALLBACK_STATUSES:
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
        send_temperature=settings.send_temperature,
        max_tokens=settings.max_tokens,
        json_mode=settings.json_mode,
        health_mode=settings.health_mode,
        protocol=settings.protocol,
        token_parameter=settings.token_parameter,
        timeout=timeout or settings.request_timeout,
    )
