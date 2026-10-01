"""A very small Ollama HTTP client (stdlib + httpx, no ollama package needed)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import httpx

DEFAULT_TIMEOUT = 60.0


class OllamaError(RuntimeError):
    """Raised when Ollama cannot be reached or returns something unexpected."""


class OllamaClient:
    """Talks to ``POST /api/chat`` and ``POST /api/generate``."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "llama3.2",
        temperature: float = 0.4,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    # ---------------------------------------------------------------- helpers
    def _post(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = self._client.post(url, json=payload, timeout=timeout or self.timeout)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise OllamaError(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")
        try:
            return response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise OllamaError(f"Ollama returned invalid JSON: {response.text[:200]}") from exc

    def _get(self, path: str, timeout: float | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = self._client.get(url, timeout=timeout or min(self.timeout, 10.0))
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise OllamaError(f"Ollama returned HTTP {response.status_code}")
        try:
            return response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise OllamaError("Ollama returned invalid JSON") from exc

    # ------------------------------------------------------------------- API
    def ping(self, timeout: float = 5.0) -> tuple[bool, str]:
        """Returns ``(reachable, detail)`` - never raises."""
        try:
            data = self._get("/api/tags", timeout=timeout)
        except OllamaError as exc:
            return False, str(exc)
        names = [model.get("name", "?") for model in data.get("models", [])]
        return True, f"{len(names)} model(s): {', '.join(names[:6])}" if names else "no models pulled yet"

    def list_models(self, timeout: float = 10.0) -> list[dict[str, Any]]:
        data = self._get("/api/tags", timeout=timeout)
        return list(data.get("models", []))

    def chat(
        self,
        messages: Iterable[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": self.temperature if temperature is None else temperature},
        }
        if system:
            payload["messages"] = [{"role": "system", "content": system}, *payload["messages"]]
        data = self._post("/api/chat", payload, timeout=timeout or self.timeout)
        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OllamaError("Ollama returned an empty response")
        return content

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature if temperature is None else temperature},
        }
        if system:
            payload["system"] = system
        data = self._post("/api/generate", payload, timeout=timeout or self.timeout)
        content = data.get("response")
        if not isinstance(content, str) or not content.strip():
            raise OllamaError("Ollama returned an empty response")
        return content

    def pull(self, model: str | None = None, timeout: float = 600.0) -> dict[str, Any]:
        """Download/pull a model (`POST /api/pull`)."""
        return self._post("/api/pull", {"model": model or self.model, "stream": False}, timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
