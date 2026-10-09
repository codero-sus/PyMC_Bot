"""A tiny thread-safe event log with async subscribers for the WebSocket UI.

Background bot/agent threads push :class:`Event` objects; the FastAPI layer
subscribes and forwards them to browsers.  Events are also kept in a bounded
ring buffer so ``/api/logs`` can serve history after a page reload.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

MAX_EVENTS = 500


@dataclass
class Event:
    level: str
    message: str
    source: str = "app"
    ts: float = field(default_factory=time.time)
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["time"] = time.strftime("%H:%M:%S", time.localtime(self.ts))
        return payload


class EventLog:
    """Bounded, thread-safe log that can broadcast to asyncio subscribers."""

    def __init__(self, maxlen: int = MAX_EVENTS) -> None:
        self._events: deque[Event] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._subscribers: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._next_id = 0
        self._listeners: list[Callable[[Event], None]] = []

    # ---------------------------------------------------------------- write
    def add(
        self,
        message: str,
        level: str = "info",
        source: str = "app",
        data: dict[str, Any] | None = None,
    ) -> Event:
        event = Event(level=level, message=message, source=source, data=data)
        with self._lock:
            self._events.append(event)
            listeners = list(self._listeners)
            subscribers = list(self._subscribers.values())
        for listener in listeners:
            try:
                listener(event)
            except Exception:  # pragma: no cover - listener bugs must not kill the bot
                pass
        for loop, queue in subscribers:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                # Loop closed (browser/thread went away) - ignore.
                pass
        return event

    # ----------------------------------------------------------------- read
    def tail(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._events)[-max(1, min(limit, MAX_EVENTS)) :]
        return [event.to_dict() for event in events]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    # -------------------------------------------------------------- publish
    def add_listener(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        """Register a synchronous callback. Returns an unsubscribe function."""
        with self._lock:
            self._listeners.append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                if callback in self._listeners:
                    self._listeners.remove(callback)

        return _unsubscribe

    def subscribe(
        self, loop: asyncio.AbstractEventLoop, maxsize: int = 200
    ) -> tuple[int, asyncio.Queue]:
        """Register an asyncio queue fed by events (used by the WebSocket)."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        with self._lock:
            sub_id = self._next_id
            self._next_id += 1
            self._subscribers[sub_id] = (loop, queue)
        return sub_id, queue

    def unsubscribe(self, sub_id: int) -> None:
        with self._lock:
            self._subscribers.pop(sub_id, None)
