"""Event log tests (used by the WebSocket UI)."""

from __future__ import annotations

import asyncio

from pymc_bot.events import Event, EventLog


def test_add_and_tail():
    log = EventLog()
    log.add("first", "info", "test")
    log.add("second", "warn", "test")
    events = log.tail(10)
    assert [e["message"] for e in events] == ["first", "second"]
    assert events[0]["level"] == "info"
    assert "time" in events[0] and len(events[0]["time"].split(":")) == 3


def test_tail_limit_and_ring_buffer():
    log = EventLog(maxlen=5)
    for index in range(20):
        log.add(f"msg-{index}")
    events = log.tail(3)
    assert [e["message"] for e in events] == ["msg-17", "msg-18", "msg-19"]
    assert len(log.tail(100)) == 5


def test_listener_receives_events_and_unsubscribes():
    log = EventLog()
    seen: list[Event] = []
    unsubscribe = log.add_listener(seen.append)
    log.add("hello")
    unsubscribe()
    log.add("not seen")
    assert [e.message for e in seen] == ["hello"]


def test_broken_listener_does_not_break_logging():
    log = EventLog()

    def explode(_event: Event) -> None:
        raise RuntimeError("boom")

    log.add_listener(explode)
    log.add("still fine")
    assert log.tail(1)[0]["message"] == "still fine"


def test_async_subscribers_receive_events():
    async def scenario() -> list[str]:
        log = EventLog()
        loop = asyncio.get_running_loop()
        sub_id, queue = log.subscribe(loop)
        log.add("from another thread")
        event = await asyncio.wait_for(queue.get(), timeout=2.0)
        log.unsubscribe(sub_id)
        log.add("after unsubscribe")
        try:
            await asyncio.wait_for(queue.get(), timeout=0.2)
            extra = "unexpected second event"
        except asyncio.TimeoutError:
            extra = "none"
        return [event.message, extra]

    assert asyncio.run(scenario()) == ["from another thread", "none"]


def test_clear():
    log = EventLog()
    log.add("something")
    log.clear()
    assert log.tail() == []


def test_slow_subscriber_keeps_latest_events_without_loop_errors():
    async def scenario():
        loop = asyncio.get_running_loop()
        errors = []
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        try:
            log = EventLog()
            sub_id, queue = log.subscribe(loop, maxsize=2)
            other_id, other = log.subscribe(loop, maxsize=12)
            for n in range(10):
                log.add(str(n))
            await asyncio.sleep(0)  # run the scheduled deliveries
            assert [queue.get_nowait().message, queue.get_nowait().message] == ["8", "9"]
            assert [other.get_nowait().message for _ in range(10)] == [str(n) for n in range(10)]
            assert errors == []
            assert len(log.tail()) == 10  # overflow only affects this subscriber
            log.unsubscribe(sub_id)
            log.unsubscribe(other_id)
        finally:
            loop.set_exception_handler(previous)

    asyncio.run(scenario())
