"""Event bus: routing, priority, wildcards, request/reply, error isolation."""

from __future__ import annotations

import asyncio

import pytest

from python_core.core.event_bus import EventBus


async def test_exact_topic_dispatch(bus: EventBus) -> None:
    seen: list = []
    bus.subscribe("task:received", lambda e: seen.append(e.payload), label="t")
    await bus.emit("task:received", {"x": 1})
    assert seen == [{"x": 1}]


async def test_wildcard_and_glob(bus: EventBus) -> None:
    hits: list[str] = []
    bus.subscribe("agent:*", lambda e: hits.append("glob:" + e.topic), label="glob")
    bus.subscribe("*", lambda e: hits.append("all:" + e.topic), label="all")
    await bus.emit("agent:coder", 1)
    await bus.emit("task:received", 2)
    assert "glob:agent:coder" in hits and "all:agent:coder" in hits
    assert "all:task:received" in hits
    assert "glob:task:received" not in hits


async def test_priority_ordering(bus: EventBus) -> None:
    order: list[str] = []
    bus.subscribe("t", lambda e: order.append("low"), priority=-10, label="low")
    bus.subscribe("t", lambda e: order.append("high"), priority=10, label="high")
    await bus.emit("t", None)
    assert order == ["high", "low"]


async def test_once_subscription(bus: EventBus) -> None:
    seen: list = []
    bus.subscribe("tick", lambda e: seen.append(1), once=True, label="once")
    await bus.emit("tick", None)
    await bus.emit("tick", None)
    assert seen == [1]
    assert bus.subscription_count == 0


async def test_handler_error_does_not_break_others(bus: EventBus) -> None:
    results = await bus.emit("boom", None)
    # both handlers ran; one errored, one returned a value

    async def bad(e) -> None:
        raise RuntimeError("kaput")

    def good(e) -> str:
        return "fine"

    bus2 = EventBus()
    bus2.subscribe("boom", bad, priority=10, label="bad")
    bus2.subscribe("boom", good, priority=0, label="good")
    results = await bus2.emit("boom", None)
    assert len(results) == 2
    assert any(r.error and "kaput" in r.error for r in results)
    assert any(r.result == "fine" for r in results)


async def test_request_returns_first_reply(bus: EventBus) -> None:
    async def replier(e) -> dict:
        return {"answer": 42}

    def noise(e) -> None:
        return None

    bus.subscribe("ask", noise, priority=10, label="noise")
    bus.subscribe("ask", replier, label="replier")
    out = await bus.request("ask", None, timeout=2)
    assert out == {"answer": 42}


async def test_request_timeout_returns_none(bus: EventBus) -> None:
    out = await bus.request("nobody-listens", None, timeout=0.1)
    assert out is None


async def test_wait_for_pattern(bus: EventBus) -> None:
    async def later() -> None:
        await asyncio.sleep(0.01)
        await bus.emit("evt:later", "payload-x")

    task = asyncio.create_task(later())
    event = await bus.wait_for("evt:*", timeout=1.0)
    await task
    assert event.payload == "payload-x"


async def test_publish_fire_and_forget(bus: EventBus) -> None:
    seen: list = []
    bus.subscribe("async-topic", lambda e: seen.append(e.payload), label="t")
    task = bus.publish("async-topic", 7)
    await task
    assert seen == [7]


async def test_unsubscribe(bus: EventBus) -> None:
    seen: list = []
    sub = bus.subscribe("u", lambda e: seen.append(1), label="t")
    assert bus.unsubscribe(sub) is True
    await bus.emit("u", None)
    assert seen == []
    assert bus.unsubscribe(sub) is False


async def test_signature_styles_accepted(bus: EventBus) -> None:
    got: dict[str, list] = {}

    def handler_event(event) -> None:
        got["event"] = [event.topic, event.payload]

    def handler_payload(payload) -> None:
        got["payload"] = payload

    def handler_two(event, payload) -> None:
        got["two"] = [event.topic, payload]

    bus.subscribe("s1", handler_event, label="h1")
    bus.subscribe("s2", handler_payload, label="h2")
    bus.subscribe("s3", handler_two, label="h3")
    await bus.emit("s1", "p1")
    await bus.emit("s2", "p2")
    await bus.emit("s3", "p3")
    assert got["event"] == ["s1", "p1"]
    assert got["payload"] == "p2"
    assert got["two"] == ["s3", "p3"]
