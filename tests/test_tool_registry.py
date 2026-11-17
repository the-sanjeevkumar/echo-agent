"""Tool registry: registration, execution, dynamic bus-driven registration."""

from __future__ import annotations

import pytest

from python_core.core.event_bus import EventBus
from python_core.core.interfaces import ToolSpec
from python_core.core.tool_registry import ToolRegistry


async def test_execute_sync_and_async_handlers(bus: EventBus) -> None:
    registry = ToolRegistry(bus)

    def sync_handler(args):
        return "sync-ok"

    async def async_handler(args):
        return "async-ok"

    registry.register(ToolSpec(name="t_sync", description=""), sync_handler)
    registry.register(ToolSpec(name="t_async", description=""), async_handler)

    r1 = await registry.execute("t_sync", {})
    r2 = await registry.execute("t_async", {})
    assert r1["ok"] and r1["result"] == "sync-ok"
    assert r2["ok"] and r2["result"] == "async-ok"


async def test_unknown_tool_and_missing_handler(bus: EventBus) -> None:
    registry = ToolRegistry(bus)
    r = await registry.execute("nope", {})
    assert r["ok"] is False and "unknown tool" in r["error"]

    registry.register(ToolSpec(name="remote_only", description=""), handler=None)
    r2 = await registry.execute("remote_only", {})
    assert r2["ok"] is False and "no local handler" in r2["error"]


async def test_handler_exception_is_data(bus: EventBus) -> None:
    registry = ToolRegistry(bus)

    def bad(args):
        raise ValueError("bad input")

    registry.register(ToolSpec(name="bad", description=""), bad)
    r = await registry.execute("bad", {})
    assert r["ok"] is False and "bad input" in r["error"]
    assert r["elapsed_ms"] >= 0


async def test_dynamic_registration_via_bus(bus: EventBus) -> None:
    """MODULE 3 Phase 3: registration events create tools at runtime."""
    registry = ToolRegistry(bus)
    spec = ToolSpec(name="hot_tool", description="registered via bus", source="test")
    await bus.emit("tool:register", {"spec": spec, "handler": lambda a: {"echo": a}})
    r = await registry.execute("hot_tool", {"x": 1})
    assert r["ok"] and r["result"] == {"echo": {"x": 1}}

    await bus.emit("tool:unregister", {"name": "hot_tool"})
    r2 = await registry.execute("hot_tool", {})
    assert r2["ok"] is False and "unknown tool" in r2["error"]


async def test_tool_executed_event_published(bus: EventBus) -> None:
    events: list = []
    bus.subscribe("tool:executed", lambda e: events.append(e.payload), label="cap")
    registry = ToolRegistry(bus)
    registry.register(ToolSpec(name="emit", description=""), lambda a: 1)
    await registry.execute("emit", {})
    assert len(events) == 1
    assert events[0]["tool"] == "emit" and events[0]["ok"] is True


def test_schemas_projection() -> None:
    registry = ToolRegistry()
    registry.register(ToolSpec(name="a", description="A", tags=["x"], source="builtin"), handler=None)
    registry.register(ToolSpec(name="b", description="B", source="skill:demo"), handler=None)
    schemas = registry.schemas()
    assert [s["name"] for s in schemas] == ["a", "b"]
    assert schemas[1]["source"] == "skill:demo"
    # handlers are never serialized into schemas
    assert "handler" not in schemas[0]


async def test_invalid_name_rejected() -> None:
    registry = ToolRegistry()
    with pytest.raises(ValueError):
        registry.register(ToolSpec(name="bad name!", description=""), handler=None)
