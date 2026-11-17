"""Dynamic tool registry — the agent's hands.

Tools are the only way the orchestrator acts on the world (call an API, edit
a file, run a test, talk to another agent). The registry is *fully dynamic*:

  * Built-in tools are registered at boot by the orchestrator.
  * The skills plugin hot-registers tools at runtime by publishing a
    ``tool:register`` event carrying the spec and a handler callable —
    no restart required (MODULE 3, Phase 3).
  * The manifest server exposes the live registry over HTTP (MODULE 5).

In-process payloads may carry callables safely; the registry never persists
handlers — persistence is the skills plugin's job (it re-registers from disk
on boot).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Callable

from .event_bus import EventBus
from .interfaces import ToolHandler, ToolSpec

logger = logging.getLogger("echo.tools")


class ToolRegistry:
    """Name → (spec, handler) map with event-bus-driven dynamic registration."""

    def __init__(self, bus: EventBus | None = None) -> None:
        self._tools: dict[str, tuple[ToolSpec, ToolHandler | None]] = {}
        self._bus = bus
        if bus is not None:
            self.bind_bus(bus)

    # ── direct registration ────────────────────────────────────────────────
    def register(self, spec: ToolSpec, handler: ToolHandler | None = None) -> ToolSpec:
        """Register (or atomically replace) a tool. Returns the stored spec."""
        if not spec.name or not spec.name.replace("_", "").isalnum():
            raise ValueError(f"invalid tool name: {spec.name!r}")
        if handler is not None and not callable(handler):
            raise TypeError(f"handler for {spec.name} must be callable")
        spec.handler = handler or spec.handler
        self._tools[spec.name] = (spec, spec.handler)
        logger.debug("tool registered: %s (source=%s)", spec.name, spec.source)
        return spec

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    # ── lookups ────────────────────────────────────────────────────────────
    def get(self, name: str) -> ToolSpec | None:
        entry = self._tools.get(name)
        return entry[0] if entry else None

    def handler(self, name: str) -> ToolHandler | None:
        entry = self._tools.get(name)
        return entry[1] if entry else None

    def list(self, *, source_prefix: str | None = None) -> list[ToolSpec]:
        specs = [spec for spec, _ in self._tools.values()]
        if source_prefix:
            specs = [s for s in specs if s.source.startswith(source_prefix)]
        return sorted(specs, key=lambda s: s.name)

    def schemas(self) -> list[dict[str, Any]]:
        """JSON-Schema projections for prompts, manifests, and caching keys."""
        return [spec.to_json_schema() for spec in self.list()]

    # ── execution ──────────────────────────────────────────────────────────
    async def execute(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a tool and return a normalized result envelope:
        ``{"ok": bool, "result" | "error": ..., "tool": name, "elapsed_ms": int}``
        Also emits ``tool:executed`` so the memory plugin can log episodes.
        """
        spec, handler = self._tools.get(name) or (None, None)
        started = time.monotonic()
        if spec is None:
            return {"ok": False, "error": f"unknown tool: {name}", "tool": name}
        if handler is None:
            return {"ok": False, "error": f"tool {name} has no local handler (remote-only)", "tool": name}

        envelope: dict[str, Any]
        try:
            out = handler(args or {})
            if inspect.isawaitable(out):
                out = await out
            envelope = {"ok": True, "result": out, "tool": name}
        except Exception as exc:  # noqa: BLE001 — tool failures are data, not crashes
            logger.info("tool %s failed: %s", name, exc)
            logger.debug("tool %s traceback", name, exc_info=True)
            envelope = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "tool": name}

        envelope["elapsed_ms"] = int((time.monotonic() - started) * 1000)
        if self._bus is not None:
            # Awaited (not fire-and-forget): downstream observers (episodic
            # memory) record the outcome before the caller sees the result —
            # deterministic ordering that tests and audits rely on.
            await self._bus.emit(
                "tool:executed",
                {"tool": name, "ok": envelope["ok"], "elapsed_ms": envelope["elapsed_ms"]},
                source="tool_registry",
            )
        return envelope

    # ── event-bus dynamic registration (MODULE 3 Phase 3) ──────────────────
    def bind_bus(self, bus: EventBus) -> None:
        """Subscribe to registration events so any plugin (or an ingested
        skill wrapper module) can hot-register tools by publishing:
            bus.publish("tool:register",   {"spec": {...}, "handler": fn})
            bus.publish("tool:unregister", {"name": "tool_name"})
        """
        self._bus = bus

        def _on_register(payload: dict[str, Any]) -> dict[str, Any]:
            spec_data = payload.get("spec")
            if isinstance(spec_data, ToolSpec):  # in-process: pass the object directly
                spec = spec_data
            else:  # dict payload (e.g. reconstructed from a persisted manifest)
                spec = ToolSpec(**{k: v for k, v in spec_data.items() if k != "handler"})
            handler: Callable | None = payload.get("handler")
            stored = self.register(spec, handler)
            return {"registered": stored.name}

        def _on_unregister(payload: dict[str, Any]) -> dict[str, Any]:
            return {"unregistered": self.unregister(payload.get("name", ""))}

        bus.subscribe("tool:register", _on_register, label="tool_registry.register")
        bus.subscribe("tool:unregister", _on_unregister, label="tool_registry.unregister")
