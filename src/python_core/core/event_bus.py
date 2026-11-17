"""Central Pub/Sub event dispatcher — the expandability core.

Every feature in echo-agent communicates through this bus, never by direct
imports of other plugins. Consequences:

  * A gateway can trigger the self-healing pipeline without knowing it exists.
  * The skills plugin can register tools at runtime by publishing one event.
  * Any future module only needs to (a) subscribe to topics it cares about
    and (b) publish events describing what it observed.

API:
    bus.subscribe(pattern, handler, priority=0, once=False) -> Subscription
    bus.unsubscribe(sub)
    await bus.emit(topic, payload)            -> list[Result]   (awaited dispatch)
    bus.publish(topic, payload)               -> Task            (fire & forget)
    await bus.request(topic, payload, t=...)  -> first reply      (request/reply)
    await bus.wait_for(pattern, timeout=...)  -> Event            (await a condition)

Topic matching mirrors the Rust bus: exact match, trailing-glob ("agent:*"),
and the universal wildcard "*".
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger("echo.event_bus")

Handler = Callable[..., Any] | Callable[..., Awaitable[Any]]

_SERIAL = itertools.count(1)


@dataclass(slots=True)
class Event:
    """A single message on the bus."""

    id: int
    topic: str
    payload: Any
    source: str = "unknown"
    ts: float = field(default_factory=time.time)


@dataclass(slots=True)
class Subscription:
    """Handle returned by :meth:`EventBus.subscribe`; pass back to unsubscribe."""

    token: int
    pattern: str
    handler: Handler
    priority: int = 0
    once: bool = False
    label: str = ""


@dataclass(slots=True)
class DispatchResult:
    """Outcome of one handler during emit(); used for introspection and tests."""

    subscriber: str
    result: Any = None
    error: str | None = None


class EventBus:
    """In-process, asyncio-native publish/subscribe bus with priority order,
    wildcard topics, once-subscriptions, request/reply, and wait-for."""

    def __init__(self) -> None:
        self._subs: dict[int, Subscription] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._event_seq = _SERIAL

    # ── registration ───────────────────────────────────────────────────────
    def subscribe(
        self,
        pattern: str,
        handler: Handler,
        *,
        priority: int = 0,
        once: bool = False,
        label: str = "",
    ) -> Subscription:
        """Register *handler* for topics matching *pattern*.

        Handlers receive either ``handler(event)`` or ``handler(event, payload)``
        — introspected automatically so plugins can use either signature.
        Higher priority numbers dispatch first.
        """
        if not callable(handler):
            raise TypeError("handler must be callable")
        token = next(_SERIAL)
        sub = Subscription(
            token=token,
            pattern=pattern,
            handler=handler,
            priority=priority,
            once=once,
            label=label or getattr(handler, "__qualname__", repr(handler)),
        )
        self._subs[token] = sub
        logger.debug("subscribed %s → %s (prio=%d)", sub.label, pattern, priority)
        return sub

    def unsubscribe(self, sub: Subscription) -> bool:
        return self._subs.pop(sub.token, None) is not None

    # ── matching ───────────────────────────────────────────────────────────
    @staticmethod
    def matches(pattern: str, topic: str) -> bool:
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return topic.startswith(pattern[:-1])
        return pattern == topic

    def _matching(self, topic: str) -> list[Subscription]:
        subs = [s for s in self._subs.values() if self.matches(s.pattern, topic)]
        subs.sort(key=lambda s: s.priority, reverse=True)  # high priority first
        return subs

    # ── dispatch ───────────────────────────────────────────────────────────
    async def emit(self, topic: str, payload: Any = None, *, source: str = "unknown") -> list[DispatchResult]:
        """Awaited dispatch: every handler runs (in priority order), one
        handler's failure never aborts the others. Returns per-handler results.
        """
        event = Event(id=next(self._event_seq), topic=topic, payload=payload, source=source)
        results: list[DispatchResult] = []
        for sub in self._matching(topic):
            if sub.once:
                self._subs.pop(sub.token, None)
            results.append(await self._invoke(sub, event))
        logger.debug("emit %s (%d handlers, source=%s)", topic, len(results), source)
        return results

    async def _invoke(self, sub: Subscription, event: Event) -> DispatchResult:
        # Accept handler(event), handler(event, payload), or handler(payload).
        kwargs: dict[str, Any] = {}
        try:
            sig = inspect.signature(sub.handler)
            params = list(sig.parameters.values())
            takes_event = any(p.name == "event" for p in params) or (
                len(params) == 1 and params[0].name in {"event", "e", "ev", "evnt"}
            )
            takes_payload = any(p.name == "payload" for p in params)
        except (TypeError, ValueError):  # C functions / callables w/o signature
            params = []
            takes_event = takes_payload = False

        args: tuple[Any, ...]
        if takes_payload and takes_event:
            args, kwargs = (), {"event": event, "payload": event.payload}
        elif takes_payload:
            args = (event.payload,)
        elif takes_event:
            args = (event,)
        elif len(params) == 2:
            args = (event, event.payload)
        else:
            args = (event,)

        try:
            out = sub.handler(*args, **kwargs)
            if inspect.isawaitable(out):
                out = await out
            return DispatchResult(subscriber=sub.label, result=out)
        except Exception as exc:  # noqa: BLE001 — one bad handler must not kill the bus
            logger.exception("handler %s failed on topic %s", sub.label, event.topic)
            return DispatchResult(subscriber=sub.label, error=f"{type(exc).__name__}: {exc}")

    def publish(self, topic: str, payload: Any = None, *, source: str = "unknown") -> asyncio.Task[list[DispatchResult]]:
        """Fire-and-forget dispatch. Task is strongly referenced until done."""
        task = asyncio.create_task(self.emit(topic, payload, source=source))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def request(self, topic: str, payload: Any = None, *, timeout: float = 10.0, source: str = "unknown") -> Any:
        """Request/reply over the bus: return the first non-None handler result."""
        results = await asyncio.wait_for(self.emit(topic, payload, source=source), timeout=timeout)
        for r in results:
            if r.error is None and r.result is not None:
                return r.result
        return None

    async def wait_for(self, pattern: str, timeout: float | None = None) -> Event:
        """Await the next event matching *pattern* (useful in tests and for
        synchronous-style daemon logic)."""
        future: asyncio.Future[Event] = asyncio.get_running_loop().create_future()

        async def _hook(event: Event) -> None:
            if not future.done():
                future.set_result(event)

        sub = self.subscribe(pattern, _hook, label="wait_for")
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self.unsubscribe(sub)

    # ── introspection ──────────────────────────────────────────────────────
    def listeners(self, topic: str) -> list[str]:
        return [s.label for s in self._matching(topic)]

    @property
    def subscription_count(self) -> int:
        return len(self._subs)


class EventBusStats:
    """Optional listener that records traffic for the research metrics guide."""

    def __init__(self, bus: EventBus) -> None:
        self.counts: dict[str, int] = {}
        self.total = 0
        bus.subscribe("*", self._on_any, priority=-100, label="bus_stats")

    def _on_any(self, event: Event) -> None:
        self.counts[event.topic] = self.counts.get(event.topic, 0) + 1
        self.total += 1
