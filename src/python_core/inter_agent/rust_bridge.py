"""Rust echo-bus bridge — Python adapter for the high-performance IPC bus.

The Rust ``echo-bus`` binary (``cargo build --release``) serves newline-
delimited JSON-RPC over a Unix socket. This bridge is a thin asyncio client:

    Rust → Python: every event on subscribed topics is re-published on the
                   Python EventBus (topics prefixed ``bus:`` for namespace clarity).
    Python → Rust: ``publish_remote(topic, payload)`` sends frames out; and
                   ``forward(topics)`` mirrors local Python topics into Rust.

Used when sub-agents (Coder/Researcher/Debugger) run as separate processes
and need shared, high-throughput message passing. Optional: the agent is
fully functional without it (the in-process Python bus is the default).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from python_core.core.event_bus import EventBus, Subscription

logger = logging.getLogger("echo.rust_bridge")

FRAME_ID = 0


class RustBusBridge:
    """Persistent connection to echo-bus with auto-reconnect."""

    def __init__(self, bus: EventBus, socket_path: str, *, patterns: tuple[str, ...] = ("agent:*",)) -> None:
        self.bus = bus
        self.socket_path = socket_path
        self.patterns = patterns
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._task: asyncio.Task[None] | None = None
        self._forward_subs: list[Subscription] = []
        self.connected = False

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="echo-rust-bridge")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._close()
        for sub in self._forward_subs:
            self.bus.unsubscribe(sub)
        self._forward_subs.clear()

    async def _close(self) -> None:
        self.connected = False
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass
        self._reader = self._writer = None

    # ── connection loop ────────────────────────────────────────────────────
    async def _run(self) -> None:
        backoff = 1.0
        while True:
            try:
                self._reader, self._writer = await asyncio.open_unix_connection(self.socket_path)
                self.connected = True
                backoff = 1.0
                logger.info("connected to echo-bus at %s", self.socket_path)
                # Subscribe to every configured pattern (per spec, one call each).
                for pattern in self.patterns:
                    await self._rpc({"jsonrpc": "2.0", "id": _next_id(), "method": "subscribe", "params": {"topic": pattern}})
                await self._read_loop()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("echo-bus connection unavailable (%s); retrying in %.0fs", exc, backoff)
                await self._close()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _read_loop(self) -> None:
        assert self._reader is not None
        while True:
            line = await self._reader.readline()
            if not line:
                self.connected = False
                return  # connection closed → reconnect loop
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            if frame.get("method") == "event":
                params = frame.get("params") or {}
                # Namespace Rust topics so they can't collide with Python ones.
                self.bus.publish(f"bus:{params.get('topic', '')}", params.get("payload"), source="rust_bus")

    async def _rpc(self, frame: dict[str, Any]) -> Any:
        """Send a request frame (responses to calls we don't await are ignored)."""
        if self._writer is None:
            raise ConnectionError("not connected to echo-bus")
        self._writer.write((json.dumps(frame) + "\n").encode())
        await self._writer.drain()

    # ── public API ─────────────────────────────────────────────────────────
    async def publish_remote(self, topic: str, payload: Any) -> None:
        """Publish to the Rust bus (broadcast to all its subscribers)."""
        await self._rpc({
            "jsonrpc": "2.0",
            "id": _next_id(),
            "method": "publish",
            "params": {"topic": topic, "payload": payload},
        })

    def forward(self, topics: tuple[str, ...]) -> None:
        """Mirror local Python-bus topics into the Rust bus (fire-and-forget)."""
        for topic in topics:
            pattern = topic

            async def _mirror(event: Any, _topic: str = pattern) -> None:
                payload = getattr(event, "payload", event)
                try:
                    await self.publish_remote(_topic, payload)
                except Exception:  # noqa: BLE001 — mirroring is best-effort
                    pass

            self._forward_subs.append(self.bus.subscribe(pattern, _mirror, label=f"rust_bridge.forward:{topic}"))

    def stats(self) -> dict[str, Any]:
        return {"connected": self.connected, "socket": self.socket_path, "patterns": list(self.patterns)}


_IDS = iter(range(1, 1_000_000))


def _next_id() -> int:
    try:
        return next(_IDS)
    except StopIteration:  # pragma: no cover — 1M frames is plenty per process
        return 0
