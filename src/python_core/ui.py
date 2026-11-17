"""Tiny terminal UI: animated spinner and compact, human-readable output.

Deliberately dependency-free. Animation is enabled only when stdout is a real
TTY (and ``NO_COLOR``/``TERM=dumb`` are absent), so piped output, CI, and tests
never see escape codes or stray background tasks.

Two pieces the CLI uses:

    spinner = Spinner("thinking")
    await spinner.start()
    try:
        result = await orchestrator.handle_task(text, on_event=spinner.on_event)
    finally:
        await spinner.stop()

    render_response(result["response"])
    render_steps(result["steps"])       # ✓ memory_recall · ✗ web_fetch
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from collections.abc import Iterable
from typing import Any, TextIO

BRAILLE_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# ANSI SGR codes
_DIM = "2"
_CYAN = "36"
_GREEN = "32"
_RED = "31"
_YELLOW = "33"
_BOLD = "1"


def color_enabled(stream: TextIO | None = None) -> bool:
    """True when ANSI styling/animation makes sense for *stream*."""
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def paint(text: str, *codes: str, enabled: bool | None = None) -> str:
    """Wrap *text* in ANSI SGR codes (no-op when styling is disabled)."""
    if enabled is None:
        enabled = color_enabled()
    if not enabled or not codes:
        return text
    return f"\x1b[{';'.join(codes)}m{text}\x1b[0m"


def _clean_tool_name(step: dict[str, Any]) -> str:
    name = str(step.get("tool") or "").strip()
    return name or "invalid-tool"


def step_summary(
    steps: Iterable[dict[str, Any]],
    *,
    color: bool | None = None,
    indent: str = "  ",
    max_groups: int = 10,
) -> str:
    """One compact line summarising tool steps.

    Consecutive identical outcomes collapse (``✗ no-tools ×3``) and unknown
    tool names surface as ``invalid-tool`` rather than an empty arrow. Returns
    an empty string when there were no tool calls, so simple answers stay clean.
    """
    if color is None:
        color = color_enabled()
    steps = list(steps)
    if not steps:
        return ""

    groups: list[tuple[str, bool, int]] = []
    for step in steps:
        name = _clean_tool_name(step)
        ok = bool(step.get("ok"))
        if groups and groups[-1][0] == name and groups[-1][1] == ok:
            prev_name, prev_ok, count = groups[-1]
            groups[-1] = (prev_name, prev_ok, count + 1)
        else:
            groups.append((name, ok, 1))

    overflow = max(0, len(groups) - max_groups)
    visible = groups[:max_groups]

    parts = []
    for name, ok, count in visible:
        mark = paint("✓", _GREEN, enabled=color) if ok else paint("✗", _RED, enabled=color)
        suffix = paint(f" ×{count}", _DIM, enabled=color) if count > 1 else ""
        parts.append(f"{mark} {name}{suffix}")
    if overflow:
        parts.append(paint(f"+{overflow} more", _DIM, enabled=color))

    return indent + paint(" · ", _DIM, enabled=color).join(parts)


def render_steps(
    steps: Iterable[dict[str, Any]],
    *,
    stream: TextIO | None = None,
    color: bool | None = None,
    indent: str = "  ",
) -> bool:
    """Print a compact step summary. Returns True if anything was printed."""
    stream = stream or sys.stdout
    summary = step_summary(steps, color=color, indent=indent)
    if not summary:
        return False
    print(summary, file=stream, flush=True)
    return True


def render_response(text: str, *, stream: TextIO | None = None) -> None:
    """Print the final answer with a little breathing room."""
    stream = stream or sys.stdout
    body = (text or "").strip()
    if body:
        print(body, file=stream, flush=True)


def banner(version: str, *, stream: TextIO | None = None) -> str:
    """Small startup banner (no box, no noise)."""
    stream = stream or sys.stdout
    color = color_enabled(stream)
    title = paint(f"echo-agent v{version}", _BOLD, _CYAN, enabled=color)
    hint = paint("local-first · type a task, 'exit' to quit", _DIM, enabled=color)
    return f"{title}\n{hint}"


class Spinner:
    """A one-line braille spinner that runs while a task is in flight.

    ``enabled`` defaults to TTY detection. ``on_event`` accepts the small
    progress dicts emitted by the orchestrator (``thinking`` / ``tool`` / …)
    so the status line reflects what the agent is actually doing.
    """

    def __init__(
        self,
        label: str = "thinking",
        *,
        stream: TextIO | None = None,
        enabled: bool | None = None,
        interval: float = 0.08,
    ) -> None:
        self._stream = stream or sys.stdout
        self._color = color_enabled(self._stream)
        self.enabled = self._color if enabled is None else enabled
        self._interval = interval
        self._label = label
        self._task: asyncio.Task[None] | None = None
        self._started = 0.0
        self._frame = 0

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        self._started = time.monotonic()
        self._task = asyncio.create_task(self._run(), name="echo-spinner")

    async def stop(self, final: str | None = None) -> None:
        """Stop animating and clear the status line (optionally leaving a note)."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self.enabled:
            self._stream.write("\r\x1b[2K")
            if final:
                self._stream.write(final + "\n")
            self._stream.flush()

    # ── status updates ─────────────────────────────────────────────────────
    def set_status(self, label: str) -> None:
        self._label = label

    def on_event(self, event: dict[str, Any]) -> None:
        """Callable handed to ``Orchestrator.handle_task(on_event=...)``."""
        kind = event.get("type")
        if kind == "thinking":
            self.set_status("thinking")
        elif kind == "tool":
            self.set_status(f"tool: {event.get('tool') or '?'}")
        elif kind == "invalid_tool":
            self.set_status(f"unknown tool: {event.get('tool') or '?'}")
        elif kind == "tool_result":
            self.set_status("thinking")
        elif kind == "cached":
            self.set_status("cached")

    # ── internals ──────────────────────────────────────────────────────────
    async def _run(self) -> None:
        while True:
            self._draw()
            await asyncio.sleep(self._interval)

    def _draw(self) -> None:
        frame = BRAILLE_FRAMES[self._frame % len(BRAILLE_FRAMES)]
        self._frame += 1
        elapsed = time.monotonic() - self._started
        label = self._label
        try:
            columns = shutil.get_terminal_size((80, 20)).columns
        except OSError:
            columns = 80
        text = f"{frame} {label} ({elapsed:4.1f}s)"
        if len(text) > max(20, columns - 1):
            text = text[: max(20, columns - 2)] + "…"
        self._stream.write("\r\x1b[2K" + paint(text, _CYAN, enabled=self._color))
        self._stream.flush()
