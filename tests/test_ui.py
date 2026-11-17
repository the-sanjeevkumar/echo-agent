"""Terminal UI helpers: compact step summaries + dependency-free spinner."""

from __future__ import annotations

import asyncio
import io

from python_core.ui import Spinner, paint, step_summary


def test_step_summary_is_compact() -> None:
    steps = [
        {"tool": "memory_recall", "ok": True},
        {"tool": "no-tools", "ok": False},
        {"tool": "no-tools", "ok": False},
        {"tool": "web_fetch", "ok": True},
    ]
    summary = step_summary(steps, color=False)
    assert summary.startswith("  ")
    assert "✓ memory_recall" in summary
    assert "✗ no-tools ×2" in summary
    assert "✓ web_fetch" in summary
    # No arrows, no repeated FAIL lines.
    assert "→" not in summary and "FAIL" not in summary


def test_step_summary_empty_when_no_tools() -> None:
    assert step_summary([], color=False) == ""


def test_step_summary_handles_missing_tool_name() -> None:
    assert "invalid-tool" in step_summary([{"tool": None, "ok": False}], color=False)
    assert "invalid-tool" in step_summary([{"tool": "  ", "ok": False}], color=False)


def test_step_summary_caps_long_runs() -> None:
    steps = [{"tool": f"tool_{i}", "ok": True} for i in range(15)]
    summary = step_summary(steps, color=False, max_groups=5)
    assert "+10 more" in summary


def test_paint_respects_toggle() -> None:
    assert paint("x", "36", enabled=False) == "x"
    assert "\x1b[36m" in paint("x", "36", enabled=True)


async def test_spinner_disabled_is_silent() -> None:
    buf = io.StringIO()
    spinner = Spinner("thinking", stream=buf, enabled=False)
    await spinner.start()
    spinner.on_event({"type": "tool", "tool": "fs_read"})
    await spinner.stop()
    assert buf.getvalue() == ""


async def test_spinner_animates_and_clears() -> None:
    buf = io.StringIO()
    spinner = Spinner("thinking", stream=buf, enabled=True, interval=0.01)
    await spinner.start()
    spinner.on_event({"type": "tool", "tool": "fs_read"})
    await asyncio.sleep(0.05)
    await spinner.stop()
    out = buf.getvalue()
    assert "fs_read" in out
    assert out.endswith("\r\x1b[2K")  # status line cleared on completion
