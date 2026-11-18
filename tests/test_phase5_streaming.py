"""PHASE 5: live activity streaming — ordered bus progress events, one
editable gateway status message per task, debounce, Hermes notices, CLI spinner.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from python_core.core.config import EchoConfig
from python_core.core.event_bus import EventBus
from python_core.plugins.gateways.base import GatewayInterface
from python_core.ui import Spinner


class FakeGateway(GatewayInterface):
    """Gateway that records status-message create/edit/finalize calls."""

    name = "fake"

    def __init__(self, config: EchoConfig) -> None:
        super().__init__()
        self.config = config
        self.created: list[tuple[str, str, str]] = []
        self.edited: list[tuple[str, str, str]] = []
        self.finalized: list[tuple[str, str, str]] = []
        self.replies: list[tuple[str, str]] = []
        self._seq = 0

    async def start(self, ctx: Any) -> None:
        self.bind_bus(ctx.bus)

    async def stop(self) -> None:
        self.unbind_bus()

    async def send_reply(self, channel_id: str, text: str) -> None:
        self.replies.append((channel_id, text))

    async def send_status_create(self, channel_id: str, text: str) -> str:
        self._seq += 1
        message_id = f"m{self._seq}"
        self.created.append((channel_id, text, message_id))
        return message_id

    async def send_status_edit(self, channel_id: str, message_id: str, text: str) -> None:
        self.edited.append((channel_id, message_id, text))

    async def send_status_finalize(self, channel_id: str, message_id: str, text: str) -> None:
        self.finalized.append((channel_id, message_id, text))


def _gateway(tmp_path: Path, *, debounce: float = 0.0, mode: str = "compact") -> tuple[EventBus, FakeGateway]:
    bus = EventBus()
    config = EchoConfig(data_dir=tmp_path / "data", gateway_progress=mode, gateway_status_debounce=debounce)
    gateway = FakeGateway(config)
    gateway.bind_bus(bus)
    return bus, gateway


# ── orchestrator event stream ─────────────────────────────────────────────────

async def test_fake_provider_task_emits_ordered_progress_events(kernel) -> None:
    kernel.record("task:progress")
    kernel.provider.script = [
        {"thought": "read", "action": "tool", "tool": "fs_read", "args": {"path": "pyproject.toml", "max_chars": 20}},
        {"thought": "reply", "action": "respond", "response": "ok"},
    ]
    await kernel.orchestrator.handle_task(
        "read a file", source="cli", reply_channel="telegram:42", platform="telegram",
        session_id="phase5",
    )
    await asyncio.sleep(0.2)  # fire-and-forget bus publishes

    events = kernel.captured["task:progress"]
    kinds = [e.get("kind") for e in events]
    assert kinds[0] == "thinking"
    assert "tool" in kinds and "tool_result" in kinds
    assert kinds.index("tool") < kinds.index("tool_result")
    for event in events:
        assert event["reply_channel"] == "telegram:42"
        assert event["platform"] == "telegram"
        assert event["session_id"] == "phase5"
        assert "detail" in event
    tool_event = next(e for e in events if e["kind"] == "tool")
    assert "fs_read" in tool_event["detail"]


async def test_file_edit_progress_has_diff_gist(kernel) -> None:
    kernel.record("task:progress")
    kernel.provider.script = [
        {"thought": "edit", "action": "tool", "tool": "fs_apply_diff",
         "args": {"path": "pyproject.toml", "diff": "--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new"}},
        {"thought": "reply", "action": "respond", "response": "edited"},
    ]
    await kernel.orchestrator.handle_task("edit a file", source="cli", session_id="phase5-edit")
    await asyncio.sleep(0.2)
    edits = [e for e in kernel.captured["task:progress"] if e["kind"] == "file_edit"]
    assert edits and "old" in edits[0]["detail"] and len(edits[0]["detail"]) <= 400


# ── gateway status message lifecycle ──────────────────────────────────────────

async def test_gateway_status_create_edit_finalize(tmp_path: Path) -> None:
    bus, gw = _gateway(tmp_path, debounce=0.0)
    progress = {
        "reply_channel": "fake:ch1", "platform": "fake", "session_id": "s1",
        "kind": "tool", "detail": "fs_read(path=pyproject.toml)",
    }
    await bus.emit("task:progress", progress, source="orchestrator")
    assert len(gw.created) == 1 and gw.created[0][0] == "ch1"
    message_id = gw.created[0][2]
    assert "fs_read" in gw.created[0][1]

    await bus.emit("task:progress", {**progress, "kind": "skill", "detail": "deploy-checklist"}, source="orchestrator")
    assert gw.edited and gw.edited[-1][1] == message_id
    assert "🛠" in gw.edited[-1][2]

    await bus.emit("task:completed", {
        "response": "final answer", "reply_channel": "fake:ch1", "platform": "fake",
        "session_id": "s1", "steps": [{"tool": "fs_read", "ok": True}],
    }, source="orchestrator")
    assert gw.finalized and gw.finalized[-1][1] == message_id
    assert "✅" in gw.finalized[-1][2]
    # Final answer is a SEPARATE message.
    assert gw.replies == [("ch1", "final answer")]


async def test_debounce_suppresses_rapid_edits(tmp_path: Path) -> None:
    bus, gw = _gateway(tmp_path, debounce=2.0)
    payload = {
        "reply_channel": "fake:ch9", "platform": "fake", "session_id": "s9",
        "kind": "tool", "detail": "tool A",
    }
    await bus.emit("task:progress", payload, source="orchestrator")
    await bus.emit("task:progress", {**payload, "detail": "tool B"}, source="orchestrator")
    assert len(gw.created) == 1 and gw.edited == []  # second edit suppressed

    # Simulate the debounce window elapsing.
    gw._status["ch9"]["last_edit"] -= 5
    await bus.emit("task:progress", {**payload, "detail": "tool C"}, source="orchestrator")
    assert len(gw.edited) == 1
    assert "tool B" in gw.edited[-1][2] and "tool C" in gw.edited[-1][2]


async def test_progress_off_mode_creates_nothing(tmp_path: Path) -> None:
    bus, gw = _gateway(tmp_path, mode="off")
    await bus.emit("task:progress", {
        "reply_channel": "fake:ch1", "platform": "fake", "session_id": "s1",
        "kind": "tool", "detail": "x",
    }, source="orchestrator")
    assert gw.created == []


def test_status_icons_and_spinner_labels() -> None:
    line = GatewayInterface._status_line({"kind": "memory", "detail": "profile saved"})
    assert "💾" in line
    spinner = Spinner(enabled=False)
    spinner.on_event({"type": "tool", "tool": "fs_read", "args": {"path": "x"}})
    assert "fs_read" in spinner._label
    spinner.on_event({"type": "file_edit", "diff": "-old +new"})
    assert "edit:" in spinner._label
    spinner.on_event({"type": "skill"})
    assert "🛠" in spinner._label
