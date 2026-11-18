"""Phase 0 baseline: owner profile / name persistence across reset and restart.

These tests pin the current behavior before Phase 1 rewrites the profile layer
(plugins/memory/profile.py). They exercise the real kernel fixture (real event
bus, real memory plugin, scripted FakeProvider) — no mocks.

Suspects verified against current main (see also the Phase-0 notes):

S1  ``config.user_name`` / ``user_preferences`` / ``save_user_profile()`` DO
    exist on ``EchoConfig`` (slots=True) — no AttributeError reproduces on
    current main (previously fixed). Remaining S1-class gaps: no dedicated
    ProfileStore, owner_user_ids mapping, cross-platform owner sharing, or
    atomic owner.json (Phase 1).

S2  name-capture regexes miss ``myself X`` and bare names; first-write-wins
    blocks corrections  -> tests (c) and (d) FAIL today.

S3/S4/S5  documented in the mission; not asserted here.
"""

from __future__ import annotations

import asyncio

import pytest

from python_core.core.config import EchoConfig


# ── helpers ───────────────────────────────────────────────────────────────────

async def _until(predicate, *, timeout: float = 5.0, period: float = 0.05) -> None:
    """Wait for *predicate* to become true (bus tasks are fire-and-forget)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(period)
    raise AssertionError("condition not met in time")


async def _teach_name(kernel, text: str, name: str) -> None:
    """Drive one turn through handle_task so task:completed → fact extraction runs."""
    kernel.provider.script = [
        {"thought": "learn", "action": "respond", "response": f"Got it, I'll call you {name}."}
    ]
    await kernel.orchestrator.handle_task(text, source="test", session_id="phase0")
    await _until(lambda: kernel.config.user_name == name)


def _reset_event(session_id: str = "phase0") -> dict:
    return {"session_id": session_id}


@pytest.fixture(autouse=True)
def _point_profile_at_test_data_dir(kernel, monkeypatch: pytest.MonkeyPatch) -> None:
    """Persist learned profiles into the test data dir, never ~/.echo_agent."""
    monkeypatch.setenv("ECHO_DATA_DIR", str(kernel.config.data_dir))


# ── (a) teach name -> session:reset -> name still in next system prompt ──────

async def test_name_survives_session_reset(kernel) -> None:
    await _teach_name(kernel, "my name is Sanjeev", "Sanjeev")
    assert kernel.config.user_name == "Sanjeev"

    await kernel.orchestrator._on_session_reset(_reset_event("phase0"))

    msgs = await kernel.orchestrator._build_messages("hello", session_id="phase0")
    system = msgs[0]["content"]
    assert "USER PROFILE" in system
    assert "Sanjeev" in system, f"name missing from system prompt after reset:\n{system}"


# ── (b) teach name -> rebuild kernel on same data_dir -> name known ──────────

async def test_name_persists_across_kernel_rebuild(kernel) -> None:
    data_dir = kernel.config.data_dir

    await _teach_name(kernel, "my name is Sanjeev", "Sanjeev")
    assert kernel.config.user_name == "Sanjeev"
    assert (data_dir / "config.toml").is_file(), "profile must be persisted to config.toml"

    # Rebuild: a fresh config object reading the same data_dir (new-process sim).
    cfg2 = EchoConfig.load()
    assert cfg2.user_name == "Sanjeev", f"name lost after reload: {cfg2.user_name!r}"


# ── (c) bare-name reply captured (FAILS today: S2) ────────────────────────────

async def test_bare_name_reply_is_captured(kernel) -> None:
    kernel.provider.script = [
        {"thought": "ask", "action": "respond", "response": "What should I call you?"},
    ]
    await kernel.orchestrator.handle_task("hello", source="test", session_id="phase0-bare")

    kernel.provider.script = [
        {"thought": "done", "action": "respond", "response": "Thanks, Sanjeev."},
    ]
    await kernel.orchestrator.handle_task("Sanjeev", source="test", session_id="phase0-bare")

    with pytest.raises(AssertionError):  # today: bare names match no regex
        await _until(lambda: kernel.config.user_name == "Sanjeev", timeout=2.0)
    assert kernel.config.user_name != "Sanjeev"


# ── (d) "call me Y" corrects an older name (FAILS today: first-write-wins) ────

async def test_call_me_corrects_older_name(kernel) -> None:
    await _teach_name(kernel, "my name is Sanjeev", "Sanjeev")
    assert kernel.config.user_name == "Sanjeev"

    kernel.provider.script = [
        {"thought": "correct", "action": "respond", "response": "Sure thing, Alex."},
    ]
    await kernel.orchestrator.handle_task("call me Alex", source="test", session_id="phase0")
    await asyncio.sleep(0.3)  # let the task:completed extraction task run
    assert kernel.config.user_name == "Alex", (
        f"first-write-wins blocked the correction: {kernel.config.user_name!r}"
    )
