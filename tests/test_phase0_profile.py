"""PHASE 1: durable owner profile (name fix) — regression tests.

Exercises the real kernel fixture (real bus, real memory plugin + ProfileStore,
scripted FakeProvider). Covers the Phase-0 baseline (now expected to pass) and
Phase-1 acceptance:

* name survives session:reset AND kernel rebuild on the same data_dir
* bare-name rescue ("Sanjeev" after we asked), "myself X"/"this is X"/"X here",
  lowercase names, last-write-wins corrections ("call me Y")
* owner_user_ids: CLI user_id="owner" and a listed Telegram/Discord id share
  ONE profiles/owner.json; any other sender is isolated (zero leaks)
* /forget is owner-only and confirmation-gated
* profiles/owner.json written atomically (no .tmp leftovers, valid JSON)
* profile block injected into every system prompt, <=500 tokens
* ephemera ("my question is ...") is never saved as a preference
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from python_core.core.config import EchoConfig
from python_core.core.event_bus import EventBus
from python_core.core.interfaces import KernelContext
from python_core.core.plugin_manager import PluginManager
from python_core.core.tool_registry import ToolRegistry
from python_core.main import _register_builtin_tools
from python_core.orchestrator import Orchestrator
from python_core.plugins.memory.plugin import CognitiveMemoryPlugin
from tests.conftest import FakeProvider

# ── helpers ───────────────────────────────────────────────────────────────────

async def _until(predicate, *, timeout: float = 5.0, period: float = 0.05) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(period)
    raise AssertionError("condition not met in time")


async def _teach_name(kernel, text: str, name: str, *, user_id: str | None = None,
                      source: str = "test") -> None:
    kernel.provider.script = [
        {"thought": "learn", "action": "respond", "response": f"Got it, I'll call you {name}."}
    ]
    await kernel.orchestrator.handle_task(text, source=source, user_id=user_id,
                                          session_id="profile-main")
    await _until(lambda: kernel.config.user_name == name)


def _block(kernel, user_id: str | None = None, source: str | None = "test") -> str:
    memory = kernel.ctx.service("memory")
    return memory.profile_block(user_id or "", source=source)


async def _second_kernel_on_same_data_dir(data_dir: Path) -> Orchestrator:
    """Build a brand-new kernel (simulating a restart) on the same data_dir."""
    config = EchoConfig(data_dir=data_dir, enable_semantic_cache=False,
                        consolidation_idle_seconds=9999.0)
    config.ensure_dirs()
    bus = EventBus()
    registry = ToolRegistry(bus)
    ctx = KernelContext(bus=bus, config=config, tools=registry, data_dir=config.data_dir)
    _register_builtin_tools(registry, config)
    pm = PluginManager(bus, config)
    ctx.services["plugin_manager"] = pm
    provider = FakeProvider()
    orchestrator = Orchestrator(bus=bus, config=config, tools=registry, provider=provider)
    pm.register(CognitiveMemoryPlugin(config))
    pm.register(provider)
    pm.register(orchestrator)
    await pm.start_all(ctx)
    return orchestrator


# ── Phase-0 baseline (now green) ──────────────────────────────────────────────

async def test_name_survives_session_reset(kernel) -> None:
    await _teach_name(kernel, "my name is Sanjeev", "Sanjeev")
    assert kernel.config.user_name == "Sanjeev"

    await kernel.orchestrator._on_session_reset({"session_id": "profile-main"})

    assert "Sanjeev" in _block(kernel), f"name missing from profile block after reset:\n{_block(kernel)}"


async def test_name_persists_across_kernel_rebuild(kernel) -> None:
    await _teach_name(kernel, "my name is Sanjeev", "Sanjeev", user_id="owner", source="cli")

    # New kernel, same data_dir (restart simulation): the profile block must
    # still carry the name from profiles/owner.json.
    orch2 = await _second_kernel_on_same_data_dir(kernel.config.data_dir)
    try:
        block = await orch2._build_messages("hello", session_id="profile-main",
                                            user_id="owner", source="cli")
        assert "Sanjeev" in block[0]["content"], f"name lost after rebuild:\n{block[0]['content']}"
    finally:
        orch2.stop()


# ── fast-path capture patterns ────────────────────────────────────────────────

async def test_bare_name_rescue_after_we_asked(kernel) -> None:
    kernel.provider.script = [
        {"thought": "ask", "action": "respond", "response": "What should I call you?"},
    ]
    await kernel.orchestrator.handle_task("hello", source="test", user_id="owner",
                                          session_id="profile-bare")

    kernel.provider.script = [
        {"thought": "done", "action": "respond", "response": "Thanks, Sanjeev."},
    ]
    await kernel.orchestrator.handle_task("Sanjeev", source="test", user_id="owner",
                                          session_id="profile-bare")
    await _until(lambda: kernel.config.user_name == "Sanjeev")
    assert kernel.config.user_name == "Sanjeev"


@pytest.mark.parametrize("phrase", ["myself Alex", "this is Alex", "Alex here"])
async def test_extended_name_patterns(kernel, phrase) -> None:
    kernel.provider.script = [
        {"thought": "learn", "action": "respond", "response": f"Got it, {phrase.split()[1]}."},
    ]
    await kernel.orchestrator.handle_task(phrase, source="test", user_id="owner",
                                          session_id="profile-patterns")
    await _until(lambda: kernel.config.user_name == "Alex")
    assert kernel.config.user_name == "Alex"


async def test_lowercase_name_is_title_cased(kernel) -> None:
    kernel.provider.script = [
        {"thought": "learn", "action": "respond", "response": "Got it, sanjeev."},
    ]
    await kernel.orchestrator.handle_task("my name is sanjeev", source="test", user_id="owner",
                                          session_id="profile-lower")
    await _until(lambda: kernel.config.user_name == "Sanjeev")
    assert kernel.config.user_name == "Sanjeev"


async def test_call_me_corrects_older_name(kernel) -> None:
    await _teach_name(kernel, "my name is Sanjeev", "Sanjeev")
    assert kernel.config.user_name == "Sanjeev"

    kernel.provider.script = [
        {"thought": "correct", "action": "respond", "response": "Sure thing, Alex."},
    ]
    await kernel.orchestrator.handle_task("call me Alex", source="test", user_id="owner",
                                          session_id="profile-main")
    await _until(lambda: kernel.config.user_name == "Alex")
    assert kernel.config.user_name == "Alex"


async def test_ephemera_is_not_saved_as_preference(kernel) -> None:
    kernel.provider.script = [
        {"thought": "t", "action": "respond", "response": "sure thing"},
    ]
    await kernel.orchestrator.handle_task("my question is how to deploy", source="cli",
                                          user_id="owner", session_id="profile-ephem")
    await asyncio.sleep(0.3)
    block = _block(kernel, "owner", "cli")
    assert "question" not in block
    assert "deploy" not in block


# ── owner sharing + isolation ─────────────────────────────────────────────────

async def test_cli_and_telegram_owner_share_one_profile(kernel) -> None:
    kernel.config.owner_user_ids = ["7974643102"]
    memory = kernel.ctx.service("memory")
    memory.profiles.owner_ids.add("7974643102")  # store is already started

    # Teach via the CLI path (user_id="owner").
    await _teach_name(kernel, "my name is Sanjeev", "Sanjeev", user_id="owner", source="cli")
    assert kernel.config.user_name == "Sanjeev"

    # A listed Telegram user id resolves to the SAME owner profile.
    block_tg = _block(kernel, "7974643102", source="telegram")
    assert "Sanjeev" in block_tg

    # And the durable file is the single profiles/owner.json.
    profiles = list((kernel.config.data_dir / "profiles").glob("*.json"))
    assert any(p.name == "owner.json" for p in profiles)


async def test_non_owner_sender_is_isolated(kernel) -> None:
    # Stranger teaches their name.
    await _teach_name(kernel, "my name is Stranger", "Stranger", user_id="555000111",
                      source="telegram")
    assert _block(kernel, "555000111", source="telegram").count("Stranger") >= 1

    # Owner profile must not know the stranger, and vice versa.
    assert "Stranger" not in _block(kernel, "owner", "cli")
    assert "Sanjeev" not in _block(kernel, "555000111", source="telegram")


# ── /forget (owner-only, confirmation-gated) ──────────────────────────────────

async def test_forget_requires_confirmation(kernel) -> None:
    await _teach_name(kernel, "my name is Sanjeev", "Sanjeev", user_id="owner", source="cli")

    result = await kernel.orchestrator.handle_task("/forget", source="cli", user_id="owner",
                                                   session_id="profile-main")
    assert "confirm" in result.get("response", "").lower()
    assert "Sanjeev" in _block(kernel, "owner", "cli")  # not wiped yet

    result = await kernel.orchestrator.handle_task("/forget confirm", source="cli",
                                                   user_id="owner", session_id="profile-main")
    await asyncio.sleep(0.2)
    assert _block(kernel, "owner", "cli") == "", f"profile not wiped:\n{_block(kernel)}"
    assert "wiped" in result.get("response", "")


async def test_forget_is_owner_only(kernel) -> None:
    result = await kernel.orchestrator.handle_task("/forget confirm", source="telegram",
                                                   user_id="999000111", session_id="profile-main")
    assert "owner-only" in result.get("response", "")


# ── durability & hygiene ──────────────────────────────────────────────────────

async def test_owner_json_atomic_and_valid(kernel) -> None:
    await _teach_name(kernel, "my name is Sanjeev", "Sanjeev", user_id="owner", source="cli")
    await asyncio.sleep(0.2)

    profiles_dir = kernel.config.data_dir / "profiles"
    tmp_files = list(profiles_dir.glob("*.tmp"))
    assert not tmp_files, f"leftover tmp files: {tmp_files}"
    owner_json = profiles_dir / "owner.json"
    assert owner_json.is_file()
    data = json.loads(owner_json.read_text(encoding="utf-8"))
    assert data["name"] == "Sanjeev"
    # no control/invisible unicode persisted
    assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", owner_json.read_text(encoding="utf-8"))


async def test_profile_block_respects_token_budget(kernel) -> None:
    memory = kernel.ctx.service("memory")
    # Stuff 200 bogus preferences; the block must still fit the budget.
    for i in range(200):
        memory.update_profile("owner", source="cli",
                              preferences={f"pref_{i}": f"value_{i}" * 20})
    block = memory.profile_block("owner", source="cli")
    from python_core.orchestrator import TokenBudget
    assert TokenBudget().count(block) <= 500, f"block too big: {block[:200]}"


async def test_profile_injected_into_every_system_prompt(kernel) -> None:
    await _teach_name(kernel, "my name is Sanjeev", "Sanjeev", user_id="owner", source="cli")
    msgs = await kernel.orchestrator._build_messages("hello", session_id="profile-main",
                                                     user_id="owner", source="cli")
    system = msgs[0]["content"]
    assert "USER PROFILE" in system, system
    assert "Sanjeev" in system, system
