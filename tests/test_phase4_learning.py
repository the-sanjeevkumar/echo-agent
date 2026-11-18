"""PHASE 4: learning loop (review tool-path writes, skill synthesis,
consolidator dedup, approval-gated sandbox patch proposals)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from python_core.core.event_bus import EventBus
from python_core.plugins.memory.consolidator import ConsolidationLoop
from python_core.plugins.memory.storage import EpisodicMemory, SemanticMemory, WorkingMemory


async def _until(predicate, *, timeout: float = 8.0, period: float = 0.05) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(period)
    raise AssertionError("condition not met in time")


def _two_step_script(response: str = "done") -> list[dict]:
    return [
        {"thought": "read", "action": "tool", "tool": "fs_read", "args": {"path": "pyproject.toml", "max_chars": 20}},
        {"thought": "read2", "action": "tool", "tool": "fs_read", "args": {"path": "pyproject.toml", "max_chars": 20}},
        {"thought": "reply", "action": "respond", "response": response},
    ]


# ── skill synthesis ───────────────────────────────────────────────────────────

async def test_same_two_step_task_twice_creates_one_skill(kernel) -> None:
    kernel.config.review_debounce_seconds = 0.0
    kernel.provider.review_return = {
        "facts": [], "preferences": [], "lessons": [],
        "skill_candidates": [{
            "name": "deploy-checklist",
            "when_to_use": "When deploying the service to production",
            "steps": ["read the checklist", "restart the service"],
            "tool_sequence": ["fs_read", "shell_exec"],
            "pitfalls": ["never skip tests"],
        }],
    }

    for _ in range(2):
        kernel.provider.script = _two_step_script()
        await kernel.orchestrator.handle_task(
            "deploy the service using the checklist", source="cli", user_id="owner", session_id="phase4"
        )

    skills_root = kernel.config.resolved_skills_dir
    await _until(lambda: (skills_root / "deploy_checklist" / "SKILL.md").is_file())
    skill_files = list(skills_root.glob("*/SKILL.md"))
    assert len(skill_files) == 1, f"expected exactly one skill, got {skill_files}"

    text = skill_files[0].read_text(encoding="utf-8")
    for section in ("When to use", "Steps", "Tool sequence", "Pitfalls"):
        assert section in text
    assert "fs_read" in text and "never skip tests" in text

    # Visible in the NEXT prompt.
    msgs = await kernel.orchestrator._build_messages(
        "next task", session_id="phase4", user_id="owner", source="cli"
    )
    assert "SKILLS" in msgs[0]["content"]
    assert "deploy_checklist" in msgs[0]["content"]


# ── review writes go through tool paths ───────────────────────────────────────

async def test_review_writes_through_memory_and_profile_tools(kernel, monkeypatch) -> None:
    kernel.config.review_debounce_seconds = 0.0
    kernel.provider.review_return = {
        "facts": ["the staging server runs on port 8790"],
        "preferences": [{"key": "reply_style", "value": "terse"}],
        "lessons": ["check file existence before fs_read"],
        "skill_candidates": [],
    }
    memory = kernel.ctx.service("memory")
    curated_calls: list[dict] = []
    profile_calls: list[dict] = []
    orig_curated, orig_profile = memory.curated_apply, memory.update_profile

    def spy_curated(target, operations):
        curated_calls.append({"target": target, "operations": operations})
        return orig_curated(target, operations)

    def spy_profile(user_id, source=None, **kwargs):
        profile_calls.append({"user_id": user_id, "source": source, **kwargs})
        return orig_profile(user_id, source, **kwargs)

    monkeypatch.setattr(memory, "curated_apply", spy_curated)
    monkeypatch.setattr(memory, "update_profile", spy_profile)

    kernel.provider.script = [
        {"thought": "read", "action": "tool", "tool": "fs_read", "args": {"path": "pyproject.toml", "max_chars": 20}},
        {"thought": "reply", "action": "respond", "response": "ok"},
    ]
    await kernel.orchestrator.handle_task(
        "inspect the project configuration", source="cli", user_id="owner", session_id="phase4-review"
    )

    await _until(lambda: len(kernel.provider.review_calls) >= 1)
    await _until(lambda: "port 8790" in memory.curated.contents("memory"))
    assert curated_calls and any(c["target"] == "memory" for c in curated_calls)
    assert "Lesson: the staging server runs on port 8790" in memory.curated.contents("memory")
    assert profile_calls and any("reply_style" in (c.get("preferences") or {}) for c in profile_calls)
    assert "terse" in (memory.profiles.get("owner").preferences.get("reply_style") or "")


# ── consolidator dedup ────────────────────────────────────────────────────────

async def test_consolidator_dedup_stops_docs_growth(tmp_path: Path) -> None:
    bus = EventBus()
    wm = WorkingMemory(capacity=32)
    em = EpisodicMemory(tmp_path / "ep.sqlite3")
    sm = SemanticMemory(tmp_path / "vec")
    wm.push("fix flaky test retry logic", salience=0.9)
    wm.push("fix flaky test retry logic!", salience=0.9)
    em.append("shell git push", "rejected", success=False)

    loop = ConsolidationLoop(bus, wm, em, sm, idle_seconds=1, min_interval_seconds=0)
    first = await loop.consolidate()
    count_after_first = sm.count()
    lines_after_first = len((tmp_path / "vec" / "docs.jsonl").read_text(encoding="utf-8").splitlines())

    second = await loop.consolidate()
    assert sm.count() == count_after_first, (first, second)
    assert len((tmp_path / "vec" / "docs.jsonl").read_text(encoding="utf-8").splitlines()) == lines_after_first
    assert second.get("merged", 0) >= 1, second
    em.close()


# ── approval-gated code patches through the sandbox ───────────────────────────

async def test_code_patch_needs_approval_then_flows_to_sandbox(kernel, monkeypatch) -> None:
    healing = kernel.ctx.service("self_healing")
    learning = kernel.ctx.service("learning")
    calls: list[dict] = []

    async def fake_propose(target_file, patch, *, test_path=None, project_root=None):
        calls.append({"target_file": target_file, "patch": patch, "test_path": test_path})
        return {"applied": True, "reason": "tests passed", "diff": ""}

    monkeypatch.setattr(healing, "propose_patch", fake_propose)

    outcome = await learning.propose_code_patch("src/python_core/plugins/memory/plugin.py", "--- a\n+++ b\n@@ ...")
    assert outcome["pending"] is True and outcome["id"]
    assert calls == [], "no sandbox call before approval"

    patch_id = outcome["id"]
    result = await kernel.orchestrator.handle_task(
        f"/patch approve {patch_id}", source="cli", user_id="owner", session_id="phase4-patch"
    )
    assert len(calls) == 1
    assert calls[0]["target_file"] == "src/python_core/plugins/memory/plugin.py"
    assert "applied" in result.get("response", "").lower() or patch_id in result.get("response", "")


async def test_auto_apply_runs_sandbox_directly(kernel, monkeypatch) -> None:
    kernel.config.auto_apply_if_tests_pass = True
    healing = kernel.ctx.service("self_healing")
    learning = kernel.ctx.service("learning")
    calls: list[dict] = []

    async def fake_propose(target_file, patch, *, test_path=None, project_root=None):
        calls.append({"target_file": target_file, "patch": patch})
        return {"applied": True, "reason": "tests passed", "diff": ""}

    monkeypatch.setattr(healing, "propose_patch", fake_propose)
    outcome = await learning.propose_code_patch("src/python_core/foo.py", "--- a\n+++ b")
    assert outcome["pending"] is False and len(calls) == 1


async def test_patch_commands_owner_only(kernel) -> None:
    result = await kernel.orchestrator.handle_task(
        "/patch list", source="telegram", user_id="999000111", session_id="phase4-patch"
    )
    assert "owner-only" in result.get("response", "")
