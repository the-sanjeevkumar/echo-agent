"""PHASE 2: curated USER.md / MEMORY.md stores and the `memory` tool.

Acceptance coverage:
* capacity overflow returns an ERROR instructing ONE consolidating batch;
  nothing is silently dropped
* exact-duplicate add rejected; patch/remove use unique-substring matching
* frozen snapshot (usage headers + § separators) injected into every system
  prompt
* optional write-approval gate (default off) with pending/approve/reject
"""

from __future__ import annotations

from pathlib import Path

from python_core.plugins.memory.curated import MEMORY_CAP, USER_CAP, CuratedMemoryStore

# ── direct store tests ────────────────────────────────────────────────────────

def test_snapshot_usage_headers_and_separators(tmp_path: Path) -> None:
    store = CuratedMemoryStore(tmp_path)
    store.apply("user", [{"action": "add", "content": "Name: Sanjeev"}])
    store.apply("memory", [{"action": "add", "content": "Server uses Nemotron via OpenAI-compatible API"}])

    snap = store.snapshot()
    assert "USER.md [" in snap and f"/{USER_CAP}]" in snap
    assert "MEMORY.md [" in snap and f"/{MEMORY_CAP}]" in snap
    assert snap.count("§") == 2
    assert "Name: Sanjeev" in snap
    assert "Nemotron" in snap
    # usage percentage is rendered
    assert "%" in snap


def test_add_rejects_exact_duplicate(tmp_path: Path) -> None:
    store = CuratedMemoryStore(tmp_path)
    first = store.apply("user", [{"action": "add", "content": "Prefer concise replies"}])
    assert first["ok"] is True

    dup = store.apply("user", [{"action": "add", "content": "Prefer concise replies"}])
    assert dup["ok"] is False and "duplicate" in dup["error"].lower()
    assert store.contents("user") == "Prefer concise replies"


def test_patch_and_remove_unique_substring(tmp_path: Path) -> None:
    store = CuratedMemoryStore(tmp_path)
    store.apply("memory", [{"action": "add", "content": "Deploy: use the release checklist"}])
    store.apply("memory", [{"action": "add", "content": "Deploy: staging needs approval"}])

    patched = store.apply("memory", [{
        "action": "patch", "old_text": "release checklist", "content": "the v2 release checklist"
    }])
    assert patched["ok"] is True
    assert "v2 release checklist" in store.contents("memory")

    ambiguous = store.apply("memory", [{"action": "patch", "old_text": "Deploy", "content": "x"}])
    assert ambiguous["ok"] is False and "ambiguous" in ambiguous["error"].lower()

    missing = store.apply("memory", [{"action": "patch", "old_text": "not present", "content": "x"}])
    assert missing["ok"] is False and "not found" in missing["error"].lower()

    removed = store.apply("memory", [{"action": "remove", "content": "staging needs approval"}])
    assert removed["ok"] is True
    assert "staging needs approval" not in store.contents("memory")
    assert "v2 release checklist" in store.contents("memory")  # other line intact


def test_capacity_error_never_silently_drops(tmp_path: Path) -> None:
    store = CuratedMemoryStore(tmp_path)
    # Fill MEMORY.md close to capacity with one line.
    filler = "x" * (MEMORY_CAP - 20)
    assert store.apply("memory", [{"action": "add", "content": filler}])["ok"] is True
    before = store.contents("memory")

    result = store.apply("memory", [{"action": "add", "content": "y" * 100}])
    assert result["ok"] is False
    assert "capacity" in result["error"].lower()
    assert "ONE" in result["error"] and "consolidat" in result["error"].lower()
    # Nothing was written — the overflowing entry was not silently dropped.
    assert store.contents("memory") == before


def test_batch_is_atomic_on_failure(tmp_path: Path) -> None:
    store = CuratedMemoryStore(tmp_path)
    result = store.apply("user", [
        {"action": "add", "content": "first fact"},
        {"action": "add", "content": "second fact"},
        {"action": "patch", "old_text": "absent", "content": "broken"},
    ])
    assert result["ok"] is False
    assert store.contents("user") == ""


def test_control_unicode_stripped(tmp_path: Path) -> None:
    store = CuratedMemoryStore(tmp_path)
    store.apply("user", [{"action": "add", "content": "Na\x00me: \u200bSanjeev"}])
    content = store.contents("user")
    assert "\x00" not in content and "\u200b" not in content
    assert "Sanjeev" in content


# ── tool registration + prompt injection (full kernel) ───────────────────────

async def test_memory_tool_registration_and_snapshot_injection(kernel) -> None:
    result = await kernel.ctx.tools.execute("memory", {
        "target": "user",
        "operations": [{"action": "add", "content": "The owner's name is Sanjeev"}],
    })
    assert result["ok"] and result["result"]["ok"]
    await kernel.ctx.tools.execute("memory", {
        "target": "memory",
        "operations": [{"action": "add", "content": "Server disk is on /dev/sda1"}],
    })

    msgs = await kernel.orchestrator._build_messages(
        "hello", session_id="phase2", user_id="owner", source="cli"
    )
    system = msgs[0]["content"]
    assert "USER.md [" in system and "The owner's name is Sanjeev" in system
    assert "MEMORY.md [" in system and "Server disk is on /dev/sda1" in system
    assert "§" in system


async def test_memory_tool_capacity_error_via_registry(kernel) -> None:
    filler = "z" * (MEMORY_CAP - 20)
    first = await kernel.ctx.tools.execute("memory", {
        "target": "memory", "operations": [{"action": "add", "content": filler}],
    })
    assert first["result"]["ok"] is True
    second = await kernel.ctx.tools.execute("memory", {
        "target": "memory", "operations": [{"action": "add", "content": "q" * 100}],
    })
    assert second["ok"] and second["result"]["ok"] is False
    assert "consolidat" in second["result"]["error"].lower()


# ── write-approval gate ───────────────────────────────────────────────────────

def test_approval_gate_pending_approve_reject(tmp_path: Path) -> None:
    store = CuratedMemoryStore(tmp_path, write_approval=True)
    result = store.apply("user", [{"action": "add", "content": "gated fact"}])
    assert result["ok"] is False and result["pending"] is True
    pend_id = result["id"]
    assert store.contents("user") == ""

    pending = store.pending_list()
    assert any(item["id"] == pend_id for item in pending)

    # pending persists across instances (CLI is a separate process)
    store2 = CuratedMemoryStore(tmp_path, write_approval=True)
    assert any(item["id"] == pend_id for item in store2.pending_list())

    approved = store2.approve(pend_id)
    assert approved["ok"] is True
    assert "gated fact" in store2.contents("user")
    assert store2.pending_list() == []

    pending2 = store2.apply("memory", [{"action": "add", "content": "discard me"}])
    rejected = store2.reject(pending2["id"])
    assert rejected["ok"] is True
    assert "discard me" not in store2.contents("memory")


def test_approval_off_by_default(tmp_path: Path) -> None:
    store = CuratedMemoryStore(tmp_path)
    assert store.write_approval is False
    result = store.apply("user", [{"action": "add", "content": "auto-saved"}])
    assert result["ok"] is True and "auto-saved" in store.contents("user")
