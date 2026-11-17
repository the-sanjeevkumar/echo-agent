"""Self-healing: diff machinery, sandboxed validation, atomic apply/rollback."""

from __future__ import annotations

from pathlib import Path

import pytest

from python_core.plugins.self_healing.patch_manager import (
    PatchApplyError,
    PatchManager,
    apply_unified_diff,
    atomic_write,
    create_diff,
)
from python_core.plugins.self_healing.sandbox import Sandbox


# ── diff roundtrip ───────────────────────────────────────────────────────────
def test_diff_roundtrip() -> None:
    original = "line1\nline2\nline3\n"
    modified = "line1\nLINE2\nline3\n"
    diff = create_diff(original, modified)
    assert "@@" in diff
    assert apply_unified_diff(original, diff) == modified


def test_diff_roundtrip_no_trailing_newline() -> None:
    original = "a\nb\nc"
    modified = "a\nb\nC"
    diff = create_diff(original, modified)
    assert apply_unified_diff(original, diff) == modified


def test_diff_multiline_hunk() -> None:
    original = "\n".join(f"def f{i}(): pass" for i in range(20)) + "\n"
    modified = original.replace("def f10(): pass", "def f10(): return 10")
    diff = create_diff(original, modified)
    assert apply_unified_diff(original, diff) == modified


def test_bad_context_rejected() -> None:
    diff = """--- a
+++ b
@@ -1,2 +1,2 @@
 missing context line that was never in the file
-also missing
+replacement
"""
    with pytest.raises(PatchApplyError):
        apply_unified_diff("totally\ndifferent\ncontent\n", diff)


def test_offset_tolerance() -> None:
    """Hunk header off by one still applies via offset search."""
    original = "one\ntwo\nthree\n"
    diff = """--- a
+++ b
@@ -3,1 +3,1 @@
-two
+TWO
"""
    # header says line 3 but 'two' is line 2 → ±1 offset search saves it
    assert apply_unified_diff(original, diff) == "one\nTWO\nthree\n"


# ── atomic write ─────────────────────────────────────────────────────────────
def test_atomic_write(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "file.txt"
    atomic_write(target, "v1")
    atomic_write(target, "v2")
    assert target.read_text() == "v2"
    # no temp residue
    assert list(tmp_path.rglob("*.tmp")) == []


# ── sandbox ──────────────────────────────────────────────────────────────────
def test_sandbox_copies_tree_with_overlays(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("X = 1\n")
    (tmp_path / "pkg" / "__pycache__" / "mod.cpython.pyc").write_text("binary")
    sandbox = Sandbox()
    root = sandbox.create(tmp_path, files={"new.txt": "hello"})
    try:
        assert (root / "pkg" / "mod.py").read_text() == "X = 1\n"
        assert (root / "new.txt").read_text() == "hello"
        assert not (root / "pkg" / "__pycache__").exists()  # ignored
    finally:
        sandbox.cleanup(root)


# ── the full self-modification pipeline ──────────────────────────────────────
@pytest.fixture
def tiny_project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    (proj / "tests").mkdir(parents=True)
    (proj / "app.py").write_text('def greet():\n    return "hello"\n')
    (proj / "tests" / "test_app.py").write_text(
        "from app import greet\n\ndef test_greet():\n    assert greet() == 'hello'\n"
    )
    return proj


async def test_propose_patch_applied_when_tests_pass(tiny_project: Path) -> None:
    """A behavior-preserving (additive) patch survives validation and applies."""
    original = (tiny_project / "app.py").read_text()
    modified = original + '\n\ndef farewell():\n    return "goodbye"\n'
    diff = create_diff(original, modified)

    pm = PatchManager()
    outcome = await pm.propose_patch(tiny_project, "app.py", diff)
    assert outcome.applied is True
    assert outcome.sandbox_result.ok is True
    # live file changed atomically
    assert (tiny_project / "app.py").read_text() == modified

    # follow-up: patch the TEST SUITE itself (agents extend their own tests)
    test_original = (tiny_project / "tests" / "test_app.py").read_text()
    test_modified = test_original + "\n\ndef test_farewell():\n    from app import farewell\n    assert farewell() == 'goodbye'\n"
    outcome2 = await pm.propose_patch(tiny_project, "tests/test_app.py", create_diff(test_original, test_modified))
    assert outcome2.applied is True


async def test_propose_patch_rejected_when_tests_fail(tiny_project: Path) -> None:
    """THE critical safety property: failing tests ⇒ live file untouched."""
    original = (tiny_project / "app.py").read_text()
    broken = original.replace('"hello"', '"broken"')
    diff = create_diff(original, broken)

    pm = PatchManager()
    outcome = await pm.propose_patch(tiny_project, "app.py", diff)
    assert outcome.applied is False
    assert outcome.reason == "tests failed in sandbox"
    assert "assert" in (outcome.meta.get("evidence", "") or outcome.sandbox_result.stderr or "")
    # LIVE FILE NEVER TOUCHED
    assert (tiny_project / "app.py").read_text() == original


async def test_propose_patch_rejects_nonapplicable_diff(tiny_project: Path) -> None:
    pm = PatchManager()
    bad_diff = "--- a\n+++ b\n@@ -1,1 +1,1 @@\n-not in file\n+nope\n"
    outcome = await pm.propose_patch(tiny_project, "app.py", bad_diff)
    assert outcome.applied is False
    assert "diff rejected" in outcome.reason


async def test_propose_patch_noop_rejected(tiny_project: Path) -> None:
    original = (tiny_project / "app.py").read_text()
    pm = PatchManager()
    outcome = await pm.propose_patch(tiny_project, "app.py", create_diff(original, original))
    assert outcome.applied is False and outcome.reason == "no-op patch"


async def test_self_healing_plugin_tool_roundtrip(kernel, tiny_project: Path, monkeypatch) -> None:
    """Tool-level flow: self_patch applies a good diff and logs an episode."""
    heal = kernel.ctx.service("self_healing")
    monkeypatch.setattr(heal.config, "project_root", str(tiny_project))

    original = (tiny_project / "app.py").read_text()
    modified = original + '\n\ndef version_tag():\n    return "v2"\n'
    diff = create_diff(original, modified)

    result = await kernel.ctx.tools.execute(
        "self_patch", {"target_file": "app.py", "patch": diff}
    )
    assert result["ok"] and result["result"]["applied"] is True
    assert (tiny_project / "app.py").read_text() == modified

    # verify the episodic record exists for the applied patch
    memory = kernel.ctx.service("memory")
    assert any("self_patch(app.py)" in e["action"] for e in memory.em.recent(limit=10))


async def test_self_healing_plugin_rejection_logged(kernel, tiny_project: Path, monkeypatch) -> None:
    """Rejected patches are LEARNED FROM: logged to episodic memory with evidence."""
    heal = kernel.ctx.service("self_healing")
    monkeypatch.setattr(heal.config, "project_root", str(tiny_project))

    original = (tiny_project / "app.py").read_text()
    diff = create_diff(original, original.replace('"hello"', '"broken"'))
    await kernel.ctx.tools.execute("self_patch", {"target_file": "app.py", "patch": diff})

    memory = kernel.ctx.service("memory")
    episodes = memory.em.recent(limit=10)
    fail = [e for e in episodes if e["action"].startswith("self_patch") and not e["success"]]
    assert fail, "rejection must be recorded as a failed episode"
    assert fail[0]["outcome"] == "tests failed in sandbox"
    assert "evidence" in fail[0]["metadata"] and fail[0]["metadata"]["evidence"]

    # and the rollback tool restores the last good content
    assert (tiny_project / "app.py").read_text() == original  # untouched after rejection
