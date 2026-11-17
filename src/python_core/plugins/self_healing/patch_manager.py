"""Diff generation, pytest validation, atomic apply/rollback (MODULE 2).

The self-modification pipeline in one call (:meth:`PatchManager.propose_patch`):

    1. Parse the proposed unified diff against the ORIGINAL file content.
    2. Materialize a full sandbox copy of the project with the patch applied.
    3. Run pytest inside the sandbox (sandbox-first imports).
    4. PASS  → atomically replace the live file (write temp → ``os.replace``),
               returning the applied diff for episodic memory.
       FAIL  → discard everything; the live tree was never touched. The caller
               (plugin glue) logs the failure to Episodic Memory so the agent
               *learns from the rejected patch*.

Only unified diffs are accepted — this is MODULE 4's delta-only policy applied
to self-modification: the agent thinks in diffs, never whole-file rewrites.
"""

from __future__ import annotations

import difflib
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .sandbox import Sandbox, SandboxResult

logger = logging.getLogger("echo.patch")


class PatchApplyError(ValueError):
    """The unified diff does not apply cleanly to the original content."""


@dataclass(slots=True)
class PatchOutcome:
    """Normalized result of a propose→validate→apply cycle."""

    applied: bool
    target_file: str
    diff: str = ""
    reason: str = ""
    sandbox_result: SandboxResult | None = None
    original_content: str = ""   # enables manual recovery / episodic storage
    patched_content: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


# ── unified diff machinery ───────────────────────────────────────────────────
def create_diff(original: str, modified: str, *, fromfile: str = "a", tofile: str = "b") -> str:
    """Unified diff of two file versions (the agent's preferred output shape).

    Note: difflib emits malformed hunks when the final line lacks a trailing
    newline ('-c+C' glued together). We therefore diff newline-normalized
    copies; ``apply_unified_diff`` preserves the ORIGINAL file's ending
    convention on apply, so the roundtrip is faithful for all practical
    patch flows (adding a trailing newline via a patch is the rare edge case
    better served by fs_write).
    """
    old_lines = original.splitlines(keepends=True)
    new_lines = modified.splitlines(keepends=True)
    if old_lines and not old_lines[-1].endswith("\n"):
        old_lines[-1] += "\n"
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"
    return "".join(
        difflib.unified_diff(old_lines, new_lines, fromfile=fromfile, tofile=tofile)
    )


def _parse_hunk_header(line: str) -> tuple[int, int]:
    """Extract (old_start, old_count) from an ``@@ -a,b +c,d @@`` header."""
    body = line.split("@@", 2)[1].strip()  # "-a,b +c,d"
    old_part = body.split("+")[0].strip().lstrip("-")
    start_s, _, count_s = old_part.partition(",")
    start = int(start_s)
    count = int(count_s) if count_s else 1
    return start, count


def apply_unified_diff(original: str, diff_text: str) -> str:
    """Apply a unified diff to *original* and return the new content.

    Implementation: classic hunk walking with tolerant context matching —
    when a hunk's context doesn't line up exactly, search nearby offsets
    (±5 lines) before failing. Raises :class:`PatchApplyError` on a genuine
    mismatch so the caller can ask the LLM for a corrected diff.
    """
    diff_lines = diff_text.splitlines(keepends=True)
    src = original.splitlines(keepends=True)
    out: list[str] = []
    pos = 0  # index into src consumed so far

    i = 0
    while i < len(diff_lines):
        line = diff_lines[i]
        if line.startswith("@@"):
            old_start, _old_count = _parse_hunk_header(line)
            # Collect the hunk body.
            j = i + 1
            hunk: list[str] = []
            while j < len(diff_lines) and not diff_lines[j].startswith("@@") and not diff_lines[j].startswith("diff "):
                hunk.append(diff_lines[j])
                j += 1

            # Optimistic target from the header (1-based → 0-based); the real
            # anchor is resolved by _apply_hunk's offset search, and the
            # untouched prefix is sliced up to the MATCHED position (not the
            # declared one) so offset-searched hunks never duplicate lines.
            target = max(0, old_start - 1)
            if target < pos:
                target = pos

            new_lines, matched_at = _apply_hunk(src, target, hunk)
            if matched_at is None:
                raise PatchApplyError(f"hunk at line {i + 1} does not apply (context mismatch)")
            out.extend(src[pos:max(matched_at, pos)])
            out.extend(new_lines)
            # Advance pos past the hunk's old span (context + deletions).
            pos = matched_at + _hunk_old_len(hunk)
            i = j
            continue
        # File headers / noise between hunks are skipped.
        i += 1

    out.extend(src[pos:])
    result = "".join(out)
    # Preserve the original's trailing-newline convention when the diff has no
    # explicit "\\ No newline at end of file" markers (difflib doesn't emit
    # them, so the file's own ending is the least-surprise default).
    if not original.endswith("\n") and "\\ No newline" not in diff_text and result.endswith("\n"):
        result = result[:-1]
    return result


def _hunk_old_len(hunk: list[str]) -> int:
    """Number of source lines a hunk consumes: context (' ...'), deletions
    ('-...'), and blank-line context. '+' lines and '\\ No newline' markers
    consume nothing."""
    n = 0
    for raw in hunk:
        if raw.startswith("+") or raw.startswith("\\"):
            continue
        n += 1
    return n


def _apply_hunk(src: list[str], start: int, hunk: list[str]) -> tuple[list[str], int | None]:
    """Try to apply one hunk at *start* (then ±offset search). Returns
    (replacement_lines, matched_index) or (…, None) on mismatch."""
    context: list[str] = []       # context/deletion lines expected from src
    replacement: list[str] = []   # full hunk body for the output
    for raw in hunk:
        body = raw.rstrip("\n")
        if raw.endswith("\n") is False and body in ("", "\\ No newline"):
            continue
        if raw.startswith(" "):
            context.append(body[1:] + "\n")
            replacement.append(body[1:] + "\n")
        elif raw.startswith("-"):
            context.append(body[1:] + "\n")
        elif raw.startswith("+"):
            replacement.append(body[1:] + "\n")
        elif raw.startswith("\\"):  # "\ No newline at end of file"
            if replacement:
                replacement[-1] = replacement[-1].rstrip("\n")
        # empty line inside hunk = blank context line
        elif raw.strip() == "":
            context.append("\n")
            replacement.append("\n")

    for offset in (0, -1, 1, -2, 2, -3, 3, -5, 5):
        at = start + offset
        if at < 0 or at + len(context) > len(src):
            continue
        candidate = src[at : at + len(context)]
        if [l.rstrip("\n") for l in candidate] == [l.rstrip("\n") for l in context]:
            return replacement, at
    return replacement, None


# ── atomic filesystem mutation ───────────────────────────────────────────────
def atomic_write(path: Path, content: str) -> None:
    """Write *content* to *path* atomically: temp file in the same directory,
    fsync, then ``os.replace``. Readers see either the old or the new file —
    never a partial write. POSIX-guaranteed; on Windows os.replace is atomic
    for same-volume targets as well.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)  # atomic on POSIX & same-volume Windows
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ── the pipeline ─────────────────────────────────────────────────────────────
class PatchManager:
    """propose → sandbox-validate → atomic-apply-or-discard."""

    def __init__(self, sandbox: Sandbox | None = None) -> None:
        self.sandbox = sandbox or Sandbox()

    async def propose_patch(
        self,
        project_root: Path,
        target_file: str,
        patch: str,
        *,
        test_path: str | None = None,
        timeout: float = 300.0,
        original_content: str | None = None,
    ) -> PatchOutcome:
        """Validate *patch* (unified diff for *target_file*) against a sandbox
        copy of *project_root*; apply atomically only if tests pass."""
        target = project_root / target_file
        if not target.is_file():
            return PatchOutcome(applied=False, target_file=target_file, reason=f"target file not found: {target_file}")

        original = original_content if original_content is not None else target.read_text(encoding="utf-8")
        try:
            patched = apply_unified_diff(original, patch)
        except PatchApplyError as exc:
            return PatchOutcome(applied=False, target_file=target_file, reason=f"diff rejected: {exc}", original_content=original)
        if patched == original:
            return PatchOutcome(applied=False, target_file=target_file, reason="no-op patch", original_content=original)

        # 1. Sandbox copy + overlay the patched file.
        files = {target_file: patched}
        # Also mirror the ORIGINAL file in case other tests import it.
        sandbox_root = self.sandbox.create(project_root, files=files, prefix="echo-patch-")
        try:
            # 2. Validate in the sandbox.
            result = await self.sandbox.run_pytest(sandbox_root, test_path=test_path, timeout=timeout)
            if not result.ok:
                # FAIL → discard. Live tree untouched. Learn from the failure.
                return PatchOutcome(
                    applied=False,
                    target_file=target_file,
                    diff=create_diff(original, patched),
                    reason="tests failed in sandbox",
                    sandbox_result=result,
                    original_content=original,
                    meta={"evidence": result.tail(3000)},
                )

            # 3. PASS → atomic apply to the live tree.
            atomic_write(target, patched)
            return PatchOutcome(
                applied=True,
                target_file=target_file,
                diff=create_diff(original, patched),
                reason="tests passed",
                sandbox_result=result,
                original_content=original,
                patched_content=patched,
            )
        finally:
            self.sandbox.cleanup(sandbox_root)
