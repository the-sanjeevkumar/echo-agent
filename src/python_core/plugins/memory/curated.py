"""Curated long-term stores: ``USER.md`` (owner identity) and ``MEMORY.md``
(agent notes: server facts, conventions, lessons). PHASE 2.

Hermes-style curated memory: two small, human-readable, bounded Markdown
files under ``<data_dir>/memory/`` that are snapshotted into *every* system
prompt. Writes go exclusively through the ``memory`` tool (add / patch /
remove operations) with:

* unique-substring matching (patch/remove apply only when the substring
  resolves to exactly one location),
* exact-duplicate rejection for adds,
* capacity errors on overflow that never silently drop content — the caller
  is instructed to consolidate into ONE batch instead,
* an optional write-approval gate (``config.memory_write_approval``;
  default off = auto-save like Hermes).

All writes are atomic (tmp + os.replace). Control/invisible unicode is
stripped before persistence.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

USER_CAP = 1375   # chars — USER.md holds the owner's identity
MEMORY_CAP = 2200  # chars — MEMORY.md holds agent notes

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize(text: str, *, preserve_newlines: bool = False) -> str:
    text = _CONTROL_CHARS.sub("", text or "")
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\t\n")
    if preserve_newlines:
        # Strip invisible/control unicode but keep the line structure intact.
        return "\n".join(" ".join(line.split()) for line in text.split("\n"))
    return " ".join(text.split())


class CuratedMemoryStore:
    """Two bounded curated files with an optional write-approval gate."""

    def __init__(self, memory_dir: Path, write_approval: bool = False) -> None:
        self.dir = Path(memory_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.user_path = self.dir / "USER.md"
        self.memory_path = self.dir / "MEMORY.md"
        self.caps = {"user": USER_CAP, "memory": MEMORY_CAP}
        self._paths = {"user": self.user_path, "memory": self.memory_path}
        self.write_approval = bool(write_approval)
        self._lock = threading.Lock()
        # Pending approvals are persisted so the CLI (a separate process) can
        # approve/reject writes queued by the live daemon.
        self._pending_path = self.dir / "pending.json"
        self.pending: dict[str, dict[str, Any]] = self._load_pending()
        self._pending_seq = max(
            (int(k.removeprefix("pend-")) for k in self.pending if k.startswith("pend-") and k[5:].isdigit()),
            default=0,
        )

    # ── pending persistence ────────────────────────────────────────────────
    def _load_pending(self) -> dict[str, dict[str, Any]]:
        try:
            raw = json.loads(self._pending_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)}

    def _save_pending(self) -> None:
        fd, tmp_name = tempfile.mkstemp(prefix=".pending-", dir=str(self.dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.pending, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_name, self._pending_path)
            tmp_name = None
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    # ── read / write ─────────────────────────────────────────────────────
    def _read(self, target: str) -> str:
        try:
            return self._paths[target].read_text(encoding="utf-8")
        except OSError:
            return ""

    def _write(self, target: str, text: str) -> None:
        path = self._paths[target]
        text = _sanitize(text, preserve_newlines=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target}-", dir=str(self.dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp_name, path)
            tmp_name = None
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    # ── public query ────────────────────────────────────────────────────
    def contents(self, target: str) -> str:
        return self._read(target)

    def snapshot(self) -> str:
        """Frozen snapshot injected into every system prompt.

        Each section carries a usage header (``USER.md [42% — 580/1375]``)
        and sections end with a ``§`` separator.
        """
        blocks: list[str] = []
        for target in ("user", "memory"):
            text = self._read(target)
            cap = self.caps[target]
            used = len(text)
            blocks.append(
                f"{target.upper()}.md [{used / cap:.0%} — {used}/{cap}]\n"
                f"{text.rstrip()}\n§"
            )
        return "\n".join(blocks) + "\n"

    # ── mutations ────────────────────────────────────────────────────────
    def apply(
        self,
        target: str,
        operations: list[dict[str, Any]],
        *,
        approved: bool = False,
    ) -> dict[str, Any]:
        """Validate + apply a batch of add/patch/remove operations.

        The whole batch is validated against the current file first; if any
        operation fails (duplicate, not found, ambiguous, capacity) the file
        is left untouched and an ERROR dict is returned — nothing is ever
        silently dropped.
        """
        if target not in ("user", "memory"):
            return {"ok": False, "error": "target must be 'user' or 'memory'"}
        cap = self.caps[target]
        current = self._read(target)
        lines = current.split("\n") if current else []

        try:
            new_lines = self._apply_operations(lines, operations)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        proposed = "\n".join(new_lines)
        if len(proposed) > cap:
            return {
                "ok": False,
                "error": (
                    f"capacity overflow: {target.upper()}.md would hold "
                    f"{len(proposed)}/{cap} chars. Consolidate into ONE "
                    f"consolidating batch (patch/remove stale entries, then "
                    f"re-add) before adding more — nothing was written."
                ),
            }

        if self.write_approval and not approved:
            self._pending_seq += 1
            pend_id = f"pend-{self._pending_seq:06d}"
            with self._lock:
                self.pending[pend_id] = {
                    "target": target,
                    "operations": operations,
                    "ts": time.time(),
                }
                self._save_pending()
            return {
                "ok": False,
                "pending": True,
                "id": pend_id,
                "target": target,
                "message": (
                    f"write-approval gate: {len(operations)} operation(s) for "
                    f"{target.upper()}.md await approval. Run /memory approve "
                    f"{pend_id} or /memory reject {pend_id}."
                ),
            }

        self._write(target, proposed)
        return {
            "ok": True,
            "target": target,
            "chars": len(proposed),
            "cap": cap,
            "operations": len(operations),
        }

    @staticmethod
    def _apply_operations(
        lines: list[str], operations: list[dict[str, Any]]
    ) -> list[str]:
        out = list(lines)
        for index, op in enumerate(operations):
            action = str(op.get("action", "")).strip().lower()
            content = _sanitize(str(op.get("content", "")))
            old_text = _sanitize(str(op.get("old_text", "")))
            if action not in {"add", "patch", "remove"}:
                raise ValueError(f"operation {index}: unknown action {action!r}")
            if action == "add":
                if not content:
                    raise ValueError(f"operation {index}: add with empty content")
                if content in out:
                    raise ValueError(
                        f"operation {index}: exact duplicate already present: {content[:80]}"
                    )
                out.append(content)
            elif action == "patch":
                if not old_text:
                    raise ValueError(f"operation {index}: patch needs old_text")
                if not content:
                    raise ValueError(f"operation {index}: patch with empty content")
                matches = [i for i, ln in enumerate(out) if old_text in ln]
                if not matches:
                    raise ValueError(f"operation {index}: patch target not found: {old_text[:80]}")
                if len(matches) > 1:
                    raise ValueError(
                        f"operation {index}: patch target is ambiguous ({len(matches)} matches): {old_text[:80]}"
                    )
                out[matches[0]] = out[matches[0]].replace(old_text, content)
            else:  # remove
                if not content:
                    raise ValueError(f"operation {index}: remove with empty content")
                matches = [i for i, ln in enumerate(out) if content in ln]
                if not matches:
                    raise ValueError(f"operation {index}: remove target not found: {content[:80]}")
                for i in reversed(matches):
                    out.pop(i)
        return out

    # ── approval gate ────────────────────────────────────────────────────
    def pending_list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{"id": k, **v} for k, v in self.pending.items()]

    def approve(self, pend_id: str) -> dict[str, Any]:
        with self._lock:
            pend = self.pending.pop(pend_id, None)
            if pend is not None:
                self._save_pending()
        if pend is None:
            return {"ok": False, "error": f"no pending write {pend_id}"}
        return self.apply(pend["target"], pend["operations"], approved=True)

    def reject(self, pend_id: str) -> dict[str, Any]:
        with self._lock:
            removed = self.pending.pop(pend_id, None)
            if removed is not None:
                self._save_pending()
        if removed is None:
            return {"ok": False, "error": f"no pending write {pend_id}"}
        return {"ok": True, "rejected": pend_id, "target": removed["target"]}

    def stats(self) -> dict[str, Any]:
        return {
            "user_chars": len(self._read("user")),
            "user_cap": USER_CAP,
            "memory_chars": len(self._read("memory")),
            "memory_cap": MEMORY_CAP,
            "pending": len(self.pending),
            "write_approval": self.write_approval,
        }
