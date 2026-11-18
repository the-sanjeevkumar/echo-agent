"""Hermes-style learning loop: background review, skill synthesis, self-healing.

PHASE 4. After every non-trivial task:completed event the loop schedules ONE
bounded structured review call (default the same API model; optionally a
cheaper one) that extracts facts / preferences / lessons / skill candidates.

Hard rules:

* all writes go through the EXISTING memory/profile/skills tool paths (never
  direct file writes from the review),
* cached and trivial turns are skipped, reviews are debounced, material is
  token-capped, and reviews are deferred while the bus is busy,
* failures are logged and NEVER break the task loop,
* recurring failures can propose a code patch through the EXISTING
  self-healing sandbox; code patches require chat approval by default.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("echo.learning")

# Structured output from the review call.
REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "facts": {"type": "array", "items": {"type": "string"}},
        "preferences": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                "required": ["key", "value"],
            },
        },
        "lessons": {"type": "array", "items": {"type": "string"}},
        "skill_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "when_to_use": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "tool_sequence": {"type": "array", "items": {"type": "string"}},
                    "pitfalls": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "when_to_use"],
            },
        },
    },
    "required": ["facts", "preferences", "lessons", "skill_candidates"],
}


class LearningLoop:
    """Post-turn background review + skill synthesis + patch proposals."""

    def __init__(
        self,
        bus: Any,
        config: Any,
        services_getter: Any,
    ) -> None:
        self.bus = bus
        self.config = config
        self._services_getter = services_getter
        self._sub = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._last_review = 0.0
        self._skill_signatures: dict[str, int] = {}
        self._failure_counts: dict[str, int] = {}
        self.reviews = 0
        self.skills_created = 0
        self.patches_proposed = 0
        self._dir = Path(config.data_dir) / "learning"
        self._patch_path = self._dir / "pending_patches.json"
        self.pending_patches: dict[str, dict[str, Any]] = self._load_patches()
        self._patch_seq = max(
            (int(k.removeprefix("patch-")) for k in self.pending_patches if k.startswith("patch-") and k[6:].isdigit()),
            default=0,
        )

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._sub is None:
            self._sub = self.bus.subscribe(
                "task:completed", self._on_task_completed, label="learning.review"
            )

    async def stop(self) -> None:
        if self._sub is not None:
            self.bus.unsubscribe(self._sub)
            self._sub = None
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def _service(self, name: str) -> Any:
        try:
            return self._services_getter().get(name)
        except Exception:  # noqa: BLE001
            return None

    # ── scheduling (debounce / skip / defer) ───────────────────────────────
    def _on_task_completed(self, event: Any) -> None:
        if not bool(getattr(self.config, "learning_enabled", True)):
            return
        payload = getattr(event, "payload", event) or {}
        # Skip cached and trivial turns (nothing durable to learn).
        if payload.get("cached"):
            return
        task_text = str(payload.get("task", "") or "")
        steps = payload.get("steps") or []
        if len(task_text) < 8 and not steps:
            return
        now = time.time()
        if now - self._last_review < float(getattr(self.config, "review_debounce_seconds", 30.0)):
            return
        pending = len(getattr(self.bus, "_tasks", ()) or ())
        if pending > 2:
            # Defer while the bus is busy — retry shortly instead of dropping
            # the review entirely (the task loop must never lose learning).
            self._schedule(self._deferred_review(dict(payload)), name="echo-learning-deferred")
            return
        self._last_review = now
        self._schedule(self._review(dict(payload)), name="echo-learning-review")

    def _schedule(self, coro: Any, *, name: str) -> None:
        try:
            task = asyncio.get_running_loop().create_task(coro, name=name)
        except RuntimeError:  # no running loop (sync invocation)
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _deferred_review(self, payload: dict[str, Any]) -> None:
        await asyncio.sleep(0.3)
        self._last_review = time.time()
        await self._review(payload)

    # ── the review itself ──────────────────────────────────────────────────
    async def _review(self, payload: dict[str, Any]) -> None:
        try:
            material = self._material(payload)
            result: dict[str, Any] = {}
            llm = self._service("llm")
            if llm is not None:
                try:
                    out = await llm.complete(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "Review one agent turn and extract durable learning. Return ONLY "
                                    "JSON matching the schema: {facts: string[], preferences: "
                                    "[{key,value}], lessons: string[], skill_candidates: "
                                    "[{name, when_to_use, steps?, tool_sequence?, pitfalls?}]}. "
                                    "Only stable facts and reusable procedures; no ephemera. "
                                    "If a fix is a recurring code bug, put the proposed unified diff "
                                    "in a lesson prefixed with PATCH: <relative/path>: <diff>."
                                ),
                            },
                            {"role": "user", "content": material[: self.config.review_max_chars]},
                        ],
                        schema=REVIEW_SCHEMA,
                        temperature=0.0,
                        max_tokens=500,
                    )
                    content = out.get("content") if isinstance(out, dict) else None
                    if isinstance(content, dict):
                        result = content
                except Exception:  # noqa: BLE001
                    logger.debug("review LLM call failed; using offline heuristics")
            if not result:
                result = self._offline_review(payload)
            await self._apply_review(result, payload)
            self.reviews += 1
        except Exception:  # review failures NEVER break the task loop
            logger.exception("background review failed")

    @staticmethod
    def _material(payload: dict[str, Any]) -> str:
        steps = payload.get("steps") or []
        actions = "; ".join(
            f"{s.get('tool')}({'ok' if s.get('ok') else 'FAIL'})" for s in steps
        ) or "no-tools"
        return (
            f"USER: {str(payload.get('task', ''))[:600]}\n"
            f"AGENT: {str(payload.get('response', ''))[:600]}\n"
            f"STEPS: {actions[:400]}"
        )

    @staticmethod
    def _offline_review(payload: dict[str, Any]) -> dict[str, Any]:
        """Zero-LLM heuristic: failures become lessons, no invented facts."""
        lessons = []
        for step in payload.get("steps") or []:
            if not step.get("ok"):
                lessons.append(f"Tool {step.get('tool')} failed; verify arguments before retrying.")
        return {"facts": [], "preferences": [], "lessons": lessons, "skill_candidates": []}

    async def _apply_review(self, result: dict[str, Any], payload: dict[str, Any]) -> None:
        memory = self._service("memory")
        user_id = str(payload.get("user_id") or "")
        source = str(payload.get("source") or "")

        # 1. Curated memory writes — THROUGH the `memory` tool path.
        facts = [str(f) for f in (result.get("facts") or []) if str(f).strip()][:5]
        lessons = [str(l) for l in (result.get("lessons") or []) if str(l).strip()][:5]
        if (facts or lessons) and memory is not None and hasattr(memory, "curated_apply"):
            memory.curated_apply("memory", [
                {"action": "add", "content": f"Lesson: {item}"}
                for item in facts + lessons
            ])

        # 2. Preferences — THROUGH the profile tool path.
        prefs = {}
        for item in result.get("preferences") or []:
            if isinstance(item, dict) and item.get("key") and item.get("value"):
                prefs[str(item["key"])] = str(item["value"])
        if prefs and memory is not None and hasattr(memory, "update_profile"):
            memory.update_profile(user_id, source, preferences=prefs)

        # 3. Skill synthesis — recurring 2+ step successes OR review candidates.
        steps = payload.get("steps") or []
        succeeded = bool(steps) and all(step.get("ok") for step in steps)
        review_candidates = [c for c in (result.get("skill_candidates") or []) if isinstance(c, dict) and c.get("name")]
        if succeeded and len(steps) >= 2 and not review_candidates:
            signature = "+".join(str(step.get("tool")) for step in steps)[:240]
            self._skill_signatures[signature] = self._skill_signatures.get(signature, 0) + 1
            if self._skill_signatures[signature] >= 2:
                candidate = {
                    "name": re.sub(r"[^a-z0-9]+", "-", signature.lower()).strip("-")[:40] or "workflow",
                    "when_to_use": f"Multi-step workflow: {signature}",
                    "steps": [f"Call {step.get('tool')}" for step in steps],
                    "tool_sequence": [str(step.get("tool")) for step in steps],
                    "pitfalls": [],
                }
                await self._synthesize(candidate)

        for candidate in review_candidates[:3]:
            if isinstance(candidate, dict) and candidate.get("name"):
                await self._synthesize(candidate, patch=bool(payload.get("correction")))

        # 4. Patch proposals from recurring failure lessons.
        for lesson in lessons:
            if lesson.startswith("PATCH:"):
                self._note_failure(lesson)

    async def _synthesize(self, candidate: dict[str, Any], *, patch: bool = False) -> dict[str, Any]:
        skills = self._service("skills")
        if skills is None or not hasattr(skills, "synthesize_skill"):
            return {"ok": False, "error": "skills service unavailable"}
        result = await skills.synthesize_skill(
            str(candidate.get("name", "workflow")),
            when_to_use=str(candidate.get("when_to_use", "")),
            steps=[str(s) for s in (candidate.get("steps") or [])],
            tool_sequence=[str(s) for s in (candidate.get("tool_sequence") or [])],
            pitfalls=[str(s) for s in (candidate.get("pitfalls") or [])],
            patch=patch,
        )
        if result.get("created"):
            self.skills_created += 1
            self.bus.publish("skill:synthesized", {"name": result.get("name")}, source="learning")
        return result

    # ── code self-improvement (approval-gated sandbox flow) ─────────────────
    def _note_failure(self, lesson: str) -> None:
        """Track recurring patch lessons; propose once the threshold is hit."""
        match = re.match(r"PATCH:\s*([^\s:]+):\s*(.+)", lesson, re.DOTALL)
        if not match:
            return
        target_file, patch = match.group(1).strip(), match.group(2).strip()
        key = f"{target_file}:{hash(patch) & 0xFFFFFFFF}"
        self._failure_counts[key] = self._failure_counts.get(key, 0) + 1
        if self._failure_counts[key] >= int(getattr(self.config, "review_min_failure_count", 3)):
            self._failure_counts[key] = 0  # propose once per threshold crossing
            coro = self.propose_code_patch(target_file, patch)
            try:
                task = asyncio.get_running_loop().create_task(coro, name="echo-learning-patch")
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            except RuntimeError:
                logger.debug("no running loop; patch proposal deferred")

    # ── pending patch persistence ──────────────────────────────────────────
    def _load_patches(self) -> dict[str, dict[str, Any]]:
        try:
            raw = json.loads(self._patch_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)} if isinstance(raw, dict) else {}

    def _save_patches(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".patches-", dir=str(self._dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.pending_patches, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_name, self._patch_path)
            tmp_name = None
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    async def propose_code_patch(
        self, target_file: str, patch: str, *, test_path: str | None = None
    ) -> dict[str, Any]:
        """Route a proposed code patch through the existing sandbox.

        Code changes require chat approval by default; only when
        ``auto_apply_if_tests_pass`` is enabled is the sandbox run immediately.
        """
        self.patches_proposed += 1
        if not bool(getattr(self.config, "auto_apply_if_tests_pass", False)):
            self._patch_seq += 1
            pend_id = f"patch-{self._patch_seq:06d}"
            self.pending_patches[pend_id] = {
                "target_file": target_file,
                "patch": patch,
                "test_path": test_path,
                "ts": time.time(),
            }
            self._save_patches()
            self.bus.publish(
                "learning:patch_proposed", {"id": pend_id, "target_file": target_file}, source="learning"
            )
            return {"ok": True, "pending": True, "id": pend_id, "target_file": target_file}

        healing = self._service("self_healing")
        if healing is None:
            return {"ok": False, "error": "self_healing unavailable"}
        outcome = await healing.propose_patch(target_file, patch, test_path=test_path)
        return {"ok": True, "pending": False, **outcome}

    async def approve_patch(self, pend_id: str) -> dict[str, Any]:
        entry = self.pending_patches.pop(pend_id, None)
        if entry is None:
            return {"ok": False, "error": f"no pending patch {pend_id}"}
        self._save_patches()
        healing = self._service("self_healing")
        if healing is None:
            return {"ok": False, "error": "self_healing unavailable"}
        outcome = await healing.propose_patch(
            entry["target_file"], entry["patch"], test_path=entry.get("test_path")
        )
        self.bus.publish(
            "learning:patch_approved", {"id": pend_id, "result": outcome}, source="learning"
        )
        return {"ok": True, "id": pend_id, **outcome}

    def reject_patch(self, pend_id: str) -> dict[str, Any]:
        if self.pending_patches.pop(pend_id, None) is None:
            return {"ok": False, "error": f"no pending patch {pend_id}"}
        self._save_patches()
        return {"ok": True, "rejected": pend_id}

    def stats(self) -> dict[str, Any]:
        return {
            "reviews": self.reviews,
            "skills_created": self.skills_created,
            "patches_proposed": self.patches_proposed,
            "pending_patches": len(self.pending_patches),
        }
