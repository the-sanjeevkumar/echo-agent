"""Background consolidation loop (MODULE 1).

Biological inspiration: during idle time ("rest"), the brain replays recent
experience and promotes what matters into durable memory. Here:

    WorkingMemory ──(idle trigger)──▶ pattern extraction ──▶ EpisodicMemory
                                                          └─▶ SemanticMemory

Trigger policy: consolidation runs when the bus has been idle for
``config.consolidation_idle_seconds`` and at most once per
``min_interval_seconds``. Activity is sensed by a low-priority "*" listener,
so any event (task, tool call, gateway ping) postpones consolidation —
the agent never "daydreams" while working.

Extraction strategy (local-first):
    1. If an LLM service is registered, one structured call produces dense,
       high-value pattern summaries (bounded, schema-forced).
    2. Otherwise a deterministic heuristic does the same job offline:
       cluster near-duplicate working-memory items (hash-embedding cosine),
       count repeated action→outcome signatures, weight by success rate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from python_core.core.event_bus import EventBus
from python_core.core.utils import cosine

from .storage import EpisodicMemory, SemanticMemory, WorkingMemory

logger = logging.getLogger("echo.consolidator")

PATTERNS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {"type": "string", "description": "one dense sentence"},
                    "kind": {"type": "string", "enum": ["skill", "fact", "preference", "cause-effect"]},
                    "salience": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["summary", "kind", "salience"],
            },
        }
    },
    "required": ["patterns"],
}


class ConsolidationLoop:
    """Async task: watch idleness, extract patterns, consolidate to EM/SM."""

    def __init__(
        self,
        bus: EventBus,
        wm: WorkingMemory,
        em: EpisodicMemory,
        sm: SemanticMemory,
        *,
        idle_seconds: float = 20.0,
        min_interval_seconds: float = 60.0,
        llm: Any | None = None,
        services_getter: Any | None = None,
    ) -> None:
        self.bus = bus
        self.wm = wm
        self.em = em
        self.sm = sm
        self.idle_seconds = idle_seconds
        self.min_interval = min_interval_seconds
        self.llm = llm  # explicit provider; usually resolved lazily via services_getter
        self._services_getter = services_getter  # callable returning the kernel services dict
        self.last_activity = time.time()
        self.last_consolidation = 0.0
        self.consolidations = 0
        self.patterns_written = 0
        self._task: asyncio.Task[None] | None = None
        self._activity_sub = bus.subscribe("*", self._on_activity, priority=-1000, label="consolidator.activity")

    # ── activity sensing ───────────────────────────────────────────────────
    def _on_activity(self, event: Any) -> None:
        self.last_activity = time.time()

    def notify_activity(self) -> None:
        self.last_activity = time.time()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="echo-consolidator")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.bus.unsubscribe(self._activity_sub)

    async def _run(self) -> None:
        """Idle-watch loop. Never raises: consolidation failures are logged
        and the loop keeps watching (self-healing at the process level)."""
        while True:
            await asyncio.sleep(2.0)
            idle_for = time.time() - self.last_activity
            since_last = time.time() - self.last_consolidation
            if idle_for >= self.idle_seconds and since_last >= self.min_interval:
                try:
                    report = await self.consolidate()
                    if report.get("patterns"):
                        logger.info("consolidation: %s", json.dumps(report["patterns"])[:200])
                except Exception:  # noqa: BLE001
                    logger.exception("consolidation pass failed; continuing")
                finally:
                    self.last_consolidation = time.time()
                    self.last_activity = time.time()  # avoid immediate re-trigger

    # ── the consolidation pass itself ──────────────────────────────────────
    async def consolidate(self) -> dict[str, Any]:
        """One pass: WM review → pattern extraction → EM/SM writes."""
        started = time.time()
        # 1. Harvest: salient WM items + recent episodes as raw material.
        wm_items = self.wm.candidates(min_salience=0.6)
        if not wm_items:
            wm_items = list(self.wm.snapshot(limit=8))
        recent_episodes = self.em.recent(limit=10)
        if not wm_items and not recent_episodes:
            return {"patterns": 0, "reason": "nothing salient"}

        # 2. Extract patterns (LLM-assisted when available, heuristic fallback).
        llm = self.llm
        if llm is None and self._services_getter is not None:
            try:
                llm = self._services_getter().get("llm")
            except Exception:  # noqa: BLE001
                llm = None
        patterns: list[dict[str, Any]] = []
        if llm is not None:
            # Pass the resolved provider explicitly — self.llm may still be
            # None because the memory plugin starts before the LLM plugin.
            patterns = await self._extract_with_llm(wm_items, recent_episodes, llm)
        if not patterns:
            patterns = self._extract_heuristic(wm_items, recent_episodes)

        # 3. Consolidate into Semantic Memory (durable abstractions).
        for p in patterns:
            doc_id = f"pattern:{uuid.uuid4().hex[:12]}"
            self.sm.upsert(
                doc_id,
                p["summary"],
                metadata={"kind": p.get("kind", "fact"), "origin": "consolidation", "ts": time.time()},
                salience=float(p.get("salience", 0.5)),
            )
            # 4. Cause-effect edges also go to the episodic graph.
            if p.get("kind") == "cause-effect":
                self.em.append(f"[pattern] {p['summary'][:200]}", "consolidated", salience=0.7)

        self.consolidations += 1
        self.patterns_written += len(patterns)
        report = {
            "patterns": len(patterns),
            "wm_items_reviewed": len(wm_items),
            "episodes_reviewed": len(recent_episodes),
            "elapsed_ms": int((time.time() - started) * 1000),
        }
        self.bus.publish("memory:consolidated", report, source="consolidator")
        return report

    # ── extraction backends ────────────────────────────────────────────────
    async def _extract_with_llm(
        self, wm_items: list[Any], episodes: list[dict[str, Any]], llm: Any
    ) -> list[dict[str, Any]]:
        """One bounded structured call; local models stay terse via schema."""
        material = "\n".join(
            [f"WM: {getattr(i, 'text', str(i))[:200]}" for i in wm_items[:10]]
            + [f"EP: action={e['action'][:120]} → outcome={e['outcome'][:120]} success={e['success']}" for e in episodes[:10]]
        )
        out = await llm.complete(
            [{"role": "user", "content": (
                "Extract reusable knowledge from this agent trace. Only high-value, "
                "durable patterns (skills, stable facts, cause-effect rules). Max 5:\n" + material
            )}],
            schema=PATTERNS_SCHEMA,
            temperature=0.1,
            max_tokens=400,
        )
        content = out.get("content") if isinstance(out, dict) else None
        if isinstance(content, dict):
            return list(content.get("patterns", []))[:5]
        return []

    def _extract_heuristic(self, wm_items: list[Any], episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Zero-LLM pattern mining: duplicate clustering + action signatures."""
        patterns: list[dict[str, Any]] = []

        # (a) Cluster near-duplicate WM items → recurring themes.
        texts = [getattr(i, "text", str(i)) for i in wm_items]
        embs = [SemanticMemory.embed(t) for t in texts]
        used = [False] * len(texts)
        for i in range(len(texts)):
            if used[i]:
                continue
            cluster = [i]
            used[i] = True
            for j in range(i + 1, len(texts)):
                if not used[j] and cosine(embs[i], embs[j]) >= 0.80:
                    cluster.append(j)
                    used[j] = True
            if len(cluster) >= 2:  # recurrence ⇒ salience
                patterns.append({
                    "summary": f"Recurring theme ({len(cluster)}x): {texts[cluster[0]][:180]}",
                    "kind": "fact",
                    "salience": min(1.0, 0.5 + 0.1 * len(cluster)),
                })

        # (b) Repeated action prefixes with consistent outcomes → cause-effect rules.
        by_prefix: dict[str, list[bool]] = {}
        for e in episodes:
            prefix = e["action"].split("(")[0][:60]
            by_prefix.setdefault(prefix, []).append(bool(e["success"]))
        for prefix, outcomes in by_prefix.items():
            if len(outcomes) >= 2:
                rate = sum(outcomes) / len(outcomes)
                verdict = "succeeds" if rate >= 0.5 else "fails"
                patterns.append({
                    "summary": f"Action '{prefix}' historically {verdict} ({rate:.0%} of {len(outcomes)} attempts).",
                    "kind": "cause-effect",
                    "salience": min(1.0, 0.4 + abs(rate - 0.5)),
                })
        return patterns[:8]
