"""Memory plugin glue: wires the dual-process stores into the kernel.

Responsibilities:
    * expose the stores behind the ``MemoryPlugin`` Protocol (services['memory'])
    * dogfood dynamic tool registration via ``tool:register`` bus events
      (memory_recall / memory_remember / memory_stats / memory_trace)
    * auto-record experiences: every ``tool:executed`` and ``task:completed``
      event becomes an episodic memory row + cause-effect graph edge
    * own the background ConsolidationLoop lifecycle
    * auto-extract personal facts (name, preferences) from conversations
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from python_core.core.config import EchoConfig
from python_core.core.event_bus import EventBus, Subscription
from python_core.core.interfaces import KernelContext, ToolSpec

from .consolidator import ConsolidationLoop
from .storage import EpisodicMemory, SemanticMemory, WorkingMemory

logger = logging.getLogger("echo.memory")

# ── personal-fact extraction patterns ──────────────────────────────────────
_NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", re.IGNORECASE),
    re.compile(r"\bi['’]m\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", re.IGNORECASE),
    re.compile(r"\bcall me\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", re.IGNORECASE),
    re.compile(r"\bi am\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", re.IGNORECASE),
]
_PREF_PATTERNS = [
    re.compile(r"\bi\s+(prefer|like|love|don.?t like|dislike|hate)\s+(.+)", re.IGNORECASE),
    re.compile(r"\bmy\s+(.+?)\s+is\s+(.+)", re.IGNORECASE),
]


class CognitiveMemoryPlugin:
    """Implements MemoryPlugin. Registered under services['memory'] so the
    orchestrator, gateways, and future plugins can all use one memory system
    without importing this module."""

    name = "memory"
    version = "0.1.0"
    capabilities = ["memory:working", "memory:episodic", "memory:semantic", "consolidation"]

    def __init__(self, config: EchoConfig | None = None) -> None:
        self.config = config or EchoConfig.load()
        self.wm = WorkingMemory(capacity=self.config.wm_capacity)
        self.em: EpisodicMemory | None = None
        self.sm: SemanticMemory | None = None
        self.consolidator: ConsolidationLoop | None = None
        self._ctx: KernelContext | None = None
        self._bus: EventBus | None = None
        self._subs: list[Subscription] = []

    # ── plugin lifecycle ───────────────────────────────────────────────────
    async def start(self, ctx: KernelContext) -> None:
        self._ctx = ctx
        self._bus = ctx.bus
        self.config = ctx.config  # adopt the kernel's live config
        self.em = EpisodicMemory(ctx.config.sqlite_path)
        self.sm = SemanticMemory(ctx.config.vector_dir)
        self.consolidator = ConsolidationLoop(
            ctx.bus, self.wm, self.em, self.sm,
            idle_seconds=ctx.config.consolidation_idle_seconds,
            services_getter=lambda: ctx.services,
        )
        ctx.services["memory"] = self

        # Auto-episodicity: observe the system and remember what happened.
        self._subs.append(
            ctx.bus.subscribe("tool:executed", self._on_tool_executed, label="memory.tool_executed")
        )
        self._subs.append(
            ctx.bus.subscribe("task:completed", self._on_task_completed, label="memory.task_completed")
        )

        # Dynamic tool registration THROUGH the event bus (MODULE 3 Phase 3
        # mechanism, reused by a built-in plugin — one code path for all).
        await ctx.bus.emit("tool:register", {
            "spec": ToolSpec(
                name="memory_recall",
                description="Search durable memory (semantic + episodic) for relevant knowledge.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "k": {"type": "integer", "minimum": 1, "maximum": 20}},
                    "required": ["query"],
                },
                source="plugin:memory",
                tags=["memory"],
            ),
            "handler": self._tool_recall,
        })
        await ctx.bus.emit("tool:register", {
            "spec": ToolSpec(
                name="memory_remember",
                description="Store a durable fact/skill into semantic memory with a salience score.",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}, "salience": {"type": "number"}},
                    "required": ["text"],
                },
                source="plugin:memory",
                tags=["memory"],
            ),
            "handler": self._tool_remember,
        })
        await ctx.bus.emit("tool:register", {
            "spec": ToolSpec(
                name="memory_stats",
                description="Report memory system statistics (WM/EM/SM, consolidation counts).",
                parameters={"type": "object", "properties": {}},
                source="plugin:memory",
                tags=["memory", "diagnostics"],
            ),
            "handler": self._tool_stats,
        })

        self.consolidator.start()
        ctx.logger.info("memory plugin online: %s", json.dumps(self.stats()))

    async def stop(self) -> None:
        if self.consolidator is not None:
            await self.consolidator.stop()
        for sub in self._subs:
            if self._bus is not None:
                self._bus.unsubscribe(sub)
        self._subs.clear()
        if self.em is not None:
            self.em.close()

    # ── MemoryPlugin Protocol ──────────────────────────────────────────────
    def write_working(self, content: str, *, role: str = "system", salience: float = 0.5) -> dict:
        self.wm.push(content, role=role, salience=salience)
        return {"wm_size": len(self.wm)}

    def working_snapshot(self, limit: int = 6) -> list[str]:
        return self.wm.snapshot(limit=limit)

    def append_episode(self, action: str, outcome: str, *, success: bool = True, metadata: dict | None = None) -> str:
        if self.em is None:
            return ""
        episode_id = self.em.append(action, outcome, success=success, salience=0.6, metadata=metadata)
        return str(episode_id)

    async def consolidate(self) -> dict:
        if self.consolidator is None:
            return {"patterns": 0}
        return await self.consolidator.consolidate()

    def recall(self, query: str, *, k: int = 5) -> list[dict]:
        """Hybrid recall: semantic hits first, then recent episodic context."""
        results: list[dict] = []
        if self.sm is not None:
            for hit in self.sm.recall(query, k=k):
                results.append({"text": hit["text"], "score": hit["score"], "kind": hit["metadata"].get("kind", "fact")})
        if self.em is not None:
            for ep in self.em.recent(limit=3):
                results.append({
                    "text": f"experience: {ep['action']} → {ep['outcome'][:120]} ({'ok' if ep['success'] else 'fail'})",
                    "score": 0.3,
                    "kind": "episode",
                })
        return results

    def trace_back(self, outcome: str) -> list[str]:
        return self.em.trace_back(outcome) if self.em else []

    def stats(self) -> dict[str, Any]:
        consolidator_stats: dict[str, Any] = {}
        if self.consolidator is not None:
            consolidator_stats = {
                "consolidations": self.consolidator.consolidations,
                "patterns_written": self.consolidator.patterns_written,
            }
        return {
            "wm_items": len(self.wm),
            "episodic": self.em.stats() if self.em else {},
            "semantic": self.sm.stats() if self.sm else {},
            **consolidator_stats,
        }

    # ── event observers (auto-episodicity) ─────────────────────────────────
    def _on_tool_executed(self, event: Any) -> None:
        payload = getattr(event, "payload", event) or {}
        if self.em is None:
            return
        self.em.append(
            action=f"{payload.get('tool', 'unknown')}",
            outcome=json.dumps({"ok": payload.get("ok")}, default=str)[:120],
            success=bool(payload.get("ok")),
            salience=0.4,
            metadata={"elapsed_ms": payload.get("elapsed_ms")},
        )

    def _on_task_completed(self, event: Any) -> None:
        payload = getattr(event, "payload", event) or {}
        response = str(payload.get("response", ""))[:400]
        if response:
            salience = 0.8 if not payload.get("cached") else 0.5
            self.wm.push(f"task: {str(payload.get('task', ''))[:160]} → {response[:160]}", role="agent", salience=salience)

        # Auto-extract personal facts from the user's task text.
        task_text = str(payload.get("task", ""))
        if task_text and self._ctx is not None:
            self._extract_personal_facts(task_text)

    def _extract_personal_facts(self, text: str) -> None:
        """Scan *text* for personal facts (name, preferences) and store them
        durably in config + semantic memory so the agent remembers them."""
        if self._ctx is None or self.config is None:
            return
        config = self._ctx.config

        # --- name extraction ---
        if not config.user_name:
            for pattern in _NAME_PATTERNS:
                m = pattern.search(text)
                if m:
                    name = m.group(1).strip()
                    config.user_name = name
                    self._remember_fact(f"User's name is {name}", salience=0.9)
                    config.save_user_profile()
                    logger.info("memory: learned user name = %s", name)
                    break

        # --- preference extraction ---
        for pattern in _PREF_PATTERNS:
            for m in pattern.finditer(text):
                if m.lastindex >= 2:
                    key = m.group(1).strip().lower()
                    value = m.group(2).strip()
                    # Skip generic matches
                    if len(value) < 2 or len(value) > 200:
                        continue
                    pref_key = key.replace(" ", "_")
                    config.user_preferences[pref_key] = value
                    self._remember_fact(f"User preference: {key} = {value}", salience=0.7)
                    logger.info("memory: learned preference %s = %s", key, value)
        if config.user_preferences:
            config.save_user_profile()

    # ── tool handlers ──────────────────────────────────────────────────────
    def _tool_recall(self, args: dict[str, Any]) -> dict[str, Any]:
        hits = self.recall(str(args.get("query", "")), k=int(args.get("k", 5)))
        return {"hits": hits}

    def _tool_remember(self, args: dict[str, Any]) -> dict[str, Any]:
        if self.sm is None:
            return {"ok": False, "error": "semantic memory unavailable"}
        import uuid

        text = str(args.get("text", "")).strip()
        if not text:
            return {"ok": False, "error": "empty text"}
        doc_id = f"fact:{uuid.uuid4().hex[:12]}"
        self.sm.upsert(doc_id, text, metadata={"origin": "tool"}, salience=float(args.get("salience", 0.7)))
        return {"ok": True, "id": doc_id}

    def _tool_stats(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.stats()


def create_plugin() -> CognitiveMemoryPlugin:
    """Discovery-contract entry point."""
    return CognitiveMemoryPlugin()
