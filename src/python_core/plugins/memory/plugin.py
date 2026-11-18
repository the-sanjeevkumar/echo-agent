"""Memory plugin glue: wires the dual-process stores into the kernel.

Responsibilities:
    * expose the stores behind the ``MemoryPlugin`` Protocol (services['memory'])
    * own the durable owner profile (ProfileStore): <data_dir>/profiles/owner.json
      plus one isolated profile per non-owner sender
    * dogfood dynamic tool registration via ``tool:register`` bus events
      (memory_recall / memory_remember / memory_stats / memory_trace)
    * auto-record experiences: every ``tool:executed`` and ``task:completed``
      event becomes an episodic memory row + cause-effect graph edge
    * own the background ConsolidationLoop lifecycle
    * auto-extract personal facts (name, preferences, facts) from conversations
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
from .profile import _NAME_STOPWORDS, ProfileStore
from .storage import EpisodicMemory, SemanticMemory, WorkingMemory

logger = logging.getLogger("echo.memory")

# ── fast-path name / preference regexes (zero LLM cost) ────────────────────
_NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)", re.IGNORECASE),
    re.compile(r"\bi['’]m\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)", re.IGNORECASE),
    re.compile(r"\bcall me\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)", re.IGNORECASE),
    re.compile(r"\bi am\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)", re.IGNORECASE),
    re.compile(r"\bmyself\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)", re.IGNORECASE),
    re.compile(r"\bthis is\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)", re.IGNORECASE),
    re.compile(r"\b([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s+here\b", re.IGNORECASE),
]
_BARE_NAME = re.compile(r"^[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?$")
_PREF_PATTERNS = [
    re.compile(r"\bi\s+(prefer|like|love|don.?t like|dislike|hate)\s+(.+)", re.IGNORECASE),
    re.compile(r"\bmy\s+(.+?)\s+is\s+(.+)", re.IGNORECASE),
]
_EPHEMERA_KEYS = {"question", "query", "task", "request", "issue", "problem", "error", "answer", "name"}

# Structured schema for the per-turn profile extraction call (temperature 0).
EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {
            "type": ["string", "null"],
            "description": "the user's name, only when explicitly stated",
        },
        "preferences": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "key": {"type": "string", "description": "preference key, snake_case"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
        },
        "facts": {
            "type": "array",
            "items": {"type": "string", "description": "durable identity fact"},
        },
    },
    "required": ["name", "preferences", "facts"],
}


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
        self.profiles: ProfileStore | None = None
        self.consolidator: ConsolidationLoop | None = None
        self._ctx: KernelContext | None = None
        self._bus: EventBus | None = None
        self._subs: list[Subscription] = []
        # assistant's most recent response per session, so a bare-name reply
        # ("Sanjeev") can be matched to the turn that asked for the name.
        self._last_assistant: dict[str, str] = {}

    # ── plugin lifecycle ───────────────────────────────────────────────────
    async def start(self, ctx: KernelContext) -> None:
        self._ctx = ctx
        self._bus = ctx.bus
        self.config = ctx.config  # adopt the kernel's live config
        self.em = EpisodicMemory(ctx.config.sqlite_path)
        self.sm = SemanticMemory(ctx.config.vector_dir)
        # Durable owner profile: loaded at plugin start, atomic writes.
        self.profiles = ProfileStore(
            ctx.config.data_dir, owner_user_ids=list(ctx.config.owner_user_ids)
        )
        ctx.services["memory"] = self
        ctx.services["profile_store"] = self.profiles

        self.consolidator = ConsolidationLoop(
            ctx.bus, self.wm, self.em, self.sm,
            idle_seconds=ctx.config.consolidation_idle_seconds,
            services_getter=lambda: ctx.services,
        )

        # Auto-episodicity: observe the system and remember what happened.
        self._subs.append(
            ctx.bus.subscribe("tool:executed", self._on_tool_executed, label="memory.tool_executed")
        )
        self._subs.append(
            ctx.bus.subscribe("task:completed", self._on_task_completed, label="memory.task_completed")
        )

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

    async def _on_task_completed(self, event: Any) -> None:
        payload = getattr(event, "payload", event) or {}
        response = str(payload.get("response", ""))[:400]
        if response:
            salience = 0.8 if not payload.get("cached") else 0.5
            self.wm.push(f"task: {str(payload.get('task', ''))[:160]} → {response[:160]}", role="agent", salience=salience)

        # Auto-extract personal facts from the user's turn and durably update
        # the sender's profile (owner profile for the owner, isolated profile
        # for anyone else). Failures here never reach the user.
        task_text = str(payload.get("task", ""))
        user_id = str(payload.get("user_id") or "")
        source = str(payload.get("source") or "")
        session_id = str(payload.get("session_id") or "main")
        assistant_text = self._last_assistant.get(session_id, "")
        if task_text and self.profiles is not None:
            try:
                await self._update_profile_from_turn(user_id, source, task_text, assistant_text)
            except Exception:  # memory extraction never breaks the loop
                logger.exception("profile extraction failed")
        # Remember this turn's response so the NEXT user turn can rescue a bare name.
        self._last_assistant[session_id] = response

    async def _update_profile_from_turn(
        self, user_id: str, source: str, task_text: str, assistant_text: str
    ) -> None:
        """One bounded structured LLM call (primary) plus zero-cost regex
        fast-path; last-write-wins merge into ProfileStore."""
        key = self.profiles.resolve_key(user_id, source)

        # 1. Fast-path regexes (no LLM cost).
        delta: dict[str, Any] = self._extract_fast(task_text, assistant_text)

        # 2. Primary: one structured LLM call per user turn.
        llm = self._ctx.services.get("llm") if self._ctx is not None else None
        if llm is not None:
            try:
                out = await llm.complete(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Extract durable profile facts about the user from their latest message. "
                                "Return ONLY JSON matching the schema: {\"name\": string|null, "
                                "\"preferences\": [{\"key\": string, \"value\": string}], \"facts\": [string]}. "
                                "Rules: set name only when the user states their name explicitly (including a "
                                "bare name when the previous assistant turn asked for it); preferences are "
                                "durable likes/dislikes only — never ephemera such as \"my question is ...\"; "
                                "facts are other stable identity facts. Omit unknown fields."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Previous assistant message: {assistant_text[:400]}\n"
                                f"User message: {task_text[:800]}"
                            ),
                        },
                    ],
                    schema=EXTRACTION_SCHEMA,
                    temperature=0.0,
                    max_tokens=250,
                )
                content = out.get("content") if isinstance(out, dict) else None
                if isinstance(content, dict):
                    llm_name = content.get("name")
                    llm_prefs = content.get("preferences") or []
                    llm_facts = content.get("facts") or []
                    llm_delta: dict[str, Any] = {}
                    if isinstance(llm_name, str) and llm_name.strip():
                        llm_delta["name"] = llm_name.strip()
                    if isinstance(llm_prefs, list):
                        prefs = {}
                        for item in llm_prefs:
                            if isinstance(item, dict) and isinstance(item.get("key"), str) and isinstance(item.get("value"), str):
                                prefs[item["key"].strip()] = item["value"].strip()
                        if prefs:
                            llm_delta["preferences"] = prefs
                    if isinstance(llm_facts, list):
                        llm_delta["facts"] = [str(f) for f in llm_facts if str(f).strip()]
                    # Fast path has final say on what it explicitly found.
                    delta = {**llm_delta, **delta}
            except Exception:  # noqa: BLE001 — regex fallback keeps working offline
                logger.debug("LLM profile extraction unavailable; using fast path only")

        if not delta:
            return

        profile = self.profiles.update(
            key,
            name=delta.get("name"),
            preferences=delta.get("preferences") or None,
            facts=delta.get("facts") or None,
        )
        # Mirror into config so legacy readers (config.user_name / config.toml)
        # stay in sync with the durable profile store.
        config = self._ctx.config if self._ctx is not None else self.config
        if config is not None:
            config.user_name = profile.name
            config.user_preferences = dict(profile.preferences)
            try:
                config.save_user_profile()
            except Exception:  # config.toml mirror is best-effort
                logger.debug("config.toml profile mirror write failed", exc_info=True)
        logger.info(
            "memory: profile updated for %s (name=%s)", key, profile.name or "(none)"
        )

    @staticmethod
    def _extract_fast(task_text: str, assistant_text: str) -> dict[str, Any]:
        """Zero-LLM fact capture: extended name patterns + bare-name rescue
        when the assistant just asked for the name."""
        delta: dict[str, Any] = {}
        asked_for_name = bool(re.search(r"\bname\b|\bcall you\b", assistant_text or "", re.IGNORECASE))

        for pattern in _NAME_PATTERNS:
            m = pattern.search(task_text)
            if m:
                delta["name"] = m.group(1).strip().title()
                break

        # Bare-name rescue: "Sanjeev" alone, right after we asked for the name.
        if "name" not in delta and asked_for_name:
            m = _BARE_NAME.match(task_text.strip())
            if m and m.group(0).lower() not in _NAME_STOPWORDS:
                delta["name"] = m.group(0).strip().title()

        # Preferences (skip ephemera).
        for pattern in _PREF_PATTERNS:
            for m in pattern.finditer(task_text):
                if m.lastindex < 2:
                    continue
                key = m.group(1).strip().lower()
                value = m.group(2).strip()
                if len(value) < 2 or len(value) > 200:
                    continue
                pref_key = key.replace(" ", "_")
                if pref_key in _EPHEMERA_KEYS:
                    continue
                # "my <question/task/...> is ..." is not a preference.
                if any(ephem in pref_key for ephem in ("question", "query", "task", "error", "issue")):
                    continue
                delta.setdefault("preferences", {})[pref_key] = value

        return delta

    # ── public profile API (used by the orchestrator and the CLI) ──────────
    def profile_block(self, user_id: str, source: str | None = None) -> str:
        """USER PROFILE block for *user_id*, always safe to inject into a
        system prompt (<=500 tokens after trimming)."""
        if self.profiles is None:
            return ""
        key = self.profiles.resolve_key(user_id, source)
        return self.profiles.block(key, budget_tokens=500)

    def update_profile(
        self,
        user_id: str,
        source: str | None = None,
        *,
        name: str | None = None,
        preferences: dict[str, str] | None = None,
        facts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Explicit profile update (tools/CLI). Last-write-wins."""
        if self.profiles is None:
            return {"ok": False, "error": "profile store unavailable"}
        key = self.profiles.resolve_key(user_id, source)
        profile = self.profiles.update(
            key, name=name, preferences=preferences, facts=facts
        )
        return {"ok": True, "user_id": key, "name": profile.name, "is_owner": self.profiles.is_owner(key)}

    def clear_profile(self, user_id: str, source: str | None = None, confirmed: bool = False) -> dict[str, Any]:
        if self.profiles is None:
            return {"ok": False, "error": "profile store unavailable"}
        key = self.profiles.resolve_key(user_id, source)
        if not self.profiles.is_owner(key):
            return {"ok": False, "error": "profile management is owner-only"}
        if not confirmed:
            self.profiles.request_clear(key)
            return {"ok": True, "pending": True, "ttl_seconds": self.profiles.confirm_ttl}
        ok = self.profiles.clear(key, confirmed=True)
        return {"ok": ok}

    # ── tool handlers ──────────────────────────────────────────────────────

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
