"""The main agent loop — LLM routing, token pruning, semantic caching.

Design: a compact ReAct-style loop where every LLM round-trip is a strict JSON
object (MODULE 4: structured outputs). The loop:

    1. Semantic-cache lookup keyed on (prompt, active tool schemas).
    2. Context assembly: Working Memory + semantic recall, pruned to budget.
    3. LLM emits either {"action":"respond"} (done) or {"action":"tool"}.
    4. Tool execution result is appended as a compact OBSERVATION and the loop
       continues (bounded by ``max_steps``).
    5. The whole interaction is written to Episodic Memory for consolidation.

Token efficiency:
    * tiktoken-exact counting (char-estimate fallback when tiktoken is absent).
    * Old turns are folded into a single "MEMORY SUMMARY" block when the
      window exceeds ``prune_target_ratio`` of the budget.
    * Tool observations are truncated; file edits request diffs only (the
      tools themselves are delta-oriented).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core.config import EchoConfig
from .core.event_bus import EventBus, Subscription
from .core.interfaces import KernelContext, LLMProvider
from .core.tool_registry import ToolRegistry
from .core.utils import cosine, hash_embedding
from .core.utils import extract_json as _extract_json
from .plugins.llm.openai_compat import LLMProviderError

ProgressCallback = Callable[[dict[str, Any]], None]

logger = logging.getLogger("echo.orchestrator")

# Strict action schema — forces terse, structured, filler-free responses.
ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "thought": {"type": "string", "description": "one short sentence max"},
        "action": {"type": "string", "enum": ["tool", "respond"]},
        "tool": {"type": ["string", "null"], "description": "tool name when action=tool"},
        "args": {"type": ["object", "null"], "description": "tool arguments object"},
        "response": {"type": ["string", "null"], "description": "final answer when action=respond"},
    },
    "required": ["thought", "action", "tool", "args", "response"],
}

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string", "description": "dense summary, <=120 tokens"}
    },
    "required": ["summary"],
}

SYSTEM_TEMPLATE = (
    "You are echo-agent, a local-first autonomous agent for coding, research, and analysis.\n"
    "Respond ONLY with JSON matching the given schema. Never narrate.\n"
    "Strategy: use tools for anything factual or mutating; prefer unified diffs over "
    "full files; keep 'thought' to one sentence.\n"
    "AVAILABLE TOOLS (JSON schema):\n{tools}\n"
    "{user_profile}{memory_block}RECENT CONTEXT:\n{working_memory}"
    "\n\nMEMORY GUIDANCE:\n"
    "- Use memory_remember to store important facts about the user (name, preferences, goals).\n"
    "- Use memory_recall to recall the user's name, preferences, and past context before answering.\n"
    "- Adapt your responses to the user's known preferences and name.\n"
    "- If you do not know the user's name, ask \"What should I call you?\" and remember it with memory_remember.\n"
)


# ── token accounting (MODULE 4) ──────────────────────────────────────────────
class TokenBudget:
    """tiktoken-exact when available; deterministic char/4 estimator otherwise."""

    def __init__(self) -> None:
        self._encoding = None
        try:
            import tiktoken  # type: ignore

            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001 — pure-local fallback keeps the system running
            logger.info("tiktoken unavailable; using char/4 token estimator")

    def count(self, text: str) -> int:
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return max(1, len(text) // 4)

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        return sum(self.count(m.get("content", "")) + 4 for m in messages)  # +4 per-message overhead


# ── deterministic local embedding lives in core.utils (hash_embedding) ──────
# (imported above; kept here so SemanticCache reads naturally)


@dataclass(slots=True)
class CacheEntry:
    key: str
    prompt: str
    embedding: list[float]
    response: dict[str, Any]
    ts: float


class SemanticCache:
    """Local, bounded LRU-ish cache: exact-hash first, cosine second."""

    def __init__(self, config: EchoConfig) -> None:
        self._entries: dict[str, CacheEntry] = {}
        self._order: list[str] = []
        self._config = config

    @staticmethod
    def _exact_key(prompt: str, tool_names: list[str], scope: str = "") -> str:
        normalized = " ".join(prompt.lower().split())
        blob = normalized + "‖" + ",".join(sorted(tool_names)) + "‖scope:" + scope
        return hashlib.sha1(blob.encode()).hexdigest()

    def lookup(self, prompt: str, tool_names: list[str], scope: str = "") -> dict[str, Any] | None:
        if not self._config.enable_semantic_cache:
            return None
        key = self._exact_key(prompt, tool_names, scope)
        entry = self._entries.get(key)
        if entry is not None:  # exact hit
            return entry.response
        # Semantic pass (only when few tools changed the key space). Sessions are
        # isolated: a cached answer from another chat must never leak into this one.
        emb = hash_embedding(prompt)
        best, best_sim = None, self._config.cache_similarity_threshold
        for e in self._entries.values():
            if not e.key.endswith("‖scope:" + scope):
                continue
            sim = cosine(emb, e.embedding)
            if sim >= best_sim:
                best, best_sim = e, sim
        if best is not None:
            logger.info("semantic cache hit (sim=%.3f)", best_sim)
            return best.response
        return None

    def store(self, prompt: str, tool_names: list[str], response: dict[str, Any], scope: str = "") -> None:
        if not self._config.enable_semantic_cache:
            return
        key = self._exact_key(prompt, tool_names, scope)
        self._entries[key] = CacheEntry(
            key=key, prompt=prompt, embedding=hash_embedding(prompt), response=response, ts=time.time()
        )
        self._order.append(key)
        if len(self._order) > 256:  # bounded memory footprint
            old = self._order.pop(0)
            self._entries.pop(old, None)


# ── the orchestrator itself (satisfies the Plugin Protocol) ─────────────────
@dataclass
class Orchestrator:
    """Core agent loop. Registered like any plugin so gateways, tests, and
    external entry points all drive it through the same event interface."""

    bus: EventBus
    config: EchoConfig
    tools: ToolRegistry
    provider: LLMProvider
    max_steps: int = 5
    capabilities: list[str] = field(default_factory=lambda: ["agent-loop", "structured-output"])
    name: str = "orchestrator"
    version: str = "0.1.0"

    def __post_init__(self) -> None:
        self.budget = TokenBudget()
        self.cache = SemanticCache(self.config)
        self._subs: list[Subscription] = []
        self.conversation: list[dict[str, str]] = []  # default ("main") session
        self._sessions: OrderedDict[str, list[dict[str, str]]] = OrderedDict({"main": self.conversation})
        self._session_locks: dict[str, Any] = {}
        self._sessions_path = Path(self.config.data_dir) / "sessions.json"
        self._load_sessions()

    # ── plugin lifecycle ───────────────────────────────────────────────────
    def start(self, ctx: KernelContext) -> None:
        self._ctx = ctx
        self._subs.append(
            self.bus.subscribe("task:incoming", self._on_incoming, label="orchestrator.incoming")
        )
        self._subs.append(
            self.bus.subscribe("session:reset", self._on_session_reset, label="orchestrator.session-reset")
        )
        self._subs.append(
            self.bus.subscribe("session:status", self._on_session_status, label="orchestrator.session-status")
        )
        ctx.logger.info("orchestrator online (model=%s)", self.config.llm_model)

    def stop(self) -> None:
        for sub in self._subs:
            self.bus.unsubscribe(sub)
        self._subs.clear()
        self._save_sessions()

    async def _on_incoming(self, event: Any) -> Any:
        payload = getattr(event, "payload", event) or {}
        return await self.handle_task(
            payload.get("text", ""),
            source=payload.get("source", "gateway"),
            reply_channel=payload.get("reply_channel"),
            platform=payload.get("platform"),
            session_id=payload.get("session_id"),
            user_id=payload.get("user_id"),
        )

    # ── sessions (per DM / group / topic) ──────────────────────────────────
    @staticmethod
    def _normalize_session(session_id: str | None) -> str:
        """Keep session keys filesystem/log friendly and bounded."""
        if not session_id:
            return "main"
        cleaned = "".join(
            ch for ch in str(session_id).strip() if ch.isalnum() or ch in ":_-.@"
        )
        return cleaned[:200] or "main"

    def _session(self, session_id: str) -> list[dict[str, str]]:
        sid = self._normalize_session(session_id)
        if sid == "main":
            return self.conversation
        if sid not in self._sessions:
            self._sessions[sid] = []
            self._evict_sessions(keep=sid)
        self._sessions.move_to_end(sid)
        return self._sessions[sid]

    def _evict_sessions(self, *, keep: str | None = None) -> None:
        limit = max(1, int(self.config.max_sessions))
        while len(self._sessions) > limit:
            victim = next(
                (
                    key
                    for key in self._sessions
                    if key != "main" and key != keep
                ),
                None,
            )
            if victim is None:
                break
            self._sessions.pop(victim, None)
            self._session_locks.pop(victim, None)

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        sid = self._normalize_session(session_id)
        lock = self._session_locks.get(sid)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[sid] = lock
        return lock

    def _append_turn(self, session_id: str, user_text: str, response: str) -> None:
        sid = self._normalize_session(session_id)
        history = self._session(sid)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": response})
        limit = max(2, int(self.config.session_history_limit))
        if len(history) > limit:
            del history[: len(history) - limit]
        if sid in self._sessions:
            self._sessions.move_to_end(sid)
        self._save_sessions()

    def _load_sessions(self) -> None:
        if not bool(getattr(self.config, "session_persistence", True)):
            return
        try:
            raw = json.loads(self._sessions_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        limit = max(2, int(self.config.session_history_limit))
        loaded: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        for raw_sid, raw_messages in raw.items():
            if not isinstance(raw_messages, list):
                continue
            messages: list[dict[str, str]] = []
            for message in raw_messages:
                if (
                    isinstance(message, dict)
                    and message.get("role") in {"user", "assistant", "system"}
                    and isinstance(message.get("content"), str)
                ):
                    messages.append({"role": message["role"], "content": message["content"]})
            if messages:
                loaded[self._normalize_session(str(raw_sid))] = messages[-limit:]

        main = loaded.pop("main", [])
        if len(main) > limit:
            main = main[-limit:]
        self.conversation = main
        self._sessions = OrderedDict({"main": self.conversation})
        for sid, messages in loaded.items():
            self._sessions[sid] = messages
        self._evict_sessions()

    def _save_sessions(self) -> None:
        if not bool(getattr(self.config, "session_persistence", True)):
            return
        limit = max(2, int(self.config.session_history_limit))
        payload = {
            sid: messages[-limit:]
            for sid, messages in self._sessions.items()
            if messages
        }
        tmp_name: str | None = None
        try:
            self._sessions_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=".sessions-", dir=str(self._sessions_path.parent)
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            os.replace(tmp_name, self._sessions_path)
            tmp_name = None
        except OSError:
            logger.debug("could not persist sessions", exc_info=True)
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    async def _on_session_reset(self, event: Any) -> dict[str, Any]:
        payload = event.payload if hasattr(event, "payload") else event
        sid = self._normalize_session((payload or {}).get("session_id"))
        async with self._lock_for(sid):
            if sid == "main":
                self.conversation = []
                self._sessions["main"] = self.conversation
            else:
                self._sessions.pop(sid, None)
            self._save_sessions()
        logger.info("session reset: %s", sid)
        return {"session_id": sid, "reset": True}

    async def _on_session_status(self, event: Any) -> dict[str, Any]:
        payload = event.payload if hasattr(event, "payload") else event
        sid = self._normalize_session((payload or {}).get("session_id"))
        history = self.conversation if sid == "main" else self._sessions.get(sid, [])
        return {
            "session_id": sid,
            "messages": len(history),
            "turns": sum(1 for message in history if message.get("role") == "user"),
            "sessions": len(self._sessions),
            "model": self.config.llm_model,
        }

    # ── memory access (loose coupling: whatever service registered itself) ──
    @property
    def memory(self) -> Any | None:
        return getattr(self, "_ctx", None) and self._ctx.service("memory")

    # ── main loop ──────────────────────────────────────────────────────────
    async def handle_task(
        self,
        text: str,
        *,
        source: str = "cli",
        reply_channel: str | None = None,
        platform: str | None = None,
        on_event: ProgressCallback | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute one user task end-to-end and return a result envelope.

        ``session_id`` isolates chat conversations (gateways pass a per-DM /
        per-channel / per-topic key). Tasks in the same session are serialized,
        so two messages from one chat cannot interleave their history.

        ``reply_channel`` is the platform-prefixed chat target (e.g.
        ``discord:<channel_id>`` or ``telegram:<chat_id>``) that the response
        should be delivered back to. ``platform`` ("telegram" | "discord")
        is used as an additional routing filter so a job triggered from one
        platform is never echoed on the other.

        ``user_id`` is the stable platform sender id (CLI ⇒ "owner"). It is
        threaded through to memory so the owner profile is shared across
        CLI/Telegram/Discord while every other sender stays isolated.

        ``on_event`` is an optional fire-and-forget hook for UIs (the CLI uses
        it to animate a spinner). It receives tiny dicts such as
        ``{"type": "thinking"}`` / ``{"type": "tool", "tool": "fs_read"}`` and
        must never be able to break the agent loop.
        """
        text = (text or "").strip()
        if not text:
            return {"response": "", "steps": [], "cached": False, "error": "empty task"}
        sid = self._normalize_session(session_id)
        async with self._lock_for(sid):
            return await self._execute_task(
                text,
                source=source,
                reply_channel=reply_channel,
                platform=platform,
                on_event=on_event,
                session_id=sid,
                user_id=user_id,
            )

    async def _execute_task(
        self,
        text: str,
        *,
        source: str,
        reply_channel: str | None,
        platform: str | None = None,
        on_event: ProgressCallback | None = None,
        session_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        self.bus.publish("task:received", {"text": text, "source": source}, source="orchestrator")
        started = time.monotonic()

        # Profile management command (owner-only, confirmation-gated).
        forget_parts = text.lstrip("/").strip().lower().split()
        if forget_parts and forget_parts[0] == "forget" and len(forget_parts) <= 2:
            result = await self._handle_forget(text, user_id=user_id, source=source, session_id=session_id)
            self._finish(
                text,
                result,
                reply_channel,
                platform,
                elapsed=time.monotonic() - started,
                cached=False,
                session_id=session_id,
                user_id=user_id,
            )
            return result

        # 1. Semantic cache (MODULE 4), scoped to this chat session.
        tool_names = [s["name"] for s in self.tools.schemas()]
        cached = self.cache.lookup(text, tool_names, scope=session_id)
        if cached is not None:
            self._notify(on_event, {"type": "cached"})
            result = {**cached, "cached": True, "session_id": session_id}
            self._append_turn(session_id, text, str(result.get("response", "")))
            self._finish(
                text,
                result,
                reply_channel,
                platform,
                elapsed=time.monotonic() - started,
                cached=True,
                session_id=session_id,
                user_id=user_id,
                source=source,
            )
            return result

        # 2. Context assembly + bounded ReAct loop.
        messages = await self._build_messages(
            text, session_id=session_id, user_id=user_id, source=source
        )
        steps: list[dict[str, Any]] = []
        response: str | None = None
        invalid_streak = 0

        for step_idx in range(self.max_steps):
            self._notify(on_event, {"type": "thinking", "step": step_idx})
            action = await self._ask(messages)
            if action.get("action") == "respond":
                response = action.get("response") or action.get("thought") or ""
                break

            tool_name = str(action.get("tool") or "").strip()
            args = action.get("args") or {}

            # A malformed/unknown tool call is DATA, not an action: record it,
            # tell the model exactly which names are valid, and don't waste an
            # execution attempt on it.
            if not tool_name or self.tools.get(tool_name) is None:
                error = f"unknown tool: {tool_name}" if tool_name else "no tool name given"
                steps.append({
                    "step": step_idx,
                    "tool": tool_name or None,
                    "args": args,
                    "ok": False,
                    "error": error,
                })
                self._notify(on_event, {"type": "invalid_tool", "step": step_idx, "tool": tool_name})
                valid = ", ".join(s.name for s in self.tools.list())[:800]
                messages.append({"role": "assistant", "content": json.dumps(action)})
                messages.append({
                    "role": "user",
                    "content": (
                        f"OBSERVATION (invalid tool): {error}. "
                        f"Call one of these exact tool names, or use action=respond: {valid}"
                    ),
                })
                invalid_streak += 1
                if invalid_streak >= 2:  # don't burn the whole budget on bad names
                    break
                continue
            invalid_streak = 0

            self._notify(on_event, {"type": "tool", "step": step_idx, "tool": tool_name, "args": args})
            result = await self.tools.execute(tool_name, args)
            ok = bool(result.get("ok"))
            steps.append({"step": step_idx, "tool": tool_name, "args": args, "ok": ok})
            self._notify(on_event, {"type": "tool_result", "step": step_idx, "tool": tool_name, "ok": ok})

            # Compact observation (delta-first policy keeps this small).
            observation = json.dumps(result, default=str)
            if len(observation) > 2000:  # hard cap; tools should emit diffs
                observation = observation[:2000] + "…[truncated]"
            messages.append({"role": "assistant", "content": json.dumps(action)})
            messages.append({"role": "user", "content": f"OBSERVATION ({tool_name}): {observation}"})

            # Re-prune after each step (MODULE 4).
            messages = (await self.prune(messages))[0]

        # 3. Guarantee a final answer even if the loop hit max_steps.
        if response is None:
            messages.append({
                "role": "user",
                "content": "MAXIMUM STEPS REACHED. Respond now with the best answer so far (action=respond).",
            })
            self._notify(on_event, {"type": "thinking", "step": "final"})
            action = await self._ask(messages)
            response = action.get("response") or action.get("thought") or ""

        # 4. Persist experience + cache + notify.
        response = response or "⚠ I couldn't process that request. Please try again."
        result = {
            "response": response,
            "steps": steps,
            "cached": False,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "source": source,
            "session_id": session_id,
        }
        self.cache.store(text, tool_names, result, scope=session_id)
        self._append_turn(session_id, text, response)
        if self.memory is not None:
            try:
                actions = "; ".join(
                    f"{s.get('tool') or 'invalid-tool'}({'ok' if s.get('ok') else 'FAIL'})" for s in steps
                ) or "no-tools"
                self.memory.append_episode(
                    actions,
                    response,
                    metadata={"task": text, "steps": steps, "session_id": session_id, "source": source},
                )
            except Exception:  # noqa: BLE001 — memory must never break the loop
                logger.exception("failed to write episode")
        self._finish(
            text,
            result,
            reply_channel,
            platform,
            elapsed=(time.monotonic() - started),
            cached=False,
            session_id=session_id,
            user_id=user_id,
            source=source,
        )
        return result

    # ── profile management (/forget, owner-only, confirmation-gated) ───────
    async def _handle_forget(
        self,
        text: str,
        *,
        user_id: str | None,
        source: str,
        session_id: str,
    ) -> dict[str, Any]:
        """``/forget`` wipes the sender's durable profile.

        First send ``/forget`` to arm a confirmation (TTL from
        ``ProfileStore``), then ``/forget confirm`` to actually wipe. Only the
        owner profile may be managed this way; isolated non-owner profiles are
        out of scope here. Memory failures never break the loop.
        """
        memory = self.memory
        if memory is None or not hasattr(memory, "clear_profile"):
            return {
                "response": "⚠ profile storage is unavailable right now.",
                "steps": [],
                "cached": False,
                "session_id": session_id,
            }
        try:
            parts = text.strip().lower().split()
            confirmed = len(parts) == 2 and parts[1] == "confirm"
            outcome = memory.clear_profile(user_id or "", source=source, confirmed=confirmed)
        except Exception:  # noqa: BLE001
            logger.exception("profile clear failed")
            return {
                "response": "⚠ couldn't process the profile clear request.",
                "steps": [],
                "cached": False,
                "session_id": session_id,
            }
        if not outcome.get("ok"):
            return {
                "response": f"⚠ {outcome.get('error', 'profile clear failed')}.",
                "steps": [],
                "cached": False,
                "session_id": session_id,
            }
        if outcome.get("pending"):
            ttl = int(outcome.get("ttl_seconds", 0) or 0)
            return {
                "response": (
                    "⚠ This will wipe your saved profile (name, preferences, "
                    f"pinned facts) for user {outcome.get('user_id')}. "
                    f"Send /forget confirm within {ttl // 60} minutes to proceed."
                ),
                "steps": [],
                "cached": False,
                "session_id": session_id,
            }
        return {
            "response": f"🗑 Your profile ({outcome.get('user_id')}) has been wiped. I'll call you by name again once you tell me.",
            "steps": [],
            "cached": False,
            "session_id": session_id,
        }

    @staticmethod
    def _notify(on_event: ProgressCallback | None, event: dict[str, Any]) -> None:
        """Deliver a UI progress event; UI failures never affect the loop."""
        if on_event is None:
            return
        try:
            on_event(event)
        except Exception:  # noqa: BLE001
            logger.debug("progress callback failed", exc_info=True)

    def _finish(
        self,
        text: str,
        result: dict[str, Any],
        reply_channel: str | None,
        platform: str | None = None,
        *,
        elapsed: float,
        cached: bool,
        session_id: str = "main",
        user_id: str | None = None,
        source: str | None = None,
    ) -> None:
        self.bus.publish(
            "task:completed",
            {
                "task": text,
                "response": result.get("response", "") or "",
                "cached": cached,
                "reply_channel": reply_channel,
                "platform": platform,
                "session_id": session_id,
                "user_id": user_id,
                "source": source,
            },
            source="orchestrator",
        )
        if cached:
            result.setdefault("elapsed_ms", int(elapsed * 1000))

    # ── LLM plumbing ───────────────────────────────────────────────────────
    async def _ask(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """One structured LLM call; defensive parsing guarantees the loop can
        continue even with sloppy local models."""
        try:
            out = await self.provider.complete(messages, schema=ACTION_SCHEMA, temperature=0.2)
        except LLMProviderError as exc:
            # Known, expected provider failures get a clean, human message —
            # no tracebacks, no raw httpx internals in the user's face. INFO
            # keeps the interactive spinner line free of interleaved log text.
            logger.info("LLM provider error: %s", exc)
            if exc.status == 429:
                friendly = ("⏳ rate limited by the LLM provider — it already retried a few times. "
                            "Wait a few seconds, then send again.")
            elif exc.status == 0:
                friendly = (f"⚠ can't reach the LLM at {self.config.llm_base_url} — "
                            "is the server running?")
            else:
                friendly = f"LLM provider error (HTTP {exc.status}): {exc}"
            return {"thought": "provider error", "action": "respond", "response": friendly}
        except Exception as exc:  # noqa: BLE001
            # One concise line in the answer; traceback only with -v.
            logger.info("LLM call failed: %s", exc)
            logger.debug("LLM call traceback", exc_info=True)
            return {
                "thought": "provider error",
                "action": "respond",
                "response": f"⚠ LLM call failed ({type(exc).__name__}): {exc}",
            }
        if isinstance(out, str):  # provider returned raw text
            out = {"content": out}
        content = out.get("content") if isinstance(out, dict) else None
        if isinstance(content, dict):  # already parsed JSON
            return self._coerce_action(content)
        if isinstance(content, str):
            return self._coerce_action(_extract_json(content))
        return self._coerce_action({})

    @staticmethod
    def _coerce_action(data: dict[str, Any]) -> dict[str, Any]:
        data = data if isinstance(data, dict) else {}
        action = data.get("action", "respond")
        if action not in ("tool", "respond"):
            action = "respond"
        thought = str(data.get("thought", ""))[:400]
        response = data.get("response")
        # When the LLM emits garbage (e.g. thought="json", response=null),
        # don't propagate nonsense as the user-facing response.
        if not response and thought.strip() in {"json", ""}:
            thought = ""
            response = "⚠ I couldn't process that request. Please try again."
        return {
            "thought": thought,
            "action": action,
            "tool": data.get("tool"),
            "args": data.get("args") if isinstance(data.get("args"), dict) else {},
            "response": response,
        }

    # ── context assembly & pruning (MODULE 4) ──────────────────────────────
    async def _build_messages(
        self, text: str, *, session_id: str = "main", user_id: str | None = None, source: str | None = None
    ) -> list[dict[str, str]]:
        tools = json.dumps(self.tools.schemas(), separators=(",", ":"))
        # Working Memory (volatile) + Semantic recall (durable), if present.
        wm_block, recall_block = "", ""
        profile_block = ""
        mem = self.memory
        if mem is not None:
            try:
                wm_items = mem.working_snapshot(limit=6)
                if wm_items:
                    wm_block = "WORKING MEMORY:\n" + "\n".join(f"- {i}" for i in wm_items) + "\n"
                recalled = mem.recall(text, k=4)
                if recalled:
                    recall_block = (
                        "RECALLED MEMORY:\n"
                        + "\n".join(f"- {r.get('text', '')[:200]}" for r in recalled)
                        + "\n"
                    )
                # IDENTITY IN EVERY PROFILE: the USER PROFILE block comes from
                # the durable ProfileStore and is injected into EVERY prompt.
                # session:reset and restarts never touch it.
                profile_block = mem.profile_block(user_id or "", source=source)
            except Exception:  # noqa: BLE001
                logger.exception("memory recall failed; continuing without it")
        system = SYSTEM_TEMPLATE.format(
            tools=tools,
            memory_block=recall_block,
            working_memory=wm_block,
            user_profile=profile_block,
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        # Fold prior conversation for *this* session, pruned to budget.
        prior = self._session(session_id)[-8:]
        if prior:
            kept, dropped = self._fit(prior, reserve=self.budget.count(system) + self.budget.count(text) + 256)
            summary = await self._summarize_dropped(dropped)
            if summary:
                messages.append({"role": "system", "content": f"MEMORY SUMMARY of earlier turns:\n{summary}"})
            messages.extend(kept)
        messages.append({"role": "user", "content": text})
        return (await self.prune(messages))[0]

    async def prune(self, messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
        """Hard token ceiling; dropped turns collapse into ONE summary block."""
        budget = self.config.max_context_tokens
        used = self.budget.count_messages(messages)
        if used <= int(budget * self.config.prune_target_ratio):
            return messages, ""
        kept, dropped = self._fit(messages, reserve=0)
        summary = await self._summarize_dropped(dropped)
        if summary:
            # Summary block goes right after leading system prompts.
            insert_at = 0
            for i, m in enumerate(kept):
                if m["role"] != "system":
                    insert_at = i
                    break
            else:
                insert_at = len(kept)
            kept.insert(insert_at, {"role": "system", "content": f"MEMORY SUMMARY of earlier turns:\n{summary}"})
        return kept, summary

    def _fit(self, messages: list[dict[str, str]], *, reserve: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Keep system prompts + the newest turns that fit; return the rest."""
        budget = int(self.config.max_context_tokens * self.config.prune_target_ratio) - reserve
        if budget <= 0:
            budget = self.config.max_context_tokens // 2
        system: list[dict[str, str]] = []
        rest: list[dict[str, str]] = []
        for m in messages:
            (system if m["role"] == "system" else rest).append(m)

        kept: list[dict[str, str]] = []
        used = self.budget.count_messages(system)
        dropped: list[dict[str, str]] = []
        for m in reversed(rest):  # newest first
            cost = self.budget.count(m["content"]) + 4
            if used + cost > budget:
                dropped = list(reversed(rest[: len(rest) - len(kept)]))  # current + all older
                break
            kept.insert(0, m)
            used += cost
        return system + kept, dropped

    async def _summarize_dropped(self, dropped: list[dict[str, str]]) -> str:
        """Summarize old turns into ONE dense block. Uses the LLM for large
        drops; falls back to deterministic truncation (bounded, zero cost)."""
        if not dropped:
            return ""
        blob = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in dropped)
        if self.budget.count(blob) > 512 and len(dropped) >= 4:
            try:  # structured summarization — never let it crash the loop
                out = await self.provider.complete(
                    [{"role": "user", "content": f"Summarize this agent transcript densely (<=120 tokens):\n{blob}"}],
                    schema=SUMMARY_SCHEMA,
                    temperature=0.0,
                    max_tokens=200,
                )
                content = out.get("content") if isinstance(out, dict) else None
                if isinstance(content, dict):
                    return str(content.get("summary", ""))[:800]
                if isinstance(content, str):
                    parsed = _extract_json(content)
                    return str(parsed.get("summary", content[:600]))
            except Exception:  # noqa: BLE001
                logger.exception("LLM summary failed; using truncation")
        lines = [f"{m['role']}: {m['content'][:160]}" for m in dropped[-6:]]
        return "\n".join(lines) + ("\n[+older turns omitted]" if len(dropped) > 6 else "")


# _extract_json is imported from core.utils (single source of truth).
