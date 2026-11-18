"""Shared gateway plumbing (MODULE 6).

A gateway is a *translator*: it converts platform-specific traffic (webhooks,
polls, gateway events) into one internal event shape and routes agent replies
back to the originating chat. The interesting behavior — access control,
pairing, mention gating, sessions, chunking — lives here so Discord, Telegram,
and future adapters stay thin and consistent.

Event contract::

    inbound:  bus.publish("task:incoming", {
                  "text": <user text>,
                  "source": <gateway name>,
                  "reply_channel": "<gateway>:<platform channel id>",
                  "session_id": "<per-chat session>",
                  "user": <display name>,
                  "user_id": <stable platform id>,
                  "message_id": <platform message id>,
              })
    outbound: subscribe to "task:completed" — payloads whose ``reply_channel``
              starts with ``<gateway>:`` are delivered through ``send_reply``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable

from python_core.core.event_bus import EventBus
from python_core.core.interfaces import KernelContext

from .pairing import PairingStore

logger = logging.getLogger("echo.gateway")

# Platform message limits (hard caps are 2000 for Discord and 4096 for Telegram).
DISCORD_CHUNK = 1900
TELEGRAM_CHUNK = 3900

_REPLY_TAG = re.compile(r"\[\[\s*reply_to(?::([^\]\s]+)|_current)?\s*\]\]", re.IGNORECASE)


# ── text helpers ────────────────────────────────────────────────────────────
def chunk_text(text: str, limit: int, *, mode: str = "newline") -> list[str]:
    """Split *text* into platform-sized chunks without cutting words or fences.

    Paragraph boundaries win, then newlines, then spaces. A chunk never ends
    inside an odd number of ``` fences when a fence boundary is available.
    """
    text = text or ""
    limit = max(1, int(limit))
    if len(text) <= limit:
        return [text] if text else []

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        # Prefer a fence boundary so code blocks stay readable.
        cut = -1
        if window.count("```") % 2 == 1:
            fence = window.rfind("```")
            # Cut *before* the opening fence when possible, so a short code
            # block moves whole into the next chunk instead of being split.
            before_fence = window.rfind("\n", 0, fence)
            candidate = before_fence if before_fence >= limit // 3 else fence
            if candidate >= limit // 3:
                cut = candidate
        if cut < 0 and mode == "newline":
            cut = window.rfind("\n\n")
            if cut < limit // 3:
                cut = window.rfind("\n")
        if cut < 0:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        piece = remaining[:cut].rstrip()
        if piece:
            chunks.append(piece)
        remaining = remaining[cut:].lstrip("\n")
    if remaining.strip():
        chunks.append(remaining)
    return chunks


def parse_reply_tags(text: str) -> tuple[str, str | None]:
    """Strip ``[[reply_to_current]]`` / ``[[reply_to:<id>]]`` from *text*.

    Returns ``(clean_text, target)`` where target is ``"current"``, a platform
    message id, or ``None``.
    """
    target: str | None = None

    def _replace(match: re.Match[str]) -> str:
        nonlocal target
        explicit = match.group(1)
        target = explicit.strip() if explicit else "current"
        return ""

    cleaned = _REPLY_TAG.sub(_replace, text or "")
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip(), target


def normalize_id(value: Any) -> str:
    """Normalize an allowlist/chat entry: drop channel prefixes and wrappers."""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    for prefix in ("telegram:", "tg:", "discord:", "user:", "id:", "channel:"):
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            lowered = text.lower()
            break
    if text.startswith("<@") and text.endswith(">"):
        text = text[3:-1].lstrip("!")
    return text.lstrip("@").strip()


def normalize_allowlist(values: Iterable[Any] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        norm = normalize_id(value)
        if norm and norm.lower() not in seen:
            seen.add(norm.lower())
            out.append(norm)
    return out


def _matches(entry: str, sender_id: Any, username: str | None) -> bool:
    if entry == "*":
        return True
    candidate = str(sender_id).strip()
    if candidate and entry == candidate:
        return True
    if username and entry.lower() == str(username).lstrip("@").lower():
        return True
    return False


def sender_allowed(
    allowlist: Iterable[Any] | None,
    sender_id: Any,
    username: str | None = None,
) -> bool:
    """Allow when the stable ID or username matches (``*`` opens the list)."""
    entries = normalize_allowlist(allowlist)
    if not entries:
        return False
    return any(_matches(entry, sender_id, username) for entry in entries)


def id_allowed(allowlist: Iterable[Any] | None, identifier: Any) -> bool:
    entries = normalize_allowlist(allowlist)
    if not entries:
        return False
    candidate = str(identifier).strip()
    return any(entry == "*" or entry == candidate for entry in entries)


def parse_slash_command(text: str, bot_username: str = "") -> tuple[str | None, str]:
    """Parse ``/cmd@bot args``; return ``(command, args)`` or ``(None, text)``.

    A command addressed to another bot returns the sentinel ``""`` so callers
    can ignore it instead of treating it as ordinary text.
    """
    text = (text or "").strip()
    if not text.startswith("/"):
        return None, text
    head, _, rest = text.partition(" ")
    name = head[1:].strip()
    target = ""
    if "@" in name:
        name, _, target = name.partition("@")
    if target and bot_username and target.lower() != bot_username.lstrip("@").lower():
        return "", ""
    return name.lower(), rest.strip()


# ── admission decisions ─────────────────────────────────────────────────────
@dataclass(slots=True)
class Admission:
    allowed: bool
    reason: str = ""
    pairing_code: str | None = None
    pairing_status: str = ""


class GatewayInterface(ABC):
    """Lifecycle + routing contract for all chat-platform gateways."""

    name: str = "gateway"
    version: str = "0.2.0"
    capabilities: list[str] = ["gateway"]

    def __init__(self) -> None:
        self._bus: EventBus | None = None
        self._ctx: KernelContext | None = None
        self._reply_sub = None
        self._started_sub = None
        self.pairing: PairingStore | None = None
        self._reply_targets: dict[str, str] = {}
        self._pending: dict[str, int] = {}
        self._typing_tasks: dict[str, asyncio.Task[None]] = {}
        self._typing_enabled = False
        self._typing_interval = 4.5
        self._notification_task: asyncio.Task[None] | None = None

    # ── lifecycle (ABC contract mirrors Plugin Protocol) ───────────────────
    @abstractmethod
    async def start(self, ctx: KernelContext) -> None:
        """Connect to the platform; register bus subscriptions."""

    @abstractmethod
    async def stop(self) -> None:
        """Disconnect cleanly."""

    @abstractmethod
    async def send_reply(self, channel_id: str, text: str) -> None:
        """Deliver *text* to a platform channel (platform-specific)."""

    async def send_typing(self, channel_id: str) -> None:  # noqa: ARG002 — platforms opt in
        """Show a typing/working indicator; default is a no-op."""

    # ── shared setup / routing ─────────────────────────────────────────────
    def bind_pairing(self, ctx: KernelContext) -> None:
        if self.pairing is not None:
            return
        existing = ctx.service("pairing")
        if isinstance(existing, PairingStore):
            self.pairing = existing
        else:
            self.pairing = PairingStore(
                ctx.data_dir / "gateways" / "pairing.json",
                ttl=ctx.config.gateway_pairing_ttl,
                max_pending=ctx.config.gateway_pairing_max_pending,
            )
            ctx.services["pairing"] = self.pairing

    def bind_bus(self, bus: EventBus) -> None:
        """Subscribe to completed tasks addressed to this gateway."""
        self._bus = bus
        prefix = f"{self.name}:"

        async def _on_completed(event: Any) -> None:
            payload = getattr(event, "payload", event) or {}
            reply_channel = str(payload.get("reply_channel") or "")
            if not reply_channel.startswith(prefix):
                return
            # Platform-specific routing: only deliver if the task's platform
            # matches this gateway (defense-in-depth alongside the prefix check).
            platform = str(payload.get("platform") or "")
            if platform and platform != self.name:
                return
            channel_id = reply_channel[len(prefix):]
            response = str(payload.get("response", "")).strip()
            if not response:
                return
            self._finish_typing(channel_id)
            try:
                await self.send_reply(channel_id, response)
                await self._on_task_completed(channel_id, response, ok=True)
            except Exception as exc:  # noqa: BLE001 — delivery failure must not kill the bus
                logger.warning("%s failed to deliver reply to %s: %s", self.name, channel_id, exc)
                logger.debug("gateway delivery traceback", exc_info=True)
                await self._on_task_completed(channel_id, response, ok=False)

        async def _on_job_started(event: Any) -> None:
            """Notify the originating chat that a scheduled job has started."""
            payload = getattr(event, "payload", event) or {}
            reply_channel = str(payload.get("reply_channel") or "")
            if not reply_channel.startswith(prefix):
                return
            platform = str(payload.get("platform") or "")
            if platform and platform != self.name:
                return
            channel_id = reply_channel[len(prefix):]
            job_name = str(payload.get("job_name") or "job")
            job_id = str(payload.get("job_id") or "?")
            try:
                await self.send_reply(
                    channel_id,
                    f"🔄 Cron job `{job_name}` ({job_id}) started…",
                )
            except Exception as exc:  # start notice is cosmetic
                logger.warning("%s failed to send job-started notice to %s: %s", self.name, channel_id, exc)

        self._reply_sub = bus.subscribe("task:completed", _on_completed, label=f"gateway.{self.name}.reply")
        self._started_sub = bus.subscribe(
            "scheduler:job_started", _on_job_started, label=f"gateway.{self.name}.job-started"
        )

    async def _on_task_completed(self, channel_id: str, response: str, *, ok: bool) -> None:
        """Hook for gateways to react or update status when a task completes.

        Default does nothing. Subclasses override to add emoji reactions,
        status-message edits, etc.
        """
        pass

    def unbind_bus(self) -> None:
        if self._bus is not None and self._reply_sub is not None:
            self._bus.unsubscribe(self._reply_sub)
            self._reply_sub = None
        if self._bus is not None and self._started_sub is not None:
            self._bus.unsubscribe(self._started_sub)
            self._started_sub = None
        for task in list(self._typing_tasks.values()):
            task.cancel()
        self._typing_tasks.clear()
        self._pending.clear()
        if self._notification_task is not None:
            self._notification_task.cancel()
            self._notification_task = None

    # ── approval notifications (OpenClaw ``pairing approve --notify``) ───────
    def start_notification_pump(self, interval: float = 15.0) -> None:
        """Poll the pairing store for confirmations queued by the CLI.

        The CLI runs in a separate process, so approvals with ``--notify`` are
        written to the shared store and delivered by the live gateway.
        """
        if self._notification_task is not None and not self._notification_task.done():
            return
        self._notification_task = asyncio.create_task(
            self._notification_loop(interval),
            name=f"echo-pairing-notify-{self.name}",
        )

    async def _notification_loop(self, interval: float) -> None:
        try:
            while True:
                await asyncio.sleep(interval)
                await self.flush_pairing_notifications()
        except asyncio.CancelledError:
            raise
        except Exception:  # never let the pump kill the gateway
            logger.debug("pairing notification pump failed", exc_info=True)

    async def flush_pairing_notifications(self) -> None:
        """Deliver every queued approval confirmation for this gateway."""
        store = self.pairing
        if store is None:
            return
        for notice in store.take_notifications(self.name):
            chat_id = str(notice.get("chat_id") or notice.get("sender_id") or "").strip()
            if not chat_id:
                continue
            try:
                await self.send_reply(chat_id, self.approval_notice(self.name.title()))
            except Exception:  # confirmation is cosmetic
                logger.debug("pairing confirmation delivery failed for %s", chat_id, exc_info=True)

    @staticmethod
    def approval_notice(platform: str = "") -> str:
        suffix = f" on {platform}" if platform else ""
        return f"✅ Pairing approved{suffix}. You can now chat with echo-agent."

    def emit_task(
        self,
        bus: EventBus,
        text: str,
        channel_id: str,
        *,
        source_user: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        message_id: str | None = None,
        chat_type: str = "dm",
        chat_id: str | None = None,
    ) -> None:
        """Standard inbound translation: platform message → internal task."""
        if message_id:
            self._reply_targets[str(channel_id)] = str(message_id)
        if self._typing_enabled:
            self._begin_typing(str(channel_id))
        bus.publish(
            "task:incoming",
            {
                "text": text,
                "source": self.name,
                "reply_channel": f"{self.name}:{channel_id}",
                "user": source_user,
                "user_id": user_id,
                "chat_id": chat_id or str(channel_id),
                "chat_type": chat_type,
                "session_id": session_id,
                "message_id": message_id,
            },
            source=f"gateway:{self.name}",
        )

    # ── typing indicator ────────────────────────────────────────────────────
    def _begin_typing(self, channel_id: str) -> None:
        self._pending[channel_id] = self._pending.get(channel_id, 0) + 1
        if channel_id not in self._typing_tasks:
            self._typing_tasks[channel_id] = asyncio.create_task(
                self._typing_loop(channel_id), name=f"echo-typing-{self.name}-{channel_id}"
            )

    async def _typing_loop(self, channel_id: str) -> None:
        try:
            # Bound the indicator: if a task never completes, stop pinging the
            # platform after ~12 minutes instead of typing forever.
            for _ in range(160):
                if self._pending.get(channel_id, 0) <= 0:
                    return
                try:
                    await self.send_typing(channel_id)
                except Exception:  # noqa: BLE001 — indicator is cosmetic
                    logger.debug("%s typing indicator failed for %s", self.name, channel_id, exc_info=True)
                await asyncio.sleep(self._typing_interval)
        except asyncio.CancelledError:
            raise
        finally:
            current = asyncio.current_task()
            if self._typing_tasks.get(channel_id) is current:
                self._typing_tasks.pop(channel_id, None)

    def _finish_typing(self, channel_id: str) -> None:
        count = self._pending.get(channel_id, 0)
        if count <= 1:
            self._pending.pop(channel_id, None)
            task = self._typing_tasks.pop(channel_id, None)
            if task is not None:
                task.cancel()
        else:
            self._pending[channel_id] = count - 1

    # ── access control ──────────────────────────────────────────────────────
    def admit_dm(
        self,
        *,
        policy: str,
        allow_from: Iterable[Any] | None,
        sender_id: Any,
        username: str = "",
        chat_id: str = "",
    ) -> Admission:
        """Decide whether a direct message may reach the agent."""
        if policy == "disabled":
            return Admission(False, "dm disabled")
        if policy == "open" or sender_allowed(allow_from, sender_id, username):
            return Admission(True)
        if policy == "allowlist":
            return Admission(False, "sender not in allowlist")
        # pairing
        if self.pairing is not None and self.pairing.is_approved(self.name, sender_id):
            return Admission(True)
        if self.pairing is None:
            return Admission(False, "pairing store unavailable")
        request = self.pairing.request(
            self.name, sender_id, username=username, chat_id=chat_id
        )
        if request.get("status") == "created":
            return Admission(False, "pairing required", str(request.get("code") or ""), "created")
        return Admission(
            False,
            "pairing pending",
            str(request.get("code") or "") or None,
            str(request.get("status") or ""),
        )

    def admit_group(
        self,
        *,
        policy: str,
        allowed_groups: Iterable[Any] | None,
        group_id: Any,
        sender_allowlist: Iterable[Any] | None,
        sender_id: Any,
        username: str = "",
    ) -> Admission:
        """Decide whether a group message may reach the agent.

        An empty sender allowlist means "any member of an allowed group"; set
        one explicitly to restrict who can trigger the bot inside a group.
        """
        if policy == "disabled":
            return Admission(False, "groups disabled")
        if policy == "allowlist" and not id_allowed(allowed_groups, group_id):
            return Admission(False, "group not in allowlist")
        if sender_allowlist and not sender_allowed(sender_allowlist, sender_id, username):
            return Admission(False, "sender not in group allowlist")
        return Admission(True)

    @staticmethod
    def pairing_prompt(code: str, *, platform: str, user_id: Any, ttl: int = 3600) -> str:
        minutes = max(1, ttl // 60)
        return (
            "🔐 Pairing required.\n\n"
            f"Your {platform} user id: {user_id}\n"
            f"Pairing code: {code}\n\n"
            "Approve from the machine running echo-agent:\n"
            f"  echo-agent pairing approve {platform.lower()} {code}\n\n"
            f"The code expires in {minutes} minutes."
        )

    @staticmethod
    def help_text(platform: str, prefix: str = "/") -> str:
        p = prefix
        return (
            f"echo-agent · {platform}\n\n"
            f"{p}ask <text>          run a task\n"
            f"{p}status              session + model status\n"
            f"{p}model [name]        show or change the LLM model\n"
            f"{p}reset / {p}new        clear this chat's session\n"
            f"{p}whoami              show your sender/chat ids\n"
            f"{p}activation always   stop requiring @mentions in this group\n"
            f"{p}activation mention  require @mentions again\n"
            f"{p}help                this message\n\n"
            "In DMs you can also just type normally.\n\n"
            "Reactions: 👀 = received, ✅ = done, ❌ = failed."
        )


def create_plugin() -> GatewayInterface:  # pragma: no cover - abstract fallback
    raise NotImplementedError("use a concrete gateway")
