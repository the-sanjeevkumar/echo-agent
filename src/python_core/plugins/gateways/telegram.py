"""Telegram gateway — OpenClaw-style DMs, groups, commands, and sessions.

Deliberately dependency-light: the Bot API is plain HTTPS JSON and ``httpx`` is
already a core dependency. Long polling is the default transport (no webhook,
no public URL required).

Behavior highlights
-------------------
* DMs default to **pairing**: unknown senders get a code, and their message is
  not processed until ``echo-agent pairing approve telegram <CODE>``.
* Groups default to an **allowlist** (``telegram_groups``), with optional
  sender filtering, @mention gating, and per-chat sessions (including forum
  topics).
* Native command menu (``/ask``, ``/status``, ``/reset``, ``/whoami`` …),
  typing indicator, long-message chunking, reply threading, and graceful
  reconnection/backoff.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

import httpx

from python_core.core.interfaces import KernelContext

from .base import (
    TELEGRAM_CHUNK,
    Admission,
    GatewayInterface,
    chunk_text,
    parse_reply_tags,
    parse_slash_command,
)

logger = logging.getLogger("echo.gateway.telegram")

API_BASE = "https://api.telegram.org"
MAX_MESSAGE = TELEGRAM_CHUNK  # Telegram hard limit is 4096
COMMANDS = [
    {"command": "ask", "description": "Run a task"},
    {"command": "status", "description": "Session and model status"},
    {"command": "reset", "description": "Clear this chat's session"},
    {"command": "whoami", "description": "Show sender and chat ids"},
    {"command": "activation", "description": "Toggle mention requirement in a group"},
    {"command": "help", "description": "List commands"},
]


class TelegramError(RuntimeError):
    pass


class TelegramUnauthorized(TelegramError):
    pass


class TelegramConflict(TelegramError):
    pass


class TelegramGateway(GatewayInterface):
    name = "telegram"
    version = "0.2.0"
    capabilities = ["gateway", "telegram", "pairing", "sessions"]

    def __init__(self, token: str = "") -> None:
        super().__init__()
        self.token = token
        self.config: Any = None
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._offset = 0
        self._offset_path: Path | None = None
        self._offset_dirty = 0
        self._http: httpx.AsyncClient | None = None
        self._bot_id: int | None = None
        self._bot_username = ""
        self._activation: dict[str, bool] = {}  # chat_id → require mention

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start(self, ctx: KernelContext) -> None:
        self._configure(ctx)
        if not self.token:
            self.token = ctx.config.telegram_token
        if not self.token:
            ctx.logger.info("telegram gateway disabled (no token)")
            return
        if self.config.telegram_dm_policy == "allowlist" and not self.config.telegram_allow_from:
            ctx.logger.warning("telegram dm_policy=allowlist but telegram_allow_from is empty — all DMs blocked")
        if self.config.telegram_group_policy == "allowlist" and not self.config.telegram_groups:
            ctx.logger.warning("telegram group_policy=allowlist but telegram_groups is empty — all groups blocked")

        self.bind_bus(ctx.bus)
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(65.0, connect=15.0))
        self._running = True
        self._task = asyncio.create_task(self._run(), name="echo-telegram")
        ctx.logger.info("telegram gateway polling")

    def _configure(self, ctx: KernelContext) -> None:
        """Pure configuration step, kept separate so tests can drive handlers."""
        self._ctx = ctx
        self.config = ctx.config
        self.bind_pairing(ctx)
        self._typing_enabled = bool(ctx.config.gateway_typing)
        self._offset_path = ctx.data_dir / "gateways" / "telegram-offset"
        self._offset = self._load_offset()

    async def stop(self) -> None:
        self._running = False
        self.unbind_bus()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None
        self._save_offset()
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception:  # noqa: BLE001
                logger.debug("telegram http close failed", exc_info=True)
            self._http = None

    # ── transport ──────────────────────────────────────────────────────────
    async def _run(self) -> None:
        backoff = 1.0
        while self._running:
            try:
                if self._bot_id is None:
                    await self._bootstrap()
                await self._poll_once()
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except TelegramUnauthorized as exc:
                logger.error("telegram rejected the bot token: %s", exc)
                self._running = False
                break
            except TelegramConflict as exc:
                logger.error("telegram polling conflict: %s", exc)
                await asyncio.sleep(max(backoff, 10.0))
            except Exception as exc:  # noqa: BLE001 — network hiccups must not kill the gateway
                logger.warning("telegram poll failed: %s; retrying in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _bootstrap(self) -> None:
        data = await self._api("getMe")
        result = data.get("result") or {}
        self._bot_id = int(result.get("id")) if result.get("id") is not None else None
        self._bot_username = str(result.get("username") or "")
        try:
            await self._api("setMyCommands", commands=COMMANDS)
        except Exception:  # noqa: BLE001 — menus are cosmetic; polling still works
            logger.debug("setMyCommands failed", exc_info=True)
        logger.info("telegram connected as @%s", self._bot_username or self._bot_id)

    async def _poll_once(self) -> None:
        data = await self._api(
            "getUpdates",
            timeout=50,
            offset=self._offset,
            allowed_updates=["message"],
        )
        for update in data.get("result") or []:
            try:
                self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)
                self._offset_dirty += 1
                await self._handle_update(update)
            except Exception as exc:  # noqa: BLE001 — one bad update must not stall the poller
                logger.warning("telegram update handling failed: %s", exc)
                logger.debug("telegram update traceback", exc_info=True)
        if self._offset_dirty >= 20:
            self._save_offset()

    async def _api(self, method: str, **params: Any) -> dict[str, Any]:
        url = f"{API_BASE}/bot{self.token}/{method}"
        client = self._http
        temp = client is None
        if temp:
            client = httpx.AsyncClient(timeout=httpx.Timeout(65.0, connect=15.0))
        try:
            resp = await client.post(url, json=params)
            status = resp.status_code
            try:
                data = resp.json()
            except ValueError:
                data = {}
            if status == 401:
                raise TelegramUnauthorized(str(data.get("description") or "unauthorized"))
            if status == 409:
                raise TelegramConflict(str(data.get("description") or "conflict"))
            if status >= 400 or data.get("ok") is False:
                raise TelegramError(str(data.get("description") or f"HTTP {status}"))
            return data
        finally:
            if temp and client is not None:
                await client.aclose()

    # ── inbound ────────────────────────────────────────────────────────────
    async def _handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message")
        if isinstance(message, dict):
            await self._handle_message(message)

    async def _handle_message(self, message: dict[str, Any]) -> None:
        chat = message.get("chat") or {}
        chat_id = str(chat.get("id", "") or "")
        chat_type = str(chat.get("type") or "private")
        sender = message.get("from") or {}
        sender_id = sender.get("id")
        if not chat_id or sender_id is None:
            return
        if sender.get("is_bot") and not bool(self.config.telegram_allow_bots):
            return

        text = str(message.get("text") or message.get("caption") or "").strip()
        if not text:
            return

        is_private = chat_type == "private"
        username = str(sender.get("username") or "")
        display = username or str(sender.get("first_name") or sender_id)

        command, args = parse_slash_command(text, self._bot_username)
        if command == "":
            return  # addressed to a different bot
        if command is not None:
            await self._handle_command(
                command=command,
                args=args,
                message=message,
                chat_id=chat_id,
                chat_type=chat_type,
                sender_id=sender_id,
                username=username,
                display=display,
                is_private=is_private,
            )
            return

        admission = self._admit(
            is_private=is_private,
            chat_id=chat_id,
            sender_id=sender_id,
            username=username,
        )
        if not admission.allowed:
            await self._notify_denied(admission, chat_id, sender_id, is_private)
            return

        if not is_private:
            if not self._mentioned(message):
                if self._require_mention(chat_id):
                    return
                if self.config.telegram_ignore_other_mentions and self._mentions_others(message):
                    return
            else:
                text = self._strip_bot_mention(text).strip()
                if not text:
                    text = "hello"

        session_id = self._session_id(chat_id, chat_type, message)
        quoted = self._quoted_context(message)
        if quoted:
            text = f"{quoted}\n{text}"
        self.emit_task(
            self._bus,
            text,
            chat_id,
            source_user=display,
            user_id=str(sender_id),
            session_id=session_id,
            message_id=str(message.get("message_id") or "") or None,
            chat_type="dm" if is_private else "group",
            chat_id=chat_id,
        )

    def _admit(
        self,
        *,
        is_private: bool,
        chat_id: str,
        sender_id: Any,
        username: str,
    ) -> Admission:
        if is_private:
            return self.admit_dm(
                policy=self.config.telegram_dm_policy,
                allow_from=self.config.telegram_allow_from,
                sender_id=sender_id,
                username=username,
                chat_id=chat_id,
            )
        return self.admit_group(
            policy=self.config.telegram_group_policy,
            allowed_groups=self.config.telegram_groups,
            group_id=chat_id,
            sender_allowlist=self.config.telegram_group_allow_from or self.config.telegram_allow_from,
            sender_id=sender_id,
            username=username,
        )

    async def _notify_denied(
        self,
        admission: Admission,
        chat_id: str,
        sender_id: Any,
        is_private: bool,
    ) -> None:
        if not is_private:
            return  # never make noise in a group that is not authorized
        if admission.reason == "pairing required" and admission.pairing_code:
            await self.send_reply(
                chat_id,
                self.pairing_prompt(
                    admission.pairing_code,
                    platform="Telegram",
                    user_id=sender_id,
                    ttl=int(self.config.gateway_pairing_ttl),
                ),
            )
        elif admission.reason == "sender not in allowlist":
            await self.send_reply(
                chat_id,
                f"⛔ Not authorized. Your Telegram user id is {sender_id}.\n"
                "Ask the operator to add it to telegram_allow_from.",
            )
        # pairing pending / disabled: stay quiet to avoid spam.

    # ── commands ───────────────────────────────────────────────────────────
    async def _handle_command(
        self,
        *,
        command: str,
        args: str,
        message: dict[str, Any],
        chat_id: str,
        chat_type: str,
        sender_id: Any,
        username: str,
        display: str,
        is_private: bool,
    ) -> None:
        admission = self._admit(
            is_private=is_private, chat_id=chat_id, sender_id=sender_id, username=username
        )

        if command in {"start", "help"}:
            if not admission.allowed:
                await self._notify_denied(admission, chat_id, sender_id, is_private)
            else:
                greeting = "echo-agent online." if command == "start" else self.help_text("Telegram")
                await self.send_reply(chat_id, greeting)
            return

        if command == "whoami":
            await self.send_reply(
                chat_id,
                f"user_id: {sender_id}\nusername: @{username or '—'}\n"
                f"chat_id: {chat_id}\nchat_type: {chat_type}",
            )
            return

        if not admission.allowed:
            await self._notify_denied(admission, chat_id, sender_id, is_private)
            return

        session_id = self._session_id(chat_id, chat_type, message)

        if command == "status":
            status = await self._bus.request(
                "session:status", {"session_id": session_id}, timeout=5.0
            ) if self._bus is not None else None
            status = status or {}
            await self.send_reply(
                chat_id,
                f"session: {session_id}\n"
                f"messages: {status.get('messages', '?')}\n"
                f"model: {getattr(self.config, 'llm_model', '?')}\n"
                f"dm policy: {self.config.telegram_dm_policy}\n"
                f"group policy: {self.config.telegram_group_policy}",
            )
            return

        if command in {"reset", "new"}:
            if self._bus is not None:
                await self._bus.emit("session:reset", {"session_id": session_id}, source="telegram")
            await self.send_reply(chat_id, "🧹 session cleared for this chat.")
            return

        if command == "activation":
            if is_private:
                await self.send_reply(chat_id, "Activation only applies to groups.")
                return
            value = args.strip().lower()
            if value in {"always", "off", "false", "0"}:
                self._activation[chat_id] = False
                await self.send_reply(chat_id, "🔔 mention requirement off for this group.")
            elif value in {"mention", "on", "true", "1"}:
                self._activation[chat_id] = True
                await self.send_reply(chat_id, "🔕 mention requirement on for this group.")
            else:
                await self.send_reply(chat_id, "usage: /activation always | mention")
            return

        if command in {"ask", "task", "echo"}:
            if not args:
                await self.send_reply(chat_id, f"usage: /{command} <text>")
                return
            self.emit_task(
                self._bus,
                args,
                chat_id,
                source_user=display,
                user_id=str(sender_id),
                session_id=session_id,
                message_id=str(message.get("message_id") or "") or None,
                chat_type="dm" if is_private else "group",
                chat_id=chat_id,
            )
            return

        await self.send_reply(chat_id, f"unknown command: /{command}\n\n" + self.help_text("Telegram"))

    # ── mention handling ───────────────────────────────────────────────────
    def _require_mention(self, chat_id: str) -> bool:
        return self._activation.get(chat_id, bool(self.config.telegram_require_mention))

    def _mentioned(self, message: dict[str, Any]) -> bool:
        text = str(message.get("text") or message.get("caption") or "")
        bot = self._bot_username.lstrip("@").lower()
        if bot and re.search(rf"(?<![\w@])@{re.escape(bot)}(?!\w)", text, re.IGNORECASE):
            return True
        for entity in message.get("entities") or message.get("caption_entities") or []:
            if not isinstance(entity, dict):
                continue
            if entity.get("type") == "text_mention":
                user = entity.get("user") or {}
                if self._bot_id is not None and user.get("id") == self._bot_id:
                    return True
        reply_to = message.get("reply_to_message") or {}
        reply_sender = reply_to.get("from") or {}
        if self._bot_id is not None and reply_sender.get("id") == self._bot_id:
            return True
        return False

    def _strip_bot_mention(self, text: str) -> str:
        bot = self._bot_username.lstrip("@")
        if not bot:
            return text
        return re.sub(rf"(?<![\w@])@{re.escape(bot)}(?!\w)", "", text, flags=re.IGNORECASE)

    def _mentions_others(self, message: dict[str, Any]) -> bool:
        text = str(message.get("text") or message.get("caption") or "")
        for entity in message.get("entities") or message.get("caption_entities") or []:
            if not isinstance(entity, dict):
                continue
            if entity.get("type") == "mention":
                offset = int(entity.get("offset", 0))
                length = int(entity.get("length", 0))
                token = text[offset:offset + length].lstrip("@").lower()
                if token and token != self._bot_username.lstrip("@").lower():
                    return True
            elif entity.get("type") == "text_mention":
                user = entity.get("user") or {}
                if self._bot_id is not None and user.get("id") != self._bot_id:
                    return True
        return False

    @staticmethod
    def _session_id(chat_id: str, chat_type: str, message: dict[str, Any]) -> str:
        scope = "dm" if chat_type == "private" else "chat"
        base = f"telegram:{scope}:{chat_id}"
        thread = message.get("message_thread_id")
        return f"{base}:topic:{thread}" if thread else base

    @staticmethod
    def _quoted_context(message: dict[str, Any]) -> str:
        reply = message.get("reply_to_message") or {}
        quoted = str(reply.get("text") or reply.get("caption") or "").strip()
        if not quoted:
            return ""
        sender = reply.get("from") or {}
        who = sender.get("username") or sender.get("first_name") or "someone"
        return f"[replying to @{who}: {quoted[:300]}]" if sender.get("username") else f"[replying to {who}: {quoted[:300]}]"

    # ── outbound ───────────────────────────────────────────────────────────
    async def send_reply(self, channel_id: str, text: str) -> None:
        clean, explicit_target = parse_reply_tags(text)
        if not clean:
            return
        target = explicit_target
        if target == "current" or (target is None and self.config.gateway_reply_to_mode != "off"):
            target = self._reply_targets.get(str(channel_id))
        self._reply_targets.pop(str(channel_id), None)

        chunks = chunk_text(clean, MAX_MESSAGE, mode="newline")
        for index, chunk in enumerate(chunks):
            payload: dict[str, Any] = {
                "chat_id": channel_id,
                "text": chunk,
                "disable_web_page_preview": not bool(self.config.telegram_link_preview),
            }
            if index == 0 and target:
                try:
                    payload["reply_parameters"] = {
                        "message_id": int(target),
                        "allow_sending_without_reply": True,
                    }
                except (TypeError, ValueError):
                    pass
            await self._api("sendMessage", **payload)

    async def send_typing(self, channel_id: str) -> None:
        await self._api("sendChatAction", chat_id=channel_id, action="typing")

    # ── offset persistence ────────────────────────────────────────────────
    def _load_offset(self) -> int:
        path = self._offset_path
        if path is None:
            return 0
        try:
            return max(0, int(path.read_text(encoding="utf-8").strip()))
        except (OSError, ValueError):
            return 0

    def _save_offset(self) -> None:
        path = self._offset_path
        if path is None or not self._offset_dirty:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(str(self._offset), encoding="utf-8")
            tmp.replace(path)
            self._offset_dirty = 0
        except OSError:
            logger.debug("could not persist telegram offset", exc_info=True)


def create_plugin() -> TelegramGateway:
    return TelegramGateway()
