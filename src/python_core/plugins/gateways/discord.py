"""Discord gateway — OpenClaw-style DMs, guild channels, slash commands.

Uses the official ``discord.py`` gateway (optional extra: ``uv sync --extra
discord``). Without the library or a token the plugin stays dormant, so the
rest of echo-agent keeps running.

Access model
------------
* DMs default to **pairing** (explicit approval from the CLI).
* Guilds default to an **allowlist** (``discord_guilds``), optionally narrowed
  by channel IDs, sender IDs, and @mention gating.
* Slash commands (``/ask``, ``/status``, ``/reset`` …) use the same policy as
  ordinary messages; text commands (``!ask``) are available too.

Replies are chunked at Discord's 2 000-character limit, native reply threading
is supported via ``[[reply_to_current]]`` / ``[[reply_to:<id>]]``, and the
client reconnects with exponential backoff.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from python_core.core.interfaces import KernelContext

from .base import (
    DISCORD_CHUNK,
    Admission,
    GatewayInterface,
    chunk_text,
    id_allowed,
    parse_reply_tags,
)

logger = logging.getLogger("echo.gateway.discord")

try:
    import discord  # type: ignore
except Exception:  # noqa: BLE001 — optional dependency
    discord = None  # type: ignore[assignment]

MAX_MESSAGE = DISCORD_CHUNK


class DiscordGateway(GatewayInterface):
    name = "discord"
    version = "0.2.0"
    capabilities = ["gateway", "discord", "pairing", "sessions", "slash-commands"]

    def __init__(self, token: str = "") -> None:
        super().__init__()
        self.token = token
        self.config: Any = None
        self._client: Any = None
        self._tree: Any = None
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._bot_user_id: int | None = None
        self._activation: dict[str, bool] = {}
        self._introduced: set[int] = set()
        self._commands_synced = False

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start(self, ctx: KernelContext) -> None:
        self._ctx = ctx
        self.config = ctx.config
        if not self.token:
            self.token = ctx.config.discord_token
        if not self.token:
            ctx.logger.info("discord gateway disabled (no token)")
            return
        if discord is None:
            ctx.logger.warning("discord gateway disabled: install with `uv sync --extra discord`")
            return
        if self.config.discord_dm_policy == "allowlist" and not self.config.discord_allow_from:
            ctx.logger.warning("discord dm_policy=allowlist but discord_allow_from is empty — all DMs blocked")
        if self.config.discord_group_policy == "allowlist" and not self.config.discord_guilds:
            ctx.logger.warning("discord group_policy=allowlist but discord_guilds is empty — all guilds blocked")

        self.bind_pairing(ctx)
        self.bind_bus(ctx.bus)
        self._typing_enabled = bool(ctx.config.gateway_typing)
        self._running = True
        self._task = asyncio.create_task(self._run(), name="echo-discord")
        self.start_notification_pump()
        ctx.logger.info("discord gateway connecting")

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
        await self._close_client()

    async def _close_client(self) -> None:
        client = self._client
        self._client = None
        self._tree = None
        if client is not None and not client.is_closed():
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                logger.debug("discord client close failed", exc_info=True)

    async def _run(self) -> None:
        backoff = 2.0
        while self._running:
            try:
                client = self._build_client()
                self._client = client
                await client.start(self.token)
                backoff = 2.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                if discord is not None and isinstance(exc, discord.LoginFailure):
                    logger.error("discord rejected the bot token")
                    self._running = False
                    break
                logger.warning("discord connection failed: %s; retrying in %.0fs", exc, backoff)
            finally:
                await self._close_client()
            if self._running:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    def _build_client(self) -> Any:
        intents = discord.Intents.default()
        intents.guilds = True
        intents.guild_messages = True
        intents.dm_messages = True
        intents.message_content = True  # privileged; enable it in the Discord portal

        client = discord.Client(intents=intents)
        tree = discord.app_commands.CommandTree(client)
        self._tree = tree

        @client.event
        async def on_ready() -> None:  # noqa: ANN202 — discord.py callback style
            self._bot_user_id = client.user.id if client.user else None
            logger.info("discord gateway connected as %s", client.user)
            if self.config.discord_slash_commands and not self._commands_synced:
                try:
                    await tree.sync()
                    self._commands_synced = True
                except Exception as exc:  # noqa: BLE001 — commands are optional
                    logger.warning("discord slash-command sync failed: %s", exc)
                    logger.debug("discord slash-command sync traceback", exc_info=True)

        @client.event
        async def on_message(message: Any) -> None:  # noqa: ANN001
            try:
                await self._on_message(message)
            except Exception as exc:  # noqa: BLE001 — one bad message must not kill the client
                logger.warning("discord message handling failed: %s", exc)
                logger.debug("discord message traceback", exc_info=True)

        @client.event
        async def on_guild_join(guild: Any) -> None:  # noqa: ANN001
            try:
                await self._on_guild_join(guild)
            except Exception:  # noqa: BLE001
                logger.debug("discord join intro failed", exc_info=True)

        @client.event
        async def on_disconnect() -> None:
            logger.warning("discord gateway disconnected; reconnecting")

        if self.config.discord_slash_commands:
            self._register_commands(tree)
        return client

    # ── slash commands ─────────────────────────────────────────────────────
    def _register_commands(self, tree: Any) -> None:
        @tree.command(name="ask", description="Give echo-agent a task")
        @discord.app_commands.describe(prompt="What should echo-agent do?")
        async def ask(interaction: Any, prompt: str) -> None:
            await self._slash_ask(interaction, prompt)

        @tree.command(name="status", description="Show session and model status")
        async def status(interaction: Any) -> None:
            await self._slash_status(interaction)

        @tree.command(name="reset", description="Clear this channel's session")
        async def reset(interaction: Any) -> None:
            await self._slash_reset(interaction)

        @tree.command(name="whoami", description="Show your Discord IDs")
        async def whoami(interaction: Any) -> None:
            await self._slash_whoami(interaction)

        @tree.command(name="activation", description="Toggle @mention requirement in this channel")
        @discord.app_commands.describe(mode="mention (default) or always")
        async def activation(interaction: Any, mode: str) -> None:
            await self._slash_activation(interaction, mode)

        @tree.command(name="help", description="List echo-agent commands")
        async def help_command(interaction: Any) -> None:
            await self._respond_ephemeral(interaction, self.help_text("Discord", "/"))

    async def _slash_context(self, interaction: Any) -> tuple[Admission, bool, str, str]:
        is_dm = interaction.guild is None
        channel_id = str(interaction.channel_id or "")
        sender_id = interaction.user.id
        username = str(interaction.user)
        if is_dm:
            admission = self.admit_dm(
                policy=self.config.discord_dm_policy,
                allow_from=self.config.discord_allow_from,
                sender_id=sender_id,
                username=username,
                chat_id=channel_id,
            )
        else:
            guild_id = str(interaction.guild_id or "")
            admission = self.admit_group(
                policy=self.config.discord_group_policy,
                allowed_groups=self.config.discord_guilds,
                group_id=guild_id,
                sender_allowlist=self.config.discord_group_allow_from or self.config.discord_allow_from,
                sender_id=sender_id,
                username=username,
            )
            if admission.allowed and self.config.discord_channels:
                parent_id = getattr(interaction.channel, "parent_id", None)
                if not id_allowed(self.config.discord_channels, channel_id) and not (
                    parent_id and id_allowed(self.config.discord_channels, parent_id)
                ):
                    admission = Admission(False, "channel not in allowlist")
        session_id = self._session_id(
            channel_id, is_dm=is_dm, guild_id=str(interaction.guild_id or "") or None
        )
        return admission, is_dm, channel_id, session_id

    async def _slash_ask(self, interaction: Any, prompt: str) -> None:
        admission, is_dm, channel_id, session_id = await self._slash_context(interaction)
        try:
            await interaction.response.defer(thinking=True, ephemeral=True)
        except Exception:  # noqa: BLE001 — already acknowledged / expired
            logger.debug("discord interaction defer failed", exc_info=True)
        if not admission.allowed:
            await self._slash_denied(interaction, admission, is_dm)
            return
        text = (prompt or "").strip()
        if not text:
            await self._respond_ephemeral(interaction, "usage: /ask <text>")
            return
        await self._respond_ephemeral(interaction, "🧠 working…")
        self.emit_task(
            self._bus,
            text,
            channel_id,
            source_user=str(interaction.user),
            user_id=str(interaction.user.id),
            session_id=session_id,
            chat_type="dm" if is_dm else "group",
            chat_id=channel_id,
        )

    async def _slash_status(self, interaction: Any) -> None:
        admission, _is_dm, _channel_id, session_id = await self._slash_context(interaction)
        if not admission.allowed:
            await self._slash_denied(interaction, admission, interaction.guild is None)
            return
        status = None
        if self._bus is not None:
            status = await self._bus.request("session:status", {"session_id": session_id}, timeout=5.0)
        status = status or {}
        await self._respond_ephemeral(
            interaction,
            f"session: {session_id}\n"
            f"messages: {status.get('messages', '?')}\n"
            f"model: {getattr(self.config, 'llm_model', '?')}\n"
            f"dm policy: {self.config.discord_dm_policy}\n"
            f"group policy: {self.config.discord_group_policy}",
        )

    async def _slash_reset(self, interaction: Any) -> None:
        admission, _is_dm, _channel_id, session_id = await self._slash_context(interaction)
        if not admission.allowed:
            await self._slash_denied(interaction, admission, interaction.guild is None)
            return
        if self._bus is not None:
            await self._bus.emit("session:reset", {"session_id": session_id}, source="discord")
        await self._respond_ephemeral(interaction, "🧹 session cleared for this channel.")

    async def _slash_whoami(self, interaction: Any) -> None:
        await self._respond_ephemeral(
            interaction,
            f"user_id: {interaction.user.id}\n"
            f"username: {interaction.user}\n"
            f"channel_id: {interaction.channel_id}\n"
            f"guild_id: {interaction.guild_id or '— (DM)'}",
        )

    async def _slash_activation(self, interaction: Any, mode: str) -> None:
        admission, is_dm, channel_id, _session_id = await self._slash_context(interaction)
        if is_dm:
            await self._respond_ephemeral(interaction, "Activation only applies to guild channels.")
            return
        if not admission.allowed:
            await self._slash_denied(interaction, admission, False)
            return
        value = (mode or "").strip().lower()
        if value in {"always", "off", "false", "0"}:
            self._activation[channel_id] = False
            await self._respond_ephemeral(interaction, "🔔 mention requirement off for this channel.")
        elif value in {"mention", "on", "true", "1"}:
            self._activation[channel_id] = True
            await self._respond_ephemeral(interaction, "🔕 mention requirement on for this channel.")
        else:
            await self._respond_ephemeral(interaction, "usage: /activation always | mention")

    async def _respond_ephemeral(self, interaction: Any, text: str) -> None:
        try:
            if interaction.response.is_done():
                await interaction.followup.send(text, ephemeral=True)
            else:
                await interaction.response.send_message(text, ephemeral=True)
        except Exception:  # noqa: BLE001 — interaction may have expired
            logger.debug("discord interaction response failed", exc_info=True)

    async def _slash_denied(self, interaction: Any, admission: Admission, is_dm: bool) -> None:
        if admission.reason == "pairing required" and admission.pairing_code and is_dm:
            await self._respond_ephemeral(
                interaction,
                self.pairing_prompt(
                    admission.pairing_code,
                    platform="Discord",
                    user_id=interaction.user.id,
                    ttl=int(self.config.gateway_pairing_ttl),
                ),
            )
        else:
            await self._respond_ephemeral(interaction, f"⛔ not authorized: {admission.reason}")

    # ── inbound messages ───────────────────────────────────────────────────
    async def _on_message(self, message: Any) -> None:
        author = message.author
        if author.bot and not bool(self.config.discord_allow_bots):
            return
        if self._bot_user_id is not None and author.id == self._bot_user_id:
            return
        if message.guild is None and discord is not None and isinstance(message.channel, discord.GroupChannel):
            if not bool(self.config.discord_group_dm_enabled):
                return

        content = str(message.content or "").strip()
        if not content and message.attachments:
            names = ", ".join(a.filename for a in message.attachments[:3])
            content = f"[attachment: {names}]"
        if not content:
            return

        admission, is_dm, channel_id, session_id = self._context_for_message(message)
        if not admission.allowed:
            await self._notify_denied(channel_id, admission, is_dm, author.id)
            return

        prefix = str(self.config.discord_command_prefix or "!")
        command, args = self._parse_prefix(content, prefix)
        if command is not None:
            await self._handle_prefix(
                command=command,
                args=args,
                message=message,
                channel_id=channel_id,
                is_dm=is_dm,
                session_id=session_id,
            )
            return

        if not is_dm:
            mentioned = self._mentioned(message)
            if not mentioned:
                if self._require_mention(channel_id):
                    return
                if self.config.discord_ignore_other_mentions and self._mentions_others(message):
                    return
            else:
                content = self._strip_bot_mentions(content).strip() or "hello"

        quoted = self._quoted_context(message)
        if quoted:
            content = f"{quoted}\n{content}"

        self.emit_task(
            self._bus,
            content,
            channel_id,
            source_user=str(author),
            user_id=str(author.id),
            session_id=session_id,
            message_id=str(message.id),
            chat_type="dm" if is_dm else "group",
            chat_id=channel_id,
        )

    def _context_for_message(self, message: Any) -> tuple[Admission, bool, str, str]:
        is_dm = message.guild is None
        channel_id = str(message.channel.id)
        sender_id = message.author.id
        username = str(message.author)
        if is_dm:
            admission = self.admit_dm(
                policy=self.config.discord_dm_policy,
                allow_from=self.config.discord_allow_from,
                sender_id=sender_id,
                username=username,
                chat_id=channel_id,
            )
        else:
            guild_id = str(message.guild.id)
            admission = self.admit_group(
                policy=self.config.discord_group_policy,
                allowed_groups=self.config.discord_guilds,
                group_id=guild_id,
                sender_allowlist=self.config.discord_group_allow_from or self.config.discord_allow_from,
                sender_id=sender_id,
                username=username,
            )
            if admission.allowed and self.config.discord_channels:
                parent_id = getattr(message.channel, "parent_id", None)
                if not id_allowed(self.config.discord_channels, channel_id) and not (
                    parent_id and id_allowed(self.config.discord_channels, parent_id)
                ):
                    admission = Admission(False, "channel not in allowlist")
        session_id = self._session_id(
            channel_id, is_dm=is_dm, guild_id=str(message.guild.id) if message.guild else None
        )
        return admission, is_dm, channel_id, session_id

    async def _notify_denied(
        self, channel_id: str, admission: Admission, is_dm: bool, user_id: Any
    ) -> None:
        if not is_dm:
            return
        if admission.reason == "pairing required" and admission.pairing_code:
            await self.send_reply(
                channel_id,
                self.pairing_prompt(
                    admission.pairing_code,
                    platform="Discord",
                    user_id=user_id,
                    ttl=int(self.config.gateway_pairing_ttl),
                ),
            )
        elif admission.reason == "sender not in allowlist":
            await self.send_reply(
                channel_id,
                f"⛔ Not authorized. Your Discord user id is {user_id}.\n"
                "Ask the operator to add it to discord_allow_from.",
            )

    # ── prefix commands ────────────────────────────────────────────────────
    @staticmethod
    def _parse_prefix(text: str, prefix: str) -> tuple[str | None, str]:
        if not prefix or not text.lower().startswith(prefix.lower()):
            return None, text
        rest = text[len(prefix):].strip()
        if not rest:
            return "", ""
        name, _, args = rest.partition(" ")
        return name.lower(), args.strip()

    async def _handle_prefix(
        self,
        *,
        command: str,
        args: str,
        message: Any,
        channel_id: str,
        is_dm: bool,
        session_id: str,
    ) -> None:
        if command in {"ask", "echo", "task"}:
            if not args:
                await self.send_reply(channel_id, f"usage: {self.config.discord_command_prefix}{command} <text>")
                return
            self.emit_task(
                self._bus,
                args,
                channel_id,
                source_user=str(message.author),
                user_id=str(message.author.id),
                session_id=session_id,
                message_id=str(message.id),
                chat_type="dm" if is_dm else "group",
                chat_id=channel_id,
            )
            return
        if command in {"help", "start"}:
            await self.send_reply(channel_id, self.help_text("Discord", self.config.discord_command_prefix))
            return
        if command == "whoami":
            await self.send_reply(
                channel_id,
                f"user_id: {message.author.id}\nusername: {message.author}\n"
                f"channel_id: {channel_id}\nguild_id: {message.guild.id if message.guild else '— (DM)'}",
            )
            return
        if command == "status":
            status = await self._bus.request("session:status", {"session_id": session_id}, timeout=5.0) if self._bus else None
            status = status or {}
            await self.send_reply(
                channel_id,
                f"session: {session_id}\nmessages: {status.get('messages', '?')}\n"
                f"model: {getattr(self.config, 'llm_model', '?')}",
            )
            return
        if command in {"reset", "new"}:
            if self._bus is not None:
                await self._bus.emit("session:reset", {"session_id": session_id}, source="discord")
            await self.send_reply(channel_id, "🧹 session cleared for this channel.")
            return
        if command == "activation":
            if is_dm:
                await self.send_reply(channel_id, "Activation only applies to guild channels.")
                return
            value = args.strip().lower()
            if value in {"always", "off", "false", "0"}:
                self._activation[channel_id] = False
                await self.send_reply(channel_id, "🔔 mention requirement off for this channel.")
            elif value in {"mention", "on", "true", "1"}:
                self._activation[channel_id] = True
                await self.send_reply(channel_id, "🔕 mention requirement on for this channel.")
            else:
                await self.send_reply(channel_id, "usage: !activation always | mention")
            return
        await self.send_reply(channel_id, f"unknown command: {command}\n\n" + self.help_text("Discord", self.config.discord_command_prefix))

    # ── mention / session helpers ──────────────────────────────────────────
    def _require_mention(self, channel_id: str) -> bool:
        return self._activation.get(channel_id, bool(self.config.discord_require_mention))

    def _mentioned(self, message: Any) -> bool:
        bot_id = self._bot_user_id or (self._client.user.id if self._client and self._client.user else None)
        if bot_id is None:
            return False
        if any(getattr(user, "id", None) == bot_id for user in message.mentions):
            return True
        content = str(message.content or "")
        if re.search(rf"<@!?{bot_id}>", content):
            return True
        reference = getattr(message, "reference", None)
        resolved = getattr(reference, "resolved", None) if reference is not None else None
        author = getattr(resolved, "author", None)
        return bool(author and getattr(author, "id", None) == bot_id)

    def _mentions_others(self, message: Any) -> bool:
        bot_id = self._bot_user_id
        if getattr(message, "mention_everyone", False):
            return True
        if getattr(message, "role_mentions", None):
            return True
        return any(getattr(user, "id", None) != bot_id for user in message.mentions)

    def _strip_bot_mentions(self, text: str) -> str:
        bot_id = self._bot_user_id
        if bot_id is None:
            return text
        return re.sub(rf"<@!?{bot_id}>", "", text).strip()

    def _quoted_context(self, message: Any) -> str:
        reference = getattr(message, "reference", None)
        resolved = getattr(reference, "resolved", None) if reference is not None else None
        content = str(getattr(resolved, "content", "") or "").strip()
        if not content:
            return ""
        author = getattr(resolved, "author", None)
        who = str(author) if author is not None else "someone"
        return f"[replying to {who}: {content[:300]}]"

    @staticmethod
    def _session_id(channel_id: str, *, is_dm: bool, guild_id: str | None) -> str:
        if is_dm:
            return f"discord:dm:{channel_id}"
        return f"discord:guild:{guild_id or 'unknown'}:channel:{channel_id}"

    # ── join introductions ─────────────────────────────────────────────────
    async def _on_guild_join(self, guild: Any) -> None:
        if not bool(self.config.discord_join_intro) or guild.id in self._introduced:
            return
        if self.config.discord_group_policy == "allowlist" and not id_allowed(self.config.discord_guilds, guild.id):
            return
        channel = guild.system_channel
        me = guild.me
        if channel is None or me is None or not channel.permissions_for(me).send_messages:
            channel = next(
                (
                    c for c in guild.text_channels
                    if me is not None and c.permissions_for(me).send_messages
                ),
                None,
            )
        if channel is None:
            return
        try:
            await channel.send(
                "Hi — I'm echo-agent. Mention me in this server or use `/ask` to give me a task.",
                allowed_mentions=discord.AllowedMentions.none(),
                suppress_embeds=True,
            )
            self._introduced.add(guild.id)
        except Exception:  # noqa: BLE001
            logger.debug("discord join intro send failed", exc_info=True)

    # ── outbound ───────────────────────────────────────────────────────────
    async def _resolve_channel(self, channel_id: str) -> Any:
        client = self._client
        if client is None or client.is_closed():
            return None
        try:
            cid = int(channel_id)
        except (TypeError, ValueError):
            return None
        channel = client.get_channel(cid)
        if channel is None:
            try:
                channel = await client.fetch_channel(cid)
            except Exception:  # noqa: BLE001
                return None
        return channel

    async def send_reply(self, channel_id: str, text: str) -> None:
        clean, explicit_target = parse_reply_tags(text)
        if not clean:
            return
        target = explicit_target
        if target == "current" or (target is None and self.config.gateway_reply_to_mode != "off"):
            target = self._reply_targets.get(str(channel_id))
        self._reply_targets.pop(str(channel_id), None)

        channel = await self._resolve_channel(channel_id)
        if channel is None:
            logger.warning("discord channel %s unavailable; dropping reply", channel_id)
            return

        chunks = chunk_text(clean, MAX_MESSAGE, mode="newline")
        for index, chunk in enumerate(chunks):
            reference = None
            if index == 0 and target:
                try:
                    guild_id = getattr(getattr(channel, "guild", None), "id", None)
                    reference = discord.MessageReference(
                        message_id=int(target),
                        channel_id=int(channel_id),
                        guild_id=guild_id,
                        fail_if_not_exists=False,
                    )
                except (TypeError, ValueError):
                    reference = None
            kwargs: dict[str, Any] = {
                "allowed_mentions": discord.AllowedMentions(
                    everyone=False, roles=False, users=True, replied_user=False
                ),
                "suppress_embeds": bool(self.config.discord_suppress_embeds),
            }
            if reference is not None:
                kwargs.update(reference=reference, mention_author=False)
            try:
                await channel.send(chunk, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if reference is not None:
                    # The referenced message may have been deleted; send plainly.
                    kwargs.pop("reference", None)
                    kwargs.pop("mention_author", None)
                    await channel.send(chunk, **kwargs)
                else:
                    logger.warning("discord send failed: %s", exc)

    async def send_typing(self, channel_id: str) -> None:
        channel = await self._resolve_channel(channel_id)
        if channel is None:
            return
        async with channel.typing():
            await asyncio.sleep(0.05)


def create_plugin() -> DiscordGateway:
    return DiscordGateway()
