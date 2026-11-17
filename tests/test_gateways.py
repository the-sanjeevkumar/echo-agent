"""Gateway access control, pairing, chunking, and platform routing."""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from python_core.core.config import EchoConfig
from python_core.core.event_bus import EventBus
from python_core.core.interfaces import KernelContext
from python_core.core.tool_registry import ToolRegistry
from python_core.plugins.gateways.base import (
    chunk_text,
    id_allowed,
    normalize_allowlist,
    parse_reply_tags,
    parse_slash_command,
    sender_allowed,
)
from python_core.plugins.gateways.discord import DiscordGateway
from python_core.plugins.gateways.pairing import PairingStore
from python_core.plugins.gateways.telegram import TelegramGateway


# ── helpers ─────────────────────────────────────────────────────────────────
def make_ctx(tmp_path, **overrides: Any) -> KernelContext:
    config = EchoConfig(
        data_dir=tmp_path / "data",
        gateway_typing=False,
        session_persistence=False,
        **overrides,
    )
    config.ensure_dirs()
    bus = EventBus()
    return KernelContext(
        bus=bus,
        config=config,
        tools=ToolRegistry(bus),
        data_dir=config.data_dir,
    )


def collect_incoming(ctx: KernelContext) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    async def _collect(payload: Any) -> None:
        events.append(payload)

    ctx.bus.subscribe("task:incoming", _collect, label="test.gateway.incoming")
    return events


class RecordingTelegram(TelegramGateway):
    def __init__(self) -> None:
        super().__init__("test-token")
        self.replies: list[tuple[str, str]] = []

    async def send_reply(self, channel_id: str, text: str) -> None:
        self.replies.append((str(channel_id), text))


class RecordingDiscord(DiscordGateway):
    def __init__(self) -> None:
        super().__init__("test-token")
        self.replies: list[tuple[str, str]] = []

    async def send_reply(self, channel_id: str, text: str) -> None:
        self.replies.append((str(channel_id), text))


def tg_message(
    text: str,
    *,
    user_id: int = 42,
    chat_id: int = 42,
    chat_type: str = "private",
    username: str = "alice",
    message_id: int = 10,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "message_id": message_id,
        "chat": {"id": chat_id, "type": chat_type},
        "from": {"id": user_id, "username": username, "is_bot": False},
        "text": text,
        **extra,
    }


def configured_telegram(ctx: KernelContext) -> RecordingTelegram:
    gateway = RecordingTelegram()
    gateway._configure(ctx)
    gateway.bind_bus(ctx.bus)
    gateway._typing_enabled = False
    gateway._bot_id = 999
    gateway._bot_username = "echo_bot"
    return gateway


class FakeUser:
    def __init__(self, user_id: int, *, bot: bool = False) -> None:
        self.id = user_id
        self.bot = bot

    def __str__(self) -> str:
        return f"user{self.id}"


def dc_message(
    text: str,
    *,
    user_id: int = 7,
    channel_id: int = 70,
    guild_id: int | None = None,
    mentions: list[FakeUser] | None = None,
    reference: Any = None,
    message_id: int = 77,
) -> Any:
    channel = SimpleNamespace(id=channel_id, parent_id=None)
    guild = SimpleNamespace(id=guild_id) if guild_id is not None else None
    return SimpleNamespace(
        id=message_id,
        author=FakeUser(user_id),
        channel=channel,
        guild=guild,
        content=text,
        mentions=mentions or [],
        role_mentions=[],
        mention_everyone=False,
        reference=reference,
        attachments=[],
    )


def configured_discord(ctx: KernelContext) -> RecordingDiscord:
    gateway = RecordingDiscord()
    gateway.config = ctx.config
    gateway.bind_pairing(ctx)
    gateway.bind_bus(ctx.bus)
    gateway._typing_enabled = False
    gateway._bot_user_id = 999
    return gateway


# ── text helpers ────────────────────────────────────────────────────────────
def test_chunk_text_respects_limit_and_boundaries() -> None:
    text = "First paragraph.\n\n" + ("word " * 300) + "\n\nlast line"
    chunks = chunk_text(text, 120)
    assert chunks
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert "First paragraph." in chunks[0]
    assert "last line" in chunks[-1]


def test_chunk_text_short_text_is_one_chunk() -> None:
    assert chunk_text("hello", 100) == ["hello"]
    assert chunk_text("", 100) == []


def test_parse_reply_tags() -> None:
    clean, target = parse_reply_tags("hello [[reply_to_current]]")
    assert clean == "hello"
    assert target == "current"

    clean, target = parse_reply_tags("answer [[reply_to:12345]]!")
    assert clean == "answer !"
    assert target == "12345"

    clean, target = parse_reply_tags("plain")
    assert clean == "plain"
    assert target is None


def test_allowlist_normalization_and_matching() -> None:
    assert normalize_allowlist(["tg:42", "@Alice", "<@!77>", "user:99", "42"]) == [
        "42",
        "Alice",
        "77",
        "99",
    ]
    assert sender_allowed(["tg:42"], 42, "alice")
    assert sender_allowed(["@Alice"], 100, "alice")
    assert not sender_allowed(["42"], 100, "alice")
    assert not sender_allowed([], 42, "alice")
    assert sender_allowed(["*"], 999, "nobody")
    assert id_allowed(["1", "2"], 2)
    assert not id_allowed([], 2)


def test_parse_slash_command() -> None:
    assert parse_slash_command("/ask hello world", "echo_bot") == ("ask", "hello world")
    assert parse_slash_command("/ask@echo_bot hi", "echo_bot") == ("ask", "hi")
    assert parse_slash_command("/ask@other_bot hi", "echo_bot") == ("", "")
    assert parse_slash_command("hello", "echo_bot") == (None, "hello")


# ── pairing store ───────────────────────────────────────────────────────────
def test_pairing_store_lifecycle(tmp_path) -> None:
    now = [1000.0]
    path = tmp_path / "pairing.json"
    store = PairingStore(path, ttl=60, max_pending=2, clock=lambda: now[0])

    created = store.request("telegram", 42, username="alice", chat_id="42")
    assert created["status"] == "created"
    code = created["code"]
    assert len(code) == 8
    assert re.fullmatch(r"[A-HJ-NP-Z2-9]{8}", code)
    assert store.is_approved("telegram", 42) is False

    pending = store.request("telegram", 42, username="alice")
    assert pending["status"] == "pending"
    assert pending["code"] == code

    approved = store.approve("telegram", code.lower())
    assert approved is not None
    assert approved["sender_id"] == "42"
    assert store.is_approved("telegram", 42) is True
    assert store.list_pending("telegram") == []
    assert store.list_approved("telegram")[0]["sender_id"] == "42"

    # File-backed: a fresh instance (e.g. the CLI) sees the approval.
    assert PairingStore(path, ttl=60).is_approved("telegram", 42) is True

    assert store.revoke("telegram", 42) is True
    assert store.is_approved("telegram", 42) is False


def test_pairing_store_expiry_and_cap(tmp_path) -> None:
    now = [1000.0]
    store = PairingStore(tmp_path / "pairing.json", ttl=10, max_pending=2, clock=lambda: now[0])
    first = store.request("discord", 1)
    store.request("discord", 2)
    full = store.request("discord", 3)
    assert first["status"] == "created"
    assert full["status"] == "full"

    now[0] += 11
    assert store.list_pending("discord") == []
    assert store.approve("discord", first["code"]) is None
    again = store.request("discord", 3)
    assert again["status"] == "created"
    assert again["code"] != first["code"]


# ── Telegram ────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_telegram_dm_pairing_then_approval(tmp_path) -> None:
    ctx = make_ctx(tmp_path)
    gateway = configured_telegram(ctx)
    events = collect_incoming(ctx)

    await gateway._handle_message(tg_message("hello"))
    assert events == []
    assert any("Pairing code" in text for _, text in gateway.replies)

    pending = ctx.service("pairing").list_pending("telegram")
    assert len(pending) == 1
    assert ctx.service("pairing").approve("telegram", pending[0]["code"])

    await gateway._handle_message(tg_message("hello again"))
    await asyncio.sleep(0.05)
    assert len(events) == 1
    payload = events[0]
    assert payload["text"] == "hello again"
    assert payload["session_id"] == "telegram:dm:42"
    assert payload["chat_type"] == "dm"
    assert payload["user_id"] == "42"


@pytest.mark.asyncio
async def test_telegram_allowlist_blocks_and_allows(tmp_path) -> None:
    ctx = make_ctx(
        tmp_path,
        telegram_dm_policy="allowlist",
        telegram_allow_from=["tg:42"],
    )
    gateway = configured_telegram(ctx)
    events = collect_incoming(ctx)

    await gateway._handle_message(tg_message("hi", user_id=100, chat_id=100))
    assert events == []
    assert any("Not authorized" in text for _, text in gateway.replies)

    await gateway._handle_message(tg_message("hi"))
    await asyncio.sleep(0.05)
    assert len(events) == 1


@pytest.mark.asyncio
async def test_telegram_group_mention_gating_and_sessions(tmp_path) -> None:
    ctx = make_ctx(tmp_path, telegram_group_policy="open", telegram_require_mention=True)
    gateway = configured_telegram(ctx)
    events = collect_incoming(ctx)

    await gateway._handle_message(
        tg_message("hello", chat_id=-100, chat_type="supergroup", message_id=20)
    )
    assert events == []

    await gateway._handle_message(
        tg_message("@echo_bot hello", chat_id=-100, chat_type="supergroup", message_id=21)
    )
    await asyncio.sleep(0.05)
    assert len(events) == 1
    assert events[0]["text"] == "hello"
    assert events[0]["session_id"] == "telegram:chat:-100"
    assert events[0]["chat_type"] == "group"


@pytest.mark.asyncio
async def test_telegram_group_topic_session_and_activation(tmp_path) -> None:
    ctx = make_ctx(tmp_path, telegram_group_policy="open", telegram_require_mention=True)
    gateway = configured_telegram(ctx)
    events = collect_incoming(ctx)

    await gateway._handle_message(
        tg_message(
            "/activation always", chat_id=-100, chat_type="supergroup", message_id=30
        )
    )
    assert any("mention requirement off" in text for _, text in gateway.replies)

    await gateway._handle_message(
        tg_message(
            "no mention now",
            chat_id=-100,
            chat_type="supergroup",
            message_id=31,
            message_thread_id=5,
        )
    )
    await asyncio.sleep(0.05)
    assert len(events) == 1
    assert events[0]["session_id"] == "telegram:chat:-100:topic:5"


@pytest.mark.asyncio
async def test_telegram_commands(tmp_path) -> None:
    ctx = make_ctx(tmp_path, telegram_dm_policy="open")
    gateway = configured_telegram(ctx)
    events = collect_incoming(ctx)

    await gateway._handle_message(tg_message("/whoami"))
    assert any("user_id: 42" in text for _, text in gateway.replies)

    await gateway._handle_message(tg_message("/status"))
    assert any("session: telegram:dm:42" in text for _, text in gateway.replies)

    await gateway._handle_message(tg_message("/ask do the thing"))
    await asyncio.sleep(0.05)
    assert events[0]["text"] == "do the thing"

    await gateway._handle_message(tg_message("/reset"))
    assert any("session cleared" in text for _, text in gateway.replies)


@pytest.mark.asyncio
async def test_telegram_send_reply_chunks_and_threads(tmp_path) -> None:
    ctx = make_ctx(tmp_path, telegram_link_preview=False, gateway_reply_to_mode="first")
    gateway = TelegramGateway("test-token")
    gateway._configure(ctx)
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True, "result": {}})

    gateway._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway._reply_targets["42"] = "10"
    try:
        await gateway.send_reply("42", "x" * 5000)
    finally:
        await gateway._http.aclose()

    assert len(calls) == 2
    assert all(len(call["text"]) <= 3900 for call in calls)
    assert calls[0]["reply_parameters"]["message_id"] == 10
    assert "reply_parameters" not in calls[1]
    assert calls[0]["disable_web_page_preview"] is True


# ── Discord ─────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_discord_dm_pairing_then_approval(tmp_path) -> None:
    ctx = make_ctx(tmp_path)
    gateway = configured_discord(ctx)
    events = collect_incoming(ctx)

    await gateway._on_message(dc_message("hello"))
    assert events == []
    assert any("Pairing code" in text for _, text in gateway.replies)

    pending = ctx.service("pairing").list_pending("discord")
    assert len(pending) == 1
    assert ctx.service("pairing").approve("discord", pending[0]["code"])

    await gateway._on_message(dc_message("hello again"))
    await asyncio.sleep(0.05)
    assert len(events) == 1
    assert events[0]["session_id"] == "discord:dm:70"
    assert events[0]["chat_type"] == "dm"


@pytest.mark.asyncio
async def test_discord_guild_allowlist_and_mention_gating(tmp_path) -> None:
    ctx = make_ctx(
        tmp_path,
        discord_group_policy="allowlist",
        discord_guilds=["1"],
        discord_require_mention=True,
    )
    gateway = configured_discord(ctx)
    events = collect_incoming(ctx)

    await gateway._on_message(dc_message("hello", guild_id=2))
    assert events == []

    await gateway._on_message(dc_message("hello", guild_id=1))
    assert events == []

    await gateway._on_message(dc_message("<@999> hello", guild_id=1, mentions=[FakeUser(999)]))
    await asyncio.sleep(0.05)
    assert len(events) == 1
    assert events[0]["text"] == "hello"
    assert events[0]["session_id"] == "discord:guild:1:channel:70"


@pytest.mark.asyncio
async def test_discord_channel_allowlist(tmp_path) -> None:
    ctx = make_ctx(
        tmp_path,
        discord_group_policy="open",
        discord_guilds=["1"],
        discord_channels=["71"],
        discord_require_mention=False,
    )
    gateway = configured_discord(ctx)
    events = collect_incoming(ctx)

    await gateway._on_message(dc_message("hello", guild_id=1, channel_id=70))
    assert events == []

    await gateway._on_message(dc_message("hello", guild_id=1, channel_id=71))
    await asyncio.sleep(0.05)
    assert len(events) == 1


@pytest.mark.asyncio
async def test_discord_prefix_commands_bypass_mention(tmp_path) -> None:
    ctx = make_ctx(
        tmp_path,
        discord_group_policy="open",
        discord_guilds=["1"],
        discord_require_mention=True,
    )
    gateway = configured_discord(ctx)
    events = collect_incoming(ctx)

    await gateway._on_message(dc_message("!whoami", guild_id=1))
    assert any("user_id: 7" in text for _, text in gateway.replies)

    await gateway._on_message(dc_message("!ask summarize", guild_id=1))
    await asyncio.sleep(0.05)
    assert events[0]["text"] == "summarize"


@pytest.mark.asyncio
async def test_discord_send_reply_chunks_and_threads(tmp_path) -> None:
    pytest.importorskip("discord")
    ctx = make_ctx(tmp_path, gateway_reply_to_mode="first", discord_suppress_embeds=True)
    gateway = DiscordGateway("test-token")
    gateway.config = ctx.config

    class FakeChannel:
        guild = None

        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        async def send(self, content: str, **kwargs: Any) -> None:
            self.calls.append((content, kwargs))

    channel = FakeChannel()

    class FakeClient:
        def is_closed(self) -> bool:
            return False

        def get_channel(self, channel_id: int) -> FakeChannel:
            assert channel_id == 70
            return channel

    gateway._client = FakeClient()
    gateway._reply_targets["70"] = "77"
    await gateway.send_reply("70", "y" * 4000)

    assert len(channel.calls) == 3
    assert all(len(content) <= 1900 for content, _ in channel.calls)
    first_text, first_kwargs = channel.calls[0]
    assert first_text
    assert first_kwargs["reference"].message_id == 77
    assert first_kwargs["suppress_embeds"] is True
    assert channel.calls[1][1].get("reference") is None

