"""Per-chat sessions: isolation, reset/status events, and persistence."""

from __future__ import annotations

from python_core.core.event_bus import EventBus
from python_core.orchestrator import Orchestrator


async def test_sessions_isolate_conversations(kernel) -> None:
    kernel.provider.script = [
        {"thought": "t", "action": "respond", "response": "answer A"},
        {"thought": "t", "action": "respond", "response": "answer B"},
    ]

    first = await kernel.orchestrator.handle_task("same prompt", session_id="chat:a")
    second = await kernel.orchestrator.handle_task("same prompt", session_id="chat:b")

    assert first["response"] == "answer A"
    assert second["response"] == "answer B"
    assert [m["content"] for m in kernel.orchestrator._sessions["chat:a"]] == [
        "same prompt",
        "answer A",
    ]
    assert [m["content"] for m in kernel.orchestrator._sessions["chat:b"]] == [
        "same prompt",
        "answer B",
    ]
    assert kernel.orchestrator.conversation == []


async def test_session_status_and_reset_events(kernel) -> None:
    kernel.provider.script = [{"thought": "t", "action": "respond", "response": "hi"}]
    await kernel.orchestrator.handle_task("hello", session_id="chat:a")

    status = await kernel.bus.request("session:status", {"session_id": "chat:a"})
    assert status["session_id"] == "chat:a"
    assert status["messages"] == 2
    assert status["turns"] == 1

    results = await kernel.bus.emit("session:reset", {"session_id": "chat:a"})
    assert any(result.result and result.result.get("reset") for result in results)

    status = await kernel.bus.request("session:status", {"session_id": "chat:a"})
    assert status["messages"] == 0


async def test_main_session_is_backwards_compatible(kernel) -> None:
    kernel.provider.script = [{"thought": "t", "action": "respond", "response": "main"}]
    await kernel.orchestrator.handle_task("hello")

    assert [m["content"] for m in kernel.orchestrator.conversation] == ["hello", "main"]
    assert kernel.orchestrator._sessions["main"] is kernel.orchestrator.conversation


async def test_sessions_persist_and_reload(kernel) -> None:
    kernel.provider.script = [
        {"thought": "t", "action": "respond", "response": "first"},
        {"thought": "t", "action": "respond", "response": "second"},
    ]
    await kernel.orchestrator.handle_task("one", session_id="telegram:dm:1")
    await kernel.orchestrator.handle_task("two", session_id="discord:dm:2")

    path = kernel.config.data_dir / "sessions.json"
    assert path.is_file()

    reloaded = Orchestrator(
        bus=EventBus(),
        config=kernel.config,
        tools=kernel.ctx.tools,
        provider=kernel.provider,
    )
    assert [m["content"] for m in reloaded._sessions["telegram:dm:1"]] == ["one", "first"]
    assert [m["content"] for m in reloaded._sessions["discord:dm:2"]] == ["two", "second"]


async def test_sessions_are_trimmed_to_history_limit(kernel) -> None:
    kernel.config.session_history_limit = 4
    kernel.provider.script = [
        {"thought": "t", "action": "respond", "response": f"r{i}"} for i in range(4)
    ]
    for i in range(4):
        await kernel.orchestrator.handle_task(f"q{i}", session_id="chat:trim")

    history = kernel.orchestrator._sessions["chat:trim"]
    assert len(history) == 4
    assert [m["content"] for m in history] == ["q2", "r2", "q3", "r3"]


async def test_sessions_are_lru_bounded(kernel) -> None:
    kernel.config.max_sessions = 3
    kernel.provider.script = [
        {"thought": "t", "action": "respond", "response": f"r{i}"} for i in range(3)
    ]
    for sid in ("chat:a", "chat:b", "chat:c"):
        await kernel.orchestrator.handle_task(sid, session_id=sid)

    assert "chat:a" not in kernel.orchestrator._sessions
    assert "chat:b" in kernel.orchestrator._sessions
    assert "chat:c" in kernel.orchestrator._sessions


async def test_semantic_cache_is_scoped_per_session(kernel) -> None:
    kernel.orchestrator.cache._config.enable_semantic_cache = True
    kernel.provider.script = [{"thought": "t", "action": "respond", "response": "cached A"}]

    first = await kernel.orchestrator.handle_task("same question", session_id="chat:a")
    second = await kernel.orchestrator.handle_task("same question", session_id="chat:b")
    third = await kernel.orchestrator.handle_task("same question", session_id="chat:a")

    assert first["cached"] is False
    assert second["cached"] is False  # other chat must not see chat A's answer
    assert third["cached"] is True
    assert third["response"] == "cached A"
