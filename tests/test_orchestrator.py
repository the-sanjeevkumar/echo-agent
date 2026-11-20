"""Orchestrator: ReAct loop, token pruning, semantic caching, structured output."""

from __future__ import annotations

import pytest

from python_core.core.config import EchoConfig
from python_core.core.event_bus import EventBus
from python_core.core.tool_registry import ToolRegistry
from python_core.orchestrator import SemanticCache, TokenBudget, _extract_json
from python_core.plugins.llm.openai_compat import LLMProviderError
from tests.conftest import FakeProvider


def _mk(config_kwargs=None) -> tuple[EchoConfig, EventBus, ToolRegistry]:
    config = EchoConfig(data_dir="/tmp/echo-test-orch", enable_semantic_cache=False)
    bus = EventBus()
    registry = ToolRegistry(bus)
    return config, bus, registry


async def test_simple_respond_flow(kernel) -> None:
    kernel.provider.script = [{"thought": "hi", "action": "respond", "response": "the answer"}]
    out = await kernel.orchestrator.handle_task("what up", source="test")
    assert out["response"] == "the answer"
    assert out["steps"] == []
    assert out["cached"] is False


async def test_tool_loop_then_respond(kernel) -> None:
    kernel.record("tool:executed")
    kernel.provider.script = [
        {"thought": "read it", "action": "tool", "tool": "fs_read", "args": {"path": "pyproject.toml", "max_chars": 200}},
        {"thought": "done", "action": "respond", "response": "read the file"},
    ]
    out = await kernel.orchestrator.handle_task("read pyproject", source="test")
    assert out["response"] == "read the file"
    assert len(out["steps"]) == 1
    assert out["steps"][0]["tool"] == "fs_read" and out["steps"][0]["ok"] is True
    # tool:executed event reached the bus (memory plugin also observes it)
    assert any(p["tool"] == "fs_read" for p in kernel.captured["tool:executed"])


async def test_unknown_tool_recovers(kernel) -> None:
    """A failing tool call must not kill the loop; the LLM gets the error."""
    kernel.provider.script = [
        {"thought": "try", "action": "tool", "tool": "does_not_exist", "args": {}},
        {"thought": "recover", "action": "respond", "response": "recovered"},
    ]
    out = await kernel.orchestrator.handle_task("x")
    assert out["response"] == "recovered"
    assert out["steps"][0]["ok"] is False


async def test_max_steps_forces_final_answer(kernel) -> None:
    kernel.orchestrator.max_steps = 2
    # script that keeps wanting tools → loop hits cap → forced final respond
    kernel.provider.script = [
        {"thought": "1", "action": "tool", "tool": "fs_read", "args": {"path": "pyproject.toml", "max_chars": 20}},
        {"thought": "2", "action": "tool", "tool": "fs_read", "args": {"path": "pyproject.toml", "max_chars": 20}},
        {"thought": "forced", "action": "respond", "response": "gave up gracefully"},
    ]
    out = await kernel.orchestrator.handle_task("loop")
    assert out["response"] == "gave up gracefully"
    assert len(out["steps"]) == kernel.orchestrator.max_steps


async def test_semantic_cache_hit(kernel) -> None:
    kernel.orchestrator.cache._config.enable_semantic_cache = True
    kernel.provider.script = [{"thought": "t", "action": "respond", "response": "cached answer"}]
    first = await kernel.orchestrator.handle_task("cache me please", source="test")
    assert first["cached"] is False
    second = await kernel.orchestrator.handle_task("cache me please", source="test")
    assert second["cached"] is True
    assert second["response"] == "cached answer"
    # provider was called exactly once
    assert len([c for c in kernel.provider.calls if any("cache me please" in m.get("content", "") for m in c)]) == 1


async def test_semantic_cache_similar_prompt(kernel) -> None:
    cache = SemanticCache(EchoConfig(enable_semantic_cache=True, cache_similarity_threshold=0.90))
    payload = {"response": "same-ish"}
    cache.store("What is the capital of France?", ["t"], payload)
    # exact key differs, semantic similarity is 1.0 (same tokens)
    assert cache.lookup("what IS the capital of france?", ["t"]) == payload
    assert cache.lookup("totally different prompt", ["t"]) is None


def test_token_budget_counts() -> None:
    budget = TokenBudget()
    n = budget.count("hello world " * 10)
    assert n > 0
    msgs = [{"role": "user", "content": "a" * 100}, {"role": "assistant", "content": "b" * 100}]
    assert budget.count_messages(msgs) > budget.count("a" * 100)


def test_extract_json_robustness() -> None:
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('noise before {"a": {"b": [1,2]}} noise after') == {"a": {"b": [1, 2]}}
    assert _extract_json('```json\n{"x": true}\n```') == {"x": True}
    assert _extract_json('say "quote {inside}" {"y": 2}') == {"y": 2}
    assert _extract_json("no json here") == {}
    assert _extract_json("") == {}


async def test_context_pruning_inserts_summary(kernel) -> None:
    """Tiny budget + long history ⇒ old turns fold into ONE summary block."""
    orch = kernel.orchestrator
    orch.config.max_context_tokens = 200
    orch.config.prune_target_ratio = 0.8
    # fill conversation with long turns
    for i in range(6):
        orch.conversation.append({"role": "user", "content": f"turn {i} " + "x" * 120})
        orch.conversation.append({"role": "assistant", "content": f"reply {i} " + "y" * 120})
    kernel.provider.script = [{"thought": "t", "action": "respond", "response": "pruned!"}]
    out = await orch.handle_task("new task now")
    assert out["response"] == "pruned!"
    first_call = kernel.provider.calls[-1]
    contents = [m["content"] for m in first_call]
    # a MEMORY SUMMARY block folded the dropped turns
    summary_blocks = [c for c in contents if "MEMORY SUMMARY of earlier turns" in c]
    assert summary_blocks, "old turns must collapse into one summary block"
    # newest user turn retained verbatim
    assert any("new task now" in c for c in contents)
    # with a 200-token budget, most of the 12 prior turns CANNOT appear verbatim
    prior_in_prompt = sum(1 for c in contents if c.startswith(("turn ", "reply ")))
    assert prior_in_prompt < 12


async def test_episode_written_to_memory(kernel) -> None:
    kernel.provider.script = [
        {"thought": "t", "action": "tool", "tool": "fs_read", "args": {"path": "pyproject.toml", "max_chars": 50}},
        {"thought": "t", "action": "respond", "response": "did it"},
    ]
    out = await kernel.orchestrator.handle_task("log me")
    memory = kernel.ctx.service("memory")
    episodes = memory.em.recent(limit=5)
    assert any("fs_read" in e["action"] for e in episodes)


async def test_invalid_tool_names_are_reported_not_executed(kernel) -> None:
    """Malformed tool calls become observations with the valid tool list."""
    kernel.provider.script = [
        {"thought": "x", "action": "tool", "tool": "no-tools", "args": {}},
        {"thought": "y", "action": "tool", "tool": "", "args": {}},
        {"thought": "done", "action": "respond", "response": "handled"},
    ]
    events: list[dict] = []
    out = await kernel.orchestrator.handle_task("hi", on_event=events.append)

    assert out["response"] == "handled"
    assert [s["tool"] for s in out["steps"]] == ["no-tools", None]
    assert all(s["ok"] is False for s in out["steps"])
    assert any(e["type"] == "invalid_tool" for e in events)
    # The model was told exactly which tools exist before its next attempt.
    second_prompt = "\n".join(m["content"] for m in kernel.provider.calls[1])
    assert "unknown tool: no-tools" in second_prompt
    assert "fs_read" in second_prompt


async def test_progress_events_describe_the_loop(kernel) -> None:
    kernel.provider.script = [
        {"thought": "read", "action": "tool", "tool": "fs_read", "args": {"path": "pyproject.toml", "max_chars": 20}},
        {"thought": "done", "action": "respond", "response": "ok"},
    ]
    events: list[dict] = []
    await kernel.orchestrator.handle_task("read", on_event=events.append)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "thinking"
    assert "tool" in kinds and "tool_result" in kinds


async def test_broken_progress_callback_cannot_break_loop(kernel) -> None:
    def boom(_event: dict) -> None:
        raise RuntimeError("UI died")

    kernel.provider.script = [{"thought": "t", "action": "respond", "response": "fine"}]
    out = await kernel.orchestrator.handle_task("x", on_event=boom)
    assert out["response"] == "fine"


async def test_plain_text_response_passes_through(kernel) -> None:
    """A normal text answer must reach the user unchanged (no JSON leak)."""
    kernel.provider.script = [
        {"thought": "done", "action": "respond", "response": "Hello, Sanjeev!"}
    ]
    out = await kernel.orchestrator.handle_task("hi", source="test")
    assert out["response"] == "Hello, Sanjeev!"


async def test_json_response_is_rephrased_not_leaked(kernel) -> None:
    """The model is told to emit only JSON, so it may leak raw JSON in the
    ``response`` field. That must be rephrased into natural language, never
    surfaced as raw JSON to the user."""
    kernel.provider.script = [
        {"thought": "done", "action": "respond", "response": '{"status": "ok", "value": 42}'},
        {"thought": "done", "action": "respond", "response": "The status is ok with value 42."},
    ]
    out = await kernel.orchestrator.handle_task("check", source="test")
    assert out["response"] == "The status is ok with value 42."
    assert "{" not in out["response"] and "}" not in out["response"]


async def test_api_error_is_told_to_user(kernel) -> None:
    """When the LLM API errors, the user must see a clear error message,
    not a silent failure or raw internals."""

    class BoomProvider(FakeProvider):
        async def complete(self, messages, *, schema=None, temperature=0.2, max_tokens=None):
            raise LLMProviderError("upstream boom", status=500)

    kernel.provider = BoomProvider()
    kernel.orchestrator.provider = kernel.provider
    out = await kernel.orchestrator.handle_task("anything", source="test")
    assert "HTTP 500" in out["response"]
    assert "upstream boom" in out["response"]
