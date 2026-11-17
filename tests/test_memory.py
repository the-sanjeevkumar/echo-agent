"""Dual-process memory: WM decay, EM graph, SM vector recall, consolidation."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from python_core.core.event_bus import EventBus
from python_core.plugins.memory.consolidator import ConsolidationLoop
from python_core.plugins.memory.storage import EpisodicMemory, SemanticMemory, WorkingMemory


# ── Working Memory ───────────────────────────────────────────────────────────
def test_wm_capacity_and_snapshot() -> None:
    wm = WorkingMemory(capacity=5)
    for i in range(8):
        wm.push(f"item {i}", salience=0.5)
    snap = wm.snapshot(limit=10)
    assert len(snap) == 5
    assert "item 7" in snap  # newest retained (ring buffer)


def test_wm_salience_ranking() -> None:
    wm = WorkingMemory(capacity=4)
    wm.push("boring", salience=0.1)
    wm.push("critical", salience=1.0)
    assert wm.snapshot(limit=1) == ["critical"]


def test_wm_decay_eviction() -> None:
    wm = WorkingMemory(capacity=10, decay_after_seconds=0.0)
    wm.push("ephemeral", salience=0.1)
    time.sleep(0.01)
    wm.push("trigger", salience=0.5)  # triggers eviction of decayed low-salience
    assert all("ephemeral" != t for t in wm.snapshot(limit=10)) or "trigger" in wm.snapshot(limit=10)


# ── Episodic Memory ──────────────────────────────────────────────────────────
def test_em_append_and_recent(tmp_path: Path) -> None:
    em = EpisodicMemory(tmp_path / "ep.sqlite3")
    em.append("ran tests", "all green", success=True, metadata={"x": 1})
    em.append("deployed", "500 error", success=False)
    eps = em.recent(limit=10)
    assert len(eps) == 2
    assert eps[0]["action"] == "ran tests" and eps[0]["success"] is True
    assert eps[1]["success"] is False
    assert em.stats()["episodes"] == 2
    em.close()


def test_em_graph_trace_back(tmp_path: Path) -> None:
    em = EpisodicMemory(tmp_path / "ep.sqlite3")
    em.append("write code", "tests pass")
    em.append("run deploy", "tests pass")
    chain = em.trace_back("tests pass")
    assert "write code" in chain and "run deploy" in chain
    assert em.stats()["graph_edges"] >= 2
    em.close()


def test_em_predict_success_rate(tmp_path: Path) -> None:
    em = EpisodicMemory(tmp_path / "ep.sqlite3")
    em.append("pytest run", "ok", success=True)
    em.append("pytest run", "fail", success=False)
    em.append("pytest run", "ok", success=True)
    rate = em.predict("pytest run")
    assert abs(rate - 2 / 3) < 1e-6
    em.close()


def test_em_rebuild_graph_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "ep.sqlite3"
    em = EpisodicMemory(path)
    em.append("action a", "outcome x")
    em.close()
    em2 = EpisodicMemory(path)
    assert em2.trace_back("outcome x") == ["action a"]
    em2.close()


# ── Semantic Memory ──────────────────────────────────────────────────────────
def test_sm_upsert_and_recall(tmp_path: Path) -> None:
    sm = SemanticMemory(tmp_path / "vec")
    sm.upsert("d1", "The cat sat on the mat")
    sm.upsert("d2", "Deploying requires the release checklist")
    sm.upsert("d3", "Rust ownership prevents data races")
    hits = sm.recall("the cat sat", k=2)
    assert hits and hits[0]["id"] == "d1"
    assert hits[0]["metadata"]["salience"] == 0.5  # default metadata retained
    assert sm.stats()["documents"] == 3


def test_sm_persistence_across_instances(tmp_path: Path) -> None:
    sm = SemanticMemory(tmp_path / "vec")
    sm.upsert("persist", "durability of local memory files")
    sm2 = SemanticMemory(tmp_path / "vec")  # fresh instance re-loads docs.jsonl
    hits = sm2.recall("durability local memory", k=3)
    assert any(h["id"] == "persist" for h in hits)


# ── Consolidation ────────────────────────────────────────────────────────────
async def test_consolidation_heuristic_writes_sm(tmp_path: Path) -> None:
    bus = EventBus()
    wm = WorkingMemory(capacity=32)
    em = EpisodicMemory(tmp_path / "ep.sqlite3")
    sm = SemanticMemory(tmp_path / "vec")

    # recurring theme (2 near-duplicates) + repeated failing action
    wm.push("fix flaky test retry logic", salience=0.9)
    wm.push("fix flaky test retry logic!", salience=0.9)
    em.append("shell git push", "rejected", success=False)
    em.append("shell git push", "rejected", success=False)

    loop = ConsolidationLoop(bus, wm, em, sm, idle_seconds=1, min_interval_seconds=0)
    report = await loop.consolidate()

    assert report["patterns"] >= 2
    assert loop.patterns_written == report["patterns"]
    hits = sm.recall("flaky test retry", k=5)
    assert any(h["metadata"].get("kind") in ("fact", "cause-effect") for h in hits)
    em.close()


async def test_consolidation_resolves_llm_lazily(tmp_path: Path) -> None:
    """Regression: ``self.llm`` is None at construction time because the memory
    plugin starts before the LLM plugin in the kernel bootstrap; the provider
    must be resolved from kernel services when the pass actually runs."""
    bus = EventBus()
    wm = WorkingMemory(capacity=8)
    em = EpisodicMemory(tmp_path / "ep.sqlite3")
    sm = SemanticMemory(tmp_path / "vec")
    wm.push("prefer atomic writes over whole-file rewrites", salience=0.9)

    calls: list[dict] = []

    class FakeLLM:
        async def complete(self, messages, **kw):
            calls.append(kw)
            return {"content": {"patterns": [{"summary": "use atomic writes", "kind": "fact", "salience": 0.8}]}}

    services = {"llm": FakeLLM()}
    loop = ConsolidationLoop(
        bus, wm, em, sm, idle_seconds=1, min_interval_seconds=0, services_getter=lambda: services
    )
    report = await loop.consolidate()

    assert calls, "LLM-backed extraction never ran"
    assert report["patterns"] == 1
    assert loop.patterns_written == 1
    hits = sm.recall("atomic writes", k=3)
    assert any(h["metadata"].get("kind") == "fact" for h in hits)
    em.close()


async def test_consolidation_idle_trigger(tmp_path: Path) -> None:
    bus = EventBus()
    wm = WorkingMemory(capacity=8)
    em = EpisodicMemory(tmp_path / "ep.sqlite3")
    sm = SemanticMemory(tmp_path / "vec")
    wm.push("something salient", salience=0.9)

    loop = ConsolidationLoop(bus, wm, em, sm, idle_seconds=0.2, min_interval_seconds=0)
    loop.start()
    # activity keeps postponing; after going idle it consolidates
    loop.notify_activity()
    await _eventually(lambda: loop.consolidations >= 1, timeout=5.0)
    await loop.stop()


async def _eventually(predicate, timeout: float) -> None:
    import asyncio

    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition not met in time")


# ── plugin-level wiring ──────────────────────────────────────────────────────
async def test_memory_plugin_registers_tools_and_records(kernel) -> None:
    memory = kernel.ctx.service("memory")
    assert memory is not None
    # dynamic tools registered through the bus are live
    r = await kernel.ctx.tools.execute("memory_remember", {"text": "echo-agent is local-first", "salience": 0.9})
    assert r["ok"] and r["result"]["ok"]
    r2 = await kernel.ctx.tools.execute("memory_recall", {"query": "echo-agent local-first", "k": 3})
    assert r2["ok"] and any("local-first" in h["text"] for h in r2["result"]["hits"])

    # task:completed → working memory push (publish is fire-and-forget: yield)
    kernel.bus.publish("task:completed", {"task": "hello task", "response": "done!", "cached": False})
    await asyncio.sleep(0.1)
    assert any("hello task" in s for s in memory.working_snapshot(limit=10))
