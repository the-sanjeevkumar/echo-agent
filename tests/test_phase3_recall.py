"""PHASE 3: FTS5 recall, session archive, cache eviction, hybrid embeddings.

Acceptance coverage:
* post-reset session_search finds pre-reset content
* session:reset evicts that session's semantic-cache entries
* recall() no longer injects unconditional "last 3 episodes" noise
* embed() interface runs real ONNX MiniLM when installed, hash otherwise;
  hybrid FTS+vector scoring is exposed
"""

from __future__ import annotations

from pathlib import Path

from python_core.plugins.memory.embeddings import EmbeddingEngine
from python_core.plugins.memory.storage import EpisodicMemory, SemanticMemory

# ── FTS + session archive (direct storage) ────────────────────────────────────

def test_fts5_episode_search(tmp_path: Path) -> None:
    em = EpisodicMemory(tmp_path / "ep.sqlite3")
    em.append("deploy service", "succeeded using the release checklist")
    em.append("run tests", "all green")
    hits = em.search_episodes("deploy checklist")
    assert hits and "deploy" in hits[0]["text"]
    assert hits[0]["kind"] == "episode"
    assert em.search_episodes("") == []  # empty query is a no-op
    em.close()


def test_session_archive_and_search(tmp_path: Path) -> None:
    em = EpisodicMemory(tmp_path / "ep.sqlite3")
    messages = [
        {"role": "user", "content": "Remember: the launch code is BLUE."},
        {"role": "assistant", "content": "Noted — the launch code is BLUE."},
    ]
    assert em.archive_session("telegram:dm:42", messages, user_id="42") == 2
    hits = em.search_sessions("launch code BLUE")
    assert len(hits) >= 2
    assert any("BLUE" in h["text"] for h in hits)
    assert all(h["kind"] == "session" for h in hits)
    # scope by session
    scoped = em.search_sessions("launch code", session_id="telegram:dm:42")
    assert scoped and all(h["session_id"] == "telegram:dm:42" for h in scoped)
    assert em.search_sessions("launch code", session_id="other") == []
    em.close()


# ── full kernel: reset archives + cache eviction ──────────────────────────────

async def test_post_reset_session_search_finds_content(kernel) -> None:
    kernel.provider.script = [
        {"thought": "note", "action": "respond", "response": "Noted — the launch code is BLUE."}
    ]
    await kernel.orchestrator.handle_task(
        "Remember the launch code is BLUE", source="cli", user_id="owner", session_id="phase3"
    )

    # Reset through the bus (as gateways do).
    result = await kernel.bus.emit("session:reset", {"session_id": "phase3"}, source="test")
    assert result

    search = await kernel.ctx.tools.execute("session_search", {"query": "launch code BLUE"})
    assert search["ok"] and search["result"]["ok"]
    assert any("BLUE" in hit["text"] for hit in search["result"]["hits"]), search["result"]["hits"]

    # The live session is really cleared.
    assert kernel.orchestrator._sessions.get("phase3") is None


async def test_reset_evicts_session_cache(kernel) -> None:
    kernel.orchestrator.cache._config.enable_semantic_cache = True
    kernel.provider.script = [
        {"thought": "a", "action": "respond", "response": "first answer"},
        {"thought": "b", "action": "respond", "response": "second answer"},
    ]
    first = await kernel.orchestrator.handle_task(
        "cache me phase3", source="cli", user_id="owner", session_id="phase3-cache"
    )
    second = await kernel.orchestrator.handle_task(
        "cache me phase3", source="cli", user_id="owner", session_id="phase3-cache"
    )
    assert first["cached"] is False and second["cached"] is True

    await kernel.bus.emit("session:reset", {"session_id": "phase3-cache"}, source="test")

    third = await kernel.orchestrator.handle_task(
        "cache me phase3", source="cli", user_id="owner", session_id="phase3-cache"
    )
    assert third["cached"] is False, "reset must evict the session's cached entries"


async def test_recall_has_no_unconditional_episode_noise(kernel) -> None:
    memory = kernel.ctx.service("memory")
    memory.append_episode("totally unrelated work", "some outcome")
    hits = memory.recall("xylophone unicorn question")
    assert all(h.get("kind") != "episode" for h in hits), hits


# ── embeddings behind one stable interface ────────────────────────────────────

def test_embedding_engine_hash_fallback(tmp_path: Path) -> None:
    engine = EmbeddingEngine(tmp_path, backend="hash")
    assert engine.active == "hash" and engine.dim == 256
    vec = engine.embed("hello world")
    assert len(vec) == 256
    assert vec == engine.embed("hello world")  # deterministic offline fallback


def test_embedding_engine_auto_degrades_without_onnx(tmp_path: Path) -> None:
    # onnxruntime/tokenizers or model files are absent in the offline test env.
    engine = EmbeddingEngine(tmp_path, backend="auto")
    assert engine.active == "hash"
    assert len(engine.embed("deterministic fallback works")) == 256


def test_semantic_memory_embed_interface_and_hybrid_scores(tmp_path: Path) -> None:
    sm = SemanticMemory(tmp_path / "vec")
    assert sm.stats()["embeddings"]["active"] == "hash"
    assert len(sm.embed("test")) == 256
    sm.upsert("d1", "Deploying requires the release checklist")
    hits = sm.recall("release checklist")
    assert hits and hits[0]["id"] == "d1"
    assert "vector_score" in hits[0] and "lexical_score" in hits[0]
    assert hits[0]["lexical_score"] > 0
