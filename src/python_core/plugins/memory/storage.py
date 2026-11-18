"""Dual-process cognitive memory — local storage layer (MODULE 1).

Three biologically-inspired stores, 100% local:

  WorkingMemory   volatile in-memory ring buffer with salience decay
                  (analogous to pre-attentive scratchpad / "stm").
  EpisodicMemory  chronological action→outcome experiences in SQLite plus a
                  NetworkX directed cause-effect graph (causal replay).
  SemanticMemory  abstracted knowledge/skills in a local vector DB; backend
                  auto-selection: chromadb (persistent) → cpp_memory native
                  index (in-memory + jsonl persistence) → numpy flat index.

All writes are tiny and serialized behind locks; SQLite is the durability
floor, so a crash never loses more than the volatile Working Memory — exactly
like biological sleep-dependent consolidation.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python_core.core.utils import cosine

from .embeddings import EmbeddingEngine

logger = logging.getLogger("echo.memory")

try:  # optional cause-effect graph (pure-python, installs cleanly)
    import networkx as nx
except Exception:  # noqa: BLE001
    nx = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# Working Memory: fast, volatile, salience-ranked
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class WMItem:
    text: str
    role: str
    salience: float
    ts: float


class WorkingMemory:
    """Bounded ring with decay: low-salience items evict first."""

    def __init__(self, capacity: int = 64, decay_after_seconds: float = 900.0) -> None:
        self.capacity = capacity
        self.decay_after = decay_after_seconds
        self._items: deque[WMItem] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def push(self, text: str, *, role: str = "system", salience: float = 0.5) -> None:
        with self._lock:
            self._items.append(WMItem(text=text, role=role, salience=max(0.0, min(1.0, salience)), ts=time.time()))
            self._evict()

    def _evict(self) -> None:
        # Drop decayed low-salience items first; keep the ring bounded.
        now = time.time()
        kept = deque(
            (i for i in self._items if i.salience >= 0.2 or now - i.ts < self.decay_after),
            maxlen=self.capacity,
        )
        self._items = kept

    def snapshot(self, limit: int = 6) -> list[str]:
        with self._lock:
            ranked = sorted(self._items, key=lambda i: i.salience, reverse=True)[:limit]
            return [i.text for i in ranked]

    def candidates(self, min_salience: float = 0.6) -> list[WMItem]:
        """High-value items eligible for consolidation into EM/SM."""
        with self._lock:
            return [i for i in self._items if i.salience >= min_salience]

    def drain(self) -> list[WMItem]:
        with self._lock:
            items = list(self._items)
            self._items.clear()
            return items

    def __len__(self) -> int:
        return len(self._items)


# ─────────────────────────────────────────────────────────────────────────────
# Episodic Memory: SQLite rows + cause-effect graph
# ─────────────────────────────────────────────────────────────────────────────
class EpisodicMemory:
    """Chronological experiences with a learned action→outcome graph.

    The graph is the interesting part: every episode adds/reinforces a
    weighted edge ``action ≈→ outcome``. Over time:
      * ``trace_back(outcome)`` reveals what typically *causes* an outcome.
      * ``predict(action)`` estimates the historical success rate, which the
        orchestrator and consolidator use as a cheap local prior.
    """

    def __init__(self, sqlite_path: Path) -> None:
        self.path = sqlite_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")   # durable + concurrent reads
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._graph = None
        if nx is not None:
            self._graph = nx.DiGraph()
        self._init_schema()
        self._rebuild_graph()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts        REAL    NOT NULL,
                    action    TEXT    NOT NULL,
                    outcome   TEXT    NOT NULL,
                    success   INTEGER NOT NULL DEFAULT 1,
                    salience  REAL    NOT NULL DEFAULT 0.5,
                    metadata  TEXT
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_action ON episodes(action)")
            # PHASE 3: full-text search over episodes and a durable session
            # archive. FTS5 is local, free (no LLM cost) and survives restarts.
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts         REAL    NOT NULL,
                    session_id TEXT    NOT NULL,
                    role       TEXT    NOT NULL,
                    content    TEXT    NOT NULL,
                    user_id    TEXT
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_sid ON sessions(session_id)")
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5("
                "action, outcome, content='episodes', content_rowid='id')"
            )
            self._conn.execute(
                "CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN "
                "INSERT INTO episodes_fts(rowid, action, outcome) "
                "VALUES (new.id, new.action, new.outcome); END"
            )
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5("
                "content, content='sessions', content_rowid='id')"
            )
            self._conn.execute(
                "CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN "
                "INSERT INTO sessions_fts(rowid, content) VALUES (new.id, new.content); END"
            )
            self._conn.commit()

    def _rebuild_graph(self) -> None:
        """The cause-effect graph is derived state: rebuildable from SQLite."""
        if self._graph is None:
            return
        with self._lock:
            rows = self._conn.execute("SELECT action, outcome, success FROM episodes").fetchall()
        for action, outcome, success in rows:
            self._bump_edge(action[:120], outcome[:120], bool(success))

    def _bump_edge(self, action: str, outcome: str, success: bool) -> None:
        if self._graph is None:
            return
        w = self._graph.get_edge_data(action, outcome) or {"weight": 0, "successes": 0}
        w["weight"] += 1
        w["successes"] += int(success)
        self._graph.add_edge(action, outcome, **w)

    def append(
        self, action: str, outcome: str, *, success: bool = True, salience: float = 0.5, metadata: dict[str, Any] | None = None
    ) -> int:
        """Record one experience. Returns the episode id."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO episodes (ts, action, outcome, success, salience, metadata) VALUES (?,?,?,?,?,?)",
                (time.time(), action[:2000], outcome[:4000], int(success), salience, json.dumps(metadata or {})),
            )
            self._conn.commit()
            episode_id = int(cur.lastrowid or 0)
        self._bump_edge(action[:120], outcome[:120], success)
        return episode_id

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, action, outcome, success, salience, metadata FROM episodes ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for row in reversed(rows):  # chronological order
            out.append(
                {"id": row[0], "ts": row[1], "action": row[2], "outcome": row[3], "success": bool(row[4]), "salience": row[5], "metadata": json.loads(row[6] or "{}")}
            )
        return out

    # ── PHASE 3: FTS + session archive ────────────────────────────────────
    @staticmethod
    def _fts_query(query: str) -> str:
        """Turn a user query into a safe FTS5 OR expression (no injection)."""
        terms = re.findall(r"\w+", query or "")
        if not terms:
            return ""
        return " OR ".join(f'"{term}"' for term in terms[:40])

    def archive_session(self, session_id: str, messages: list[dict[str, str]], *, user_id: str | None = None) -> int:
        """Archive messages BEFORE a session reset clears live history."""
        stored = 0
        with self._lock:
            for message in messages or []:
                role = str(message.get("role", "") or "")
                content = str(message.get("content", "") or "")
                if role not in {"user", "assistant"} or not content.strip():
                    continue
                self._conn.execute(
                    "INSERT INTO sessions (ts, session_id, role, content, user_id) VALUES (?,?,?,?,?)",
                    (time.time(), session_id[:200], role, content[:8000], user_id or ""),
                )
                stored += 1
            if stored:
                self._conn.commit()
        return stored

    def search_sessions(
        self, query: str, *, session_id: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Return real past messages matching *query* (zero LLM cost)."""
        expression = self._fts_query(query)
        if not expression:
            return []
        sql = (
            "SELECT s.session_id, s.role, s.content, s.ts, bm25(sessions_fts) AS rank "
            "FROM sessions_fts f JOIN sessions s ON s.id = f.rowid "
            "WHERE sessions_fts MATCH ?"
        )
        params: list[Any] = [expression]
        if session_id:
            sql += " AND s.session_id = ?"
            params.append(session_id)
        sql += " ORDER BY rank LIMIT ?"
        params.append(max(1, min(int(limit), 50)))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "session_id": row[0],
                "role": row[1],
                "text": row[2],
                "ts": row[3],
                "score": round(-float(row[4]), 4),
                "kind": "session",
            }
            for row in rows
        ]

    def search_episodes(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """FTS over episode action/outcome (scoped search, no noise)."""
        expression = self._fts_query(query)
        if not expression:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT e.id, e.action, e.outcome, e.success, bm25(episodes_fts) AS rank "
                "FROM episodes_fts f JOIN episodes e ON e.id = f.rowid "
                "WHERE episodes_fts MATCH ? ORDER BY rank LIMIT ?",
                (expression, max(1, min(int(limit), 50))),
            ).fetchall()
        return [
            {
                "id": row[0],
                "text": f"experience: {row[1]} → {str(row[2])[:160]} ({'ok' if row[3] else 'fail'})",
                "score": round(-float(row[4]), 4),
                "kind": "episode",
            }
            for row in rows
        ]

    def trace_back(self, outcome_substring: str, *, max_depth: int = 3) -> list[str]:
        """BFS predecessors in the cause-effect graph — 'what led to this?'."""
        if self._graph is None or outcome_substring not in self._graph:
            return []
        seen: set[str] = set()
        frontier = {outcome_substring}
        chain: list[str] = []
        for _ in range(max_depth):
            nxt: set[str] = set()
            for node in frontier:
                for pred in self._graph.predecessors(node):
                    if pred not in seen:
                        seen.add(pred)
                        chain.append(pred)
                        nxt.add(pred)
            frontier = nxt
            if not frontier:
                break
        return chain

    def predict(self, action_prefix: str) -> float:
        """Historical success rate for actions starting with *action_prefix*.
        Cheap local prior for planning (no LLM involved)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT AVG(success) FROM episodes WHERE action LIKE ?",
                (action_prefix[:60] + "%",),
            ).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.5

    def stats(self) -> dict[str, Any]:
        with self._lock:
            n = int(self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])
        return {"episodes": n, "graph_nodes": self._graph.number_of_nodes() if self._graph else 0, "graph_edges": self._graph.number_of_edges() if self._graph else 0}

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Memory: local vector DB with backend auto-selection
# ─────────────────────────────────────────────────────────────────────────────
class SemanticMemory:
    """Abstracted knowledge store. Documents (text + metadata) persist to a
    jsonl sidecar; vectors live in the best available local backend:

        1. chromadb  — persistent local vector DB (full fidelity)
        2. cpp_memory— C++ exact cosine index (fast, in-memory, jsonl reload)
        3. numpy     — flat cosine scan (always available)

    Embeddings use the deterministic local ``hash_embedding`` by default so
    the system works fully offline; swap ``embed`` for a local ONNX encoder
    (e.g. all-MiniLM) without touching any other layer.
    """

    def __init__(self, vector_dir: Path, config: Any | None = None) -> None:
        self.dir = vector_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._docs_path = self.dir / "docs.jsonl"
        self._docs: dict[str, dict[str, Any]] = {}
        # PHASE 3: real ONNX embeddings when available; hash offline fallback.
        self.engine = EmbeddingEngine(
            getattr(config, "data_dir", vector_dir.parent) if config else vector_dir.parent,
            backend=getattr(config, "embeddings_backend", "auto"),
            model_dir=getattr(config, "embedding_model_dir", ""),
            enabled=getattr(config, "enable_embeddings", True),
        )
        self._load_docs()
        self._backend_name, self._backend = self._pick_backend()
        self._hydrate_backend()

    def _hydrate_backend(self) -> None:
        """Re-add vectors for persisted docs into in-memory backends (chromadb
        is self-persistent; numpy/cpp_memory are rebuilt from docs.jsonl)."""
        if self._backend_name == "chromadb":
            return
        for doc_id, doc in self._docs.items():
            emb = self.embed(doc["text"])
            if self._backend_name == "cpp_memory":
                self._backend.upsert(doc_id, emb)
            else:
                self._backend.add(doc_id, emb)

    # ── backend selection ──────────────────────────────────────────────────
    def _pick_backend(self) -> tuple[str, Any]:
        try:  # 1. chromadb: real persistent vector store
            import chromadb  # type: ignore

            client = chromadb.PersistentClient(path=str(self.dir / "chroma"))
            collection = client.get_or_create_collection("semantic_memory")
            return "chromadb", collection
        except Exception as exc:  # noqa: BLE001
            logger.info("chromadb unavailable (%s); trying native index", type(exc).__name__)
        try:  # 2. C++ exact index (reloaded from docs.jsonl on boot)
            import cpp_memory  # type: ignore

            return "cpp_memory", cpp_memory.VectorIndex()
        except Exception as exc:  # noqa: BLE001
            logger.info("cpp_memory unavailable (%s); using numpy flat index", type(exc).__name__)
        return "numpy", _NumpyIndex()

    def _load_docs(self) -> None:
        if self._docs_path.exists():
            with open(self._docs_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    self._docs[doc["id"]] = doc

    def _save_docs(self) -> None:
        tmp = self._docs_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for doc in self._docs.values():
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        tmp.replace(self._docs_path)  # atomic swap

    # ── public API ─────────────────────────────────────────────────────────
    def embed(self, text: str) -> list[float]:
        """Embed text behind one stable interface: ONNX MiniLM when the
        runtime + model are installed, deterministic hash otherwise."""
        return self.engine.embed(text)

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        """FTS-style lexical overlap score (local, zero cost) for hybrid rank."""
        terms = set(re.findall(r"\w+", (query or "").lower()))
        if not terms:
            return 0.0
        tokens = set(re.findall(r"\w+", (text or "").lower()))
        return len(terms & tokens) / len(terms)

    def upsert(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None, *, salience: float = 0.5) -> None:
        doc = {"id": doc_id, "text": text, "metadata": {**(metadata or {}), "salience": salience}}
        emb = self.embed(text)
        if self._backend_name == "chromadb":
            self._backend.upsert(ids=[doc_id], embeddings=[emb], documents=[text], metadatas=[doc["metadata"]])
        elif self._backend_name == "cpp_memory":
            self._backend.upsert(doc_id, emb)
        else:
            self._backend.add(doc_id, emb)
        self._docs[doc_id] = doc
        self._save_docs()

    def recall(self, query: str, *, k: int = 5, min_score: float = 0.05) -> list[dict[str, Any]]:
        emb = self.embed(query)
        hits: list[tuple[str, float]] = []
        if self._backend_name == "chromadb":
            res = self._backend.query(query_embeddings=[emb], n_results=min(k, max(1, len(self._docs))))
            ids = res.get("ids", [[]])[0]
            dists = res.get("distances", [[]])[0]
            hits = [(i, 1.0 - float(d)) for i, d in zip(ids, dists)]
        elif self._backend_name == "cpp_memory":
            hits = [(i, float(s)) for i, s in self._backend.top_k(emb, k)]
        else:
            hits = self._backend.query(emb, k)
        out = []
        for doc_id, score in hits[: max(k * 3, k)]:
            doc = self._docs.get(doc_id)
            if doc is None or score < min_score:
                continue
            vector_score = float(score)
            lexical = self._lexical_score(query, doc["text"])
            # Hybrid: vector similarity is primary, lexical FTS signal adds a
            # local boost for exact terms (NEVER an LLM call).
            hybrid = 0.75 * min(1.0, vector_score) + 0.25 * lexical
            out.append({
                "id": doc_id,
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": round(hybrid, 4),
                "vector_score": round(vector_score, 4),
                "lexical_score": round(lexical, 4),
            })
        out.sort(key=lambda hit: hit["score"], reverse=True)
        return out[:k]

    # ── PHASE 4: pattern dedup / decay ────────────────────────────────────
    def find_similar(self, text: str, *, threshold: float = 0.90) -> dict[str, Any] | None:
        """Find a consolidated pattern with cosine >= threshold (pure local)."""
        emb = self.embed(text)
        best_id, best_sim = None, threshold
        for doc_id, doc in self._docs.items():
            if (doc.get("metadata") or {}).get("origin") != "consolidation":
                continue
            sim = cosine(emb, self.embed(doc.get("text", "")))
            if sim >= best_sim:
                best_id, best_sim = doc_id, sim
        return self._docs[best_id] if best_id is not None else None

    def reinforce(self, doc_id: str, salience: float = 0.5) -> None:
        """Bump salience/timestamp of an existing pattern instead of duplicating."""
        doc = self._docs.get(doc_id)
        if doc is None:
            return
        meta = doc.setdefault("metadata", {})
        meta["salience"] = round(max(float(meta.get("salience", 0.0)), salience), 4)
        meta["ts"] = time.time()
        meta["hits"] = int(meta.get("hits", 0)) + 1
        self._save_docs()

    def decay_patterns(
        self, *, decay_per_day: float = 0.02, min_salience: float = 0.1, max_age_days: float = 60.0
    ) -> int:
        """Decay stale consolidated patterns; evict weak, old ones."""
        now = time.time()
        removed = 0
        for doc_id, doc in list(self._docs.items()):
            meta = doc.get("metadata") or {}
            if meta.get("origin") != "consolidation":
                continue
            age_days = max(0.0, (now - float(meta.get("ts") or now)) / 86400.0)
            salience = float(meta.get("salience", 0.5)) * ((1.0 - decay_per_day) ** age_days)
            if age_days > max_age_days and salience < min_salience:
                self._docs.pop(doc_id, None)
                removed += 1
            else:
                meta["salience"] = round(salience, 4)
        if removed:
            self._save_docs()
        return removed

    def count(self) -> int:
        return len(self._docs)

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self._backend_name,
            "documents": self.count(),
            "embeddings": self.engine.stats(),
        }


class _NumpyIndex:
    """Flat cosine index — the always-available local fallback."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._matrix: list[list[float]] = []

    def add(self, doc_id: str, emb: list[float]) -> None:
        if doc_id in self._ids:
            idx = self._ids.index(doc_id)
            self._matrix[idx] = emb
        else:
            self._ids.append(doc_id)
            self._matrix.append(emb)

    def query(self, emb: list[float], k: int) -> list[tuple[str, float]]:
        scored = [(doc_id, cosine(emb, vec)) for doc_id, vec in zip(self._ids, self._matrix)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
