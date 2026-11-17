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
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python_core.core.utils import cosine, hash_embedding

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

    def __init__(self, vector_dir: Path) -> None:
        self.dir = vector_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._docs_path = self.dir / "docs.jsonl"
        self._docs: dict[str, dict[str, Any]] = {}
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
    @staticmethod
    def embed(text: str) -> list[float]:
        """Local, deterministic embedding. Replace with a local ONNX encoder
        for production-grade recall; the interface stays identical."""
        return hash_embedding(text)

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
        for doc_id, score in hits[:k]:
            doc = self._docs.get(doc_id)
            if doc is None or score < min_score:
                continue
            out.append({"id": doc_id, "text": doc["text"], "metadata": doc["metadata"], "score": round(float(score), 4)})
        return out

    def count(self) -> int:
        return len(self._docs)

    def stats(self) -> dict[str, Any]:
        return {"backend": self._backend_name, "documents": self.count()}


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
