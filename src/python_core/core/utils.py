"""Small, dependency-free utilities shared by the core and all plugins."""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["extract_json", "hash_embedding", "cosine"]


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from free text.

    Local models are notorious for wrapping JSON in prose or code fences;
    this scanner handles the common cases (raw object, fenced block, prose
    with an embedded balanced object).
    """
    if not isinstance(text, str):
        return {}
    text = text.strip()
    # Strip a single markdown fence if present.
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Scan for the first balanced {...} block.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return {}


def hash_embedding(text: str, dim: int = 256) -> list[float]:
    """Hashed bag-of-words embedding: cheap, dependency-free, stable.

    Quality is below a neural encoder, but for *near-duplicate detection*
    (semantic cache hits, similar-task matching) cosine similarity over
    hashed unigrams is a proven local-first technique.
    """
    vec = [0.0] * dim
    words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    for w in words:
        bucket = int(hashlib.md5(w.encode()).hexdigest(), 16) % dim
        vec[bucket] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for equal-length float vectors (0.0 on mismatch)."""
    if len(a) != len(b) or not a:
        return 0.0
    return sum(x * y for x, y in zip(a, b))
