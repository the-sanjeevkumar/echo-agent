"""Local embedding engine behind the ``embed(text) -> list[float]`` interface.

PHase 3: real vector embeddings by default when the ONNX all-MiniLM model is
available (onnxruntime + tokenizers, ~100MB CPU-only encoder — NOT a local
LLM), with the deterministic ``hash_embedding`` as the offline fallback.

Model layout (config ``embedding_model_dir`` or ``<data_dir>/models/all-MiniLM-L6-v2``)::

    model.onnx        # ONNX encoder
    tokenizer.json    # HuggingFace fast tokenizer

When the runtime or model files are absent, the engine silently degrades to
lexical hash embeddings so the agent works fully offline.
"""

from __future__ import annotations

import logging
from pathlib import Path

from python_core.core.utils import hash_embedding

logger = logging.getLogger("echo.memory.embeddings")


class EmbeddingEngine:
    """Selects ONNX MiniLM when available; otherwise hash embeddings."""

    def __init__(
        self,
        data_dir: Path | None = None,
        *,
        backend: str = "auto",
        model_dir: str = "",
        enabled: bool = True,
    ) -> None:
        self.requested = str(backend or "auto").strip().lower()
        self.enabled = bool(enabled)
        self.model_dir = (
            Path(model_dir).expanduser()
            if model_dir
            else (Path(data_dir) / "models" / "all-MiniLM-L6-v2" if data_dir else Path("."))
        )
        self.active = "hash"
        self.dim = 256
        self._session = None
        self._tokenizer = None
        self._last_error = ""
        if self.enabled and self.requested != "hash":
            self._try_load_onnx()

    def _try_load_onnx(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore
            from tokenizers import Tokenizer  # type: ignore

            model_path = self.model_dir / "model.onnx"
            tokenizer_path = self.model_dir / "tokenizer.json"
            if not model_path.is_file() or not tokenizer_path.is_file():
                self._last_error = f"model files not found in {self.model_dir}"
                return
            self._session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self.active = "onnx"
            self.dim = 384
            logger.info("embeddings: ONNX all-MiniLM active (%s)", self.model_dir)
        except Exception as exc:  # noqa: BLE001 — optional native layer
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.info("embeddings: ONNX unavailable (%s); using hash fallback", self._last_error)

    def embed(self, text: str) -> list[float]:
        """Return a normalized embedding vector (384-D ONNX or 256-D hash)."""
        if self.enabled and self._session is not None:
            try:
                import numpy as np  # type: ignore

                encoding = self._tokenizer.encode(text or "", add_special_tokens=True)
                input_ids = np.array([encoding.ids], dtype=np.int64)
                attention = np.array([encoding.attention_mask], dtype=np.int64)
                outputs = self._session.run(
                    None, {"input_ids": input_ids, "attention_mask": attention}
                )
                hidden = outputs[0][0]  # (seq_len, hidden)
                mask = attention[0].reshape(-1, 1)
                pooled = (hidden * mask).sum(axis=0) / np.maximum(mask.sum(axis=0), 1)
                norm = float(np.linalg.norm(pooled))
                if norm > 0:
                    pooled = pooled / norm
                return [float(v) for v in pooled]
            except Exception as exc:  # noqa: BLE001 — never break recall
                logger.debug("ONNX embedding failed (%s); hash fallback", exc)
        if self.enabled and self.requested == "hash":
            return hash_embedding(text)
        # auto with no onnx model, or embeddings disabled: deterministic hash.
        return hash_embedding(text)

    def stats(self) -> dict[str, object]:
        return {
            "active": self.active,
            "requested": self.requested,
            "enabled": self.enabled,
            "dim": self.dim,
            "model_dir": str(self.model_dir),
            "last_error": self._last_error,
        }
