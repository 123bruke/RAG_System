"""Cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2).

BI-ENCODER (what vector search uses): query and passage are embedded
SEPARATELY, then compared. Fast, but the model never sees them together.

CROSS-ENCODER: the model reads "(query, passage)" TOGETHER and outputs one
relevance score. Much more accurate, but too slow to run on thousands of
chunks - so we only run it on the 10-20 candidates that hybrid search already
shortlisted.

The score is a raw logit: higher = more relevant. It can be negative and is NOT
a probability; only compare scores within the same query.
"""

import logging
from dataclasses import replace
from typing import Any, Protocol

from app.exceptions import RetrievalError
from app.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class RerankerBackend(Protocol):
    """Anything that can rerank candidates (real model or test fake)."""

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder wrapper (model loaded lazily)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RetrievalError(
                    "sentence-transformers is not installed. Run: pip install -r requirements.txt"
                ) from exc
            logger.info("Loading reranker model: %s", self.model_name)
            try:
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                raise RetrievalError(
                    f"Could not load reranker model '{self.model_name}': {exc}. "
                    "The first run needs internet access to download the model."
                ) from exc
        return self._model

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Score every (query, chunk) pair and return the ``top_k`` best, best first."""
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Reranking failed: {exc}") from exc

        if len(scores) != len(candidates):
            raise RetrievalError(
                f"Reranker returned {len(scores)} scores for {len(candidates)} candidates."
            )
        scored = [replace(c, rerank_score=float(s)) for c, s in zip(candidates, scores, strict=True)]
        scored.sort(key=lambda c: c.rerank_score, reverse=True)  # type: ignore[arg-type,return-value]
        top = scored[:top_k]
        logger.info("Reranker selected %d documents", len(top))
        return top
