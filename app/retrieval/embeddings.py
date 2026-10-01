"""Embedding service (sentence-transformers / all-MiniLM-L6-v2).

An embedding turns text into a list of numbers (384 of them for MiniLM) so that
texts with similar MEANING end up close together in vector space.

DOCUMENT vs QUERY embeddings
  * ``embed_documents`` encodes the passages we store (many texts, batched).
  * ``embed_query`` encodes the user's question at search time (one text).
  Some models (E5, BGE, ...) are *asymmetric*: they need different prefixes such
  as "passage: " vs "query: ". all-MiniLM-L6-v2 is *symmetric*, so both methods
  currently do the same thing - but keeping two methods means swapping to an
  asymmetric model later only changes this file.

NORMALISATION
  Vectors are scaled to length 1, so cosine similarity == dot product and
  ChromaDB's cosine distance behaves predictably.

The (large) model is loaded lazily on first use so importing the app, checking
``/health`` or opening the UI stays fast.
"""

import logging
from collections.abc import Sequence
from typing import Any, Protocol

from app.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    """Anything that can embed documents and queries (real model or test fake)."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


class EmbeddingService:
    """sentence-transformers wrapper with lazy loading and clear errors."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. Run: pip install -r requirements.txt"
                ) from exc
            logger.info("Loading embedding model: %s", self.model_name)
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # download failure, corrupt cache, bad name...
                raise EmbeddingError(
                    f"Could not load embedding model '{self.model_name}': {exc}. "
                    "The first run needs internet access to download the model."
                ) from exc
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many passages. Returns one normalised vector per text."""
        if not texts:
            return []
        logger.info("Embedding %d document chunks", len(texts))
        try:
            vectors = self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding documents failed: {exc}") from exc
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed one search query. Returns a single normalised vector."""
        if not query.strip():
            raise EmbeddingError("Cannot embed an empty query.")
        try:
            vector = self.model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding query failed: {exc}") from exc
        return vector.tolist()
