"""ChromaDB vector store (persistent).

Chroma stores, for every chunk: an id, the text, its embedding and metadata.
``PersistentClient`` writes to disk, so the index survives restarts - the
database is opened, never recreated, when the app starts.

We compute embeddings ourselves (with the injected embedder) and pass them to
Chroma, so Chroma never downloads or runs a model of its own. The collection
uses cosine distance; we report similarity = 1 - distance.
"""

import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.exceptions import EmbeddingError, VectorStoreError
from app.retrieval.embeddings import EmbeddingBackend
from app.schemas import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

_WRITE_BATCH = 256
_READ_BATCH = 1000


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Chroma only accepts str/int/float/bool metadata values - drop the rest."""
    return {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}


class VectorStore:
    """Reusable wrapper around one persistent Chroma collection."""

    def __init__(
        self,
        embedder: EmbeddingBackend,
        persist_path: str | Path,
        collection_name: str = "documents",
    ) -> None:
        self._embedder = embedder
        self.persist_path = Path(persist_path)
        self.collection_name = collection_name
        try:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._open_collection()
        except Exception as exc:
            raise VectorStoreError(
                f"Could not open ChromaDB at '{self.persist_path}': {exc}"
            ) from exc
        logger.info(
            "ChromaDB ready at %s (collection '%s', %d chunks)",
            self.persist_path,
            collection_name,
            self.count(),
        )

    def _open_collection(self) -> Any:
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,  # we supply embeddings ourselves
        )

    # ---------------------------------------------------------------- writes
    def add_documents(self, chunks: Sequence[Chunk]) -> int:
        """Embed and store chunks (upsert: same chunk id overwrites). Returns count."""
        if not chunks:
            return 0
        embeddings = self._embedder.embed_documents([c.text for c in chunks])
        if len(embeddings) != len(chunks):
            raise EmbeddingError(
                f"Embedder returned {len(embeddings)} vectors for {len(chunks)} chunks."
            )
        try:
            for i in range(0, len(chunks), _WRITE_BATCH):
                batch = chunks[i : i + _WRITE_BATCH]
                self._collection.upsert(
                    ids=[c.chunk_id for c in batch],
                    documents=[c.text for c in batch],
                    embeddings=embeddings[i : i + _WRITE_BATCH],
                    metadatas=[_clean_metadata(c.metadata) for c in batch],
                )
        except Exception as exc:
            raise VectorStoreError(
                f"Could not write to ChromaDB: {exc}. If you changed EMBEDDING_MODEL, "
                "clear the database first (vectors of different sizes cannot be mixed)."
            ) from exc
        logger.info("Stored %d chunks in ChromaDB", len(chunks))
        return len(chunks)

    def delete_by_source(self, source: str) -> None:
        """Remove every chunk of one file (used before re-ingesting it)."""
        try:
            self._collection.delete(where={"source": source})
        except Exception as exc:
            raise VectorStoreError(f"Could not delete chunks of '{source}': {exc}") from exc

    def delete_collection(self) -> None:
        """Delete ALL stored chunks. The store stays usable (empty) afterwards."""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._open_collection()
        except Exception as exc:
            raise VectorStoreError(f"Could not delete collection: {exc}") from exc
        logger.info("Deleted collection '%s'", self.collection_name)

    # ----------------------------------------------------------------- reads
    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as exc:
            raise VectorStoreError(f"Could not count chunks: {exc}") from exc

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Semantic search: the ``top_k`` chunks closest in meaning to ``query``."""
        total = self.count()
        if total == 0 or top_k <= 0:
            return []
        query_vector = self._embedder.embed_query(query)
        try:
            result = self._collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, total),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Vector search failed: {exc}. If you changed EMBEDDING_MODEL, "
                "clear the database and re-ingest."
            ) from exc

        hits = [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                metadata=dict(metadata or {}),
                vector_rank=rank,
                vector_score=1.0 - float(distance),
            )
            for rank, (chunk_id, text, metadata, distance) in enumerate(
                zip(
                    result["ids"][0],
                    result["documents"][0],
                    result["metadatas"][0],
                    result["distances"][0],
                    strict=True,
                ),
                start=1,
            )
        ]
        logger.info("Vector search returned %d results", len(hits))
        return hits

    def get_all(self) -> list[Chunk]:
        """Return every stored chunk (used to (re)build the BM25 index)."""
        chunks: list[Chunk] = []
        offset = 0
        try:
            while True:
                batch = self._collection.get(
                    limit=_READ_BATCH, offset=offset, include=["documents", "metadatas"]
                )
                ids = batch["ids"]
                if not ids:
                    break
                for chunk_id, text, metadata in zip(ids, batch["documents"], batch["metadatas"], strict=True):
                    chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=dict(metadata or {})))
                offset += len(ids)
                if len(ids) < _READ_BATCH:
                    break
        except Exception as exc:
            raise VectorStoreError(f"Could not read chunks from ChromaDB: {exc}") from exc
        return chunks

    def list_sources(self) -> dict[str, int]:
        """Map filename -> number of chunks stored for it."""
        return dict(Counter(c.metadata.get("source", "unknown") for c in self.get_all()))
