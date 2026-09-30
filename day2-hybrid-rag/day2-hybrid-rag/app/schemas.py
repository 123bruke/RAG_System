"""Plain data containers passed between pipeline stages.

Using small dataclasses (instead of loose dicts) gives us type hints,
autocomplete and one obvious place to see what flows through the system.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A piece of a document, ready to be embedded and indexed."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class RetrievedChunk:
    """A chunk returned by a retrieval stage, with the scores collected so far.

    Each stage fills in its own fields, so one object tells the whole story:
    where the chunk ranked in vector search, in BM25, after RRF fusion and
    after cross-encoder reranking.
    """

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    vector_rank: int | None = None
    vector_score: float | None = None  # cosine similarity (1.0 = identical direction)
    bm25_rank: int | None = None
    bm25_score: float | None = None  # raw BM25 score (only comparable within one query)
    rrf_score: float | None = None
    rerank_score: float | None = None  # cross-encoder logit (higher = more relevant)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    @property
    def chunk_index(self) -> int:
        return int(self.metadata.get("chunk_index", -1))

    @property
    def page(self) -> int | None:
        page = self.metadata.get("page")
        return int(page) if page is not None else None

    def citation(self) -> str:
        """Human-readable reference, e.g. ``refund_policy.pdf — chunk 4 (page 2)``."""
        page = f" (page {self.page})" if self.page is not None else ""
        return f"{self.source} — chunk {self.chunk_index}{page}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HybridResult:
    """Output of the hybrid retrieval step (before reranking)."""

    vector_hits: list[RetrievedChunk] = field(default_factory=list)
    bm25_hits: list[RetrievedChunk] = field(default_factory=list)
    fused: list[RetrievedChunk] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Everything retrieval produced for one question (used by the UI/evaluator)."""

    query: str
    vector_hits: list[RetrievedChunk]
    bm25_hits: list[RetrievedChunk]
    fused: list[RetrievedChunk]
    reranked: list[RetrievedChunk]


@dataclass
class FileIngestResult:
    """Outcome of ingesting one file."""

    filename: str
    status: str  # "ok" | "error"
    chunks: int = 0
    error: str | None = None


@dataclass
class IngestionReport:
    """Outcome of ingesting one or more files."""

    files: list[FileIngestResult] = field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        return sum(f.chunks for f in self.files)

    @property
    def ok_count(self) -> int:
        return sum(1 for f in self.files if f.status == "ok")

    @property
    def failed_count(self) -> int:
        return sum(1 for f in self.files if f.status != "ok")


@dataclass
class RAGResponse:
    """Final answer plus everything needed to explain it."""

    question: str
    answer: str
    sources: list[RetrievedChunk]
    cited: set[int] = field(default_factory=set)  # 1-based source numbers the LLM cited
    answered: bool = True  # False when we short-circuited with "I don't know"
    retrieval: RetrievalResult | None = None
