"""RAGPipeline - the one class that wires every stage together.

    Documents -> Load -> Chunk -> Embed -> ChromaDB -> BM25
    Question  -> Vector search + BM25 -> RRF -> Cross-encoder -> Top context
              -> LLM -> Answer + Sources

Heavy components (embedding model, reranker model, LLM client) are created
lazily on first use, and every component can be injected through the
constructor. Injection is what makes the pipeline unit-testable without model
downloads or API keys.
"""

import logging
import re
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from app.config import Settings, get_settings
from app.exceptions import EmptyDocumentError, NoDocumentsError, RAGError
from app.generation.llm import BaseLLM, create_llm
from app.generation.prompts import NO_ANSWER_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from app.ingestion.chunker import TextChunker
from app.ingestion.loader import DocumentLoader
from app.retrieval.embeddings import EmbeddingBackend, EmbeddingService
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.keyword_search import KeywordIndex
from app.retrieval.reranker import CrossEncoderReranker, RerankerBackend
from app.retrieval.vector_store import VectorStore
from app.schemas import (
    FileIngestResult,
    HybridResult,
    IngestionReport,
    RAGResponse,
    RetrievalResult,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(answer: str, source_count: int) -> set[int]:
    """Return the 1-based source numbers cited in the answer, e.g. '[1][3]' -> {1, 3}."""
    cited: set[int] = set()
    for group in _CITATION_RE.findall(answer):
        for number in group.split(","):
            value = int(number)
            if 1 <= value <= source_count:
                cited.add(value)
    return cited


class RAGPipeline:
    """Ingest documents, retrieve, rerank, generate, and answer with sources."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: EmbeddingBackend | None = None,
        reranker: RerankerBackend | None = None,
        llm: BaseLLM | None = None,
        vector_store: VectorStore | None = None,
        keyword_index: KeywordIndex | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.loader = DocumentLoader()
        self.chunker = TextChunker(self.settings.chunk_size, self.settings.chunk_overlap)
        self.keyword_index = keyword_index or KeywordIndex()

        self._embedder = embedder
        self._reranker = reranker
        self._llm = llm
        self._vector_store = vector_store
        self._searcher: HybridSearcher | None = None
        self._keyword_ready = False
        self._lock = threading.RLock()

    # ------------------------------------------------- lazily created parts
    @property
    def embedder(self) -> EmbeddingBackend:
        if self._embedder is None:
            self._embedder = EmbeddingService(
                self.settings.embedding_model, self.settings.embedding_batch_size
            )
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore(
                self.embedder, self.settings.chroma_path, self.settings.collection_name
            )
        return self._vector_store

    @property
    def reranker(self) -> RerankerBackend:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.settings.reranker_model)
        return self._reranker

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = create_llm(self.settings)  # raises LLMAuthError if no API key
        return self._llm

    @property
    def searcher(self) -> HybridSearcher:
        if self._searcher is None:
            self._searcher = HybridSearcher(
                self.vector_store, self.keyword_index, self.settings.rrf_k
            )
        return self._searcher

    # ---------------------------------------------------------- BM25 upkeep
    def _rebuild_keyword_index(self) -> None:
        """BM25 lives in memory, so rebuild it from what ChromaDB holds."""
        self.keyword_index.build(self.vector_store.get_all())
        self._keyword_ready = True

    def _ensure_keyword_index(self) -> None:
        if not self._keyword_ready:
            with self._lock:
                if not self._keyword_ready:
                    self._rebuild_keyword_index()

    # ------------------------------------------------------------ ingestion
    def ingest_documents(self, paths: Sequence[str | Path]) -> IngestionReport:
        """Load -> chunk -> embed -> store each file. One bad file never blocks the rest.

        Re-ingesting a file replaces its old chunks (no duplicates).
        """
        report = IngestionReport()
        with self._lock:
            for raw_path in paths:
                path = Path(raw_path)
                try:
                    document = self.loader.load(path)
                    chunks = self.chunker.chunk_document(document)
                    if not chunks:
                        raise EmptyDocumentError(f"'{path.name}' produced no chunks.")
                    self.vector_store.delete_by_source(document.source)
                    self.vector_store.add_documents(chunks)
                    report.files.append(FileIngestResult(path.name, "ok", len(chunks)))
                except RAGError as exc:
                    logger.error("Failed to ingest %s: %s", path.name, exc)
                    report.files.append(FileIngestResult(path.name, "error", 0, str(exc)))
            if report.ok_count:
                self._rebuild_keyword_index()
        logger.info(
            "Ingestion finished: %d ok, %d failed, %d chunks",
            report.ok_count,
            report.failed_count,
            report.total_chunks,
        )
        return report

    def ingest_directory(self, directory: str | Path | None = None) -> IngestionReport:
        """Ingest every supported file in a folder (default: DOCUMENTS_DIR)."""
        folder = Path(directory or self.settings.documents_dir)
        files = (
            sorted(p for p in folder.iterdir() if p.is_file() and self.loader.is_supported(p))
            if folder.is_dir()
            else []
        )
        if not files:
            logger.warning("No supported documents found in %s", folder)
        return self.ingest_documents(files)

    # ------------------------------------------------------------ retrieval
    def retrieve(
        self,
        question: str,
        top_k_vector: int | None = None,
        top_k_bm25: int | None = None,
        top_k_hybrid: int | None = None,
    ) -> HybridResult:
        """Vector search + BM25, fused with RRF (no reranking yet)."""
        if self.vector_store.count() == 0:
            raise NoDocumentsError(
                "No documents have been ingested yet. Upload documents first "
                "(POST /ingest, or the sidebar in the Streamlit app)."
            )
        self._ensure_keyword_index()
        s = self.settings
        return self.searcher.search(
            question,
            top_k_vector or s.top_k_vector,
            top_k_bm25 or s.top_k_bm25,
            top_k_hybrid or s.top_k_hybrid,
        )

    def rerank(
        self, question: str, candidates: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Cross-encoder rerank of the fused candidates; returns the best ``top_k``."""
        return self.reranker.rerank(question, candidates, top_k or self.settings.top_k_rerank)

    # ----------------------------------------------------------- generation
    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        """Ask the LLM to answer from the (numbered) context chunks."""
        logger.info("Generating answer")
        started = time.perf_counter()
        answer = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, contexts))
        logger.info("LLM answered in %.2fs", time.perf_counter() - started)
        return answer

    def ask(
        self,
        question: str,
        *,
        top_k_vector: int | None = None,
        top_k_bm25: int | None = None,
        top_k_hybrid: int | None = None,
        top_k_rerank: int | None = None,
    ) -> RAGResponse:
        """Full RAG flow: retrieve -> fuse -> rerank -> generate -> answer + sources."""
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        logger.info("Question: %s", question)

        hybrid = self.retrieve(question, top_k_vector, top_k_bm25, top_k_hybrid)
        reranked = self.rerank(question, hybrid.fused, top_k_rerank)
        retrieval = RetrievalResult(
            query=question,
            vector_hits=hybrid.vector_hits,
            bm25_hits=hybrid.bm25_hits,
            fused=hybrid.fused,
            reranked=reranked,
        )

        if not reranked:
            logger.warning("Retrieval returned nothing; skipping the LLM call")
            return RAGResponse(
                question=question,
                answer=NO_ANSWER_MESSAGE,
                sources=[],
                answered=False,
                retrieval=retrieval,
            )

        answer = self.generate(question, reranked)
        return RAGResponse(
            question=question,
            answer=answer,
            sources=reranked,
            cited=extract_citations(answer, len(reranked)),
            answered=NO_ANSWER_MESSAGE.lower() not in answer.lower(),
            retrieval=retrieval,
        )

    # ------------------------------------------------------------ management
    def stats(self) -> dict:
        """Index statistics and active configuration (no model loading needed)."""
        s = self.settings
        sources = self.vector_store.list_sources()
        return {
            "chunks": sum(sources.values()),
            "documents": sources,
            "embedding_model": s.embedding_model,
            "reranker_model": s.reranker_model,
            "llm_provider": s.llm_provider,
            "llm_model": s.resolved_llm_model,
            "chunk_size": s.chunk_size,
            "chunk_overlap": s.chunk_overlap,
            "top_k": {
                "vector": s.top_k_vector,
                "bm25": s.top_k_bm25,
                "hybrid": s.top_k_hybrid,
                "rerank": s.top_k_rerank,
            },
            "rrf_k": s.rrf_k,
        }

    def clear(self) -> None:
        """Delete every stored chunk and empty the BM25 index."""
        with self._lock:
            self.vector_store.delete_collection()
            self.keyword_index.build([])
            self._keyword_ready = True
