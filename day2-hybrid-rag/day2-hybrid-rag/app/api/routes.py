"""FastAPI routes: /health, /stats, /ingest, /query."""

import logging
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Response, UploadFile
from pydantic import BaseModel, StringConstraints

from app.ingestion.loader import DocumentLoader
from app.pipeline import RAGPipeline
from app.schemas import FileIngestResult

logger = logging.getLogger(__name__)
router = APIRouter()


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """One shared pipeline per process (tests override this dependency)."""
    return RAGPipeline()


# ------------------------------------------------------------------- models
class QueryRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    top_k_rerank: int | None = None  # optional override of TOP_K_RERANK


class SourceOut(BaseModel):
    source: str
    chunk: int
    chunk_id: str
    page: int | None = None
    rerank_score: float | None = None
    rrf_score: float | None = None
    cited: bool = False
    text: str


class QueryResponse(BaseModel):
    answer: str
    answered: bool
    sources: list[SourceOut]


class FileResultOut(BaseModel):
    filename: str
    status: str
    chunks: int
    error: str | None = None


class IngestResponse(BaseModel):
    succeeded: int
    failed: int
    total_chunks: int
    files: list[FileResultOut]


# ---------------------------------------------------------------- endpoints
@router.get("/health")
def health() -> dict[str, str]:
    """Liveness check (does not load any model)."""
    return {"status": "ok"}


@router.get("/stats")
def stats(pipeline: RAGPipeline = Depends(get_pipeline)) -> dict:
    """Number of chunks, documents and the active configuration."""
    return pipeline.stats()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> QueryResponse:
    """Answer a question from the ingested documents, with sources."""
    result = pipeline.ask(request.question, top_k_rerank=request.top_k_rerank)
    return QueryResponse(
        answer=result.answer,
        answered=result.answered,
        sources=[
            SourceOut(
                source=chunk.source,
                chunk=chunk.chunk_index,
                chunk_id=chunk.chunk_id,
                page=chunk.page,
                rerank_score=chunk.rerank_score,
                rrf_score=chunk.rrf_score,
                cited=number in result.cited,
                text=chunk.text,
            )
            for number, chunk in enumerate(result.sources, start=1)
        ],
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    response: Response,
    files: list[UploadFile] | None = File(default=None),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> IngestResponse:
    """Upload PDF/TXT/DOCX files and index them.

    Send no files to (re-)index everything in the server's documents folder.
    A bad file is reported in ``files`` with status "error"; the others still
    succeed. If *every* file fails the status code is 422.
    """
    results: list[FileIngestResult] = []
    if not files:
        report = pipeline.ingest_directory()
        results.extend(report.files)
    else:
        target_dir = Path(pipeline.settings.documents_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for upload in files:
            name = Path(upload.filename or "").name  # strips any path -> no traversal
            if not name or name.startswith(".") or not DocumentLoader.is_supported(name):
                allowed = ", ".join(DocumentLoader.supported_extensions())
                results.append(
                    FileIngestResult(
                        name or "(unnamed)", "error", 0,
                        f"Unsupported or invalid file. Supported types: {allowed}.",
                    )
                )
                continue
            destination = target_dir / name
            with destination.open("wb") as out:
                shutil.copyfileobj(upload.file, out)
            saved.append(destination)
        if saved:
            results.extend(pipeline.ingest_documents(saved).files)

    ok = sum(1 for r in results if r.status == "ok")
    if results and ok == 0:
        response.status_code = 422
    return IngestResponse(
        succeeded=ok,
        failed=len(results) - ok,
        total_chunks=sum(r.chunks for r in results),
        files=[FileResultOut(**r.__dict__) for r in results],
    )
