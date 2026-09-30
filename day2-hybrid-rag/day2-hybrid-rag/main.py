"""FastAPI entry point.  Run with:  uvicorn main:app --reload"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import router
from app.config import get_settings
from app.exceptions import (
    CorruptedDocumentError,
    DocumentNotFoundError,
    EmptyDocumentError,
    LLMAuthError,
    LLMError,
    NoDocumentsError,
    RAGError,
    UnsupportedFileError,
)
from app.logging_config import setup_logging

# Most specific first: the first isinstance() match wins.
_STATUS_CODES: list[tuple[type[RAGError], int]] = [
    (UnsupportedFileError, 415),
    (EmptyDocumentError, 422),
    (CorruptedDocumentError, 422),
    (DocumentNotFoundError, 404),
    (NoDocumentsError, 409),
    (LLMAuthError, 503),  # our server is misconfigured, not the client's fault
    (LLMError, 502),  # upstream provider problem
]


def create_app() -> FastAPI:
    setup_logging(get_settings().log_level)
    api = FastAPI(
        title="Hybrid RAG Knowledge Assistant",
        version=__version__,
        description="Hybrid (vector + BM25) retrieval with RRF, cross-encoder reranking and cited answers.",
    )
    api.include_router(router)

    @api.exception_handler(RAGError)
    async def handle_rag_error(_: Request, exc: RAGError) -> JSONResponse:
        status = next((code for cls, code in _STATUS_CODES if isinstance(exc, cls)), 500)
        return JSONResponse(status_code=status, content={"error": type(exc).__name__, "detail": str(exc)})

    @api.exception_handler(ValueError)
    async def handle_value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": "ValueError", "detail": str(exc)})

    return api


app = create_app()
