"""Application-specific exceptions.

Every error we raise on purpose inherits from :class:`RAGError`, so callers
(the API, the Streamlit UI, tests) can catch one base class and still show a
precise, human-readable message. Unexpected bugs are *not* wrapped: they
propagate so they are never silently swallowed.
"""


class RAGError(Exception):
    """Base class for all expected application errors."""


# --- document ingestion -----------------------------------------------------
class DocumentNotFoundError(RAGError):
    """The given file path does not exist."""


class UnsupportedFileError(RAGError):
    """The file extension is not supported (only PDF, TXT and DOCX)."""


class EmptyDocumentError(RAGError):
    """The document contains no extractable text (e.g. a scanned PDF)."""


class CorruptedDocumentError(RAGError):
    """The file could not be parsed (corrupted, encrypted, wrong format)."""


# --- retrieval --------------------------------------------------------------
class EmbeddingError(RAGError):
    """The embedding model failed to load or to encode text."""


class VectorStoreError(RAGError):
    """ChromaDB failed to read or write."""


class RetrievalError(RAGError):
    """Keyword search, fusion or reranking failed."""


class NoDocumentsError(RAGError):
    """A question was asked but nothing has been ingested yet."""


# --- generation -------------------------------------------------------------
class LLMError(RAGError):
    """The LLM provider call failed (network, rate limit, blocked output...)."""


class LLMAuthError(LLMError):
    """The LLM API key is missing or was rejected by the provider."""


# --- configuration ----------------------------------------------------------
class ConfigurationError(RAGError):
    """The application is configured incorrectly."""
