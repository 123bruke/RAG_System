"""Logging setup shared by the API, the Streamlit app and the evaluator."""

import logging

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers", "google_genai")


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once (safe to call repeatedly, e.g. on Streamlit reruns)."""
    root = logging.getLogger()
    if not any(getattr(h, "_rag_handler", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler._rag_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level.upper())
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
