"""Text chunking.

WHY DO WE CHUNK?
  1. Embedding models have an input limit and produce ONE vector per input.
     A 50-page PDF squeezed into one vector becomes a blurry average of every
     topic in it, so search quality collapses. Small chunks -> focused vectors.
  2. The LLM has a limited context window and costs money per token. We only
     want to send the few most relevant passages, not whole documents.
  3. Citations become precise: "chunk 4 of refund_policy.pdf" beats "the PDF".

WHY OVERLAP?
  A fact can straddle a chunk boundary ("...refunds are available for" | "30
  days after purchase"). Repeating the last ``chunk_overlap`` characters at the
  start of the next chunk keeps such sentences intact in at least one chunk.

HOW IT WORKS
  A sliding window of ``chunk_size`` characters. Instead of cutting blindly, the
  cut is moved back to the nicest boundary in the second half of the window
  (paragraph > line > sentence > clause > word), and the next window starts on
  a word boundary ``chunk_overlap`` characters earlier.

Limitation: chunks never span PDF pages, so a sentence continuing onto the next
page is split in two.
"""

import logging
import re

from app.ingestion.loader import LoadedDocument
from app.schemas import Chunk

logger = logging.getLogger(__name__)

# Preferred cut points, best first.
_BREAKS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


class TextChunker:
    """Split text (or whole documents) into overlapping chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------ text
    def split_text(self, text: str) -> list[str]:
        """Split one string into chunks of at most ``chunk_size`` characters."""
        text = self._normalize(text)
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start, length = 0, len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            if end < length:
                end = self._find_break(text, start, end)
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= length:
                break
            start = self._next_start(text, start, end)
        return chunks

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _find_break(self, text: str, start: int, end: int) -> int:
        """Move ``end`` back to the best boundary inside the window.

        The search region starts at ``min_end`` so that (a) chunks are at least
        half full and (b) the window always advances even with a large overlap.
        """
        min_end = min(start + max(self.chunk_size // 2, self.chunk_overlap + 1), end)
        for separator in _BREAKS:
            index = text.rfind(separator, min_end, end)
            if index != -1:
                return index + len(separator)
        return end

    def _next_start(self, text: str, start: int, end: int) -> int:
        """Start the next window ``chunk_overlap`` chars back, on a word boundary."""
        candidate = end - self.chunk_overlap
        if self.chunk_overlap and candidate > 0:
            while candidate < end and not text[candidate - 1].isspace():
                candidate += 1
        return max(candidate, start + 1)  # always make progress

    # -------------------------------------------------------------- documents
    def chunk_document(self, document: LoadedDocument) -> list[Chunk]:
        """Chunk every page of a document and attach metadata to each chunk."""
        chunks: list[Chunk] = []
        index = 0
        for page in document.pages:
            for piece in self.split_text(page.text):
                chunk_id = f"{document.source}::chunk-{index}"
                metadata: dict[str, object] = {
                    "source": document.source,
                    "doc_type": document.doc_type,
                    "chunk_index": index,
                    "chunk_id": chunk_id,
                }
                if page.page is not None:
                    metadata["page"] = page.page
                chunks.append(Chunk(chunk_id=chunk_id, text=piece, metadata=metadata))
                index += 1
        logger.info("Created %d chunks from %s", len(chunks), document.source)
        return chunks
