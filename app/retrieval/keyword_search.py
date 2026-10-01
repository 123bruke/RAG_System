"""BM25 keyword (lexical) search using rank-bm25.

VECTOR SEARCH  = finds text with similar MEANING ("car" ~ "automobile").
BM25           = finds text containing the same WORDS / TOKENS, weighting rare
                 tokens more (a token found in 1 of 1000 chunks is a strong
                 signal; "the" is worthless).

Why we need both: embeddings blur exact identifiers. For the query
"HP:0001945" a vector model may return any phenotype term, but BM25 sees the
rare token ``hp:0001945`` and ranks "HP:0001945 Fever" first.

Tokenizer: lower-cased word tokens that keep compound identifiers intact
(``hp:0001945``, ``invoice-2024-001``) and ALSO emit their parts (``hp``,
``0001945``), so both exact and partial matches work.

The BM25 index lives in memory (rank-bm25 has no persistence); the pipeline
rebuilds it from ChromaDB at start-up and after every ingest/clear.
"""

import logging
import re
import threading
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from app.schemas import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[:_\-./][a-z0-9]+)*")
_SPLIT_RE = re.compile(r"[:_\-./]")
_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from has have how i if in is it its "
    "of on or that the their this to was were what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case, keep compound identifiers, add their parts, drop stopwords."""
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        if match in _STOPWORDS:
            continue
        tokens.append(match)
        if _SPLIT_RE.search(match):
            tokens.extend(p for p in _SPLIT_RE.split(match) if p and p not in _STOPWORDS)
    return tokens


class KeywordIndex:
    """In-memory BM25 index over chunks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
        self._token_sets: list[frozenset[str]] = []

    @property
    def count(self) -> int:
        return len(self._chunks)

    def build(self, chunks: Sequence[Chunk]) -> None:
        """(Re)build the index from scratch. An empty list clears it."""
        tokenized = [tokenize(c.text) for c in chunks]
        if not chunks or not any(tokenized):
            bm25, kept, sets = None, [], []
        else:
            bm25 = BM25Okapi(tokenized)
            kept = list(chunks)
            sets = [frozenset(t) for t in tokenized]
        with self._lock:  # swap in one step so readers never see a half-built index
            self._bm25, self._chunks, self._token_sets = bm25, kept, sets
        logger.info("BM25 index built over %d chunks", len(kept))

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks sharing at least one token with the query."""
        with self._lock:
            bm25, chunks, token_sets = self._bm25, self._chunks, self._token_sets
        query_tokens = tokenize(query)
        if bm25 is None or not query_tokens or top_k <= 0:
            return []

        scores = bm25.get_scores(query_tokens)
        query_set = set(query_tokens)
        # Keep only chunks with a real lexical match. (BM25's IDF can be zero or
        # negative on tiny corpora, so "score > 0" would wrongly drop valid hits.)
        matching = [i for i in range(len(chunks)) if query_set & token_sets[i]]
        matching.sort(key=lambda i: (-scores[i], i))

        hits = [
            RetrievedChunk(
                chunk_id=chunks[i].chunk_id,
                text=chunks[i].text,
                metadata=dict(chunks[i].metadata),
                bm25_rank=rank,
                bm25_score=float(scores[i]),
            )
            for rank, i in enumerate(matching[:top_k], start=1)
        ]
        logger.info("BM25 returned %d results", len(hits))
        return hits
