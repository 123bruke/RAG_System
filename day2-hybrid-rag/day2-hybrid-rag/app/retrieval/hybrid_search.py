"""Hybrid search = vector search + BM25, fused with RRF."""

import logging
from dataclasses import replace

from app.retrieval.keyword_search import KeywordIndex
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.vector_store import VectorStore
from app.schemas import HybridResult, RetrievedChunk

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Run both retrievers, then merge their rankings with Reciprocal Rank Fusion."""

    def __init__(self, vector_store: VectorStore, keyword_index: KeywordIndex, rrf_k: int = 60) -> None:
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.rrf_k = rrf_k

    def search(
        self, query: str, top_k_vector: int, top_k_bm25: int, top_k_hybrid: int
    ) -> HybridResult:
        vector_hits = self.vector_store.search(query, top_k_vector)
        bm25_hits = self.keyword_index.search(query, top_k_bm25)

        # Merge the two hit lists by chunk id into copies carrying both sets of scores.
        merged: dict[str, RetrievedChunk] = {h.chunk_id: replace(h) for h in vector_hits}
        for hit in bm25_hits:
            if hit.chunk_id in merged:
                merged[hit.chunk_id].bm25_rank = hit.bm25_rank
                merged[hit.chunk_id].bm25_score = hit.bm25_score
            else:
                merged[hit.chunk_id] = replace(hit)

        fused_scores = reciprocal_rank_fusion(
            [[h.chunk_id for h in vector_hits], [h.chunk_id for h in bm25_hits]],
            k=self.rrf_k,
        )
        fused: list[RetrievedChunk] = []
        for chunk_id, score in fused_scores[:top_k_hybrid]:
            candidate = merged[chunk_id]
            candidate.rrf_score = score
            fused.append(candidate)

        logger.info("RRF produced %d candidates", len(fused))
        return HybridResult(vector_hits=vector_hits, bm25_hits=bm25_hits, fused=fused)
