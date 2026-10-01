"""Reciprocal Rank Fusion (RRF).

Problem: vector similarity (0..1) and BM25 scores (0..20+) live on different
scales, so adding or averaging them is meaningless. RRF ignores the scores and
uses only the RANK each retriever gave a document:

    RRF(d) = sum over retrievers r of   1 / (k + rank_r(d))

A document ranked high by BOTH retrievers wins; one ranked high by only one
still gets credit. ``k`` (default 60, from the original paper) dampens the
advantage of rank 1 over rank 2 so no single retriever dominates.

Worked example, k=60:
    vector: [A, B]      bm25: [B, C]
    A = 1/61            = 0.01639
    B = 1/62 + 1/61     = 0.03252   <- found by both -> first
    C = 1/62            = 0.01613
"""

from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse several ranked id lists into one ranked ``(id, rrf_score)`` list.

    Each inner sequence is one retriever's results, best first. Ties keep the
    order in which ids were first seen, so the output is deterministic.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):  # rank is 1-based
            if doc_id in seen:  # ignore accidental duplicates inside one list
                continue
            seen.add(doc_id)
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])  # stable sort
