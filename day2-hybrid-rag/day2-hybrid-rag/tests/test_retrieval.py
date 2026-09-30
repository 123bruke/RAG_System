import pytest

from app.ingestion.chunker import TextChunker  # noqa: F401  (import check)
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.keyword_search import KeywordIndex, tokenize
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.vector_store import VectorStore
from app.schemas import Chunk
from tests.conftest import FakeEmbedder

HPO = [
    ("HP:0001945 Fever - Elevated body temperature.", "hpo.txt"),
    ("HP:0002315 Headache - Pain in the head.", "hpo.txt"),
    ("HP:0012378 Fatigue - A feeling of tiredness.", "hpo.txt"),
    ("HP:0002094 Dyspnea - Difficult or labored breathing.", "hpo.txt"),
    ("HP:0012735 Cough - Sudden expulsion of air from the lungs.", "hpo.txt"),
    ("HP:0002018 Nausea - An urge to vomit.", "hpo.txt"),
    ("Refunds are available within 30 days of purchase.", "refund.txt"),
]


def make_chunks() -> list[Chunk]:
    return [
        Chunk(f"{src}::chunk-{i}", text, {"source": src, "chunk_index": i, "chunk_id": f"{src}::chunk-{i}"})
        for i, (text, src) in enumerate(HPO)
    ]


@pytest.fixture
def store(tmp_path):
    vs = VectorStore(FakeEmbedder(), tmp_path / "chroma", "test_docs")
    vs.add_documents(make_chunks())
    return vs


@pytest.fixture
def index():
    idx = KeywordIndex()
    idx.build(make_chunks())
    return idx


# ------------------------------------------------------------------- RRF
def test_rrf_matches_the_formula():
    fused = dict(reciprocal_rank_fusion([["A", "B"], ["B", "C"]], k=60))
    assert fused["A"] == pytest.approx(1 / 61)
    assert fused["B"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["C"] == pytest.approx(1 / 62)


def test_rrf_ranks_documents_found_by_both_retrievers_first():
    ranked = reciprocal_rank_fusion([["A", "B"], ["B", "C"]])
    assert [doc for doc, _ in ranked] == ["B", "A", "C"]


def test_rrf_k_changes_scores():
    low = dict(reciprocal_rank_fusion([["A"]], k=1))["A"]
    high = dict(reciprocal_rank_fusion([["A"]], k=100))["A"]
    assert low > high


def test_rrf_handles_empty_and_invalid_input():
    assert reciprocal_rank_fusion([[], []]) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["A"]], k=0)


# ------------------------------------------------------------------ BM25
def test_tokenizer_keeps_identifiers_and_their_parts():
    tokens = tokenize("HP:0001945 Fever")
    assert "hp:0001945" in tokens and "0001945" in tokens and "fever" in tokens


def test_bm25_finds_exact_identifier(index):
    hits = index.search("HP:0001945", top_k=3)
    assert hits[0].text.startswith("HP:0001945 Fever")
    assert hits[0].bm25_rank == 1


def test_bm25_returns_nothing_without_keyword_overlap(index):
    assert index.search("zebra", top_k=5) == []


def test_bm25_empty_index_returns_nothing():
    assert KeywordIndex().search("anything", top_k=5) == []


def test_bm25_respects_top_k(index):
    assert len(index.search("HP", top_k=2)) == 2


# ---------------------------------------------------------------- vector
def test_vector_search_ranks_by_similarity(store):
    hits = store.search("difficult labored breathing", top_k=3)
    assert hits[0].text.startswith("HP:0002094 Dyspnea")
    assert hits[0].vector_rank == 1 and hits[0].vector_score > hits[1].vector_score


def test_vector_store_count_and_upsert_is_idempotent(store):
    assert store.count() == len(HPO)
    store.add_documents(make_chunks())
    assert store.count() == len(HPO)


def test_vector_store_persists_between_instances(store, tmp_path):
    reopened = VectorStore(FakeEmbedder(), tmp_path / "chroma", "test_docs")
    assert reopened.count() == len(HPO)


def test_delete_collection_empties_store_but_keeps_it_usable(store):
    store.delete_collection()
    assert store.count() == 0 and store.search("fever", 3) == []
    store.add_documents(make_chunks()[:2])
    assert store.count() == 2


def test_delete_by_source(store):
    store.delete_by_source("refund.txt")
    assert store.count() == len(HPO) - 1
    assert "refund.txt" not in store.list_sources()


# ---------------------------------------------------------------- hybrid
def test_hybrid_search_merges_both_retrievers(store, index):
    searcher = HybridSearcher(store, index, rrf_k=60)
    result = searcher.search("HP:0001945", top_k_vector=4, top_k_bm25=4, top_k_hybrid=5)
    assert result.fused[0].text.startswith("HP:0001945 Fever")
    top = result.fused[0]
    assert top.vector_rank is not None and top.bm25_rank is not None  # found by both
    assert top.rrf_score == pytest.approx(1 / (60 + top.vector_rank) + 1 / (60 + top.bm25_rank))
    scores = [c.rrf_score for c in result.fused]
    assert scores == sorted(scores, reverse=True)
    assert len(result.fused) <= 5


def test_hybrid_includes_bm25_only_hits(store, index):
    searcher = HybridSearcher(store, index)
    # top_k_vector=1 -> vector search returns just one chunk; BM25 adds the others.
    result = searcher.search("HP", top_k_vector=1, top_k_bm25=5, top_k_hybrid=10)
    assert len(result.vector_hits) == 1
    assert len(result.fused) > 1
    assert any(c.vector_rank is None and c.bm25_rank is not None for c in result.fused)


# -------------------------------------------------------------- reranker
def test_cross_encoder_reranker_orders_by_score_and_rejects_bad_model_output():
    from app.exceptions import RetrievalError
    from app.retrieval.reranker import CrossEncoderReranker
    from app.schemas import RetrievedChunk

    class FakeCrossEncoder:
        def __init__(self, scores):
            self.scores, self.pairs = scores, None

        def predict(self, pairs, show_progress_bar=False):
            self.pairs = pairs
            return self.scores

    candidates = [RetrievedChunk(f"id{i}", f"text {i}", {"source": "a.txt"}) for i in range(3)]
    reranker = CrossEncoderReranker()
    reranker._model = FakeCrossEncoder([-3.0, 7.5, 0.2])

    top = reranker.rerank("my question", candidates, top_k=2)
    assert [c.chunk_id for c in top] == ["id1", "id2"]
    assert top[0].rerank_score == 7.5
    assert reranker._model.pairs[0] == ("my question", "text 0")  # (query, document) pairs

    reranker._model = FakeCrossEncoder([1.0])  # wrong number of scores
    with pytest.raises(RetrievalError, match="scores"):
        reranker.rerank("q", candidates, top_k=2)
    assert reranker.rerank("q", [], top_k=2) == []
