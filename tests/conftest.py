"""Shared fixtures: fake embedder / reranker / LLM so tests need no models or API keys."""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from app.config import Settings
from app.generation.llm import BaseLLM
from app.pipeline import RAGPipeline
from app.retrieval.keyword_search import tokenize
from app.schemas import RetrievedChunk


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (similar words -> similar vectors)."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)


class FakeReranker:
    """Scores candidates by how many distinct non-stopword query tokens they contain.

    A deterministic stand-in for the cross-encoder. It checks the pipeline's
    wiring (ordering, top-k, scores attached); it does NOT judge semantic quality.
    """

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        wanted = set(tokenize(query))
        scored = [
            replace(c, rerank_score=float(len(wanted & set(tokenize(c.text))))) for c in candidates
        ]
        scored.sort(key=lambda c: c.rerank_score, reverse=True)
        return scored[:top_k]


class FakeLLM(BaseLLM):
    """Records the prompts it receives and returns a canned, cited answer."""

    def __init__(self, answer: str = "The refund period is 30 days. [1]") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.answer


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        chroma_path=str(tmp_path / "chroma"),
        documents_dir=str(tmp_path / "docs"),
        chunk_size=300,
        chunk_overlap=50,
        top_k_vector=6,
        top_k_bm25=6,
        top_k_hybrid=6,
        top_k_rerank=3,
    )


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def pipeline(settings: Settings, llm: FakeLLM) -> RAGPipeline:
    return RAGPipeline(settings, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=llm)


@pytest.fixture
def sample_files(tmp_path: Path) -> list[Path]:
    """Copies of the three demo documents in a temp folder."""
    source_dir = Path(__file__).resolve().parents[1] / "data" / "documents"
    target = tmp_path / "samples"
    target.mkdir()
    files = []
    for f in sorted(source_dir.glob("*.txt")):
        copy = target / f.name
        copy.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
        files.append(copy)
    return files
