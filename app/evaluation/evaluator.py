"""Evaluation: how good is retrieval, and how good are the answers?

DATASET (JSON list):
    {"question": "...", "expected_answer": "30 days", "expected_sources": ["refund_policy.txt"]}

WHAT COUNTS AS "RELEVANT"?
  A retrieved chunk is relevant if it CONTAINS the expected answer text
  (case/whitespace-insensitive). It is a cheap, automatic proxy that needs no
  hand-labelled chunk ids. Limitation: the answer must appear verbatim in the
  documents, and a chunk can contain the string without truly answering.

RETRIEVAL METRICS (computed at K, for each stage: vector, bm25, hybrid, rerank)
  Recall@K     relevant chunks found in the top K / relevant chunks that exist.
               "Did we find what we needed?"
  Precision@K  relevant chunks in the top K / K.
               "How much of what we returned is useful?"
  MRR          mean of 1 / rank of the FIRST relevant chunk (0 if none).
               "How near the top is the first good result?" 1.0 = always first.

END-TO-END METRICS
  Source coverage    fraction of expected_sources present among the chunks given
                     to the LLM. (No LLM needed.)
  Answer correctness fraction of questions whose generated answer contains the
                     expected answer text. (Needs an LLM: use --generate.)
                     Simple string matching - a real system would add an
                     LLM-as-judge or human review.

Run:  python -m app.evaluation.evaluator --dataset data/eval/eval_dataset.json --k 5
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.logging_config import setup_logging
from app.pipeline import RAGPipeline
from app.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

STAGES = ("vector", "bm25", "hybrid_rrf", "hybrid_rrf+rerank")


# --------------------------------------------------------------------- metrics
def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def contains_answer(text: str, expected: str) -> bool:
    return normalize(expected) in normalize(text)


def recall_at_k(relevant_flags: list[bool], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return min(sum(relevant_flags[:k]), total_relevant) / total_relevant


def precision_at_k(relevant_flags: list[bool], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(relevant_flags[:k]) / k


def reciprocal_rank(relevant_flags: list[bool], k: int) -> float:
    for rank, flag in enumerate(relevant_flags[:k], start=1):
        if flag:
            return 1.0 / rank
    return 0.0


# --------------------------------------------------------------------- results
@dataclass
class QuestionResult:
    question: str
    expected_answer: str
    total_relevant_chunks: int
    stage_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    source_coverage: float | None = None
    answer: str | None = None
    answer_correct: bool | None = None


@dataclass
class EvaluationReport:
    k: int
    questions: list[QuestionResult]
    skipped: list[str]  # questions whose expected answer is not in any chunk
    summary: dict[str, dict[str, float]]
    source_coverage: float | None
    answer_accuracy: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "summary": self.summary,
            "source_coverage": self.source_coverage,
            "answer_accuracy": self.answer_accuracy,
            "skipped_questions": self.skipped,
            "questions": [q.__dict__ for q in self.questions],
        }

    def format(self) -> str:
        lines = [
            f"Retrieval quality @K={self.k} ({len(self.questions)} questions evaluated)",
            f"{'stage':<20}{'Recall@K':>10}{'Precision@K':>13}{'MRR':>8}",
            "-" * 51,
        ]
        for stage in STAGES:
            m = self.summary[stage]
            lines.append(f"{stage:<20}{m['recall']:>10.3f}{m['precision']:>13.3f}{m['mrr']:>8.3f}")
        if self.source_coverage is not None:
            lines.append(f"\nSource coverage : {self.source_coverage:.3f}")
        if self.answer_accuracy is not None:
            lines.append(f"Answer accuracy : {self.answer_accuracy:.3f}")
        if self.skipped:
            lines.append(f"\nSkipped (expected answer not found in any chunk): {self.skipped}")
        return "\n".join(lines)


# ------------------------------------------------------------------ evaluation
def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must be a JSON list of objects.")
    for i, item in enumerate(data):
        if "question" not in item or "expected_answer" not in item:
            raise ValueError(f"Item {i} needs 'question' and 'expected_answer'.")
    return data


def evaluate_pipeline(
    pipeline: RAGPipeline,
    dataset: list[dict[str, Any]],
    k: int | None = None,
    generate: bool = False,
) -> EvaluationReport:
    """Evaluate every stage of retrieval (and optionally the generated answers)."""
    k = k or pipeline.settings.top_k_rerank
    all_chunks = pipeline.vector_store.get_all()
    results: list[QuestionResult] = []
    skipped: list[str] = []

    for item in dataset:
        question, expected = item["question"], item["expected_answer"]
        total_relevant = sum(contains_answer(c.text, expected) for c in all_chunks)
        if total_relevant == 0:
            logger.warning("Expected answer %r is not in any chunk; skipping %r", expected, question)
            skipped.append(question)
            continue

        hybrid = pipeline.retrieve(question)
        reranked = pipeline.rerank(question, hybrid.fused, k)
        stage_lists: dict[str, list[RetrievedChunk]] = {
            "vector": hybrid.vector_hits,
            "bm25": hybrid.bm25_hits,
            "hybrid_rrf": hybrid.fused,
            "hybrid_rrf+rerank": reranked,
        }

        result = QuestionResult(question, expected, total_relevant)
        for stage, hits in stage_lists.items():
            flags = [contains_answer(h.text, expected) for h in hits[:k]]
            result.stage_metrics[stage] = {
                "recall": recall_at_k(flags, total_relevant, k),
                "precision": precision_at_k(flags, k),
                "mrr": reciprocal_rank(flags, k),
            }

        expected_sources = item.get("expected_sources") or []
        if expected_sources:
            found = {c.source for c in reranked}
            result.source_coverage = sum(s in found for s in expected_sources) / len(expected_sources)

        if generate:
            result.answer = pipeline.generate(question, reranked)
            result.answer_correct = contains_answer(result.answer, expected)
        results.append(result)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary = {
        stage: {
            metric: mean([r.stage_metrics[stage][metric] for r in results])
            for metric in ("recall", "precision", "mrr")
        }
        for stage in STAGES
    }
    coverages = [r.source_coverage for r in results if r.source_coverage is not None]
    correct = [r.answer_correct for r in results if r.answer_correct is not None]
    return EvaluationReport(
        k=k,
        questions=results,
        skipped=skipped,
        summary=summary,
        source_coverage=mean(coverages) if coverages else None,
        answer_accuracy=mean([float(c) for c in correct]) if correct else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the hybrid RAG pipeline.")
    parser.add_argument("--dataset", default="data/eval/eval_dataset.json")
    parser.add_argument("--k", type=int, default=None, help="cut-off K (default: TOP_K_RERANK)")
    parser.add_argument("--generate", action="store_true", help="also call the LLM and score answers")
    parser.add_argument("--ingest", action="store_true", help="ingest DOCUMENTS_DIR before evaluating")
    parser.add_argument("--output", default=None, help="write the full report as JSON to this file")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    setup_logging(pipeline.settings.log_level)
    if args.ingest or pipeline.vector_store.count() == 0:
        pipeline.ingest_directory()
    report = evaluate_pipeline(pipeline, load_dataset(args.dataset), args.k, args.generate)
    print(report.format())
    if args.output:
        Path(args.output).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
