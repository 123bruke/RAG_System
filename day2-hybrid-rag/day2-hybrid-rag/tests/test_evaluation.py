import json
from pathlib import Path

import pytest

from app.evaluation.evaluator import (
    contains_answer,
    evaluate_pipeline,
    load_dataset,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

DATASET = Path(__file__).resolve().parents[1] / "data" / "eval" / "eval_dataset.json"


def test_metric_definitions():
    flags = [False, True, False, True]
    assert precision_at_k(flags, 4) == 0.5
    assert precision_at_k(flags, 2) == 0.5
    assert recall_at_k(flags, total_relevant=2, k=2) == 0.5
    assert recall_at_k(flags, total_relevant=2, k=4) == 1.0
    assert reciprocal_rank(flags, 4) == 0.5
    assert reciprocal_rank([False, False], 2) == 0.0
    assert recall_at_k([], total_relevant=0, k=3) == 0.0


def test_contains_answer_ignores_case_and_whitespace():
    assert contains_answer("Refunds within  30\ndays.", "30 DAYS")
    assert not contains_answer("Refunds within 14 days", "30 days")


def test_load_dataset_validates_items(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"question": "no answer field"}]))
    with pytest.raises(ValueError):
        load_dataset(bad)
    assert len(load_dataset(DATASET)) >= 5


def test_end_to_end_evaluation_with_fakes(pipeline, sample_files):
    pipeline.ingest_documents(sample_files)
    report = evaluate_pipeline(pipeline, load_dataset(DATASET), k=3, generate=True)

    assert report.skipped == []
    assert len(report.questions) == len(load_dataset(DATASET))
    for stage in ("vector", "bm25", "hybrid_rrf", "hybrid_rrf+rerank"):
        metrics = report.summary[stage]
        assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert report.summary["hybrid_rrf+rerank"]["mrr"] > 0.5
    assert report.source_coverage is not None and report.source_coverage > 0.5
    assert report.answer_accuracy is not None  # FakeLLM always says "30 days"
    assert "Recall@K" in report.format()
    json.dumps(report.to_dict())  # report is JSON-serialisable


def test_questions_with_unfindable_answers_are_skipped(pipeline, sample_files):
    pipeline.ingest_documents(sample_files)
    report = evaluate_pipeline(pipeline, [{"question": "q?", "expected_answer": "not anywhere in docs"}], k=3)
    assert report.skipped == ["q?"] and report.questions == []
