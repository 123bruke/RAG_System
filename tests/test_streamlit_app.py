"""Drives the real streamlit_app.py headlessly (fake models, no network)."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.config import get_settings
from app.pipeline import RAGPipeline
from tests.conftest import FakeEmbedder, FakeLLM, FakeReranker

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app_test(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("DOCUMENTS_DIR", str(ROOT / "data" / "documents"))
    monkeypatch.setattr(RAGPipeline, "embedder", property(lambda self: FakeEmbedder()))
    monkeypatch.setattr(RAGPipeline, "reranker", property(lambda self: FakeReranker()))
    monkeypatch.setattr(RAGPipeline, "llm", property(lambda self: FakeLLM("Refunds are possible within 30 days. [1]")))
    get_settings.cache_clear()
    st.cache_resource.clear()
    yield AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30)
    get_settings.cache_clear()
    st.cache_resource.clear()


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_page_loads_with_title_and_empty_index(app_test):
    at = app_test.run()
    assert not at.exception
    assert at.title[0].value == "🤖 Hybrid RAG Knowledge Assistant"
    assert at.text_input[0].label == "Ask a question about your documents..."
    assert at.metric[0].value == "0"


def test_ingest_ask_and_inspect_results(app_test):
    at = app_test.run()
    _button(at, "Ingest sample documents").click().run()
    assert not at.exception
    assert int(at.metric[0].value) > 3
    assert len(at.success) >= 3  # at least one success banner per sample file

    at.text_input[0].set_value("HP:0001945").run()
    _button(at, "Ask").click().run()
    assert not at.exception
    assert any("Refunds are possible within 30 days" in m.value for m in at.markdown)
    assert len(at.tabs) == 4  # vector / BM25 / RRF / reranker
    assert len(at.dataframe) >= 3
    # numbered source lines with citation info, and one expander per source chunk
    assert any(m.value.startswith("[1] ") and "chunk" in m.value for m in at.markdown)
    assert len(at.expander) == 5  # slider default = TOP_K_RERANK (5): one expander per source chunk
    assert any(m.value.startswith("[1] ") and "✅ cited" in m.value for m in at.markdown)  # FakeLLM cites [1]


def test_question_before_ingest_shows_a_helpful_warning(app_test):
    at = app_test.run()
    at.text_input[0].set_value("What is the refund period?").run()
    _button(at, "Ask").click().run()
    assert not at.exception
    assert "No documents have been ingested" in at.warning[0].value


def test_clear_database_requires_confirmation_then_empties_index(app_test):
    at = app_test.run()
    _button(at, "Ingest sample documents").click().run()
    assert _button(at, "Clear database").disabled
    at.checkbox[0].check().run()
    _button(at, "Clear database").click().run()
    assert not at.exception
    assert at.metric[0].value == "0"
