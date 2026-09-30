"""Streamlit UI.  Run with:  streamlit run streamlit_app.py"""

import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import get_settings
from app.exceptions import NoDocumentsError, RAGError
from app.ingestion.loader import DocumentLoader
from app.logging_config import setup_logging
from app.pipeline import RAGPipeline
from app.schemas import IngestionReport, RetrievedChunk

st.set_page_config(page_title="Hybrid RAG Knowledge Assistant", page_icon="🤖", layout="wide")


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    """Create the pipeline once per server process (models load lazily on first use)."""
    setup_logging(get_settings().log_level)
    return RAGPipeline()


def show_ingestion_report(report: IngestionReport) -> None:
    for result in report.files:
        if result.status == "ok":
            st.success(f"✅ {result.filename}: {result.chunks} chunks indexed")
        else:
            st.error(f"❌ {result.filename}: {result.error}")


def hits_table(hits: list[RetrievedChunk], columns: dict[str, str]) -> pd.DataFrame:
    """Turn retrieved chunks into a table. ``columns`` maps attribute -> column title."""
    rows = []
    for i, hit in enumerate(hits, start=1):
        row = {"#": i, "Source": hit.source, "Chunk": hit.chunk_index}
        for attribute, title in columns.items():
            value = getattr(hit, attribute)
            row[title] = round(value, 4) if isinstance(value, float) else value
        row["Preview"] = hit.text[:110].replace("\n", " ") + ("…" if len(hit.text) > 110 else "")
        rows.append(row)
    return pd.DataFrame(rows)


pipeline = get_pipeline()
settings = pipeline.settings

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("📁 Documents")
    uploads = st.file_uploader(
        "Upload documents",
        type=[e.lstrip(".") for e in DocumentLoader.supported_extensions()],
        accept_multiple_files=True,
    )
    if st.button("Ingest uploaded files", disabled=not uploads, use_container_width=True):
        target = Path(settings.documents_dir)
        target.mkdir(parents=True, exist_ok=True)
        saved = []
        for upload in uploads:
            destination = target / Path(upload.name).name
            with destination.open("wb") as out:
                shutil.copyfileobj(upload, out)
            saved.append(destination)
        with st.spinner("Loading, chunking and embedding…"):
            try:
                show_ingestion_report(pipeline.ingest_documents(saved))
            except RAGError as exc:
                st.error(str(exc))
    if st.button("Ingest sample documents", use_container_width=True):
        with st.spinner("Indexing data/documents…"):
            try:
                show_ingestion_report(pipeline.ingest_directory())
            except RAGError as exc:
                st.error(str(exc))

    st.header("⚙️ Retrieval settings")
    n_retrieved = st.slider("Retrieved documents (per retriever)", 3, 30, settings.top_k_vector)
    n_reranked = st.slider("Reranked documents (sent to LLM)", 1, 10, settings.top_k_rerank)

    st.header("📊 Index")
    try:
        stats = pipeline.stats()
        st.metric("Chunks indexed", stats["chunks"])
        for name, count in stats["documents"].items():
            st.caption(f"• {name} — {count} chunks")
    except RAGError as exc:
        st.error(str(exc))
    st.caption(f"LLM: {settings.llm_provider} / {settings.resolved_llm_model}")

    st.header("🗑️ Danger zone")
    confirm = st.checkbox("Yes, delete everything")
    if st.button("Clear database", disabled=not confirm, use_container_width=True):
        pipeline.clear()
        st.session_state.pop("response", None)
        st.rerun()

# --------------------------------------------------------------------- main
st.title("🤖 Hybrid RAG Knowledge Assistant")
st.caption("Vector search + BM25 → Reciprocal Rank Fusion → cross-encoder reranking → cited answer")

with st.form("ask"):
    question = st.text_input("Ask a question about your documents...")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("Searching, reranking and generating…"):
        try:
            st.session_state["response"] = pipeline.ask(
                question,
                top_k_vector=n_retrieved,
                top_k_bm25=n_retrieved,
                top_k_hybrid=n_retrieved,
                top_k_rerank=n_reranked,
            )
        except NoDocumentsError as exc:
            st.session_state.pop("response", None)
            st.warning(str(exc))
        except RAGError as exc:
            st.session_state.pop("response", None)
            st.error(f"{type(exc).__name__}: {exc}")

response = st.session_state.get("response")
if response is not None:
    st.subheader("Answer")
    st.markdown(response.answer)
    if response.sources:
        st.markdown("**Sources**")
        for number, source in enumerate(response.sources, start=1):
            mark = " ✅ cited" if number in response.cited else ""
            st.markdown(f"[{number}] {source.citation()}{mark}")

    retrieval = response.retrieval
    if retrieval is not None:
        st.subheader("Retrieval process")
        tab_vec, tab_bm25, tab_rrf, tab_rerank = st.tabs(
            ["1 · Vector search", "2 · BM25", "3 · RRF fusion", "4 · Reranker"]
        )
        with tab_vec:
            st.caption("Meaning-based. Score = cosine similarity (higher is closer).")
            st.dataframe(hits_table(retrieval.vector_hits, {"vector_score": "Similarity"}), hide_index=True)
        with tab_bm25:
            st.caption("Keyword-based. Only chunks sharing words with the question appear.")
            if retrieval.bm25_hits:
                st.dataframe(hits_table(retrieval.bm25_hits, {"bm25_score": "BM25 score"}), hide_index=True)
            else:
                st.info("No chunk shares a keyword with the question.")
        with tab_rrf:
            st.caption("Rank-based merge: RRF = Σ 1 / (k + rank). Chunks found by both retrievers rise.")
            st.dataframe(
                hits_table(
                    retrieval.fused,
                    {"rrf_score": "RRF score", "vector_rank": "Vector rank", "bm25_rank": "BM25 rank"},
                ),
                hide_index=True,
            )
        with tab_rerank:
            st.caption("Cross-encoder reads (question, chunk) together. Higher = more relevant (not a probability).")
            st.dataframe(
                hits_table(retrieval.reranked, {"rerank_score": "Reranker score", "rrf_score": "RRF score"}),
                hide_index=True,
            )

    if response.sources:
        st.subheader("Retrieved chunks (sent to the LLM)")
        for number, source in enumerate(response.sources, start=1):
            score = f"{source.rerank_score:.2f}" if source.rerank_score is not None else "n/a"
            with st.expander(f"[{number}] {source.citation()} · relevance {score}"):
                st.caption(f"chunk_id: {source.chunk_id}")
                st.write(source.text)
