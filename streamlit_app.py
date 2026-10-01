"""Streamlit UI. Run with: streamlit run streamlit_app.py"""

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

st.set_page_config(
    page_title="Hybrid RAG Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ Custom ChatGPT-like Styling
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Background & Container */
    .stApp {
        background: radial-gradient(circle at top right, #171d26, #0e1217 60%, #090c10);
        color: #e6edf3;
    }

    /* Top Header & Badges */
    .chat-header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 0.5rem;
        margin-bottom: 0.2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .chat-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(16, 163, 127, 0.15);
        color: #10a37f;
        border: 1px solid rgba(16, 163, 127, 0.3);
        border-radius: 9999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .pipeline-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.4rem;
        margin-bottom: 1rem;
    }
    .pipeline-pill {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        color: #8b949e;
    }
    .pipeline-pill.active {
        color: #58a6ff;
        border-color: rgba(88, 166, 255, 0.3);
        background: rgba(88, 166, 255, 0.1);
    }

    /* Message Cards */
    .user-message-card {
        background: rgba(33, 38, 45, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.25rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    .assistant-message-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.85) 0%, rgba(13, 17, 23, 0.95) 100%);
        border: 1px solid rgba(16, 163, 127, 0.35);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(16, 163, 127, 0.08), 0 2px 8px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }
    .msg-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .user-avatar {
        background: #30363d;
        color: #f0f6fc;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
    }
    .assistant-avatar {
        background: linear-gradient(135deg, #10a37f, #0d8a6a);
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
        box-shadow: 0 0 10px rgba(16, 163, 127, 0.5);
    }

    /* Hero / Empty State */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px dashed rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        background: rgba(22, 27, 34, 0.4);
    }
    .hero-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        display: inline-block;
        filter: drop-shadow(0 0 12px rgba(16, 163, 127, 0.4));
    }
    .hero-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f0f6fc;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        color: #8b949e;
        font-size: 0.9rem;
        max-width: 540px;
        margin: 0 auto 1.25rem auto;
        line-height: 1.5;
    }

    /* Citation Badges */
    .source-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.65rem;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 500;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #c9d1d9;
        margin: 0.2rem 0.3rem 0.2rem 0;
    }
    .source-pill.cited {
        background: rgba(16, 163, 127, 0.15);
        border-color: rgba(16, 163, 127, 0.4);
        color: #2ea043;
        font-weight: 600;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0b0e14 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] .stMarkdown h3 {
        color: #f0f6fc;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.88rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.06);
        color: #f0f6fc;
    }
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(255, 255, 255, 0.25);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }

    /* Form Submit (Ask) Button */
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #10a37f 0%, #0d8a6a 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 10px rgba(16, 163, 127, 0.3) !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, #13b890 0%, #0fa07c 100%) !important;
        box-shadow: 0 4px 16px rgba(16, 163, 127, 0.45) !important;
        transform: translateY(-1px);
    }

    /* Text Input */
    [data-testid="stTextInput"] input {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        background: rgba(13, 17, 23, 0.8) !important;
        color: #f0f6fc !important;
        padding: 0.65rem 1rem !important;
        font-size: 0.95rem !important;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2);
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #10a37f !important;
        box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.25) !important;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.75rem 1rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        color: #8b949e !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    [data-testid="stMetricValue"] {
        color: #10a37f !important;
        font-weight: 700 !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: rgba(22, 27, 34, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        margin-bottom: 0.6rem !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(22, 27, 34, 0.5);
        padding: 0.3rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.4rem 0.9rem;
        color: #8b949e;
        font-weight: 500;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(16, 163, 127, 0.15) !important;
        color: #10a37f !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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

# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:1rem;">
            <div style="background:#10a37f; border-radius:8px; width:32px; height:32px; display:flex; align-items:center; justify-content:center; color:white; font-size:1.1rem; box-shadow:0 0 12px rgba(16,163,127,0.4);">⚡</div>
            <div>
                <div style="font-weight:700; font-size:1rem; color:#f0f6fc; line-height:1.1;">Hybrid RAG</div>
                <div style="font-size:0.75rem; color:#8b949e;">Multi-Stage Knowledge Base</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("📁 Documents")
    uploads = st.file_uploader(
        "Upload documents",
        type=[e.lstrip(".") for e in DocumentLoader.supported_extensions()],
        accept_multiple_files=True,
        help="Upload PDF, TXT, or DOCX documents to add to your knowledge base.",
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
    n_retrieved = st.slider(
        "Retrieved documents (per retriever)",
        3,
        30,
        settings.top_k_vector,
        help="Candidates extracted independently from ChromaDB vector search and BM25.",
    )
    n_reranked = st.slider(
        "Reranked documents (sent to LLM)",
        1,
        10,
        settings.top_k_rerank,
        help="Top high-relevance chunks selected by cross-encoder for the final prompt.",
    )

    st.header("📊 Index")
    try:
        stats = pipeline.stats()
        st.metric("Chunks indexed", stats["chunks"])
        for name, count in stats["documents"].items():
            st.caption(f"• {name} — {count} chunks")
    except RAGError as exc:
        st.error(str(exc))

    st.markdown(
        f"""
        <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:0.55rem 0.8rem; margin-top:0.8rem;">
            <div style="font-size:0.75rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.04em;">Active LLM Provider</div>
            <div style="font-size:0.85rem; font-weight:600; color:#58a6ff; margin-top:0.15rem;">
                🟢 {settings.llm_provider.capitalize()} · {settings.resolved_llm_model}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("🗑️ Danger zone")
    confirm = st.checkbox("Yes, delete everything")
    if st.button("Clear database", disabled=not confirm, use_container_width=True):
        pipeline.clear()
        st.session_state.pop("response", None)
        st.session_state.pop("last_question", None)
        st.rerun()

# --------------------------------------------------------------------- Main Content
col_title, col_status = st.columns([4, 1])
with col_title:
    st.title("🤖 Hybrid RAG Knowledge Assistant")
    st.caption("Vector search + BM25 → Reciprocal Rank Fusion → cross-encoder reranking → cited answer")

st.markdown(
    """
    <div class="pipeline-pills">
        <span class="pipeline-pill active">1 · 🧠 Semantic Embeddings</span>
        <span class="pipeline-pill active">2 · 🔤 BM25 Keywords</span>
        <span class="pipeline-pill active">3 · ⚡ RRF Fusion</span>
        <span class="pipeline-pill active">4 · 🎯 Cross-Encoder Reranker</span>
        <span class="pipeline-pill active">5 · 🤖 Grounded Generation</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Search Input Form
with st.form("ask"):
    question = st.text_input(
        "Ask a question about your documents...",
        placeholder="Type your question or identifier (e.g. 'What is the refund period?' or 'HP:0001945')...",
    )
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
            st.session_state["last_question"] = question
        except NoDocumentsError as exc:
            st.session_state.pop("response", None)
            st.warning(str(exc))
        except RAGError as exc:
            st.session_state.pop("response", None)
            st.error(f"{type(exc).__name__}: {exc}")

response = st.session_state.get("response")
last_question = st.session_state.get("last_question", "")

if response is not None:
    # User Question Bubble
    if last_question:
        st.markdown(
            f"""
            <div class="user-message-card">
                <div class="msg-header" style="color: #c9d1d9;">
                    <span class="user-avatar">👤</span>
                    <span>You</span>
                </div>
                <div style="font-size: 1.05rem; font-weight: 500; color: #f0f6fc; margin-left: 2.2rem;">
                    {last_question}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Assistant Response Card
    st.markdown(
        """
        <div class="msg-header" style="color: #10a37f; margin-bottom: 0.4rem;">
            <span class="assistant-avatar">🤖</span>
            <span>Knowledge Assistant</span>
            <span style="background: rgba(16,163,127,0.15); border: 1px solid rgba(16,163,127,0.3); border-radius: 9999px; padding: 0.15rem 0.5rem; font-size: 0.72rem; color: #10a37f; margin-left: auto;">Grounded & Cited</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Answer")
    st.markdown(response.answer)

    # Sources & Citations
    if response.sources:
        st.markdown("**Sources**")
        for number, source in enumerate(response.sources, start=1):
            mark = " ✅ cited" if number in response.cited else ""
            st.markdown(f"[{number}] {source.citation()}{mark}")

    # Retrieval Process Inspector
    retrieval = response.retrieval
    if retrieval is not None:
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.subheader("Retrieval process")
        tab_vec, tab_bm25, tab_rrf, tab_rerank = st.tabs(
            ["1 · Vector search", "2 · BM25", "3 · RRF fusion", "4 · Reranker"]
        )
        with tab_vec:
            st.caption("🧠 Meaning-based semantic retrieval. Score = cosine similarity (higher is closer).")
            st.dataframe(hits_table(retrieval.vector_hits, {"vector_score": "Similarity"}), hide_index=True)
        with tab_bm25:
            st.caption("🔤 Exact keyword matching. Ideal for codes, policy ids, and exact terminology.")
            if retrieval.bm25_hits:
                st.dataframe(hits_table(retrieval.bm25_hits, {"bm25_score": "BM25 score"}), hide_index=True)
            else:
                st.info("No chunk shares a keyword with the question.")
        with tab_rrf:
            st.caption("⚡ Rank-based merge: RRF = Σ 1 / (k + rank). Chunks found by both retrievers rise.")
            st.dataframe(
                hits_table(
                    retrieval.fused,
                    {"rrf_score": "RRF score", "vector_rank": "Vector rank", "bm25_rank": "BM25 rank"},
                ),
                hide_index=True,
            )
        with tab_rerank:
            st.caption("🎯 Cross-encoder joint evaluation (question + chunk). Higher score = higher true relevance.")
            st.dataframe(
                hits_table(retrieval.reranked, {"rerank_score": "Reranker score", "rrf_score": "RRF score"}),
                hide_index=True,
            )

    if response.sources:
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.subheader("Retrieved chunks (sent to the LLM)")
        for number, source in enumerate(response.sources, start=1):
            score = f"{source.rerank_score:.2f}" if source.rerank_score is not None else "n/a"
            with st.expander(f"[{number}] {source.citation()} · relevance {score}"):
                st.caption(f"chunk_id: {source.chunk_id}")
                st.write(source.text)

else:
    # Empty State Hero
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-icon">✨</div>
            <div class="hero-title">How can I assist your document research?</div>
            <div class="hero-subtitle">
                Ask questions about your uploaded documents. Our 4-stage pipeline combines dense semantic search and BM25 keywords, fuses ranks, and re-ranks evidence before generating an exact citation-backed answer.
            </div>
            <div style="display: flex; justify-content: center; gap: 0.75rem; flex-wrap: wrap;">
                <span class="source-pill">💡 "What is the refund period?"</span>
                <span class="source-pill">🔍 "HP:0001945"</span>
                <span class="source-pill">📋 "Customer support SLAs"</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
