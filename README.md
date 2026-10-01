# Production-Ready Hybrid RAG Knowledge Assistant

Ask questions about your own PDF / TXT / DOCX files and get **answers with citations**.
Retrieval combines **semantic vector search** and **BM25 keyword search**, merges them with
**Reciprocal Rank Fusion**, and re-scores the shortlist with a **cross-encoder** before an LLM writes the answer.

## Overview

**Problem.** An LLM alone doesn't know your documents and will happily invent answers. Plain vector search
misses exact identifiers ("HP:0001945", invoice numbers, error codes). Plain keyword search misses
paraphrases ("money back" vs "refund").

**Solution.** Use both retrievers, fuse their rankings, rerank with a stronger model, and force the LLM to answer
*only* from the retrieved text, cite it, and say *"I don't know based on the provided documents."* when the
answer isn't there.

<<<<<<< HEAD
## System Architecture & Workflow

### 📊 End-to-End System Flow

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    HYBRID RAG SYSTEM ARCHITECTURE                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

                         ╔════════════════════════════════════╗
                         ║   📄 USER DOCUMENTS                ║
                         ║  (PDF / TXT / DOCX)                ║
                         ╚════════════════════════════════════╝
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │   INGESTION PIPELINE (Once)     │
                    └─────────────────────────────────┘
                    │
        ┌───────────┴───────────┬────────────────┐
        ▼                       ▼                ▼
    ┏━━━━━━━━┓            ┏━━━━━━━━┓       ┏━━━━━━━━┓
    ┃ Loader ┃            ┃ Chunker┃       ┃Embedder┃
    ┃        ┃──────────▶ ┃        ┃──────▶┃        ┃
    ┗━━━━━━━━┛            ┗━━━━━━━━┘       ┗━━━━━━━━┛
        Text                Chunks           Vectors
                                               │
                        ┌──────────────────────┼──────────────────────┐
                        ▼                      ▼                      ▼
                    ┏━━━━━━━━━┓           ┏━━━━━━━━━┓            ┏━━━━━━━┓
                    ┃ChromaDB ┃           ┃BM25 Index          ┃ Config┃
                    ┃(Vector) ┃           ┃(Keyword)           ┃ Cache ┃
                    ┗━━━━━━━━━┛           ┗━━━━━━━━━┘           ┗━━━━━━━┛
                         📦                   📦                    ⚙️


                    ╔════════════════════════════════════╗
                    ║   ❓ USER QUESTION                 ║
                    ╚════════════════════════════════════╝
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │   RETRIEVAL PIPELINE (Per Query)│
                    └─────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
            ┏━━━━━━━━━━━━━━━┓                  ┏━━━━━━━━━━━━━━━┓
            ┃Vector Search  ┃                  ┃ BM25 Search   ┃
            ┃ (Semantic)    ┃                  ┃ (Keyword)     ┃
            ┃ Top 10        ┃                  ┃ Top 10        ┃
            ┗━━━━━━━━━━━━━━━┛                  ┗━━━━━━━━━━━━━━━┛
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
                            ┏━━━━━━━━━━━━━━━┓
                            ┃  RRF Fusion   ┃
                            ┃  Merge Ranks  ┃
                            ┃  Top 10 → 5   ┃
                            ┗━━━━━━━━━━━━━━━┛
                                      │
                                      ▼
                            ┏━━━━━━━━━━━━━━━┓
                            ┃ Cross-Encoder ┃
                            ┃  Reranker     ┃
                            ┃ Scoring: 0-10 ┃
                            ┗━━━━━━━━━━━━━━━┛
                                      │
                                      ▼
                    ╔════════════════════════════════════╗
                    ║  Top 5 Relevant Chunks Ready      ║
                    ╚════════════════════════════════════╝
                                      │
                                      ▼
                            ┏━━━━━━━━━━━━━━━┓
                            ┃  LLM (Gemini  ┃
                            ┃  or OpenAI)   ┃
                            ┃  Generate     ┃
                            ┃  Answer       ┃
                            ┗━━━━━━━━━━━━━━━┛
                                      │
                                      ▼
                    ╔════════════════════════════════════╗
                    ║   🎯 GROUNDED ANSWER + CITATIONS  ║
                    ║   📌 Source References Included    ║
                    ╚════════════════════════════════════╝
```

### 🔄 Detailed Processing Steps

#### **INGESTION PHASE** (Once per file)
```
File Upload
    │
    ├─▶ Load & Extract
    │   └─ PDF/TXT/DOCX → Plain text
    │
    ├─▶ Chunk & Segment
    │   ├─ 500 char chunks
    │   ├─ 100 char overlap
    │   └─ Preserve page metadata
    │
    ├─▶ Generate Embeddings
    │   ├─ Model: all-MiniLM-L6-v2
    │   ├─ Normalize vectors
    │   └─ 384-dim embeddings
    │
    └─▶ Store & Index
        ├─ ChromaDB (persistent)
        └─ BM25 in-memory index
```

#### **RETRIEVAL PHASE** (Per question)
```
Question Input
    │
    ├─▶ Parallel Search 1: Vector Similarity
    │   ├─ Cosine distance
    │   ├─ Top 10 results
    │   └─ Semantic matching
    │
    ├─▶ Parallel Search 2: BM25 Keyword
    │   ├─ Term frequency
    │   ├─ Top 10 results
    │   └─ Exact match scoring
    │
    ├─▶ Reciprocal Rank Fusion (RRF)
    │   ├─ Merge rankings
    │   ├─ Formula: 1/(k+rank)
    │   └─ Top 10 → Top 5
    │
    ├─▶ Cross-Encoder Reranking
    │   ├─ Model: ms-marco-MiniLM-L-6-v2
    │   ├─ Score each chunk
    │   └─ Re-sort by relevance
    │
    └─▶ LLM Answer Generation
        ├─ Input: Question + Top 5 chunks
        ├─ Force grounding in text
        ├─ Generate citations [1][2]
        └─ Output: Answer + Sources
```

## 🏗️ Technology Stack & Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                     TECHNOLOGY STACK                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🐍 LANGUAGE & RUNTIME                                          │
│  └─ Python 3.11+                                               │
│                                                                 │
│  🔗 VECTOR & EMBEDDING                                          │
│  ├─ ChromaDB (Vector Database)                                 │
│  ├─ sentence-transformers                                      │
│  │  ├─ all-MiniLM-L6-v2 (384-dim embeddings)                   │
│  │  └─ ms-marco-MiniLM-L-6-v2 (reranker)                       │
│  └─ NumPy (vector operations)                                  │
│                                                                 │
│  📚 RETRIEVAL & SEARCH                                          │
│  ├─ rank-bm25 (Keyword search)                                 │
│  ├─ ChromaDB API (Vector similarity)                           │
│  └─ Custom RRF Fusion (Ranking merge)                          │
│                                                                 │
│  🤖 LLM INTEGRATIONS                                            │
│  ├─ Google Gemini (gemini-2.5-flash)                           │
│  ├─ OpenAI (gpt-4o-mini)                                       │
│  └─ Provider-agnostic interface                                │
│                                                                 │
│  🎨 INTERFACES                                                  │
│  ├─ FastAPI (REST API)                                         │
│  ├─ Streamlit (Web UI)                                         │
│  └─ OpenAPI/Swagger docs (/docs)                               │
│                                                                 │
│  📋 CONFIGURATION                                               │
│  ├─ Pydantic Settings (Type-safe config)                       │
│  └─ .env file management                                       │
│                                                                 │
│  📦 DOCUMENT PROCESSING                                         │
│  ├─ pypdf (PDF extraction)                                     │
│  ├─ python-docx (DOCX parsing)                                 │
│  └─ Text chunking (Overlapping windows)                        │
│                                                                 │
│  ✅ TESTING & QUALITY                                           │
│  ├─ pytest (85 unit tests)                                     │
│  ├─ Deterministic test mocks                                   │
│  └─ No API keys needed for tests                               │
│                                                                 │
│  🐳 DEPLOYMENT                                                  │
│  ├─ Docker (Containerization)                                  │
│  ├─ docker-compose (Multi-service)                             │
│  └─ Volume management for persistence                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Output Example

### 📤 Query Flow Visualization

```
Input Query: "What is the refund period?"

    STEP 1: VECTOR SEARCH              STEP 2: BM25 SEARCH
    ┌──────────────────┐              ┌──────────────────┐
    │ Embed question   │              │ Tokenize query   │
    │ Find similar     │              │ Match keywords   │
    │ vectors          │              │ Calculate IDF    │
    └────────┬─────────┘              └────────┬─────────┘
             │                                 │
    Result:  │ "Refund",                   │ "refund_policy.txt",
    [Score]  │ "Return",                   │ "customer_policy.docx",
             │ "Money back"                │ "policy_updated.txt"
             │                                 │
             └───────────────┬─────────────────┘
                             │
                    STEP 3: RRF FUSION
                    ┌──────────────────┐
                    │ Reciprocal Ranks │
                    │ Merge results    │
                    │ Remove duplicates│
                    └────────┬─────────┘
                             │
                    Top 10 merged results
                             │
                    STEP 4: CROSS-ENCODER RERANKING
                    ┌──────────────────────────┐
                    │ Score each chunk pair:   │
                    │ (question, chunk_text)   │
                    │ Logit scores: -10 to +10 │
                    └────────┬─────────────────┘
                             │
                    ┌────────▼─────────────────────────────┐
                    │ Top 5 Ranked Chunks:                 │
                    │ [1] refund_policy.txt (score: 8.41)  │
                    │ [2] customer_policy.docx (score: 7.9)│
                    │ [3] policy_updated.txt (score: 6.2)  │
                    │ [4] faq.txt (score: 5.1)             │
                    │ [5] terms.docx (score: 4.8)          │
                    └────────┬─────────────────────────────┘
                             │
                    STEP 5: LLM GENERATION
                    ┌──────────────────────────────────┐
                    │ Context Window:                  │
                    │ Question +                       │
                    │ Top 5 chunks +                   │
                    │ Grounding prompt                 │
                    └────────┬─────────────────────────┘
                             │
    ╔════════════════════════▼══════════════════════════════╗
    ║                     OUTPUT                            ║
    ╠════════════════════════════════════════════════════════╣
    ║                                                        ║
    ║ ✅ ANSWER:                                            ║
    ║ "Customers can request a full refund within 30 days  ║
    ║  of purchase. [1]"                                    ║
    ║                                                        ║
    ║ 📌 SOURCES:                                           ║
    ║ [1] refund_policy.txt (page: 1, chunk: 0)            ║
    ║     Score: 8.41 (rerank) | RRF: 0.0325              ║
    ║     Text: "ACME Store - Refund Policy...             ║
    ║           Customers can request a full refund..."   ║
    ║                                                        ║
    ╚════════════════════════════════════════════════════════╝
```

=======
>>>>>>> c367202 (Initial commit – add Hybrid‑RAG project files)
## Architecture

```
                         INGESTION (once per file)
 Documents ─► Loader ─► Chunker ─► Embeddings ─► ChromaDB (persistent, on disk)
 PDF/TXT/DOCX  text     500 chars   MiniLM-L6      │
                        +100 overlap               └─► BM25 index (in memory, rebuilt from Chroma)

                         QUERY (every question)
 Question ─┬─► Vector search (Chroma, top 10) ─┐
           └─► BM25 keyword search  (top 10) ──┴─► RRF fusion ─► Cross-encoder ─► Top 5 ─► LLM ─► Answer
<<<<<<< HEAD
                                                    1/(k+rank)     reranker                        + Sources
=======
                                                   1/(k+rank)     reranker                        + Sources
>>>>>>> c367202 (Initial commit – add Hybrid‑RAG project files)
```

| Stage | Module | Job |
|---|---|---|
| Load | `app/ingestion/loader.py` | PDF/TXT/DOCX → text + page metadata |
| Chunk | `app/ingestion/chunker.py` | Overlapping chunks with metadata & unique ids |
| Embed | `app/retrieval/embeddings.py` | `all-MiniLM-L6-v2`, normalised vectors |
| Store | `app/retrieval/vector_store.py` | Persistent ChromaDB (cosine) |
| Keyword | `app/retrieval/keyword_search.py` | BM25 (`rank-bm25`) |
| Fuse | `app/retrieval/rrf.py`, `hybrid_search.py` | Reciprocal Rank Fusion |
| Rerank | `app/retrieval/reranker.py` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generate | `app/generation/` | Provider-agnostic LLM + grounded prompt |
| Orchestrate | `app/pipeline.py` | `RAGPipeline` |
| Serve | `main.py`, `app/api/routes.py`, `streamlit_app.py` | FastAPI + Streamlit |
| Measure | `app/evaluation/evaluator.py` | Recall@K, Precision@K, MRR |

## Features

- PDF / TXT / DOCX ingestion with per-chunk metadata (source, type, page, chunk id, chunk index)
- Configurable chunking; idempotent re-ingestion (re-uploading a file replaces its chunks)
- Persistent ChromaDB — the index survives restarts
- Hybrid retrieval (vector + BM25) fused with RRF, then cross-encoder reranking
- Grounded answers with `[n]` citations and inspectable source chunks
- Swappable LLM provider: **Gemini** or **OpenAI** (one line in `.env`)
- Streamlit UI that shows every retrieval stage; FastAPI with `/ingest`, `/query`, `/health`, `/stats`
- Typed config (Pydantic Settings), structured logging, explicit error classes with helpful messages
- Evaluation harness comparing vector-only vs BM25-only vs hybrid vs hybrid+rerank
- 85 pytest tests that run without model downloads or API keys; Dockerfile + docker-compose

## Technologies

Python 3.11+ · FastAPI · Pydantic / pydantic-settings · Streamlit · ChromaDB · sentence-transformers
(`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`) · rank-bm25 · pypdf · python-docx · google-genai / openai · pytest · Docker

## Installation

```bash
git clone <your-repo-url> day2-hybrid-rag && cd day2-hybrid-rag

python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# Optional but recommended on machines without a GPU: CPU-only PyTorch (~200 MB instead of ~2 GB+)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
cp .env.example .env                 # Windows: copy .env.example .env
```

The **first** run downloads two small models (~90 MB + ~90 MB) from Hugging Face, so it needs internet once.

## Environment variables

Copy `.env.example` to `.env` and put your key in it. **`.env` is git-ignored — never commit it.**

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `openai` |
| `GOOGLE_API_KEY` | – | Gemini key (Google AI Studio, starts with `AIza…`) |
| `OPENAI_API_KEY` | – | OpenAI key (starts with `sk-…`) |
| `LLM_MODEL` | provider default | Blank → `gemini-2.5-flash` / `gpt-4o-mini` |
| `LLM_TEMPERATURE` | `0.0` | 0 = most deterministic |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `100` | Characters per chunk / shared between neighbours |
| `TOP_K_VECTOR` / `TOP_K_BM25` | `10` / `10` | Candidates from each retriever |
| `TOP_K_HYBRID` | `10` | Candidates kept after RRF (fed to the reranker) |
| `TOP_K_RERANK` | `5` | Chunks sent to the LLM |
| `RRF_K` | `60` | RRF smoothing constant |
| `CHROMA_PATH` | `./chroma_db` | Where the vector DB lives |
| `DOCUMENTS_DIR` | `./data/documents` | Folder for uploads / sample docs |
| `EMBEDDING_MODEL`, `RERANKER_MODEL` | MiniLM models | Hugging Face model ids |

Changing `EMBEDDING_MODEL` changes the vector size, so clear the database and re-ingest afterwards.

## Running

```bash
# Streamlit UI  ->  http://localhost:8501
streamlit run streamlit_app.py

# FastAPI       ->  http://localhost:8000  (interactive docs at /docs)
uvicorn main:app --reload
```

In the UI: click **Ingest sample documents** (or upload your own), then ask e.g. *"What is the refund period?"*
or just *"HP:0001945"* — the latter shows BM25 earning its keep.

### Docker

```bash
docker build -t hybrid-rag .
docker run --env-file .env -p 8000:8000 hybrid-rag          # API only

docker compose up --build                                    # API :8000 + UI :8501
```

`docker compose` gives the API and the UI **separate** ChromaDB volumes (see the note at the top of
`docker-compose.yml`). Secrets are passed at runtime; `.dockerignore` keeps `.env` out of the image.

## API examples

```bash
# 1) Index documents (upload)…
curl -X POST http://localhost:8000/ingest \
  -F "files=@refund_policy.pdf" -F "files=@customer_policy.docx"
# …or index everything in DOCUMENTS_DIR (the bundled samples):
curl -X POST http://localhost:8000/ingest
```
```json
{
  "succeeded": 3, "failed": 0, "total_chunks": 17,
  "files": [{"filename": "refund_policy.txt", "status": "ok", "chunks": 6, "error": null}, "..."]
}
```
A bad file is reported per file (`"status": "error"` + reason) without blocking the others; if *every* file fails the status is `422`.

```bash
# 2) Ask
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "What is the refund period?"}'
```
```json
{
  "answer": "Customers can request a full refund within 30 days of purchase. [1]",
  "answered": true,
  "sources": [
    {
      "source": "refund_policy.txt", "chunk": 0, "chunk_id": "refund_policy.txt::chunk-0",
      "page": null, "rerank_score": 8.41, "rrf_score": 0.0325, "cited": true,
      "text": "ACME Store - Refund Policy\n\nCustomers can request a full refund within 30 days…"
    }
  ]
}
```
(Example output — scores depend on the model and your documents.)

```bash
curl http://localhost:8000/health   # {"status":"ok"}
curl http://localhost:8000/stats    # chunk count, documents, active models & settings
```

| Status | Meaning |
|---|---|
| 409 | Nothing ingested yet |
| 415 / 422 | Unsupported, empty or corrupted file; invalid request body |
| 502 | LLM provider failure (rate limit, network, blocked answer) |
| 503 | LLM API key missing or rejected |

## Tests

```bash
pytest -q
```
Unit tests use deterministic fakes for the embedder, reranker and LLM, so they need **no downloads and no API key**.
They verify the *wiring* (chunking, BM25, RRF maths, hybrid merge, pipeline, API, Streamlit flow, error handling);
they cannot judge the *quality* of the real models — that is what the evaluation below is for.

## Evaluation

```bash
python -m app.evaluation.evaluator --ingest --k 5              # retrieval metrics only (no LLM cost)
python -m app.evaluation.evaluator --k 5 --generate            # + answer correctness (calls the LLM)
python -m app.evaluation.evaluator --output report.json
```

Dataset: `data/eval/eval_dataset.json` — `question`, `expected_answer`, `expected_sources`. A retrieved chunk counts as
**relevant** if it contains the expected answer text.

| Metric | Meaning |
|---|---|
| **Recall@K** | Of all relevant chunks that exist, what fraction is in the top K? *Did we find what we needed?* |
| **Precision@K** | Of the K chunks returned, what fraction is relevant? *How much noise?* |
| **MRR** | Average of 1/rank of the first relevant chunk. *How near the top is the first good result?* |
| **Source coverage** | Fraction of `expected_sources` among the chunks given to the LLM |
| **Answer correctness** | Generated answer contains the expected answer (simple string match) |

The report compares four stages — `vector`, `bm25`, `hybrid_rrf`, `hybrid_rrf+rerank` — so you can *measure* what each
component adds instead of assuming it. Add your own questions to the JSON file.

## Design choices & known limitations

- **Character-based chunking.** Simple and predictable; not token- or structure-aware. Chunks never span PDF pages.
- **No OCR.** Scanned PDFs are rejected as "no extractable text". DOCX tables are appended after paragraphs.
- **BM25 is in memory.** `rank-bm25` can't persist, so it's rebuilt from ChromaDB at start-up and after each ingest — fine for
  thousands of chunks, not millions. On very small corpora BM25's IDF is weak (a term in half the chunks scores ~0).
- **Local ChromaDB is single-process.** Don't point two servers at the same folder (hence the separate compose volumes).
- **Reranker scores are logits**, not probabilities; only compare them within one query. There is no relevance
  threshold, so out-of-scope questions rely on the LLM's "I don't know" instruction.
- **No authentication, rate limiting or upload size limit** on the API — do not expose it publicly as is.
- **Evaluation is a proxy** (string containment). Real systems add human-labelled chunks and LLM-as-judge scoring.
<<<<<<< HEAD
- **English-centric models** (`all-MiniLM-L6-v2` is trained mostly on English).
=======
- English-centric models (`all-MiniLM-L6-v2` is trained mostly on English).
>>>>>>> c367202 (Initial commit – add Hybrid‑RAG project files)

## Future improvements

Qdrant (or Chroma in server mode) for a shared, scalable index · PostgreSQL + pgvector · structure/semantic-aware chunking ·
multilingual embeddings · query rewriting / multi-query · contextual compression of retrieved chunks · streaming answers ·
observability (OpenTelemetry, Langfuse, per-stage latency) · authentication & rate limiting · async ingestion queue ·
CI/CD and cloud deployment · OCR for scanned PDFs · relevance thresholds & abstention.
