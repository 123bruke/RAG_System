FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# 1) CPU-only PyTorch first (the default PyPI wheel bundles multi-GB CUDA libraries
#    we do not need). requirements.txt then sees torch as already satisfied.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# 2) Dependencies (separate layer so code changes do not re-install everything)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Bake the two models into the image so containers start without downloading.
#    (Needs internet at BUILD time. Keep the names in sync with EMBEDDING_MODEL / RERANKER_MODEL.)
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# 4) Application code (.dockerignore keeps .env and chroma_db out of the image)
COPY . .
RUN useradd --create-home appuser \
    && mkdir -p /app/chroma_db /app/data/documents \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Secrets are NOT baked in: pass them at runtime (docker run --env-file .env ...)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
