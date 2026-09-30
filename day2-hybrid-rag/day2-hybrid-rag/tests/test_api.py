import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_pipeline
from app.exceptions import LLMAuthError, LLMError
from main import app


@pytest.fixture
def client(pipeline):
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_stats_on_empty_index(client):
    body = client.get("/stats").json()
    assert body["chunks"] == 0 and body["documents"] == {}


def test_query_before_ingest_returns_409(client):
    response = client.post("/query", json={"question": "What is the refund policy?"})
    assert response.status_code == 409
    assert response.json()["error"] == "NoDocumentsError"


def test_ingest_upload_then_query(client, sample_files):
    files = [("files", (f.name, f.read_bytes(), "text/plain")) for f in sample_files]
    response = client.post("/ingest", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["succeeded"] == 3 and body["failed"] == 0 and body["total_chunks"] > 3

    answer = client.post("/query", json={"question": "What is the refund period?"}).json()
    assert answer["answer"].startswith("The refund period is 30 days")
    first = answer["sources"][0]
    assert first["source"].endswith(".txt") and isinstance(first["chunk"], int)
    assert first["chunk_id"] == f"{first['source']}::chunk-{first['chunk']}"
    assert first["cited"] is True and first["text"] and first["rerank_score"] is not None
    assert [s["cited"] for s in answer["sources"][1:]] == [False] * (len(answer["sources"]) - 1)

    assert client.get("/stats").json()["chunks"] == body["total_chunks"]


def test_ingest_rejects_unsupported_file_with_422(client):
    response = client.post("/ingest", files=[("files", ("evil.exe", b"MZ...", "application/octet-stream"))])
    assert response.status_code == 422
    body = response.json()
    assert body["failed"] == 1 and "Unsupported" in body["files"][0]["error"]


def test_ingest_partial_success_returns_200(client, sample_files):
    files = [
        ("files", (sample_files[0].name, sample_files[0].read_bytes(), "text/plain")),
        ("files", ("empty.txt", b"  ", "text/plain")),
    ]
    response = client.post("/ingest", files=files)
    assert response.status_code == 200
    assert response.json()["succeeded"] == 1 and response.json()["failed"] == 1


def test_ingest_ignores_path_traversal_in_filename(client, pipeline):
    from pathlib import Path

    client.post("/ingest", files=[("files", ("../../escape.txt", b"hello world text", "text/plain"))])
    assert (Path(pipeline.settings.documents_dir) / "escape.txt").exists()
    assert not (Path(pipeline.settings.documents_dir).parent.parent / "escape.txt").exists()


def test_ingest_without_files_indexes_documents_dir(client, pipeline, sample_files):
    from pathlib import Path

    docs = Path(pipeline.settings.documents_dir)
    docs.mkdir(parents=True)
    (docs / "one.txt").write_text("The office opens at 8 am every weekday morning.")
    response = client.post("/ingest")
    assert response.status_code == 200 and response.json()["succeeded"] == 1


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 2001}])
def test_query_validation_errors(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_llm_errors_map_to_http_status(client, pipeline, sample_files, llm, monkeypatch):
    pipeline.ingest_documents(sample_files)

    def auth_fail(*args, **kwargs):
        raise LLMAuthError("Gemini rejected the API key (check GOOGLE_API_KEY).")

    monkeypatch.setattr(llm, "generate", auth_fail)
    response = client.post("/query", json={"question": "refund?"})
    assert response.status_code == 503 and "API key" in response.json()["detail"]

    def provider_down(*args, **kwargs):
        raise LLMError("Could not reach Gemini")

    monkeypatch.setattr(llm, "generate", provider_down)
    assert client.post("/query", json={"question": "refund?"}).status_code == 502
