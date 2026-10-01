import pytest

from app.ingestion.chunker import TextChunker
from app.ingestion.loader import DocumentPage, LoadedDocument

WORDS = " ".join(f"word{i}" for i in range(300))


def test_chunks_never_exceed_chunk_size():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split_text(WORDS)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_consecutive_chunks_overlap():
    chunker = TextChunker(chunk_size=100, chunk_overlap=30)
    chunks = chunker.split_text(WORDS)
    for previous, current in zip(chunks, chunks[1:], strict=False):  # neighbouring pairs
        assert current[:15] in previous  # the start of a chunk repeats the end of the last


def test_no_words_are_lost_or_cut():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split_text(WORDS)
    found = {w for c in chunks for w in c.split()}
    assert found == set(WORDS.split())


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t  \n"])
def test_empty_input_returns_no_chunks(text):
    assert TextChunker(100, 10).split_text(text) == []


def test_short_text_is_one_chunk():
    assert TextChunker(500, 100).split_text("Hello world.") == ["Hello world."]


def test_large_overlap_still_terminates():
    chunks = TextChunker(chunk_size=50, chunk_overlap=49).split_text(WORDS)
    assert chunks and all(len(c) <= 50 for c in chunks)


def test_text_without_spaces_is_hard_split():
    chunks = TextChunker(chunk_size=40, chunk_overlap=10).split_text("x" * 200)
    assert len(chunks) > 1 and all(len(c) <= 40 for c in chunks)


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_configuration_raises(size, overlap):
    with pytest.raises(ValueError):
        TextChunker(size, overlap)


def test_chunk_document_metadata_and_ids():
    doc = LoadedDocument(
        source="report.pdf",
        doc_type="pdf",
        pages=[DocumentPage(WORDS[:400], page=1), DocumentPage(WORDS[400:900], page=2)],
    )
    chunks = TextChunker(100, 20).chunk_document(doc)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)  # ids are unique
    assert {c.metadata["page"] for c in chunks} == {1, 2}
    assert all(c.metadata["source"] == "report.pdf" and c.metadata["doc_type"] == "pdf" for c in chunks)
    assert all(c.metadata["chunk_id"] == c.chunk_id for c in chunks)


def test_page_is_omitted_when_format_has_no_pages():
    doc = LoadedDocument("a.txt", "txt", [DocumentPage("hello world", page=None)])
    (chunk,) = TextChunker(100, 10).chunk_document(doc)
    assert "page" not in chunk.metadata
