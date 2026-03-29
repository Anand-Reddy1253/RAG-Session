"""Tests for src.splitter — text chunking utilities."""

from __future__ import annotations

from langchain_core.documents import Document

from src.splitter import split_documents


def _make_doc(text: str, source: str = "test.txt") -> Document:
    return Document(page_content=text, metadata={"source": source})


class TestSplitDocuments:
    def test_returns_list_of_documents(self):
        docs = [_make_doc("Hello world. " * 10)]
        chunks = split_documents(docs, chunk_size=50, chunk_overlap=0)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Document) for c in chunks)

    def test_chunks_are_no_longer_than_chunk_size(self):
        long_text = "word " * 200
        docs = [_make_doc(long_text)]
        chunk_size = 100
        chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=0)
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size

    def test_short_document_is_not_split(self):
        short_text = "Short sentence."
        docs = [_make_doc(short_text)]
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=0)
        assert len(chunks) == 1
        assert chunks[0].page_content == short_text

    def test_metadata_is_preserved(self):
        docs = [_make_doc("Some content.", source="myfile.pdf")]
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=0)
        assert chunks[0].metadata["source"] == "myfile.pdf"

    def test_overlap_creates_shared_content(self):
        # A predictable text: "AB" repeated so chunks overlap by one "word".
        text = "Alpha Beta " * 20
        docs = [_make_doc(text)]
        chunks = split_documents(docs, chunk_size=30, chunk_overlap=10)
        # With overlap, adjacent chunks should share some characters.
        if len(chunks) >= 2:
            overlap_text = chunks[0].page_content[-10:]
            assert overlap_text in chunks[1].page_content

    def test_empty_document_list_returns_empty(self):
        assert split_documents([]) == []

    def test_multiple_documents_are_all_split(self):
        docs = [_make_doc("word " * 100), _make_doc("text " * 100)]
        chunks = split_documents(docs, chunk_size=50, chunk_overlap=0)
        assert len(chunks) > 2
