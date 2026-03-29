"""Tests for src.retriever — similarity search utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.retriever import retrieve


def _make_doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "test"})


class TestRetrieve:
    def test_calls_similarity_search_with_correct_args(self):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [_make_doc("result")]

        retrieve(mock_store, "What is ACME?", k=3)

        mock_store.similarity_search.assert_called_once_with("What is ACME?", k=3)

    def test_returns_documents_from_store(self):
        expected = [_make_doc("chunk A"), _make_doc("chunk B")]
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = expected

        result = retrieve(mock_store, "query")
        assert result == expected

    def test_default_k_is_four(self):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []

        retrieve(mock_store, "question")
        mock_store.similarity_search.assert_called_once_with("question", k=4)

    def test_returns_empty_list_when_store_is_empty(self):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []

        result = retrieve(mock_store, "anything")
        assert result == []
