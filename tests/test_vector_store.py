"""Tests for src.vector_store — FAISS vector store utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.vector_store import build_vector_store, load_vector_store, save_vector_store


def _make_doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "test"})


class TestBuildVectorStore:
    def test_calls_faiss_from_documents(self):
        docs = [_make_doc("chunk one"), _make_doc("chunk two")]
        mock_embeddings = MagicMock()
        mock_store = MagicMock()

        with patch("src.vector_store.FAISS") as MockFAISS:
            MockFAISS.from_documents.return_value = mock_store
            result = build_vector_store(docs, embeddings=mock_embeddings)

        MockFAISS.from_documents.assert_called_once_with(docs, mock_embeddings)
        assert result is mock_store

    def test_uses_default_embeddings_when_none_provided(self):
        docs = [_make_doc("text")]
        mock_embeddings = MagicMock()
        mock_store = MagicMock()

        with (
            patch("src.vector_store.get_embeddings_model", return_value=mock_embeddings),
            patch("src.vector_store.FAISS") as MockFAISS,
        ):
            MockFAISS.from_documents.return_value = mock_store
            build_vector_store(docs)

        MockFAISS.from_documents.assert_called_once_with(docs, mock_embeddings)


class TestSaveVectorStore:
    def test_delegates_to_store_save_local(self, tmp_path):
        mock_store = MagicMock()
        save_vector_store(mock_store, tmp_path / "index")
        mock_store.save_local.assert_called_once_with(str(tmp_path / "index"))


class TestLoadVectorStore:
    def test_calls_faiss_load_local(self, tmp_path):
        mock_embeddings = MagicMock()
        mock_store = MagicMock()

        with patch("src.vector_store.FAISS") as MockFAISS:
            MockFAISS.load_local.return_value = mock_store
            result = load_vector_store(tmp_path, embeddings=mock_embeddings)

        MockFAISS.load_local.assert_called_once_with(
            str(tmp_path), mock_embeddings, allow_dangerous_deserialization=True
        )
        assert result is mock_store

    def test_uses_default_embeddings_when_none_provided(self, tmp_path):
        mock_embeddings = MagicMock()
        mock_store = MagicMock()

        with (
            patch("src.vector_store.get_embeddings_model", return_value=mock_embeddings),
            patch("src.vector_store.FAISS") as MockFAISS,
        ):
            MockFAISS.load_local.return_value = mock_store
            load_vector_store(tmp_path)

        MockFAISS.load_local.assert_called_once_with(
            str(tmp_path), mock_embeddings, allow_dangerous_deserialization=True
        )
