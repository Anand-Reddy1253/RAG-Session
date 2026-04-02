"""Unit tests for src.embedder — VectorStoreManager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.embedder import VectorStoreManager


# ===========================================================================
# Tests
# ===========================================================================


class TestBuild:
    def test_build_calls_faiss_from_documents(self, mocker, sample_chunks):
        mock_embeddings = MagicMock()
        mocker.patch("src.embedder.OpenAIEmbeddings", return_value=mock_embeddings)
        mock_faiss_cls = mocker.patch("src.embedder.FAISS")
        mock_store = MagicMock()
        mock_faiss_cls.from_documents.return_value = mock_store

        manager = VectorStoreManager(openai_api_key="sk-test")
        manager.build(sample_chunks)

        mock_faiss_cls.from_documents.assert_called_once_with(
            sample_chunks, mock_embeddings
        )

    def test_build_raises_on_empty_chunks(self, mocker):
        mocker.patch("src.embedder.OpenAIEmbeddings")
        mocker.patch("src.embedder.FAISS")

        manager = VectorStoreManager(openai_api_key="sk-test")
        with pytest.raises(ValueError, match="empty"):
            manager.build([])

    def test_build_returns_faiss_instance(self, mocker, sample_chunks):
        mocker.patch("src.embedder.OpenAIEmbeddings")
        mock_faiss_cls = mocker.patch("src.embedder.FAISS")
        mock_store = MagicMock()
        mock_faiss_cls.from_documents.return_value = mock_store

        manager = VectorStoreManager(openai_api_key="sk-test")
        result = manager.build(sample_chunks)

        assert result is mock_store


class TestSave:
    def test_save_calls_save_local(self, mocker):
        mocker.patch("src.embedder.OpenAIEmbeddings")
        mocker.patch("src.embedder.FAISS")

        mock_store = MagicMock()
        manager = VectorStoreManager(openai_api_key="sk-test")
        manager.save(mock_store)

        mock_store.save_local.assert_called_once()

    def test_save_uses_correct_path(self, mocker, tmp_path):
        mocker.patch("src.embedder.OpenAIEmbeddings")
        mocker.patch("src.embedder.FAISS")

        mock_store = MagicMock()
        manager = VectorStoreManager(
            openai_api_key="sk-test",
            vector_store_path=str(tmp_path / "my_store"),
        )
        manager.save(mock_store)

        mock_store.save_local.assert_called_once_with(str(tmp_path / "my_store"))


class TestLoad:
    def test_load_calls_faiss_load_local(self, mocker):
        mocker.patch("src.embedder.OpenAIEmbeddings")
        mock_faiss_cls = mocker.patch("src.embedder.FAISS")
        mock_faiss_cls.load_local.return_value = MagicMock()

        manager = VectorStoreManager(openai_api_key="sk-test")
        manager.load()

        mock_faiss_cls.load_local.assert_called_once()

    def test_load_passes_allow_dangerous_deserialization(self, mocker):
        mocker.patch("src.embedder.OpenAIEmbeddings")
        mock_faiss_cls = mocker.patch("src.embedder.FAISS")
        mock_faiss_cls.load_local.return_value = MagicMock()

        manager = VectorStoreManager(
            openai_api_key="sk-test", vector_store_path="./vector_store"
        )
        manager.load()

        call_kwargs = mock_faiss_cls.load_local.call_args.kwargs
        assert call_kwargs.get("allow_dangerous_deserialization") is True

    def test_load_raises_if_path_missing(self, mocker):
        mocker.patch("src.embedder.OpenAIEmbeddings")
        mock_faiss_cls = mocker.patch("src.embedder.FAISS")
        mock_faiss_cls.load_local.side_effect = FileNotFoundError("no index")

        manager = VectorStoreManager(openai_api_key="sk-test")
        with pytest.raises(FileNotFoundError):
            manager.load()


class TestBuildAndSave:
    def test_build_and_save_calls_both(self, mocker, sample_chunks):
        mocker.patch("src.embedder.OpenAIEmbeddings")
        mocker.patch("src.embedder.FAISS")

        manager = VectorStoreManager(openai_api_key="sk-test")
        build_spy = mocker.spy(manager, "build")
        save_spy = mocker.spy(manager, "save")

        manager.build_and_save(sample_chunks)

        build_spy.assert_called_once_with(sample_chunks)
        save_spy.assert_called_once()


class TestSimilaritySearch:
    def test_similarity_search_delegates_to_faiss(self, mocker, sample_chunks):
        mocker.patch("src.embedder.OpenAIEmbeddings")
        mocker.patch("src.embedder.FAISS")

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = sample_chunks[:2]

        manager = VectorStoreManager(openai_api_key="sk-test")
        manager.similarity_search("query text", mock_store, top_k=3)

        mock_store.similarity_search.assert_called_once_with("query text", k=3)

    def test_similarity_search_returns_documents(self, mocker, sample_chunks):
        mocker.patch("src.embedder.OpenAIEmbeddings")
        mocker.patch("src.embedder.FAISS")

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = sample_chunks[:2]

        manager = VectorStoreManager(openai_api_key="sk-test")
        result = manager.similarity_search("query", mock_store, top_k=2)

        assert result == sample_chunks[:2]


class TestEmbeddingsProperty:
    def test_embeddings_model_name_configured(self, mocker):
        mock_embeddings_cls = mocker.patch("src.embedder.OpenAIEmbeddings")
        mocker.patch("src.embedder.FAISS")

        VectorStoreManager(
            openai_api_key="sk-test",
            embedding_model="text-embedding-ada-002",
        )

        call_kwargs = mock_embeddings_cls.call_args.kwargs
        assert call_kwargs.get("model") == "text-embedding-ada-002"

    def test_embeddings_property_returns_instance(self, mocker):
        mock_instance = MagicMock()
        mocker.patch("src.embedder.OpenAIEmbeddings", return_value=mock_instance)
        mocker.patch("src.embedder.FAISS")

        manager = VectorStoreManager(openai_api_key="sk-test")
        assert manager.embeddings is mock_instance
