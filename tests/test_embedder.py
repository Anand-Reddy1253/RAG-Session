"""Tests for src.embedder — embedding utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.embedder import embed_texts, get_embeddings_model


class TestGetEmbeddingsModel:
    def test_returns_openai_embeddings_instance(self):
        from langchain_openai import OpenAIEmbeddings

        with patch("src.embedder.OpenAIEmbeddings") as MockEmb:
            instance = MagicMock(spec=OpenAIEmbeddings)
            MockEmb.return_value = instance
            result = get_embeddings_model()
            MockEmb.assert_called_once_with(model="text-embedding-ada-002")
            assert result is instance

    def test_custom_model_name_is_forwarded(self):
        with patch("src.embedder.OpenAIEmbeddings") as MockEmb:
            get_embeddings_model(model="text-embedding-3-small")
            MockEmb.assert_called_once_with(model="text-embedding-3-small")


class TestEmbedTexts:
    def test_returns_list_of_vectors(self):
        fake_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        with patch("src.embedder.OpenAIEmbeddings") as MockEmb:
            mock_instance = MagicMock()
            mock_instance.embed_documents.return_value = fake_vectors
            MockEmb.return_value = mock_instance

            result = embed_texts(["hello", "world"])

            mock_instance.embed_documents.assert_called_once_with(["hello", "world"])
            assert result == fake_vectors

    def test_empty_list_returns_empty_list(self):
        with patch("src.embedder.OpenAIEmbeddings") as MockEmb:
            mock_instance = MagicMock()
            mock_instance.embed_documents.return_value = []
            MockEmb.return_value = mock_instance

            result = embed_texts([])
            assert result == []

    def test_custom_model_is_used(self):
        with patch("src.embedder.OpenAIEmbeddings") as MockEmb:
            mock_instance = MagicMock()
            mock_instance.embed_documents.return_value = [[0.0]]
            MockEmb.return_value = mock_instance

            embed_texts(["test"], model="text-embedding-3-large")
            MockEmb.assert_called_once_with(model="text-embedding-3-large")
