"""Tests for rag.embedder — OpenAI embedding wrapper."""

from unittest.mock import MagicMock, patch

import pytest

from rag.embedder import build_embedder


class TestBuildEmbedder:
    def test_returns_openai_embeddings_instance(self):
        with patch("rag.embedder.OpenAIEmbeddings") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            result = build_embedder(model="text-embedding-ada-002")

        mock_cls.assert_called_once_with(model="text-embedding-ada-002")
        assert result is mock_instance

    def test_passes_api_key_when_provided(self):
        with patch("rag.embedder.OpenAIEmbeddings") as mock_cls:
            build_embedder(model="text-embedding-ada-002", api_key="sk-test")

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["api_key"] == "sk-test"

    def test_omits_api_key_when_none(self):
        with patch("rag.embedder.OpenAIEmbeddings") as mock_cls:
            build_embedder(model="text-embedding-ada-002", api_key=None)

        call_kwargs = mock_cls.call_args[1]
        assert "api_key" not in call_kwargs

    def test_default_model_is_ada(self):
        with patch("rag.embedder.OpenAIEmbeddings") as mock_cls:
            build_embedder()

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model"] == "text-embedding-ada-002"
