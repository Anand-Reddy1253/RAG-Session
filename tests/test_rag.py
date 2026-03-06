"""Unit tests for src.rag — RAGPipeline and RAGResponse."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.llm import LLMResponse
from src.rag import RAGPipeline, RAGResponse


# ===========================================================================
# Helpers
# ===========================================================================


def _make_pipeline(mocker, top_k: int = 4, *, chat_model: str = "gpt-3.5-turbo"):
    """Create a RAGPipeline with all three collaborators mocked."""
    mock_loader_cls = mocker.patch("src.rag.DocumentLoader")
    mock_vsm_cls = mocker.patch("src.rag.VectorStoreManager")
    mock_llm_cls = mocker.patch("src.rag.LLMClient")
    return (
        RAGPipeline(
            docs_dir="./docs",
            openai_api_key="sk-test",
            top_k=top_k,
            chat_model=chat_model,
        ),
        mock_loader_cls.return_value,
        mock_vsm_cls.return_value,
        mock_llm_cls.return_value,
    )


# ===========================================================================
# Group — build_index()
# ===========================================================================


class TestBuildIndex:
    def test_build_index_calls_load_and_split(self, mocker, sample_chunks):
        pipeline, mock_loader, mock_vsm, _ = _make_pipeline(mocker)
        mock_loader.load_and_split.return_value = sample_chunks
        mock_vsm.build.return_value = MagicMock()

        pipeline.build_index()

        mock_loader.load_and_split.assert_called_once()

    def test_build_index_calls_vsm_build(self, mocker, sample_chunks):
        pipeline, mock_loader, mock_vsm, _ = _make_pipeline(mocker)
        mock_loader.load_and_split.return_value = sample_chunks
        mock_vsm.build.return_value = MagicMock()

        pipeline.build_index()

        mock_vsm.build.assert_called_once_with(sample_chunks)

    def test_build_index_calls_vsm_save(self, mocker, sample_chunks):
        pipeline, mock_loader, mock_vsm, _ = _make_pipeline(mocker)
        mock_store = MagicMock()
        mock_loader.load_and_split.return_value = sample_chunks
        mock_vsm.build.return_value = mock_store

        pipeline.build_index()

        mock_vsm.save.assert_called_once_with(mock_store)


# ===========================================================================
# Group — ask()
# ===========================================================================


class TestAsk:
    def test_ask_loads_index_if_not_cached(self, mocker, sample_chunks):
        pipeline, _, mock_vsm, mock_llm = _make_pipeline(mocker)
        mock_vsm.load.return_value = MagicMock()
        mock_vsm.similarity_search.return_value = sample_chunks
        mock_llm.generate.return_value = LLMResponse(answer="42", sources=sample_chunks)

        pipeline.ask("What is the answer?")

        mock_vsm.load.assert_called_once()

    def test_ask_does_not_reload_if_cached(self, mocker, sample_chunks):
        pipeline, _, mock_vsm, mock_llm = _make_pipeline(mocker)
        mock_vsm.load.return_value = MagicMock()
        mock_vsm.similarity_search.return_value = sample_chunks
        mock_llm.generate.return_value = LLMResponse(answer="x", sources=sample_chunks)

        pipeline.ask("Q1?")
        pipeline.ask("Q2?")

        assert mock_vsm.load.call_count == 1

    def test_ask_calls_similarity_search(self, mocker, sample_chunks):
        pipeline, _, mock_vsm, mock_llm = _make_pipeline(mocker)
        mock_vsm.load.return_value = MagicMock()
        mock_vsm.similarity_search.return_value = sample_chunks
        mock_llm.generate.return_value = LLMResponse(answer="a", sources=sample_chunks)

        pipeline.ask("What are payment terms?")

        mock_vsm.similarity_search.assert_called_once()
        call_args = mock_vsm.similarity_search.call_args
        assert call_args[0][0] == "What are payment terms?"

    def test_ask_passes_top_k(self, mocker, sample_chunks):
        pipeline, _, mock_vsm, mock_llm = _make_pipeline(mocker, top_k=6)
        mock_vsm.load.return_value = MagicMock()
        mock_vsm.similarity_search.return_value = sample_chunks
        mock_llm.generate.return_value = LLMResponse(answer="a", sources=sample_chunks)

        pipeline.ask("Q?")

        call_kwargs = mock_vsm.similarity_search.call_args.kwargs
        assert call_kwargs.get("top_k") == 6

    def test_ask_calls_llm_generate(self, mocker, sample_chunks):
        pipeline, _, mock_vsm, mock_llm = _make_pipeline(mocker)
        mock_vsm.load.return_value = MagicMock()
        mock_vsm.similarity_search.return_value = sample_chunks
        mock_llm.generate.return_value = LLMResponse(answer="a", sources=sample_chunks)

        pipeline.ask("Q?")

        mock_llm.generate.assert_called_once()

    def test_ask_returns_rag_response(self, mocker, sample_chunks):
        pipeline, _, mock_vsm, mock_llm = _make_pipeline(mocker)
        mock_vsm.load.return_value = MagicMock()
        mock_vsm.similarity_search.return_value = sample_chunks
        mock_llm.generate.return_value = LLMResponse(answer="42", sources=sample_chunks)

        result = pipeline.ask("Q?")

        assert isinstance(result, RAGResponse)

    def test_ask_response_contains_answer(self, mocker, sample_chunks):
        pipeline, _, mock_vsm, mock_llm = _make_pipeline(mocker)
        mock_vsm.load.return_value = MagicMock()
        mock_vsm.similarity_search.return_value = sample_chunks
        mock_llm.generate.return_value = LLMResponse(answer="42", sources=sample_chunks)

        result = pipeline.ask("What is the answer?")

        assert result.answer == "42"

    def test_ask_response_contains_sources(self, mocker, sample_chunks):
        pipeline, _, mock_vsm, mock_llm = _make_pipeline(mocker)
        mock_vsm.load.return_value = MagicMock()
        mock_vsm.similarity_search.return_value = sample_chunks[:2]
        mock_llm.generate.return_value = LLMResponse(
            answer="a", sources=sample_chunks[:2]
        )

        result = pipeline.ask("Q?")

        assert result.sources == sample_chunks[:2]

    def test_ask_empty_question_propagates_value_error(self, mocker, sample_chunks):
        pipeline, _, mock_vsm, mock_llm = _make_pipeline(mocker)
        mock_vsm.load.return_value = MagicMock()
        mock_vsm.similarity_search.return_value = sample_chunks
        mock_llm.generate.side_effect = ValueError("empty question")

        with pytest.raises(ValueError):
            pipeline.ask("")


# ===========================================================================
# Group — is_index_loaded property
# ===========================================================================


class TestIsIndexLoaded:
    def test_is_index_loaded_false_initially(self, mocker):
        mocker.patch("src.rag.DocumentLoader")
        mocker.patch("src.rag.VectorStoreManager")
        mocker.patch("src.rag.LLMClient")

        pipeline = RAGPipeline(docs_dir="./docs", openai_api_key="sk-test")
        assert pipeline.is_index_loaded is False

    def test_is_index_loaded_true_after_ask(self, mocker, sample_chunks):
        pipeline, _, mock_vsm, mock_llm = _make_pipeline(mocker)
        mock_vsm.load.return_value = MagicMock()
        mock_vsm.similarity_search.return_value = sample_chunks
        mock_llm.generate.return_value = LLMResponse(answer="a", sources=sample_chunks)

        pipeline.ask("Q?")

        assert pipeline.is_index_loaded is True

    def test_is_index_loaded_true_after_build_index(self, mocker, sample_chunks):
        pipeline, mock_loader, mock_vsm, _ = _make_pipeline(mocker)
        mock_loader.load_and_split.return_value = sample_chunks
        mock_vsm.build.return_value = MagicMock()

        pipeline.build_index()

        assert pipeline.is_index_loaded is True


# ===========================================================================
# Group — RAGResponse dataclass
# ===========================================================================


class TestRAGResponseDataclass:
    def test_rag_response_is_dataclass(self):
        assert dataclasses.is_dataclass(RAGResponse) is True

    def test_rag_response_default_fields(self):
        resp = RAGResponse(question="Q?", answer="A")
        assert resp.sources == []
        assert resp.model == ""
        assert resp.total_tokens == 0

    def test_rag_response_all_fields(self, sample_document):
        resp = RAGResponse(
            question="Q?",
            answer="A",
            sources=[sample_document],
            model="gpt-4",
            total_tokens=100,
        )
        assert resp.question == "Q?"
        assert resp.answer == "A"
        assert resp.sources == [sample_document]
        assert resp.model == "gpt-4"
        assert resp.total_tokens == 100
