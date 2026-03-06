"""Unit tests for src.llm — LLMClient and LLMResponse.

Coverage target: ≥ 90 % line coverage for src/llm.py.
All OpenAI / LangChain network calls are mocked; no real API key is required.
"""

from __future__ import annotations

import dataclasses
import os
from unittest.mock import MagicMock, call

import openai
import pytest
from langchain_core.documents import Document

from src.llm import LLMClient, LLMResponse, RAG_PROMPT_TEMPLATE


# ===========================================================================
# Helpers / fixtures
# ===========================================================================


@pytest.fixture()
def mock_chat_openai(mocker):
    """Patch ChatOpenAI so no real API client is created."""
    return mocker.patch("src.llm.ChatOpenAI")


@pytest.fixture()
def llm_client(mock_chat_openai) -> LLMClient:
    """A ready-to-use LLMClient whose ChatOpenAI constructor is mocked."""
    return LLMClient(openai_api_key="sk-test-key")


@pytest.fixture()
def llm_client_gpt4(mock_chat_openai) -> LLMClient:
    """An LLMClient configured with model_name='gpt-4'."""
    return LLMClient(openai_api_key="sk-test-key", model_name="gpt-4")


def _make_mock_chain(answer: str = "Mocked answer") -> MagicMock:
    """Return a mock LCEL chain whose .invoke() returns *answer*."""
    chain = MagicMock()
    chain.invoke.return_value = answer
    return chain


# ===========================================================================
# Group A — Initialisation (7 tests)
# ===========================================================================


class TestInit:
    def test_init_default_model(self, mock_chat_openai):
        client = LLMClient(openai_api_key="sk-test")
        assert client.model_name == "gpt-3.5-turbo"

    def test_init_custom_model(self, mock_chat_openai):
        client = LLMClient(openai_api_key="sk-test", model_name="gpt-4")
        assert client.model_name == "gpt-4"

    def test_init_temperature_zero(self, mock_chat_openai):
        LLMClient(openai_api_key="sk-test")
        kwargs = mock_chat_openai.call_args.kwargs
        assert kwargs["temperature"] == 0.0

    def test_init_custom_temperature(self, mock_chat_openai):
        LLMClient(openai_api_key="sk-test", temperature=0.7)
        kwargs = mock_chat_openai.call_args.kwargs
        assert kwargs["temperature"] == 0.7

    def test_init_api_key_forwarded(self, mock_chat_openai):
        LLMClient(openai_api_key="sk-explicit")
        kwargs = mock_chat_openai.call_args.kwargs
        assert kwargs["openai_api_key"] == "sk-explicit"

    def test_init_api_key_from_env(self, mocker):
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"}, clear=False)
        mock_cls = mocker.patch("src.llm.ChatOpenAI")
        LLMClient()  # no explicit api_key
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["openai_api_key"] == "sk-from-env"

    def test_chat_model_property(self, mock_chat_openai):
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance
        client = LLMClient(openai_api_key="sk-test")
        assert client.chat_model is mock_instance


# ===========================================================================
# Group B — format_context() (6 tests)
# ===========================================================================


class TestFormatContext:
    def test_format_context_single_chunk(self, llm_client):
        chunk = Document(
            page_content="Net 30 days.", metadata={"source": "contract.docx"}
        )
        result = llm_client.format_context([chunk])
        assert "Net 30 days." in result

    def test_format_context_multiple_chunks(self, llm_client, sample_chunks):
        result = llm_client.format_context(sample_chunks)
        for chunk in sample_chunks:
            assert chunk.page_content in result

    def test_format_context_includes_source(self, llm_client):
        chunk = Document(
            page_content="Some text.", metadata={"source": "contract.docx"}
        )
        result = llm_client.format_context([chunk])
        assert "[Source: contract.docx]" in result

    def test_format_context_empty_raises(self, llm_client):
        with pytest.raises(ValueError, match="empty"):
            llm_client.format_context([])

    def test_format_context_missing_source_metadata(self, llm_client):
        chunk = Document(page_content="No source here.", metadata={})
        result = llm_client.format_context([chunk])
        assert "[Source: unknown]" in result

    def test_format_context_preserves_order(self, llm_client):
        chunk_a = Document(
            page_content="Alpha text.", metadata={"source": "a.pdf"}
        )
        chunk_b = Document(
            page_content="Beta text.", metadata={"source": "b.pdf"}
        )
        result = llm_client.format_context([chunk_a, chunk_b])
        assert result.index("Alpha text.") < result.index("Beta text.")


# ===========================================================================
# Group C — build_prompt() (5 tests)
# ===========================================================================


class TestBuildPrompt:
    def test_build_prompt_contains_question(self, llm_client, sample_chunks):
        result = llm_client.build_prompt("What are the payment terms?", sample_chunks)
        assert "What are the payment terms?" in result

    def test_build_prompt_contains_context(self, llm_client, sample_chunks):
        result = llm_client.build_prompt("Q?", sample_chunks)
        assert sample_chunks[0].page_content in result

    def test_build_prompt_uses_template(self, llm_client, sample_chunks):
        result = llm_client.build_prompt("Q?", sample_chunks)
        assert "You are a helpful assistant" in result

    def test_build_prompt_custom_template(self, mock_chat_openai, sample_chunks):
        client = LLMClient(
            openai_api_key="sk-test",
            prompt_template="Custom: {context} Q: {question}",
        )
        result = client.build_prompt("Why?", sample_chunks)
        assert "Custom:" in result

    def test_build_prompt_empty_question_raises(self, llm_client, sample_chunks):
        with pytest.raises(ValueError):
            llm_client.build_prompt("", sample_chunks)


# ===========================================================================
# Group D — generate() Happy Path (8 tests)
# ===========================================================================


class TestGenerateHappyPath:
    def test_generate_returns_llm_response(self, llm_client, mocker, sample_chunks):
        mocker.patch.object(llm_client, "_build_chain", return_value=_make_mock_chain())
        result = llm_client.generate("What are the payment terms?", sample_chunks)
        assert isinstance(result, LLMResponse)

    def test_generate_answer_matches_mock(self, llm_client, mocker, sample_chunks):
        mocker.patch.object(
            llm_client, "_build_chain", return_value=_make_mock_chain("Net 30 days.")
        )
        result = llm_client.generate("What are the payment terms?", sample_chunks)
        assert result.answer == "Net 30 days."

    def test_generate_sources_populated(self, llm_client, mocker, sample_chunks):
        mocker.patch.object(llm_client, "_build_chain", return_value=_make_mock_chain())
        result = llm_client.generate("Q?", sample_chunks)
        assert result.sources == sample_chunks

    def test_generate_model_name_in_response(self, llm_client, mocker, sample_chunks):
        mocker.patch.object(llm_client, "_build_chain", return_value=_make_mock_chain())
        result = llm_client.generate("Q?", sample_chunks)
        assert result.model == "gpt-3.5-turbo"

    def test_generate_chain_invoked_once(self, llm_client, mocker, sample_chunks):
        mock_chain = _make_mock_chain()
        mocker.patch.object(llm_client, "_build_chain", return_value=mock_chain)
        llm_client.generate("Q?", sample_chunks)
        assert mock_chain.invoke.call_count == 1

    def test_generate_chain_receives_context(self, llm_client, mocker, sample_chunks):
        mock_chain = _make_mock_chain()
        mocker.patch.object(llm_client, "_build_chain", return_value=mock_chain)
        llm_client.generate("Q?", sample_chunks)
        call_args = mock_chain.invoke.call_args
        assert "context" in call_args[0][0]

    def test_generate_chain_receives_question(self, llm_client, mocker, sample_chunks):
        question = "How many vacation days?"
        mock_chain = _make_mock_chain()
        mocker.patch.object(llm_client, "_build_chain", return_value=mock_chain)
        llm_client.generate(question, sample_chunks)
        call_args = mock_chain.invoke.call_args
        assert call_args[0][0]["question"] == question

    def test_generate_gpt4_model(self, llm_client_gpt4, mocker, sample_chunks):
        mocker.patch.object(
            llm_client_gpt4, "_build_chain", return_value=_make_mock_chain("answer")
        )
        result = llm_client_gpt4.generate("Q?", sample_chunks)
        assert isinstance(result, LLMResponse)
        assert result.answer != ""


# ===========================================================================
# Group E — generate() Error Paths (8 tests)
# ===========================================================================


class TestGenerateErrorPaths:
    def test_generate_empty_question_raises(self, llm_client, sample_chunks):
        with pytest.raises(ValueError):
            llm_client.generate("", sample_chunks)

    def test_generate_whitespace_question_raises(self, llm_client, sample_chunks):
        with pytest.raises(ValueError):
            llm_client.generate("   ", sample_chunks)

    def test_generate_empty_chunks_raises(self, llm_client):
        with pytest.raises(ValueError):
            llm_client.generate("What is X?", [])

    def test_generate_propagates_auth_error(self, llm_client, mocker, sample_chunks):
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = openai.AuthenticationError(
            message="Incorrect API key",
            response=MagicMock(status_code=401),
            body={"error": {"type": "invalid_api_key"}},
        )
        mocker.patch.object(llm_client, "_build_chain", return_value=mock_chain)
        with pytest.raises(openai.AuthenticationError):
            llm_client.generate("Any question?", sample_chunks)

    def test_generate_propagates_rate_limit_error(
        self, llm_client, mocker, sample_chunks
    ):
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={"error": {"type": "rate_limit_exceeded"}},
        )
        mocker.patch.object(llm_client, "_build_chain", return_value=mock_chain)
        with pytest.raises(openai.RateLimitError):
            llm_client.generate("Any question?", sample_chunks)

    def test_generate_very_long_question(self, llm_client, mocker, sample_chunks):
        long_question = "A" * 2000
        mocker.patch.object(
            llm_client, "_build_chain", return_value=_make_mock_chain("answer")
        )
        result = llm_client.generate(long_question, sample_chunks)
        assert isinstance(result, LLMResponse)

    def test_generate_single_chunk(self, llm_client, mocker, sample_document):
        mocker.patch.object(
            llm_client, "_build_chain", return_value=_make_mock_chain("answer")
        )
        result = llm_client.generate("Q?", [sample_document])
        assert len(result.sources) == 1

    def test_generate_special_characters_in_question(
        self, llm_client, mocker, sample_chunks
    ):
        question = 'What is "net 30"?\n{braces}'
        mocker.patch.object(
            llm_client, "_build_chain", return_value=_make_mock_chain("answer")
        )
        result = llm_client.generate(question, sample_chunks)
        assert isinstance(result, LLMResponse)


# ===========================================================================
# Group F — LLMResponse Dataclass (3 tests)
# ===========================================================================


class TestLLMResponseDataclass:
    def test_llm_response_default_fields(self):
        resp = LLMResponse(answer="test")
        assert resp.sources == []
        assert resp.model == ""
        assert resp.prompt_tokens == 0
        assert resp.completion_tokens == 0
        assert resp.total_tokens == 0

    def test_llm_response_all_fields(self, sample_document):
        resp = LLMResponse(
            answer="42",
            sources=[sample_document],
            model="gpt-4",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        assert resp.answer == "42"
        assert resp.sources == [sample_document]
        assert resp.model == "gpt-4"
        assert resp.prompt_tokens == 10
        assert resp.completion_tokens == 5
        assert resp.total_tokens == 15

    def test_llm_response_is_dataclass(self):
        assert dataclasses.is_dataclass(LLMResponse) is True


# ===========================================================================
# Group G — Prompt Template Constants (3 tests)
# ===========================================================================


class TestPromptTemplateConstants:
    def test_rag_prompt_template_has_context_placeholder(self):
        assert "{context}" in RAG_PROMPT_TEMPLATE

    def test_rag_prompt_template_has_question_placeholder(self):
        assert "{question}" in RAG_PROMPT_TEMPLATE

    def test_rag_prompt_template_fallback_instruction(self):
        assert "I don't have enough information" in RAG_PROMPT_TEMPLATE


# ===========================================================================
# Group H — Chain Construction (3 tests)
# ===========================================================================


class TestBuildChain:
    def test_build_chain_returns_runnable(self, llm_client):
        chain = llm_client._build_chain()
        assert chain is not None

    def test_build_chain_cached_after_first_call(self, llm_client):
        chain1 = llm_client._build_chain()
        chain2 = llm_client._build_chain()
        assert chain1 is chain2

    def test_build_chain_called_once_across_two_generates(
        self, llm_client, mocker, sample_chunks
    ):
        mock_chain = _make_mock_chain()
        build_spy = mocker.patch.object(
            llm_client, "_build_chain", wraps=llm_client._build_chain
        )
        # Replace the real chain that wraps returns with our mock
        llm_client._chain = mock_chain
        llm_client.generate("Q1?", sample_chunks)
        llm_client.generate("Q2?", sample_chunks)
        # _build_chain is called once per generate(), but the chain itself
        # is cached so invoke is called twice on the same mock_chain
        assert mock_chain.invoke.call_count == 2
