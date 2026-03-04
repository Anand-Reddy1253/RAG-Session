"""Tests for src.generator — LLM response-generation utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.generator import generate_answer


def _make_doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "test"})


class TestGenerateAnswer:
    def test_returns_llm_response_content(self):
        mock_response = MagicMock()
        mock_response.content = "42 days per year."

        with patch("src.generator.ChatOpenAI") as MockLLM:
            mock_llm_instance = MagicMock()
            MockLLM.return_value = mock_llm_instance
            # Simulate the chain: (prompt | llm).invoke(...)
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            mock_llm_instance.__ror__ = MagicMock(return_value=mock_chain)

            with patch("src.generator._PROMPT_TEMPLATE") as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)
                result = generate_answer("How many days?", [_make_doc("42 days.")])

        assert result == "42 days per year."

    def test_context_is_built_from_all_docs(self):
        """The content of every context doc must appear in the prompt input."""
        captured: dict = {}

        mock_response = MagicMock()
        mock_response.content = "answer"

        with patch("src.generator.ChatOpenAI") as MockLLM:
            mock_llm_instance = MagicMock()
            MockLLM.return_value = mock_llm_instance

            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = lambda x: (captured.update(x), mock_response)[1]

            with patch("src.generator._PROMPT_TEMPLATE") as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)
                generate_answer(
                    "question",
                    [_make_doc("Doc A content."), _make_doc("Doc B content.")],
                )

        assert "Doc A content." in captured.get("context", "")
        assert "Doc B content." in captured.get("context", "")

    def test_uses_specified_llm_model(self):
        mock_response = MagicMock()
        mock_response.content = "ok"

        with patch("src.generator.ChatOpenAI") as MockLLM:
            mock_llm_instance = MagicMock()
            MockLLM.return_value = mock_llm_instance

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response

            with patch("src.generator._PROMPT_TEMPLATE") as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)
                generate_answer("q", [_make_doc("ctx")], model="gpt-4o")

        MockLLM.assert_called_once_with(model="gpt-4o", temperature=0.0)

    def test_empty_context_still_calls_llm(self):
        mock_response = MagicMock()
        mock_response.content = "no info"

        with patch("src.generator.ChatOpenAI") as MockLLM:
            mock_llm_instance = MagicMock()
            MockLLM.return_value = mock_llm_instance

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response

            with patch("src.generator._PROMPT_TEMPLATE") as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)
                result = generate_answer("anything", [])

        assert result == "no info"
