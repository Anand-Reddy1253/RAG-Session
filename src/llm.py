"""LLM client for the RAG pipeline — wraps ChatOpenAI via LangChain LCEL."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Default prompt template
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """\
You are a helpful assistant that answers questions using only the context provided below.
If the answer cannot be found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Structured response returned by :meth:`LLMClient.generate`."""

    answer: str
    sources: List[Document] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


class LLMClient:
    """Wraps a LangChain LCEL chain built from ``ChatOpenAI``."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 512,
        prompt_template: str = RAG_PROMPT_TEMPLATE,
    ) -> None:
        self._openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._prompt_template = prompt_template
        self._chat_model = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=self._openai_api_key,
        )
        self._chain: Optional[Runnable] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def chat_model(self) -> ChatOpenAI:
        return self._chat_model

    @property
    def model_name(self) -> str:
        return self._model_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format_context(self, chunks: List[Document]) -> str:
        """Format *chunks* into a single context string with source headers."""
        if not chunks:
            raise ValueError("Cannot format context from an empty chunk list.")

        parts: List[str] = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            parts.append(f"[Source: {source}]\n{chunk.page_content}")
        return "\n\n".join(parts)

    def build_prompt(self, question: str, chunks: List[Document]) -> str:
        """Return the fully rendered prompt string (for debugging / logging)."""
        if not question or not question.strip():
            raise ValueError("Question must not be empty or whitespace.")

        context = self.format_context(chunks)
        return self._prompt_template.format(context=context, question=question)

    def generate(self, question: str, chunks: List[Document]) -> LLMResponse:
        """Generate an answer grounded in *chunks* for the given *question*."""
        if not question or not question.strip():
            raise ValueError("Question must not be empty or whitespace.")
        if not chunks:
            raise ValueError("Cannot generate an answer without context chunks.")

        context = self.format_context(chunks)
        chain = self._build_chain()
        answer = chain.invoke({"context": context, "question": question})

        return LLMResponse(
            answer=answer,
            sources=chunks,
            model=self._model_name,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_chain(self) -> Runnable:
        """Build and cache the LCEL chain: prompt | ChatOpenAI | StrOutputParser."""
        if self._chain is None:
            prompt = ChatPromptTemplate.from_template(self._prompt_template)
            self._chain = prompt | self._chat_model | StrOutputParser()
        return self._chain
