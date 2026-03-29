"""Response-generation utilities for the RAG pipeline.

Given a user query and a list of retrieved context chunks, this module
builds a prompt and calls an OpenAI chat model to produce a grounded answer.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "information provided in the context below. If the answer is not contained "
    "in the context, say \"I don't have enough information to answer that.\"\n\n"
    "Context:\n{context}"
)

_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def generate_answer(
    query: str,
    context_docs: List[Document],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> str:
    """Generate a grounded answer for *query* using *context_docs*.

    Args:
        query: The user's natural-language question.
        context_docs: Retrieved document chunks used as context.
        model: OpenAI chat model name.
        temperature: Sampling temperature (0 = deterministic).

    Returns:
        The model's answer as a plain string.
    """
    context = "\n\n".join(doc.page_content for doc in context_docs)
    llm = ChatOpenAI(model=model, temperature=temperature)
    chain = _PROMPT_TEMPLATE | llm
    response = chain.invoke({"context": context, "question": query})
    content = getattr(response, "content", None)
    if not isinstance(content, str):
        raise ValueError(f"Unexpected response type from LLM: {type(response)!r}")
    return content
