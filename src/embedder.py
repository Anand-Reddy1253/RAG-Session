"""Embedding utilities for the RAG pipeline.

Wraps the OpenAI embedding model so that the rest of the application can
call a single function to obtain vector embeddings for a list of texts.
"""

from __future__ import annotations

from typing import List

from langchain_openai import OpenAIEmbeddings


def get_embeddings_model(model: str = "text-embedding-ada-002") -> OpenAIEmbeddings:
    """Return an ``OpenAIEmbeddings`` instance for the given *model*.

    The OpenAI API key is read from the ``OPENAI_API_KEY`` environment
    variable (or from a loaded ``.env`` file).
    """
    return OpenAIEmbeddings(model=model)


def embed_texts(texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
    """Return embedding vectors for each string in *texts*.

    Args:
        texts: Plain-text strings to embed.
        model: OpenAI embedding model name.

    Returns:
        A list of float vectors, one per input text.
    """
    embeddings = get_embeddings_model(model)
    return embeddings.embed_documents(texts)
