"""Retrieval utilities for the RAG pipeline.

Wraps a FAISS vector store to provide a simple similarity-search interface.
"""

from __future__ import annotations

from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def retrieve(
    store: FAISS,
    query: str,
    k: int = 4,
) -> List[Document]:
    """Return the *k* most relevant document chunks for *query*.

    Args:
        store: A populated FAISS vector store.
        query: The user's natural-language question.
        k: Number of chunks to retrieve.

    Returns:
        A list of at most *k* ``Document`` objects ordered by relevance.
    """
    return store.similarity_search(query, k=k)
