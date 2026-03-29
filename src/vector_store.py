"""Vector store utilities for the RAG pipeline.

Builds and persists a FAISS index from a list of LangChain ``Document``
objects, and loads a previously saved index from disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.embedder import get_embeddings_model


def build_vector_store(
    documents: List[Document],
    embeddings: OpenAIEmbeddings | None = None,
) -> FAISS:
    """Create a FAISS vector store from *documents*.

    Args:
        documents: Text chunks to index.
        embeddings: Embedding model to use. Defaults to the OpenAI model.

    Returns:
        An in-memory ``FAISS`` vector store.
    """
    if embeddings is None:
        embeddings = get_embeddings_model()
    return FAISS.from_documents(documents, embeddings)


def save_vector_store(store: FAISS, path: str | Path) -> None:
    """Persist *store* to *path* on disk."""
    store.save_local(str(path))


def load_vector_store(
    path: str | Path,
    embeddings: OpenAIEmbeddings | None = None,
) -> FAISS:
    """Load a previously saved FAISS vector store from *path*.

    Args:
        path: Directory that was previously passed to ``save_vector_store``.
        embeddings: Embedding model used when the store was created.

    Returns:
        The deserialized ``FAISS`` vector store.
    """
    if embeddings is None:
        embeddings = get_embeddings_model()
    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
