"""FAISS vector store helpers for the RAG pipeline."""

import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def build_vector_store(
    documents: list[Document],
    embeddings: OpenAIEmbeddings,
    persist_path: str | None = None,
) -> FAISS:
    """Create a FAISS vector store from *documents*.

    Args:
        documents: Chunked documents to index.
        embeddings: Embedding model used to encode the documents.
        persist_path: If provided, the vector store is saved to this directory
            so it can be reloaded later without re-embedding.

    Returns:
        A :class:`~langchain_community.vectorstores.FAISS` instance.
    """
    vector_store = FAISS.from_documents(documents, embeddings)
    if persist_path:
        os.makedirs(persist_path, exist_ok=True)
        vector_store.save_local(persist_path)
    return vector_store


def load_vector_store(
    persist_path: str,
    embeddings: OpenAIEmbeddings,
) -> FAISS:
    """Load a previously persisted FAISS vector store.

    Args:
        persist_path: Directory where the vector store was saved by
            :func:`build_vector_store`.  The directory **must** already exist;
            call :func:`build_vector_store` first if it does not.
        embeddings: Embedding model used when the store was originally built.

    Returns:
        A :class:`~langchain_community.vectorstores.FAISS` instance.

    Raises:
        FileNotFoundError: If *persist_path* does not exist.

    .. warning::
        FAISS uses pickle-based serialisation under the hood.  Only load stores
        from trusted sources — never from user-supplied or untrusted paths.
    """
    if not os.path.exists(persist_path):
        raise FileNotFoundError(
            f"Vector store not found at '{persist_path}'. "
            "Run the ingestion step first."
        )
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
