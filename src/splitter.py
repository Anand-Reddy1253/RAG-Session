"""Text splitting utilities for the RAG pipeline.

Splits LangChain ``Document`` objects into smaller chunks so that each
chunk fits comfortably inside the context window of an embedding model.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Split *documents* into overlapping text chunks.

    Args:
        documents: Source documents produced by the loader.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.

    Returns:
        A list of ``Document`` objects where each ``page_content`` is a chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)
