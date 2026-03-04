"""End-to-end RAG pipeline orchestration.

This module ties together loading, splitting, embedding, storing, retrieving,
and generating into a single ``RAGPipeline`` class.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.embedder import get_embeddings_model
from src.generator import generate_answer
from src.loader import load_documents
from src.retriever import retrieve
from src.splitter import split_documents
from src.vector_store import build_vector_store, load_vector_store, save_vector_store


class RAGPipeline:
    """Orchestrates the full RAG workflow.

    Usage::

        pipeline = RAGPipeline(docs_dir="./docs", vector_store_path="./vector_store")
        pipeline.ingest()                     # load, split, embed, persist
        answer = pipeline.query("What …?")   # retrieve + generate
    """

    def __init__(
        self,
        docs_dir: str | Path = "./docs",
        vector_store_path: str | Path = "./vector_store",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_k: int = 4,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.vector_store_path = Path(vector_store_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.llm_model = llm_model
        self._store = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self) -> List[Document]:
        """Load documents, split them, build a vector store, and persist it.

        Returns:
            The list of text chunks that were indexed.
        """
        raw_docs = load_documents(self.docs_dir)
        chunks = split_documents(raw_docs, self.chunk_size, self.chunk_overlap)
        embeddings = get_embeddings_model()
        self._store = build_vector_store(chunks, embeddings)
        save_vector_store(self._store, self.vector_store_path)
        return chunks

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def load_store(self) -> None:
        """Load a previously persisted vector store from disk."""
        embeddings = get_embeddings_model()
        self._store = load_vector_store(self.vector_store_path, embeddings)

    def query(self, question: str) -> str:
        """Answer *question* using the vector store.

        If the store has not been loaded yet, ``load_store`` is called
        automatically.

        Args:
            question: The user's natural-language question.

        Returns:
            A grounded answer string.
        """
        if self._store is None:
            self.load_store()
        context_docs = retrieve(self._store, question, k=self.retrieval_k)
        return generate_answer(question, context_docs, model=self.llm_model)
