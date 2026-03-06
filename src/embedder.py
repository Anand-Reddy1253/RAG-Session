"""Vector store management (embeddings + FAISS) for the RAG pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


class VectorStoreManager:
    """Manages OpenAI embeddings and a FAISS vector store."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
        vector_store_path: str | Path = "./vector_store",
    ) -> None:
        self._openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._embedding_model = embedding_model
        self.vector_store_path = Path(vector_store_path)
        self._embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=self._openai_api_key,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        return self._embeddings

    def build(self, chunks: List[Document]) -> FAISS:
        """Create a FAISS index from *chunks*."""
        if not chunks:
            raise ValueError("Cannot build a vector store from an empty chunk list.")
        return FAISS.from_documents(chunks, self._embeddings)

    def save(self, vector_store: FAISS) -> None:
        """Persist *vector_store* to :attr:`vector_store_path`."""
        vector_store.save_local(str(self.vector_store_path))

    def load(self) -> FAISS:
        """Load a previously saved FAISS index from :attr:`vector_store_path`."""
        return FAISS.load_local(
            str(self.vector_store_path),
            self._embeddings,
            allow_dangerous_deserialization=True,
        )

    def build_and_save(self, chunks: List[Document]) -> FAISS:
        """Build and immediately persist a FAISS index."""
        vector_store = self.build(chunks)
        self.save(vector_store)
        return vector_store

    def similarity_search(
        self,
        query: str,
        vector_store: FAISS,
        top_k: int = 4,
    ) -> List[Document]:
        """Return the *top_k* most relevant chunks for *query*."""
        return vector_store.similarity_search(query, k=top_k)
