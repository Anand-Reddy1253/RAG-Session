"""End-to-end RAG pipeline orchestrating loader, embedder, and LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.embedder import VectorStoreManager
from src.llm import LLMClient, LLMResponse
from src.loader import DocumentLoader


@dataclass
class RAGResponse:
    """Structured response returned by :meth:`RAGPipeline.ask`."""

    question: str
    answer: str
    sources: List[Document] = field(default_factory=list)
    model: str = ""
    total_tokens: int = 0


class RAGPipeline:
    """Orchestrates document loading, vector indexing, retrieval, and generation."""

    def __init__(
        self,
        docs_dir: str | Path = "./docs",
        openai_api_key: Optional[str] = None,
        vector_store_path: str | Path = "./vector_store",
        embedding_model: str = "text-embedding-ada-002",
        chat_model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 4,
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.top_k = top_k

        self._loader = DocumentLoader(
            docs_dir=docs_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._vsm = VectorStoreManager(
            openai_api_key=openai_api_key,
            embedding_model=embedding_model,
            vector_store_path=vector_store_path,
        )
        self._llm = LLMClient(
            openai_api_key=openai_api_key,
            model_name=chat_model,
            temperature=temperature,
        )
        self._vector_store: Optional[FAISS] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_index_loaded(self) -> bool:
        return self._vector_store is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(self) -> None:
        """Load documents, build the vector index, and persist it to disk."""
        chunks = self._loader.load_and_split()
        vector_store = self._vsm.build(chunks)
        self._vsm.save(vector_store)
        self._vector_store = vector_store

    def ask(self, question: str) -> RAGResponse:
        """Answer *question* using the vector store and LLM."""
        vector_store = self._ensure_index_loaded()
        chunks = self._vsm.similarity_search(question, vector_store, top_k=self.top_k)
        llm_response: LLMResponse = self._llm.generate(question, chunks)

        return RAGResponse(
            question=question,
            answer=llm_response.answer,
            sources=llm_response.sources,
            model=llm_response.model,
            total_tokens=llm_response.total_tokens,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_index_loaded(self) -> FAISS:
        if self._vector_store is None:
            self._vector_store = self._vsm.load()
        return self._vector_store
