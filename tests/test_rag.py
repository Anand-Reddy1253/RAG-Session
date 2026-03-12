"""Tests for rag.chain and rag.vector_store."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.chain import build_rag_chain
from rag.memory import ConversationMemory
from rag.vector_store import build_vector_store, load_vector_store


class TestBuildVectorStore:
    def test_creates_faiss_from_documents(self):
        docs = [Document(page_content="hello world", metadata={"source": "test"})]
        mock_embeddings = MagicMock()

        with patch("rag.vector_store.FAISS") as mock_faiss_cls:
            mock_store = MagicMock()
            mock_faiss_cls.from_documents.return_value = mock_store

            result = build_vector_store(docs, mock_embeddings)

        mock_faiss_cls.from_documents.assert_called_once_with(docs, mock_embeddings)
        assert result is mock_store

    def test_saves_to_disk_when_persist_path_given(self, tmp_path):
        docs = [Document(page_content="text", metadata={})]
        mock_embeddings = MagicMock()
        persist_path = str(tmp_path / "store")

        with patch("rag.vector_store.FAISS") as mock_faiss_cls:
            mock_store = MagicMock()
            mock_faiss_cls.from_documents.return_value = mock_store

            build_vector_store(docs, mock_embeddings, persist_path=persist_path)

        mock_store.save_local.assert_called_once_with(persist_path)

    def test_does_not_save_when_no_persist_path(self):
        docs = [Document(page_content="text", metadata={})]
        mock_embeddings = MagicMock()

        with patch("rag.vector_store.FAISS") as mock_faiss_cls:
            mock_store = MagicMock()
            mock_faiss_cls.from_documents.return_value = mock_store

            build_vector_store(docs, mock_embeddings)

        mock_store.save_local.assert_not_called()


class TestLoadVectorStore:
    def test_loads_from_existing_path(self, tmp_path):
        store_path = str(tmp_path)
        mock_embeddings = MagicMock()

        with patch("rag.vector_store.FAISS") as mock_faiss_cls:
            mock_store = MagicMock()
            mock_faiss_cls.load_local.return_value = mock_store

            result = load_vector_store(store_path, mock_embeddings)

        mock_faiss_cls.load_local.assert_called_once_with(
            store_path, mock_embeddings, allow_dangerous_deserialization=True
        )
        assert result is mock_store

    def test_raises_when_path_does_not_exist(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_vector_store("/nonexistent/path", MagicMock())


class TestBuildRagChain:
    def test_returns_runnable_with_message_history(self):
        from langchain_core.runnables.history import RunnableWithMessageHistory

        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        memory = ConversationMemory()

        with (
            patch("rag.chain.create_history_aware_retriever") as mock_har,
            patch("rag.chain.create_stuff_documents_chain") as mock_qa,
            patch("rag.chain.create_retrieval_chain") as mock_rc,
        ):
            mock_har.return_value = MagicMock()
            mock_qa.return_value = MagicMock()
            mock_rc.return_value = MagicMock()

            chain = build_rag_chain(mock_llm, mock_retriever, memory)

        assert isinstance(chain, RunnableWithMessageHistory)

    def test_chain_uses_memory_get_session_history(self):
        from langchain_core.runnables.history import RunnableWithMessageHistory

        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        memory = ConversationMemory()

        with (
            patch("rag.chain.create_history_aware_retriever"),
            patch("rag.chain.create_stuff_documents_chain"),
            patch("rag.chain.create_retrieval_chain") as mock_rc,
        ):
            mock_rc.return_value = MagicMock()

            chain = build_rag_chain(mock_llm, mock_retriever, memory)

        # Both callables refer to the same underlying function bound to the same object.
        assert chain.get_session_history.__func__ is memory.get_session_history.__func__
        assert chain.get_session_history.__self__ is memory.get_session_history.__self__
