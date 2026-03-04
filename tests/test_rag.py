"""Tests for src.rag — end-to-end RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.documents import Document

from src.rag import RAGPipeline


def _make_doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "test"})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pipeline(tmp_path):
    """Return a RAGPipeline whose paths point to temporary directories."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    # Write a minimal CSV so load_documents finds something.
    (docs_dir / "data.csv").write_text("col\nvalue\n", encoding="utf-8")

    store_path = tmp_path / "store"
    return RAGPipeline(
        docs_dir=docs_dir,
        vector_store_path=store_path,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestRAGPipelineInit:
    def test_default_attributes(self, tmp_path):
        p = RAGPipeline()
        assert p.docs_dir == Path("./docs")
        assert p.vector_store_path == Path("./vector_store")
        assert p.chunk_size == 500
        assert p.chunk_overlap == 50
        assert p.retrieval_k == 4
        assert p.llm_model == "gpt-4o-mini"
        assert p._store is None

    def test_custom_attributes_are_stored(self, tmp_path):
        p = RAGPipeline(
            docs_dir=tmp_path / "d",
            vector_store_path=tmp_path / "s",
            chunk_size=256,
            chunk_overlap=32,
            retrieval_k=6,
            llm_model="gpt-4o",
        )
        assert p.chunk_size == 256
        assert p.retrieval_k == 6
        assert p.llm_model == "gpt-4o"


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


class TestIngest:
    def test_ingest_returns_chunks(self, pipeline):
        mock_store = MagicMock()
        mock_embeddings = MagicMock()

        with (
            patch("src.rag.get_embeddings_model", return_value=mock_embeddings),
            patch("src.rag.build_vector_store", return_value=mock_store) as mock_build,
            patch("src.rag.save_vector_store") as mock_save,
        ):
            chunks = pipeline.ingest()

        assert isinstance(chunks, list)
        mock_build.assert_called_once()
        mock_save.assert_called_once_with(mock_store, pipeline.vector_store_path)

    def test_ingest_sets_internal_store(self, pipeline):
        mock_store = MagicMock()

        with (
            patch("src.rag.get_embeddings_model"),
            patch("src.rag.build_vector_store", return_value=mock_store),
            patch("src.rag.save_vector_store"),
        ):
            pipeline.ingest()

        assert pipeline._store is mock_store


# ---------------------------------------------------------------------------
# load_store
# ---------------------------------------------------------------------------


class TestLoadStore:
    def test_load_store_sets_internal_store(self, pipeline):
        mock_store = MagicMock()
        mock_embeddings = MagicMock()

        with (
            patch("src.rag.get_embeddings_model", return_value=mock_embeddings),
            patch("src.rag.load_vector_store", return_value=mock_store) as mock_load,
        ):
            pipeline.load_store()

        mock_load.assert_called_once_with(pipeline.vector_store_path, mock_embeddings)
        assert pipeline._store is mock_store


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_returns_answer_string(self, pipeline):
        pipeline._store = MagicMock()  # pre-populate store

        with (
            patch("src.rag.retrieve", return_value=[_make_doc("ctx")]) as mock_ret,
            patch("src.rag.generate_answer", return_value="The answer.") as mock_gen,
        ):
            result = pipeline.query("What is ACME?")

        mock_ret.assert_called_once_with(pipeline._store, "What is ACME?", k=pipeline.retrieval_k)
        mock_gen.assert_called_once()
        assert result == "The answer."

    def test_query_auto_loads_store_when_none(self, pipeline, tmp_path):
        mock_store = MagicMock()

        with (
            patch("src.rag.get_embeddings_model"),
            patch("src.rag.load_vector_store", return_value=mock_store),
            patch("src.rag.retrieve", return_value=[]),
            patch("src.rag.generate_answer", return_value="auto loaded"),
        ):
            result = pipeline.query("question")

        assert pipeline._store is mock_store
        assert result == "auto loaded"

    def test_query_uses_llm_model_from_config(self, pipeline):
        pipeline._store = MagicMock()
        pipeline.llm_model = "gpt-4o"

        with (
            patch("src.rag.retrieve", return_value=[_make_doc("ctx")]),
            patch("src.rag.generate_answer", return_value="ok") as mock_gen,
        ):
            pipeline.query("q")

        mock_gen.assert_called_once_with("q", [_make_doc("ctx")], model="gpt-4o")
