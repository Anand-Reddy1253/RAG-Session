"""Unit tests for src.loader — DocumentLoader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.loader import DocumentLoader


# ===========================================================================
# Helpers
# ===========================================================================


def _make_doc(text: str = "sample text", source: str = "file.pdf") -> Document:
    return Document(page_content=text, metadata={"source": source})


# ===========================================================================
# Tests
# ===========================================================================


class TestLoad:
    def test_load_raises_if_dir_missing(self):
        loader = DocumentLoader(docs_dir="/nonexistent/path/that/does/not/exist")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_raises_if_no_supported_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("plain text")
        loader = DocumentLoader(docs_dir=tmp_path)
        with pytest.raises(ValueError, match="No supported files"):
            loader.load()

    def test_load_pdf_dispatches_pypdf_loader(self, mocker, tmp_path):
        pdf_file = tmp_path / "fake.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")
        mock_loader_cls = mocker.patch("src.loader.PyPDFLoader")
        mock_instance = MagicMock()
        mock_instance.load.return_value = [_make_doc(source=str(pdf_file))]
        mock_loader_cls.return_value = mock_instance

        loader = DocumentLoader(docs_dir=tmp_path)
        loader.load()

        mock_loader_cls.assert_called_once_with(str(pdf_file))

    def test_load_docx_dispatches_docx2txt_loader(self, mocker, tmp_path):
        docx_file = tmp_path / "fake.docx"
        docx_file.write_bytes(b"PK\x03\x04fake")
        mock_loader_cls = mocker.patch("src.loader.Docx2txtLoader")
        mock_instance = MagicMock()
        mock_instance.load.return_value = [_make_doc(source=str(docx_file))]
        mock_loader_cls.return_value = mock_instance

        loader = DocumentLoader(docs_dir=tmp_path)
        loader.load()

        mock_loader_cls.assert_called_once_with(str(docx_file))

    def test_load_csv_dispatches_csv_loader(self, mocker, tmp_path):
        csv_file = tmp_path / "fake.csv"
        csv_file.write_text("name,value\nfoo,bar\n")
        mock_loader_cls = mocker.patch("src.loader.CSVLoader")
        mock_instance = MagicMock()
        mock_instance.load.return_value = [_make_doc(source=str(csv_file))]
        mock_loader_cls.return_value = mock_instance

        loader = DocumentLoader(docs_dir=tmp_path)
        loader.load()

        mock_loader_cls.assert_called_once_with(str(csv_file))

    def test_load_json_dispatches_json_loader(self, mocker, tmp_path):
        json_file = tmp_path / "fake.json"
        json_file.write_text('{"key": "value"}')
        mock_loader_cls = mocker.patch("src.loader.JSONLoader")
        mock_instance = MagicMock()
        mock_instance.load.return_value = [_make_doc(source=str(json_file))]
        mock_loader_cls.return_value = mock_instance

        loader = DocumentLoader(docs_dir=tmp_path)
        loader.load()

        mock_loader_cls.assert_called_once()
        call_kwargs = mock_loader_cls.call_args.kwargs
        assert call_kwargs.get("jq_schema") == "."

    def test_load_returns_document_list(self, mocker, tmp_path):
        for name in ("a.pdf", "b.docx", "c.csv", "d.json"):
            (tmp_path / name).write_text("content")

        doc = _make_doc()
        for patch_target in (
            "src.loader.PyPDFLoader",
            "src.loader.Docx2txtLoader",
            "src.loader.CSVLoader",
            "src.loader.JSONLoader",
        ):
            mock_cls = mocker.patch(patch_target)
            mock_instance = MagicMock()
            mock_instance.load.return_value = [doc]
            mock_cls.return_value = mock_instance

        loader = DocumentLoader(docs_dir=tmp_path)
        result = loader.load()

        assert isinstance(result, list)
        assert all(isinstance(d, Document) for d in result)
        assert len(result) == 4

    def test_load_aggregates_multi_page_pdf(self, mocker, tmp_path):
        pdf_file = tmp_path / "multi.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")
        mock_loader_cls = mocker.patch("src.loader.PyPDFLoader")
        mock_instance = MagicMock()
        pages = [_make_doc(f"Page {i}") for i in range(3)]
        mock_instance.load.return_value = pages
        mock_loader_cls.return_value = mock_instance

        loader = DocumentLoader(docs_dir=tmp_path)
        result = loader.load()

        assert len(result) == 3


class TestSplit:
    def test_split_produces_smaller_chunks(self, tmp_path):
        long_doc = Document(
            page_content="x " * 3000, metadata={"source": "big.pdf"}
        )
        loader = DocumentLoader(docs_dir=tmp_path, chunk_size=500, chunk_overlap=50)
        chunks = loader.split([long_doc])
        assert len(chunks) >= 5

    def test_split_chunk_index_metadata(self, tmp_path):
        long_doc = Document(
            page_content="word " * 1000, metadata={"source": "big.pdf"}
        )
        loader = DocumentLoader(docs_dir=tmp_path, chunk_size=200, chunk_overlap=20)
        chunks = loader.split([long_doc])
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata

    def test_split_empty_input_returns_empty(self, tmp_path):
        loader = DocumentLoader(docs_dir=tmp_path)
        result = loader.split([])
        assert result == []

    def test_source_metadata_preserved_after_split(self, mocker, tmp_path):
        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF fake")
        long_text = "sentence. " * 500
        doc = Document(page_content=long_text, metadata={"source": str(pdf_file)})

        mock_loader_cls = mocker.patch("src.loader.PyPDFLoader")
        mock_instance = MagicMock()
        mock_instance.load.return_value = [doc]
        mock_loader_cls.return_value = mock_instance

        loader = DocumentLoader(docs_dir=tmp_path, chunk_size=200)
        chunks = loader.load_and_split()

        for chunk in chunks:
            assert "source" in chunk.metadata


class TestLoadAndSplit:
    def test_load_and_split_chains_both_steps(self, mocker, tmp_path):
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF fake")
        doc = Document(page_content="word " * 500, metadata={"source": str(pdf_file)})

        mock_loader_cls = mocker.patch("src.loader.PyPDFLoader")
        mock_instance = MagicMock()
        mock_instance.load.return_value = [doc]
        mock_loader_cls.return_value = mock_instance

        loader = DocumentLoader(docs_dir=tmp_path, chunk_size=200)
        load_spy = mocker.spy(loader, "load")
        split_spy = mocker.spy(loader, "split")

        loader.load_and_split()

        load_spy.assert_called_once()
        split_spy.assert_called_once()
