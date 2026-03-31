"""Tests for rag.loader — document loading and splitting utilities."""

import csv
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.loader import _load_csv, _load_json, load_documents, split_documents


class TestLoadCsv:
    def test_returns_one_document_per_row(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,value\nAlice,1\nBob,2\n")

        docs = _load_csv(str(csv_file))

        assert len(docs) == 2
        assert "name: Alice" in docs[0].page_content
        assert "value: 1" in docs[0].page_content
        assert docs[0].metadata["row"] == 0
        assert docs[1].metadata["row"] == 1

    def test_metadata_contains_source(self, tmp_path):
        csv_file = tmp_path / "sample.csv"
        csv_file.write_text("col\nval\n")

        docs = _load_csv(str(csv_file))

        assert docs[0].metadata["source"] == str(csv_file)


class TestLoadJson:
    def test_returns_single_document(self, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps({"key": "value"}))

        docs = _load_json(str(json_file))

        assert len(docs) == 1
        assert "key" in docs[0].page_content

    def test_metadata_contains_source(self, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text("{}")

        docs = _load_json(str(json_file))

        assert docs[0].metadata["source"] == str(json_file)


class TestLoadDocuments:
    def test_loads_csv_and_json(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        json_file = tmp_path / "data.json"
        json_file.write_text('{"x": 1}')

        with (
            patch("rag.loader.PyPDFLoader") as mock_pdf,
            patch("rag.loader.Docx2txtLoader") as mock_docx,
        ):
            docs = load_documents(str(tmp_path))

        assert len(docs) == 2  # 1 CSV row + 1 JSON doc

    def test_skips_unsupported_extensions(self, tmp_path):
        (tmp_path / "readme.txt").write_text("ignored")

        docs = load_documents(str(tmp_path))

        assert docs == []

    def test_loads_pdf_via_loader(self, tmp_path):
        pdf_file = tmp_path / "file.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_doc = Document(page_content="PDF content", metadata={"source": str(pdf_file)})
        with patch("rag.loader.PyPDFLoader") as mock_pdf_cls:
            mock_pdf_cls.return_value.load.return_value = [mock_doc]
            docs = load_documents(str(tmp_path))

        assert len(docs) == 1
        assert docs[0].page_content == "PDF content"

    def test_loads_docx_via_loader(self, tmp_path):
        docx_file = tmp_path / "file.docx"
        docx_file.write_bytes(b"PK fake docx content")

        mock_doc = Document(page_content="DOCX content", metadata={"source": str(docx_file)})
        with patch("rag.loader.Docx2txtLoader") as mock_docx_cls:
            mock_docx_cls.return_value.load.return_value = [mock_doc]
            docs = load_documents(str(tmp_path))

        assert len(docs) == 1
        assert docs[0].page_content == "DOCX content"


class TestSplitDocuments:
    def test_splits_large_document(self):
        long_text = "word " * 500  # ~2500 chars
        docs = [Document(page_content=long_text, metadata={"source": "test"})]

        chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)

        assert len(chunks) > 1

    def test_preserves_metadata(self):
        docs = [Document(page_content="short text", metadata={"source": "test.pdf"})]

        chunks = split_documents(docs, chunk_size=500, chunk_overlap=0)

        assert all(c.metadata["source"] == "test.pdf" for c in chunks)

    def test_returns_empty_list_for_empty_input(self):
        assert split_documents([]) == []
