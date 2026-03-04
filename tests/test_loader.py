"""Tests for src.loader — document loading utilities."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.loader import load_csv, load_documents, load_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------


class TestLoadCsv:
    def test_returns_one_document_per_data_row(self, tmp_path):
        csv_path = _write(tmp_path, "data.csv", "name,score\nAlice,90\nBob,85\n")
        docs = load_csv(csv_path)
        assert len(docs) == 2

    def test_document_content_contains_column_values(self, tmp_path):
        csv_path = _write(tmp_path, "data.csv", "region,revenue\nWest,1000\n")
        docs = load_csv(csv_path)
        assert "West" in docs[0].page_content
        assert "1000" in docs[0].page_content

    def test_document_metadata_contains_source(self, tmp_path):
        csv_path = _write(tmp_path, "sales.csv", "a,b\n1,2\n")
        docs = load_csv(csv_path)
        assert docs[0].metadata["source"] == str(csv_path)

    def test_empty_csv_returns_no_documents(self, tmp_path):
        csv_path = _write(tmp_path, "empty.csv", "col1,col2\n")
        docs = load_csv(csv_path)
        assert docs == []


# ---------------------------------------------------------------------------
# load_json
# ---------------------------------------------------------------------------


class TestLoadJson:
    def test_returns_single_document(self, tmp_path):
        data = {"key": "value", "number": 42}
        json_path = _write(tmp_path, "data.json", json.dumps(data))
        docs = load_json(json_path)
        assert len(docs) == 1

    def test_document_content_is_json_string(self, tmp_path):
        data = {"company": "Globex", "score": 62}
        json_path = _write(tmp_path, "crm.json", json.dumps(data))
        docs = load_json(json_path)
        assert "Globex" in docs[0].page_content
        assert "62" in docs[0].page_content

    def test_document_metadata_contains_source(self, tmp_path):
        json_path = _write(tmp_path, "file.json", '{"x": 1}')
        docs = load_json(json_path)
        assert docs[0].metadata["source"] == str(json_path)

    def test_nested_json_is_preserved(self, tmp_path):
        data = {"outer": {"inner": "deep"}}
        json_path = _write(tmp_path, "nested.json", json.dumps(data))
        docs = load_json(json_path)
        assert "deep" in docs[0].page_content


# ---------------------------------------------------------------------------
# load_documents
# ---------------------------------------------------------------------------


class TestLoadDocuments:
    def test_raises_for_nonexistent_directory(self, tmp_path):
        with pytest.raises(ValueError, match="not a directory"):
            load_documents(tmp_path / "no_such_dir")

    def test_skips_unsupported_file_types(self, tmp_path):
        _write(tmp_path, "readme.txt", "plain text file")
        _write(tmp_path, "script.py", "print('hello')")
        docs = load_documents(tmp_path)
        assert docs == []

    def test_loads_csv_and_json_together(self, tmp_path):
        _write(tmp_path, "data.csv", "col\nvalue\n")
        _write(tmp_path, "data.json", '{"key": "val"}')
        docs = load_documents(tmp_path)
        assert len(docs) == 2

    def test_empty_directory_returns_empty_list(self, tmp_path):
        docs = load_documents(tmp_path)
        assert docs == []
