"""Shared pytest fixtures for all test modules."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document


@pytest.fixture()
def sample_document() -> Document:
    """A single document with standard metadata."""
    return Document(
        page_content="Payment is due within 30 days of invoice receipt.",
        metadata={"source": "ACME_Globex_Contract.docx"},
    )


@pytest.fixture()
def sample_chunks() -> list[Document]:
    """Three document chunks representing different source files."""
    return [
        Document(
            page_content="Payment is due within 30 days of invoice receipt.",
            metadata={"source": "ACME_Globex_Contract.docx", "chunk_index": 0},
        ),
        Document(
            page_content="Full-time employees receive 15 days of paid vacation per year.",
            metadata={"source": "ACME_HR_Handbook_v4.2.pdf", "chunk_index": 1},
        ),
        Document(
            page_content="The Western region led Q3 with $2.4 M in revenue.",
            metadata={"source": "acme_sales_q3.csv", "chunk_index": 2},
        ),
    ]
