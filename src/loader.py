"""Document loading utilities for the RAG pipeline.

Supports PDF (.pdf), Word (.docx), CSV (.csv), and JSON (.json) files.
Each file is loaded as a list of LangChain ``Document`` objects.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.documents import Document


def load_pdf(path: str | Path) -> List[Document]:
    """Load a PDF file and return one Document per page."""
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(path))
    return loader.load()


def load_docx(path: str | Path) -> List[Document]:
    """Load a Word (.docx) file and return a single Document."""
    from langchain_community.document_loaders import Docx2txtLoader

    loader = Docx2txtLoader(str(path))
    return loader.load()


def load_csv(path: str | Path) -> List[Document]:
    """Load a CSV file and return one Document per row."""
    df = pd.read_csv(path)
    docs: List[Document] = []
    for _, row in df.iterrows():
        content = ", ".join(f"{col}: {val}" for col, val in row.items())
        docs.append(Document(page_content=content, metadata={"source": str(path)}))
    return docs


def load_json(path: str | Path) -> List[Document]:
    """Load a JSON file and return a single Document."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    content = json.dumps(data, indent=2)
    return [Document(page_content=content, metadata={"source": str(path)})]


def load_documents(docs_dir: str | Path) -> List[Document]:
    """Load all supported documents from *docs_dir*.

    Supported extensions: .pdf, .docx, .csv, .json
    Files with other extensions are silently skipped.
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.is_dir():
        raise ValueError(f"docs_dir is not a directory: {docs_dir}")

    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".csv": load_csv,
        ".json": load_json,
    }

    all_docs: List[Document] = []
    for file_path in sorted(docs_dir.iterdir()):
        suffix = file_path.suffix.lower()
        if suffix in loaders:
            all_docs.extend(loaders[suffix](file_path))
    return all_docs
