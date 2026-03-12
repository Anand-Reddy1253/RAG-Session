"""Document loading utilities for the RAG pipeline.

Supports PDF, Word (.docx), CSV, and JSON files from the docs directory.
"""

import csv
import json
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _load_csv(path: str) -> list[Document]:
    """Load a CSV file and convert each row to a Document."""
    documents = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            content = "\n".join(f"{k}: {v}" for k, v in row.items())
            documents.append(
                Document(page_content=content, metadata={"source": path, "row": i})
            )
    return documents


def _load_json(path: str) -> list[Document]:
    """Load a JSON file and convert it to a Document."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    content = json.dumps(data, indent=2)
    return [Document(page_content=content, metadata={"source": path})]


def load_documents(docs_dir: str) -> list[Document]:
    """Load all supported documents from *docs_dir*.

    Supported formats: ``.pdf``, ``.docx``, ``.csv``, ``.json``.

    Args:
        docs_dir: Path to the directory containing documents.

    Returns:
        A flat list of :class:`~langchain_core.documents.Document` objects.
    """
    documents: list[Document] = []
    docs_path = Path(docs_dir)

    for entry in sorted(docs_path.iterdir()):
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        path = str(entry)

        if suffix == ".pdf":
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        elif suffix == ".docx":
            loader = Docx2txtLoader(path)
            documents.extend(loader.load())
        elif suffix == ".csv":
            documents.extend(_load_csv(path))
        elif suffix == ".json":
            documents.extend(_load_json(path))

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into smaller chunks for embedding.

    Args:
        documents: Documents to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of chunked :class:`~langchain_core.documents.Document` objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)
