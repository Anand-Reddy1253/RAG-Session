"""Document loading and splitting for the RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
    PyPDFLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".csv", ".json"}


class DocumentLoader:
    """Loads and splits documents from a directory into LangChain ``Document`` chunks."""

    def __init__(
        self,
        docs_dir: str | Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        json_content_key: str = ".",
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.json_content_key = json_content_key
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> List[Document]:
        """Discover and load all supported documents from ``docs_dir``."""
        if not self.docs_dir.exists():
            raise FileNotFoundError(
                f"Documents directory not found: {self.docs_dir}"
            )

        files = self._discover_files()
        if not files:
            raise ValueError(
                f"No supported files ({', '.join(_SUPPORTED_EXTENSIONS)}) "
                f"found in {self.docs_dir}"
            )

        documents: List[Document] = []
        for file_path in files:
            documents.extend(self._load_single_file(file_path))
        return documents

    def split(self, documents: List[Document]) -> List[Document]:
        """Split *documents* into chunks and attach ``chunk_index`` metadata."""
        if not documents:
            return []

        chunks = self._splitter.split_documents(documents)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
        return chunks

    def load_and_split(self) -> List[Document]:
        """Convenience wrapper — ``load()`` then ``split()``."""
        documents = self.load()
        return self.split(documents)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_files(self) -> List[Path]:
        """Return all files with a supported extension inside ``docs_dir``."""
        return [
            p
            for p in sorted(self.docs_dir.iterdir())
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
        ]

    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Dispatch to the correct LangChain loader based on file extension."""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(file_path))
        elif suffix == ".csv":
            loader = CSVLoader(str(file_path))
        elif suffix == ".json":
            loader = JSONLoader(
                file_path=str(file_path),
                jq_schema=self.json_content_key,
                text_content=False,
            )
        else:
            return []

        docs = loader.load()
        # Ensure every document carries a ``source`` metadata key.
        for doc in docs:
            doc.metadata.setdefault("source", str(file_path.resolve()))
        return docs
