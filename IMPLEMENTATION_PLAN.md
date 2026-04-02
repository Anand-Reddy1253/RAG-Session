# RAG Pipeline — Implementation Plan

> **Repository:** RAG-Session  
> **Target Issue:** Unit test generation for LLM  
> **Python Version:** 3.9+  
> **Last Updated:** 2025  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Repository Layout After Implementation](#2-repository-layout-after-implementation)
3. [Dependency Specification — `requirements.txt`](#3-dependency-specification--requirementstxt)
4. [Module-by-Module Interface Specification](#4-module-by-module-interface-specification)
   - 4.1 [loader.py](#41-loaderpy)
   - 4.2 [embedder.py](#42-embedderpy)
   - 4.3 [llm.py](#43-llmpy)
   - 4.4 [rag.py](#44-ragpy)
   - 4.5 [main.py](#45-mainpy)
5. [LangChain & OpenAI API Mapping](#5-langchain--openai-api-mapping)
6. [Test Infrastructure](#6-test-infrastructure)
   - 6.1 [pytest.ini](#61-pytestini)
   - 6.2 [tests/__init__.py](#62-tests__init__py)
   - 6.3 [tests/conftest.py](#63-testsconftestpy)
7. [Unit Test Specifications](#7-unit-test-specifications)
   - 7.1 [tests/test_llm.py — Primary Focus](#71-teststest_llmpy--primary-focus)
   - 7.2 [tests/test_loader.py](#72-teststest_loaderpy)
   - 7.3 [tests/test_embedder.py](#73-teststest_embedderpy)
   - 7.4 [tests/test_rag.py](#74-teststest_ragpy)
8. [Mocking Strategy — External API Calls](#8-mocking-strategy--external-api-calls)
9. [Data Flow & Component Interaction Diagram](#9-data-flow--component-interaction-diagram)
10. [Implementation Task Breakdown](#10-implementation-task-breakdown)
11. [Acceptance Criteria](#11-acceptance-criteria)
12. [Risk Register](#12-risk-register)

---

## 1. Executive Summary

This plan describes the complete implementation of a **Retrieval-Augmented Generation (RAG)** pipeline for the RAG-Session repository. The system ingests the four documents already present in `docs/` (`.docx`, `.pdf`, `.csv`, `.json`), converts them into searchable vector embeddings stored in FAISS, and answers natural-language questions by grounding GPT-4 responses in retrieved context.

The primary deliverable called out in the issue is **unit tests for the LLM layer** (`tests/test_llm.py`), but robust testing requires the full test pyramid: loader, embedder, LLM, and end-to-end RAG tests — all with **zero real API calls** during CI.

### Guiding Principles

| Principle | Decision |
|---|---|
| No real API calls in tests | All OpenAI and FAISS I/O is mocked via `unittest.mock` |
| Deterministic tests | Fixed seeds, controlled mock return values, no randomness |
| Single responsibility | Each module owns exactly one pipeline stage |
| LangChain v0.2 idioms | Use `langchain-openai`, `langchain-community`, `langchain-core` split packages |
| Typed interfaces | All public functions carry full type annotations |

---

## 2. Repository Layout After Implementation

```
RAG-Session/
│
├── docs/                              # Existing knowledge-base documents (unchanged)
│   ├── ACME_Globex_Contract.docx
│   ├── ACME_HR_Handbook_v4.2.pdf
│   ├── acme_sales_q3.csv
│   ├── globex_crm_profile.json
│   └── filelist.txt
│
├── src/                               # NEW — all production source code lives here
│   ├── __init__.py
│   ├── loader.py                      # Stage 1 & 2: Load + Split
│   ├── embedder.py                    # Stage 3 & 4: Embed + Store
│   ├── llm.py                         # Stage 6: Generate (prompt + LLM call)
│   └── rag.py                         # Orchestrator: wires all stages together
│
├── tests/                             # NEW — test suite
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures
│   ├── test_loader.py
│   ├── test_embedder.py
│   ├── test_llm.py                    # PRIMARY FOCUS
│   └── test_rag.py
│
├── vector_store/                      # Runtime artefact (git-ignored)
│
├── main.py                            # NEW — CLI entry point
├── requirements.txt                   # NEW — pinned dependencies
├── pytest.ini                         # NEW — test runner configuration
├── .env.example                       # NEW — environment variable template
├── .gitignore                         # NEW — ignore venv, vector_store, .env
├── ReadMe.md                          # Existing (unchanged)
└── IMPLEMENTATION_PLAN.md             # This file
```

> **Why `src/` layout?**  
> Placing production modules under `src/` prevents test collection from accidentally importing the wrong package and keeps the root clean. Tests import as `from src.loader import DocumentLoader`.

---

## 3. Dependency Specification — `requirements.txt`

Below is the complete, annotated dependency list. Every package is pinned to a minimum version that guarantees API stability.

```
# ── LangChain ecosystem (v0.2 split-package architecture) ─────────────────────
langchain>=0.2.0            # Core orchestration primitives (chains, runnable)
langchain-core>=0.2.0       # Base classes: BaseMessage, PromptTemplate, Runnable
langchain-openai>=0.1.8     # ChatOpenAI, OpenAIEmbeddings wrappers
langchain-community>=0.2.0  # Document loaders, FAISS vector store integration
langchain-text-splitters>=0.2.0  # RecursiveCharacterTextSplitter

# ── OpenAI SDK ────────────────────────────────────────────────────────────────
openai>=1.14.0              # Underlying HTTP client used by langchain-openai

# ── Vector Store ──────────────────────────────────────────────────────────────
faiss-cpu>=1.7.4            # Facebook AI Similarity Search (CPU build)
                             # Swap for faiss-gpu on CUDA machines

# ── Document parsers (used by LangChain loaders internally) ───────────────────
pypdf>=3.17.0               # PDF text extraction for PyPDFLoader
docx2txt>=0.8               # .docx plain-text extraction for Docx2txtLoader
python-docx>=1.1.0          # Alternative/fallback DOCX handling
jq>=1.6.0                   # JSON path filtering for JSONLoader

# ── Numerical / ML utilities ──────────────────────────────────────────────────
numpy>=1.24.0               # Required by FAISS Python bindings
tiktoken>=0.6.0             # Token counting for chunk-size calculations

# ── Environment & configuration ───────────────────────────────────────────────
python-dotenv>=1.0.0        # Loads .env into os.environ at startup

# ── Testing ───────────────────────────────────────────────────────────────────
pytest>=7.4.0               # Test runner
pytest-mock>=3.11.0         # `mocker` fixture — thin wrapper over unittest.mock
pytest-cov>=4.1.0           # Coverage reporting (--cov flag)
coverage>=7.3.0             # Underlying coverage engine

# ── Development / linting (not required at runtime) ──────────────────────────
# black>=24.0.0
# ruff>=0.3.0
# mypy>=1.8.0
```

### Dependency Rationale

| Package | Why needed | Alternative |
|---|---|---|
| `langchain-openai` | Official LangChain wrapper for `ChatOpenAI` and `OpenAIEmbeddings` | `openai` SDK directly (more boilerplate) |
| `langchain-community` | Provides `PyPDFLoader`, `CSVLoader`, `JSONLoader`, `Docx2txtLoader`, `FAISS` | Individual loaders from source |
| `faiss-cpu` | In-process vector similarity search with no server dependency | ChromaDB, Pinecone (require extra infra) |
| `jq` | `JSONLoader` uses jq expressions to extract text fields from JSON | Custom JSON parsing in loader |
| `tiktoken` | Token-aware chunk sizing prevents embedding API overflows | Character-based sizing (less precise) |
| `pytest-mock` | `mocker.patch` is cleaner than raw `unittest.mock.patch` in fixtures | `unittest.mock` directly |

---

## 4. Module-by-Module Interface Specification

### 4.1 `loader.py`

**Responsibility:** Discover files in `docs/`, dispatch to the correct LangChain loader by file extension, and optionally split the resulting `Document` objects into chunks.

#### Class: `DocumentLoader`

```python
# src/loader.py

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    JSONLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Supported extensions mapped to their LangChain loader class
LOADER_MAP: Dict[str, type] = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".csv":  CSVLoader,
    ".json": JSONLoader,
}


class DocumentLoader:
    """
    Loads and optionally splits documents from a directory.

    Args:
        docs_dir (str | Path): Path to the directory containing source documents.
        chunk_size (int): Target character count per chunk. Default: 1000.
        chunk_overlap (int): Overlap between consecutive chunks. Default: 200.
        json_content_key (str): jq expression to extract text from JSON files.
                                Default: ".[]" (each top-level item).
    """

    def __init__(
        self,
        docs_dir: str | Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        json_content_key: str = ".",
    ) -> None: ...

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> List[Document]:
        """
        Discover all supported files in docs_dir and load them.

        Returns:
            Flat list of Document objects (one or more per file depending
            on the loader; e.g. PyPDFLoader returns one Document per page).

        Raises:
            FileNotFoundError: If docs_dir does not exist.
            ValueError: If docs_dir contains zero supported files.
        """
        ...

    def load_and_split(self) -> List[Document]:
        """
        Load all documents then apply RecursiveCharacterTextSplitter.

        Returns:
            List of chunked Document objects, each with metadata keys:
            - source (str): original file path
            - chunk_index (int): zero-based chunk position within the source file
        """
        ...

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split an already-loaded list of Documents into chunks.

        Args:
            documents: Output from load().

        Returns:
            Chunked Document list.
        """
        ...

    # ── Private helpers ───────────────────────────────────────────────────────

    def _discover_files(self) -> List[Path]:
        """Return all files under docs_dir with a supported extension."""
        ...

    def _load_single_file(self, file_path: Path) -> List[Document]:
        """
        Instantiate the correct loader for the given file and call .load().

        Special handling:
        - JSONLoader requires `jq_schema` and `text_content=False` kwargs.
        - CSVLoader automatically uses the first row as column headers.
        """
        ...
```

#### Key Design Decisions for `loader.py`

| Decision | Rationale |
|---|---|
| `json_content_key` defaults to `"."` | The `globex_crm_profile.json` file is a dict; `"."` serialises the whole object as the document text, which is safe for arbitrary JSON shapes |
| `chunk_index` metadata field | Enables tests and downstream consumers to verify chunk ordering |
| `load()` separate from `load_and_split()` | Allows `test_loader.py` to test loading and splitting independently |
| `_load_single_file` is private but testable | Tests call it directly via `loader._load_single_file(path)` for isolated file-type tests |

---

### 4.2 `embedder.py`

**Responsibility:** Convert `Document` chunks into vector embeddings using `text-embedding-ada-002` and persist/retrieve a FAISS vector store.

#### Class: `VectorStoreManager`

```python
# src/embedder.py

from __future__ import annotations
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class VectorStoreManager:
    """
    Manages creation, persistence, and retrieval from a FAISS vector store.

    Args:
        openai_api_key (str): API key forwarded to OpenAIEmbeddings.
                              Defaults to os.environ["OPENAI_API_KEY"].
        embedding_model (str): OpenAI embedding model name.
                               Default: "text-embedding-ada-002".
        vector_store_path (str | Path): Directory where the FAISS index is
                                        saved and loaded. Default: "./vector_store".
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
        vector_store_path: str | Path = "./vector_store",
    ) -> None: ...

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, chunks: List[Document]) -> FAISS:
        """
        Embed all chunks and build a new FAISS index in memory.

        Internally calls FAISS.from_documents(chunks, self._embeddings).

        Args:
            chunks: Chunked Document list from DocumentLoader.load_and_split().

        Returns:
            In-memory FAISS vector store.

        Raises:
            ValueError: If chunks is empty.
            openai.AuthenticationError: Propagated if API key is invalid.
        """
        ...

    def save(self, vector_store: FAISS) -> None:
        """
        Persist the in-memory FAISS index to vector_store_path.

        Calls vector_store.save_local(str(self._vector_store_path)).

        Args:
            vector_store: The FAISS instance returned by build().
        """
        ...

    def load(self) -> FAISS:
        """
        Load a previously saved FAISS index from disk.

        Calls FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True).

        Returns:
            Loaded FAISS vector store.

        Raises:
            FileNotFoundError: If vector_store_path does not contain a valid index.
        """
        ...

    def build_and_save(self, chunks: List[Document]) -> FAISS:
        """
        Convenience method: build() then save() in a single call.

        Returns:
            The same FAISS instance that was saved.
        """
        ...

    def similarity_search(
        self,
        query: str,
        vector_store: FAISS,
        top_k: int = 4,
    ) -> List[Document]:
        """
        Return the top_k most semantically similar chunks for a query string.

        Calls vector_store.similarity_search(query, k=top_k).

        Args:
            query: The user's natural-language question.
            vector_store: Loaded or freshly built FAISS instance.
            top_k: Number of chunks to retrieve. Default: 4.

        Returns:
            List of Document objects ordered by similarity (most similar first).
        """
        ...

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Expose the underlying OpenAIEmbeddings instance (needed for mocking)."""
        ...
```

#### Key Design Decisions for `embedder.py`

| Decision | Rationale |
|---|---|
| `allow_dangerous_deserialization=True` on `load_local` | Required by LangChain ≥ 0.2 for pickle-based FAISS index files |
| `similarity_search` lives on `VectorStoreManager` | Centralises retrieval logic; tests can mock `vector_store.similarity_search` independently |
| `build_and_save` convenience method | Used by `rag.py` for the indexing path; keeps the orchestrator clean |

---

### 4.3 `llm.py`

**Responsibility:** Build RAG prompts and invoke ChatGPT to produce grounded answers. This is the **primary focus** of the test suite.

#### Dataclass: `LLMResponse`

```python
# src/llm.py  (partial)

from dataclasses import dataclass, field
from typing import List
from langchain_core.documents import Document


@dataclass
class LLMResponse:
    """
    Structured response returned by LLMClient.generate().

    Attributes:
        answer (str): The LLM-generated answer text.
        sources (List[Document]): The context chunks that were injected.
        model (str): The model name used (e.g. "gpt-4").
        prompt_tokens (int): Number of tokens in the prompt (from usage metadata).
        completion_tokens (int): Number of tokens in the completion.
        total_tokens (int): prompt_tokens + completion_tokens.
    """
    answer: str
    sources: List[Document] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
```

#### Class: `LLMClient`

```python
# src/llm.py  (continued)

from __future__ import annotations
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


# ── Prompt template ──────────────────────────────────────────────────────────
#
# Using a plain string template (not f-string) so it can be stored, versioned,
# and inspected in tests.

RAG_PROMPT_TEMPLATE = """\
You are a helpful assistant that answers questions using only the provided context.
If the answer cannot be found in the context, reply with:
"I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Answer:"""


class LLMClient:
    """
    Wraps ChatOpenAI with RAG-specific prompt construction and response parsing.

    Args:
        openai_api_key (str): OpenAI API key.
                              Defaults to os.environ["OPENAI_API_KEY"].
        model_name (str): Chat model to use. Default: "gpt-3.5-turbo".
        temperature (float): Sampling temperature. Default: 0.0 for determinism.
        max_tokens (int): Maximum tokens in the completion. Default: 512.
        prompt_template (str): Override the default RAG_PROMPT_TEMPLATE.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 512,
        prompt_template: str = RAG_PROMPT_TEMPLATE,
    ) -> None: ...

    # ── Public API ────────────────────────────────────────────────────────────

    def format_context(self, chunks: List[Document]) -> str:
        """
        Concatenate retrieved chunks into a single context string.

        Format:
            [Source: <source_filename>]
            <page_content>

            [Source: <source_filename>]
            <page_content>
            ...

        Args:
            chunks: List of Document objects from the retriever.

        Returns:
            Multi-line string ready to be injected into the prompt.

        Raises:
            ValueError: If chunks is empty.
        """
        ...

    def build_prompt(self, question: str, chunks: List[Document]) -> str:
        """
        Render the prompt template with the question and formatted context.

        Args:
            question: The user's natural-language question.
            chunks: Retrieved Document chunks (passed to format_context).

        Returns:
            Fully rendered prompt string.
        """
        ...

    def generate(self, question: str, chunks: List[Document]) -> LLMResponse:
        """
        Send the rendered prompt to the LLM and return a structured response.

        Internally uses a LangChain LCEL chain:
            prompt_template | chat_model | StrOutputParser()

        For token-usage metadata, inspect the AIMessage.usage_metadata
        attribute returned by ChatOpenAI with include_response_headers=True,
        or fall back to a best-effort count via tiktoken.

        Args:
            question: The user's question.
            chunks: Context documents to inject into the prompt.

        Returns:
            LLMResponse dataclass instance.

        Raises:
            ValueError: If question is empty or whitespace only.
            openai.AuthenticationError: Propagated if API key is invalid.
            openai.RateLimitError: Propagated on rate limit; callers should retry.
        """
        ...

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def chat_model(self) -> ChatOpenAI:
        """Expose the underlying ChatOpenAI instance (needed for mocking)."""
        ...

    @property
    def model_name(self) -> str:
        """The model name this client was initialised with."""
        ...

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_chain(self):
        """
        Build and cache the LangChain LCEL chain:
            ChatPromptTemplate | ChatOpenAI | StrOutputParser

        Returns a Runnable that accepts {"context": str, "question": str}.
        """
        ...
```

#### LLM Chain Construction (LCEL)

```
┌──────────────────────────────────┐
│  ChatPromptTemplate              │  ← RAG_PROMPT_TEMPLATE with {context}/{question}
│  .from_template(prompt_template) │
└──────────────┬───────────────────┘
               │  pipe operator  |
               ▼
┌──────────────────────────────────┐
│  ChatOpenAI                      │  ← model_name, temperature, max_tokens
│  (gpt-3.5-turbo / gpt-4)         │
└──────────────┬───────────────────┘
               │  pipe operator  |
               ▼
┌──────────────────────────────────┐
│  StrOutputParser()               │  ← extracts .content from AIMessage
└──────────────────────────────────┘
```

The chain is called with `.invoke({"context": context_str, "question": question})`.

---

### 4.4 `rag.py`

**Responsibility:** Orchestrate all pipeline stages. Acts as the single public entry point for both the indexing phase (build) and the query phase (ask).

#### Dataclass: `RAGResponse`

```python
# src/rag.py  (partial)

from dataclasses import dataclass, field
from typing import List
from langchain_core.documents import Document


@dataclass
class RAGResponse:
    """
    Full response from RAGPipeline.ask().

    Attributes:
        question (str): The original user question.
        answer (str): LLM-generated answer.
        sources (List[Document]): Retrieved chunks used as context.
        model (str): The LLM model name.
        total_tokens (int): Total tokens consumed in the LLM call.
    """
    question: str
    answer: str
    sources: List[Document] = field(default_factory=list)
    model: str = ""
    total_tokens: int = 0
```

#### Class: `RAGPipeline`

```python
# src/rag.py  (continued)

from __future__ import annotations
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS

from src.loader import DocumentLoader
from src.embedder import VectorStoreManager
from src.llm import LLMClient


class RAGPipeline:
    """
    End-to-end RAG orchestrator.

    Usage — Build index then query:
        pipeline = RAGPipeline(docs_dir="./docs", openai_api_key="sk-...")
        pipeline.build_index()
        response = pipeline.ask("What are the payment terms?")
        print(response.answer)

    Usage — Query against a pre-built index:
        pipeline = RAGPipeline(docs_dir="./docs", openai_api_key="sk-...")
        response = pipeline.ask("What are the payment terms?")  # loads index from disk

    Args:
        docs_dir (str | Path): Directory of source documents.
        openai_api_key (str): OpenAI API key.
        vector_store_path (str | Path): Where to save/load the FAISS index.
        embedding_model (str): Embedding model name.
        chat_model (str): Chat completion model name.
        temperature (float): LLM temperature.
        chunk_size (int): Characters per chunk.
        chunk_overlap (int): Overlap between chunks.
        top_k (int): Chunks to retrieve per query.
    """

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
    ) -> None: ...

    # ── Public API ────────────────────────────────────────────────────────────

    def build_index(self) -> None:
        """
        Full ingestion pipeline: Load → Split → Embed → Store.

        Steps:
            1. DocumentLoader.load_and_split()  → chunks
            2. VectorStoreManager.build(chunks)  → vector_store
            3. VectorStoreManager.save(vector_store)

        Raises:
            FileNotFoundError: If docs_dir does not exist.
            ValueError: If no supported documents are found.
        """
        ...

    def ask(self, question: str) -> RAGResponse:
        """
        Query pipeline: Retrieve → Generate.

        If the vector store is not already loaded in memory, loads it from disk.

        Steps:
            1. VectorStoreManager.load()  (if not cached)
            2. VectorStoreManager.similarity_search(question, top_k=self._top_k)
            3. LLMClient.generate(question, chunks)
            4. Return RAGResponse

        Args:
            question: Natural-language question string.

        Returns:
            RAGResponse dataclass instance.

        Raises:
            FileNotFoundError: If the vector store has not been built yet.
            ValueError: If question is empty.
        """
        ...

    @property
    def is_index_loaded(self) -> bool:
        """True if the vector store is currently held in memory."""
        ...

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_index_loaded(self) -> FAISS:
        """Load the vector store from disk if not already in memory."""
        ...
```

---

### 4.5 `main.py`

**Responsibility:** CLI entry point. Reads environment, builds the pipeline, and runs an interactive question loop.

```python
# main.py  (interface sketch — not full implementation)

"""
Usage:
    python main.py                    # interactive REPL
    python main.py --build            # re-index docs then start REPL
    python main.py --question "..."   # single-shot query (non-interactive)
"""

import argparse
import os
from dotenv import load_dotenv
from src.rag import RAGPipeline, RAGResponse


def parse_args() -> argparse.Namespace:
    """
    CLI argument parser.

    Flags:
        --build          Force re-index of docs/ before querying.
        --question TEXT  Run a single query and exit (non-interactive mode).
        --top-k INT      Number of chunks to retrieve (default: 4).
        --model TEXT     Chat model override (default: gpt-3.5-turbo).
    """
    ...


def print_response(response: RAGResponse) -> None:
    """Pretty-print a RAGResponse to stdout with source citations."""
    ...


def interactive_loop(pipeline: RAGPipeline) -> None:
    """Read questions from stdin, print answers until the user types 'exit'."""
    ...


def main() -> None:
    load_dotenv()
    args = parse_args()
    pipeline = RAGPipeline(
        docs_dir=os.getenv("DOCS_DIR", "./docs"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        vector_store_path=os.getenv("VECTOR_STORE_PATH", "./vector_store"),
        chat_model=args.model,
        top_k=args.top_k,
    )
    if args.build:
        pipeline.build_index()
    if args.question:
        print_response(pipeline.ask(args.question))
    else:
        interactive_loop(pipeline)


if __name__ == "__main__":
    main()
```

---

## 5. LangChain & OpenAI API Mapping

This table maps every external API call to the module and method that owns it.

| Stage | Module | LangChain / OpenAI Class | Method Called | Import Path |
|---|---|---|---|---|
| Load PDF | `loader.py` | `PyPDFLoader` | `.load()` | `langchain_community.document_loaders` |
| Load DOCX | `loader.py` | `Docx2txtLoader` | `.load()` | `langchain_community.document_loaders` |
| Load CSV | `loader.py` | `CSVLoader` | `.load()` | `langchain_community.document_loaders` |
| Load JSON | `loader.py` | `JSONLoader` | `.load()` | `langchain_community.document_loaders` |
| Split | `loader.py` | `RecursiveCharacterTextSplitter` | `.split_documents()` | `langchain_text_splitters` |
| Embed | `embedder.py` | `OpenAIEmbeddings` | *(called internally by FAISS)* | `langchain_openai` |
| Store (build) | `embedder.py` | `FAISS` | `.from_documents()` | `langchain_community.vectorstores` |
| Store (save) | `embedder.py` | `FAISS` | `.save_local()` | `langchain_community.vectorstores` |
| Store (load) | `embedder.py` | `FAISS` | `.load_local()` | `langchain_community.vectorstores` |
| Retrieve | `embedder.py` | `FAISS` | `.similarity_search()` | `langchain_community.vectorstores` |
| Prompt build | `llm.py` | `ChatPromptTemplate` | `.from_template()` | `langchain_core.prompts` |
| Generate | `llm.py` | `ChatOpenAI` | `.invoke()` (via LCEL chain) | `langchain_openai` |
| Parse output | `llm.py` | `StrOutputParser` | `.invoke()` (via LCEL chain) | `langchain_core.output_parsers` |

### Constructor Parameters for Key Classes

```python
# OpenAIEmbeddings
OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=api_key,         # str
)

# ChatOpenAI
ChatOpenAI(
    model_name="gpt-3.5-turbo",     # or "gpt-4"
    temperature=0.0,
    max_tokens=512,
    openai_api_key=api_key,
)

# RecursiveCharacterTextSplitter
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)

# JSONLoader
JSONLoader(
    file_path="./docs/globex_crm_profile.json",
    jq_schema=".",                  # serialize entire JSON as text
    text_content=False,             # field value may be a dict, not a string
)

# CSVLoader
CSVLoader(
    file_path="./docs/acme_sales_q3.csv",
    encoding="utf-8",
)
```

---

## 6. Test Infrastructure

### 6.1 `pytest.ini`

```ini
# pytest.ini
[pytest]
testpaths        = tests
python_files     = test_*.py
python_classes   = Test*
python_functions = test_*

# Show short tracebacks, verbose output, and color
addopts =
    -v
    --tb=short
    --strict-markers
    --cov=src
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=80

# Custom markers (prevents PytestUnknownMarkWarning)
markers =
    unit: Pure unit tests with no I/O
    integration: Tests that touch the real filesystem
    slow: Tests that are intentionally slow
```

### 6.2 `tests/__init__.py`

```python
# tests/__init__.py
# Empty — marks tests/ as a Python package so `from src.X import Y` resolves correctly.
```

### 6.3 `tests/conftest.py`

The shared fixture file eliminates duplication across the four test modules.

```python
# tests/conftest.py

"""
Shared pytest fixtures for the RAG test suite.

Fixture hierarchy:
    sample_document       → single Document object
    sample_chunks         → list of 3 Document objects (simulates split output)
    mock_embeddings       → MagicMock replacing OpenAIEmbeddings
    mock_faiss_store      → MagicMock replacing FAISS vector store
    mock_chat_openai      → MagicMock replacing ChatOpenAI
    mock_llm_client       → LLMClient with mock_chat_openai injected
    tmp_docs_dir          → tmp_path with minimal fake .txt file (for loader tests)
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.llm import LLMClient, LLMResponse
from src.embedder import VectorStoreManager


# ── Document fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def sample_document() -> Document:
    return Document(
        page_content="Payment is due within 30 days of invoice date.",
        metadata={"source": "docs/ACME_Globex_Contract.docx", "page": 0},
    )


@pytest.fixture
def sample_chunks() -> list[Document]:
    return [
        Document(
            page_content="Section 1: Payment terms — net 30 days.",
            metadata={"source": "docs/ACME_Globex_Contract.docx", "chunk_index": 0},
        ),
        Document(
            page_content="Section 2: SLA — 99.9% uptime guaranteed.",
            metadata={"source": "docs/ACME_Globex_Contract.docx", "chunk_index": 1},
        ),
        Document(
            page_content="Full-time employees receive 15 vacation days per year.",
            metadata={"source": "docs/ACME_HR_Handbook_v4.2.pdf", "chunk_index": 0},
        ),
    ]


# ── Mock fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_embeddings() -> MagicMock:
    """MagicMock standing in for OpenAIEmbeddings."""
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 1536   # ada-002 dimension
    mock.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
    return mock


@pytest.fixture
def mock_faiss_store(sample_chunks) -> MagicMock:
    """MagicMock standing in for a FAISS vector store instance."""
    mock = MagicMock()
    mock.similarity_search.return_value = sample_chunks[:2]
    return mock


@pytest.fixture
def mock_chat_openai() -> MagicMock:
    """
    MagicMock standing in for ChatOpenAI.
    The .invoke() method returns a string (post-StrOutputParser).
    """
    mock = MagicMock()
    mock.invoke.return_value = "Payment is due within 30 days of invoice date."
    return mock


@pytest.fixture
def llm_client_with_mock(mock_chat_openai) -> LLMClient:
    """
    LLMClient instance whose internal ChatOpenAI is replaced with a mock.
    Achieved by patching at the class-instantiation level.
    """
    with patch("src.llm.ChatOpenAI", return_value=mock_chat_openai):
        client = LLMClient(openai_api_key="sk-test-key")
    return client
```

---

## 7. Unit Test Specifications

The following sections define **every test case** that must be implemented. Each test is described with its inputs, expected behaviour, and the mock targets.

---

### 7.1 `tests/test_llm.py` — Primary Focus

This file is the most important deliverable. It must cover `LLMClient` exhaustively with no real network calls.

#### Mock Target Reference

```
src.llm.ChatOpenAI          ← patch target for the chat model constructor
src.llm.ChatPromptTemplate  ← patch target if testing prompt building in isolation
```

#### Full Test Case Inventory

```python
# tests/test_llm.py  — SPECIFICATION (not runnable skeleton — see implementation notes)

"""
Test module for src.llm.LLMClient and src.llm.LLMResponse.

Covers:
    A. LLMClient initialisation
    B. format_context()
    C. build_prompt()
    D. generate() — happy path
    E. generate() — edge cases and error paths
    F. LLMResponse dataclass
    G. Prompt template content assertions
    H. Chain construction (_build_chain)
"""
```

##### A — Initialisation Tests

| Test ID | Test Name | Description | Mock | Expected |
|---|---|---|---|---|
| A1 | `test_init_default_model` | No `model_name` arg → defaults to `"gpt-3.5-turbo"` | `patch("src.llm.ChatOpenAI")` | `client.model_name == "gpt-3.5-turbo"` |
| A2 | `test_init_custom_model` | `model_name="gpt-4"` is stored correctly | `patch("src.llm.ChatOpenAI")` | `client.model_name == "gpt-4"` |
| A3 | `test_init_temperature_zero` | Default `temperature=0.0` is passed to `ChatOpenAI` | `patch("src.llm.ChatOpenAI")` | `ChatOpenAI` called with `temperature=0.0` |
| A4 | `test_init_custom_temperature` | `temperature=0.7` is forwarded to `ChatOpenAI` | `patch("src.llm.ChatOpenAI")` | `ChatOpenAI` called with `temperature=0.7` |
| A5 | `test_init_api_key_forwarded` | `openai_api_key` is passed to `ChatOpenAI` | `patch("src.llm.ChatOpenAI")` | `ChatOpenAI` called with the expected key value |
| A6 | `test_init_api_key_from_env` | `openai_api_key=None` → reads from `os.environ["OPENAI_API_KEY"]` | `patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env"})` + patch ChatOpenAI | `ChatOpenAI` called with `"sk-env"` |
| A7 | `test_chat_model_property` | `.chat_model` returns the mocked `ChatOpenAI` instance | `patch("src.llm.ChatOpenAI")` | `client.chat_model is mock_instance` |

##### B — `format_context()` Tests

| Test ID | Test Name | Description | Mock | Expected |
|---|---|---|---|---|
| B1 | `test_format_context_single_chunk` | One chunk → contains its `page_content` in output | None (pure function) | `"Section 1" in result` |
| B2 | `test_format_context_multiple_chunks` | Three chunks → all three `page_content` strings appear | None | All three strings present |
| B3 | `test_format_context_includes_source` | Output contains `[Source: ...]` header for each chunk | None | `"[Source:" in result` |
| B4 | `test_format_context_empty_raises` | Empty `chunks=[]` → raises `ValueError` | None | `pytest.raises(ValueError)` |
| B5 | `test_format_context_missing_source_metadata` | Chunk with no `source` key in metadata → uses `"unknown"` fallback | None | `"[Source: unknown]"` appears |
| B6 | `test_format_context_preserves_order` | First chunk's content appears before second chunk's in output | None | `result.index(chunks[0].page_content) < result.index(chunks[1].page_content)` |

##### C — `build_prompt()` Tests

| Test ID | Test Name | Description | Mock | Expected |
|---|---|---|---|---|
| C1 | `test_build_prompt_contains_question` | Rendered prompt includes the question string | None | `question in result` |
| C2 | `test_build_prompt_contains_context` | Rendered prompt includes chunk text | None | `chunk.page_content in result` |
| C3 | `test_build_prompt_uses_template` | Default template's static text is present | None | `"You are a helpful assistant" in result` |
| C4 | `test_build_prompt_custom_template` | `LLMClient(prompt_template=custom)` uses the custom string | patch ChatOpenAI | Custom static text in result |
| C5 | `test_build_prompt_empty_question_raises` | `question=""` → raises `ValueError` | None | `pytest.raises(ValueError)` |

##### D — `generate()` Happy Path Tests

| Test ID | Test Name | Description | Mock | Expected |
|---|---|---|---|---|
| D1 | `test_generate_returns_llm_response` | `generate()` returns an `LLMResponse` instance | Mock chain `.invoke()` → string | `isinstance(result, LLMResponse)` |
| D2 | `test_generate_answer_matches_mock` | `result.answer` equals the mocked return value | Mock chain `.invoke()` → `"Mocked answer"` | `result.answer == "Mocked answer"` |
| D3 | `test_generate_sources_populated` | `result.sources` equals the input `chunks` list | Mock chain | `result.sources == chunks` |
| D4 | `test_generate_model_name_in_response` | `result.model` reflects the configured model name | Mock chain | `result.model == "gpt-3.5-turbo"` |
| D5 | `test_generate_chain_invoked_once` | The LCEL chain `.invoke()` is called exactly one time | Mock chain | `mock_chain.invoke.call_count == 1` |
| D6 | `test_generate_chain_receives_context` | `.invoke()` is called with dict containing `"context"` key | Mock chain + `call_args` assertion | `"context" in mock_chain.invoke.call_args[0][0]` |
| D7 | `test_generate_chain_receives_question` | `.invoke()` is called with dict containing `"question"` key | Mock chain | `"question" in mock_chain.invoke.call_args[0][0]` |
| D8 | `test_generate_gpt4_model` | Works identically when `model_name="gpt-4"` | patch ChatOpenAI | `result.answer` is non-empty string |

##### E — `generate()` Edge Cases and Error Paths

| Test ID | Test Name | Description | Mock | Expected |
|---|---|---|---|---|
| E1 | `test_generate_empty_question_raises` | `question=""` → `ValueError` before API call | None needed | `pytest.raises(ValueError)` |
| E2 | `test_generate_whitespace_question_raises` | `question="   "` → `ValueError` | None needed | `pytest.raises(ValueError)` |
| E3 | `test_generate_empty_chunks_raises` | `chunks=[]` → `ValueError` (no context to inject) | None needed | `pytest.raises(ValueError)` |
| E4 | `test_generate_propagates_auth_error` | Chain raises `openai.AuthenticationError` → propagated | Mock chain `.invoke.side_effect = AuthenticationError(...)` | `pytest.raises(openai.AuthenticationError)` |
| E5 | `test_generate_propagates_rate_limit_error` | Chain raises `openai.RateLimitError` → propagated | Mock chain `.invoke.side_effect = RateLimitError(...)` | `pytest.raises(openai.RateLimitError)` |
| E6 | `test_generate_very_long_question` | 2000-character question does not cause internal error | Mock chain | Returns `LLMResponse` without raising |
| E7 | `test_generate_single_chunk` | `chunks` with exactly one item works correctly | Mock chain | `result.sources` has length 1 |
| E8 | `test_generate_special_characters_in_question` | Question with `"`, `\n`, `{`, `}` does not break template | Mock chain | Returns `LLMResponse` |

##### F — `LLMResponse` Dataclass Tests

| Test ID | Test Name | Description | Mock | Expected |
|---|---|---|---|---|
| F1 | `test_llm_response_default_fields` | Instantiate with only `answer` → other fields have defaults | None | `response.sources == []`, `response.model == ""` |
| F2 | `test_llm_response_all_fields` | All fields set correctly | None | Each field matches its input |
| F3 | `test_llm_response_is_dataclass` | `LLMResponse` is a proper dataclass | None | `dataclasses.is_dataclass(LLMResponse)` is `True` |

##### G — Prompt Template Assertions

| Test ID | Test Name | Description | Mock | Expected |
|---|---|---|---|---|
| G1 | `test_rag_prompt_template_has_context_placeholder` | `RAG_PROMPT_TEMPLATE` contains `{context}` | None (module-level constant) | `"{context}" in RAG_PROMPT_TEMPLATE` |
| G2 | `test_rag_prompt_template_has_question_placeholder` | `RAG_PROMPT_TEMPLATE` contains `{question}` | None | `"{question}" in RAG_PROMPT_TEMPLATE` |
| G3 | `test_rag_prompt_template_fallback_instruction` | Template contains the "I don't have enough information" fallback | None | Fallback string present |

##### H — Chain Construction Tests

| Test ID | Test Name | Description | Mock | Expected |
|---|---|---|---|---|
| H1 | `test_build_chain_returns_runnable` | `_build_chain()` returns a non-None object | patch ChatOpenAI | `chain is not None` |
| H2 | `test_build_chain_called_once` | Chain is lazily built and cached, not rebuilt on every `generate()` call | patch ChatOpenAI + spy | `_build_chain` called exactly once across two `generate()` invocations |

---

### 7.2 `tests/test_loader.py`

#### Test Case Inventory

```python
"""
Tests for src.loader.DocumentLoader.

Uses tmp_path fixture to create real (but minimal) fake files so that
LangChain loaders can be exercised without real OpenAI calls.
LangChain loaders are patched at the class level to return controlled
Document lists.
"""
```

| Test ID | Test Name | Description | Mock / Fixture |
|---|---|---|---|
| L1 | `test_load_raises_if_dir_missing` | Non-existent `docs_dir` → `FileNotFoundError` | `tmp_path` |
| L2 | `test_load_raises_if_no_supported_files` | Dir with only `.txt` → `ValueError` | `tmp_path` with `.txt` file |
| L3 | `test_load_pdf_dispatches_pypdf_loader` | `.pdf` file → `PyPDFLoader` is instantiated | `patch("src.loader.PyPDFLoader")` |
| L4 | `test_load_docx_dispatches_docx2txt_loader` | `.docx` file → `Docx2txtLoader` is instantiated | `patch("src.loader.Docx2txtLoader")` |
| L5 | `test_load_csv_dispatches_csv_loader` | `.csv` file → `CSVLoader` is instantiated | `patch("src.loader.CSVLoader")` |
| L6 | `test_load_json_dispatches_json_loader` | `.json` file → `JSONLoader` is instantiated | `patch("src.loader.JSONLoader")` |
| L7 | `test_load_returns_document_list` | Mixed directory → returns `List[Document]` | All loaders patched with fake Documents |
| L8 | `test_load_aggregates_multi_page_pdf` | PDF loader returns 3 pages → all 3 in result | `patch PyPDFLoader` → returns 3 Documents |
| L9 | `test_split_produces_smaller_chunks` | 5000-char document with `chunk_size=500` → ≥10 chunks | Real `RecursiveCharacterTextSplitter` |
| L10 | `test_split_chunk_overlap_metadata` | Each chunk has `chunk_index` in metadata | Real splitter |
| L11 | `test_load_and_split_chains_both_steps` | `load_and_split()` calls `load()` then `split()` | Spy on both methods |
| L12 | `test_split_empty_input_returns_empty` | `split([])` returns `[]` without error | None |
| L13 | `test_source_metadata_preserved` | `source` key from original Document is present in chunks | Patched loader |

---

### 7.3 `tests/test_embedder.py`

#### Test Case Inventory

```python
"""
Tests for src.embedder.VectorStoreManager.

FAISS.from_documents and FAISS.load_local are class-level functions;
they must be patched at the module import level:
    patch("src.embedder.FAISS.from_documents")
    patch("src.embedder.FAISS.load_local")
"""
```

| Test ID | Test Name | Description | Mock / Fixture |
|---|---|---|---|
| E1 | `test_build_calls_faiss_from_documents` | `build()` calls `FAISS.from_documents` with chunks and embeddings | `patch("src.embedder.FAISS")` |
| E2 | `test_build_raises_on_empty_chunks` | `build([])` → `ValueError` | None |
| E3 | `test_build_returns_faiss_instance` | `build()` returns the mock FAISS object | `patch("src.embedder.FAISS")` |
| E4 | `test_save_calls_save_local` | `save(mock_store)` → `mock_store.save_local()` called | `mock_faiss_store` fixture |
| E5 | `test_save_uses_correct_path` | `save_local` receives the configured `vector_store_path` | `mock_faiss_store` fixture |
| E6 | `test_load_calls_faiss_load_local` | `load()` calls `FAISS.load_local` with correct args | `patch("src.embedder.FAISS")` |
| E7 | `test_load_passes_allow_dangerous` | `FAISS.load_local` called with `allow_dangerous_deserialization=True` | `patch("src.embedder.FAISS")` |
| E8 | `test_load_raises_if_path_missing` | `vector_store_path` doesn't exist → `FileNotFoundError` | `tmp_path` (no index files) |
| E9 | `test_build_and_save_calls_both` | `build_and_save()` calls both `build()` and `save()` | Spy on both methods |
| E10 | `test_similarity_search_delegates_to_faiss` | `similarity_search()` calls `vector_store.similarity_search(query, k=top_k)` | `mock_faiss_store` fixture |
| E11 | `test_similarity_search_returns_documents` | Result is a `List[Document]` | `mock_faiss_store` fixture |
| E12 | `test_embeddings_model_name_configured` | `OpenAIEmbeddings` instantiated with the correct model name | `patch("src.embedder.OpenAIEmbeddings")` |

---

### 7.4 `tests/test_rag.py`

#### Test Case Inventory

```python
"""
Tests for src.rag.RAGPipeline and src.rag.RAGResponse.

The pipeline is tested by replacing its three collaborators:
    - DocumentLoader
    - VectorStoreManager
    - LLMClient
with MagicMock instances, ensuring that RAGPipeline correctly
orchestrates them without testing their internals again.
"""
```

| Test ID | Test Name | Description | Mock / Fixture |
|---|---|---|---|
| R1 | `test_build_index_calls_load_and_split` | `build_index()` calls `DocumentLoader.load_and_split()` | `patch("src.rag.DocumentLoader")` |
| R2 | `test_build_index_calls_vsm_build` | `build_index()` passes chunks to `VectorStoreManager.build()` | `patch VectorStoreManager` |
| R3 | `test_build_index_calls_vsm_save` | `build_index()` calls `VectorStoreManager.save()` | `patch VectorStoreManager` |
| R4 | `test_ask_loads_index_if_not_cached` | First `ask()` call triggers `VectorStoreManager.load()` | `patch VectorStoreManager` |
| R5 | `test_ask_does_not_reload_if_cached` | Second `ask()` does NOT call `VectorStoreManager.load()` again | `patch VectorStoreManager` |
| R6 | `test_ask_calls_similarity_search` | `ask()` calls `similarity_search` with the question | `patch VectorStoreManager` |
| R7 | `test_ask_passes_top_k` | `similarity_search` called with the configured `top_k` | `patch VectorStoreManager` |
| R8 | `test_ask_calls_llm_generate` | `ask()` calls `LLMClient.generate()` with question and chunks | `patch LLMClient` |
| R9 | `test_ask_returns_rag_response` | `ask()` returns a `RAGResponse` instance | All collaborators mocked |
| R10 | `test_ask_response_contains_answer` | `RAGResponse.answer` matches the mocked LLM answer | All collaborators mocked |
| R11 | `test_ask_response_contains_sources` | `RAGResponse.sources` contains retrieved chunks | All collaborators mocked |
| R12 | `test_ask_empty_question_raises` | `ask("")` → `ValueError` | None (guard is in `LLMClient`) |
| R13 | `test_is_index_loaded_false_initially` | `pipeline.is_index_loaded` is `False` before first `ask()` | None |
| R14 | `test_is_index_loaded_true_after_ask` | `pipeline.is_index_loaded` is `True` after `ask()` | All collaborators mocked |
| R15 | `test_rag_response_dataclass` | `RAGResponse` is a proper dataclass | `dataclasses.is_dataclass` |

---

## 8. Mocking Strategy — External API Calls

This section is the definitive reference for **how to mock** each external boundary. Zero real HTTP requests must occur during `pytest`.

### 8.1 Patching Philosophy

```
Rule 1: Patch at the point of USE, not the point of definition.
        If src/llm.py imports ChatOpenAI from langchain_openai,
        patch "src.llm.ChatOpenAI", NOT "langchain_openai.ChatOpenAI".

Rule 2: Prefer pytest-mock's `mocker.patch()` in fixtures.
        It auto-restores the patch after each test with no context managers.

Rule 3: For class-level methods (FAISS.from_documents), patch the class.
        patch("src.embedder.FAISS") replaces the entire class;
        FAISS.from_documents becomes a method on the mock class.

Rule 4: Use autospec=True where the call signature matters.
        mocker.patch("src.llm.ChatOpenAI", autospec=True)
        prevents calling it with wrong argument counts.
```

### 8.2 Mock Target Map

```
External Boundary               Patch Target String              Return Value Shape
─────────────────────────────────────────────────────────────────────────────────────
OpenAI chat completion          src.llm.ChatOpenAI               MagicMock with
                                                                  .invoke() → str

OpenAI embeddings               src.embedder.OpenAIEmbeddings    MagicMock

FAISS.from_documents            src.embedder.FAISS               MagicMock with
                                                                  .similarity_search()
                                                                  → List[Document]

FAISS.load_local                src.embedder.FAISS               Same MagicMock

FAISS.save_local                mock_faiss_store.save_local      MagicMock (no return)

PyPDFLoader                     src.loader.PyPDFLoader           MagicMock with
                                                                  .load() → List[Document]

Docx2txtLoader                  src.loader.Docx2txtLoader        Same pattern

CSVLoader                       src.loader.CSVLoader             Same pattern

JSONLoader                      src.loader.JSONLoader            Same pattern

DocumentLoader (in rag tests)   src.rag.DocumentLoader           MagicMock with
                                                                  .load_and_split()
                                                                  → List[Document]

VectorStoreManager (in rag)     src.rag.VectorStoreManager       MagicMock

LLMClient (in rag tests)        src.rag.LLMClient                MagicMock with
                                                                  .generate() →
                                                                  LLMResponse(...)
```

### 8.3 Canonical Mock Patterns

#### Pattern 1 — Patching `ChatOpenAI` in `test_llm.py`

```python
# Pattern used for most test_llm.py tests
def test_generate_returns_llm_response(mocker, sample_chunks):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Payment is due within 30 days."

    # Patch ChatOpenAI constructor so LLMClient doesn't hit the network
    mocker.patch("src.llm.ChatOpenAI")

    # Patch _build_chain so we control what the chain returns
    client = LLMClient(openai_api_key="sk-test")
    mocker.patch.object(client, "_build_chain", return_value=mock_chain)

    result = client.generate("What are the payment terms?", sample_chunks)

    assert isinstance(result, LLMResponse)
    assert result.answer == "Payment is due within 30 days."
```

#### Pattern 2 — `side_effect` for Error Propagation Tests

```python
import openai

def test_generate_propagates_auth_error(mocker, sample_chunks):
    mocker.patch("src.llm.ChatOpenAI")
    client = LLMClient(openai_api_key="sk-bad")

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = openai.AuthenticationError(
        message="Invalid API key",
        response=MagicMock(),
        body={},
    )
    mocker.patch.object(client, "_build_chain", return_value=mock_chain)

    with pytest.raises(openai.AuthenticationError):
        client.generate("What are the terms?", sample_chunks)
```

#### Pattern 3 — Patching a Class-Level Method (`FAISS.from_documents`)

```python
def test_build_calls_faiss_from_documents(mocker, sample_chunks, mock_embeddings):
    mock_faiss_class = mocker.patch("src.embedder.FAISS")
    mock_faiss_class.from_documents.return_value = MagicMock()

    mocker.patch("src.embedder.OpenAIEmbeddings", return_value=mock_embeddings)

    manager = VectorStoreManager(openai_api_key="sk-test")
    manager.build(sample_chunks)

    mock_faiss_class.from_documents.assert_called_once_with(
        sample_chunks,
        mock_embeddings,
    )
```

#### Pattern 4 — Environment Variable Mocking

```python
def test_init_api_key_from_env(mocker):
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"})
    mock_chat = mocker.patch("src.llm.ChatOpenAI")

    LLMClient()  # no api_key arg

    call_kwargs = mock_chat.call_args.kwargs
    assert call_kwargs.get("openai_api_key") == "sk-from-env"
```

#### Pattern 5 — Spying on a Method Without Replacing Its Logic

```python
def test_build_and_save_calls_both(mocker, sample_chunks):
    manager = VectorStoreManager(openai_api_key="sk-test")

    build_spy = mocker.patch.object(manager, "build", return_value=MagicMock())
    save_spy  = mocker.patch.object(manager, "save")

    manager.build_and_save(sample_chunks)

    build_spy.assert_called_once_with(sample_chunks)
    save_spy.assert_called_once()
```

### 8.4 What NOT to Mock

| Boundary | Reason to keep real |
|---|---|
| `RecursiveCharacterTextSplitter` | Pure in-memory text processing; fast and deterministic |
| `LLMResponse` / `RAGResponse` dataclasses | Plain Python; mocking adds zero value |
| `format_context()` method | Pure string logic; test with real inputs |
| `build_prompt()` method | Pure string rendering; test with real inputs |
| `pathlib.Path` operations | Cheap; use `tmp_path` fixture from pytest |

---

## 9. Data Flow & Component Interaction Diagram

```
╔══════════════════════════════════════════════════════════════════════╗
║  INDEXING PHASE  (build_index)                                       ║
╚══════════════════════════════════════════════════════════════════════╝

  docs/
  ├── ACME_Globex_Contract.docx
  ├── ACME_HR_Handbook_v4.2.pdf     ──►  DocumentLoader.load()
  ├── acme_sales_q3.csv                       │
  └── globex_crm_profile.json                 │ List[Document]
                                              ▼
                                   DocumentLoader.split()
                                              │
                                              │ List[Document] (chunks)
                                              ▼
                                  VectorStoreManager.build()
                                       │           │
                           OpenAIEmbeddings    FAISS.from_documents()
                           (ada-002)                │
                                                    │ FAISS index
                                                    ▼
                                       VectorStoreManager.save()
                                              │
                                       vector_store/
                                       ├── index.faiss
                                       └── index.pkl


╔══════════════════════════════════════════════════════════════════════╗
║  QUERY PHASE  (ask)                                                  ║
╚══════════════════════════════════════════════════════════════════════╝

  User: "What are the payment terms?"
                │
                ▼
       RAGPipeline.ask(question)
                │
                ├──► VectorStoreManager.load()   (if not cached)
                │         │
                │    vector_store/ ──► FAISS index in memory
                │
                ├──► VectorStoreManager.similarity_search(question, top_k=4)
                │         │
                │    OpenAIEmbeddings.embed_query(question)  ──► query vector
                │         │
                │    FAISS.similarity_search()  ──► List[Document] (top 4 chunks)
                │
                └──► LLMClient.generate(question, chunks)
                          │
                          ├── format_context(chunks)  ──► context string
                          ├── build_prompt(question, context)  ──► rendered prompt
                          │
                          └── LCEL chain.invoke({"context": ..., "question": ...})
                                    │
                               ChatOpenAI (GPT-3.5 / GPT-4)
                                    │
                               StrOutputParser  ──► answer string
                                    │
                              LLMResponse(
                                answer=...,
                                sources=chunks,
                                model="gpt-3.5-turbo",
                              )
                                    │
                              RAGResponse(
                                question=...,
                                answer=...,
                                sources=...,
                              )
                                    │
                                    ▼
                             User sees answer
```

---

## 10. Implementation Task Breakdown

Tasks are ordered by dependency. Items marked **[BLOCKER]** must be complete before dependent tasks can start.

### Phase 0 — Repository Scaffolding

| Task | File(s) | Notes |
|---|---|---|
| 0.1 Create `src/` package | `src/__init__.py` | Empty file |
| 0.2 Create `tests/` package | `tests/__init__.py` | Empty file |
| 0.3 Create `requirements.txt` | `requirements.txt` | See Section 3 |
| 0.4 Create `pytest.ini` | `pytest.ini` | See Section 6.1 |
| 0.5 Create `.env.example` | `.env.example` | `OPENAI_API_KEY=`, `DOCS_DIR=./docs`, `VECTOR_STORE_PATH=./vector_store` |
| 0.6 Create `.gitignore` | `.gitignore` | Ignore `.env`, `vector_store/`, `__pycache__/`, `.venv/`, `htmlcov/` |

### Phase 1 — Production Modules **[BLOCKER for Phase 2]**

| Task | File | Interfaces to implement |
|---|---|---|
| 1.1 Implement `DocumentLoader` | `src/loader.py` | `__init__`, `load`, `split`, `load_and_split`, `_discover_files`, `_load_single_file` |
| 1.2 Implement `VectorStoreManager` | `src/embedder.py` | `__init__`, `build`, `save`, `load`, `build_and_save`, `similarity_search`, `embeddings` property |
| 1.3 Define `LLMResponse` dataclass | `src/llm.py` | `answer`, `sources`, `model`, `prompt_tokens`, `completion_tokens`, `total_tokens` |
| 1.4 Implement `LLMClient` | `src/llm.py` | `__init__`, `format_context`, `build_prompt`, `generate`, `chat_model` property, `_build_chain` |
| 1.5 Define `RAGResponse` dataclass | `src/rag.py` | `question`, `answer`, `sources`, `model`, `total_tokens` |
| 1.6 Implement `RAGPipeline` | `src/rag.py` | `__init__`, `build_index`, `ask`, `is_index_loaded`, `_ensure_index_loaded` |
| 1.7 Implement CLI | `main.py` | `parse_args`, `print_response`, `interactive_loop`, `main` |

### Phase 2 — Test Infrastructure **[BLOCKER for Phase 3]**

| Task | File | Contents |
|---|---|---|
| 2.1 Create shared fixtures | `tests/conftest.py` | All fixtures from Section 6.3 |

### Phase 3 — Unit Tests (can be implemented in parallel)

| Task | File | Test Count | Priority |
|---|---|---|---|
| 3.1 Write LLM tests | `tests/test_llm.py` | 34 tests (A–H) | **Highest** |
| 3.2 Write loader tests | `tests/test_loader.py` | 13 tests | High |
| 3.3 Write embedder tests | `tests/test_embedder.py` | 12 tests | High |
| 3.4 Write RAG tests | `tests/test_rag.py` | 15 tests | High |

### Phase 4 — Validation

| Task | Command | Pass Criteria |
|---|---|---|
| 4.1 Install dependencies | `pip install -r requirements.txt` | No errors |
| 4.2 Run full test suite | `pytest` | All tests green, coverage ≥ 80% |
| 4.3 Run LLM tests only | `pytest tests/test_llm.py -v` | All 34 tests pass |
| 4.4 Check coverage report | `open htmlcov/index.html` | `src/llm.py` at ≥ 90% |
| 4.5 Smoke test (optional) | `OPENAI_API_KEY=sk-... python main.py --question "..."` | Returns a grounded answer |

---

## 11. Acceptance Criteria

The implementation is complete when **all** of the following are true:

### Functional Acceptance Criteria

- [ ] **AC-01** `DocumentLoader.load()` successfully loads all four files in `docs/` without raising exceptions  
- [ ] **AC-02** `DocumentLoader.load_and_split()` produces chunks where `len(chunk.page_content) <= chunk_size + chunk_overlap`  
- [ ] **AC-03** `VectorStoreManager.build_and_save()` creates files `vector_store/index.faiss` and `vector_store/index.pkl`  
- [ ] **AC-04** `RAGPipeline.ask("What are the payment terms?")` returns an `RAGResponse` with a non-empty `answer`  
- [ ] **AC-05** `LLMClient.generate()` raises `ValueError` for empty or whitespace-only questions  
- [ ] **AC-06** `LLMClient.generate()` raises `ValueError` when `chunks=[]`  

### Test Acceptance Criteria

- [ ] **AC-07** `pytest tests/test_llm.py` passes with **zero failures** and **zero real API calls**  
- [ ] **AC-08** `pytest` (full suite) passes with **≥ 80% overall coverage** of the `src/` package  
- [ ] **AC-09** `src/llm.py` has **≥ 90% line coverage**  
- [ ] **AC-10** No test makes a network connection (verified by running tests with network disabled)  
- [ ] **AC-11** All tests complete in **< 10 seconds** (no I/O waits)  

### Code Quality Acceptance Criteria

- [ ] **AC-12** All public functions in `src/` have type annotations  
- [ ] **AC-13** No hardcoded API keys in any source file  
- [ ] **AC-14** `.env` is listed in `.gitignore`  
- [ ] **AC-15** `vector_store/` directory is listed in `.gitignore`  

---

## 12. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **LangChain v0.2 API breaking changes** between minor versions | Medium | High | Pin exact versions in `requirements.txt`; add a `pip-audit` step to CI |
| **FAISS `allow_dangerous_deserialization`** flag causes `ValueError` if forgotten | High | Medium | Encode it in the spec (Section 4.2) and test for it (E7) |
| **JSONLoader `jq` schema mismatch** with `globex_crm_profile.json` structure | Medium | Medium | Default to `"."` schema; add a `test_load_json_dispatches_json_loader` test with the real file path |
| **`tiktoken` encoding mismatch** when counting tokens for chunk sizing | Low | Low | Use character-based `len` as `length_function`; tiktoken only needed for `max_tokens` validation |
| **Mock drift** — mocks don't match the real LangChain API after upgrades | Medium | Medium | Use `autospec=True` on critical mocks (`ChatOpenAI`, `OpenAIEmbeddings`) |
| **Test isolation failure** — one test mutates shared fixture state | Low | High | Use `deepcopy` for mutable fixtures; mark fixtures with appropriate scope (`function` by default) |
| **`faiss-cpu` not available on ARM Macs (M1/M2)** | Medium | Medium | Document the `pip install faiss-cpu` workaround via Rosetta, or use `chromadb` as a fallback |
| **OpenAI API key leaked in test logs** | Low | Critical | Always use `"sk-test-key"` placeholder strings in tests; add `--no-header` to pytest output |
| **CSV/JSON encoding errors** on non-UTF-8 files | Low | Medium | Specify `encoding="utf-8"` in `CSVLoader` and handle `UnicodeDecodeError` in `_load_single_file` |

---

*End of Implementation Plan*

> **Next Step:** Begin with Phase 0 (scaffolding) and Phase 1 (production modules), then proceed to Phase 3 test implementation. The `tests/test_llm.py` file should be treated as the primary deliverable and implemented first within Phase 3.
