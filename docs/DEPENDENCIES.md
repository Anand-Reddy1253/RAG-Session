# Dependency Specification

> Complete annotated `requirements.txt` for the RAG pipeline.
> All version pins are minimum bounds — tested against LangChain v0.2 split-package architecture.

---

## Production Dependencies

```
# ── LangChain ecosystem (v0.2 split-package architecture) ──────────────────────
#
# LangChain v0.2 split the monolithic package into purpose-specific sub-packages.
# All four packages below are required.

langchain>=0.2.0
# Core orchestration layer: Chain, Runnable, ConversationMemory, etc.
# Minimum 0.2.0 for stable LCEL (LangChain Expression Language) pipe operator.

langchain-core>=0.2.0
# Base abstractions: BaseMessage, ChatMessage, Document, PromptTemplate,
# Runnable, BaseOutputParser. Required by both langchain-openai and
# langchain-community.

langchain-openai>=0.1.8
# Official LangChain wrappers for OpenAI services:
#   - ChatOpenAI        (chat completion — GPT-3.5-turbo, GPT-4)
#   - OpenAIEmbeddings  (text-embedding-ada-002)
# Minimum 0.1.8 for stable usage_metadata on AIMessage.

langchain-community>=0.2.0
# Community-maintained integrations:
#   Document loaders: PyPDFLoader, Docx2txtLoader, CSVLoader, JSONLoader
#   Vector stores:    FAISS
# Minimum 0.2.0 for deprecation-free FAISS.load_local signature.

langchain-text-splitters>=0.2.0
# RecursiveCharacterTextSplitter (split by paragraph → sentence → word → char)
# TokenTextSplitter (tiktoken-aware, optional upgrade path)

# ── OpenAI SDK ──────────────────────────────────────────────────────────────────

openai>=1.14.0
# Underlying HTTP client used by langchain-openai. Must be ≥ 1.0.0 for
# the v1 API (openai.OpenAI() client). Version 1.14.0 introduces stable
# AuthenticationError and RateLimitError classes used in error-path tests.

# ── Vector Store ────────────────────────────────────────────────────────────────

faiss-cpu>=1.7.4
# Facebook AI Similarity Search — CPU build.
# Provides in-process approximate nearest neighbour search with no server.
# On CUDA machines replace with: faiss-gpu>=1.7.4
# On Apple Silicon (M1/M2) may need: pip install faiss-cpu --no-binary faiss-cpu

# ── Document parsers ────────────────────────────────────────────────────────────
# These packages are called internally by LangChain loaders.
# They do NOT need to be imported directly in application code.

pypdf>=3.17.0
# PDF text extraction engine used by PyPDFLoader.
# Minimum 3.17.0 for stable multi-page extraction and metadata support.
# (Note: older versions used the 'PyPDF2' package name — do not use PyPDF2.)

docx2txt>=0.8
# .docx plain-text extraction used by Docx2txtLoader.
# Converts Word XML structure to plain text; preserves paragraph order.

python-docx>=1.1.0
# Full Word document API — used as a fallback and for richer DOCX metadata.
# Also required by some LangChain community loader variants.

jq>=1.6.0
# Python bindings for the jq JSON query language.
# Required by JSONLoader for jq_schema-based field extraction.
# The globex_crm_profile.json is loaded with jq_schema="."

# ── Numerical utilities ─────────────────────────────────────────────────────────

numpy>=1.24.0
# Required by FAISS Python bindings for vector array operations.
# Also used internally by OpenAIEmbeddings for embedding normalisation.

tiktoken>=0.6.0
# OpenAI tokeniser — used for accurate token counting.
# Needed when implementing chunk_size in tokens rather than characters,
# and for populating LLMResponse.prompt_tokens / completion_tokens fields.

# ── Environment & configuration ─────────────────────────────────────────────────

python-dotenv>=1.0.0
# Reads .env file and injects variables into os.environ.
# Called as: load_dotenv() at the top of main.py.
# Does nothing if .env does not exist (safe for CI environments where
# secrets are injected via environment variables directly).
```

---

## Test Dependencies

```
# ── Test runner ─────────────────────────────────────────────────────────────────

pytest>=7.4.0
# Test discovery, fixtures, parametrize, markers, and assertions.
# Minimum 7.4.0 for stable tmp_path fixture and --tb=short output.

pytest-mock>=3.11.0
# Provides the `mocker` fixture — a thin wrapper over unittest.mock.
# Preferred over raw unittest.mock.patch because:
#   - Patches are automatically restored after each test (no context managers)
#   - mocker.patch.object() syntax is cleaner than @patch decorators
#   - Works correctly with pytest fixture dependency injection

pytest-cov>=4.1.0
# Coverage plugin for pytest.
# Invoked via: pytest --cov=src --cov-report=term-missing
# Configured in pytest.ini to fail if coverage < 80%.

coverage>=7.3.0
# Underlying coverage measurement engine used by pytest-cov.
# Minimum 7.3.0 for branch coverage support and HTML report generation.
```

---

## Development / Optional Dependencies

These are NOT required to run the application or tests, but are recommended for
contributors. Add them to a `requirements-dev.txt` file.

```
# Formatting
black>=24.0.0           # Opinionated code formatter
isort>=5.13.0           # Import sorter (compatible with black)

# Linting
ruff>=0.3.0             # Fast Python linter (replaces flake8 + pylint for most uses)

# Type checking
mypy>=1.8.0             # Static type checker
types-docutils          # Type stubs for common libraries

# Security
pip-audit>=2.7.0        # Scans requirements.txt for known CVEs
```

---

## Version Compatibility Matrix

| Python Version | Supported | Notes |
|---|---|---|
| 3.9 | ✅ | Minimum per README; `str \| Path` syntax requires `from __future__ import annotations` |
| 3.10 | ✅ | Native `str \| Path` syntax |
| 3.11 | ✅ | Recommended (faster startup, better error messages) |
| 3.12 | ✅ | Tested with LangChain 0.2 |
| 3.8 or below | ❌ | Incompatible with langchain-core 0.2 |

---

## Installation Notes

```bash
# Standard installation
pip install -r requirements.txt

# With development extras
pip install -r requirements.txt -r requirements-dev.txt

# CUDA GPU acceleration (replace faiss-cpu)
pip install faiss-gpu>=1.7.4

# Apple Silicon workaround if faiss-cpu binary fails
pip install faiss-cpu --no-binary faiss-cpu
```

---

## Known Conflicts

| Package | Conflict | Resolution |
|---|---|---|
| `faiss-gpu` vs `faiss-cpu` | Cannot install both | Choose one; `faiss-cpu` in `requirements.txt` |
| `PyPDF2` (old) vs `pypdf` (new) | Different package names | Use `pypdf>=3.17.0` only; never `PyPDF2` |
| `langchain` < 0.2 (monolithic) | Different import paths | Pins in requirements.txt prevent this |
