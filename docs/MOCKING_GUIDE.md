# Mocking Strategy Reference

> Quick-reference guide for every mock target in the RAG test suite.
> All patterns use `pytest-mock`'s `mocker` fixture unless noted otherwise.

---

## Core Rule

> **Patch at the point of USE, not the point of definition.**

```python
# ✅ Correct — patch where the name is imported/used
mocker.patch("src.llm.ChatOpenAI")

# ❌ Wrong — patching the source module has no effect on src.llm
mocker.patch("langchain_openai.ChatOpenAI")
```

---

## Full Mock Target Map

| Module | External Boundary | Patch Target String | Shape of Return Value |
|---|---|---|---|
| `src/llm.py` | `ChatOpenAI` constructor | `src.llm.ChatOpenAI` | `MagicMock` with `.invoke()` |
| `src/llm.py` | LCEL chain `.invoke()` | `mocker.patch.object(client, "_build_chain", return_value=mock_chain)` | `mock_chain.invoke → str` |
| `src/embedder.py` | `OpenAIEmbeddings` constructor | `src.embedder.OpenAIEmbeddings` | `MagicMock` |
| `src/embedder.py` | `FAISS.from_documents` (classmethod) | `src.embedder.FAISS` | `MagicMock` with `.from_documents` |
| `src/embedder.py` | `FAISS.load_local` (classmethod) | `src.embedder.FAISS` | Same mock class |
| `src/embedder.py` | `faiss_store.save_local` | `mock_faiss_store.save_local` (fixture method) | `MagicMock()` |
| `src/loader.py` | `PyPDFLoader` | `src.loader.PyPDFLoader` | `MagicMock` with `.load() → List[Document]` |
| `src/loader.py` | `Docx2txtLoader` | `src.loader.Docx2txtLoader` | Same pattern |
| `src/loader.py` | `CSVLoader` | `src.loader.CSVLoader` | Same pattern |
| `src/loader.py` | `JSONLoader` | `src.loader.JSONLoader` | Same pattern |
| `src/rag.py` | `DocumentLoader` (collaborator) | `src.rag.DocumentLoader` | `MagicMock` with `.load_and_split()` |
| `src/rag.py` | `VectorStoreManager` (collaborator) | `src.rag.VectorStoreManager` | `MagicMock` with `.load()`, `.similarity_search()`, `.build()`, `.save()` |
| `src/rag.py` | `LLMClient` (collaborator) | `src.rag.LLMClient` | `MagicMock` with `.generate() → LLMResponse` |
| All | `os.environ` | `mocker.patch.dict(os.environ, {...})` | Dict replacement |

---

## Pattern 1 — Replacing a Constructor (`ChatOpenAI`)

Use this when you want `LLMClient.__init__` to run normally but the
`ChatOpenAI` class it calls to be a mock.

```python
def test_init_temperature_zero(mocker):
    mock_chat_class = mocker.patch("src.llm.ChatOpenAI")

    LLMClient(openai_api_key="sk-test")

    # Inspect how the constructor was called
    mock_chat_class.assert_called_once()
    call_kwargs = mock_chat_class.call_args.kwargs
    assert call_kwargs["temperature"] == 0.0
```

---

## Pattern 2 — Controlling `generate()` Output via `_build_chain`

The LCEL chain is built lazily inside `_build_chain()`. Patch the method
to inject a controllable mock chain.

```python
def test_generate_answer_matches_mock(mocker, sample_chunks):
    mocker.patch("src.llm.ChatOpenAI")          # prevent real constructor
    client = LLMClient(openai_api_key="sk-test")

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Net 30 days."
    mocker.patch.object(client, "_build_chain", return_value=mock_chain)

    result = client.generate("What are the payment terms?", sample_chunks)

    assert result.answer == "Net 30 days."
    mock_chain.invoke.assert_called_once()
```

---

## Pattern 3 — `side_effect` for Exception Propagation

Use `side_effect` when the mock must raise an exception.

```python
import openai
from unittest.mock import MagicMock

def test_generate_propagates_auth_error(mocker, sample_chunks):
    mocker.patch("src.llm.ChatOpenAI")
    client = LLMClient(openai_api_key="sk-bad-key")

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = openai.AuthenticationError(
        message="Incorrect API key",
        response=MagicMock(status_code=401),
        body={"error": {"type": "invalid_api_key"}},
    )
    mocker.patch.object(client, "_build_chain", return_value=mock_chain)

    with pytest.raises(openai.AuthenticationError):
        client.generate("Any question", sample_chunks)
```

---

## Pattern 4 — Patching a Class-Level Method (`FAISS.from_documents`)

`FAISS.from_documents` is a classmethod. Patch the entire class and set
the attribute on the mock class object.

```python
def test_build_calls_faiss_from_documents(mocker, sample_chunks):
    mock_embeddings = MagicMock()
    mocker.patch("src.embedder.OpenAIEmbeddings", return_value=mock_embeddings)

    mock_faiss_class = mocker.patch("src.embedder.FAISS")
    mock_store = MagicMock()
    mock_faiss_class.from_documents.return_value = mock_store

    manager = VectorStoreManager(openai_api_key="sk-test")
    result = manager.build(sample_chunks)

    mock_faiss_class.from_documents.assert_called_once_with(
        sample_chunks, mock_embeddings
    )
    assert result is mock_store
```

---

## Pattern 5 — Patching `FAISS.load_local`

```python
def test_load_passes_allow_dangerous(mocker):
    mock_embeddings = MagicMock()
    mocker.patch("src.embedder.OpenAIEmbeddings", return_value=mock_embeddings)

    mock_faiss_class = mocker.patch("src.embedder.FAISS")
    mock_faiss_class.load_local.return_value = MagicMock()

    manager = VectorStoreManager(
        openai_api_key="sk-test",
        vector_store_path="./vector_store",
    )
    manager.load()

    # Verify the critical safety kwarg is present
    call_kwargs = mock_faiss_class.load_local.call_args.kwargs
    assert call_kwargs.get("allow_dangerous_deserialization") is True
```

---

## Pattern 6 — Environment Variable Override

```python
import os

def test_init_api_key_from_env(mocker):
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"}, clear=False)
    mock_chat_class = mocker.patch("src.llm.ChatOpenAI")

    LLMClient()   # no explicit api_key

    call_kwargs = mock_chat_class.call_args.kwargs
    assert call_kwargs.get("openai_api_key") == "sk-from-env"
```

---

## Pattern 7 — Spying on Collaborator Calls in `RAGPipeline`

In `test_rag.py`, replace the three collaborator classes at the
`src.rag` module level so `RAGPipeline.__init__` picks up the mocks.

```python
def test_build_index_calls_vsm_build(mocker, sample_chunks):
    mock_loader_class = mocker.patch("src.rag.DocumentLoader")
    mock_loader_instance = MagicMock()
    mock_loader_instance.load_and_split.return_value = sample_chunks
    mock_loader_class.return_value = mock_loader_instance

    mock_vsm_class = mocker.patch("src.rag.VectorStoreManager")
    mock_vsm_instance = MagicMock()
    mock_vsm_class.return_value = mock_vsm_instance

    pipeline = RAGPipeline(docs_dir="./docs", openai_api_key="sk-test")
    pipeline.build_index()

    mock_vsm_instance.build.assert_called_once_with(sample_chunks)
```

---

## Pattern 8 — Spying Without Replacing Logic

When you want to verify a method is called without changing its behaviour,
use `mocker.spy()`.

```python
def test_load_and_split_chains_both_steps(mocker, tmp_docs_dir):
    loader = DocumentLoader(docs_dir=tmp_docs_dir)

    mocker.patch.object(loader, "load", wraps=loader.load)
    split_spy = mocker.spy(loader, "split")

    loader.load_and_split()

    assert split_spy.call_count == 1
```

---

## Pattern 9 — `autospec=True` for Signature Safety

Use `autospec=True` when you want the mock to enforce the real method
signature. This catches bugs where tests pass wrong argument counts.

```python
def test_similarity_search_called_with_correct_args(mocker, sample_chunks):
    mock_store = mocker.MagicMock(spec=FAISS)   # spec enforces real FAISS interface
    mock_store.similarity_search.return_value = sample_chunks[:2]

    manager = VectorStoreManager.__new__(VectorStoreManager)  # skip __init__
    result = manager.similarity_search("query text", mock_store, top_k=3)

    mock_store.similarity_search.assert_called_once_with("query text", k=3)
```

---

## What NOT to Mock

| Item | Reason |
|---|---|
| `RecursiveCharacterTextSplitter` | Pure in-memory; fast and deterministic |
| `LLMResponse` / `RAGResponse` dataclasses | Plain Python data containers |
| `format_context()` | Pure string transformation — test real behaviour |
| `build_prompt()` | Pure string rendering — test real behaviour |
| `pathlib.Path` | Cheap; use `tmp_path` for real temporary directories |
| `dataclasses.is_dataclass` | Introspection function; should be called on real class |

---

## Fixture Scoping Rules

| Fixture | Scope | Why |
|---|---|---|
| `sample_document` | `function` (default) | Mutable — reset between tests |
| `sample_chunks` | `function` | Same |
| `mock_embeddings` | `function` | Fresh mock per test |
| `mock_faiss_store` | `function` | Fresh mock per test |
| `mock_chat_openai` | `function` | Fresh mock per test |
| `llm_client_with_mock` | `function` | Depends on `mock_chat_openai` |
| `tmp_docs_dir` | `function` | `tmp_path` is always function-scoped |
