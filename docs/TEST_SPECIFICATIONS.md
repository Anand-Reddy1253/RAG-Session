# Test Suite Specification

> Companion to `IMPLEMENTATION_PLAN.md` — complete test case list with mock targets
> and expected assertions for every test file.

---

## Overview

| File | Test Count | Primary Class Under Test | Real I/O? |
|---|---|---|---|
| `tests/test_llm.py` | 34 | `LLMClient`, `LLMResponse` | ❌ None |
| `tests/test_loader.py` | 13 | `DocumentLoader` | ✅ `tmp_path` only |
| `tests/test_embedder.py` | 12 | `VectorStoreManager` | ✅ `tmp_path` only |
| `tests/test_rag.py` | 15 | `RAGPipeline`, `RAGResponse` | ❌ None |
| **Total** | **74** | | |

All OpenAI network calls are mocked. No test requires `OPENAI_API_KEY` to be set.

---

## `tests/test_llm.py` — 34 Tests

### Group A — Initialisation (7 tests)

```
test_init_default_model
    patch: src.llm.ChatOpenAI
    assert: client.model_name == "gpt-3.5-turbo"

test_init_custom_model
    patch: src.llm.ChatOpenAI
    input: model_name="gpt-4"
    assert: client.model_name == "gpt-4"

test_init_temperature_zero
    patch: src.llm.ChatOpenAI (capture call_args)
    assert: ChatOpenAI called with temperature=0.0

test_init_custom_temperature
    patch: src.llm.ChatOpenAI (capture call_args)
    input: temperature=0.7
    assert: ChatOpenAI called with temperature=0.7

test_init_api_key_forwarded
    patch: src.llm.ChatOpenAI (capture call_args)
    input: openai_api_key="sk-explicit"
    assert: ChatOpenAI called with openai_api_key="sk-explicit"

test_init_api_key_from_env
    patch: src.llm.ChatOpenAI + os.environ["OPENAI_API_KEY"]="sk-env"
    input: LLMClient()  — no api_key arg
    assert: ChatOpenAI called with openai_api_key="sk-env"

test_chat_model_property
    patch: src.llm.ChatOpenAI → returns mock_instance
    assert: client.chat_model is mock_instance
```

### Group B — `format_context()` (6 tests)

```
test_format_context_single_chunk
    input: [Document(page_content="Net 30 days", metadata={"source": "contract.docx"})]
    assert: "Net 30 days" in result

test_format_context_multiple_chunks
    input: 3 chunks (sample_chunks fixture)
    assert: all three page_content strings present in result

test_format_context_includes_source
    input: chunk with source="contract.docx"
    assert: "[Source: contract.docx]" in result  (or similar format)

test_format_context_empty_raises
    input: chunks=[]
    assert: raises ValueError

test_format_context_missing_source_metadata
    input: Document(page_content="text", metadata={})   — no source key
    assert: "[Source: unknown]" in result

test_format_context_preserves_order
    input: [chunk_a, chunk_b]
    assert: result.index(chunk_a.page_content) < result.index(chunk_b.page_content)
```

### Group C — `build_prompt()` (5 tests)

```
test_build_prompt_contains_question
    input: question="What are payment terms?", sample_chunks
    assert: "What are payment terms?" in result

test_build_prompt_contains_context
    input: question="Q?", sample_chunks
    assert: sample_chunks[0].page_content in result

test_build_prompt_uses_template
    assert: "You are a helpful assistant" in result

test_build_prompt_custom_template
    input: prompt_template="Custom: {context} Q: {question}"
    assert: "Custom:" in result

test_build_prompt_empty_question_raises
    input: question=""
    assert: raises ValueError
```

### Group D — `generate()` Happy Path (8 tests)

```
test_generate_returns_llm_response
    mock: chain.invoke → "Mocked answer"
    assert: isinstance(result, LLMResponse)

test_generate_answer_matches_mock
    mock: chain.invoke → "Mocked answer"
    assert: result.answer == "Mocked answer"

test_generate_sources_populated
    mock: chain.invoke → "answer"
    assert: result.sources == sample_chunks

test_generate_model_name_in_response
    mock: chain.invoke → "answer"
    assert: result.model == "gpt-3.5-turbo"

test_generate_chain_invoked_once
    mock: chain
    assert: mock_chain.invoke.call_count == 1

test_generate_chain_receives_context
    mock: chain (capture call_args)
    assert: "context" in call_args[0][0]

test_generate_chain_receives_question
    mock: chain (capture call_args)
    assert: call_args[0][0]["question"] == question

test_generate_gpt4_model
    input: model_name="gpt-4"
    mock: chain.invoke → "answer"
    assert: isinstance(result, LLMResponse) and result.answer != ""
```

### Group E — `generate()` Error Paths (8 tests)

```
test_generate_empty_question_raises
    input: question=""
    assert: raises ValueError (before chain is called)

test_generate_whitespace_question_raises
    input: question="   "
    assert: raises ValueError

test_generate_empty_chunks_raises
    input: chunks=[]
    assert: raises ValueError

test_generate_propagates_auth_error
    mock: chain.invoke.side_effect = openai.AuthenticationError(...)
    assert: raises openai.AuthenticationError

test_generate_propagates_rate_limit_error
    mock: chain.invoke.side_effect = openai.RateLimitError(...)
    assert: raises openai.RateLimitError

test_generate_very_long_question
    input: question="A" * 2000, sample_chunks
    mock: chain.invoke → "answer"
    assert: isinstance(result, LLMResponse)

test_generate_single_chunk
    input: chunks=[single_chunk]
    mock: chain.invoke → "answer"
    assert: len(result.sources) == 1

test_generate_special_characters_in_question
    input: question='What is "net 30"?\n{braces}'
    mock: chain.invoke → "answer"
    assert: isinstance(result, LLMResponse)   — no KeyError / format error
```

### Group F — `LLMResponse` Dataclass (3 tests)

```
test_llm_response_default_fields
    input: LLMResponse(answer="test")
    assert: sources==[], model=="", prompt_tokens==0, completion_tokens==0, total_tokens==0

test_llm_response_all_fields
    input: LLMResponse(answer="a", sources=[doc], model="gpt-4", prompt_tokens=10,
                        completion_tokens=5, total_tokens=15)
    assert: all fields match

test_llm_response_is_dataclass
    assert: dataclasses.is_dataclass(LLMResponse) is True
```

### Group G — Prompt Template Constants (3 tests)

```
test_rag_prompt_template_has_context_placeholder
    assert: "{context}" in RAG_PROMPT_TEMPLATE

test_rag_prompt_template_has_question_placeholder
    assert: "{question}" in RAG_PROMPT_TEMPLATE

test_rag_prompt_template_fallback_instruction
    assert: "I don't have enough information" in RAG_PROMPT_TEMPLATE
```

### Group H — Chain Construction (2 tests)

```
test_build_chain_returns_runnable
    patch: src.llm.ChatOpenAI
    assert: client._build_chain() is not None

test_build_chain_called_once
    patch: src.llm.ChatOpenAI
    spy: client._build_chain
    action: call generate() twice
    assert: _build_chain called exactly once (cached)
```

---

## `tests/test_loader.py` — 13 Tests

```
test_load_raises_if_dir_missing
    input: docs_dir="/nonexistent/path"
    assert: raises FileNotFoundError

test_load_raises_if_no_supported_files
    setup: tmp_path with only a .txt file
    assert: raises ValueError

test_load_pdf_dispatches_pypdf_loader
    setup: tmp_path with fake.pdf
    patch: src.loader.PyPDFLoader → returns [sample_document]
    assert: PyPDFLoader instantiated with the pdf path

test_load_docx_dispatches_docx2txt_loader
    setup: tmp_path with fake.docx
    patch: src.loader.Docx2txtLoader → returns [sample_document]
    assert: Docx2txtLoader instantiated with the docx path

test_load_csv_dispatches_csv_loader
    setup: tmp_path with fake.csv (one header row + one data row)
    patch: src.loader.CSVLoader → returns [sample_document]
    assert: CSVLoader instantiated with the csv path

test_load_json_dispatches_json_loader
    setup: tmp_path with fake.json ({"key": "value"})
    patch: src.loader.JSONLoader → returns [sample_document]
    assert: JSONLoader instantiated with jq_schema="."

test_load_returns_document_list
    setup: tmp_path with one file of each type
    patch: all four loaders → return [sample_document] each
    assert: result is List[Document] with len == 4

test_load_aggregates_multi_page_pdf
    patch: PyPDFLoader.load() → returns [doc1, doc2, doc3]
    assert: len(result) == 3

test_split_produces_smaller_chunks
    input: [Document(page_content="x" * 5000)], chunk_size=500
    (real RecursiveCharacterTextSplitter — no mock)
    assert: len(chunks) >= 10

test_split_chunk_overlap_metadata
    input: long document
    assert: each chunk has "chunk_index" in metadata

test_load_and_split_chains_both_steps
    spy: loader.load, loader.split
    call: loader.load_and_split()
    assert: both load and split were called

test_split_empty_input_returns_empty
    input: documents=[]
    assert: result == []

test_source_metadata_preserved
    patch: loader to return doc with source metadata
    call: load_and_split()
    assert: "source" in chunk.metadata for all chunks
```

---

## `tests/test_embedder.py` — 12 Tests

```
test_build_calls_faiss_from_documents
    patch: src.embedder.FAISS, src.embedder.OpenAIEmbeddings
    assert: FAISS.from_documents called once with (sample_chunks, embeddings)

test_build_raises_on_empty_chunks
    patch: src.embedder.FAISS
    input: chunks=[]
    assert: raises ValueError

test_build_returns_faiss_instance
    patch: src.embedder.FAISS → .from_documents returns mock_store
    assert: result is mock_store

test_save_calls_save_local
    fixture: mock_faiss_store
    call: manager.save(mock_faiss_store)
    assert: mock_faiss_store.save_local.called

test_save_uses_correct_path
    fixture: mock_faiss_store
    input: vector_store_path="./my_store"
    assert: mock_faiss_store.save_local called with "./my_store"

test_load_calls_faiss_load_local
    patch: src.embedder.FAISS
    call: manager.load()
    assert: FAISS.load_local called once

test_load_passes_allow_dangerous
    patch: src.embedder.FAISS (capture call_args)
    assert: call_args includes allow_dangerous_deserialization=True

test_load_raises_if_path_missing
    setup: vector_store_path=tmp_path (no index files exist)
    patch: FAISS.load_local → raises FileNotFoundError
    assert: raises FileNotFoundError

test_build_and_save_calls_both
    spy: manager.build, manager.save
    call: manager.build_and_save(sample_chunks)
    assert: build called with sample_chunks, save called once

test_similarity_search_delegates_to_faiss
    fixture: mock_faiss_store
    call: manager.similarity_search("query", mock_faiss_store, top_k=3)
    assert: mock_faiss_store.similarity_search("query", k=3) called

test_similarity_search_returns_documents
    fixture: mock_faiss_store (returns sample_chunks[:2])
    assert: result == sample_chunks[:2]

test_embeddings_model_name_configured
    patch: src.embedder.OpenAIEmbeddings (capture call_args)
    input: embedding_model="text-embedding-ada-002"
    assert: OpenAIEmbeddings called with model="text-embedding-ada-002"
```

---

## `tests/test_rag.py` — 15 Tests

```
test_build_index_calls_load_and_split
    patch: src.rag.DocumentLoader → mock with .load_and_split() returning chunks
    call: pipeline.build_index()
    assert: DocumentLoader.load_and_split called once

test_build_index_calls_vsm_build
    patch: src.rag.DocumentLoader, src.rag.VectorStoreManager
    assert: VectorStoreManager().build(chunks) called

test_build_index_calls_vsm_save
    patch: src.rag.VectorStoreManager
    assert: VectorStoreManager().save(vector_store) called

test_ask_loads_index_if_not_cached
    patch: src.rag.VectorStoreManager (fresh pipeline, not pre-loaded)
    call: pipeline.ask("question")
    assert: VectorStoreManager().load() called once

test_ask_does_not_reload_if_cached
    patch: src.rag.VectorStoreManager
    call: pipeline.ask("q1"), pipeline.ask("q2")
    assert: VectorStoreManager().load() called exactly once total

test_ask_calls_similarity_search
    patch: src.rag.VectorStoreManager
    call: pipeline.ask("What are payment terms?")
    assert: similarity_search called with "What are payment terms?"

test_ask_passes_top_k
    input: top_k=6
    patch: src.rag.VectorStoreManager (capture call_args)
    assert: similarity_search called with top_k=6

test_ask_calls_llm_generate
    patch: src.rag.LLMClient
    assert: LLMClient().generate(question, chunks) called

test_ask_returns_rag_response
    patch: all three collaborators
    assert: isinstance(result, RAGResponse)

test_ask_response_contains_answer
    mock: LLMClient().generate returns LLMResponse(answer="42")
    assert: result.answer == "42"

test_ask_response_contains_sources
    mock: similarity_search returns [doc1, doc2]
    assert: result.sources == [doc1, doc2]

test_ask_empty_question_raises
    (delegates to LLMClient — test that ValueError propagates)
    mock: LLMClient().generate.side_effect = ValueError("empty question")
    assert: raises ValueError

test_is_index_loaded_false_initially
    call: RAGPipeline(...)  — no ask() yet
    assert: pipeline.is_index_loaded is False

test_is_index_loaded_true_after_ask
    patch: all collaborators
    call: pipeline.ask("q")
    assert: pipeline.is_index_loaded is True

test_rag_response_dataclass
    assert: dataclasses.is_dataclass(RAGResponse) is True
```

---

## Running the Tests

```bash
# Full suite with coverage
pytest

# LLM tests only (primary deliverable)
pytest tests/test_llm.py -v

# Single test by ID
pytest tests/test_llm.py::test_generate_propagates_auth_error -v

# Coverage for a specific module
pytest --cov=src.llm --cov-report=term-missing tests/test_llm.py

# Exclude slow tests
pytest -m "not slow"

# Show all print output (useful for debugging)
pytest -s tests/test_llm.py
```

## Coverage Targets

| Module | Minimum Line Coverage |
|---|---|
| `src/llm.py` | 90% |
| `src/loader.py` | 80% |
| `src/embedder.py` | 80% |
| `src/rag.py` | 80% |
| **Overall `src/`** | **80%** |
