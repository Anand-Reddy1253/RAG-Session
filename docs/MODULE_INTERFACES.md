# Module Interface Quick-Reference

> Companion to `IMPLEMENTATION_PLAN.md` — concise signatures for all public APIs.

---

## `src/loader.py` — `DocumentLoader`

```python
class DocumentLoader:
    def __init__(
        self,
        docs_dir: str | Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        json_content_key: str = ".",
    ) -> None

    def load(self) -> List[Document]
    def split(self, documents: List[Document]) -> List[Document]
    def load_and_split(self) -> List[Document]

    # Private — accessible in tests via loader._load_single_file(path)
    def _discover_files(self) -> List[Path]
    def _load_single_file(self, file_path: Path) -> List[Document]
```

### Metadata guaranteed on every output Document

| Key | Type | Source |
|---|---|---|
| `source` | `str` | Absolute path to the originating file |
| `chunk_index` | `int` | Zero-based position within the file's chunks |

### File-extension → Loader mapping

| Extension | LangChain Class | Import |
|---|---|---|
| `.pdf` | `PyPDFLoader` | `langchain_community.document_loaders` |
| `.docx` | `Docx2txtLoader` | `langchain_community.document_loaders` |
| `.csv` | `CSVLoader` | `langchain_community.document_loaders` |
| `.json` | `JSONLoader` | `langchain_community.document_loaders` |

---

## `src/embedder.py` — `VectorStoreManager`

```python
class VectorStoreManager:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,       # falls back to env var
        embedding_model: str = "text-embedding-ada-002",
        vector_store_path: str | Path = "./vector_store",
    ) -> None

    def build(self, chunks: List[Document]) -> FAISS
    def save(self, vector_store: FAISS) -> None
    def load(self) -> FAISS
    def build_and_save(self, chunks: List[Document]) -> FAISS
    def similarity_search(
        self,
        query: str,
        vector_store: FAISS,
        top_k: int = 4,
    ) -> List[Document]

    @property
    def embeddings(self) -> OpenAIEmbeddings
```

### Critical LangChain call signatures

```python
# Building the index
FAISS.from_documents(chunks, embeddings_instance)

# Saving
faiss_instance.save_local(str(vector_store_path))

# Loading  ← note the required kwarg
FAISS.load_local(
    str(vector_store_path),
    embeddings_instance,
    allow_dangerous_deserialization=True,   # REQUIRED in LangChain ≥ 0.2
)

# Retrieval
faiss_instance.similarity_search(query_string, k=top_k)
```

---

## `src/llm.py` — `LLMClient` + `LLMResponse`

```python
@dataclass
class LLMResponse:
    answer: str
    sources: List[Document] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMClient:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 512,
        prompt_template: str = RAG_PROMPT_TEMPLATE,
    ) -> None

    def format_context(self, chunks: List[Document]) -> str
    def build_prompt(self, question: str, chunks: List[Document]) -> str
    def generate(self, question: str, chunks: List[Document]) -> LLMResponse

    @property
    def chat_model(self) -> ChatOpenAI

    @property
    def model_name(self) -> str

    # Private
    def _build_chain(self) -> Runnable
```

### LCEL Chain

```
ChatPromptTemplate.from_template(prompt_template)
    | ChatOpenAI(model_name=..., temperature=..., max_tokens=...)
    | StrOutputParser()
```

Invoked as: `chain.invoke({"context": context_str, "question": question})`

### Prompt Template Placeholders

| Placeholder | Provided by | Description |
|---|---|---|
| `{context}` | `format_context(chunks)` | Formatted chunk text with source headers |
| `{question}` | Caller | User's raw question string |

### Guard Conditions (raise `ValueError`)

- `question` is empty string or whitespace-only
- `chunks` is an empty list

---

## `src/rag.py` — `RAGPipeline` + `RAGResponse`

```python
@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: List[Document] = field(default_factory=list)
    model: str = ""
    total_tokens: int = 0


class RAGPipeline:
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
    ) -> None

    def build_index(self) -> None
    def ask(self, question: str) -> RAGResponse

    @property
    def is_index_loaded(self) -> bool

    # Private
    def _ensure_index_loaded(self) -> FAISS
```

### Orchestration Sequence

**`build_index()`**
```
DocumentLoader.load_and_split()
    → VectorStoreManager.build(chunks)
        → VectorStoreManager.save(vector_store)
```

**`ask(question)`**
```
_ensure_index_loaded()              # loads from disk if not cached
    → VectorStoreManager.similarity_search(question, top_k)
        → LLMClient.generate(question, chunks)
            → RAGResponse(...)
```

---

## `main.py` — CLI Entry Point

```python
def parse_args() -> argparse.Namespace
    # --build         : re-index before querying
    # --question TEXT : single-shot mode
    # --top-k INT     : retrieval depth (default 4)
    # --model TEXT    : chat model override

def print_response(response: RAGResponse) -> None
def interactive_loop(pipeline: RAGPipeline) -> None
def main() -> None
```

### Environment Variables (loaded from `.env`)

| Variable | Default | Used by |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | `LLMClient`, `VectorStoreManager` |
| `DOCS_DIR` | `./docs` | `DocumentLoader` |
| `VECTOR_STORE_PATH` | `./vector_store` | `VectorStoreManager` |
