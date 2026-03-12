# RAG Session — Retrieval-Augmented Generation Demo

A hands-on demo project that showcases how to build a **Retrieval-Augmented Generation (RAG)** pipeline using enterprise documents. The system ingests internal company files, indexes them into a vector store, and lets users ask natural-language questions that are answered using the most relevant document chunks.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Sample Data](#sample-data)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Example Queries](#example-queries)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)

---

## Overview

This project demonstrates a complete RAG workflow:

1. **Load** — Read documents from the `docs/` folder (PDFs, Word files, CSVs, JSON).
2. **Split** — Chunk documents into manageable pieces for embedding.
3. **Embed** — Convert chunks into vector embeddings using an LLM embedding model.
4. **Store** — Persist embeddings in a vector database for fast similarity search.
5. **Retrieve** — Given a user query, fetch the most relevant chunks.
6. **Generate** — Pass the retrieved context plus the query to an LLM to produce a grounded answer.

---

## Project Structure

```
RAG-Session/
├── docs/                          # Sample knowledge-base documents
│   ├── ACME_Globex_Contract.docx  # Vendor contract between ACME Corp and Globex
│   ├── ACME_HR_Handbook_v4.2.pdf  # ACME Corp HR policy handbook
│   ├── acme_sales_q3.csv          # Q3 sales data for ACME Corp
│   ├── globex_crm_profile.json    # Globex CRM customer profile export
│   └── filelist.txt               # Index of available documents
├── rag/                           # Core RAG pipeline package
│   ├── __init__.py
│   ├── loader.py                  # Document loading (PDF, DOCX, CSV, JSON)
│   ├── embedder.py                # OpenAI embedding model wrapper
│   ├── vector_store.py            # FAISS vector store build / load helpers
│   ├── memory.py                  # Conversation memory (per-session history)
│   └── chain.py                   # History-aware RAG chain (LCEL + memory)
├── tests/                         # Unit tests (pytest)
│   ├── test_loader.py
│   ├── test_embedder.py
│   ├── test_memory.py
│   └── test_rag.py
├── main.py                        # Interactive CLI chatbot
├── config.py                      # Environment-based configuration
├── requirements.txt
├── pytest.ini
├── .env.example                   # Template for .env configuration
└── ReadMe.md                      # Project documentation (this file)
```

---

## Sample Data

The `docs/` directory contains fictional company documents used as the RAG knowledge base:

| File | Type | Description |
|---|---|---|
| `ACME_Globex_Contract.docx` | Word | Service agreement between ACME Corp and Globex Inc. covering SLAs and pricing |
| `ACME_HR_Handbook_v4.2.pdf` | PDF | HR policies, leave management, and code-of-conduct guidelines for ACME Corp |
| `acme_sales_q3.csv` | CSV | Quarterly sales figures broken down by region and product line |
| `globex_crm_profile.json` | JSON | Customer relationship data including contact history and deal pipeline |

> **Note:** All data is entirely fictional and created for demonstration purposes only.

---

## Getting Started

### Prerequisites

- Python 3.9+
- An OpenAI API key (or any compatible LLM provider)
- `pip` or `conda` for package management

### Installation

```bash
# Clone the repository
git clone https://github.com/Anand-Reddy1253/RAG-Session.git
cd RAG-Session

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
VECTOR_STORE_PATH=./vector_store
DOCS_DIR=./docs
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVER_K=4
```

### Run the Demo

```bash
python main.py
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## How It Works

```
User Query + Chat History
    │
    ▼
[ History-Aware Retriever ]  ← rewrites follow-up questions into standalone queries
    │
    ▼
[ Embedding Model ] ──► Query Vector
                              │
                              ▼
                     [ Vector Store (FAISS) ]
                              │
                    Top-K Relevant Chunks
                              │
                              ▼
               [ LLM (GPT-4 / GPT-3.5-turbo) ]
                              │
                        Grounded Answer
                              │
                              ▼
                  [ Conversation Memory ]  ← stores Q&A pairs per session
```

1. The query (plus any prior chat history) is sent to a **history-aware retriever** that
   rewrites follow-up questions into self-contained queries before hitting the vector store.
2. The rewritten query is converted to an embedding vector and used for cosine-similarity
   search, returning the *k* most relevant document chunks.
3. The retrieved chunks, the original query, and the full conversation history are injected
   into a prompt and passed to the LLM.
4. The LLM generates a factual, grounded answer.
5. Both the user message and the AI response are stored in **per-session conversation memory**
   so that follow-up questions have full context.

### Conversation Memory

The memory subsystem (`rag/memory.py`) keeps a separate
`ChatMessageHistory` for every *session ID*. This allows multiple independent
conversations to coexist without interfering with each other.

| Operation | Method |
|---|---|
| Access session history | `memory.get_session_history(session_id)` |
| Clear a session | `memory.clear_session(session_id)` |
| List active sessions | `memory.list_sessions()` |
| Count messages | `memory.get_message_count(session_id)` |

During a conversation the CLI automatically appends each turn to the history.
Type **`clear`** at the prompt to wipe the history for the current session.

---

## Example Queries

```
Q: What are the payment terms in the ACME-Globex contract?
A: According to the contract, payment is due within 30 days of invoice...

Q: How many vacation days do ACME employees get per year?
A: Per the HR handbook (Section 4.2), full-time employees receive 15 days...

Q: What was ACME's best-performing region in Q3?
A: Based on the sales data, the Western region led with $2.4 M in revenue...
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| LLM | OpenAI GPT-4 / GPT-3.5-turbo |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Vector Store | FAISS / ChromaDB |
| Document Loaders | LangChain document loaders |
| Orchestration | LangChain / LlamaIndex |

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

*This project is for educational and demo purposes.*
