"""Interactive CLI for the RAG chatbot with conversation memory.

Usage::

    python main.py

Environment variables (see ``.env.example``):

* ``OPENAI_API_KEY`` — Required. Your OpenAI API key.
* ``DOCS_DIR``       — Directory with knowledge-base documents (default: ./docs).
* ``VECTOR_STORE_PATH`` — Where to persist/load the FAISS index (default: ./vector_store).
* ``LLM_MODEL``     — Chat model to use (default: gpt-3.5-turbo).
* ``EMBEDDING_MODEL`` — Embedding model (default: text-embedding-ada-002).
* ``CHUNK_SIZE``    — Characters per chunk (default: 1000).
* ``CHUNK_OVERLAP`` — Overlap between chunks (default: 200).
* ``RETRIEVER_K``   — Number of chunks to retrieve per query (default: 4).
"""

import os
import sys

from langchain_openai import ChatOpenAI

import config
from rag.embedder import build_embedder
from rag.loader import load_documents, split_documents
from rag.memory import ConversationMemory
from rag.chain import build_rag_chain
from rag.vector_store import build_vector_store, load_vector_store


def _build_retriever():
    """Return a FAISS retriever, loading from disk or building from docs."""
    embeddings = build_embedder(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY or None,
    )

    if os.path.exists(config.VECTOR_STORE_PATH):
        print(f"Loading vector store from '{config.VECTOR_STORE_PATH}' …")
        vector_store = load_vector_store(config.VECTOR_STORE_PATH, embeddings)
    else:
        print(f"Building vector store from '{config.DOCS_DIR}' …")
        documents = load_documents(config.DOCS_DIR)
        chunks = split_documents(documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        print(f"  Loaded {len(documents)} document(s), {len(chunks)} chunk(s).")
        vector_store = build_vector_store(chunks, embeddings, config.VECTOR_STORE_PATH)
        print(f"  Vector store saved to '{config.VECTOR_STORE_PATH}'.")

    return vector_store.as_retriever(search_kwargs={"k": config.RETRIEVER_K})


def main() -> None:
    """Run the interactive RAG chatbot loop."""
    if not config.OPENAI_API_KEY:
        print(
            "Error: OPENAI_API_KEY is not set. "
            "Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        sys.exit(1)

    retriever = _build_retriever()
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0,
    )
    memory = ConversationMemory()
    chain = build_rag_chain(llm, retriever, memory)

    session_id = "default"
    print("\nRAG Chatbot with Memory — type 'quit' to exit, 'clear' to reset history.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            memory.clear_session(session_id)
            print("Conversation history cleared.\n")
            continue

        response = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"Bot: {response['answer']}\n")


if __name__ == "__main__":
    main()
