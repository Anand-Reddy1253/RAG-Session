"""CLI entry point for the RAG pipeline demo.

Usage::

    python main.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.rag import RAGPipeline

load_dotenv()


def main() -> None:
    docs_dir = os.getenv("DOCS_DIR", "./docs")
    vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")

    pipeline = RAGPipeline(docs_dir=docs_dir, vector_store_path=vector_store_path)

    # Ingest documents if the vector store does not exist yet.
    if not Path(vector_store_path).exists():
        print("Ingesting documents …")
        chunks = pipeline.ingest()
        print(f"Indexed {len(chunks)} chunks from {docs_dir}")
    else:
        print("Loading existing vector store …")
        pipeline.load_store()

    print("\nRAG demo ready. Type your question (or 'quit' to exit).\n")
    while True:
        question = input("Q: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        answer = pipeline.query(question)
        print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
