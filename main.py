"""CLI entry point for the RAG demo."""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from src.rag import RAGPipeline, RAGResponse

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Session — ask questions about your docs")
    parser.add_argument("--build", action="store_true", help="Re-index docs before querying")
    parser.add_argument("--question", "-q", type=str, help="Single-shot question mode")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI chat model to use",
    )
    return parser.parse_args()


def print_response(response: RAGResponse) -> None:
    print("\n" + "=" * 60)
    print(f"Q: {response.question}")
    print("-" * 60)
    print(f"A: {response.answer}")
    if response.sources:
        print("\nSources:")
        seen: set[str] = set()
        for doc in response.sources:
            src = doc.metadata.get("source", "unknown")
            if src not in seen:
                print(f"  • {src}")
                seen.add(src)
    print("=" * 60 + "\n")


def interactive_loop(pipeline: RAGPipeline) -> None:
    print("RAG Session — type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if question.lower() in {"quit", "exit", ""}:
            print("Bye!")
            break
        try:
            response = pipeline.ask(question)
            print_response(response)
        except ValueError as exc:
            print(f"[Error] {exc}\n")


def main() -> None:
    args = parse_args()

    pipeline = RAGPipeline(
        docs_dir=os.getenv("DOCS_DIR", "./docs"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        vector_store_path=os.getenv("VECTOR_STORE_PATH", "./vector_store"),
        chat_model=args.model,
        top_k=args.top_k,
    )

    if args.build:
        print("Building index …")
        pipeline.build_index()
        print("Index built.\n")

    if args.question:
        response = pipeline.ask(args.question)
        print_response(response)
    else:
        interactive_loop(pipeline)


if __name__ == "__main__":
    main()
