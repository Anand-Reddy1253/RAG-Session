"""
RAG-Session entry point.

Demonstrates a minimal Retrieval-Augmented Generation loop that persists
conversation history in Redis with a 1-day TTL.
"""

import os
import uuid
from dotenv import load_dotenv

from cache import ConversationCache

load_dotenv()


def build_prompt(history: list[dict], query: str, context: str) -> str:
    """Build a simple prompt from conversation history, retrieved context, and the query."""
    lines = [
        "You are a helpful assistant. Use the context below to answer the question.",
        "",
        "### Context",
        context,
        "",
        "### Conversation so far",
    ]
    for msg in history:
        lines.append(f"{msg['role'].capitalize()}: {msg['content']}")
    lines += ["", f"User: {query}", "Assistant:"]
    return "\n".join(lines)


def run_rag_loop(session_id: str | None = None) -> None:
    """Interactive RAG loop backed by Redis conversation history."""
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))

    cache = ConversationCache(host=redis_host, port=redis_port)
    session_id = session_id or str(uuid.uuid4())
    print(f"Session ID: {session_id}")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue

        # Retrieve context (placeholder — replace with real FAISS / vector-store lookup)
        context = "[Retrieved document chunks would appear here]"

        history = cache.get_history(session_id)
        prompt = build_prompt(history, query, context)

        # Generate answer (placeholder — replace with real LLM call)
        answer = f"[LLM answer based on context for: {query!r}]"

        # Persist both turns
        cache.add_message(session_id, "user", query)
        cache.add_message(session_id, "assistant", answer)

        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    run_rag_loop()
