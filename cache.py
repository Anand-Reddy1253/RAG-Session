"""
Redis-backed conversation history cache with a 1-day (86 400 s) TTL.

Each conversation is keyed by a session/user identifier and stores an ordered
list of {"role": ..., "content": ...} message dicts, serialised as JSON.
"""

import json
import redis

# 1 day in seconds
CONVERSATION_TTL = 86_400


class ConversationCache:
    """Manages conversation history in Redis with a 1-day TTL."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self._client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, session_id: str) -> str:
        return f"conversation:{session_id}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> list[dict]:
        """Return the full message history for *session_id* (may be empty)."""
        raw = self._client.get(self._key(session_id))
        if raw is None:
            return []
        return json.loads(raw)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message and refresh the TTL to 1 day from now."""
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})
        self._client.setex(self._key(session_id), CONVERSATION_TTL, json.dumps(history))

    def clear_history(self, session_id: str) -> None:
        """Delete the conversation history for *session_id*."""
        self._client.delete(self._key(session_id))

    def get_ttl(self, session_id: str) -> int:
        """Return seconds remaining before expiry, or -2 if the key does not exist."""
        return self._client.ttl(self._key(session_id))
