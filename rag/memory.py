"""Conversation memory management for multi-turn RAG sessions.

Each chat session is identified by a *session_id* string. The session's
message history is stored in memory (no external persistence) and can be
cleared or inspected at any time.
"""

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


class ConversationMemory:
    """Manages per-session conversation history for the RAG chain.

    The memory store is an in-process dict keyed by *session_id*. A new
    :class:`~langchain_community.chat_message_histories.ChatMessageHistory`
    is created automatically the first time a session ID is seen.

    Example::

        memory = ConversationMemory()
        history = memory.get_session_history("user-123")
        history.add_user_message("What is the payment term?")
        history.add_ai_message("Payment is due within 30 days.")
    """

    def __init__(self) -> None:
        self._store: dict[str, ChatMessageHistory] = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Return (or create) the message history for *session_id*.

        This method is compatible with LangChain's
        :class:`~langchain_core.runnables.history.RunnableWithMessageHistory`
        ``get_session_history`` callback.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            The :class:`~langchain_core.chat_history.BaseChatMessageHistory`
            associated with *session_id*.
        """
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]

    def clear_session(self, session_id: str) -> None:
        """Remove the message history for *session_id*.

        Args:
            session_id: Session whose history should be cleared.
        """
        self._store.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        """Return all currently active session IDs.

        Returns:
            Sorted list of session ID strings.
        """
        return sorted(self._store.keys())

    def get_message_count(self, session_id: str) -> int:
        """Return the number of messages stored for *session_id*.

        Args:
            session_id: Session to inspect.

        Returns:
            Number of messages (0 if the session does not exist).
        """
        if session_id not in self._store:
            return 0
        return len(self._store[session_id].messages)
