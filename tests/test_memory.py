"""Tests for rag.memory — ConversationMemory."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from rag.memory import ConversationMemory


class TestConversationMemory:
    def test_creates_new_session_on_first_access(self):
        memory = ConversationMemory()
        history = memory.get_session_history("session-1")

        assert history is not None

    def test_returns_same_history_for_same_session(self):
        memory = ConversationMemory()
        h1 = memory.get_session_history("session-1")
        h2 = memory.get_session_history("session-1")

        assert h1 is h2

    def test_returns_different_history_for_different_sessions(self):
        memory = ConversationMemory()
        h1 = memory.get_session_history("session-1")
        h2 = memory.get_session_history("session-2")

        assert h1 is not h2

    def test_messages_are_persisted_within_session(self):
        memory = ConversationMemory()
        history = memory.get_session_history("session-1")
        history.add_user_message("Hello")
        history.add_ai_message("Hi there!")

        retrieved = memory.get_session_history("session-1")
        assert len(retrieved.messages) == 2
        assert isinstance(retrieved.messages[0], HumanMessage)
        assert isinstance(retrieved.messages[1], AIMessage)

    def test_clear_session_removes_history(self):
        memory = ConversationMemory()
        history = memory.get_session_history("session-1")
        history.add_user_message("Hello")

        memory.clear_session("session-1")

        assert "session-1" not in memory.list_sessions()

    def test_clear_nonexistent_session_is_a_noop(self):
        memory = ConversationMemory()
        memory.clear_session("does-not-exist")  # should not raise

    def test_list_sessions_returns_all_active_sessions(self):
        memory = ConversationMemory()
        memory.get_session_history("alpha")
        memory.get_session_history("beta")

        sessions = memory.list_sessions()

        assert "alpha" in sessions
        assert "beta" in sessions

    def test_list_sessions_returns_sorted(self):
        memory = ConversationMemory()
        memory.get_session_history("z-session")
        memory.get_session_history("a-session")

        sessions = memory.list_sessions()

        assert sessions == sorted(sessions)

    def test_get_message_count_zero_for_new_session(self):
        memory = ConversationMemory()
        assert memory.get_message_count("nonexistent") == 0

    def test_get_message_count_increases_with_messages(self):
        memory = ConversationMemory()
        history = memory.get_session_history("s1")
        history.add_user_message("Q1")
        history.add_ai_message("A1")
        history.add_user_message("Q2")

        assert memory.get_message_count("s1") == 3

    def test_clear_session_resets_message_count(self):
        memory = ConversationMemory()
        history = memory.get_session_history("s1")
        history.add_user_message("Q")
        memory.clear_session("s1")

        assert memory.get_message_count("s1") == 0

    def test_multiple_sessions_are_isolated(self):
        memory = ConversationMemory()
        h1 = memory.get_session_history("user-a")
        h2 = memory.get_session_history("user-b")

        h1.add_user_message("message for user-a")

        assert memory.get_message_count("user-a") == 1
        assert memory.get_message_count("user-b") == 0
