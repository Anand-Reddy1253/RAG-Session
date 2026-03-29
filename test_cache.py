"""Tests for the Redis-backed ConversationCache."""

import json
from unittest.mock import MagicMock, patch

import pytest

from cache import ConversationCache, CONVERSATION_TTL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_redis():
    """Patch redis.Redis so no real Redis server is needed."""
    with patch("cache.redis.Redis") as MockRedis:
        client = MagicMock()
        MockRedis.return_value = client
        yield client


@pytest.fixture
def cache(mock_redis):
    return ConversationCache(host="localhost", port=6379, db=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetHistory:
    def test_returns_empty_list_when_key_missing(self, cache, mock_redis):
        mock_redis.get.return_value = None
        assert cache.get_history("sess-1") == []

    def test_returns_parsed_history(self, cache, mock_redis):
        messages = [{"role": "user", "content": "hello"}]
        mock_redis.get.return_value = json.dumps(messages)
        assert cache.get_history("sess-1") == messages


class TestAddMessage:
    def test_appends_message_and_sets_ttl(self, cache, mock_redis):
        mock_redis.get.return_value = None  # empty history
        cache.add_message("sess-1", "user", "What is RAG?")

        expected_payload = json.dumps([{"role": "user", "content": "What is RAG?"}])
        mock_redis.setex.assert_called_once_with(
            "conversation:sess-1",
            CONVERSATION_TTL,
            expected_payload,
        )

    def test_ttl_is_one_day(self):
        assert CONVERSATION_TTL == 86_400

    def test_appends_to_existing_history(self, cache, mock_redis):
        existing = [{"role": "user", "content": "Hi"}]
        mock_redis.get.return_value = json.dumps(existing)

        cache.add_message("sess-1", "assistant", "Hello!")

        expected_payload = json.dumps([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ])
        mock_redis.setex.assert_called_once_with(
            "conversation:sess-1",
            CONVERSATION_TTL,
            expected_payload,
        )


class TestClearHistory:
    def test_deletes_key(self, cache, mock_redis):
        cache.clear_history("sess-1")
        mock_redis.delete.assert_called_once_with("conversation:sess-1")


class TestGetTTL:
    def test_returns_ttl_from_redis(self, cache, mock_redis):
        mock_redis.ttl.return_value = 80000
        assert cache.get_ttl("sess-1") == 80000
        mock_redis.ttl.assert_called_once_with("conversation:sess-1")

    def test_returns_minus_two_when_key_missing(self, cache, mock_redis):
        mock_redis.ttl.return_value = -2
        assert cache.get_ttl("nonexistent") == -2
