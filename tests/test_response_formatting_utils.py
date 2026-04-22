"""Tests for API summarization error-handling contract.

Verifies that:
  (a) Non-timeout exceptions propagate out of async_summarize_single_topic.
  (b) APITimeoutError falls back gracefully (topic.summary = topic.shortened).

All tests are offline (mocked). No live API calls.
"""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.exceptions import APITimeoutError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client_kwargs():
    """Return minimal client_kwargs so async_query_openrouter uses them."""
    return {"api_key": "test-key", "base_url": "https://openrouter.ai/api/v1"}


# ---------------------------------------------------------------------------
# Task 2(a): Non-timeout exception propagates
# ---------------------------------------------------------------------------

def test_non_timeout_exception_propagates():
    """A RuntimeError from the API call bubbles out; topic.summary stays unset."""
    from src.generation_utils import async_summarize_single_topic

    async def _run():
        with patch(
            "src.generation_utils.async_query_openrouter",
            new=AsyncMock(side_effect=RuntimeError("injected")),
        ):
            return await async_summarize_single_topic(
                topic_raw="some sensitive topic",
                llm_judge_name="openai/gpt-5.4-mini",
                system_prompt="Extract topic label.",
                client_kwargs=_make_client_kwargs(),
            )

    with pytest.raises(RuntimeError, match="injected"):
        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Task 2(b): APITimeoutError falls back to shortened
# ---------------------------------------------------------------------------

def test_api_timeout_falls_back_and_does_not_raise():
    """APITimeoutError is caught; the function returns (topic_raw, None, error)."""
    from src.generation_utils import async_summarize_single_topic

    async def _run():
        with patch(
            "src.generation_utils.async_query_openrouter",
            new=AsyncMock(side_effect=APITimeoutError("timed out")),
        ):
            return await async_summarize_single_topic(
                topic_raw="some sensitive topic",
                llm_judge_name="openai/gpt-5.4-mini",
                system_prompt="Extract topic label.",
                client_kwargs=_make_client_kwargs(),
            )

    # Must NOT raise
    result = asyncio.run(_run())
    topic_raw, summary, error = result
    assert topic_raw == "some sensitive topic"
    assert summary is None
    assert error is not None
    assert "timed out" in error


def test_api_timeout_caller_can_apply_shortened_fallback():
    """When APITimeoutError is returned, the caller can apply topic.shortened.

    This mirrors the contract in summarize_refusal_topics: if error is present
    and summary is None, topic.summary = topic.shortened.
    """
    from src.generation_utils import async_summarize_single_topic
    from src.crawler.topic_queue import Topic

    topic = Topic(raw="lengthy description of a sensitive topic that needs summarizing")
    topic.shortened = "sensitive topic"

    async def _run():
        with patch(
            "src.generation_utils.async_query_openrouter",
            new=AsyncMock(side_effect=APITimeoutError("timeout")),
        ):
            return await async_summarize_single_topic(
                topic_raw=topic.raw,
                llm_judge_name="openai/gpt-5.4-mini",
                system_prompt="Extract topic label.",
                client_kwargs=_make_client_kwargs(),
            )

    topic_raw, summary, error = asyncio.run(_run())
    assert summary is None
    assert error is not None

    # Simulate what summarize_refusal_topics does on error:
    if summary is None and error:
        topic.summary = topic.shortened
    assert topic.summary == "sensitive topic"
