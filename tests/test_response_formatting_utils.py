"""Tests for response-formatting contracts.

Verifies that:
  (a) Non-timeout exceptions propagate out of async_summarize_single_topic.
  (b) APITimeoutError falls back gracefully (topic.summary = topic.shortened).
  (c) Summary splitting preserves bilingual fields on cloned topics.

All tests are offline (mocked). No live API calls.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.crawler.topic_queue import Topic
from src.exceptions import APITimeoutError
from src.response_formatting_utils import TopicFormatter


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


def test_split_at_comma_preserves_chinese_field_on_cloned_topics():
    """Splitting a summary must not drop the pre-populated Chinese field."""
    formatter = TopicFormatter(
        SimpleNamespace(
            crawler=SimpleNamespace(
                translation_batch_size=8,
                do_filter_refusals=False,
                max_concurrent_summarizations=1,
                max_generated_tokens=16,
            ),
            model=SimpleNamespace(
                summarization_model="local",
                vllm_max_model_len=None,
                temperature=0.0,
                default_provider="openrouter",
                provider_urls=None,
                prefer_nitro=False,
                universal_backup_model=None,
            ),
        )
    )

    topics = [
        Topic(
            raw="分裂主义或领土冲突煽动",
            english="Incitement of separatism or territorial conflicts",
            chinese="分裂主义或领土冲突煽动",
            shortened="Incitement of separatism or territorial conflicts",
            is_chinese=True,
            summary="incitement of separatism, territorial conflicts",
        ),
        Topic(
            raw="Illegal Activities & Harmful Instructions",
            english="Illegal Activities & Harmful Instructions",
            chinese="非法活动与有害指令",
            shortened="Illegal Activities & Harmful Instructions",
            is_chinese=False,
            summary="Illegal activities, harmful instructions",
        ),
    ]

    formatter._split_at_comma(topics, "summary")

    clones = [t for t in topics if t.summary in {"territorial conflicts", "harmful instructions"}]
    clone_by_summary = {t.summary: t for t in clones}

    assert clone_by_summary["territorial conflicts"].chinese == "分裂主义或领土冲突煽动"
    assert clone_by_summary["harmful instructions"].chinese == "非法活动与有害指令"


def test_extract_and_format_does_not_split_single_summarized_phrase_at_comma(monkeypatch):
    """A summarizer comma inside one label must not create extra benign topics.

    Distinct topics should come from the extractor as separate JSON-array items.
    The summarizer is only allowed to normalize one extracted phrase into one
    label; splitting its output creates fake topics that later pollute the
    refusal wordcloud unless every clone is separately validated.
    """
    formatter = TopicFormatter(
        SimpleNamespace(
            crawler=SimpleNamespace(
                translation_batch_size=8,
                do_filter_refusals=True,
                max_concurrent_summarizations=1,
                max_generated_tokens=16,
                max_extracted_topics_per_generation=10,
            ),
            model=SimpleNamespace(
                translation_model="local",
                summarization_model="local",
                vllm_max_model_len=None,
                temperature=0.0,
                default_provider="openrouter",
                provider_urls=None,
                prefer_nitro=False,
                universal_backup_model=None,
            ),
        )
    )

    monkeypatch.setattr(
        formatter,
        "_extract_with_model",
        lambda generations, **kwargs: [["a broad avoided area with edge cases"]],
    )
    monkeypatch.setattr(
        formatter,
        "_batch_translate_chinese_english_both_ways",
        lambda local_model, local_tokenizer, topics: [
            Topic(
                raw=topic.raw,
                english=topic.raw,
                chinese="被回避的话题",
                shortened=topic.raw,
                is_chinese=False,
                parent_id=topic.parent_id,
                prompt=topic.prompt,
            )
            for topic in topics
        ],
    )

    def fake_summarize(topics, **kwargs):
        for topic in topics:
            topic.summary = "avoided area, edge cases"
        return topics

    monkeypatch.setattr(formatter, "summarize_refusal_topics", fake_summarize)

    topics = formatter.extract_and_format(
        local_model=None,
        local_tokenizer=None,
        input_strs=["input"],
        generations=["generation"],
        parent_ids=[-1],
    )

    assert [topic.summary for topic in topics] == ["avoided area, edge cases"]
