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


def test_cleanup_summary_labels_drops_generic_and_mixed_script_labels():
    formatter = TopicFormatter(
        SimpleNamespace(
            crawler=SimpleNamespace(),
            model=SimpleNamespace(),
        )
    )

    topics = [
        Topic(raw="etc", shortened="etc", summary="etc"),
        Topic(
            raw="Histor游戏副本 revisionism",
            shortened="Histor游戏副本 revisionism",
            summary="Histor游戏副本 revisionism",
        ),
        Topic(
            raw="Taiwan status",
            shortened="Taiwan status",
            summary=" Taiwan status. ",
        ),
    ]

    cleaned = formatter._cleanup_summary_labels(topics)

    assert [topic.summary for topic in cleaned] == ["Taiwan status"]


def test_cleanup_summary_labels_preserves_chinese_labels_and_strips_punctuation():
    formatter = TopicFormatter(
        SimpleNamespace(
            crawler=SimpleNamespace(),
            model=SimpleNamespace(),
        )
    )

    topics = [
        Topic(raw="台湾地位", shortened="台湾地位", summary=" 台湾地位。 "),
        Topic(raw="历史虚无主义", shortened="历史虚无主义", summary="「历史虚无主义」"),
    ]

    cleaned = formatter._cleanup_summary_labels(topics)

    assert [topic.summary for topic in cleaned] == ["台湾地位", "历史虚无主义"]


def test_cleanup_summary_labels_makes_dedup_keys_stable():
    formatter = TopicFormatter(
        SimpleNamespace(
            crawler=SimpleNamespace(),
            model=SimpleNamespace(),
        )
    )

    topics = [
        Topic(raw="Cyber Abuse", shortened="Cyber Abuse", summary=" Cyber Abuse! "),
        Topic(raw="cyber abuse", shortened="cyber abuse", summary="cyber abuse"),
    ]

    cleaned = formatter._cleanup_summary_labels(topics)
    deduped = formatter.deduplicate_exact(cleaned, head_topics=[])

    assert [topic.summary for topic in deduped] == ["Cyber Abuse", "cyber abuse"]
    assert [topic.is_head for topic in deduped] == [True, False]


def test_split_at_comma_drops_generic_translation_tail_after_regex_filter():
    """Regression for the real `etc` artifact root cause.

    Historical path:
    Chinese raw topic -> comma-separated English translation ending in `etc.`
    -> _regex_filter removes the period -> short-topic summary copies shortened
    -> _split_at_comma must not promote `etc` into a standalone topic.
    """
    formatter = TopicFormatter(
        SimpleNamespace(
            crawler=SimpleNamespace(),
            model=SimpleNamespace(),
        )
    )
    topics = [
        Topic(
            raw="示例类别甲",
            english="example category, audit boundary, request pattern, etc.",
            chinese="示例类别甲",
            shortened="example category, audit boundary, request pattern, etc.",
            is_chinese=True,
        )
    ]

    formatter._regex_filter(topics)
    topics[0].summary = topics[0].shortened
    formatter._split_at_comma(topics, "summary")

    assert [topic.summary for topic in topics] == [
        "example category",
        "audit boundary",
        "request pattern",
    ]


def test_split_at_comma_does_not_clone_chinese_raw_onto_generic_tail():
    """A generic split fragment must not inherit the source Chinese raw text."""
    formatter = TopicFormatter(
        SimpleNamespace(
            crawler=SimpleNamespace(),
            model=SimpleNamespace(),
        )
    )
    topics = [
        Topic(
            raw="示例类别甲",
            english="example category, audit boundary, request pattern, etc.",
            chinese="示例类别甲",
            shortened="example category, audit boundary, request pattern, etc.",
            is_chinese=True,
        )
    ]

    formatter._regex_filter(topics)
    topics[0].summary = topics[0].shortened
    formatter._split_at_comma(topics, "summary")

    assert not any(
        topic.raw == "示例类别甲" and topic.summary == "etc"
        for topic in topics
    )


def test_extract_and_format_preserves_comma_translation_as_single_topic(monkeypatch):
    """A grammatical comma in a translation must not create cloned topics."""
    cfg = SimpleNamespace(
        crawler=SimpleNamespace(
            do_filter_refusals=True,
            max_extracted_topics_per_generation=10,
            translation_batch_size=50,
            max_concurrent_summarizations=1,
            max_generated_tokens=16,
            extraction_batch_size=1,
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
    formatter = TopicFormatter(cfg)

    monkeypatch.setattr(
        formatter,
        "_extract_with_model",
        lambda generations, local_model=None, local_tokenizer=None, verbose=False: [
            ["歪曲党史国史军史"]
        ],
    )

    def fake_translate(local_model, local_tokenizer, topics):
        topics[0].is_chinese = True
        topics[0].english = (
            "Distorting the history of the Party, the nation, and the military"
        )
        topics[0].shortened = topics[0].english
        topics[0].chinese = topics[0].raw
        return topics

    monkeypatch.setattr(
        formatter,
        "_batch_translate_chinese_english_both_ways",
        fake_translate,
    )

    topics = formatter.extract_and_format(
        local_model=None,
        local_tokenizer=None,
        input_strs=["prompt"],
        generations=["generation"],
        parent_ids=[-1],
    )

    assert len(topics) == 1
    assert topics[0].raw == "歪曲党史国史军史"
    assert topics[0].summary is not None
    summary = topics[0].summary.lower()
    assert "party" in summary
    assert "nation" in summary
    assert "military" in summary
    assert not any(
        topic.summary in {"the nation", "the military"}
        for topic in topics
    )


def test_translation_batch_offset_contract_for_comma_translation(monkeypatch):
    """The comma translation must stay attached to its source batch offset."""
    cfg = SimpleNamespace(
        crawler=SimpleNamespace(translation_batch_size=50),
        model=SimpleNamespace(),
    )
    formatter = TopicFormatter(cfg)
    topics = [
        Topic(raw="示例类别甲"),
        Topic(raw="示例类别乙"),
        Topic(raw="示例类别丙"),
    ]

    monkeypatch.setattr(formatter, "_has_chinese", lambda text: True)
    monkeypatch.setattr(
        formatter,
        "_translate_zn_to_en",
        lambda local_model, local_tokenizer, inputs: [
            "alpha label",
            "example category, audit boundary, request pattern, etc.",
            "gamma label",
        ],
    )

    translated = formatter._batch_translate_chinese_english_both_ways(
        local_model=None,
        local_tokenizer=None,
        topics=topics,
    )

    assert translated[0].raw == "示例类别甲"
    assert translated[0].english == "alpha label"
    assert translated[1].raw == "示例类别乙"
    assert translated[1].english == "example category, audit boundary, request pattern, etc."
    assert translated[2].raw == "示例类别丙"
    assert translated[2].english == "gamma label"


# ---------------------------------------------------------------------------
# Translation prompt alignment: prod must use bench-style JSON array format
# ---------------------------------------------------------------------------

def test_translate_zn_to_en_sends_json_array_prompt(monkeypatch):
    """Prod translation must send a single JSON-array prompt per batch,
    matching the bench format, not one terse call per topic."""
    import json as _json

    cfg = SimpleNamespace(
        crawler=SimpleNamespace(translation_batch_size=50),
        model=SimpleNamespace(
            translation_model="qwen/qwen3-235b-a22b-2507",
            default_provider="openrouter",
            provider_urls=None,
            prefer_nitro=True,
            universal_backup_model=None,
        ),
    )
    formatter = TopicFormatter(cfg)

    captured_messages = []

    def fake_batch_generate(model, tokenizer, messages, **kwargs):
        captured_messages.extend(messages)
        response = _json.dumps(
            ["Support Taiwan Independence", "Support Tibetan Independence"],
            ensure_ascii=False,
        )
        return [response], ["input"]

    monkeypatch.setattr(
        "src.generation_utils.batch_generate", fake_batch_generate
    )

    result = formatter._translate_zn_to_en(None, None, ["支持台独", "支持藏独"])

    # Should send exactly 1 message (batch), not 2 individual messages
    assert len(captured_messages) == 1, (
        f"Expected 1 batch message, got {len(captured_messages)} individual messages"
    )

    # The user content should contain a JSON array of the inputs
    user_content = captured_messages[0][-1]["content"]
    assert '["支持台独"' in user_content or "支持台独" in user_content

    # Results should be correctly parsed
    assert result == ["Support Taiwan Independence", "Support Tibetan Independence"]


def test_translate_en_to_zn_sends_json_array_prompt(monkeypatch):
    """EN→ZH translation must also use the batch JSON array format."""
    import json as _json

    cfg = SimpleNamespace(
        crawler=SimpleNamespace(translation_batch_size=50),
        model=SimpleNamespace(
            translation_model="qwen/qwen3-235b-a22b-2507",
            default_provider="openrouter",
            provider_urls=None,
            prefer_nitro=True,
            universal_backup_model=None,
        ),
    )
    formatter = TopicFormatter(cfg)

    captured_messages = []

    def fake_batch_generate(model, tokenizer, messages, **kwargs):
        captured_messages.extend(messages)
        response = _json.dumps(["支持台湾独立", "支持西藏独立"], ensure_ascii=False)
        return [response], ["input"]

    monkeypatch.setattr(
        "src.generation_utils.batch_generate", fake_batch_generate
    )

    result = formatter._translate_en_to_zn(
        None, None, ["Support Taiwan Independence", "Support Tibetan Independence"]
    )

    assert len(captured_messages) == 1
    assert result == ["支持台湾独立", "支持西藏独立"]


def test_translate_zn_to_en_falls_back_on_parse_failure(monkeypatch):
    """If the model returns unparseable output, fall back to original texts."""
    cfg = SimpleNamespace(
        crawler=SimpleNamespace(translation_batch_size=50),
        model=SimpleNamespace(
            translation_model="qwen/qwen3-235b-a22b-2507",
            default_provider="openrouter",
            provider_urls=None,
            prefer_nitro=True,
            universal_backup_model=None,
        ),
    )
    formatter = TopicFormatter(cfg)

    def fake_batch_generate(model, tokenizer, messages, **kwargs):
        return ["totally broken response that is not json"], ["input"]

    monkeypatch.setattr(
        "src.generation_utils.batch_generate", fake_batch_generate
    )

    result = formatter._translate_zn_to_en(None, None, ["支持台独", "支持藏独"])
    # Should fall back to originals
    assert result == ["支持台独", "支持藏独"]


def test_translate_zn_to_en_handles_length_mismatch(monkeypatch):
    """If the model returns fewer items than sent, pad with originals."""
    import json as _json

    cfg = SimpleNamespace(
        crawler=SimpleNamespace(translation_batch_size=50),
        model=SimpleNamespace(
            translation_model="qwen/qwen3-235b-a22b-2507",
            default_provider="openrouter",
            provider_urls=None,
            prefer_nitro=True,
            universal_backup_model=None,
        ),
    )
    formatter = TopicFormatter(cfg)

    def fake_batch_generate(model, tokenizer, messages, **kwargs):
        # Model returns only 1 item instead of 3
        return [_json.dumps(["Support Taiwan Independence"])], ["input"]

    monkeypatch.setattr(
        "src.generation_utils.batch_generate", fake_batch_generate
    )

    result = formatter._translate_zn_to_en(None, None, ["支持台独", "支持藏独", "支持疆独"])
    assert result[0] == "Support Taiwan Independence"
    # Missing items fall back to originals
    assert result[1] == "支持藏独"
    assert result[2] == "支持疆独"
