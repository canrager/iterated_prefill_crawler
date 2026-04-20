"""Unit tests for S0g — summarization exception handling.

The broad `except Exception` in both local and API summarization paths of
summarize_refusal_topics was narrowed: APITimeoutError is recoverable
(log + fallback to shortened); all other exceptions re-raise.

Tests:
- test_summarize_local_reraises_unexpected_exception
- test_summarize_local_falls_back_on_api_timeout_with_warning
- test_summarize_api_reraises_unexpected_exception
- test_summarize_api_falls_back_on_api_timeout_with_warning
"""

import logging
import pytest
from unittest.mock import patch, MagicMock

from src.crawler.config import CrawlerConfig
from src.crawler.topic_queue import Topic
from src.exceptions import APITimeoutError
from src.response_formatting_utils import TopicFormatter


def _make_topic(raw: str, shortened: str = "short") -> Topic:
    t = Topic(raw=raw, shortened=shortened)
    t.is_head = None  # triggers summarization for all topics
    t.is_refusal = True
    t.summary = None
    return t


def _make_local_config() -> CrawlerConfig:
    cfg = CrawlerConfig()
    cfg.model.summarization_model = "local"
    cfg.model.default_provider = "vllm"
    cfg.model.provider_urls = None
    cfg.model.provider_max_concurrency = None
    cfg.model.temperature = 0.0
    cfg.model.vllm_max_model_len = None
    cfg.crawler.max_generated_tokens = 512
    cfg.crawler.do_filter_refusals = True
    cfg.crawler.max_concurrent_summarizations = 10
    return cfg


def _make_api_config() -> CrawlerConfig:
    cfg = CrawlerConfig()
    cfg.model.summarization_model = "moonshotai/kimi-k2-0905"
    cfg.model.default_provider = "openrouter"
    cfg.model.provider_urls = None
    cfg.model.provider_max_concurrency = None
    cfg.crawler.max_concurrent_summarizations = 10
    cfg.crawler.do_filter_refusals = True
    return cfg


class TestS0gSummarizationLocalErrors:
    """S0g: local-model summarization — narrow except."""

    def test_summarize_local_reraises_unexpected_exception(self):
        """OOM / config errors in local-model summarization must propagate."""
        cfg = _make_local_config()
        tf = TopicFormatter(cfg)

        # Use a topic with enough words to bypass the word-count shortcut (<=7)
        topics = [_make_topic("this is a very long topic name that needs summarization")]

        with patch(
            "src.generation_utils.batch_generate",
            side_effect=RuntimeError("OOM in vllm"),
        ):
            with pytest.raises(RuntimeError, match="OOM in vllm"):
                tf.summarize_refusal_topics(
                    topics=topics, local_model=None, local_tokenizer=None
                )

    def test_summarize_local_falls_back_on_api_timeout_with_warning(
        self, caplog
    ):
        """APITimeoutError in local summarization triggers fallback to shortened + logs."""
        cfg = _make_local_config()
        tf = TopicFormatter(cfg)

        topics = [
            _make_topic(
                "this is a very long topic name that needs summarization",
                shortened="short label",
            )
        ]

        with caplog.at_level(logging.WARNING, logger="root"):
            with patch(
                "src.generation_utils.batch_generate",
                side_effect=APITimeoutError("timeout"),
            ):
                result = tf.summarize_refusal_topics(
                    topics=topics, local_model=None, local_tokenizer=None
                )

        # Summary should fall back to shortened
        assert result[0].summary == "short label", (
            f"Expected fallback to 'short label', got {result[0].summary!r}"
        )

        # Must emit a warning
        assert any(
            r.levelno >= logging.WARNING for r in caplog.records
        ), f"Expected WARNING-or-higher log; got: {caplog.records}"


class TestS0gSummarizationAPIErrors:
    """S0g: API summarization — narrow except."""

    def test_summarize_api_reraises_unexpected_exception(self):
        """Unexpected exceptions in API summarization must propagate."""
        cfg = _make_api_config()
        tf = TopicFormatter(cfg)

        topics = [_make_topic("this is a very long topic name that needs summarization")]

        with patch(
            "src.provider_config.get_provider_client_kwargs",
            return_value=("moonshotai/kimi-k2-0905", {}),
        ):
            with patch(
                "src.generation_utils.async_batch_summarize_topics",
                side_effect=RuntimeError("connection reset"),
            ):
                with pytest.raises(RuntimeError, match="connection reset"):
                    tf.summarize_refusal_topics(
                        topics=topics, local_model=None, local_tokenizer=None
                    )

    def test_summarize_api_falls_back_on_api_timeout_with_warning(
        self, caplog
    ):
        """APITimeoutError in API summarization triggers fallback to shortened + logs."""
        cfg = _make_api_config()
        tf = TopicFormatter(cfg)

        topics = [
            _make_topic(
                "this is a very long topic name that needs summarization",
                shortened="short api label",
            )
        ]

        with caplog.at_level(logging.WARNING, logger="root"):
            with patch(
                "src.provider_config.get_provider_client_kwargs",
                return_value=("moonshotai/kimi-k2-0905", {}),
            ):
                with patch(
                    "src.generation_utils.async_batch_summarize_topics",
                    side_effect=APITimeoutError("api timeout"),
                ):
                    result = tf.summarize_refusal_topics(
                        topics=topics, local_model=None, local_tokenizer=None
                    )

        # Summary should fall back to shortened
        assert result[0].summary == "short api label", (
            f"Expected fallback to 'short api label', got {result[0].summary!r}"
        )

        # Must emit a warning
        assert any(
            r.levelno >= logging.WARNING for r in caplog.records
        ), f"Expected WARNING-or-higher log; got: {caplog.records}"
