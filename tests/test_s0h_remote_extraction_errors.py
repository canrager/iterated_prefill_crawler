"""Unit tests for S0h — remote extraction _call_one_chunk must not silently discard.

Tests:
- test_remote_extract_403_still_triggers_split_retry
- test_remote_extract_api_timeout_reraises
- test_remote_extract_unexpected_exception_reraises_with_log
- test_remote_extract_split_retry_still_works_for_shape_mismatch
"""

import json
import logging
import pytest
from unittest.mock import patch

from src.crawler.config import CrawlerConfig
from src.exceptions import APITimeoutError
from src.response_formatting_utils import TopicFormatter


def _make_remote_config(extraction_batch_size: int = 4) -> CrawlerConfig:
    cfg = CrawlerConfig()
    cfg.model.summarization_model = "moonshotai/kimi-k2-0905"
    cfg.model.default_provider = "openrouter"
    cfg.model.provider_urls = None
    cfg.model.provider_max_concurrency = None
    cfg.crawler.max_topics_per_step_lang = 50
    cfg.crawler.max_concurrent_summarizations = 10
    cfg.crawler.do_filter_refusals = False
    cfg.crawler.extraction_batch_size = extraction_batch_size
    return cfg


class _Fake403(Exception):
    def __init__(self):
        super().__init__("moderation blocked")
        self.status_code = 403


class TestS0hRemoteExtractionErrors:
    """S0h: _call_one_chunk must re-raise non-403 exceptions, not return empties."""

    def test_remote_extract_403_still_triggers_split_retry(self):
        """403 moderation still returns None to trigger split-retry — unchanged by S0h."""
        cfg = _make_remote_config(extraction_batch_size=4)
        tf = TopicFormatter(cfg)

        texts = ["r0", "r1", "r2", "r3"]

        responses_iter = iter([
            "_raise_403",
            json.dumps([["a"], ["b"]]),   # left half [0:2]
            json.dumps([["c"], ["d"]]),   # right half [2:4]
        ])

        async def _fake_query(**kwargs):
            nxt = next(responses_iter)
            if nxt == "_raise_403":
                raise _Fake403()
            return nxt

        with patch(
            "src.generation_utils.async_query_openrouter",
            side_effect=_fake_query,
        ) as mock_query:
            result = tf._extract_with_model(texts)

        assert mock_query.call_count == 3, (
            f"Expected 3 calls (1 mod + 2 split halves), got {mock_query.call_count}"
        )
        assert result == [["a"], ["b"], ["c"], ["d"]]

    def test_remote_extract_api_timeout_reraises(self):
        """APITimeoutError from async_query_openrouter must propagate — not swallowed."""
        cfg = _make_remote_config(extraction_batch_size=4)
        tf = TopicFormatter(cfg)

        texts = ["r0", "r1", "r2", "r3"]

        async def _fake_query(**kwargs):
            raise APITimeoutError("timed out")

        with patch(
            "src.generation_utils.async_query_openrouter",
            side_effect=_fake_query,
        ) as mock_query:
            with pytest.raises(APITimeoutError):
                tf._extract_with_model(texts)

        # One call was made — the exception stops recursion, not split-retry
        assert mock_query.call_count == 1, (
            f"Expected 1 call before re-raise, got {mock_query.call_count}"
        )

    def test_remote_extract_unexpected_exception_reraises_with_log(
        self, caplog
    ):
        """Unexpected exceptions re-raise and emit logging.exception (non-gated)."""
        cfg = _make_remote_config(extraction_batch_size=4)
        tf = TopicFormatter(cfg)

        texts = ["r0", "r1", "r2", "r3"]

        async def _fake_query(**kwargs):
            raise RuntimeError("unexpected boom")

        with caplog.at_level(logging.ERROR, logger="root"):
            with patch(
                "src.generation_utils.async_query_openrouter",
                side_effect=_fake_query,
            ):
                with pytest.raises(RuntimeError, match="unexpected boom"):
                    tf._extract_with_model(texts)

        # logging.exception emits at ERROR level and includes the message
        assert any(
            "unexpected" in r.message.lower() or "extract" in r.message.lower()
            for r in caplog.records
        ), f"Expected a log record mentioning the error; got: {caplog.records}"

    def test_remote_extract_split_retry_still_works_for_shape_mismatch(self):
        """Regression: shape-mismatch still triggers split-retry after S0h fix."""
        cfg = _make_remote_config(extraction_batch_size=4)
        tf = TopicFormatter(cfg)

        texts = ["r0", "r1", "r2", "r3"]

        responses = iter([
            json.dumps([["a"], ["b"], ["c"]]),  # shape mismatch: len=3 != 4 → None
            json.dumps([["a"], ["b"]]),           # left half [0:2] → OK
            json.dumps([["c"], ["d"]]),           # right half [2:4] → OK
        ])

        async def _fake_query(**kwargs):
            return next(responses)

        with patch(
            "src.generation_utils.async_query_openrouter",
            side_effect=_fake_query,
        ) as mock_query:
            result = tf._extract_with_model(texts)

        assert mock_query.call_count == 3, (
            f"Expected 3 calls (1 mismatch + 2 split halves), got {mock_query.call_count}"
        )
        assert result == [["a"], ["b"], ["c"], ["d"]]
