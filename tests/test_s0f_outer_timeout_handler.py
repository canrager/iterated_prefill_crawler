"""Unit tests for S0f — outer APITimeoutError handlers must re-raise.

Both extract_and_format and extract_and_translate had outer
`except APITimeoutError: return []` handlers that swallowed the error.
After S0h makes _extract_with_model propagate APITimeoutError, these
outer handlers must re-raise rather than silently returning an empty list.

Tests:
- test_extract_and_format_reraises_api_timeout
- test_extract_and_translate_reraises_api_timeout
"""

import pytest
from unittest.mock import patch

from src.crawler.config import CrawlerConfig
from src.exceptions import APITimeoutError
from src.response_formatting_utils import TopicFormatter


def _make_config() -> CrawlerConfig:
    cfg = CrawlerConfig()
    cfg.model.summarization_model = "moonshotai/kimi-k2-0905"
    cfg.model.default_provider = "openrouter"
    cfg.model.provider_urls = None
    cfg.crawler.max_topics_per_step_lang = 50
    cfg.crawler.do_filter_refusals = False
    cfg.crawler.extraction_batch_size = 1
    return cfg


class TestS0fOuterTimeoutHandler:
    """S0f: outer APITimeoutError handlers must re-raise, not return []."""

    def test_extract_and_format_reraises_api_timeout(self):
        """extract_and_format must propagate APITimeoutError from _extract_with_model."""
        cfg = _make_config()
        tf = TopicFormatter(cfg)

        with patch.object(
            tf,
            "_extract_with_model",
            side_effect=APITimeoutError("inner timeout"),
        ):
            with pytest.raises(APITimeoutError):
                tf.extract_and_format(
                    local_model=None,
                    local_tokenizer=None,
                    input_strs=["prompt"],
                    generations=["gen"],
                    parent_ids=[-1],
                )

    def test_extract_and_translate_reraises_api_timeout(self):
        """extract_and_translate must propagate APITimeoutError from _extract_with_model."""
        cfg = _make_config()
        tf = TopicFormatter(cfg)

        with patch.object(
            tf,
            "_extract_with_model",
            side_effect=APITimeoutError("inner timeout"),
        ):
            with pytest.raises(APITimeoutError):
                tf.extract_and_translate(
                    local_model=None,
                    local_tokenizer=None,
                    input_strs=["prompt"],
                    generations=["gen"],
                    parent_ids=[-1],
                )
