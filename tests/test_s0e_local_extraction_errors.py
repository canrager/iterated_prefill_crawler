"""Unit tests for S0e — local-model extraction exception handling.

The broad `except Exception` in the local-model path of _extract_with_model
was narrowed: json.JSONDecodeError is recoverable (log + return empties);
all other exceptions re-raise.

Tests:
- test_local_extract_reraises_unexpected_exception
- test_local_extract_returns_empties_on_json_decode_error_with_warning
"""

import json
import logging
import pytest
from unittest.mock import patch

from src.crawler.config import CrawlerConfig
from src.response_formatting_utils import TopicFormatter


def _make_local_config(num_texts: int = 3) -> CrawlerConfig:
    cfg = CrawlerConfig()
    cfg.model.summarization_model = "local"
    cfg.model.default_provider = "vllm"
    cfg.model.provider_urls = None
    cfg.model.provider_max_concurrency = None
    cfg.crawler.extraction_batch_size = 1
    return cfg


class TestS0eLocalExtractionErrors:
    """S0e: local-model extraction broad except narrowed to json.JSONDecodeError."""

    def test_local_extract_reraises_unexpected_exception(self):
        """OOM / config errors (not json.JSONDecodeError) must propagate."""
        cfg = _make_local_config()
        tf = TopicFormatter(cfg)

        texts = ["response A", "response B"]

        with patch(
            "src.generation_utils.batch_generate",
            side_effect=RuntimeError("OOM"),
        ):
            with pytest.raises(RuntimeError, match="OOM"):
                tf._extract_with_model(texts, local_model=None, local_tokenizer=None)

    def test_local_extract_returns_empties_on_json_decode_error_with_warning(
        self, caplog
    ):
        """json.JSONDecodeError is recoverable: return empties and emit a warning."""
        cfg = _make_local_config()
        tf = TopicFormatter(cfg)

        texts = ["response A", "response B"]

        with caplog.at_level(logging.WARNING, logger="root"):
            with patch(
                "src.generation_utils.batch_generate",
                side_effect=json.JSONDecodeError("bad json", "", 0),
            ):
                result = tf._extract_with_model(
                    texts, local_model=None, local_tokenizer=None
                )

        # Must return empties aligned with texts
        assert result == [[], []], f"Expected [[], []], got {result}"

        # Must emit a warning (not gated by verbose)
        assert len(caplog.records) >= 1, "Expected at least one log record"
        assert any(
            r.levelno >= logging.WARNING for r in caplog.records
        ), f"Expected a WARNING-or-higher log record; got: {caplog.records}"

    def test_local_extract_logs_before_reraising_unexpected_exception(self, caplog):
        """Unexpected exceptions must be logged (with traceback) before re-raising."""
        cfg = _make_local_config()
        tf = TopicFormatter(cfg)

        texts = ["response A", "response B"]

        with caplog.at_level(logging.ERROR):
            with patch(
                "src.generation_utils.batch_generate",
                side_effect=RuntimeError("GPU OOM"),
            ):
                with pytest.raises(RuntimeError, match="GPU OOM"):
                    tf._extract_with_model(
                        texts, local_model=None, local_tokenizer=None
                    )

        # (a) exception propagated — asserted by pytest.raises above

        # (b) a log record with the failure message and exc_info populated
        matching = [
            r
            for r in caplog.records
            if r.levelno >= logging.ERROR
            and "unexpected exception" in r.message.lower()
            and r.exc_info is not None
        ]
        assert matching, (
            "Expected an ERROR-level log record containing 'unexpected exception' "
            f"with exc_info set; got records: {caplog.records}"
        )
