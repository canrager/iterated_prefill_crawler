"""Unit tests for APITimeoutError — S2c configurable timeout with explicit failure."""
import asyncio
import pytest
from unittest.mock import MagicMock, patch

from src.exceptions import APITimeoutError


class TestAPITimeoutErrorRaised:
    """When every API attempt times out, _async_api_single raises APITimeoutError."""

    def test_timeout_error_raised_after_exhaustion(self):
        """_async_api_single should raise APITimeoutError when retries exhaust."""
        from src.generation_utils import _async_api_single

        fake_client = MagicMock()

        async def _always_timeout(**kwargs):
            raise asyncio.TimeoutError()

        fake_client.chat.completions.create = _always_timeout

        async def _run():
            with pytest.raises(APITimeoutError):
                await _async_api_single(
                    fake_client,
                    "test-model",
                    [{"role": "user", "content": "hi"}],
                    max_new_tokens=10,
                    temperature=0.0,
                    timeout=0.01,
                    request_max_total_s=0.05,
                )

        asyncio.run(_run())


class TestRefusalCheckInconclusive:
    """check_refusal_progressive marks topic as refusal_check_inconclusive=True
    when _query_target raises APITimeoutError, not is_refusal=False."""

    def test_timeout_marks_inconclusive(self):
        """When all Phase-A responses are empty due to timeout,
        the topic is marked inconclusive, not compliant."""
        from src.crawler.progressive_refusal import classify_refusal_triage

        # Simulate: all empty responses after timeout
        result = classify_refusal_triage(
            refusal_flags=[False, False, False],
            response_texts=["", "", ""],
        )
        assert result.inconclusive is True
        # The existing logic already handles this correctly:
        # all-empty -> inconclusive, not compliant.


class TestBatchGenerateTimeout:
    """batch_generate propagates APITimeoutError."""

    def test_batch_generate_raises_on_timeout(self, monkeypatch):
        """When _api_batch_generate hits APITimeoutError, it propagates."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        # Patch _async_api_single to always raise APITimeoutError
        async def _fake_async_single(client, model_name, messages, max_new_tokens,
                                     temperature, timeout=120.0, request_max_total_s=3600.0,
                                     extra_body=None, fallback_specs=None):
            raise APITimeoutError("timeout")

        with patch("src.generation_utils._async_api_single", _fake_async_single):
            from src.generation_utils import batch_generate
            with pytest.raises(APITimeoutError):
                batch_generate(
                    model="openai:gpt-4o-mini",
                    tokenizer=None,
                    messages=[[{"role": "user", "content": "hi"}]],
                    max_new_tokens=10,
                    request_max_total_s=0.01,
                )
