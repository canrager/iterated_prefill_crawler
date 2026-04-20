"""Unit test: model fallback chain in generation infrastructure.

When the primary model fails (timeout, 5xx after retries, parse error), the
fallback chain retries with the next model in the caller-provided list.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.openrouter_utils import async_query_openrouter
from src.generation_utils import _async_api_single


def _make_choice(content="test response"):
    """Build a mock choice with the given content."""
    msg = MagicMock()
    msg.content = content
    msg.reasoning = None
    msg.reasoning_content = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    return choice


def _make_completion(choices=None):
    """Build a mock completion object."""
    comp = MagicMock()
    comp.choices = choices or [_make_choice()]
    return comp


class TestFallbackOpenRouterUtils:
    """Test fallback chain in async_query_openrouter."""

    def test_fallback_on_timeout(self):
        """Primary model times out -> fallback model succeeds."""
        async def _test():
            primary = "primary-model"
            fallback = "fallback-model"

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[
                    asyncio.TimeoutError(),
                    _make_completion([_make_choice("fallback response")]),
                ]
            )

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                result = await async_query_openrouter(
                    model_name=primary,
                    prompt="test",
                    fallback_models=[fallback],
                    client_kwargs={"api_key": "test", "base_url": "https://test"},
                )

            assert result == "fallback response"
            assert mock_client.chat.completions.create.call_count == 2

        asyncio.run(_test())

    def test_fallback_on_empty_response(self):
        """Primary model returns empty -> fallback model succeeds."""
        async def _test():
            primary = "primary-model"
            fallback = "fallback-model"

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[
                    _make_completion([_make_choice("")]),
                    _make_completion([_make_choice("fallback response")]),
                ]
            )

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                result = await async_query_openrouter(
                    model_name=primary,
                    prompt="test",
                    fallback_models=[fallback],
                    client_kwargs={"api_key": "test", "base_url": "https://test"},
                )

            assert result == "fallback response"

        asyncio.run(_test())

    def test_all_models_exhausted(self):
        """All models fail -> raises APITimeoutError (new contract: loud failure, not silent empty)."""
        from src.exceptions import APITimeoutError

        async def _test():
            from openai import APIStatusError

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=asyncio.TimeoutError()
            )

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                with pytest.raises(APITimeoutError):
                    await async_query_openrouter(
                        model_name="primary",
                        prompt="test",
                        fallback_models=["fallback1"],
                        client_kwargs={"api_key": "test", "base_url": "https://test"},
                        request_max_total_s=0.0,
                    )

        asyncio.run(_test())

    def test_no_fallback_backwards_compat(self):
        """Without fallback_models, existing behavior is unchanged."""
        async def _test():
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_make_completion([_make_choice("normal response")])
            )

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                result = await async_query_openrouter(
                    model_name="primary",
                    prompt="test",
                    client_kwargs={"api_key": "test", "base_url": "https://test"},
                )

            assert result == "normal response"
            assert mock_client.chat.completions.create.call_count == 1

        asyncio.run(_test())

    def test_4xx_raises_immediately(self):
        """Auth/bad-request errors raise immediately, no fallback."""
        async def _test():
            from openai import APIStatusError

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=APIStatusError(
                    message="not found",
                    response=MagicMock(status_code=404),
                    body=None,
                )
            )

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                with pytest.raises(APIStatusError):
                    await async_query_openrouter(
                        model_name="primary",
                        prompt="test",
                        fallback_models=["fallback"],
                        client_kwargs={"api_key": "test", "base_url": "https://test"},
                    )

        asyncio.run(_test())

    def test_multiple_fallbacks(self):
        """Primary fails, first fallback fails, second fallback succeeds."""
        async def _test():
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[
                    asyncio.TimeoutError(),
                    asyncio.TimeoutError(),
                    _make_completion([_make_choice("second fallback response")]),
                ]
            )

            with patch("openai.AsyncOpenAI", return_value=mock_client):
                result = await async_query_openrouter(
                    model_name="primary",
                    prompt="test",
                    fallback_models=["fallback1", "fallback2"],
                    client_kwargs={"api_key": "test", "base_url": "https://test"},
                )

            assert result == "second fallback response"
            assert mock_client.chat.completions.create.call_count == 3

        asyncio.run(_test())


class TestFallbackGenerationUtils:
    """Test fallback chain in _async_api_single."""

    def test_fallback_on_timeout(self):
        """Primary model times out -> fallback model succeeds."""
        async def _test():
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[
                    asyncio.TimeoutError(),
                    _make_completion([_make_choice("fallback response")]),
                ]
            )

            result = await _async_api_single(
                client=mock_client,
                model_name="primary",
                messages=[{"role": "user", "content": "test"}],
                max_new_tokens=100,
                temperature=0.0,
                fallback_specs=[(mock_client, "fallback")],
            )

            assert result == "fallback response"
            assert mock_client.chat.completions.create.call_count == 2

        asyncio.run(_test())

    def test_cross_provider_fallback(self):
        """Fallback client is used when primary's timeout budget is exhausted."""
        async def _test():
            primary_client = MagicMock()
            primary_client.chat.completions.create = AsyncMock(
                side_effect=asyncio.TimeoutError()
            )
            fallback_client = MagicMock()
            fallback_client.chat.completions.create = AsyncMock(
                return_value=_make_completion([_make_choice("cross-provider response")])
            )

            # request_max_total_s=0.0 causes the primary to exhaust its budget
            # immediately on the first timeout, then fall through to the fallback.
            result = await _async_api_single(
                client=primary_client,
                model_name="primary",
                messages=[{"role": "user", "content": "test"}],
                max_new_tokens=100,
                temperature=0.0,
                fallback_specs=[(fallback_client, "other-provider-model")],
                request_max_total_s=0.0,
            )

            assert result == "cross-provider response"
            assert primary_client.chat.completions.create.call_count == 1
            assert fallback_client.chat.completions.create.call_count == 1

        asyncio.run(_test())

    def test_extra_body_forwarded(self):
        """extra_body is forwarded to the create() call."""
        async def _test():
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_make_completion([_make_choice("response")])
            )

            result = await _async_api_single(
                client=mock_client,
                model_name="test-model",
                messages=[{"role": "user", "content": "test"}],
                max_new_tokens=100,
                temperature=0.0,
                extra_body={"reasoning": {"effort": "none"}},
            )

            assert result == "response"
            call_kwargs = mock_client.chat.completions.create.call_args
            assert call_kwargs.kwargs.get("extra_body") == {"reasoning": {"effort": "none"}}

        asyncio.run(_test())

    def test_no_fallback_no_extra_body(self):
        """Without fallback or extra_body, behavior is unchanged."""
        async def _test():
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_make_completion([_make_choice("normal")])
            )

            result = await _async_api_single(
                client=mock_client,
                model_name="test-model",
                messages=[{"role": "user", "content": "test"}],
                max_new_tokens=100,
                temperature=0.0,
            )

            assert result == "normal"
            call_kwargs = mock_client.chat.completions.create.call_args
            assert "extra_body" not in call_kwargs.kwargs

        asyncio.run(_test())
