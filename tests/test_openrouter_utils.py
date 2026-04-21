"""Tests for _apply_nitro helper and async_query_openrouter prefer_nitro plumbing.

All tests are pure (mocks only). No live API calls.
"""
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

from src.openrouter_utils import _apply_nitro, _OPENROUTER_BASE_URL


OPENROUTER_BASE = _OPENROUTER_BASE_URL  # "https://openrouter.ai/api/v1"
OPENAI_BASE = "https://api.openai.com/v1"
OLLAMA_BASE = "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# Unit tests for _apply_nitro
# ---------------------------------------------------------------------------

class TestApplyNitro:
    def test_nitro_appended_on_openrouter_when_enabled(self):
        result = _apply_nitro("google/gemini-flash", OPENROUTER_BASE, prefer=True)
        assert result == "google/gemini-flash:nitro"

    def test_nitro_not_appended_when_disabled(self):
        result = _apply_nitro("google/gemini-flash", OPENROUTER_BASE, prefer=False)
        assert result == "google/gemini-flash"

    def test_nitro_not_double_appended_nitro_suffix(self):
        result = _apply_nitro("google/gemini-flash:nitro", OPENROUTER_BASE, prefer=True)
        assert result == "google/gemini-flash:nitro"

    def test_nitro_not_double_appended_floor_suffix(self):
        result = _apply_nitro("google/gemini-flash:floor", OPENROUTER_BASE, prefer=True)
        assert result == "google/gemini-flash:floor"

    def test_floor_suffix_preserved_when_present(self):
        """Verifying :floor suffix is left unchanged even with prefer_nitro=True."""
        result = _apply_nitro("anthropic/claude-3.5-haiku:floor", OPENROUTER_BASE, prefer=True)
        assert result == "anthropic/claude-3.5-haiku:floor"

    def test_nitro_skipped_for_openai_provider(self):
        result = _apply_nitro("gpt-4o", OPENAI_BASE, prefer=True)
        assert result == "gpt-4o"

    def test_nitro_skipped_for_ollama_provider(self):
        result = _apply_nitro("llama3", OLLAMA_BASE, prefer=True)
        assert result == "llama3"

    def test_nitro_normalises_trailing_slash(self):
        """Base URL with trailing slash should still match."""
        result = _apply_nitro("some/model", OPENROUTER_BASE + "/", prefer=True)
        assert result == "some/model:nitro"


# ---------------------------------------------------------------------------
# Integration: async_query_openrouter passes prefer_nitro to the model= kwarg
# Uses asyncio.run() directly — no pytest-asyncio required.
# ---------------------------------------------------------------------------

def test_nitro_appended_in_async_query_openrouter():
    """When prefer_nitro=True and base_url is openrouter, model= kwarg has :nitro suffix."""
    from src.openrouter_utils import async_query_openrouter

    mock_choice = MagicMock()
    mock_choice.message.content = "test response"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_create = AsyncMock(return_value=mock_completion)
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    async def _run():
        return await async_query_openrouter(
            model_name="google/gemini-flash",
            prompt="hello",
            prefer_nitro=True,
        )

    with patch("src.openrouter_utils.log_model_call"):
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                result = asyncio.run(_run())

    assert result == "test response"
    call_kwargs = mock_create.call_args
    assert call_kwargs.kwargs["model"] == "google/gemini-flash:nitro"


def test_nitro_not_appended_when_prefer_nitro_false():
    """When prefer_nitro=False, model= kwarg stays unchanged."""
    from src.openrouter_utils import async_query_openrouter

    mock_choice = MagicMock()
    mock_choice.message.content = "test response"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_create = AsyncMock(return_value=mock_completion)
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    async def _run():
        return await async_query_openrouter(
            model_name="google/gemini-flash",
            prompt="hello",
            prefer_nitro=False,
        )

    with patch("src.openrouter_utils.log_model_call"):
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                result = asyncio.run(_run())

    assert result == "test response"
    call_kwargs = mock_create.call_args
    assert call_kwargs.kwargs["model"] == "google/gemini-flash"
