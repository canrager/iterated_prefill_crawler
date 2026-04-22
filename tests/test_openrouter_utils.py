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


# ---------------------------------------------------------------------------
# Task 3: batch_generate → _api_batch_generate path applies :nitro
# ---------------------------------------------------------------------------

def _make_mock_client_for_batch():
    """Build a mock AsyncOpenAI client suitable for _api_batch_generate."""
    mock_choice = MagicMock()
    mock_choice.message.content = "judge answer"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_create = AsyncMock(return_value=mock_completion)
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create
    return mock_client, mock_create


def test_batch_generate_nitro_appended_for_openrouter():
    """batch_generate with prefer_nitro=True on an OpenRouter model appends :nitro."""
    from src.generation_utils import batch_generate

    mock_client, mock_create = _make_mock_client_for_batch()
    messages = [[{"role": "user", "content": "Is this a refusal?"}]]

    with patch("src.generation_utils.log_model_call"):
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                texts, _ = batch_generate(
                    model="openai/gpt-5.4-mini",
                    tokenizer=None,
                    messages=messages,
                    max_new_tokens=16,
                    temperature=0.0,
                    prefer_nitro=True,
                )

    assert texts == ["judge answer"]
    call_kwargs = mock_create.call_args
    assert call_kwargs.kwargs["model"] == "openai/gpt-5.4-mini:nitro"


def test_batch_generate_nitro_not_appended_when_disabled():
    """batch_generate with prefer_nitro=False leaves model string unchanged."""
    from src.generation_utils import batch_generate

    mock_client, mock_create = _make_mock_client_for_batch()
    messages = [[{"role": "user", "content": "Is this a refusal?"}]]

    with patch("src.generation_utils.log_model_call"):
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                texts, _ = batch_generate(
                    model="openai/gpt-5.4-mini",
                    tokenizer=None,
                    messages=messages,
                    max_new_tokens=16,
                    temperature=0.0,
                    prefer_nitro=False,
                )

    assert texts == ["judge answer"]
    call_kwargs = mock_create.call_args
    assert call_kwargs.kwargs["model"] == "openai/gpt-5.4-mini"


def test_batch_generate_nitro_not_appended_for_non_openrouter():
    """batch_generate with prefer_nitro=True on a non-OpenRouter base_url leaves model unchanged."""
    from src.generation_utils import batch_generate

    mock_client, mock_create = _make_mock_client_for_batch()
    messages = [[{"role": "user", "content": "Is this a refusal?"}]]

    # Use openai: prefix — resolves to api.openai.com, not openrouter
    with patch("src.generation_utils.log_model_call"):
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                texts, _ = batch_generate(
                    model="openai:gpt-5.4-mini",
                    tokenizer=None,
                    messages=messages,
                    max_new_tokens=16,
                    temperature=0.0,
                    prefer_nitro=True,
                )

    assert texts == ["judge answer"]
    call_kwargs = mock_create.call_args
    # Provider is openai (not openrouter) — :nitro must NOT be appended
    assert call_kwargs.kwargs["model"] == "gpt-5.4-mini"
    assert not call_kwargs.kwargs["model"].endswith(":nitro")


# ---------------------------------------------------------------------------
# REASONING_DISABLED constant and extra_body plumbing
# ---------------------------------------------------------------------------

def test_reasoning_disabled_constant_value():
    """REASONING_DISABLED must equal the expected dict that turns off reasoning tokens."""
    from src.openrouter_utils import REASONING_DISABLED
    assert REASONING_DISABLED == {"reasoning": {"effort": "none"}}


def test_async_query_openrouter_forwards_extra_body():
    """async_query_openrouter passes extra_body through to client.chat.completions.create()."""
    from src.openrouter_utils import REASONING_DISABLED, async_query_openrouter

    mock_choice = MagicMock()
    mock_choice.message.content = "helper response"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_create = AsyncMock(return_value=mock_completion)
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    async def _run():
        return await async_query_openrouter(
            model_name="openai/gpt-5.4-mini",
            prompt="summarize this",
            extra_body=REASONING_DISABLED,
        )

    with patch("src.openrouter_utils.log_model_call"):
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                result = asyncio.run(_run())

    assert result == "helper response"
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["extra_body"] == {"reasoning": {"effort": "none"}}


def test_async_query_openrouter_extra_body_none_by_default():
    """When extra_body is not passed, create() is called with extra_body=None (the default)."""
    from src.openrouter_utils import async_query_openrouter

    mock_choice = MagicMock()
    mock_choice.message.content = "response"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_create = AsyncMock(return_value=mock_completion)
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    async def _run():
        return await async_query_openrouter(
            model_name="openai/gpt-5.4-mini",
            prompt="hello",
        )

    with patch("src.openrouter_utils.log_model_call"):
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                asyncio.run(_run())

    call_kwargs = mock_create.call_args.kwargs
    # extra_body should be absent or None when not explicitly passed
    assert call_kwargs.get("extra_body") is None


def test_batch_generate_forwards_extra_body_to_create():
    """batch_generate(extra_body=REASONING_DISABLED) forwards it through to create()."""
    from src.generation_utils import batch_generate
    from src.openrouter_utils import REASONING_DISABLED

    mock_choice = MagicMock()
    mock_choice.message.content = "extracted topics"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_create = AsyncMock(return_value=mock_completion)
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    messages = [[{"role": "user", "content": "extract topics from: foo bar"}]]

    with patch("src.generation_utils.log_model_call"):
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                texts, _ = batch_generate(
                    model="openai/gpt-5.4-mini",
                    tokenizer=None,
                    messages=messages,
                    max_new_tokens=100,
                    temperature=0.0,
                    extra_body=REASONING_DISABLED,
                )

    assert texts == ["extracted topics"]
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["extra_body"] == {"reasoning": {"effort": "none"}}


# ---------------------------------------------------------------------------
# return_usage kwarg: Task 1 tests
# ---------------------------------------------------------------------------

def test_return_usage_true_returns_tuple_with_token_counts():
    """When return_usage=True, async_query_openrouter returns (str, dict) with token counts."""
    from src.openrouter_utils import async_query_openrouter

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 42
    mock_usage.completion_tokens = 17

    mock_choice = MagicMock()
    mock_choice.message.content = "extracted list"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = mock_usage

    mock_create = AsyncMock(return_value=mock_completion)
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    async def _run():
        return await async_query_openrouter(
            model_name="openai/gpt-5.4-mini",
            prompt="extract topics",
            return_usage=True,
        )

    with patch("src.openrouter_utils.log_model_call"):
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                result = asyncio.run(_run())

    assert isinstance(result, tuple), "return_usage=True must return a tuple"
    text, usage = result
    assert text == "extracted list"
    assert usage["prompt_tokens"] == 42
    assert usage["completion_tokens"] == 17


def test_return_usage_false_returns_bare_string():
    """When return_usage=False (default), async_query_openrouter returns a bare string."""
    from src.openrouter_utils import async_query_openrouter

    mock_choice = MagicMock()
    mock_choice.message.content = "bare string response"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = MagicMock()

    mock_create = AsyncMock(return_value=mock_completion)
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    async def _run():
        return await async_query_openrouter(
            model_name="openai/gpt-5.4-mini",
            prompt="extract topics",
            # return_usage defaults to False
        )

    with patch("src.openrouter_utils.log_model_call"):
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                result = asyncio.run(_run())

    assert isinstance(result, str), "return_usage=False must return a bare string, not a tuple"
    assert result == "bare string response"


# ---------------------------------------------------------------------------
# universal_backup_model fallback
# ---------------------------------------------------------------------------

def test_universal_backup_fires_on_timeout():
    """On a timeout from the primary, async_query_openrouter retries once
    against universal_backup_model and returns its response."""
    from src.openrouter_utils import async_query_openrouter
    from openai import APITimeoutError as _SDKTimeout

    primary_choice = MagicMock()
    primary_choice.message.content = "primary would have said this"
    primary_completion = MagicMock()
    primary_completion.choices = [primary_choice]

    backup_choice = MagicMock()
    backup_choice.message.content = "backup response"
    backup_completion = MagicMock()
    backup_completion.choices = [backup_choice]
    backup_completion.usage = None

    call_count = {"n": 0}

    async def _create(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _SDKTimeout(request=MagicMock())
        return backup_completion

    mock_client = MagicMock()
    mock_client.chat.completions.create = _create

    async def _run():
        return await async_query_openrouter(
            model_name="qwen/qwen3-235b-a22b-2507",
            prompt="extract topics",
            universal_backup_model="moonshotai/kimi-k2.5",
        )

    with patch("src.openrouter_utils.log_model_call"):
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                result = asyncio.run(_run())

    assert result == "backup response"
    assert call_count["n"] == 2


def test_universal_backup_not_fired_on_auth_error():
    """403 moderation / 400 / 401 / 404 must NOT trigger the backup — these
    are config errors, fallback would just mask the problem."""
    from src.openrouter_utils import async_query_openrouter
    from openai import APIStatusError as _SDKStatus

    call_count = {"n": 0}

    async def _create(*args, **kwargs):
        call_count["n"] += 1
        resp = MagicMock()
        resp.status_code = 401
        err = _SDKStatus("unauthorized", response=resp, body={"error": {"message": "unauth"}})
        err.status_code = 401
        raise err

    mock_client = MagicMock()
    mock_client.chat.completions.create = _create

    async def _run():
        try:
            return await async_query_openrouter(
                model_name="qwen/qwen3-235b-a22b-2507",
                prompt="extract",
                universal_backup_model="moonshotai/kimi-k2.5",
            )
        except _SDKStatus:
            return "AUTH_RAISED"

    with patch("src.openrouter_utils.log_model_call"):
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                result = asyncio.run(_run())

    assert result == "AUTH_RAISED"
    assert call_count["n"] == 1, "Auth errors must raise immediately; no backup retry"


def test_both_primary_and_backup_timeout_returns_failure_sentinel():
    """When primary times out AND the universal backup also times out, the
    function returns API_CALL_FAILED_SENTINEL -- distinct from "" so callers
    can tell an infrastructure failure apart from a valid empty response."""
    from src.openrouter_utils import async_query_openrouter, API_CALL_FAILED_SENTINEL
    from openai import APITimeoutError as _SDKTimeout

    call_count = {"n": 0}
    models_called = []

    async def _create(*args, **kwargs):
        call_count["n"] += 1
        models_called.append(kwargs.get("model"))
        raise _SDKTimeout(request=MagicMock())

    mock_client = MagicMock()
    mock_client.chat.completions.create = _create

    async def _run():
        return await async_query_openrouter(
            model_name="qwen/qwen3-235b-a22b-2507",
            prompt="extract",
            universal_backup_model="moonshotai/kimi-k2.5",
        )

    with patch("src.openrouter_utils.log_model_call"):
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                result = asyncio.run(_run())

    assert result == API_CALL_FAILED_SENTINEL
    assert call_count["n"] == 2, "Both primary and backup should be tried"


def test_universal_backup_noop_when_same_as_primary():
    """If backup == primary, a timeout should NOT recurse — return the
    call-failure sentinel instead."""
    from src.openrouter_utils import async_query_openrouter, API_CALL_FAILED_SENTINEL
    from openai import APITimeoutError as _SDKTimeout

    call_count = {"n": 0}

    async def _create(*args, **kwargs):
        call_count["n"] += 1
        raise _SDKTimeout(request=MagicMock())

    mock_client = MagicMock()
    mock_client.chat.completions.create = _create

    async def _run():
        return await async_query_openrouter(
            model_name="qwen/qwen3-235b-a22b-2507",
            prompt="extract",
            universal_backup_model="qwen/qwen3-235b-a22b-2507",  # same
        )

    with patch("src.openrouter_utils.log_model_call"):
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                result = asyncio.run(_run())

    assert result == API_CALL_FAILED_SENTINEL
    assert call_count["n"] == 1, "Same-model backup must not recurse"
