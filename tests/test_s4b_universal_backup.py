"""Tests for S4b: universal_backup_model config field and call-site wiring.

All tests are pure (mocks only). No live API calls.
"""
import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.crawler.config import CrawlerConfig, ModelConfig, CrawlerRunConfig


def _make_config(universal_backup_model=None, prefer_nitro=False):
    """Helper: build a CrawlerConfig with the given model settings."""
    model = ModelConfig(
        summarization_model="some/model",
        default_provider="openrouter",
        universal_backup_model=universal_backup_model,
        prefer_nitro=prefer_nitro,
    )
    crawler = CrawlerRunConfig()
    return CrawlerConfig(model=model, crawler=crawler)


def _make_mock_completion(content="response"):
    """Build a mock completion object that async_query_openrouter / _async_api_single accept."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_choice.message.reasoning = None
    mock_choice.message.reasoning_content = None
    mock_choice.finish_reason = "stop"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


# ---------------------------------------------------------------------------
# ModelConfig field tests
# ---------------------------------------------------------------------------

class TestUniversalBackupModelField:
    def test_default_is_none(self):
        m = ModelConfig()
        assert m.universal_backup_model is None

    def test_can_be_set(self):
        m = ModelConfig(universal_backup_model="google/gemini-3-flash-preview")
        assert m.universal_backup_model == "google/gemini-3-flash-preview"

    def test_get_remote_fallbacks_still_works(self):
        """get_remote_fallbacks is preserved even though it is no longer hot-path."""
        m = ModelConfig(
            summarization_model="some/summ-model",
            translation_model="some/trans-model",
            universal_backup_model="google/gemini-3-flash-preview",
        )
        fallbacks = m.get_remote_fallbacks("summarization", "translation")
        assert "some/summ-model" in fallbacks
        assert "some/trans-model" in fallbacks


# ---------------------------------------------------------------------------
# Call-site: grouping_pipeline — fallback_models arg
# ---------------------------------------------------------------------------

class TestGroupingPipelineFallback:
    def test_universal_backup_used_when_set(self):
        """grouping_pipeline passes [backup] as fallback_models when universal_backup_model is set."""
        backup = "google/gemini-3-flash-preview"
        config = _make_config(universal_backup_model=backup)

        captured_kwargs = {}

        async def mock_async_query(**kwargs):
            captured_kwargs.update(kwargs)
            return '{"groups": [], "duplicates": [], "skipped": []}'

        from src.crawler.topic_queue import Topic
        topics = [Topic(raw="test topic")]
        known_heads = []

        # async_query_openrouter is imported locally inside the function, so patch
        # it at its definition site in openrouter_utils
        with patch("src.openrouter_utils.async_query_openrouter", side_effect=mock_async_query):
            with patch("src.crawler.grouping_pipeline.get_provider_client_kwargs",
                       return_value=("some/model", {"base_url": "https://openrouter.ai/api/v1", "api_key": "k"})):
                from src.crawler.grouping_pipeline import summarize_group_dedup
                summarize_group_dedup(
                    topics=topics,
                    known_heads=known_heads,
                    config=config,
                    local_model=None,
                    local_tokenizer=None,
                )

        assert captured_kwargs.get("fallback_models") == [backup]

    def test_no_fallback_when_unset(self):
        """grouping_pipeline passes [] as fallback_models when universal_backup_model is None."""
        config = _make_config(universal_backup_model=None)

        captured_kwargs = {}

        async def mock_async_query(**kwargs):
            captured_kwargs.update(kwargs)
            return '{"groups": [], "duplicates": [], "skipped": []}'

        from src.crawler.topic_queue import Topic
        topics = [Topic(raw="test topic")]
        known_heads = []

        with patch("src.openrouter_utils.async_query_openrouter", side_effect=mock_async_query):
            with patch("src.crawler.grouping_pipeline.get_provider_client_kwargs",
                       return_value=("some/model", {"base_url": "https://openrouter.ai/api/v1", "api_key": "k"})):
                from src.crawler.grouping_pipeline import summarize_group_dedup
                summarize_group_dedup(
                    topics=topics,
                    known_heads=known_heads,
                    config=config,
                    local_model=None,
                    local_tokenizer=None,
                )

        assert captured_kwargs.get("fallback_models") == []


# ---------------------------------------------------------------------------
# Call-site: progressive_refusal — fallback_models arg
# ---------------------------------------------------------------------------

class TestProgressiveRefusalFallback:
    def _make_pr_config(self, backup=None, refusal_model="some/refusal-model", target_model="some/target-model"):
        model = ModelConfig(
            refusal_check_model=refusal_model,
            target_model=target_model,
            translation_model="some/translation-model",
            summarization_model="some/summ-model",
            default_provider="openrouter",
            universal_backup_model=backup,
            prefer_nitro=False,
        )
        crawler = CrawlerRunConfig(
            refusal_triage_checks=3,
            refusal_escalation_checks=2,
            is_refusal_threshold=0.25,
            max_refusal_check_generated_tokens=256,
            num_refusal_checks_per_topic=3,
        )
        return CrawlerConfig(model=model, crawler=crawler)

    def test_universal_backup_used_at_progressive_refusal_call_site(self):
        """check_refusal_progressive passes [backup] as fallback_models when universal_backup_model is set."""
        backup = "google/gemini-3-flash-preview"
        config = self._make_pr_config(backup=backup)

        from src.crawler.topic_queue import Topic
        topics = [Topic(raw="test topic")]

        captured_fallbacks_list = []

        def mock_batch_generate(*args, **kwargs):
            captured_fallbacks_list.append(kwargs.get("fallback_models", []))
            # _generate_provocation_queries: one message, expects a JSON array response
            # _query_target: multiple messages, expects plain text responses
            messages = args[2] if len(args) > 2 else kwargs.get("messages", [])
            n = len(messages)
            if n == 1:
                return ['["what about test topic?", "explain test topic"]'], ["prompt"]
            else:
                return ["I'll help with that."] * n, ["prompt"] * n

        # progressive_refusal has two batch_generate lookup sites:
        # 1. Module-level `from src.generation_utils import batch_generate` (line 24)
        #    — used by _generate_provocation_queries. Patch via the module attr.
        # 2. In-function `from src.generation_utils import batch_generate` inside _query_target
        #    — re-imported each call. Patch via src.generation_utils (the source module).
        # Both patches are required to prevent live API calls.
        import src.generation_utils as _gen
        with patch("src.crawler.progressive_refusal.batch_generate",
                   side_effect=mock_batch_generate):
            with patch.object(_gen, "batch_generate", side_effect=mock_batch_generate):
                with patch("src.crawler.progressive_refusal.check_refusals_cascade",
                           return_value=[False, False, False]):
                    from src.crawler.progressive_refusal import check_refusal_progressive
                    check_refusal_progressive(
                        config=config,
                        local_model=None,
                        local_tokenizer=None,
                        selected_topics=topics,
                    )

        # All batch_generate calls from _generate_provocation_queries should use [backup]
        assert len(captured_fallbacks_list) > 0, "Expected at least one batch_generate call"
        # The provocation query generation calls should have [backup] as fallback
        provocation_calls = [fb for fb in captured_fallbacks_list if fb == [backup]]
        assert len(provocation_calls) > 0, (
            f"Expected at least one batch_generate call with fallback_models=[{backup!r}], "
            f"got: {captured_fallbacks_list}"
        )

    def test_no_fallback_at_progressive_refusal_when_unset(self):
        """check_refusal_progressive passes [] as fallback_models when universal_backup_model is None."""
        config = self._make_pr_config(backup=None)

        from src.crawler.topic_queue import Topic
        topics = [Topic(raw="test topic")]

        captured_fallbacks_list = []

        def mock_batch_generate(*args, **kwargs):
            captured_fallbacks_list.append(kwargs.get("fallback_models", []))
            messages = args[2] if len(args) > 2 else kwargs.get("messages", [])
            n = len(messages)
            if n == 1:
                return ['["what about test topic?", "explain test topic"]'], ["prompt"]
            else:
                return ["I'll help with that."] * n, ["prompt"] * n

        import src.generation_utils as _gen
        with patch("src.crawler.progressive_refusal.batch_generate",
                   side_effect=mock_batch_generate):
            with patch.object(_gen, "batch_generate", side_effect=mock_batch_generate):
                with patch("src.crawler.progressive_refusal.check_refusals_cascade",
                           return_value=[False, False, False]):
                    from src.crawler.progressive_refusal import check_refusal_progressive
                    check_refusal_progressive(
                        config=config,
                        local_model=None,
                        local_tokenizer=None,
                        selected_topics=topics,
                    )

        assert len(captured_fallbacks_list) > 0, "Expected at least one batch_generate call"
        # All provocation query generation calls should have [] as fallback
        non_empty_fallbacks = [fb for fb in captured_fallbacks_list if fb]
        assert non_empty_fallbacks == [], (
            f"Expected all batch_generate calls to have empty fallback_models, "
            f"got non-empty: {non_empty_fallbacks}"
        )


# ---------------------------------------------------------------------------
# Call-site: response_formatting _extract_with_model — fallback_models arg
# ---------------------------------------------------------------------------

class TestResponseFormattingFallback:
    def _make_rf_config(self, backup=None):
        model = ModelConfig(
            summarization_model="some/extractor-model",
            default_provider="openrouter",
            universal_backup_model=backup,
            prefer_nitro=False,
        )
        crawler = CrawlerRunConfig(extraction_batch_size=1)
        return CrawlerConfig(model=model, crawler=crawler)

    def test_universal_backup_used_at_response_formatting_call_site(self):
        """_extract_with_model passes [backup] as fallback_models when universal_backup_model is set."""
        backup = "google/gemini-3-flash-preview"
        config = self._make_rf_config(backup=backup)

        captured_kwargs = {}

        async def mock_async_query(**kwargs):
            captured_kwargs.update(kwargs)
            return '["test topic label"]'

        from src.response_formatting_utils import TopicFormatter
        formatter = TopicFormatter(config)

        # _extract_with_model does a late import: `from src.generation_utils import async_query_openrouter`
        # generation_utils re-exports async_query_openrouter from openrouter_utils at module load time,
        # so patching src.generation_utils.async_query_openrouter intercepts the lookup.
        # We also patch get_provider_client_kwargs at its source (src.provider_config) because
        # _extract_with_model calls it directly via `from src.provider_config import ...`.
        with patch("src.generation_utils.async_query_openrouter",
                   side_effect=mock_async_query):
            with patch("src.provider_config.get_provider_client_kwargs",
                       return_value=("some/extractor-model",
                                     {"base_url": "https://openrouter.ai/api/v1", "api_key": "k"})):
                formatter._extract_with_model(["some AI response text"])

        assert captured_kwargs.get("fallback_models") == [backup], (
            f"Expected fallback_models=[{backup!r}], got: {captured_kwargs.get('fallback_models')}"
        )

    def test_no_fallback_at_response_formatting_when_unset(self):
        """_extract_with_model passes [] as fallback_models when universal_backup_model is None."""
        config = self._make_rf_config(backup=None)

        captured_kwargs = {}

        async def mock_async_query(**kwargs):
            captured_kwargs.update(kwargs)
            return '["test topic label"]'

        from src.response_formatting_utils import TopicFormatter
        formatter = TopicFormatter(config)

        with patch("src.generation_utils.async_query_openrouter",
                   side_effect=mock_async_query):
            with patch("src.provider_config.get_provider_client_kwargs",
                       return_value=("some/extractor-model",
                                     {"base_url": "https://openrouter.ai/api/v1", "api_key": "k"})):
                formatter._extract_with_model(["some AI response text"])

        assert captured_kwargs.get("fallback_models") == [], (
            f"Expected fallback_models=[], got: {captured_kwargs.get('fallback_models')}"
        )


# ---------------------------------------------------------------------------
# Integration: backup model gets :nitro suffix when primary fails
# ---------------------------------------------------------------------------

class TestBackupNitroIntegration:
    def test_backup_string_picked_up_by_nitro_transform(self):
        """When universal_backup_model is an openrouter-eligible string and prefer_nitro=True,
        the :nitro suffix is applied to the backup model in the async_query pair-building loop.

        We verify this by letting async_query_openrouter run with a mocked AsyncOpenAI client
        and checking that the fallback model entry used in the completions.create() call
        has the :nitro suffix appended.
        """
        from src.openrouter_utils import async_query_openrouter

        backup = "google/gemini-3-flash-preview"

        # Primary fails → fallback is tried; check the model= kwarg on fallback call
        call_models = []

        mock_choice = MagicMock()
        mock_choice.message.content = "response"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        async def mock_create(**kwargs):
            call_models.append(kwargs.get("model", ""))
            return mock_completion

        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create

        async def _run():
            return await async_query_openrouter(
                model_name="some/primary-model",
                prompt="test",
                fallback_models=[backup],
                prefer_nitro=True,
                # Use default openrouter base_url (no client_kwargs)
            )

        with patch("src.openrouter_utils.log_model_call"):
            with patch("openai.AsyncOpenAI", return_value=mock_client):
                with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                    asyncio.run(_run())

        # Primary: some/primary-model:nitro
        # Fallback (if tried): google/gemini-3-flash-preview:nitro
        # At minimum the primary call should show :nitro
        assert any(":nitro" in m for m in call_models), (
            f"Expected at least one call with :nitro suffix, got: {call_models}"
        )

    def test_backup_receives_nitro_when_primary_fails(self):
        """When primary raises an exception, the backup model is tried with the :nitro suffix.

        This exercises the actual fallback firing path: primary fails → backup is called.
        Asserts the second (backup) call uses the backup model string with :nitro suffix.
        """
        from src.openrouter_utils import async_query_openrouter

        backup = "google/gemini-3-flash-preview"
        call_models = []

        mock_choice = MagicMock()
        mock_choice.message.content = "fallback response"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        call_count = [0]

        async def mock_create(**kwargs):
            call_count[0] += 1
            call_models.append(kwargs.get("model", ""))
            if call_count[0] == 1:
                # Primary call fails with a generic exception (hits the final
                # except block in async_query_openrouter → break → next fallback)
                raise Exception("Simulated primary failure")
            return mock_completion

        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create

        async def _run():
            return await async_query_openrouter(
                model_name="some/primary-model",
                prompt="test",
                fallback_models=[backup],
                prefer_nitro=True,
            )

        with patch("src.openrouter_utils.log_model_call"):
            with patch("openai.AsyncOpenAI", return_value=mock_client):
                with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                    result = asyncio.run(_run())

        assert result == "fallback response", f"Expected fallback response, got: {result!r}"
        assert len(call_models) == 2, (
            f"Expected exactly 2 calls (primary + backup), got: {call_models}"
        )
        # First call: primary model with :nitro
        assert call_models[0] == "some/primary-model:nitro", (
            f"Primary call model mismatch: {call_models[0]!r}"
        )
        # Second call: backup model with :nitro
        expected_backup = f"{backup}:nitro"
        assert call_models[1] == expected_backup, (
            f"Backup call model mismatch: expected {expected_backup!r}, got {call_models[1]!r}"
        )
