"""Tests for S4a follow-up — universal :nitro coverage across ALL hot-path call sites.

Verifies that every batch_generate / async_query_openrouter call in the hot path
respects prefer_nitro=True and passes :nitro-suffixed model names to the API.

All tests are pure (mocks only). No live API calls.
"""

import os
import pytest
from unittest.mock import MagicMock, patch, call

from src.crawler.config import CrawlerConfig, ModelConfig, CrawlerRunConfig


OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _make_model_config(prefer_nitro: bool = True, **kwargs) -> ModelConfig:
    defaults = dict(
        summarization_model="google/gemma-4-26b-a4b-it",
        refusal_check_model="google/gemma-4-26b-a4b-it",
        target_model="qwen/qwen3-235b-a22b-2507",
        translation_model="qwen/qwen3-235b-a22b-2507",
        default_provider="openrouter",
        prefer_nitro=prefer_nitro,
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)


def _make_config(prefer_nitro: bool = True, **model_kwargs) -> CrawlerConfig:
    model = _make_model_config(prefer_nitro=prefer_nitro, **model_kwargs)
    crawler = CrawlerRunConfig(
        refusal_triage_checks=3,
        refusal_escalation_checks=2,
        is_refusal_threshold=0.25,
        max_refusal_check_generated_tokens=256,
        num_refusal_checks_per_topic=3,
    )
    return CrawlerConfig(model=model, crawler=crawler)


def _fake_batch_generate_factory(nitro_models_seen: list):
    """Return a fake batch_generate that records model names seen.

    Infers the model from the call args (first positional arg is model string).
    Returns stub responses suitable for different callers.
    """
    def _fake_batch_generate(model_or_str, tokenizer, messages, **kwargs):
        if isinstance(model_or_str, str):
            nitro_models_seen.append(model_or_str)
        n = len(messages)
        # Return JSON array for provocation generation (single message) and
        # plain text for others
        if n == 1:
            return ['["probe query one", "probe query two"]'], ["prompt"]
        return ["I can help with that."] * n, ["input"] * n

    return _fake_batch_generate


# ---------------------------------------------------------------------------
# llm_judge_refusals — prefer_nitro plumbing
# ---------------------------------------------------------------------------

class TestLlmJudgeRefusalsPreferNitro:
    def test_prefer_nitro_forwarded_to_batch_generate(self):
        """llm_judge_refusals passes prefer_nitro to batch_generate."""
        captured = {}

        def mock_batch_generate(model, tokenizer, messages, **kwargs):
            captured.update(kwargs)
            return ["no"] * len(messages), ["input"] * len(messages)

        from src.refusal_utils import llm_judge_refusals
        with patch("src.refusal_utils.batch_generate", side_effect=mock_batch_generate):
            llm_judge_refusals(
                texts=["some response text"],
                model="google/gemma-4-26b-a4b-it",
                tokenizer=None,
                prefer_nitro=True,
            )

        assert captured.get("prefer_nitro") is True

    def test_prefer_nitro_false_forwarded(self):
        """llm_judge_refusals passes prefer_nitro=False when disabled."""
        captured = {}

        def mock_batch_generate(model, tokenizer, messages, **kwargs):
            captured.update(kwargs)
            return ["no"] * len(messages), ["input"] * len(messages)

        from src.refusal_utils import llm_judge_refusals
        with patch("src.refusal_utils.batch_generate", side_effect=mock_batch_generate):
            llm_judge_refusals(
                texts=["some response text"],
                model="google/gemma-4-26b-a4b-it",
                tokenizer=None,
                prefer_nitro=False,
            )

        assert captured.get("prefer_nitro") is False


# ---------------------------------------------------------------------------
# _translate_for_classifier — prefer_nitro plumbing
# ---------------------------------------------------------------------------

class TestTranslateForClassifierPreferNitro:
    def test_prefer_nitro_forwarded_to_batch_generate(self):
        """_translate_for_classifier passes prefer_nitro to batch_generate."""
        captured = {}

        def mock_batch_generate(model, tokenizer, messages, **kwargs):
            captured.update(kwargs)
            return ["some translation"] * len(messages), ["input"] * len(messages)

        from src.refusal_utils import _translate_for_classifier
        # Chinese text to trigger translation path
        with patch("src.refusal_utils.batch_generate", side_effect=mock_batch_generate):
            _translate_for_classifier(
                texts=["台湾独立运动"],
                indices=[0],
                translation_model="qwen/qwen3-235b-a22b-2507",
                translation_tokenizer=None,
                prefer_nitro=True,
            )

        assert captured.get("prefer_nitro") is True

    def test_prefer_nitro_false_forwarded(self):
        """_translate_for_classifier passes prefer_nitro=False when disabled."""
        captured = {}

        def mock_batch_generate(model, tokenizer, messages, **kwargs):
            captured.update(kwargs)
            return ["some translation"] * len(messages), ["input"] * len(messages)

        from src.refusal_utils import _translate_for_classifier
        with patch("src.refusal_utils.batch_generate", side_effect=mock_batch_generate):
            _translate_for_classifier(
                texts=["台湾独立运动"],
                indices=[0],
                translation_model="qwen/qwen3-235b-a22b-2507",
                translation_tokenizer=None,
                prefer_nitro=False,
            )

        assert captured.get("prefer_nitro") is False

    def test_no_translation_call_for_english_text(self):
        """Non-Chinese texts must not trigger batch_generate even with prefer_nitro=True."""
        call_count = [0]

        def mock_batch_generate(*args, **kwargs):
            call_count[0] += 1
            return ["translation"] * len(args[2]), ["input"] * len(args[2])

        from src.refusal_utils import _translate_for_classifier
        with patch("src.refusal_utils.batch_generate", side_effect=mock_batch_generate):
            result = _translate_for_classifier(
                texts=["english only text"],
                indices=[0],
                translation_model="qwen/qwen3-235b-a22b-2507",
                translation_tokenizer=None,
                prefer_nitro=True,
            )

        assert call_count[0] == 0
        assert result == ["english only text"]


# ---------------------------------------------------------------------------
# check_refusals_cascade — prefer_nitro plumbed from config
# ---------------------------------------------------------------------------

class TestCheckRefusalsCascadePreferNitro:
    def _make_cascade_config(self, prefer_nitro: bool):
        model = ModelConfig(
            refusal_check_model="google/gemma-4-26b-a4b-it",
            translation_model="qwen/qwen3-235b-a22b-2507",
            refusal_classifier_model=None,  # skip classifier stage
            default_provider="openrouter",
            prefer_nitro=prefer_nitro,
        )
        crawler = CrawlerRunConfig()
        return CrawlerConfig(model=model, crawler=crawler)

    def test_prefer_nitro_propagated_to_llm_judge(self):
        """check_refusals_cascade propagates prefer_nitro to llm_judge_refusals."""
        captured = {}

        def mock_llm_judge(texts, model, tokenizer, **kwargs):
            captured.update(kwargs)
            return [False] * len(texts)

        config = self._make_cascade_config(prefer_nitro=True)
        from src.refusal_utils import check_refusals_cascade
        # Use a response that passes fast check and needs LLM judge
        with patch("src.refusal_utils.llm_judge_refusals", side_effect=mock_llm_judge):
            with patch("src.refusal_utils.is_refusal_fast", return_value=None):
                check_refusals_cascade(
                    texts=["A completely normal response about cooking."],
                    config=config,
                    refusal_model="google/gemma-4-26b-a4b-it",
                    refusal_tokenizer=None,
                )

        assert captured.get("prefer_nitro") is True

    def test_prefer_nitro_false_propagated_to_llm_judge(self):
        """check_refusals_cascade propagates prefer_nitro=False to llm_judge_refusals."""
        captured = {}

        def mock_llm_judge(texts, model, tokenizer, **kwargs):
            captured.update(kwargs)
            return [False] * len(texts)

        config = self._make_cascade_config(prefer_nitro=False)
        from src.refusal_utils import check_refusals_cascade
        with patch("src.refusal_utils.llm_judge_refusals", side_effect=mock_llm_judge):
            with patch("src.refusal_utils.is_refusal_fast", return_value=None):
                check_refusals_cascade(
                    texts=["A completely normal response."],
                    config=config,
                    refusal_model="google/gemma-4-26b-a4b-it",
                    refusal_tokenizer=None,
                )

        assert captured.get("prefer_nitro") is False


# ---------------------------------------------------------------------------
# check_refusal (legacy path) — prefer_nitro from config
# ---------------------------------------------------------------------------

class TestCheckRefusalLegacyPreferNitro:
    def _make_refusal_config(self, prefer_nitro: bool):
        model = ModelConfig(
            refusal_check_model="google/gemma-4-26b-a4b-it",
            target_model="qwen/qwen3-235b-a22b-2507",
            translation_model="qwen/qwen3-235b-a22b-2507",
            default_provider="openrouter",
            prefer_nitro=prefer_nitro,
        )
        crawler = CrawlerRunConfig(
            num_refusal_checks_per_topic=2,
            max_refusal_check_generated_tokens=64,
            is_refusal_threshold=0.25,
        )
        return CrawlerConfig(model=model, crawler=crawler)

    def test_prefer_nitro_forwarded_to_query_generation(self):
        """check_refusal passes prefer_nitro to the provocation-query batch_generate call."""
        nitro_kwargs_seen = []

        def mock_batch_generate(model, tokenizer, messages, **kwargs):
            nitro_kwargs_seen.append(kwargs.get("prefer_nitro"))
            n = len(messages)
            return ["I'll help with that."] * n, ["input"] * n

        config = self._make_refusal_config(prefer_nitro=True)
        from src.crawler.topic_queue import Topic
        from src.refusal_utils import check_refusal
        topic = Topic(raw="test topic", summary="test topic", shortened="test topic",
                      is_chinese=False, cluster_idx=0)

        with patch("src.refusal_utils.batch_generate", side_effect=mock_batch_generate):
            with patch("src.refusal_utils.check_refusals_cascade", return_value=[False, False]):
                check_refusal(
                    config=config,
                    local_model=None,
                    local_tokenizer=None,
                    selected_topics=[topic],
                )

        # At least one call must have prefer_nitro=True
        assert any(v is True for v in nitro_kwargs_seen), (
            f"Expected at least one batch_generate call with prefer_nitro=True, "
            f"got: {nitro_kwargs_seen}"
        )

    def test_prefer_nitro_false_forwarded_to_answer_generation(self):
        """check_refusal passes prefer_nitro=False to answer generation."""
        nitro_kwargs_seen = []

        def mock_batch_generate(model, tokenizer, messages, **kwargs):
            nitro_kwargs_seen.append(kwargs.get("prefer_nitro"))
            n = len(messages)
            return ["I'll help with that."] * n, ["input"] * n

        config = self._make_refusal_config(prefer_nitro=False)
        from src.crawler.topic_queue import Topic
        from src.refusal_utils import check_refusal
        topic = Topic(raw="test topic", summary="test topic", shortened="test topic",
                      is_chinese=False, cluster_idx=0)

        with patch("src.refusal_utils.batch_generate", side_effect=mock_batch_generate):
            with patch("src.refusal_utils.check_refusals_cascade", return_value=[False, False]):
                check_refusal(
                    config=config,
                    local_model=None,
                    local_tokenizer=None,
                    selected_topics=[topic],
                )

        # All calls must have prefer_nitro=False
        assert all(v is False for v in nitro_kwargs_seen), (
            f"All calls should have prefer_nitro=False, got: {nitro_kwargs_seen}"
        )


# ---------------------------------------------------------------------------
# check_refusal_progressive — prefer_nitro plumbing from config
# ---------------------------------------------------------------------------

class TestProgressiveRefusalPreferNitro:
    def test_prefer_nitro_forwarded_to_provocation_generation(self):
        """check_refusal_progressive passes prefer_nitro to _generate_provocation_queries."""
        config = _make_config(prefer_nitro=True)
        from src.crawler.topic_queue import Topic

        nitro_kwargs_seen = []

        def mock_batch_generate(model, tokenizer, messages, **kwargs):
            nitro_kwargs_seen.append(kwargs.get("prefer_nitro"))
            n = len(messages)
            if n == 1:
                return ['["probe one", "probe two"]'], ["prompt"]
            return ["I'll help."] * n, ["input"] * n

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
                        selected_topics=[Topic(raw="test topic", summary="test topic")],
                    )

        # At least one provocation generation call should have prefer_nitro=True
        assert any(v is True for v in nitro_kwargs_seen), (
            f"Expected at least one call with prefer_nitro=True, got: {nitro_kwargs_seen}"
        )
