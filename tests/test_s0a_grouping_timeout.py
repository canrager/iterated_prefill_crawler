"""Unit tests for S0a — grouping timeout must fail loudly, not drop batch.

Tests:
- test_grouping_timeout_reraises_by_default
- test_grouping_timeout_preserves_batch_when_configured
- test_grouping_timeout_does_not_silently_drop_topics
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.crawler.config import CrawlerConfig
from src.crawler.topic_queue import Topic
from src.crawler.grouping_pipeline import summarize_group_dedup, _preserve_batch_as_heads
from src.exceptions import APITimeoutError


def _make_topic(raw: str, shortened: str = None, summary: str = None) -> Topic:
    return Topic(
        raw=raw,
        shortened=shortened or raw,
        summary=summary,
        english=raw,
    )


def _make_config(preserves_batch: bool = False) -> CrawlerConfig:
    config = CrawlerConfig()
    config.model.summarization_model = "openrouter:test/model"
    config.crawler.semantic_group_batch_size = 100
    config.crawler.grouping_timeout_preserves_batch = preserves_batch
    return config


class TestGroupingTimeoutReraises:
    """Default behaviour: APITimeoutError propagates to the caller."""

    def test_grouping_timeout_reraises_by_default(self):
        """With grouping_timeout_preserves_batch=False, APITimeoutError must propagate."""
        topics = [
            _make_topic("Taiwan independence"),
            _make_topic("Tiananmen Square protests"),
        ]
        config = _make_config(preserves_batch=False)

        async def _fake_timeout(**kwargs):
            raise APITimeoutError("timeout")

        with patch(
            "src.openrouter_utils.async_query_openrouter",
            new=_fake_timeout,
        ):
            # get_provider_client_kwargs must not fail
            with patch(
                "src.crawler.grouping_pipeline.get_provider_client_kwargs",
                return_value=("test/model", {}),
            ):
                with pytest.raises(APITimeoutError):
                    summarize_group_dedup(
                        topics=topics,
                        known_heads=[],
                        config=config,
                        verbose=False,
                    )

    def test_grouping_timeout_does_not_silently_drop_topics(self):
        """When grouping_timeout_preserves_batch=False, the exception propagates.
        Topics are NOT silently discarded (the exception is the signal).
        """
        topics = [_make_topic("Topic A"), _make_topic("Topic B")]
        config = _make_config(preserves_batch=False)

        async def _fake_timeout(**kwargs):
            raise APITimeoutError("timeout")

        with patch(
            "src.openrouter_utils.async_query_openrouter",
            new=_fake_timeout,
        ):
            with patch(
                "src.crawler.grouping_pipeline.get_provider_client_kwargs",
                return_value=("test/model", {}),
            ):
                caught = False
                try:
                    summarize_group_dedup(
                        topics=topics,
                        known_heads=[],
                        config=config,
                        verbose=False,
                    )
                except APITimeoutError:
                    caught = True

                assert caught, "APITimeoutError must not be swallowed"


class TestGroupingTimeoutPreservesBatch:
    """Opt-in preservation path: grouping_timeout_preserves_batch=True."""

    def test_grouping_timeout_preserves_batch_when_configured(self):
        """With grouping_timeout_preserves_batch=True, each topic in the timed-out
        batch is preserved as a new head with a fallback summary (no exception raised).
        """
        topics = [
            _make_topic("Taiwan independence"),
            _make_topic("Drug synthesis"),
        ]
        config = _make_config(preserves_batch=True)

        async def _fake_timeout(**kwargs):
            raise APITimeoutError("timeout")

        with patch(
            "src.openrouter_utils.async_query_openrouter",
            new=_fake_timeout,
        ):
            with patch(
                "src.crawler.grouping_pipeline.get_provider_client_kwargs",
                return_value=("test/model", {}),
            ):
                # Should NOT raise
                result = summarize_group_dedup(
                    topics=topics,
                    known_heads=[],
                    config=config,
                    verbose=False,
                )

        # All topics preserved as heads
        assert len(result) == 2
        for t in result:
            assert t.is_head is True, f"Expected is_head=True for {t.raw}"
            assert t.summary is not None, f"Expected non-None summary for {t.raw}"
            assert t.cluster_idx is not None and t.cluster_idx >= 0

    def test_preserved_batch_has_valid_cluster_indices(self):
        """Preserved topics get contiguous cluster_idx starting from len(known_heads)."""
        known_heads = [
            Topic(raw="existing", is_head=True, cluster_idx=0, summary="existing"),
        ]
        batch = [
            _make_topic("New A"),
            _make_topic("New B"),
        ]
        _preserve_batch_as_heads(batch, known_heads)

        # Must start from len(known_heads) = 1
        assert batch[0].cluster_idx == 1
        assert batch[1].cluster_idx == 2
        assert all(t.is_head for t in batch)
        assert all(t.summary is not None for t in batch)
        assert all(t.cluster_member_count == 1 for t in batch)

    def test_preserved_batch_summaries_are_non_empty(self):
        """Preserved topics use shortened or raw as summary — never None.
        This ensures they pass the downstream `if t.summary is not None` filter.
        """
        batch = [
            Topic(raw="Some topic", shortened="short topic"),
            Topic(raw="Another topic", shortened=None),
        ]
        _preserve_batch_as_heads(batch, known_heads=[])

        assert batch[0].summary == "short topic"
        assert batch[1].summary == "Another topic"
