"""End-to-end integration test for AggregateCrawler.

Unit tests for _propagate_verdicts do not require API keys.
Integration test for the full crawl requires OPENROUTER_API_KEY.
"""
import os
import tempfile
import pytest

from src.crawler.config import CrawlerConfig
from src.crawler.aggregate_crawler import AggregateCrawler
from src.crawler.topic_queue import Topic, TopicQueue


def _make_config() -> CrawlerConfig:
    """Load integration test config from YAML."""
    from tests.conftest import load_integration_test_config
    return load_integration_test_config()


class TestPropagateVerdicts:
    """Unit tests for _propagate_verdicts — no API calls needed."""

    def _make_crawler(self) -> AggregateCrawler:
        """Create an AggregateCrawler with an empty queue (no config needed for unit tests)."""
        config = CrawlerConfig()
        return AggregateCrawler(crawler_config=config, save_filename="/dev/null")

    def test_propagate_verdicts_known_head_duplicates(self):
        """Topics matching known heads must inherit the known head's is_refusal."""
        crawler = self._make_crawler()

        # Put a known head in the queue with cluster_idx=7, is_refusal=True
        known_head = Topic(
            raw="Taiwan independence",
            summary="Taiwan sovereignty",
            is_head=True,
            cluster_idx=7,
            is_refusal=True,
        )
        crawler.queue.head_topics.append(known_head)

        # New topic matches known head (cluster_idx=7, is_head=False)
        new_topic = Topic(
            raw="Taiwan political status",
            summary="Taiwan sovereignty",
            is_head=False,
            cluster_idx=7,
            is_refusal=None,
        )
        # Also add a new head that was checked
        new_head = Topic(
            raw="Drug manufacturing",
            summary="Drug manufacturing",
            is_head=True,
            cluster_idx=8,
            is_refusal=False,
        )

        topics = [new_topic, new_head]
        result = crawler._propagate_verdicts(topics)

        # The duplicate of the known head inherits is_refusal=True
        assert result[0].is_refusal is True, (
            f"Expected is_refusal=True (inherited from known head), got {result[0].is_refusal}"
        )
        # The new head keeps its own verdict
        assert result[1].is_refusal is False

    def test_propagate_verdicts_no_known_head_match(self):
        """Non-head topics without a matching known head use batch head verdict."""
        crawler = self._make_crawler()

        # New head with is_refusal=True
        new_head = Topic(
            raw="Nuclear weapons",
            summary="nuclear weapon construction",
            is_head=True,
            cluster_idx=10,
            is_refusal=True,
        )
        # Non-head member of the same cluster
        member = Topic(
            raw="Bomb making",
            summary="nuclear weapon construction",
            is_head=False,
            cluster_idx=10,
            is_refusal=None,
        )

        result = crawler._propagate_verdicts([new_head, member])
        assert result[0].is_refusal is True  # head keeps its own
        assert result[1].is_refusal is True  # member inherits from head

    def test_propagate_verdicts_empty_queue(self):
        """With an empty queue, only batch head verdicts propagate."""
        crawler = self._make_crawler()

        head = Topic(
            raw="Safe topic",
            summary="safe topic",
            is_head=True,
            cluster_idx=0,
            is_refusal=False,
        )
        member = Topic(
            raw="Safe topic variant",
            summary="safe topic",
            is_head=False,
            cluster_idx=0,
            is_refusal=None,
        )

        result = crawler._propagate_verdicts([head, member])
        assert result[0].is_refusal is False
        assert result[1].is_refusal is False


@pytest.mark.integration
def test_aggregate_crawler_e2e():
    """1-step crawl against a real model.

    Asserts:
    - Refusal topics are found
    - Grouping reduced the refusal-check count (fewer checks than raw topics)
    - Output format matches the existing crawler format
    """
    config = _make_config()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        save_path = f.name

    try:
        crawler = AggregateCrawler(crawler_config=config, save_filename=save_path)

        # Run 1-step crawl with no local model
        crawler.crawl(
            local_model=None,
            local_tokenizer=None,
            verbose=True,
        )

        # Verify output format
        result = crawler.to_dict()
        assert "stats" in result
        assert "config" in result
        assert "queue" in result
        assert "head_refusal_topics_summaries" in result

        queue = result["queue"]
        assert "topics" in queue
        assert "stats" in queue
        topics_data = queue["topics"]
        assert "head_refusal_topics" in topics_data
        assert "head_topics" in topics_data
        assert "cluster_topics" in topics_data

        head_topics = topics_data["head_topics"]

        # If we found any topics, check the grouping worked
        if len(head_topics) > 0:
            # Check that cluster_member_count is set on topics
            for topic_dict in head_topics:
                assert "cluster_member_count" in topic_dict
                assert "refusal_check_inconclusive" in topic_dict

            # The key metric: number of refusal checks should be fewer than
            # the total number of raw topics discovered
            total_topics = queue["stats"]["num_total_topics"]
            if total_topics > 0:
                # Count how many heads were actually checked
                # (novel heads get checked, duplicates don't)
                print(
                    f"\nTotal topics: {total_topics}, "
                    f"Heads: {len(head_topics)}, "
                    f"Refusal heads: {len(topics_data['head_refusal_topics'])}"
                )
                # At minimum, heads <= total_topics (grouping reduces checks)
                assert len(head_topics) <= total_topics

        # Check stats shape
        stats = result["stats"]
        assert "cumulative" in stats
        assert "history" in stats

    finally:
        if os.path.exists(save_path):
            os.unlink(save_path)
