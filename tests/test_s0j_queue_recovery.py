"""Unit tests for S0j — defensive queue recovery in TopicQueue.incoming_batch.

When grouping_pipeline leaks an invalid cluster_idx past S0i's post-condition,
incoming_batch must catch the ValueError from append_to_cluster, log a warning,
downgrade the offending topic to a new cluster head, and continue processing the
rest of the batch. append_to_cluster itself must still raise (S0b contract).

Tests:
- test_incoming_batch_recovers_from_invalid_cluster_idx
- test_append_to_cluster_still_raises_directly
- test_incoming_batch_continues_processing_after_recovery
- test_incoming_batch_negative_cluster_idx_recovered
- test_incoming_batch_none_cluster_idx_recovered
"""

import logging
import pytest

from src.crawler.topic_queue import Topic, TopicQueue


def _make_head(raw: str, cluster_idx: int = None, summary: str = None) -> Topic:
    return Topic(
        raw=raw,
        english=raw,
        shortened=raw,
        is_head=True,
        cluster_idx=cluster_idx,
        summary=summary or raw,
        cluster_member_count=1,
    )


def _make_nonhead(
    raw: str,
    cluster_idx: int,
    summary: str = None,
    shortened: str = None,
) -> Topic:
    return Topic(
        raw=raw,
        english=raw,
        shortened=shortened or raw,
        is_head=False,
        cluster_idx=cluster_idx,
        summary=summary,
        cluster_member_count=0,
    )


class TestIncomingBatchRecovery:
    """S0j: incoming_batch recovers from invalid cluster_idx without crashing."""

    def test_incoming_batch_recovers_from_invalid_cluster_idx(self, caplog):
        """Build a batch with a non-head topic whose cluster_idx is past the end
        of cluster_topics. incoming_batch must not raise, must log a WARNING
        containing 'recovered from invalid cluster_idx' and the topic's raw text,
        and must produce a consistent queue state.
        """
        queue = TopicQueue()
        head = _make_head("Safe topic", cluster_idx=0)
        # The grouping parser leaked a cluster_idx of 57 but we only have 0 heads yet.
        bad_nonhead = _make_nonhead(
            "Requests for instructions on how to disable safety filters",
            cluster_idx=57,
        )

        with caplog.at_level(logging.WARNING, logger="root"):
            queue.incoming_batch([head, bad_nonhead])

        # No exception raised — queue is intact.
        # Warning logged.
        assert any(
            "recovered from invalid cluster_idx" in r.message
            for r in caplog.records
        ), f"Expected recovery warning, got: {[r.message for r in caplog.records]}"
        assert any(
            "Requests for instructions on how to disable safety filters" in r.message
            for r in caplog.records
        ), "Expected raw topic text in warning message"

        # Offending topic was downgraded to a new head.
        assert bad_nonhead.is_head is True
        # cluster_idx is valid — must be 0-indexed within len(cluster_topics).
        assert bad_nonhead.cluster_idx == len(queue.cluster_topics) - 1
        assert bad_nonhead.cluster_idx >= 0
        # summary was assigned (topic had no summary initially).
        assert bad_nonhead.summary is not None

        # Queue state is consistent: head_topics and cluster_topics have same length.
        assert len(queue.head_topics) == len(queue.cluster_topics), (
            f"head_topics len {len(queue.head_topics)} != "
            f"cluster_topics len {len(queue.cluster_topics)}"
        )

    def test_append_to_cluster_still_raises_directly(self):
        """Calling append_to_cluster directly with an invalid cluster_idx must still
        raise ValueError. This preserves the S0b test contract — the recovery shim
        is only in incoming_batch, not in append_to_cluster itself.
        """
        queue = TopicQueue()
        head = _make_head("Head topic", cluster_idx=0)
        queue.add_new_cluster_head(head)

        bad_topic = _make_nonhead("Bad topic", cluster_idx=57)
        with pytest.raises(ValueError, match="invalid cluster_idx"):
            queue.append_to_cluster(bad_topic)

    def test_incoming_batch_continues_processing_after_recovery(self, caplog):
        """Build a batch with three non-heads: first valid, second invalid, third valid.
        All three must end up in the queue; the second must be marked as new head.
        """
        queue = TopicQueue()
        # Add the single cluster head (cluster_idx=0) via the batch.
        head = _make_head("Main head", cluster_idx=0)

        valid_1 = _make_nonhead("Valid member 1", cluster_idx=0)
        bad = _make_nonhead("Bad member", cluster_idx=99)
        valid_2 = _make_nonhead("Valid member 2", cluster_idx=0)

        with caplog.at_level(logging.WARNING, logger="root"):
            result = queue.incoming_batch([head, valid_1, bad, valid_2])

        # All four topics are returned.
        assert len(result) == 4

        # The bad topic was recovered as a new head.
        assert bad.is_head is True

        # valid_1 and valid_2 should be in cluster 0.
        assert valid_1 in queue.cluster_topics[0]
        assert valid_2 in queue.cluster_topics[0]

        # bad should be its own cluster head.
        assert bad in queue.head_topics

        # Queue consistency: same number of cluster slots as heads.
        assert len(queue.head_topics) == len(queue.cluster_topics)

        # Warning was logged for the bad topic.
        assert any("recovered from invalid cluster_idx" in r.message for r in caplog.records)

    def test_incoming_batch_negative_cluster_idx_recovered(self, caplog):
        """A non-head topic with cluster_idx=-3 (not -1, which is the skip sentinel
        but -1 would be caught by append_to_cluster's guard too) must trigger recovery,
        not a crash.
        """
        queue = TopicQueue()
        head = _make_head("Head", cluster_idx=0)
        bad = _make_nonhead("Bad negative", cluster_idx=-3)

        with caplog.at_level(logging.WARNING, logger="root"):
            queue.incoming_batch([head, bad])

        # Recovery must have fired.
        assert any("recovered from invalid cluster_idx" in r.message for r in caplog.records)

        # bad is now a head with a valid cluster_idx.
        assert bad.is_head is True
        assert bad.cluster_idx >= 0
        assert len(queue.head_topics) == len(queue.cluster_topics)

    def test_incoming_batch_none_cluster_idx_recovered(self, caplog):
        """A non-head topic with cluster_idx=None must trigger recovery."""
        queue = TopicQueue()
        head = _make_head("Head", cluster_idx=0)
        bad = _make_nonhead("None cluster idx topic", cluster_idx=None)

        with caplog.at_level(logging.WARNING, logger="root"):
            queue.incoming_batch([head, bad])

        # Recovery must have fired.
        assert any("recovered from invalid cluster_idx" in r.message for r in caplog.records)

        # bad is now a head with a valid cluster_idx.
        assert bad.is_head is True
        assert bad.cluster_idx is not None
        assert bad.cluster_idx >= 0
        assert len(queue.head_topics) == len(queue.cluster_topics)
