"""Unit tests for S0b — validate cluster_idx in _parse_grouping_response.

Tests:
- test_negative_cluster_idx_downgrades_to_head
- test_out_of_range_positive_cluster_idx_downgrades_to_head
- test_valid_known_head_reference_preserved
- test_queue_never_receives_invalid_cluster_idx
"""

import json
import pytest

from src.crawler.topic_queue import Topic, TopicQueue
from src.crawler.grouping_pipeline import _parse_grouping_response


def _make_topic(raw: str, summary: str = None, shortened: str = None, **kwargs) -> Topic:
    return Topic(
        raw=raw,
        summary=summary,
        shortened=shortened or raw,
        english=raw,
        **kwargs,
    )


class TestNegativeClusterIdxDowngrades:
    """Out-of-range negative cluster_idx must downgrade to new head, not stay invalid."""

    def test_negative_cluster_idx_out_of_range_downgrades_to_head(self):
        """cluster_idx=-2 when known_heads has only 1 entry must downgrade to a new head."""
        known_heads = [
            _make_topic("Taiwan", summary="Taiwan sovereignty", is_head=True, cluster_idx=0),
        ]
        topics = [
            _make_topic("Some other topic"),
        ]
        # LLM returns cluster_idx=-2, but known_heads only has index 0 (valid: -1)
        response = json.dumps({
            "groups": [],
            "duplicates": [
                {
                    "original_index": 0,
                    "cluster_idx": -2,
                    "is_head": False,
                    "summary": "Some other topic",
                },
            ],
            "skipped": [],
        })

        result = _parse_grouping_response(response, topics, known_heads=known_heads)
        # Must be downgraded to a new head, NOT left with cluster_idx=-2
        assert result[0].is_head is True, (
            f"Out-of-range negative cidx must downgrade to new head, got is_head={result[0].is_head}"
        )
        assert result[0].cluster_idx >= 0, (
            f"Downgraded topic must have non-negative cluster_idx, got {result[0].cluster_idx}"
        )
        assert result[0].summary is not None

    def test_negative_cluster_idx_within_range_resolves_correctly(self):
        """cluster_idx=-1 with 1 known head resolves to the known head's cluster_idx."""
        known_heads = [
            _make_topic("Taiwan", summary="Taiwan sovereignty", is_head=True, cluster_idx=7),
        ]
        topics = [
            _make_topic("Taiwan political status"),
        ]
        response = json.dumps({
            "groups": [],
            "duplicates": [
                {
                    "original_index": 0,
                    "cluster_idx": -1,  # resolves to known_heads[0] → cluster_idx=7
                    "is_head": False,
                    "summary": "Taiwan sovereignty",
                },
            ],
            "skipped": [],
        })

        result = _parse_grouping_response(response, topics, known_heads=known_heads)
        # Must resolve to the head's actual cluster_idx
        assert result[0].is_head is False
        assert result[0].cluster_idx == 7


class TestOutOfRangePositiveClusterIdx:
    """Unresolvable positive cluster_idx (not in local_to_global or known_heads)
    must downgrade to new head."""

    def test_out_of_range_positive_cluster_idx_downgrades_to_head(self):
        """cluster_idx=999 with no matching group or known head must downgrade."""
        known_heads = [
            _make_topic("Taiwan", summary="Taiwan sovereignty", is_head=True, cluster_idx=0),
        ]
        topics = [
            _make_topic("Some mystery topic"),
        ]
        response = json.dumps({
            "groups": [],
            "duplicates": [
                {
                    "original_index": 0,
                    "cluster_idx": 999,  # not in local_to_global, not a known head
                    "is_head": False,
                    "summary": "Some mystery topic",
                },
            ],
            "skipped": [],
        })

        result = _parse_grouping_response(response, topics, known_heads=known_heads)
        # Must be downgraded to a new head
        assert result[0].is_head is True
        assert result[0].cluster_idx >= 0
        assert result[0].cluster_idx != 999

    def test_valid_known_head_direct_reference_preserved(self):
        """A duplicate pointing directly at a known head's cluster_idx (positive)
        must be preserved as a non-head, not downgraded."""
        known_heads = [
            _make_topic("Taiwan", summary="Taiwan sovereignty", is_head=True, cluster_idx=5),
        ]
        topics = [
            _make_topic("Taiwan political status"),
        ]
        # Direct reference to known head's cluster_idx=5 (not using negative convention)
        response = json.dumps({
            "groups": [],
            "duplicates": [
                {
                    "original_index": 0,
                    "cluster_idx": 5,  # directly references known head at cluster_idx=5
                    "is_head": False,
                    "summary": "Taiwan sovereignty",
                },
            ],
            "skipped": [],
        })

        result = _parse_grouping_response(response, topics, known_heads=known_heads)
        # Must be preserved as a non-head with cluster_idx=5
        assert result[0].is_head is False
        assert result[0].cluster_idx == 5


class TestValidKnownHeadReference:
    """Valid negative references must still resolve correctly."""

    def test_valid_known_head_reference_preserved(self):
        """Negative cluster_idx within valid range must resolve to the known head's cluster_idx."""
        known_heads = [
            _make_topic("Head 0", summary="Head 0", is_head=True, cluster_idx=0),
            _make_topic("Head 1", summary="Head 1", is_head=True, cluster_idx=1),
        ]
        topics = [
            _make_topic("Dup of Head 1"),
        ]
        response = json.dumps({
            "groups": [],
            "duplicates": [
                {
                    "original_index": 0,
                    "cluster_idx": -2,  # known_heads[1] → cluster_idx=1
                    "is_head": False,
                    "summary": "Head 1",
                },
            ],
            "skipped": [],
        })

        result = _parse_grouping_response(response, topics, known_heads=known_heads)
        assert result[0].is_head is False
        assert result[0].cluster_idx == 1


class TestQueueNeverReceivesInvalidClusterIdx:
    """TopicQueue.append_to_cluster must raise loudly on invalid cluster_idx."""

    def test_queue_raises_on_negative_cluster_idx(self):
        """append_to_cluster raises ValueError when cluster_idx < 0."""
        queue = TopicQueue()
        # Create a head first so cluster_topics is non-empty
        head = Topic(raw="head", is_head=True, cluster_idx=0, summary="head")
        queue.add_new_cluster_head(head)

        bad_topic = Topic(raw="bad", is_head=False, cluster_idx=-1, summary="bad")
        with pytest.raises(ValueError, match="invalid cluster_idx"):
            queue.append_to_cluster(bad_topic)

    def test_queue_raises_on_out_of_range_cluster_idx(self):
        """append_to_cluster raises ValueError when cluster_idx >= len(cluster_topics)."""
        queue = TopicQueue()
        head = Topic(raw="head", is_head=True, cluster_idx=0, summary="head")
        queue.add_new_cluster_head(head)

        bad_topic = Topic(raw="bad", is_head=False, cluster_idx=5, summary="bad")
        with pytest.raises(ValueError, match="invalid cluster_idx"):
            queue.append_to_cluster(bad_topic)

    def test_queue_raises_on_none_cluster_idx(self):
        """append_to_cluster raises ValueError when cluster_idx is None."""
        queue = TopicQueue()
        head = Topic(raw="head", is_head=True, cluster_idx=0, summary="head")
        queue.add_new_cluster_head(head)

        bad_topic = Topic(raw="bad", is_head=False, cluster_idx=None, summary="bad")
        with pytest.raises(ValueError, match="invalid cluster_idx"):
            queue.append_to_cluster(bad_topic)

    def test_queue_accepts_valid_cluster_idx(self):
        """append_to_cluster succeeds when cluster_idx is within range."""
        queue = TopicQueue()
        head = Topic(raw="head", is_head=True, cluster_idx=0, summary="head")
        queue.add_new_cluster_head(head)

        good_topic = Topic(raw="member", is_head=False, cluster_idx=0, summary="head")
        queue.append_to_cluster(good_topic)
        assert len(queue.cluster_topics[0]) == 2

    def test_parse_grouping_response_invalid_not_propagated_to_queue(self):
        """The _parse_grouping_response downgrade ensures TopicQueue never sees
        invalid cluster_idx, so incoming_batch can safely process the result.
        """
        known_heads = [
            _make_topic("Head", summary="head", is_head=True, cluster_idx=0),
        ]
        topics = [
            _make_topic("Unresolvable dup"),
        ]
        # cluster_idx=-99 is out of range (only 1 known head)
        response = json.dumps({
            "groups": [],
            "duplicates": [
                {"original_index": 0, "cluster_idx": -99, "is_head": False, "summary": "x"},
            ],
            "skipped": [],
        })
        result = _parse_grouping_response(response, topics, known_heads=known_heads)

        # All topics must have non-negative cluster_idx
        for t in result:
            if t.is_head or t.cluster_idx != -1:  # -1 is the valid sentinel for skipped
                assert t.cluster_idx >= 0, (
                    f"Topic {t.raw!r} has invalid cluster_idx={t.cluster_idx}"
                )
