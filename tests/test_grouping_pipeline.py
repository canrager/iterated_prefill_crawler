"""Integration tests for the grouping pipeline.

Marked with @pytest.mark.integration — run with:
    pytest tests/test_grouping_pipeline.py -v -s -m integration
"""
import json
import pytest

from src.crawler.config import CrawlerConfig
from src.crawler.topic_queue import Topic
from src.crawler.grouping_pipeline import (
    summarize_group_dedup,
    _build_known_heads_section,
    _build_new_topics_section,
    _parse_grouping_response,
)



def _make_topic(raw: str, summary: str = None, shortened: str = None, **kwargs) -> Topic:
    """Helper to create a Topic with sensible defaults."""
    return Topic(
        raw=raw,
        summary=summary,
        shortened=shortened or raw,
        english=raw,
        **kwargs,
    )


class TestParseGroupingResponse:
    """Unit tests for _parse_grouping_response — no LLM needed."""

    def test_basic_grouping(self):
        """Five variants should produce 2 groups, not 5."""
        topics = [
            _make_topic("Taiwan independence movement"),
            _make_topic("Taiwan political status"),
            _make_topic("Tiananmen Square protests"),
            _make_topic("June Fourth incident"),
            _make_topic("Drug manufacturing"),
        ]
        response = json.dumps({
            "groups": [
                {
                    "cluster_idx": 0,
                    "summary": "Taiwan sovereignty",
                    "member_indices": [0, 1],
                    "is_head": True,
                },
                {
                    "cluster_idx": 1,
                    "summary": "Tiananmen protests",
                    "member_indices": [2, 3],
                    "is_head": True,
                },
                {
                    "cluster_idx": 2,
                    "summary": "Drug manufacturing",
                    "member_indices": [4],
                    "is_head": True,
                },
            ],
            "duplicates": [],
            "skipped": [],
        })

        result = _parse_grouping_response(response, topics, known_heads=[])
        # Group 0: Taiwan
        assert result[0].is_head is True
        assert result[0].cluster_idx == 0
        assert result[0].summary == "Taiwan sovereignty"
        assert result[0].cluster_member_count == 2
        assert result[1].is_head is False
        assert result[1].cluster_idx == 0

        # Group 1: Tiananmen
        assert result[2].is_head is True
        assert result[2].cluster_idx == 1
        assert result[3].is_head is False

        # Group 2: Drugs
        assert result[4].is_head is True
        assert result[4].cluster_idx == 2

    def test_duplicate_of_known_head(self):
        """A topic matching a known head should be is_head=False with the head's cluster_idx."""
        known_heads = [
            _make_topic("Taiwan", summary="Taiwan sovereignty", is_head=True, cluster_idx=0),
        ]
        topics = [
            _make_topic("Taiwan independence"),
        ]
        response = json.dumps({
            "groups": [],
            "duplicates": [
                {
                    "original_index": 0,
                    "cluster_idx": 0,
                    "is_head": False,
                    "summary": "Taiwan sovereignty",
                },
            ],
            "skipped": [],
        })

        result = _parse_grouping_response(response, topics, known_heads=known_heads)
        assert result[0].is_head is False
        assert result[0].cluster_idx == 0
        assert result[0].summary == "Taiwan sovereignty"

    def test_skipped_preamble_garbage(self):
        """Preamble/garbage topics should be filtered (summary=None)."""
        topics = [
            _make_topic("As an AI, I cannot help with illegal activities"),
            _make_topic("Drug manufacturing"),
        ]
        response = json.dumps({
            "groups": [
                {
                    "cluster_idx": 0,
                    "summary": "Drug manufacturing",
                    "member_indices": [1],
                    "is_head": True,
                },
            ],
            "duplicates": [],
            "skipped": [0],
        })

        result = _parse_grouping_response(response, topics, known_heads=[])
        # Topic 0 is skipped
        assert result[0].summary is None
        assert result[0].cluster_idx == -1
        # Topic 1 is a new head
        assert result[1].is_head is True
        assert result[1].summary == "Drug manufacturing"

    def test_cluster_member_count(self):
        """cluster_member_count should reflect group size."""
        topics = [
            _make_topic("Taiwan independence"),
            _make_topic("Taiwan status"),
            _make_topic("Taiwan sovereignty"),
        ]
        response = json.dumps({
            "groups": [
                {
                    "cluster_idx": 0,
                    "summary": "Taiwan sovereignty",
                    "member_indices": [0, 1, 2],
                    "is_head": True,
                },
            ],
            "duplicates": [],
            "skipped": [],
        })

        result = _parse_grouping_response(response, topics, known_heads=[])
        assert result[0].cluster_member_count == 3
        assert result[1].cluster_member_count == 3
        assert result[2].cluster_member_count == 3

    def test_malformed_json_fallback(self):
        """Malformed JSON should fall back to treating all topics as new heads."""
        topics = [
            _make_topic("Topic A"),
            _make_topic("Topic B"),
        ]
        result = _parse_grouping_response("not json at all", topics, known_heads=[])
        # All topics become heads
        assert all(t.is_head for t in result)
        assert len(result) == 2

    def test_unassigned_topic_gets_new_cluster(self):
        """Topics not mentioned in the LLM response get a new cluster."""
        topics = [
            _make_topic("Mentioned topic"),
            _make_topic("Unmentioned topic"),
        ]
        response = json.dumps({
            "groups": [
                {
                    "cluster_idx": 0,
                    "summary": "Mentioned topic",
                    "member_indices": [0],
                    "is_head": True,
                },
            ],
            "duplicates": [],
            "skipped": [],
        })

        result = _parse_grouping_response(response, topics, known_heads=[])
        # Topic 0 is assigned, topic 1 gets a new cluster
        assert result[0].cluster_idx == 0
        assert result[1].is_head is True
        assert result[1].cluster_idx == 1  # new cluster


class TestNoOverMerging:
    """The pipeline must deduplicate rephrases, NOT categorize distinct topics."""

    def test_abortion_not_grouped_with_sensitive_topics(self):
        """Abortion is a specific topic, not a subcategory of 'sensitive topics'."""
        # Simulate Kimi incorrectly grouping them together
        topics = [
            _make_topic("Abortion"),
            _make_topic("Controversial sensitive topics"),
            _make_topic("Euthanasia"),
        ]
        # BAD response: all three merged into one group
        bad_response = json.dumps({
            "groups": [
                {
                    "cluster_idx": 0,
                    "summary": "Controversial sensitive topics",
                    "member_indices": [0, 1, 2],
                    "is_head": True,
                },
            ],
            "duplicates": [],
            "skipped": [],
        })
        # Even if Kimi returns this, the test documents that it's WRONG behavior.
        # This test exists to catch regressions if we weaken the prompt.
        result = _parse_grouping_response(bad_response, topics, known_heads=[])
        # With a bad LLM response, parse will follow it — the fix is the prompt,
        # not the parser. This test documents what we DON'T want.
        # The real guard is the integration test below.

    @pytest.mark.integration
    def test_distinct_topics_stay_separate_live(self):
        """Live Kimi call: distinct topics must not be collapsed into categories."""
        from tests.conftest import load_integration_test_config
        config = load_integration_test_config()

        topics = [
            _make_topic("Abortion"),
            _make_topic("Euthanasia"),
            _make_topic("Controversial sensitive topics"),
            _make_topic("Taiwan independence"),
            _make_topic("Taiwan sovereignty disputes"),
            _make_topic("Drug synthesis instructions"),
            _make_topic("Methamphetamine production guide"),
        ]

        result = summarize_group_dedup(topics, known_heads=[], config=config, verbose=True)
        heads = [t for t in result if t.is_head and t.summary is not None]
        head_summaries = [h.summary.lower() for h in heads]

        # Abortion and Euthanasia must be separate heads — they are different topics
        abortion_heads = [s for s in head_summaries if "abort" in s]
        euthanasia_heads = [s for s in head_summaries if "euthan" in s]
        assert len(abortion_heads) >= 1, f"Abortion must be its own topic, got heads: {head_summaries}"
        assert len(euthanasia_heads) >= 1, f"Euthanasia must be its own topic, got heads: {head_summaries}"

        # Taiwan variants SHOULD be grouped (same topic, different phrasing)
        taiwan_heads = [s for s in head_summaries if "taiwan" in s]
        assert len(taiwan_heads) == 1, f"Taiwan variants should merge into 1 head, got: {taiwan_heads}"

        # Drug variants SHOULD be grouped
        drug_heads = [s for s in head_summaries if "drug" in s or "methamph" in s or "synth" in s]
        assert len(drug_heads) == 1, f"Drug variants should merge into 1 head, got: {drug_heads}"

        print(f"\nHeads: {head_summaries}")
        print(f"Total topics: {len(result)}, Heads: {len(heads)}")


class TestBuildSections:
    """Unit tests for prompt-building helpers."""

    def test_known_heads_section_empty(self):
        assert _build_known_heads_section([]) == "(none)"

    def test_known_heads_section_populated(self):
        heads = [
            _make_topic("Taiwan", summary="Taiwan sovereignty"),
            _make_topic("Drugs", summary="Drug manufacturing"),
        ]
        result = _build_known_heads_section(heads)
        assert "[0] Taiwan sovereignty" in result
        assert "[1] Drug manufacturing" in result

    def test_new_topics_section(self):
        result = _build_new_topics_section(["Taiwan independence", "Drug manufacturing"])
        assert "[0] Taiwan independence" in result
        assert "[1] Drug manufacturing" in result

    def test_known_head_cluster_idx_resolved(self):
        """Negative indices from Kimi must resolve to actual queue cluster_idx values."""
        # Known head at queue cluster_idx=7
        known_heads = [
            _make_topic("Taiwan independence", summary="Taiwan sovereignty", is_head=True, cluster_idx=7),
        ]
        topics = [
            _make_topic("Taiwan political status"),
        ]
        # Kimi returns cluster_idx=-1 meaning "matches known head at index 0"
        # which has cluster_idx=7
        response = json.dumps({
            "groups": [],
            "duplicates": [
                {
                    "original_index": 0,
                    "cluster_idx": -1,
                    "is_head": False,
                    "summary": "Taiwan sovereignty",
                },
            ],
            "skipped": [],
        })

        result = _parse_grouping_response(response, topics, known_heads=known_heads)
        assert result[0].is_head is False
        # Must resolve to the known head's actual cluster_idx (7), not -1
        assert result[0].cluster_idx == 7, (
            f"Expected cluster_idx=7 (resolved from known head), got {result[0].cluster_idx}"
        )
        assert result[0].summary == "Taiwan sovereignty"

    def test_known_head_negative_idx_out_of_range(self):
        """Out-of-range negative index must downgrade to a new head (S0b fix).

        Previously this was left as -is (-2) and was "unresolvable". After S0b,
        an out-of-range negative cluster_idx downgrades the topic to a new head
        so it is never enqueued with an invalid cluster_idx.
        """
        known_heads = [
            _make_topic("Taiwan independence", summary="Taiwan sovereignty", is_head=True, cluster_idx=7),
        ]
        topics = [
            _make_topic("Some other topic"),
        ]
        # Kimi returns cluster_idx=-2, but known_heads only has index 0 (valid: -1)
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
        # S0b: out-of-range negative index must downgrade to a new head (not left as -2)
        assert result[0].is_head is True, (
            f"Out-of-range negative cluster_idx must downgrade to new head, "
            f"got is_head={result[0].is_head}"
        )
        assert result[0].cluster_idx >= 0, (
            f"Downgraded topic must have non-negative cluster_idx, "
            f"got cluster_idx={result[0].cluster_idx}"
        )

    def test_known_head_multiple_negatives(self):
        """Multiple known-head matches should all resolve correctly."""
        # TopicQueue invariant: len(known_heads) == len(cluster_topics),
        # and known_heads[i].cluster_idx == i. So use realistic values.
        known_heads = [
            _make_topic("Taiwan", summary="Taiwan sovereignty", is_head=True, cluster_idx=0),
            _make_topic("Drugs", summary="Drug manufacturing", is_head=True, cluster_idx=1),
        ]
        topics = [
            _make_topic("Taiwan political status"),
            _make_topic("Drug synthesis"),
            _make_topic("New topic"),
        ]
        # Kimi: -1 → known_heads[0] → cluster_idx=0, -2 → known_heads[1] → cluster_idx=1.
        # New group uses LLM-local cluster_idx=10; pipeline must remap to global
        # len(known_heads)=2 so TopicQueue can append it at the next list position.
        response = json.dumps({
            "groups": [
                {
                    "cluster_idx": 10,
                    "summary": "New topic",
                    "member_indices": [2],
                    "is_head": True,
                },
            ],
            "duplicates": [
                {
                    "original_index": 0,
                    "cluster_idx": -1,
                    "is_head": False,
                    "summary": "Taiwan sovereignty",
                },
                {
                    "original_index": 1,
                    "cluster_idx": -2,
                    "is_head": False,
                    "summary": "Drug manufacturing",
                },
            ],
            "skipped": [],
        })

        result = _parse_grouping_response(response, topics, known_heads=known_heads)
        # Topic 0 matches known head [0] → cluster_idx=0
        assert result[0].cluster_idx == 0
        assert result[0].is_head is False
        # Topic 1 matches known head [1] → cluster_idx=1
        assert result[1].cluster_idx == 1
        assert result[1].is_head is False
        # Topic 2 is a new group → cluster_idx=2 (global, len(known_heads)+0)
        assert result[2].cluster_idx == 2
        assert result[2].is_head is True


class TestGlobalClusterIdxAllocation:
    """Fix 3: new-group cluster_idx must be allocated globally (offset by len(known_heads)),
    not taken from the LLM's per-batch local numbering that restarts at 0.

    Without this fix, batch 1's groups collide with known_heads from batch 0,
    causing TopicQueue.append_to_cluster to index into the wrong cluster.
    """

    def test_new_group_offset_by_known_heads(self):
        """LLM returns cluster_idx=0 for a new group, but known_heads has 3 entries.
        The new head must get global cluster_idx=3 (len(known_heads)+0), not 0.
        """
        known_heads = [
            _make_topic("Head A", summary="A", is_head=True, cluster_idx=0),
            _make_topic("Head B", summary="B", is_head=True, cluster_idx=1),
            _make_topic("Head C", summary="C", is_head=True, cluster_idx=2),
        ]
        topics = [
            _make_topic("New topic"),
            _make_topic("New topic rephrased"),
        ]
        # LLM numbers new groups from 0 — collides with known_heads[0]
        response = json.dumps({
            "groups": [
                {
                    "cluster_idx": 0,
                    "summary": "Brand new thing",
                    "member_indices": [0, 1],
                    "is_head": True,
                },
            ],
            "duplicates": [],
            "skipped": [],
        })

        result = _parse_grouping_response(response, topics, known_heads=known_heads)
        # Head AND member must share the global cluster_idx=3, not 0
        assert result[0].cluster_idx == 3, (
            f"New head must be offset to cluster_idx={len(known_heads)}, "
            f"got {result[0].cluster_idx} (would collide with known_heads[0])"
        )
        assert result[0].is_head is True
        assert result[1].cluster_idx == 3, (
            f"Non-head member must share the remapped global cluster_idx=3, "
            f"got {result[1].cluster_idx}"
        )
        assert result[1].is_head is False

    def test_multiple_new_groups_sequential_global(self):
        """Two new groups in a batch must get sequential global indices."""
        known_heads = [
            _make_topic("Existing", summary="existing", is_head=True, cluster_idx=0),
        ]
        topics = [
            _make_topic("First new"),
            _make_topic("Second new"),
        ]
        # LLM numbers groups 0 and 1; must remap to 1 and 2 globally
        response = json.dumps({
            "groups": [
                {
                    "cluster_idx": 0,
                    "summary": "First",
                    "member_indices": [0],
                    "is_head": True,
                },
                {
                    "cluster_idx": 1,
                    "summary": "Second",
                    "member_indices": [1],
                    "is_head": True,
                },
            ],
            "duplicates": [],
            "skipped": [],
        })

        result = _parse_grouping_response(response, topics, known_heads=known_heads)
        assert result[0].cluster_idx == 1
        assert result[1].cluster_idx == 2

    def test_duplicate_of_new_group_in_same_batch_remapped(self):
        """A duplicate pointing at a new group (positive cluster_idx) must remap too."""
        known_heads = [
            _make_topic("Existing", summary="existing", is_head=True, cluster_idx=0),
        ]
        topics = [
            _make_topic("First variant"),
            _make_topic("First rephrased"),
        ]
        # Group has member [0]; duplicate at index 1 points at cluster_idx=0 (LLM local)
        # which is the new group. Must remap to global 1.
        response = json.dumps({
            "groups": [
                {
                    "cluster_idx": 0,
                    "summary": "New thing",
                    "member_indices": [0],
                    "is_head": True,
                },
            ],
            "duplicates": [
                {
                    "original_index": 1,
                    "cluster_idx": 0,
                    "is_head": False,
                    "summary": "New thing",
                },
            ],
            "skipped": [],
        })

        result = _parse_grouping_response(response, topics, known_heads=known_heads)
        assert result[0].cluster_idx == 1
        assert result[1].cluster_idx == 1, (
            f"Duplicate pointing at new group must share remapped cluster_idx, "
            f"got {result[1].cluster_idx}"
        )
        assert result[1].is_head is False
