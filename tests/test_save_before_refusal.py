"""Fix 1: grouped topics must be persisted to disk BEFORE the refusal check
hangs/crashes, so the queue survives a SIGKILL mid-step.

Also verifies that after the refusal check mutates topics in-place, the
queue's `head_refusal_topics` is refreshed — not stale from add-time when
is_refusal was still None.
"""
import json
import os
import tempfile
from unittest.mock import patch

import pytest

from src.crawler.config import CrawlerConfig
from src.crawler.aggregate_crawler import AggregateCrawler
from src.crawler.topic_queue import Topic, TopicQueue


class TestRefreshRefusalMembership:
    """Pure unit tests for TopicQueue.refresh_refusal_membership."""

    def test_refresh_picks_up_mutated_is_refusal(self):
        """Heads added with is_refusal=None then mutated to True must appear
        in head_refusal_topics after refresh."""
        q = TopicQueue()
        t1 = Topic(raw="A", summary="A", is_head=True, cluster_idx=0, is_refusal=None)
        t2 = Topic(raw="B", summary="B", is_head=True, cluster_idx=1, is_refusal=None)
        q.add_new_cluster_head(t1)
        q.add_new_cluster_head(t2)
        assert q.num_head_refusal_topics == 0
        assert q.head_refusal_topics == []

        # Simulate refusal check mutating topics in place
        t1.is_refusal = True
        t2.is_refusal = False

        q.refresh_refusal_membership()

        assert q.num_head_refusal_topics == 1
        assert q.head_refusal_topics == [t1]

    def test_refresh_drops_stale_entries(self):
        """If a head's is_refusal flips True → False, it must be removed from
        head_refusal_topics on refresh."""
        q = TopicQueue()
        t = Topic(raw="A", summary="A", is_head=True, cluster_idx=0, is_refusal=True)
        q.add_new_cluster_head(t)
        assert q.num_head_refusal_topics == 1

        t.is_refusal = False
        q.refresh_refusal_membership()

        assert q.num_head_refusal_topics == 0
        assert q.head_refusal_topics == []

    def test_refresh_idempotent(self):
        """Calling refresh twice must not duplicate entries."""
        q = TopicQueue()
        t = Topic(raw="A", summary="A", is_head=True, cluster_idx=0, is_refusal=True)
        q.add_new_cluster_head(t)
        q.refresh_refusal_membership()
        q.refresh_refusal_membership()
        assert q.num_head_refusal_topics == 1
        assert q.head_refusal_topics == [t]


class TestSaveBeforeRefusalCheck:
    """When refusal check raises, the grouped topics must already be persisted."""

    def _make_crawler(self, save_path: str) -> AggregateCrawler:
        config = CrawlerConfig()
        config.crawler.num_crawl_steps = 1
        config.crawler.prompt_languages = ["english"]
        config.crawler.seed_warmup_steps = 0
        config.crawler.generation_batch_size = 1
        config.crawler.do_filter_refusals = True
        # Skip initial-topic seeding — requires a translator we don't have in unit tests
        config.initial_topics = []
        return AggregateCrawler(crawler_config=config, save_filename=save_path)

    def test_queue_persisted_when_refusal_check_raises(self):
        """Simulate the real failure mode: build_messages → generate → group →
        refusal_check RAISES. The on-disk save must contain the grouped head
        even though the refusal check never completed.
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_path = f.name

        try:
            crawler = self._make_crawler(save_path)

            # Stub prompt_builder to return 1 trivial message
            crawler.prompt_builder.build_messages = lambda *a, **kw: (
                [[{"role": "user", "content": "hi"}]], [-1]
            )

            # Stub batch_generate to return a fake generation
            fake_topic = Topic(
                raw="sensitive subject X",
                summary="sensitive subject X",
                shortened="sensitive subject X",
                english="sensitive subject X",
                is_head=True,
                is_refusal=None,
            )

            with patch(
                "src.crawler.aggregate_crawler.batch_generate",
                return_value=(["Topics:\n1. sensitive subject X"], ["<prompt>"]),
            ), patch.object(
                crawler.formatter,
                "extract_and_translate",
                return_value=[fake_topic],
            ), patch(
                "src.crawler.aggregate_crawler.summarize_group_dedup",
                side_effect=lambda topics, **kw: [
                    Topic(
                        raw=t.raw,
                        summary=t.raw,
                        shortened=t.raw,
                        english=t.raw,
                        is_head=True,
                        cluster_idx=i,
                        cluster_member_count=1,
                        is_refusal=None,
                    )
                    for i, t in enumerate(topics)
                ],
            ), patch(
                "src.crawler.aggregate_crawler.check_refusal_progressive",
                side_effect=RuntimeError("simulated refusal-check crash / SIGKILL"),
            ):
                with pytest.raises(RuntimeError, match="simulated"):
                    crawler.crawl(local_model=None, local_tokenizer=None, verbose=False)

            # The queue must have been saved to disk with the grouped head
            # BEFORE the refusal check was attempted.
            with open(save_path) as f:
                saved = json.load(f)
            head_topics = saved["queue"]["topics"]["head_topics"]
            raws = [h["raw"] for h in head_topics]
            assert "sensitive subject X" in raws, (
                f"Grouped topic must be persisted before refusal check; "
                f"found head_topics={raws}"
            )
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)

    def test_head_refusal_topics_refreshed_after_check(self):
        """When refusal check succeeds and flips is_refusal=True on a head that
        was added with is_refusal=None, head_refusal_topics must reflect it.
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_path = f.name

        try:
            crawler = self._make_crawler(save_path)

            crawler.prompt_builder.build_messages = lambda *a, **kw: (
                [[{"role": "user", "content": "hi"}]], [-1]
            )

            def _mark_refusal(config, local_model, local_tokenizer, selected_topics, verbose=False):
                for t in selected_topics:
                    t.is_refusal = True
                return selected_topics

            with patch(
                "src.crawler.aggregate_crawler.batch_generate",
                return_value=(["Topics:\n1. topic X"], ["<prompt>"]),
            ), patch.object(
                crawler.formatter,
                "extract_and_translate",
                return_value=[Topic(
                    raw="topic X",
                    summary="topic X",
                    shortened="topic X",
                    english="topic X",
                    is_head=True,
                    is_refusal=None,
                )],
            ), patch(
                "src.crawler.aggregate_crawler.summarize_group_dedup",
                side_effect=lambda topics, **kw: [
                    Topic(
                        raw=t.raw,
                        summary=t.raw,
                        shortened=t.raw,
                        english=t.raw,
                        is_head=True,
                        cluster_idx=i,
                        cluster_member_count=1,
                        is_refusal=None,
                    )
                    for i, t in enumerate(topics)
                ],
            ), patch(
                "src.crawler.aggregate_crawler.check_refusal_progressive",
                side_effect=_mark_refusal,
            ):
                crawler.crawl(local_model=None, local_tokenizer=None, verbose=False)

            with open(save_path) as f:
                saved = json.load(f)
            head_refusal = saved["queue"]["topics"]["head_refusal_topics"]
            assert len(head_refusal) == 1, (
                f"After refusal check flips is_refusal=True, the head must "
                f"appear in head_refusal_topics. Got {len(head_refusal)}."
            )
            assert head_refusal[0]["raw"] == "topic X"
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)
