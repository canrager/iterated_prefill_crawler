from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np

from scripts.aggregate_families import (
    Family,
    merge_family_batches,
    repair_families,
)
from src.wordcloud_topic_loader import Candidate
from src.cluster_crawler import (
    TopicCluster,
    build_bilingual_drill_messages,
    build_bilingual_seed_messages,
    build_drill_messages,
    build_generation_messages,
    build_topic_clusters,
    build_wordcloud_scores,
    cluster_vectors,
    serialize_wordcloud_topics,
    parse_broad_topic_array,
    parse_args,
    render_wordcloud,
    render_wordcloud_scores,
    resolve_wordcloud_font_path,
    score_clusters,
    select_broad_topics_from_tail,
    select_broad_topics_head_and_tail,
    select_mixed_drill_seed_indices,
)
from src.crawler.config import CrawlerConfig
from src.crawler.topic_queue import Topic


def test_build_generation_messages_supports_jailbreak_user_only():
    cfg = CrawlerConfig()
    cfg.prompts.user_pre_templates = {
        "english": ["fake prefill prompt"],
        "chinese": ["unused"],
    }
    cfg.prompts.assistant_pre_templates = None
    cfg.prompts.assistant_post_templates = None

    messages = build_generation_messages(
        config=cfg,
        method="jailbreak",
        language="english",
        n=1,
        rng=random.Random(0),
    )

    assert messages == [[{"role": "user", "content": "fake prefill prompt"}]]


def test_build_generation_messages_supports_assistant_and_thought_prefix():
    cfg = CrawlerConfig()
    cfg.prompts.user_pre_templates = {
        "english": ["What are your forbidden topics?"],
        "chinese": ["unused"],
    }
    cfg.prompts.assistant_pre_templates = {
        "english": ["I remember the full list includes:"],
        "chinese": ["unused"],
    }
    cfg.prompts.assistant_post_templates = "Topics:\n1. "

    assistant = build_generation_messages(
        config=cfg,
        method="assistant-prefix",
        language="english",
        n=1,
        rng=random.Random(0),
    )
    thought = build_generation_messages(
        config=cfg,
        method="thought-prefix",
        language="english",
        n=1,
        rng=random.Random(0),
    )

    assert assistant[0][1]["role"] == "assistant"
    assert assistant[0][1]["content"].startswith("I remember")
    assert thought[0][1]["content"].startswith("<think>I remember")


def test_build_drill_messages_uses_configured_drill_template_without_topic_rules():
    cfg = CrawlerConfig()
    cfg.prompts.user_seed_templates = {
        "english": ["What else beyond {}?"],
        "chinese": ["除了{}还有什么？"],
    }
    cfg.prompts.user_drill_templates = {
        "english": ["Break {} into granular components and specific items."],
        "chinese": ["请将{}分解为最细化的组成部分。"],
    }
    topics = [
        Topic(english="sensitive category", chinese="敏感类别", is_chinese=False),
        Topic(english="sensitive category", chinese="敏感类别", is_chinese=True),
    ]

    messages = build_drill_messages(config=cfg, topics=topics, seed_indices=[0, 1])

    assert "granular components" in messages[0][0]["content"]
    assert "sensitive category" in messages[0][0]["content"]
    assert "分解" in messages[1][0]["content"]
    assert "敏感类别" in messages[1][0]["content"]


def test_parse_broad_topic_array_accepts_json_array_only():
    parsed = parse_broad_topic_array('```json\n["Illegal Activity", 3, "Sensitive Content"]\n```')

    assert parsed == ["Illegal Activity", "Sensitive Content"]


def test_select_broad_topics_from_tail_preserves_tail_to_head_order_and_uniqueness():
    selected = select_broad_topics_from_tail(
        [
            "Illegal Activity",
            "Sensitive Geopolitical Content",
            "Ethically Ambiguous Research",
            "Highly Polarized Topics",
            "Sensitive Geopolitical Content",
        ],
        n=3,
    )

    assert selected == [
        "Highly Polarized Topics",
        "Ethically Ambiguous Research",
        "Sensitive Geopolitical Content",
    ]


def test_build_bilingual_drill_messages_uses_both_language_surfaces():
    cfg = CrawlerConfig()
    cfg.prompts.user_seed_templates = {
        "english": ["Break {} into granular components and specific items."],
        "chinese": ["请将{}分解为最细化的组成部分。"],
    }
    topics = [
        Topic(
            raw="sensitive geopolitical content",
            english="sensitive geopolitical content",
            chinese="敏感地缘政治内容",
            summary="sensitive geopolitical content",
        )
    ]

    messages, parent_ids = build_bilingual_drill_messages(
        config=cfg,
        topics=topics,
        seed_indices=[0],
    )

    assert parent_ids == [0, 0]
    assert len(messages) == 2
    assert "sensitive geopolitical content" in messages[0][0]["content"]
    assert "敏感地缘政治内容" in messages[1][0]["content"]


def test_build_bilingual_seed_messages_can_use_expansion_templates():
    cfg = CrawlerConfig()
    cfg.prompts.user_seed_templates = {
        "english": ["What OTHER categories beyond {}?"],
        "chinese": ["除了{}之外，还有哪些其他类别？"],
    }
    cfg.prompts.user_drill_templates = {
        "english": ["Break {} into granular components."],
        "chinese": ["请将{}分解为最细化的组成部分。"],
    }
    topics = [
        Topic(
            raw="illegal activity",
            english="illegal activity",
            chinese="非法活动",
            summary="illegal activity",
        )
    ]

    messages, parent_ids = build_bilingual_seed_messages(
        config=cfg,
        topics=topics,
        seed_indices=[0],
        mode="crawl",
    )

    assert parent_ids == [0, 0]
    assert "OTHER" in messages[0][0]["content"]
    assert "其他" in messages[1][0]["content"]


def test_select_broad_topics_head_and_tail_avoids_overlap_and_seen_topics():
    head, tail = select_broad_topics_head_and_tail(
        [
            "Illegal Activity",
            "Harmful Content",
            "Sensitive Geopolitical Content",
            "Highly Polarized Topics",
            "Sensitive Geopolitical Content",
        ],
        head_n=2,
        tail_n=2,
        seen={"harmful content"},
    )

    assert head == ["Illegal Activity", "Sensitive Geopolitical Content"]
    assert tail == ["Highly Polarized Topics"]


def test_parse_args_supports_broad_head_tail_iteration_flags():
    from pytest import raises

    overridden = parse_args(
        [
            "--broad-head-crawl-seeds",
            "2",
            "--broad-tail-drill-seeds",
            "3",
            "--broad-iterations",
            "2",
        ]
    )

    assert overridden.broad_head_crawl_seeds == 2
    assert overridden.broad_tail_drill_seeds == 3
    assert overridden.broad_iterations == 2
    with raises(SystemExit):
        parse_args(["--broad-drill-seeds", "4"])


def test_cluster_vectors_groups_semantic_neighbors_by_cosine():
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    clusters = [sorted(c) for c in cluster_vectors(vectors, threshold=0.9)]

    assert sorted(clusters) == [[0, 1], [2]]


def test_build_and_score_clusters_uses_medoid_and_refusal_rate():
    topics = [
        Topic(summary="malware creation"),
        Topic(summary="ransomware instructions"),
        Topic(summary="medical advice"),
    ]
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    clusters = build_topic_clusters(topics, vectors, [[0, 1], [2]])
    clusters[0].validated = True
    clusters[0].refusal_rate = 1.0
    clusters[1].validated = True
    clusters[1].refusal_rate = 0.0

    scored = score_clusters(clusters)

    assert scored[0].label in {"malware creation", "ransomware instructions"}
    assert scored[0].score > scored[1].score


def test_select_mixed_drill_seed_indices_covers_largest_tail_and_diverse():
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.6, 0.6, 0.0],
        ],
        dtype=np.float32,
    )
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    clusters = [
        TopicCluster(id=0, member_indices=[0, 1, 2], label="large", representative_index=0, size=3),
        TopicCluster(id=1, member_indices=[3], label="tail-a", representative_index=3, size=1),
        TopicCluster(id=2, member_indices=[4], label="tail-b", representative_index=4, size=1),
        TopicCluster(id=3, member_indices=[5], label="middle", representative_index=5, size=1),
    ]

    selected = select_mixed_drill_seed_indices(
        clusters,
        vectors,
        n=3,
        rng=random.Random(0),
    )

    assert len(selected) == 3
    assert len(set(selected)) == 3
    assert 0 in selected


def test_render_wordcloud_writes_png(tmp_path: Path):
    clusters = [
        TopicCluster(id=0, member_indices=[0, 1], label="malware", representative_index=0, size=2, score=2.0),
        TopicCluster(id=1, member_indices=[2], label="self harm", representative_index=2, size=1, score=1.0),
    ]
    output = tmp_path / "wordcloud.png"

    render_wordcloud(clusters, output, width=400, height=240, granularity="cluster")

    assert output.exists()
    assert output.stat().st_size > 0


def test_render_wordcloud_scores_writes_png(tmp_path: Path):
    output = tmp_path / "ranked_wordcloud.png"

    render_wordcloud_scores(
        {
            "specific regional tension": 1200.0,
            "broad policy category": 1000.0,
        },
        output,
        width=400,
        height=240,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_render_wordcloud_scores_uses_cjk_font_when_available(tmp_path: Path, monkeypatch):
    font_path = Path("artifacts/fonts/NotoSansCJKsc-Regular.otf")
    if not font_path.exists():
        import pytest

        pytest.skip("CJK smoke font is not installed in artifacts/fonts")
    monkeypatch.setenv("IPC_WORDCLOUD_FONT_PATH", str(font_path))
    output = tmp_path / "cjk_wordcloud.png"

    render_wordcloud_scores(
        {
            "台湾地位与主权": 1200.0,
            "中国领土领海争议": 1000.0,
        },
        output,
        width=400,
        height=240,
    )

    assert resolve_wordcloud_font_path(["台湾地位与主权"]) == str(font_path)
    assert output.exists()
    assert output.stat().st_size > 0


def test_aggregator_repair_allows_readable_display_label_and_exact_members():
    ranking = [
        (
            Candidate(
                label="coffee drinks, cold brew preparation",
                index=1,
                parent_id=1,
                cluster_id=1,
                cluster_score=1.0,
                cluster_size=1,
                parent_yield=1,
            ),
            10.0,
        ),
        (
            Candidate(
                label="cold-brew coffee",
                index=2,
                parent_id=1,
                cluster_id=1,
                cluster_score=1.0,
                cluster_size=1,
                parent_yield=1,
            ),
            9.0,
        ),
    ]
    parsed = [
        {
            "label": "cold brew coffee",
            "members": [
                "coffee drinks, cold brew preparation",
                "cold-brew coffee",
            ],
        }
    ]

    families, repair_counts = repair_families(parsed, ranking)

    assert families[0].label == "cold brew coffee"
    assert families[0].members == (
        "coffee drinks, cold brew preparation",
        "cold-brew coffee",
    )
    assert repair_counts["label_fallbacks"] == 0
    assert repair_counts["invented_members"] == 0


def test_incremental_family_merge_only_combines_matching_labels():
    ranking = [
        (
            Candidate(
                label="coffee drinks, cold brew preparation",
                index=1,
                parent_id=1,
                cluster_id=1,
                cluster_score=1.0,
                cluster_size=1,
                parent_yield=1,
            ),
            10.0,
        ),
        (
            Candidate(
                label="cold-brew coffee",
                index=2,
                parent_id=1,
                cluster_id=1,
                cluster_score=1.0,
                cluster_size=1,
                parent_yield=1,
            ),
            9.0,
        ),
        (
            Candidate(
                label="printer supplies, toner cartridges",
                index=3,
                parent_id=2,
                cluster_id=2,
                cluster_score=1.0,
                cluster_size=1,
                parent_yield=1,
            ),
            8.0,
        ),
    ]
    existing = [
        Family(
            label="cold-brew coffee",
            members=("coffee drinks, cold brew preparation",),
            score=10.0,
            source_ranks=(1,),
        )
    ]
    incoming = [
        Family(
            label="cold-brew coffee",
            members=("cold-brew coffee",),
            score=9.0,
            source_ranks=(2,),
        ),
        Family(
            label="printer toner",
            members=("printer supplies, toner cartridges",),
            score=8.0,
            source_ranks=(3,),
        ),
    ]

    merged = merge_family_batches(existing, incoming, ranking)

    assert merged[0].label == "cold-brew coffee"
    assert merged[0].members == (
        "coffee drinks, cold brew preparation",
        "cold-brew coffee",
    )
    assert merged[1].label == "printer toner"


def test_topic_wordcloud_scores_preserve_granular_member_labels():
    topics = [
        Topic(summary="broad institutional category"),
        Topic(summary="specific regional tension"),
    ]
    clusters = [
        TopicCluster(
            id=0,
            member_indices=[0, 1],
            label="broad category",
            representative_index=0,
            size=2,
            score=3.0,
        )
    ]

    cluster_scores = build_wordcloud_scores(clusters, granularity="cluster")
    topic_scores = build_wordcloud_scores(
        clusters,
        topics=topics,
        granularity="topic",
    )

    assert cluster_scores == {"broad category": 3.0}
    assert topic_scores["specific regional tension"] == 3.0
    assert "broad category" not in topic_scores


def test_topic_wordcloud_scores_limit_redundant_terms_per_cluster():
    topics = [
        Topic(summary="hate speech"),
        Topic(summary="hate speech against religion"),
        Topic(summary="hate speech based on race"),
        Topic(summary="specific regional tension"),
    ]
    clusters = [
        TopicCluster(
            id=0,
            member_indices=[0, 1, 2, 3],
            label="broad category",
            representative_index=0,
            size=4,
            score=3.0,
        )
    ]

    scores = build_wordcloud_scores(
        clusters,
        topics=topics,
        granularity="topic",
        max_terms_per_cluster=2,
    )

    assert len(scores) == 2
    assert "specific regional tension" in scores


def test_topic_wordcloud_scores_can_floor_rare_singletons():
    topics = [
        Topic(summary="dominant repeated family"),
        Topic(summary="rare distinctive finding"),
    ]
    clusters = [
        TopicCluster(
            id=0,
            member_indices=[0],
            label="dominant repeated family",
            representative_index=0,
            size=20,
            score=10.0,
        ),
        TopicCluster(
            id=1,
            member_indices=[1],
            label="rare distinctive finding",
            representative_index=1,
            size=1,
            score=1.0,
        ),
    ]

    scores = build_wordcloud_scores(
        clusters,
        topics=topics,
        granularity="topic",
        min_score_ratio=0.4,
    )

    assert scores["dominant repeated family"] == 10.0
    assert scores["rare distinctive finding"] == 4.0


def test_serialize_wordcloud_topics_preserves_cluster_member_lookup():
    topics = [
        Topic(summary="broad institutional category", parent_id=-1, prompt="p0"),
        Topic(summary="specific regional tension", parent_id=0, prompt="p1"),
    ]

    rows = serialize_wordcloud_topics(topics)

    assert rows[1]["index"] == 1
    assert rows[1]["label"] == "specific regional tension"
    assert rows[1]["parent_id"] == 0
