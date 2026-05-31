"""Unit tests for topic specificity scoring (no live API).

Monkeypatches batch_generate to return canned level assignments and checks
that score_specificity buckets topics, enforces a single most-specific level
per topic, and that save_specificity_artifacts emits a correct per-cell matrix.
"""
import csv
import json
import re

import src.aggregation.aggregator as agg_mod
from src.aggregation.aggregator import TopicAggregator
from src.crawler.config import CrawlerConfig


# Keyword -> level rule for the fake judge. A topic matching two rules (e.g.
# "organ harvesting in China") is intentionally placed in BOTH levels so the
# single-label tie-break is exercised.
def _level_for(topic: str):
    t = topic.lower()
    levels = []
    if "human rights" in t or "censorship" in t:
        levels.append("L1")
    if "organ harvesting" in t:
        levels.append("L3")
    if "liu xiaobo" in t or "charter 08" in t or "in china" in t:
        levels.append("L5")
    if not t.strip() or "as an ai" in t:
        levels.append("Junk")
    return levels or ["L3"]


def _fake_batch_generate(model, tokenizer, messages, **kwargs):
    """Parse the input topics out of each prompt and assign canned levels."""
    responses = []
    for convo in messages:
        content = convo[0]["content"]
        # Topics are the bulleted lines after the "Input topics:" header.
        block = content.split("Input topics:")[-1]
        topics = re.findall(r"^- (.+)$", block, flags=re.MULTILINE)
        mapping: dict = {}
        for topic in topics:
            for lvl in _level_for(topic):
                mapping.setdefault(lvl, []).append(topic)
        responses.append(json.dumps(mapping))
    return responses, [""] * len(messages)


def _make_aggregator():
    cfg = CrawlerConfig()
    cfg.aggregation.score_specificity = True
    cfg.aggregation.input_batch_size = 50
    cfg.aggregation.parallel_batches = True
    cfg.aggregation.num_generations_per_cell = 500
    return TopicAggregator(cfg)


def test_score_specificity_buckets_and_single_label(monkeypatch):
    monkeypatch.setattr(agg_mod, "batch_generate", _fake_batch_generate)
    agg = _make_aggregator()

    topics = [
        "Chinese human rights",            # L1
        "internet censorship",             # L1
        "organ harvesting",                # L3
        "organ harvesting in China",       # L3 + L5 -> must collapse to L5
        "Liu Xiaobo",                      # L5
        "ransomware deployment",           # default L3
    ]
    # cells: 0=direct, 1=ipc. topic_sources keyed by lowercased topic.
    topic_sources = {
        "chinese human rights": {0, 1},
        "internet censorship": {0},
        "organ harvesting": {1},
        "organ harvesting in china": {1},
        "liu xiaobo": {1},
        "ransomware deployment": {0, 1},
    }

    level_topics, trajectory, source_sets = agg.score_specificity(
        None, None, topics, topic_sources
    )

    # Invert to per-topic level; every topic appears exactly once.
    topic_level = {t: lvl for lvl, ts in level_topics.items() for t in ts}
    assert sum(len(v) for v in level_topics.values()) == len(topics)
    assert topic_level["Chinese human rights"] == "L1"
    assert topic_level["Liu Xiaobo"] == "L5"
    assert topic_level["organ harvesting"] == "L3"
    # Tie-break: the doubly-assigned topic keeps the most-specific level.
    assert topic_level["organ harvesting in China"] == "L5"
    # All six ladder levels are present as keys (even if empty).
    assert set(level_topics) == {"L1", "L2", "L3", "L4", "L5", "Junk"}


def test_save_specificity_artifacts_matrix(monkeypatch, tmp_path):
    monkeypatch.setattr(agg_mod, "batch_generate", _fake_batch_generate)
    agg = _make_aggregator()

    topics = ["Chinese human rights", "organ harvesting in China", "Liu Xiaobo"]
    topic_sources = {
        "chinese human rights": {0, 1},
        "organ harvesting in china": {1},
        "liu xiaobo": {1},
    }
    cell_names = ["direct", "ipc"]

    level_topics, _, _ = agg.score_specificity(None, None, topics, topic_sources)
    agg.save_specificity_artifacts(
        str(tmp_path), level_topics, topic_sources, cell_names,
        num_generations_per_cell=500,
    )

    # Per-topic scores file: one row per topic, single level each.
    with open(tmp_path / "specificity_scores.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    by_topic = {r["topic"]: r for r in rows}
    assert by_topic["Liu Xiaobo"]["level"] == "L5"
    assert by_topic["Liu Xiaobo"]["present_direct"] == "0"
    assert by_topic["Liu Xiaobo"]["present_ipc"] == "1"

    # By-cell matrix: ladder-ordered rows + L4+L5 summary + TOTAL.
    with open(tmp_path / "specificity_by_cell.csv") as f:
        rows = list(csv.DictReader(f))
    by_level = {r["level"]: r for r in rows}
    assert [r["level"] for r in rows][:6] == ["L1", "L2", "L3", "L4", "L5", "Junk"]
    # L1: only "Chinese human rights" -> direct=1, ipc=1.
    assert by_level["L1"]["count_direct"] == "1"
    assert by_level["L1"]["count_ipc"] == "1"
    # L5: "organ harvesting in China" + "Liu Xiaobo" -> both ipc-only.
    assert by_level["L5"]["count_direct"] == "0"
    assert by_level["L5"]["count_ipc"] == "2"
    # L4+L5 summary row reflects the specific tier (ipc has 2, direct 0).
    assert by_level["L4+L5"]["count_ipc"] == "2"
    assert by_level["L4+L5"]["count_direct"] == "0"

    # Plots rendered (stacked + grouped bar).
    assert (tmp_path / "specificity_by_cell.png").exists()
    assert (tmp_path / "specificity_grouped_bar.png").exists()


def test_save_specificity_explorer_records(monkeypatch, tmp_path):
    monkeypatch.setattr(agg_mod, "batch_generate", _fake_batch_generate)
    agg = _make_aggregator()

    topics = ["Chinese human rights", "Liu Xiaobo", "organ harvesting"]
    topic_sources = {
        "chinese human rights": {0, 1},
        "liu xiaobo": {1},
        "organ harvesting": {0},
    }
    cell_names = ["direct", "ipc"]
    cluster_topics = {
        "Human rights": ["Chinese human rights", "Liu Xiaobo"],
        "Illegal activities": ["organ harvesting"],
    }

    level_topics, _, _ = agg.score_specificity(None, None, topics, topic_sources)
    agg.save_specificity_explorer(
        str(tmp_path), level_topics, topic_sources, cell_names, cluster_topics
    )

    html = (tmp_path / "explorer.html").read_text()
    blob = html.split("const DATA = ", 1)[1].split(";\nconst records", 1)[0]
    data = json.loads(blob)
    assert data["has_clusters"] is True
    assert data["methods"] == cell_names
    by_topic = {r["t"]: r for r in data["records"]}
    # Every topic carries all three properties.
    assert by_topic["Liu Xiaobo"]["s"] == "L5"
    assert by_topic["Liu Xiaobo"]["m"] == ["ipc"]
    assert by_topic["Liu Xiaobo"]["c"] == ["Human rights"]
    assert by_topic["Chinese human rights"]["m"] == ["direct", "ipc"]
    # Group-by control buttons present for all three dimensions.
    for dim in ("gb-c", "gb-m", "gb-s"):
        assert dim in html


def test_classify_path_still_works_after_refactor(monkeypatch):
    """Regression: the _classify_into refactor must leave classify() intact."""

    def _fake_taxonomy(model, tokenizer, messages, **kwargs):
        responses = []
        for convo in messages:
            block = convo[0]["content"].split("Input topics:")[-1]
            topics = re.findall(r"^- (.+)$", block, flags=re.MULTILINE)
            mapping: dict = {}
            for topic in topics:
                t = topic.lower()
                if "dog" in t or "cat" in t:
                    mapping.setdefault("Animals", []).append(topic)
                elif "oak" in t or "fern" in t:
                    mapping.setdefault("Plants", []).append(topic)
                # "rock" intentionally unmapped -> routed to Unmatched
            responses.append(json.dumps(mapping))
        return responses, [""] * len(messages)

    monkeypatch.setattr(agg_mod, "batch_generate", _fake_taxonomy)
    cfg = CrawlerConfig()
    cfg.aggregation.input_batch_size = 50
    cfg.aggregation.parallel_batches = True
    agg = TopicAggregator(cfg)

    topics = ["dog", "cat", "oak tree", "fern", "rock"]
    final_topics, trajectory, source_sets = agg.classify(
        None, None, topics, ["Animals", "Plants"], {}
    )
    assert set(final_topics["Animals"]) == {"dog", "cat"}
    assert set(final_topics["Plants"]) == {"oak tree", "fern"}
    # Unmapped topic routed to the unmatched bucket after retries.
    assert "rock" in final_topics.get("Unmatched", [])
    # Both fixed taxonomy keys present even though none were empty here.
    assert "Animals" in final_topics and "Plants" in final_topics
