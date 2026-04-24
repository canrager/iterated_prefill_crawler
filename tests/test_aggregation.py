import json

from src.aggregation.aggregator import TopicAggregator
from src.aggregation.coverage import load_crawl_topics
from src.aggregation.topic_normalization import normalize_topic_key
from src.crawler.config import CrawlerConfig


def _write_crawler_output(path, summaries):
    path.write_text(
        json.dumps({"head_refusal_topics_summaries": summaries}),
        encoding="utf-8",
    )


def test_load_topics_preaggregates_punctuation_case_variants_without_llm(
    tmp_path, monkeypatch
):
    run0 = tmp_path / "run0.json"
    run1 = tmp_path / "run1.json"
    _write_crawler_output(run0, ["Cyber Abuse!", "Taiwan status"])
    _write_crawler_output(run1, ["cyber abuse", "Hong Kong status"])

    def fail_batch_generate(*args, **kwargs):
        raise AssertionError("load_topics must not call the LLM generation path")

    monkeypatch.setattr("src.aggregation.aggregator.batch_generate", fail_batch_generate)

    topics, sources = TopicAggregator(CrawlerConfig()).load_topics(
        [str(run0), str(run1)]
    )

    assert topics == ["Cyber Abuse!", "Taiwan status", "Hong Kong status"]
    assert sources["cyber abuse"] == {0, 1}
    assert sources["taiwan status"] == {0}
    assert sources["hong kong status"] == {1}


def test_load_topics_keeps_distinct_leaves_after_normalization(tmp_path):
    run = tmp_path / "run.json"
    _write_crawler_output(
        run,
        [
            "Taiwan status.",
            "Hong Kong status!",
            "taiwan status",
            "Hong Kong status",
        ],
    )

    topics, sources = TopicAggregator(CrawlerConfig()).load_topics([str(run)])

    assert topics == ["Taiwan status.", "Hong Kong status!"]
    assert set(sources) == {"taiwan status", "hong kong status"}


def test_coverage_and_aggregator_loaders_share_topic_normalization(tmp_path):
    run = tmp_path / "run.json"
    _write_crawler_output(
        run,
        [
            "Taiwan status.",
            "taiwan status",
            "Hong Kong status!",
            "Hong Kong status",
            "台湾地位。",
            "台湾地位",
        ],
    )

    aggregator_topics, aggregator_sources = TopicAggregator(CrawlerConfig()).load_topics(
        [str(run)]
    )
    coverage_topics = load_crawl_topics([str(run)])

    assert aggregator_topics == ["Taiwan status.", "Hong Kong status!", "台湾地位。"]
    assert coverage_topics == aggregator_topics
    assert [normalize_topic_key(t) for t in coverage_topics] == list(
        aggregator_sources
    )
