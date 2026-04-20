"""Tests for S4e: initialize_topics bulk translation in AggregateCrawler.

All tests are pure (mocks only). No live API calls.

S4e: replace per-seed _translate_zn_to_en / _translate_en_to_zn calls with
single bulk calls (one per direction), so N string seeds produce at most 2 API
calls instead of N.
"""
import pytest
from unittest.mock import MagicMock, patch, call

from src.crawler.config import CrawlerConfig, ModelConfig, CrawlerRunConfig


def _make_config():
    model = ModelConfig(
        translation_model="some/translation-model",
        summarization_model="some/summ-model",
        default_provider="openrouter",
    )
    crawler = CrawlerRunConfig()
    return CrawlerConfig(model=model, crawler=crawler)


def _make_aggregate_crawler(tmp_path):
    """Create an AggregateCrawler wired to a temp save file."""
    from src.crawler.aggregate_crawler import AggregateCrawler
    config = _make_config()
    save_file = str(tmp_path / "test_crawl.json")
    with patch.object(AggregateCrawler, "save", return_value=None):
        crawler = AggregateCrawler.__new__(AggregateCrawler)
        crawler.config = config
        from src.crawler.topic_queue import TopicQueue
        from src.crawler.crawler_stats import CrawlerStats
        from src.response_formatting_utils import TopicFormatter
        from src.prompt_builder import PromptBuilder
        crawler.queue = TopicQueue()
        crawler.stats = CrawlerStats()
        crawler.formatter = TopicFormatter(config)
        crawler.save_filename = save_file
    return crawler


class TestInitializeTopicsBulkTranslation:
    def test_bulk_translates_chinese_seeds(self, tmp_path):
        """One bulk zn_to_en call for all Chinese string seeds, not N calls."""
        crawler = _make_aggregate_crawler(tmp_path)
        seeds = ["天安门事件", "西藏独立", "台湾独立"]

        call_count = {"zn_to_en": 0}

        def mock_zn_to_en(local_model, local_tokenizer, inputs):
            call_count["zn_to_en"] += 1
            # Return English translation for each input
            if isinstance(inputs, list):
                return [f"EN:{s}" for s in inputs]
            return f"EN:{inputs}"

        with patch.object(crawler.formatter, "_translate_zn_to_en", side_effect=mock_zn_to_en):
            with patch.object(crawler.formatter, "_translate_en_to_zn", return_value=[]):
                with patch.object(crawler, "save", return_value=None):
                    with patch.object(crawler.queue, "incoming_batch", side_effect=lambda t: t):
                        topics = crawler.initialize_topics(
                            local_model=None,
                            local_tokenizer=None,
                            initial_topics=seeds,
                        )

        # Must be exactly ONE bulk call, not 3 individual calls
        assert call_count["zn_to_en"] == 1

    def test_bulk_translates_english_seeds(self, tmp_path):
        """One bulk en_to_zn call for all English string seeds, not N calls."""
        crawler = _make_aggregate_crawler(tmp_path)
        seeds = ["Taiwan independence", "Tiananmen Square", "Falun Gong"]

        call_count = {"en_to_zn": 0}

        def mock_en_to_zn(local_model, local_tokenizer, inputs):
            call_count["en_to_zn"] += 1
            if isinstance(inputs, list):
                return [f"ZH:{s}" for s in inputs]
            return f"ZH:{inputs}"

        with patch.object(crawler.formatter, "_translate_zn_to_en", return_value=[]):
            with patch.object(crawler.formatter, "_translate_en_to_zn", side_effect=mock_en_to_zn):
                with patch.object(crawler, "save", return_value=None):
                    with patch.object(crawler.queue, "incoming_batch", side_effect=lambda t: t):
                        topics = crawler.initialize_topics(
                            local_model=None,
                            local_tokenizer=None,
                            initial_topics=seeds,
                        )

        # Must be exactly ONE bulk call, not 3 individual calls
        assert call_count["en_to_zn"] == 1

    def test_dict_path_skips_translation(self, tmp_path):
        """Pre-translated dict entries must not trigger any translation call."""
        crawler = _make_aggregate_crawler(tmp_path)
        seeds = [
            {"english": "Tiananmen Square protests", "chinese": "天安门广场抗议"},
            {"english": "Taiwan independence", "chinese": "台湾独立"},
        ]

        zn_to_en_calls = []
        en_to_zn_calls = []

        with patch.object(crawler.formatter, "_translate_zn_to_en", side_effect=lambda *a, **kw: zn_to_en_calls.append(a) or []):
            with patch.object(crawler.formatter, "_translate_en_to_zn", side_effect=lambda *a, **kw: en_to_zn_calls.append(a) or []):
                with patch.object(crawler, "save", return_value=None):
                    with patch.object(crawler.queue, "incoming_batch", side_effect=lambda t: t):
                        topics = crawler.initialize_topics(
                            local_model=None,
                            local_tokenizer=None,
                            initial_topics=seeds,
                        )

        assert zn_to_en_calls == [], "No zn_to_en call expected for dict entries"
        assert en_to_zn_calls == [], "No en_to_zn call expected for dict entries"
        assert len(topics) == 2
        assert topics[0].english == "Tiananmen Square protests"
        assert topics[0].chinese == "天安门广场抗议"

    def test_mixed_list_only_strings_translated(self, tmp_path):
        """Mixed list with both strings and dicts: only strings are translated."""
        crawler = _make_aggregate_crawler(tmp_path)
        seeds = [
            {"english": "pre-translated topic", "chinese": "预翻译话题"},  # dict: no call
            "Tibet independence",   # English string: en_to_zn bulk call
            "台湾独立",              # Chinese string: zn_to_en bulk call
        ]

        zn_to_en_calls = []
        en_to_zn_calls = []

        def mock_zn_to_en(local_model, local_tokenizer, inputs):
            zn_to_en_calls.append(inputs)
            return [f"EN:{s}" for s in inputs]

        def mock_en_to_zn(local_model, local_tokenizer, inputs):
            en_to_zn_calls.append(inputs)
            return [f"ZH:{s}" for s in inputs]

        with patch.object(crawler.formatter, "_translate_zn_to_en", side_effect=mock_zn_to_en):
            with patch.object(crawler.formatter, "_translate_en_to_zn", side_effect=mock_en_to_zn):
                with patch.object(crawler, "save", return_value=None):
                    with patch.object(crawler.queue, "incoming_batch", side_effect=lambda t: t):
                        topics = crawler.initialize_topics(
                            local_model=None,
                            local_tokenizer=None,
                            initial_topics=seeds,
                        )

        # Exactly one bulk call per direction, each with only the string seeds
        assert len(zn_to_en_calls) == 1
        assert zn_to_en_calls[0] == ["台湾独立"]

        assert len(en_to_zn_calls) == 1
        assert en_to_zn_calls[0] == ["Tibet independence"]

        # Dict entry preserved exactly
        assert topics[0].english == "pre-translated topic"
        assert topics[0].chinese == "预翻译话题"
        # cluster_idx preserved per original order
        assert topics[0].cluster_idx == 0
        assert topics[1].cluster_idx == 1
        assert topics[2].cluster_idx == 2

    def test_translated_topics_correct_fields(self, tmp_path):
        """Verify english/chinese fields are correctly set after bulk translation."""
        crawler = _make_aggregate_crawler(tmp_path)
        seeds = ["Tibet autonomy", "自由西藏"]

        def mock_zn_to_en(local_model, local_tokenizer, inputs):
            return ["Free Tibet"]

        def mock_en_to_zn(local_model, local_tokenizer, inputs):
            return ["西藏自治"]

        with patch.object(crawler.formatter, "_translate_zn_to_en", side_effect=mock_zn_to_en):
            with patch.object(crawler.formatter, "_translate_en_to_zn", side_effect=mock_en_to_zn):
                with patch.object(crawler, "save", return_value=None):
                    with patch.object(crawler.queue, "incoming_batch", side_effect=lambda t: t):
                        topics = crawler.initialize_topics(
                            local_model=None,
                            local_tokenizer=None,
                            initial_topics=seeds,
                        )

        # English seed → english=original, chinese=translated
        assert topics[0].english == "Tibet autonomy"
        assert topics[0].chinese == "西藏自治"
        assert topics[0].raw == "Tibet autonomy"

        # Chinese seed → english=translated, chinese=original
        assert topics[1].english == "Free Tibet"
        assert topics[1].chinese == "自由西藏"
        assert topics[1].raw == "自由西藏"
