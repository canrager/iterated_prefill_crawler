import pytest
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from core.crawler import Crawler
from core.crawler_config import CrawlerConfig
from core.topic_queue import Topic
from core.llm_utils import load_zh_en_translation_model
from core.project_config import CACHE_DIR, DEVICE


@pytest.fixture
def crawler():
    """Create a basic crawler instance for testing"""
    config = CrawlerConfig(
        num_samples_per_topic=1,
        num_crawl_steps=5,
        generation_batch_size=2,  # Small batch size for translation testing
    )
    return Crawler(crawler_config=config, save_filename="test_crawler.json")


def test_translate_chinese_english(crawler):
    """Test the Chinese to English translation method"""

    # Create test topics with mix of Chinese and English
    test_topics = [
        Topic(text="Hello", raw="Hello"),  # English
        Topic(text="你好", raw="你好"),  # Chinese
        Topic(text="Test", raw="Test"),  # English
        Topic(text="世界", raw="世界"),  # Chinese
    ]

    model_zh_en, tokenizer_zh_en = load_zh_en_translation_model(CACHE_DIR, DEVICE)

    # Run translation
    translated_topics = crawler._translate_chinese_english(
        model_zh_en=model_zh_en, tokenizer_zh_en=tokenizer_zh_en, topics=test_topics
    )

    # Verify results
    assert len(translated_topics) == 4

    # Check English topics remain unchanged
    assert not translated_topics[0].is_chinese
    assert translated_topics[0].translation is None
    assert not translated_topics[2].is_chinese
    assert translated_topics[2].translation is None

    # Check Chinese topics are translated
    assert translated_topics[1].is_chinese
    assert translated_topics[1].translation is not None  # Changed since we're using real model
    assert translated_topics[3].is_chinese
    assert translated_topics[3].translation is not None

    # Verify order is preserved (non-Chinese first, then Chinese)
    assert not translated_topics[0].is_chinese
    assert not translated_topics[2].is_chinese
    assert translated_topics[1].is_chinese
    assert translated_topics[3].is_chinese
