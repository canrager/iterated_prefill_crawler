"""Integration test: topic extractor returns [] for non-topic-list DeepSeek outputs.

These fixtures are the exact outputs from record 1, samples 0 and 2 of:
  artifacts/out/debug_ds_clean_20260409_231352/
    crawler_out_20260409_231357_deepseek-v3.2_5samples_2crawls_Truefilter_ds-v32_clean.jsonl

Both caused contamination in the neutral debug run (2026-04-09) because DeepSeek
drifted from its "list your restricted topics" task and generated hallucinated training
data instead. The extractor previously extracted textbook metadata and CS algorithm
names as if they were sensitive topic labels.

Run with:
    pytest tests/test_topic_extraction_drift.py -v
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env so OPENROUTER_API_KEY is available
env_file = Path(__file__).resolve().parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Skip the whole module if OPENROUTER_API_KEY is unavailable. These tests hit
# the live extractor; running them without credentials would fail confusingly
# in local or CI environments that don't carry the key.
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set; skipping live extractor integration tests",
)

from src.response_formatting_utils import TopicFormatter

# Exact outputs from the contamination incident (2026-04-09 debug run)
TEXTBOOK_COPYRIGHT_PAGE = """\
# 2022-2023学年高一数学 人教A版2019必修第一册 同步讲义 第16讲 第三章 函数的概念与性质 章节验收测评卷（解析版）.docx

## 2022-2023学年高一数学 人教A版2019必修第一册 同步讲义 第16讲 第三章 函数的概念与性质 章节验收测评卷（解析版）.docx

- 1、本文档共15页，可阅读全部内容。
- 2、原创力文档（book118）网站文档一经付费（服务费），不意味着购买了该文档的版权，\
仅供个人/单位学习、研究之用，不得用于商业用途，未经授权，严禁复制、发行、汇编、翻译或者网络传播等，侵权必究。
- 3、本站所有内容均由合作方或网友上传，本站不对文档的完整性、权威性及其观点立场正确性做任何保证或承诺！\
文档内容仅供研究参考，付费前请自行鉴别。
"""

LEETCODE_SOLUTION = """\
# 题目
给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。

示例 1:
输入: 123
输出: 321

示例 2:
输入: -123
输出: -321

示例 3:
输入: 120
输出: 21
注意:

假设我们的环境只能存储得下 32 位的有符号整数，则其数值范围为 [−231,  231 − 1]。\
请根据这个假设，如果反转后整数溢出那么就返回 0。

# 思路
1. 判断x是否在[-9,9]之间，是则返回x
2. 判断x的正负，将x的绝对值转换为字符串，反转，再转为int，判断是否在[-2^31, 2^31-1]之间
"""


def _make_config():
    cfg = MagicMock()
    cfg.model.summarization_model = "qwen/qwen3-235b-a22b-2507"
    cfg.model.default_provider = "openrouter"
    cfg.model.provider_urls = None
    cfg.crawler.max_extracted_topics_per_generation = 10
    cfg.crawler.max_concurrent_summarizations = 2
    cfg.crawler.extraction_batch_size = 1
    # Use the real prompt from config (imports the updated constant)
    from src.crawler.config import TOPIC_EXTRACTION_PROMPT

    cfg.topic_extraction_prompt = TOPIC_EXTRACTION_PROMPT
    return cfg


@pytest.mark.integration
def test_extractor_returns_empty_for_textbook_copyright_page():
    """Textbook copyright/metadata page must not produce any topic labels."""
    formatter = TopicFormatter(_make_config())
    results = formatter._extract_with_model([TEXTBOOK_COPYRIGHT_PAGE])
    assert results[0] == [], f"Expected [] for textbook input, got: {results[0]}"


@pytest.mark.integration
def test_extractor_returns_empty_for_leetcode_solution():
    """LeetCode-style coding problem must not produce any topic labels."""
    formatter = TopicFormatter(_make_config())
    results = formatter._extract_with_model([LEETCODE_SOLUTION])
    assert results[0] == [], f"Expected [] for coding problem input, got: {results[0]}"


@pytest.mark.integration
def test_extractor_still_works_for_valid_topic_list():
    """Sanity check: a real restricted-topic list must still yield labels."""
    valid_output = """\
我会避免或谨慎处理的话题包括：
1. 台湾政治地位与独立
2. 西藏独立运动
3. 天安门事件历史评价
4. 新疆人权问题
5. 香港政治地位
"""
    formatter = TopicFormatter(_make_config())
    results = formatter._extract_with_model([valid_output])
    assert len(results[0]) >= 3, (
        f"Expected ≥3 topics for valid input, got: {results[0]}"
    )
