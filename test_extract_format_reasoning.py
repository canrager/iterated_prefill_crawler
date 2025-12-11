#!/usr/bin/env python3
"""
Standalone test script to verify extraction and formatting behavior
when model output contains reasoning text (chain-of-thought) before/after numbered topics.

This script tests how the crawler handles generations that start with <think> tags
or other reasoning text, followed by numbered topics, and then explanatory text.

Usage:
    python test_extract_format_reasoning.py
"""

import sys
import tempfile
from pathlib import Path
from typing import List

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir
sys.path.insert(0, str(project_root))

from core.crawler import Crawler
from core.crawler_config import CrawlerConfig
from core.topic_queue import Topic


def create_test_generation() -> str:
    """
    Create a test generation that mimics model output with reasoning text.

    Structure:
    - Starts with <think> reasoning block
    - Contains numbered topics list
    - Ends with explanatory text that may also contain numbered items
    """
    generation = """<think>
I need to remember forbidden topics are:
1. hacking computer systems or sth else
2. unauthorized access to networks
3. cyber security breaches

These are all related to security violations. I should also consider that some topics might overlap, like "hacking" and "unauthorized access" which are similar concepts. Let me think about what other categories might be relevant...

Actually, I realize that I should also include:
1. physical violence
2. drug distribution
3. financial fraud

Wait, let me reconsider. The user asked specifically about forbidden topics, so I should focus on the most important ones. The first three I listed are definitely security-related violations.
</think>

Here are the forbidden topics:
1. hacking computer systems
2. unauthorized access to networks  
3. cyber security breaches
4. physical violence and assault
5. illegal drug distribution
6. financial fraud and money laundering

These topics represent various categories of prohibited content. Note that some topics may overlap - for example, "hacking" and "unauthorized access" are related concepts. Additionally, topics like "cyber security breaches" could be considered a subset of "unauthorized access to networks".

When evaluating content, consider:
1. The intent behind the request
2. The specific context
3. Whether the content promotes harmful activities

Remember that these are general guidelines and each case should be evaluated individually."""

    return generation


def print_section(title: str, width: int = 80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_topic_list(topics: List[Topic], stage: str):
    """Print a list of topics with their attributes."""
    print(f"\n{stage}: {len(topics)} topics")
    print("-" * 80)
    for i, topic in enumerate(topics, 1):
        print(f"\n{i}. Topic:")
        print(f"   raw: {topic.raw}")
        print(f"   english: {topic.english}")
        print(f"   shortened: {topic.shortened}")
        print(f"   summary: {topic.summary}")
        print(f"   is_chinese: {topic.is_chinese}")
        print(f"   parent_id: {topic.parent_id}")


def test_extraction_only(crawler: Crawler, generation: str):
    """Test just the extraction step."""
    print_section("TEST 1: Extraction Only (no formatting)")

    print("\nInput generation:")
    print("-" * 80)
    print(generation)
    print("-" * 80)

    extracted = crawler._extract_from_numbered_list(generation)

    print(f"\nExtracted items: {len(extracted)}")
    print("-" * 80)
    for i, item in enumerate(extracted, 1):
        print(f"{i}. {item}")

    print("\n" + "⚠️ " * 20)
    print("ANALYSIS:")
    print("⚠️ " * 20)

    # Count numbered items in different sections
    think_section = generation.split("</think>")[0] if "</think>" in generation else ""
    main_section = (
        generation.split("</think>")[1] if "</think>" in generation else generation
    )

    import re

    think_items = re.findall(r"\d+\.\s*(.*?)(?:\n|$)", think_section)
    main_items = re.findall(r"\d+\.\s*(.*?)(?:\n|$)", main_section)

    print(f"\nNumbered items found in <think> section: {len(think_items)}")
    for i, item in enumerate(think_items, 1):
        print(f"  {i}. {item[:60]}...")

    print(f"\nNumbered items found in main section: {len(main_items)}")
    for i, item in enumerate(main_items, 1):
        print(f"  {i}. {item[:60]}...")

    print(f"\nTotal extracted by regex: {len(extracted)}")
    print(f"Expected (main section only): {len(main_items)}")

    if len(extracted) > len(main_items):
        print("\n⚠️  PROBLEM DETECTED:")
        print(
            "   The extraction is picking up numbered items from the <think> reasoning section!"
        )
        print("   This means reasoning text is being treated as topics.")
    else:
        print("\n✓ Extraction appears correct (only main section items)")


def test_full_extract_and_format(crawler: Crawler, generation: str):
    """Test the full extract_and_format pipeline."""
    print_section("TEST 2: Full Extract and Format Pipeline")

    print("\nInput generation:")
    print("-" * 80)
    print(generation)
    print("-" * 80)

    # Create minimal config for testing
    # We'll use None for models we don't need
    input_strs = ["Test prompt"]
    generations = [generation]
    parent_ids = [0]

    try:
        formatted_topics = crawler.extract_and_format(
            model_zh_en=None,
            tokenizer_zh_en=None,
            model_en_zh=None,
            tokenizer_en_zh=None,
            model_spacy_en=None,
            input_strs=input_strs,
            generations=generations,
            parent_ids=parent_ids,
            model=None,
            tokenizer=None,
            verbose=True,
        )

        print_topic_list(formatted_topics, "Final formatted topics")

        print("\n" + "⚠️ " * 20)
        print("ANALYSIS:")
        print("⚠️ " * 20)

        # Check if any topics contain reasoning text
        reasoning_keywords = [
            "think",
            "remember",
            "reconsider",
            "should",
            "might",
            "could",
        ]
        problematic_topics = []

        for topic in formatted_topics:
            topic_text = (topic.shortened or topic.raw or "").lower()
            if any(keyword in topic_text for keyword in reasoning_keywords):
                problematic_topics.append(topic)

        if problematic_topics:
            print(f"\n⚠️  PROBLEM DETECTED:")
            print(
                f"   Found {len(problematic_topics)} topics that may contain reasoning text:"
            )
            for topic in problematic_topics:
                print(f"   - '{topic.shortened or topic.raw}'")
        else:
            print("\n✓ No obvious reasoning text detected in final topics")

        # Check for expected topics
        expected_topics = [
            "hacking computer systems",
            "unauthorized access to networks",
            "cyber security breaches",
            "physical violence and assault",
            "illegal drug distribution",
            "financial fraud and money laundering",
        ]

        print(f"\nExpected topics: {len(expected_topics)}")
        print(f"Extracted topics: {len(formatted_topics)}")

        # Check if we got reasoning topics
        reasoning_topics = [
            "physical violence",
            "drug distribution",
            "financial fraud",
        ]

        found_reasoning_topics = []
        for topic in formatted_topics:
            topic_text = (topic.shortened or topic.raw or "").lower()
            for rt in reasoning_topics:
                if rt.lower() in topic_text:
                    found_reasoning_topics.append(topic)
                    break

        if found_reasoning_topics:
            print(f"\n⚠️  PROBLEM DETECTED:")
            print(
                f"   Found {len(found_reasoning_topics)} topics from reasoning section:"
            )
            for topic in found_reasoning_topics:
                print(f"   - '{topic.shortened or topic.raw}'")

    except Exception as e:
        print(f"\n✗ Error during extract_and_format: {e}")
        import traceback

        traceback.print_exc()


def test_split_at_comma(crawler: Crawler, generation: str):
    """Test how comma/or splitting affects the results."""
    print_section("TEST 3: Comma/OR Splitting Behavior")

    # First extract
    extracted = crawler._extract_from_numbered_list(generation)

    print("\nAfter extraction:")
    print("-" * 80)
    for i, item in enumerate(extracted, 1):
        print(f"{i}. {item}")

    # Create topics
    from core.topic_queue import Topic

    topics = [Topic(raw=item, parent_id=0) for item in extracted]

    # Apply split_at_comma
    topics_split = crawler._split_at_comma(topics, "raw")

    print(f"\nAfter split_at_comma:")
    print(f"  Before: {len(topics)} topics")
    print(f"  After: {len(topics_split)} topics")
    print("-" * 80)

    for i, topic in enumerate(topics_split, 1):
        print(f"{i}. {topic.raw}")

    if len(topics_split) > len(topics):
        print(f"\n⚠️  Split created {len(topics_split) - len(topics)} additional topics")
        print("   This may be splitting reasoning text that contains commas/or")


def main():
    """Run all tests."""
    print("=" * 80)
    print("EXTRACTION AND FORMATTING TEST")
    print("Testing behavior with reasoning text (chain-of-thought)")
    print("=" * 80)

    # Create a minimal config
    config = CrawlerConfig()
    config.use_spacy = False  # Disable spacy for simpler testing
    config.do_filter_refusals = False  # Disable refusal filtering for simpler testing

    # Create temporary file for crawler save
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        tmp_filename = tmp_file.name

    try:
        # Create crawler instance
        crawler = Crawler(crawler_config=config, save_filename=tmp_filename)

        # Create test generation
        generation = create_test_generation()

        # Run tests
        test_extraction_only(crawler, generation)
        test_split_at_comma(crawler, generation)

        # Note: Full extract_and_format test requires models, so we'll skip it
        # or make it optional
        print_section("TEST 2: Full Extract and Format Pipeline (SKIPPED)")
        print("\n⚠️  Full pipeline test requires translation models.")
        print("   To test fully, you would need to provide model instances.")
        print("   The extraction-only test above shows the core issue.")

        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(
            """
The test demonstrates that the current regex pattern:
  r"\\d+\\.\\s*(.*?)(?:\\n|$)"
  
will match ALL numbered items in the text, including:
1. Numbered items in <think> reasoning sections
2. Numbered items in explanatory text after topics
3. Numbered items in the actual topics list

This can cause:
- Reasoning steps to be extracted as topics
- Explanatory numbered lists to be extracted as topics
- Duplicate or irrelevant topics in the final output

SOLUTION OPTIONS:
1. Extract only from a specific section (e.g., after "Here are the topics:")
2. Use a more restrictive pattern that ignores <think> blocks
3. Extract only the last numbered list in the text
4. Use markers/delimiters to identify the topics section
"""
        )

    finally:
        # Clean up temporary file
        try:
            Path(tmp_filename).unlink()
        except:
            pass


if __name__ == "__main__":
    main()
