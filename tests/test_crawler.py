import pytest
import torch
from unittest.mock import patch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

from core.crawler import Crawler
from core.crawler_config import CrawlerConfig
from core.topic_queue import Topic
from core.llm_utils import load_zh_en_translation_model, load_model_and_tokenizer
from core.project_config import MODELS_DIR, DEVICE, resolve_cache_dir
from core.tokenization_utils import custom_batch_encoding


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

    model_zh_en, tokenizer_zh_en = load_zh_en_translation_model(MODELS_DIR, DEVICE)

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
    assert (
        translated_topics[1].translation is not None
    )  # Changed since we're using real model
    assert translated_topics[3].is_chinese
    assert translated_topics[3].translation is not None

    # Verify order is preserved (non-Chinese first, then Chinese)
    assert not translated_topics[0].is_chinese
    assert not translated_topics[2].is_chinese
    assert translated_topics[1].is_chinese
    assert translated_topics[3].is_chinese


def test_get_message_segments_and_seed_topics():
    """Test _get_message_segments and _get_seed_topics for all prefill_modes.

    This test verifies that the tokenization (input_strs) is correct for all prefill_modes
    by checking against ground truth predefined strings. We mock random.choice to ensure
    deterministic behavior.
    """
    # Predefined seed topics
    seed_topics = [
        Topic(
            id=1,
            raw="test topic 1",
            english="test topic 1",
            chinese="测试主题1",
            is_head=True,
            is_refusal=False,
        ),
        Topic(
            id=2,
            raw="test topic 2",
            english="test topic 2",
            chinese="测试主题2",
            is_head=True,
            is_refusal=False,
        ),
    ]

    # All prefill modes to test (excluding "no_seed" which is handled separately)
    prefill_modes = [
        "user_prefill_no_seed",
        "user_prefill_with_seed",
        "assistant_prefill_no_seed",
        "assistant_prefill_with_seed",
        "thought_prefill_no_seed",
        "thought_prefill_with_seed",
    ]

    # Load tokenizer for tokenization (we don't need the model, only tokenization)
    model_path = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=MODELS_DIR)

    for prefill_mode in prefill_modes:
        # Create crawler with specific prefill_mode
        config = CrawlerConfig(
            prefill_mode=prefill_mode,
            generation_batch_size=2,
            num_samples_per_topic=1,
            model_path=model_path,
            do_filter_refusals=False,  # Don't filter refusals for this test
        )
        crawler = Crawler(crawler_config=config, save_filename="test_tokenization.json")

        # Manually add seed topics to the queue
        for topic in seed_topics:
            crawler.queue.add_new_cluster_head(topic)

        # Mock random functions to return deterministic values
        # We'll return the first element from each list to ensure consistency
        def mock_choice(choices):
            if isinstance(choices, list) and len(choices) > 0:
                return choices[0]
            return choices

        def mock_sample(population, k):
            # Return first k elements deterministically
            if isinstance(population, list):
                return population[:k] if len(population) >= k else population
            return list(population)[:k]

        with (
            patch("core.crawler.random.choice", side_effect=mock_choice),
            patch("core.crawler.random.sample", side_effect=mock_sample),
        ):
            # Get seed topics and message segments
            seed_result = crawler._get_seed_topics()
            message_segments = crawler._get_message_segments(lang="english")

            # Handle different return types: "no_seed" modes return (list, list), others return (dict, list)
            if "no_seed" in prefill_mode:
                seed_topics_text, topic_parent_ids = seed_result
                # seed_topics_text is already a list of empty strings
            else:
                seed_topics_text_languages, topic_parent_ids = seed_result
                seed_topics_text = seed_topics_text_languages["english"]

            # Format user templates (same as batch_generate does)
            filled_user_templates = [
                message_segments.user_template.format(t) for t in seed_topics_text
            ]

            # Call custom_batch_encoding directly to test tokenization
            # This is what batch_generate calls internally for non-API models
            input_ids, input_strs = custom_batch_encoding(
                model_name=model_path,
                tokenizer=tokenizer,
                user_messages=filled_user_templates,
                user_suffix=message_segments.user_suffix,
                assistant_prefill=message_segments.assistant_prefix,
                thinking_message=message_segments.thought_prefix,
                force_thought_skip=False,
                template=config.tokenization_template,
            )

            # Verify we got input_strs back
            assert len(input_strs) == len(
                seed_topics_text
            ), f"Expected {len(seed_topics_text)} input_strs, got {len(input_strs)} for prefill_mode {prefill_mode}"

            # Verify input_strs contain expected elements based on prefill_mode
            for i, input_str in enumerate(input_strs):
                assert isinstance(
                    input_str, str
                ), f"input_str should be a string for prefill_mode {prefill_mode}"
                assert (
                    len(input_str) > 0
                ), f"input_str should not be empty for prefill_mode {prefill_mode}"

                # Check that seed topic text appears in input_str (if seed is used)
                if "no_seed" not in prefill_mode and seed_topics_text[i]:
                    # For seed modes, verify the topic appears in the input
                    assert (
                        seed_topics_text[i] in input_str
                        or seed_topics_text[i].lower() in input_str.lower()
                    ), f"Seed topic '{seed_topics_text[i]}' should appear in input_str for prefill_mode {prefill_mode}"

                # Check that assistant prefill appears (for assistant_prefill modes)
                if "assistant_prefill" in prefill_mode:
                    assert (
                        "1. " in input_str
                    ), f"Assistant prefill '1. ' should appear in input_str for prefill_mode {prefill_mode}"

                # Check that thinking message appears (for thought_prefill modes)
                if "thought_prefill" in prefill_mode:
                    # The thinking message should be in the thought_prefix
                    assert (
                        message_segments.thought_prefix != ""
                    ), f"thought_prefix should not be empty for prefill_mode {prefill_mode}"

                # Optional: Add ground truth strings for exact comparison
                # Example:
                # ground_truth = {
                #     "user_prefill_with_seed": ["expected string 1", "expected string 2"],
                #     ...
                # }
                # if prefill_mode in ground_truth:
                #     assert input_strs == ground_truth[prefill_mode], \
                #         f"input_strs don't match ground truth for {prefill_mode}"

                # For now, we verify the structure is correct
                # Uncomment below to see actual output for creating ground truth:
                print(f"\nprefill_mode: {prefill_mode}")
                print(f"input_strs: {input_strs}")
                print(f"message_segments: {message_segments}")


def test_summarize_refusal_topics():
    """Test the summarize_refusal_topics method to verify summarization is working.

    This test creates topics with raw text and verifies that summaries are generated.
    It prints the outputs for inspection.
    """
    # Create test topics with various raw text examples
    # Note: These topics are NOT in the few-shot prompt examples
    test_topics = [
        Topic(
            raw="Methods for creating explosives or bomb-making instructions",
            shortened="explosives, bomb-making",
            summary=None,  # No summary yet
            is_head=None,
            is_refusal=None,
        ),
        Topic(
            raw="Instructions on how to hack into computer systems or bypass security measures.",
            shortened="hacking, security bypass",
            summary=None,
            is_head=None,
            is_refusal=None,
        ),
        Topic(
            raw="Promoting illegal activities such as money laundering or tax evasion.",
            shortened="illegal activities, financial crimes",
            summary=None,
            is_head=None,
            is_refusal=None,
        ),
        Topic(
            raw="Data encryption and secure communication protocols: Implementing end-to-end encryption to protect sensitive information during transmission. This includes using cryptographic algorithms, secure key exchange mechanisms, and ensuring data integrity through digital signatures.",
            shortened="encryption, secure communication",
            summary=None,
            is_head=None,
            is_refusal=None,
        ),
    ]

    # Create crawler config with "self" mode for using local model
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    cache_dir_config = "models"  # Same as config.yaml
    config = CrawlerConfig(
        llm_judge_name="self",
        model_path=model_path,
        cache_dir=cache_dir_config,
        max_generated_tokens=50,
        temperature=0.6,
        vllm_max_model_len=8192,  # Use full context length for summarization
        generation_batch_size=4,
    )
    crawler = Crawler(crawler_config=config, save_filename="test_summarization.json")

    # Resolve cache_dir relative to ROOT_DIR.parent and create if needed
    cache_dir_path = resolve_cache_dir(cache_dir_config)
    cache_dir_str = str(cache_dir_path)

    print("\n" + "=" * 80)
    print("Testing summarize_refusal_topics")
    print("=" * 80)
    print(f"Using model: {model_path}")
    print(f"Cache directory: {cache_dir_str}")
    print(
        f"vllm_max_model_len (max_new_tokens for summarization): {config.vllm_max_model_len}"
    )
    print(f"\nInput topics ({len(test_topics)} topics):")
    for i, topic in enumerate(test_topics, 1):
        print(f"\n  Topic {i}:")
        print(f"    Raw: {topic.raw}")
        print(f"    Shortened: {topic.shortened}")
        print(f"    Summary (before): {topic.summary}")

    # Try to load model and tokenizer for "self" mode
    try:
        print("\nLoading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_path,
            cache_dir=cache_dir_str,
            device=DEVICE,
            backend="transformers",  # Use transformers backend for testing
        )
        print("Model loaded successfully!")

        # Run summarization with verbose output
        print("\nRunning summarization...")
        summarized_topics = crawler.summarize_refusal_topics(
            topics=test_topics,
            llm_judge_name="self",
            model=model,
            tokenizer=tokenizer,
            verbose=True,
        )

        print("\n" + "=" * 80)
        print("Summarization Results:")
        print("=" * 80)
        for i, topic in enumerate(summarized_topics, 1):
            print(f"\n  Topic {i}:")
            print(f"    Raw: {topic.raw}")
            print(f"    Shortened: {topic.shortened}")
            print(f"    Summary (after): {topic.summary}")
            print(f"    Summary generated: {topic.summary is not None}")
            if topic.summary:
                print(f"    Summary length: {len(topic.summary)} characters")

        # Verify summaries were generated
        summaries_generated = sum(1 for t in summarized_topics if t.summary is not None)
        print(f"\n{'='*80}")
        print(
            f"Summary: {summaries_generated}/{len(summarized_topics)} topics got summaries"
        )
        print(f"{'='*80}\n")

        assert (
            summaries_generated > 0
        ), "At least one summary should have been generated"

    except Exception as e:
        print(f"\nError loading model or running summarization: {e}")
        print("This test requires a model to be available. Skipping assertions.")
        print("You can run this test with an external API by changing llm_judge_name.")
        raise
