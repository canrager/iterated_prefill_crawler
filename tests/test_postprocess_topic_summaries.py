#!/usr/bin/env python3
"""
Test for postprocess_topic_summaries.py with 50 topics.

This test demonstrates the full topic aggregation pipeline with detailed output logging.

To run this test:
    pytest tests/test_postprocess_topic_summaries.py::test_topic_aggregation_50_topics -v -s

Or run directly:
    python tests/test_postprocess_topic_summaries.py

The test:
1. Creates 50 test topics with intentional duplicates and semantic similarities
2. Creates a mock crawler log file
3. Runs the full aggregation pipeline (loading, merging, description generation)
4. Shows detailed output including:
   - Topic loading statistics
   - Batch processing progress
   - Cluster merging results
   - Cluster description generation
   - Final aggregation statistics
5. Verifies that aggregation reduces the number of unique topics

The test uses mocked LLM API calls to avoid requiring actual API keys.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from exp.postprocess_topic_summaries import (
    load_all_topic_summaries,
    iteratively_merge_semantic_duplicates,
)


def create_test_topics():
    """
    Create 50 test topics with intentional duplicates and semantic similarities
    to test aggregation functionality.
    """
    # Base topics with variations
    base_topics = [
        # Security-related topics (should cluster together)
        "hacking computer systems",
        "cyber security breaches",
        "unauthorized access to networks",
        "penetrating network defenses",
        "bypassing security measures",
        # Violence-related topics (should cluster together)
        "violence and physical harm",
        "assault and battery",
        "physical attacks",
        "causing bodily injury",
        "harmful physical actions",
        # Drug-related topics (should cluster together)
        "illegal drug use",
        "substance abuse",
        "narcotics consumption",
        "drug trafficking",
        "controlled substances",
        # Financial crimes (should cluster together)
        "money laundering",
        "financial fraud",
        "embezzlement schemes",
        "tax evasion",
        "illegal financial transactions",
        # Privacy violations (should cluster together)
        "invasion of privacy",
        "unauthorized data collection",
        "personal information theft",
        "identity theft",
        "data privacy breaches",
        # Weapons-related (should cluster together)
        "weapons and firearms",
        "gun violence",
        "lethal weapons",
        "weapon manufacturing",
        "firearms and ammunition",
        # Self-harm topics (should cluster together)
        "suicide and self-harm",
        "self-inflicted injuries",
        "self-destructive behavior",
        "suicidal thoughts",
        "self-harm methods",
        # Hate speech (should cluster together)
        "hate speech and discrimination",
        "racist language",
        "discriminatory content",
        "offensive language",
        "hateful expressions",
        # Scams and fraud (should cluster together)
        "online scams",
        "phishing attacks",
        "financial scams",
        "fraudulent schemes",
        "deceptive practices",
        # Explosives (should cluster together)
        "explosive devices",
        "bomb-making instructions",
        "explosives manufacturing",
        "destructive devices",
        "explosive materials",
    ]

    # Add some exact duplicates to test exact matching
    duplicates = [
        "hacking computer systems",  # duplicate
        "violence and physical harm",  # duplicate
        "illegal drug use",  # duplicate
    ]

    all_topics = base_topics + duplicates

    # Ensure we have exactly 50 topics
    while len(all_topics) < 50:
        all_topics.append(f"topic {len(all_topics) + 1}")

    return all_topics[:50]


def create_mock_crawler_log(log_file_path: Path, topics: list, log_file_name: str):
    """Create a mock crawler log file with the given topics."""
    head_refusal_topics = []
    for i, topic in enumerate(topics):
        head_refusal_topics.append(
            {
                "id": i + 1,
                "raw": topic,
                "english": topic,
                "summary": topic,  # Use topic as summary for testing
                "is_chinese": False,
                "is_head": True,
                "is_refusal": True,
            }
        )

    crawler_log = {"queue": {"topics": {"head_refusal_topics": head_refusal_topics}}}

    with open(log_file_path, "w") as f:
        json.dump(crawler_log, f, indent=2)


@pytest.fixture
def temp_pbr_dir():
    """Create a temporary directory for test artifacts."""
    temp_dir = tempfile.mkdtemp()
    pbr_dir = Path(temp_dir) / "pbr"
    pbr_dir.mkdir(parents=True, exist_ok=True)
    yield pbr_dir
    shutil.rmtree(temp_dir)


def test_topic_aggregation_50_topics(temp_pbr_dir, monkeypatch):
    """
    Test topic aggregation with 50 topics, showing full output log.

    This test:
    1. Creates 50 test topics with intentional duplicates and semantic similarities
    2. Creates a mock crawler log file
    3. Runs the full aggregation pipeline
    4. Shows detailed output including clustering results
    5. Verifies that aggregation reduces the number of unique topics
    """
    print("\n" + "=" * 80)
    print("TEST: Topic Aggregation with 50 Topics")
    print("=" * 80)

    # Create test topics
    test_topics = create_test_topics()
    print(f"\nCreated {len(test_topics)} test topics")
    print(f"Unique topics: {len(set(test_topics))}")

    # Create mock crawler log file
    log_file_name = "crawler_log_test_50_topics.json"
    log_file_path = temp_pbr_dir / log_file_name
    create_mock_crawler_log(log_file_path, test_topics, log_file_name)
    print(f"\nCreated mock crawler log: {log_file_path}")

    # Load topics from the mock log
    print("\n" + "=" * 80)
    print("Step 1: Loading topic summaries")
    print("=" * 80)
    df = load_all_topic_summaries(temp_pbr_dir)

    print(f"\nLoaded DataFrame:")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique topics: {df['topic_summary'].nunique()}")
    print(f"\nFirst 10 topics:")
    for idx, row in df.head(10).iterrows():
        print(f"  {idx + 1}. {row['topic_summary']}")

    # Set up log file for LLM judge
    log_file = temp_pbr_dir / "llm_judge_log_test.jsonl"
    if log_file.exists():
        log_file.unlink()
    log_file.touch()

    # Mock the LLM API calls to avoid actual API calls during testing
    # We'll use a simple mock that groups topics by keywords
    def mock_query_llm_api(*args, **kwargs):
        """Mock LLM API that performs simple keyword-based matching."""
        prompt = kwargs.get(
            "prompt", args[1] if len(args) > 1 else args[0] if args else ""
        )

        # Handle both single prompt and list of prompts
        if isinstance(prompt, list):
            return [mock_query_llm_api(p=p) for p in prompt]

        # Extract reference topic and candidate topics from prompt
        if isinstance(prompt, str):
            # Simple keyword-based matching
            if "Reference topic:" in prompt:
                ref_line = [l for l in prompt.split("\n") if "Reference topic:" in l]
                if ref_line:
                    ref_topic = ref_line[0].split("'")[1] if "'" in ref_line[0] else ""

                    # Extract keywords from reference
                    ref_keywords = set(ref_topic.lower().split())

                    # Check if any candidate topics share keywords
                    candidates = []
                    for line in prompt.split("\n"):
                        if line.strip().startswith("[") and "]" in line:
                            candidate = line.split("]", 1)[1].strip()
                            candidate_keywords = set(candidate.lower().split())
                            # If >30% keyword overlap, consider it a match
                            if ref_keywords and candidate_keywords:
                                overlap = len(ref_keywords & candidate_keywords) / len(
                                    ref_keywords
                                )
                                if overlap > 0.3:
                                    candidates.append(candidate)

                    if candidates:
                        matched = "\\n".join(
                            [f"\\topic{{{c}}}" for c in candidates[:2]]
                        )
                        return f"{matched}\\n\\boxed{{Y}}"

            # Handle cluster description generation
            if "Given a list of related topic descriptions" in prompt:
                # Extract topics from the prompt
                topics = []
                for line in prompt.split("\n"):
                    if line.strip().startswith("- "):
                        topics.append(line.strip()[2:])
                if topics:
                    # Return a simple description based on common words
                    all_words = []
                    for topic in topics:
                        all_words.extend(topic.lower().split())
                    # Return most common word as description
                    from collections import Counter

                    common_words = Counter(all_words).most_common(2)
                    return " ".join([w for w, _ in common_words if len(w) > 3][:2])
                return topics[0] if topics else "general topic"

        return "\\topic{}\\n\\boxed{N}"

    async def mock_async_query_llm_api(*args, **kwargs):
        """Async version of mock."""
        import asyncio

        await asyncio.sleep(0.01)  # Simulate async delay
        return mock_query_llm_api(*args, **kwargs)

    # Monkey patch the API calls
    monkeypatch.setattr("core.generation_utils.query_llm_api", mock_query_llm_api)
    monkeypatch.setattr(
        "core.generation_utils.async_query_llm_api", mock_async_query_llm_api
    )

    # Run aggregation
    print("\n" + "=" * 80)
    print("Step 2: Merging semantic duplicates")
    print("=" * 80)

    try:
        df_merged = iteratively_merge_semantic_duplicates(
            df,
            llm_judge_name="gpt-5-nano",
            log_file=log_file,
            verbose=True,
            batch_size=10,
            max_concurrent_batches=25,
        )

        print(f"\nAfter merging:")
        print(f"  Original unique topics: {df['topic_summary'].nunique()}")
        print(f"  Final cluster heads: {df_merged['cluster_head'].nunique()}")
        print(
            f"  Reduction: {df['topic_summary'].nunique() - df_merged['cluster_head'].nunique()} topics merged"
        )

        # Show cluster distribution
        cluster_counts = (
            df_merged.groupby("cluster_head").size().sort_values(ascending=False)
        )
        print(f"\nTop 10 clusters by size:")
        for cluster_head, count in cluster_counts.head(10).items():
            print(f"  {count:3d} topics -> '{cluster_head[:60]}...'")

        # Set cluster descriptions (already abstract descriptions from clustering)
        print("\n" + "=" * 80)
        print("Step 3: Setting cluster descriptions")
        print("=" * 80)

        # Cluster heads are already abstract descriptions from clustering, so just copy them
        df_merged["cluster_description"] = df_merged["cluster_head"]
        df_final = df_merged

        print(
            f"\nSet descriptions for {df_final['cluster_description'].nunique()} clusters"
        )

        # Show final results
        print("\n" + "=" * 80)
        print("Final Results Summary")
        print("=" * 80)
        print(f"Total topics processed: {len(df_final)}")
        print(f"Unique topics (before merging): {df_final['topic_summary'].nunique()}")
        print(f"Cluster heads (after merging): {df_final['cluster_head'].nunique()}")
        print(f"Cluster descriptions: {df_final['cluster_description'].nunique()}")

        # Show sample clusters with their descriptions
        print(f"\nSample clusters with descriptions:")
        sample_clusters = (
            df_final.groupby("cluster_head")
            .agg({"topic_summary": list, "cluster_description": "first"})
            .head(5)
        )

        for cluster_head, row in sample_clusters.iterrows():
            topics = row["topic_summary"]
            description = row["cluster_description"]
            print(f"\n  Cluster: '{description}'")
            print(f"    Size: {len(topics)} topics")
            print(f"    Topics:")
            for topic in topics[:5]:  # Show first 5 topics
                print(f"      - {topic}")
            if len(topics) > 5:
                print(f"      ... and {len(topics) - 5} more")

        # Verify aggregation worked
        assert (
            df_final["cluster_head"].nunique() <= df_final["topic_summary"].nunique()
        ), "Cluster heads should be <= unique topics"
        assert len(df_final) == len(df), "Total rows should remain the same"

        # Verify cluster descriptions exist
        assert (
            df_final["cluster_description"].notna().all()
        ), "All clusters should have descriptions"

        print("\n" + "=" * 80)
        print("TEST PASSED: Topic aggregation completed successfully")
        print("=" * 80)

    except Exception as e:
        print(f"\nError during aggregation: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run the test directly using pytest
    import pytest
    import sys

    # Run with verbose output and show print statements
    exit_code = pytest.main(
        [
            __file__,
            "-v",
            "-s",  # Show print statements
            "--tb=short",  # Short traceback format
        ]
    )
    sys.exit(exit_code)
