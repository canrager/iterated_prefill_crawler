#!/usr/bin/env python3
"""
Standalone Topic Aggregation Test Script

This script demonstrates the full topic aggregation pipeline using real vLLM models.
It processes all crawler log files in the test_output/pbr directory, loads a real vLLM model,
and runs the complete aggregation process with detailed logging - no mocks involved.

The script:
1. Loads all crawler_log_*.json files from test_output/pbr
2. Aggregates topics within each log file
3. Aggregates cluster heads across log files (if multiple files exist)
4. Generates hierarchy log showing the full aggregation path

Usage:
    python standalone_topic_aggregation_test.py --model allenai/Olmo-3-7B-Think

    # With custom settings:
    python standalone_topic_aggregation_test.py --model allenai/Olmo-3-7B-Think --batch_size 3 --gpu_memory 0.8

Requirements:
    - GPU with sufficient memory
    - vLLM installed
    - transformers installed
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import argparse

# Set environment variable to force spawn method before any imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import multiprocessing

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir
sys.path.insert(0, str(project_root))

from exp.postprocess_topic_summaries import (
    load_all_topic_summaries,
    iteratively_merge_semantic_duplicates,
    aggregate_cluster_heads_across_log_files,
    load_vllm_judge_model,
    write_detailed_log,
    build_hierarchy_tree,
    generate_markdown_hierarchy_tree,
)


def create_test_topics():
    """
    Create 12 test topics with intentional duplicates and semantic similarities
    to test aggregation functionality.
    """
    # Base topics with variations - designed to cluster into semantic groups
    base_topics = [
        # Security-related topics (should cluster together)
        "hacking computer systems",
        "cyber security breaches",
        "unauthorized access to networks",
        # Violence-related topics (should cluster together)
        "violence and physical harm",
        "assault and battery",
        "physical attacks",
        # Drug-related topics (should cluster together)
        "illegal drug use",
        "substance abuse",
        # Financial crimes (should cluster together)
        "money laundering",
        "financial fraud",
        # Privacy violations (should cluster together)
        "invasion of privacy",
        "unauthorized data collection",
    ]

    # Add some exact duplicates to test exact matching
    duplicates = [
        "hacking computer systems",  # duplicate
    ]

    all_topics = base_topics + duplicates

    # Ensure we have exactly 12 topics
    while len(all_topics) < 12:
        all_topics.append(f"miscellaneous topic {len(all_topics) + 1}")

    return all_topics[:12]


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


def main():
    parser = argparse.ArgumentParser(
        description="Standalone topic aggregation test with real vLLM model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/Olmo-3-7B-Think",
        help="HuggingFace model name for vLLM (default: allenai/Olmo-3-7B-Think)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/home/can/models",
        help="Cache directory for vLLM model (default: /home/can/models)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for vLLM tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--gpu_memory",
        type=float,
        default=0.9,
        help="GPU memory fraction to use for vLLM (default: 0.9)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=5000,
        help="Maximum sequence length for vLLM (default: 5000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size for clustering (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Maximum retries for JSON parsing (default: 10)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_output",
        help="Output directory for results (default: ./test_output)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STANDALONE TOPIC AGGREGATION TEST")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"GPU memory: {args.gpu_memory}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pbr_dir = output_dir / "pbr"
    pbr_dir.mkdir(parents=True, exist_ok=True)

    # Create test topics
    print("\nStep 1: Creating test dataset")
    print("-" * 40)
    test_topics = create_test_topics()
    print(f"Created {len(test_topics)} test topics")
    print(f"Unique topics: {len(set(test_topics))}")
    print("\nSample topics:")
    for i, topic in enumerate(test_topics[:10]):
        print(f"  {i+1}. {topic}")
    if len(test_topics) > 10:
        print(f"  ... and {len(test_topics) - 10} more")

    # Create mock crawler log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"crawler_log_test_{timestamp}.json"
    log_file_path = pbr_dir / log_file_name
    create_mock_crawler_log(log_file_path, test_topics, log_file_name)
    print(f"\nCreated mock crawler log: {log_file_path}")

    # Set up logging files
    llm_judge_log_file = pbr_dir / f"llm_judge_log_{timestamp}.jsonl"
    detailed_log_file = pbr_dir / f"detailed_log_{timestamp}.txt"

    # Initialize log files
    if llm_judge_log_file.exists():
        llm_judge_log_file.unlink()
    llm_judge_log_file.touch()

    if detailed_log_file.exists():
        detailed_log_file.unlink()
    detailed_log_file.touch()

    # Write header to detailed log file
    with open(detailed_log_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED PROCESSING LOG (STANDALONE TEST)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Test: Standalone Topic Aggregation with {len(test_topics)} Topics\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write("=" * 80 + "\n\n")

    print(f"LLM judge log file: {llm_judge_log_file}")
    print(f"Detailed log file: {detailed_log_file}")

    # Load vLLM model
    print(f"\nStep 2: Loading vLLM model")
    print("-" * 40)
    print(f"Loading model: {args.model}")
    print("This may take several minutes...")

    try:
        vllm_model, vllm_tokenizer = load_vllm_judge_model(
            model_name=args.model,
            cache_dir=args.cache_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory,
            max_model_len=args.max_model_len,
        )
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nTroubleshooting tips:")
        print("- Ensure you have sufficient GPU memory")
        print("- Try reducing --gpu_memory (e.g., 0.7)")
        print("- Try reducing --max_model_len (e.g., 3000)")
        print("- Ensure vLLM is properly installed")
        return 1

    # Load topics from crawler log
    print(f"\nStep 3: Loading topics from crawler log")
    print("-" * 40)
    try:
        log_file_dfs = load_all_topic_summaries(pbr_dir)
        print(f"Loaded {len(log_file_dfs)} log files")

        total_topics = sum(len(df) for df in log_file_dfs.values())
        print(f"Total topics: {total_topics}")

        for log_name, df in log_file_dfs.items():
            print(
                f"  {log_name}: {len(df)} topics ({df['topic_summary'].nunique()} unique)"
            )

    except Exception as e:
        print(f"✗ Error loading topics: {e}")
        return 1

    # Step 4: Aggregate topics within each log file
    print(f"\nStep 4: Aggregating topics within each log file")
    print("-" * 40)
    aggregated_log_file_dfs = {}
    per_file_hierarchies: Dict[str, Dict[str, Any]] = {}

    try:
        for log_file_name, df in log_file_dfs.items():
            print(f"\n{'='*80}")
            print(f"Processing log file: {log_file_name}")
            print(f"{'='*80}")
            print(f"  Total topics: {len(df)}")
            print(f"  Unique topics: {df['topic_summary'].nunique()}")

            # Aggregate within this log file
            df_aggregated, hierarchy_info = iteratively_merge_semantic_duplicates(
                df,
                model=vllm_model,
                tokenizer=vllm_tokenizer,
                log_file=llm_judge_log_file,
                verbose=True,
                batch_size=args.batch_size,
                detailed_log_file=detailed_log_file,
                temperature=args.temperature,
                max_retries=args.max_retries,
            )
            print(
                f"  After aggregation: {df_aggregated['cluster_head'].nunique()} cluster heads"
            )

            # Store hierarchy info for this log file
            per_file_hierarchies[log_file_name] = hierarchy_info

            # Cluster heads are already abstract descriptions from clustering, so just copy them
            df_aggregated["cluster_description"] = df_aggregated["cluster_head"]

            aggregated_log_file_dfs[log_file_name] = df_aggregated

            # Save per-file results
            per_file_output = (
                pbr_dir
                / f"topic_summaries_{log_file_name.replace('.json', '')}_aggregated.json"
            )
            df_aggregated.to_json(per_file_output, orient="records", indent=2)
            print(f"  Saved per-file aggregated results to: {per_file_output}")

    except Exception as e:
        print(f"✗ Error during within-file merging: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Step 5: Aggregate cluster heads across log files (if multiple files)
    if len(log_file_dfs) > 1:
        print(f"\nStep 5: Aggregating cluster heads across log files")
        print("-" * 40)
        try:
            df_final, cross_file_mapping = aggregate_cluster_heads_across_log_files(
                aggregated_log_file_dfs,
                model=vllm_model,
                tokenizer=vllm_tokenizer,
                log_file=llm_judge_log_file,
                verbose=True,
                batch_size=args.batch_size,
                detailed_log_file=detailed_log_file,
                temperature=args.temperature,
                max_retries=args.max_retries,
            )
            print(
                f"✓ Cross-file aggregation completed!"
            )
            print(f"  Final cluster heads: {df_final['cluster_head'].nunique()}")
        except Exception as e:
            print(f"✗ Error during cross-file aggregation: {e}")
            import traceback

            traceback.print_exc()
            return 1
    else:
        # Single file: no cross-file aggregation needed
        print(f"\nStep 5: Single file - skipping cross-file aggregation")
        print("-" * 40)
        df_final = list(aggregated_log_file_dfs.values())[0]
        # Create identity mapping for single file
        cross_file_mapping = {
            cluster_head: cluster_head
            for cluster_head in df_final["cluster_head"].unique()
        }
        print(f"  Using within-file clusters directly")

    # Set cluster descriptions (already abstract descriptions from clustering)
    print(f"\nStep 6: Setting cluster descriptions")
    print("-" * 40)
    df_final["cluster_description"] = df_final["cluster_head"]

    # Write final summary to detailed log
    write_detailed_log(
        detailed_log_file,
        "Standalone Test - Clustering Complete",
        f"✓ Merging completed!\n"
        f"  Number of log files processed: {len(log_file_dfs)}\n"
        f"  Total topics: {len(df_final)}\n"
        f"  Unique topics (before merging): {df_final['topic_summary'].nunique()}\n"
        f"  Final cluster heads: {df_final['cluster_head'].nunique()}\n"
        f"  Reduction: {df_final['topic_summary'].nunique() - df_final['cluster_head'].nunique()} topics merged\n\n"
        + "Final cluster distribution:\n"
        + "\n".join(
            [
                f"  {cluster_head}: {count} topics"
                for cluster_head, count in df_final.groupby("cluster_head")
                .size()
                .sort_values(ascending=False)
                .items()
            ]
        ),
    )

    # Ensure all writes are flushed
    if detailed_log_file.exists():
        with open(detailed_log_file, "a", encoding="utf-8") as f:
            f.flush()
            os.fsync(f.fileno())

    # Generate hierarchy log
    print(f"\nStep 7: Generating hierarchy log")
    print("-" * 40)

    hierarchy_log_file = pbr_dir / f"hierarchy_log_{timestamp}.md"

    try:
        # Build hierarchy tree
        hierarchy_tree = build_hierarchy_tree(
            per_file_hierarchies, cross_file_mapping, df_final
        )

        # Generate markdown with per-file hierarchies for better documentation
        markdown_content = generate_markdown_hierarchy_tree(
            hierarchy_tree, df_final, per_file_hierarchies=per_file_hierarchies
        )

        # Save hierarchy log
        with open(hierarchy_log_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"✓ Generated hierarchy log: {hierarchy_log_file}")

    except Exception as e:
        print(f"✗ Error generating hierarchy log: {e}")
        import traceback

        traceback.print_exc()
        # Don't return error - hierarchy log is optional

    # Save results
    print(f"\nStep 8: Saving results")
    print("-" * 40)

    # Save final DataFrame
    output_file = pbr_dir / f"topic_summaries_merged_{timestamp}.json"
    df_final.to_json(output_file, orient="records", indent=2)
    print(f"✓ Saved merged topics: {output_file}")

    # Show final results
    print(f"\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total topics processed: {len(df_final)}")
    print(f"Unique topics (before merging): {df_final['topic_summary'].nunique()}")
    print(f"Cluster heads (after merging): {df_final['cluster_head'].nunique()}")
    print(f"Cluster descriptions: {df_final['cluster_description'].nunique()}")

    # Show cluster distribution
    cluster_counts = (
        df_final.groupby("cluster_head").size().sort_values(ascending=False)
    )
    print(f"\nTop clusters by size:")
    for cluster_head, count in cluster_counts.head(10).items():
        cluster_desc = df_final[df_final["cluster_head"] == cluster_head][
            "cluster_description"
        ].iloc[0]
        print(f"  {count:3d} topics -> '{cluster_desc}'")

    # Show sample clusters with their topics
    print(f"\nSample clusters with topics:")
    sample_clusters = (
        df_final.groupby("cluster_head")
        .agg({"topic_summary": list, "cluster_description": "first"})
        .head(5)
    )

    for cluster_head, row in sample_clusters.iterrows():
        topics = row["topic_summary"]
        description = row["cluster_description"]
        print(f"\n  📁 Cluster: '{description}'")
        print(f"     Size: {len(topics)} topics")
        print(f"     Topics:")
        for topic in topics[:5]:
            print(f"       • {topic}")
        if len(topics) > 5:
            print(f"       ... and {len(topics) - 5} more")

    # Show log file information
    print(f"\n" + "=" * 80)
    print("LOG FILES GENERATED")
    print("=" * 80)

    if detailed_log_file.exists():
        size_bytes = detailed_log_file.stat().st_size
        size_kb = size_bytes / 1024
        print(f"📄 Detailed log: {detailed_log_file}")
        print(f"   Size: {size_bytes:,} bytes ({size_kb:.1f} KB)")

        # Show sample content
        print(f"   Sample content:")
        with open(detailed_log_file, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")
            print("   " + "─" * 50)
            for line in lines[:20]:  # Show first 20 lines
                print(f"   {line}")
            if len(lines) > 20:
                print(f"   ... and {len(lines) - 20} more lines")
            print("   " + "─" * 50)

    if llm_judge_log_file.exists():
        size_bytes = llm_judge_log_file.stat().st_size
        size_kb = size_bytes / 1024
        print(f"\n📄 LLM judge log: {llm_judge_log_file}")
        print(f"   Size: {size_bytes:,} bytes ({size_kb:.1f} KB)")

        # Count entries
        with open(llm_judge_log_file, "r") as f:
            entries = sum(1 for line in f if line.strip())
        print(f"   Entries: {entries}")

    if hierarchy_log_file.exists():
        size_bytes = hierarchy_log_file.stat().st_size
        size_kb = size_bytes / 1024
        print(f"\n📄 Hierarchy log: {hierarchy_log_file}")
        print(f"   Size: {size_bytes:,} bytes ({size_kb:.1f} KB)")

        # Show sample content
        print(f"   Sample content:")
        with open(hierarchy_log_file, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")
            print("   " + "─" * 50)
            for line in lines[:15]:  # Show first 15 lines
                print(f"   {line}")
            if len(lines) > 15:
                print(f"   ... and {len(lines) - 15} more lines")
            print("   " + "─" * 50)

    print(f"\n✅ Test completed successfully!")
    print(f"📁 All files saved in: {output_dir}")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
