#!/usr/bin/env python3
"""
Analyze baseline evaluation results per model.

This script:
1. Groups crawler logs in /artifacts/baseline_eval by model
2. For each model, runs topic summarization on each run (user_prefill, assistant/thought_prefill)
3. Creates a merged summarization across runs for that model
4. Calculates recall metrics: what fraction of merged topics does each run cover?
5. Saves results to /artifacts/baseline_eval/model_analysis/
"""

import os

# Set environment variable to force spawn method before any imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import multiprocessing

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# MUST be set before importing torch or any CUDA libraries
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

import sys
import json
import glob
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import from postprocess_topic_summaries.py
from exp.postprocess_topic_summaries import (
    load_vllm_judge_model,
    load_topics_from_log_file,
    iteratively_merge_semantic_duplicates,
    aggregate_cluster_heads_across_log_files,
    build_hierarchy_tree,
    generate_markdown_hierarchy_tree,
    write_detailed_log,
)


def parse_crawler_log_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse crawler log filename to extract model and prefill method.

    Format: crawler_log_YYYYMMDD_HHMMSS_MODEL_Nsamples_Mcrawls_TRUEFILTERfilter_PREFILL_METHOD_BACKEND.json
    """
    basename = filename.replace(".json", "")
    # Pattern: crawler_log_timestamp_model_samples_crawls_filter_prefill_backend
    pattern = r"crawler_log_(\d{8})_(\d{6})_(.+?)_(\d+)samples_(\d+)crawls_(.+?)filter_(.+?)_vllm"

    match = re.match(pattern, basename)
    if not match:
        return None

    (
        timestamp_date,
        timestamp_time,
        model,
        samples,
        crawls,
        filter_type,
        prefill_method,
    ) = match.groups()

    return {
        "model": model,
        "prefill_method": prefill_method,
        "timestamp": f"{timestamp_date}_{timestamp_time}",
        "filename": filename,
    }


def group_logs_by_model(baseline_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    """
    Group crawler logs by model name.

    Args:
        baseline_dir: Directory containing baseline evaluation logs

    Returns:
        Dictionary mapping model name to list of log file info dicts
    """
    log_files = glob.glob(str(baseline_dir / "crawler_log_*.json"))

    model_groups = defaultdict(list)

    for log_file in log_files:
        filename = os.path.basename(log_file)
        parsed = parse_crawler_log_filename(filename)

        if parsed:
            model_groups[parsed["model"]].append(
                {
                    "path": log_file,
                    "filename": filename,
                    "prefill_method": parsed["prefill_method"],
                    "timestamp": parsed["timestamp"],
                }
            )
        else:
            print(f"Warning: Could not parse filename: {filename}")

    return dict(model_groups)


def analyze_model_runs(
    model_name: str,
    log_files: List[Dict[str, str]],
    output_dir: Path,
    vllm_model,
    vllm_tokenizer,
    batch_size: int = 5,
    temperature: float = 0.6,
    max_retries: int = 10,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Analyze all runs for a single model.

    Args:
        model_name: Name of the model
        log_files: List of log file info dicts for this model
        output_dir: Output directory for results
        vllm_model: vLLM model for topic clustering
        vllm_tokenizer: Tokenizer for vLLM model
        batch_size: Batch size for clustering
        temperature: Temperature for clustering
        max_retries: Max retries for clustering
        verbose: Verbose output

    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*80}")
    print(f"Analyzing model: {model_name}")
    print(f"{'='*80}")
    print(f"Found {len(log_files)} runs:")
    for log_info in log_files:
        print(f"  - {log_info['prefill_method']}: {log_info['filename']}")

    # Create output directory for this model
    model_output_dir = output_dir / model_name.replace("/", "_")
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Set up log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    llm_judge_log = model_output_dir / f"llm_judge_log_{timestamp}.jsonl"
    detailed_log = model_output_dir / f"detailed_log_{timestamp}.txt"

    # Initialize log files
    if llm_judge_log.exists():
        llm_judge_log.unlink()
    llm_judge_log.touch()

    if detailed_log.exists():
        detailed_log.unlink()
    detailed_log.touch()

    # Write header to detailed log
    with open(detailed_log, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"BASELINE ANALYSIS FOR MODEL: {model_name}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Number of runs: {len(log_files)}\n")
        f.write("=" * 80 + "\n\n")

    # Step 1: Load topics from each run and aggregate within each run
    print(f"\nStep 1: Loading and aggregating topics for each run")
    print(f"{'='*80}")

    run_results = {}
    per_run_hierarchies = {}

    for log_info in log_files:
        run_name = log_info["prefill_method"]
        log_path = Path(log_info["path"])

        print(f"\nProcessing run: {run_name}")
        print(f"  File: {log_info['filename']}")

        # Load topics from this log file
        df = load_topics_from_log_file(log_path)
        print(f"  Loaded {len(df)} topics ({df['topic_summary'].nunique()} unique)")

        if len(df) == 0:
            print(f"  Warning: No topics found in {run_name}, skipping...")
            continue

        # Aggregate topics within this run
        print(f"  Aggregating topics for {run_name}...")
        df_aggregated, hierarchy_info = iteratively_merge_semantic_duplicates(
            df,
            model=vllm_model,
            tokenizer=vllm_tokenizer,
            log_file=llm_judge_log,
            verbose=verbose,
            batch_size=batch_size,
            detailed_log_file=detailed_log,
            temperature=temperature,
            max_retries=max_retries,
        )

        print(
            f"  After aggregation: {df_aggregated['cluster_head'].nunique()} cluster heads"
        )

        # Store results
        run_results[run_name] = df_aggregated
        per_run_hierarchies[run_name] = hierarchy_info

        # Save per-run aggregated results
        per_run_output = model_output_dir / f"topics_{run_name}_aggregated.json"
        df_aggregated.to_json(per_run_output, orient="records", indent=2)
        print(f"  Saved to: {per_run_output}")

    if not run_results:
        print(f"Warning: No valid runs found for model {model_name}")
        return {}

    # Step 2: Merge cluster heads across all runs for this model
    print(f"\n{'='*80}")
    print(f"Step 2: Merging cluster heads across runs")
    print(f"{'='*80}")

    df_merged, cross_run_mapping = aggregate_cluster_heads_across_log_files(
        run_results,
        model=vllm_model,
        tokenizer=vllm_tokenizer,
        log_file=llm_judge_log,
        verbose=verbose,
        batch_size=batch_size,
        detailed_log_file=detailed_log,
        temperature=temperature,
        max_retries=max_retries,
    )

    # Set cluster descriptions
    df_merged["cluster_description"] = df_merged["cluster_head"]

    # Save merged results
    merged_output = model_output_dir / f"topics_merged_all_runs.json"
    df_merged.to_json(merged_output, orient="records", indent=2)
    print(f"Saved merged results to: {merged_output}")
    print(f"  Total topics: {len(df_merged)}")
    print(f"  Unique cluster heads (union): {df_merged['cluster_head'].nunique()}")

    # Step 3: Calculate recall metrics
    print(f"\n{'='*80}")
    print(f"Step 3: Calculating recall metrics")
    print(f"{'='*80}")

    # Get union of all topics (merged cluster heads)
    union_topics = set(df_merged["cluster_head"].unique())
    union_size = len(union_topics)

    print(f"Union of topics across all runs: {union_size}")

    # For each run, calculate what fraction of union topics it covers
    recall_metrics = {}

    for run_name, df_run in run_results.items():
        run_topics = set(df_run["cluster_head"].unique())

        # Map run topics to merged topics using cross_run_mapping
        # cross_run_mapping maps within-run cluster heads to final merged cluster heads
        mapped_topics = set()
        for topic in run_topics:
            if topic in cross_run_mapping:
                mapped_topics.add(cross_run_mapping[topic])
            else:
                # If not in mapping, it might already be a final cluster head
                mapped_topics.add(topic)

        # Calculate recall
        coverage = len(mapped_topics & union_topics)
        recall = coverage / union_size if union_size > 0 else 0.0

        recall_metrics[run_name] = {
            "topics_discovered": len(run_topics),
            "topics_in_union": coverage,
            "recall": recall,
            "unique_to_this_run": len(
                mapped_topics - union_topics
            ),  # Should be 0 if union is correct
        }

        print(f"\n{run_name}:")
        print(f"  Topics discovered (before merging): {len(run_topics)}")
        print(f"  Topics in union (after merging): {coverage}")
        print(f"  Recall: {recall:.4f} ({coverage}/{union_size})")

    # Step 4: Generate hierarchy visualization
    print(f"\n{'='*80}")
    print(f"Step 4: Generating hierarchy visualization")
    print(f"{'='*80}")

    try:
        hierarchy_tree = build_hierarchy_tree(
            per_run_hierarchies, cross_run_mapping, df_merged
        )
        markdown_content = generate_markdown_hierarchy_tree(
            hierarchy_tree, df_merged, per_file_hierarchies=per_run_hierarchies
        )
        hierarchy_log_file = model_output_dir / f"hierarchy_log_{timestamp}.md"
        with open(hierarchy_log_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Saved hierarchy log to: {hierarchy_log_file}")
    except Exception as e:
        print(f"Warning: Failed to generate hierarchy log: {e}")
        import traceback

        traceback.print_exc()

    # Step 5: Save analysis summary
    print(f"\n{'='*80}")
    print(f"Step 5: Saving analysis summary")
    print(f"{'='*80}")

    summary = {
        "model_name": model_name,
        "timestamp": timestamp,
        "runs_analyzed": len(run_results),
        "run_info": {
            run_name: {
                "filename": next(
                    (
                        lf["filename"]
                        for lf in log_files
                        if lf["prefill_method"] == run_name
                    ),
                    None,
                ),
                "topics_before_aggregation": len(run_results[run_name]),
                "topics_after_aggregation": run_results[run_name][
                    "cluster_head"
                ].nunique(),
            }
            for run_name in run_results.keys()
        },
        "union_topics": union_size,
        "recall_metrics": recall_metrics,
    }

    summary_file = model_output_dir / f"analysis_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved analysis summary to: {summary_file}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary for {model_name}")
    print(f"{'='*80}")
    print(f"Runs analyzed: {len(run_results)}")
    print(f"Union of topics: {union_size}")
    print(f"\nRecall per run:")
    for run_name, metrics in sorted(
        recall_metrics.items(), key=lambda x: x[1]["recall"], reverse=True
    ):
        print(
            f"  {run_name:30s} Recall: {metrics['recall']:.4f} ({metrics['topics_in_union']}/{union_size})"
        )

    return summary


def main(
    vllm_model_name: str = "allenai/Olmo-3-7B-Think",
    vllm_cache_dir: Optional[str] = None,
    vllm_tensor_parallel_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_max_model_len: Optional[int] = None,
    batch_size: int = 5,
    temperature: float = 0.6,
    max_retries: int = 10,
    verbose: bool = False,
):
    """
    Main function.

    Args:
        vllm_model_name: HuggingFace model name for vLLM clustering
        vllm_cache_dir: Cache directory for vLLM model
        vllm_tensor_parallel_size: Number of GPUs for tensor parallelism
        vllm_gpu_memory_utilization: GPU memory fraction to use
        vllm_max_model_len: Maximum sequence length
        batch_size: Batch size for clustering
        temperature: Temperature for clustering
        max_retries: Max retries for clustering
        verbose: Verbose output
    """
    # Set up paths
    baseline_dir = Path(project_root) / "artifacts" / "baseline_eval"
    output_dir = baseline_dir / "model_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BASELINE EVALUATION ANALYSIS PER MODEL")
    print("=" * 80)
    print(f"Baseline directory: {baseline_dir}")
    print(f"Output directory: {output_dir}")

    # Load vLLM model for clustering
    print(f"\nLoading vLLM model: {vllm_model_name}")
    vllm_model, vllm_tokenizer = load_vllm_judge_model(
        model_name=vllm_model_name,
        cache_dir=vllm_cache_dir,
        tensor_parallel_size=vllm_tensor_parallel_size,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
        max_model_len=vllm_max_model_len,
    )
    print(f"vLLM model loaded successfully\n")

    # Group logs by model
    print("=" * 80)
    print("Grouping crawler logs by model")
    print("=" * 80)
    model_groups = group_logs_by_model(baseline_dir)

    print(f"Found {len(model_groups)} unique models:")
    for model_name, log_files in model_groups.items():
        print(f"  {model_name}: {len(log_files)} runs")
        for log_info in log_files:
            print(f"    - {log_info['prefill_method']}")

    # Analyze each model
    all_summaries = {}

    for model_name, log_files in model_groups.items():
        try:
            summary = analyze_model_runs(
                model_name=model_name,
                log_files=log_files,
                output_dir=output_dir,
                vllm_model=vllm_model,
                vllm_tokenizer=vllm_tokenizer,
                batch_size=batch_size,
                temperature=temperature,
                max_retries=max_retries,
                verbose=verbose,
            )
            all_summaries[model_name] = summary
        except Exception as e:
            print(f"Error analyzing model {model_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save overall summary
    print(f"\n{'='*80}")
    print("Saving overall summary")
    print(f"{'='*80}")

    overall_summary = {
        "timestamp": datetime.now().isoformat(),
        "models_analyzed": len(all_summaries),
        "models": all_summaries,
    }

    overall_summary_file = output_dir / "overall_summary.json"
    with open(overall_summary_file, "w") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved overall summary to: {overall_summary_file}")

    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Models analyzed: {len(all_summaries)}")
    for model_name, summary in all_summaries.items():
        if not summary:
            continue
        print(f"\n{model_name}:")
        print(f"  Runs: {summary.get('runs_analyzed', 0)}")
        print(f"  Union topics: {summary.get('union_topics', 0)}")
        recall_metrics = summary.get("recall_metrics", {})
        if recall_metrics:
            print(f"  Recall per run:")
            for run_name, metrics in sorted(
                recall_metrics.items(), key=lambda x: x[1]["recall"], reverse=True
            ):
                print(f"    {run_name:30s} {metrics['recall']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze baseline evaluation results per model"
    )
    parser.add_argument(
        "--vllm_model_name",
        type=str,
        help="HuggingFace model name for vLLM (e.g., 'allenai/Olmo-3-7B-Think')",
        default="allenai/Olmo-3-7B-Think",
    )
    parser.add_argument(
        "--vllm_cache_dir",
        type=str,
        default="/home/can/models",
        help="Cache directory for vLLM model",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for vLLM tensor parallelism",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory fraction to use for vLLM (0.0-1.0)",
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=5000,
        help="Maximum sequence length for vLLM",
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
        help="Sampling temperature for vLLM (default: 0.6)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Maximum number of retries if JSON is not parsable (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    main(
        vllm_model_name=args.vllm_model_name,
        vllm_cache_dir=args.vllm_cache_dir,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_retries=args.max_retries,
        verbose=args.verbose,
    )


