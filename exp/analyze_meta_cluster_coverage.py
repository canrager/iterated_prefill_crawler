#!/usr/bin/env python3
"""
Analyze meta cluster coverage across models and prefill modes.

This script:
1. Loads meta_clusters.json
2. Creates a DataFrame with meta clusters as rows and crawler log filenames as columns (boolean)
3. Parses crawler log filenames to extract model name and prefill mode
4. For each model, computes union of topics across prefill modes = model refusal topics
5. Reports recall of each model+prefill mode for the corresponding model refusal topics
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

import pandas as pd
import sys

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from configs.directory_config import INTERIM_DIR


def parse_crawler_log_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse crawler log filename to extract model and prefill mode.

    Format: crawler_log_YYYYMMDD_HHMMSS_MODEL_Nsamples_Mcrawls_TRUEFILTERfilter_PREFILL_MODE_prompt_BACKEND.json
    or: crawler_log_YYYYMMDD_HHMMSS_MODEL_Nsamples_Mcrawls_TRUEFILTERfilter_PREFILL_MODE_with_seedprompt_vllm

    Returns:
        Dictionary with model, prefill_mode, and filename, or None if parsing fails
    """
    basename = filename.replace(".json", "")

    # Try pattern with "prompt" separator
    pattern1 = r"crawler_log_(\d{8})_(\d{6})_(.+?)_(\d+)samples_(\d+)crawls_(.+?)filter_(.+?)_prompt_(.+)"
    match = re.match(pattern1, basename)

    if not match:
        # Try pattern with "prefill" separator (e.g., user_prefill_with_seedprompt_vllm)
        pattern2 = r"crawler_log_(\d{8})_(\d{6})_(.+?)_(\d+)samples_(\d+)crawls_(.+?)filter_(.+?)_prefill_(.+)"
        match = re.match(pattern2, basename)

    if not match:
        return None

    date_str, time_str, model, samples, crawls, filter_val, prefill_mode, backend = (
        match.groups()
    )

    return {
        "model": model,
        "prefill_mode": prefill_mode,
        "filename": filename,
        "date_str": date_str,
        "time_str": time_str,
    }


def load_topics_from_log_file(log_file_path: Path) -> Set[str]:
    """
    Load topic summaries from a single crawler log file.

    Args:
        log_file_path: Path to the crawler log JSON file (should have .json extension)

    Returns:
        Set of topic summaries (using 'raw' field)
    """
    topics = set()

    try:
        with open(log_file_path, "r") as f:
            data = json.load(f)

        # Extract head_refusal_topics
        head_refusal_topics = (
            data.get("queue", {}).get("topics", {}).get("head_refusal_topics", [])
        )

        # Extract topic summaries
        for topic in head_refusal_topics:
            raw = topic.get("raw")
            if raw:
                topics.add(raw)

    except Exception as e:
        print(f"  Warning: Error loading topics from {log_file_path.name}: {e}")

    return topics


def load_meta_clusters(meta_clusters_file: Path) -> Dict:
    """Load meta clusters JSON file."""
    with open(meta_clusters_file, "r") as f:
        return json.load(f)


def extract_log_files_from_meta_clusters(meta_clusters: Dict) -> Set[str]:
    """
    Extract all unique crawler log filenames from meta clusters.

    Args:
        meta_clusters: Dictionary of meta clusters

    Returns:
        Set of crawler log filenames (normalized to include .json extension)
    """
    log_files = set()

    def extract_from_value(value):
        """Recursively extract log filenames from nested structure."""
        if isinstance(value, list):
            for item in value:
                if isinstance(item, list):
                    # Each topic is [topic_dict, log_filename]
                    if len(item) >= 2 and isinstance(item[1], str):
                        filename = item[1]
                        # Normalize to always include .json extension
                        if not filename.endswith(".json"):
                            filename = f"{filename}.json"
                        log_files.add(filename)
                    else:
                        extract_from_value(item)
                elif isinstance(item, dict):
                    extract_from_value(item)
        elif isinstance(value, dict):
            for v in value.values():
                extract_from_value(v)

    extract_from_value(meta_clusters)
    return log_files


def build_meta_cluster_coverage(meta_clusters: Dict) -> Dict[str, Set[str]]:
    """
    Build a mapping from meta cluster names to sets of crawler log filenames.

    Args:
        meta_clusters: Dictionary of meta clusters

    Returns:
        Dictionary mapping meta cluster name -> set of log filenames (normalized to include .json)
    """
    coverage = defaultdict(set)

    for meta_cluster_name, meta_cluster_data in meta_clusters.items():

        def extract_from_value(value, cluster_name):
            """Recursively extract log filenames from nested structure."""
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, list):
                        # Each topic is [topic_dict, log_filename]
                        if len(item) >= 2 and isinstance(item[1], str):
                            filename = item[1]
                            # Normalize to always include .json extension
                            if not filename.endswith(".json"):
                                filename = f"{filename}.json"
                            coverage[cluster_name].add(filename)
                        else:
                            extract_from_value(item, cluster_name)
                    elif isinstance(item, dict):
                        extract_from_value(item, cluster_name)
            elif isinstance(value, dict):
                for v in value.values():
                    extract_from_value(v, cluster_name)

        extract_from_value(meta_cluster_data, meta_cluster_name)

    return dict(coverage)


def create_coverage_dataframe(
    meta_clusters: Dict,
    log_files: Set[str],
    interim_dir: Path,
) -> pd.DataFrame:
    """
    Create DataFrame with meta clusters as rows and log files as columns (boolean).

    Args:
        meta_clusters: Dictionary of meta clusters
        log_files: Set of all crawler log filenames
        interim_dir: Directory where crawler logs are stored

    Returns:
        DataFrame with meta clusters as index and log files as columns
    """
    # Build coverage mapping
    coverage = build_meta_cluster_coverage(meta_clusters)

    # Create DataFrame
    meta_cluster_names = sorted(meta_clusters.keys())
    log_file_list = sorted(log_files)

    # Initialize DataFrame with False
    df = pd.DataFrame(
        False,
        index=meta_cluster_names,
        columns=log_file_list,
    )

    # Fill in True values
    for meta_cluster_name, covered_logs in coverage.items():
        for log_file in covered_logs:
            if log_file in df.columns:
                df.loc[meta_cluster_name, log_file] = True

    return df


def compute_model_refusal_topics(
    log_files: Set[str],
    interim_dir: Path,
) -> Dict[str, Set[str]]:
    """
    For each model, compute union of topics across prefill modes = model refusal topics.

    Args:
        log_files: Set of crawler log filenames
        interim_dir: Directory where crawler logs are stored

    Returns:
        Dictionary mapping model name -> set of topic summaries (model refusal topics)
    """
    # Group log files by model
    model_logs = defaultdict(list)

    for log_file in log_files:
        parsed = parse_crawler_log_filename(log_file)
        if parsed:
            model_logs[parsed["model"]].append((log_file, parsed["prefill_mode"]))

    # For each model, compute union of topics across prefill modes
    model_refusal_topics = {}

    for model, logs in model_logs.items():
        all_topics = set()
        for log_file, prefill_mode in logs:
            # Try with and without .json extension
            log_path = interim_dir / log_file
            if not log_path.exists():
                log_path = interim_dir / f"{log_file}.json"

            if log_path.exists():
                topics = load_topics_from_log_file(log_path)
                all_topics.update(topics)
            else:
                print(
                    f"  Warning: Log file not found: {log_file} (tried {interim_dir / log_file} and {interim_dir / f'{log_file}.json'} )"
                )

        model_refusal_topics[model] = all_topics
        print(f"Model {model}: {len(all_topics)} total refusal topics")

    return model_refusal_topics


def compute_model_meta_cluster_union(
    coverage_df: pd.DataFrame,
    log_files: Set[str],
) -> Dict[str, Set[str]]:
    """
    For each model, compute union of meta-clusters across all prefill modes.

    Args:
        coverage_df: DataFrame with meta clusters as rows and log files as columns (boolean)
        log_files: Set of crawler log filenames

    Returns:
        Dictionary mapping model name -> set of meta cluster names (model meta-cluster union)
    """
    # Group log files by model
    model_logs = defaultdict(set)

    for log_file in log_files:
        parsed = parse_crawler_log_filename(log_file)
        if parsed:
            # Normalize filename
            normalized_file = log_file
            if not normalized_file.endswith(".json"):
                normalized_file = f"{normalized_file}.json"
            model_logs[parsed["model"]].add(normalized_file)

    # For each model, compute union of meta-clusters
    model_meta_cluster_union = {}

    for model, model_log_set in model_logs.items():
        # Find meta-clusters that appear in ANY log file for this model
        model_log_list = [log for log in model_log_set if log in coverage_df.columns]
        if model_log_list:
            # For each meta-cluster, check if it appears in any log file for this model
            model_meta_clusters = set()
            for meta_cluster in coverage_df.index:
                if coverage_df.loc[meta_cluster, model_log_list].any():
                    model_meta_clusters.add(meta_cluster)
            model_meta_cluster_union[model] = model_meta_clusters
            print(f"Model {model}: {len(model_meta_clusters)} meta-clusters in union")
        else:
            model_meta_cluster_union[model] = set()
            print(f"Model {model}: 0 meta-clusters (no matching log files found)")

    return model_meta_cluster_union


def compute_meta_cluster_recall_per_model(
    coverage_df: pd.DataFrame,
    log_files: Set[str],
    model_meta_cluster_union: Dict[str, Set[str]],
) -> pd.DataFrame:
    """
    Compute meta-cluster recall for each model+prefill mode against model meta-cluster union.

    Args:
        coverage_df: DataFrame with meta clusters as rows and log files as columns (boolean)
        log_files: Set of crawler log filenames
        model_meta_cluster_union: Dictionary mapping model -> set of meta cluster names

    Returns:
        DataFrame with columns: model, prefill_mode, meta_cluster_recall, num_meta_clusters_in_mode, num_model_meta_clusters
    """
    results = []

    for log_file in sorted(log_files):
        parsed = parse_crawler_log_filename(log_file)
        if not parsed:
            continue

        model = parsed["model"]
        prefill_mode = parsed["prefill_mode"]

        # Normalize filename
        normalized_file = log_file
        if not normalized_file.endswith(".json"):
            normalized_file = f"{normalized_file}.json"

        if normalized_file not in coverage_df.columns:
            print(f"  Warning: Log file not in coverage DataFrame: {normalized_file}")
            continue

        # Find meta-clusters covered by this log file
        meta_clusters_in_mode = set(
            coverage_df.index[coverage_df[normalized_file] == True].tolist()
        )

        # Get model meta-cluster union
        model_meta_clusters = model_meta_cluster_union.get(model, set())

        if len(model_meta_clusters) == 0:
            recall = 0.0
        else:
            # Recall = meta-clusters in this mode / total model meta-clusters
            recall = len(meta_clusters_in_mode & model_meta_clusters) / len(
                model_meta_clusters
            )

        results.append(
            {
                "model": model,
                "prefill_mode": prefill_mode,
                "log_file": normalized_file,
                "meta_cluster_recall": recall,
                "num_meta_clusters_in_mode": len(meta_clusters_in_mode),
                "num_model_meta_clusters": len(model_meta_clusters),
                "num_recalled_meta_clusters": len(
                    meta_clusters_in_mode & model_meta_clusters
                ),
            }
        )

    return pd.DataFrame(results)


def compute_recall_per_prefill_mode(
    log_files: Set[str],
    model_refusal_topics: Dict[str, Set[str]],
    interim_dir: Path,
) -> pd.DataFrame:
    """
    Compute recall for each model+prefill mode against model refusal topics.

    Args:
        log_files: Set of crawler log filenames
        model_refusal_topics: Dictionary mapping model -> set of topic summaries
        interim_dir: Directory where crawler logs are stored

    Returns:
        DataFrame with columns: model, prefill_mode, recall, num_topics_in_mode, num_model_refusal_topics
    """
    results = []

    for log_file in sorted(log_files):
        parsed = parse_crawler_log_filename(log_file)
        if not parsed:
            continue

        model = parsed["model"]
        prefill_mode = parsed["prefill_mode"]

        # Load topics for this log file
        # Ensure .json extension
        if not log_file.endswith(".json"):
            log_file = f"{log_file}.json"

        log_path = interim_dir / log_file
        if not log_path.exists():
            print(f"  Warning: Log file not found: {log_path}")
            continue

        topics_in_mode = load_topics_from_log_file(log_path)
        model_topics = model_refusal_topics.get(model, set())

        if len(model_topics) == 0:
            recall = 0.0
        else:
            # Recall = topics in this mode / total model refusal topics
            recall = len(topics_in_mode & model_topics) / len(model_topics)

        results.append(
            {
                "model": model,
                "prefill_mode": prefill_mode,
                "log_file": log_file,
                "recall": recall,
                "num_topics_in_mode": len(topics_in_mode),
                "num_model_refusal_topics": len(model_topics),
                "num_covered_topics": len(topics_in_mode & model_topics),
            }
        )

    return pd.DataFrame(results)


def main(
    meta_clusters_file: Optional[Path] = None,
    interim_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
):
    """
    Main function.

    Args:
        meta_clusters_file: Path to meta_clusters.json (default: artifacts/pbr/meta_clusters.json)
        interim_dir: Directory where crawler logs are stored (default: artifacts/interim)
        output_dir: Directory to save output files (default: artifacts/pbr)
    """
    # Set up paths
    if meta_clusters_file is None:
        meta_clusters_file = project_root / "artifacts" / "pbr" / "meta_clusters.json"

    if interim_dir is None:
        interim_dir = INTERIM_DIR

    if output_dir is None:
        output_dir = project_root / "artifacts" / "pbr"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Meta Cluster Coverage Analysis")
    print("=" * 80)

    # Load meta clusters
    print(f"\nLoading meta clusters from {meta_clusters_file}...")
    meta_clusters = load_meta_clusters(meta_clusters_file)
    print(f"Found {len(meta_clusters)} meta clusters")

    # Extract all log files
    print("\nExtracting crawler log filenames from meta clusters...")
    log_files = extract_log_files_from_meta_clusters(meta_clusters)
    print(f"Found {len(log_files)} unique crawler log files")

    # Create coverage DataFrame
    print("\nCreating coverage DataFrame...")
    coverage_df = create_coverage_dataframe(meta_clusters, log_files, interim_dir)
    print(f"Coverage DataFrame shape: {coverage_df.shape}")
    print(f"  Rows (meta clusters): {coverage_df.shape[0]}")
    print(f"  Columns (log files): {coverage_df.shape[1]}")

    # Save coverage DataFrame
    coverage_output = output_dir / "meta_cluster_coverage.csv"
    coverage_df.to_csv(coverage_output)
    print(f"\nSaved coverage DataFrame to: {coverage_output}")

    # Compute model meta-cluster union (union across prefill modes per model)
    print("\n" + "=" * 80)
    print(
        "Computing model meta-cluster union (union across prefill modes per model)..."
    )
    print("=" * 80)
    model_meta_cluster_union = compute_model_meta_cluster_union(coverage_df, log_files)

    # Compute meta-cluster recall per model+prefill mode
    print("\n" + "=" * 80)
    print("Computing meta-cluster recall per model+prefill mode...")
    print("=" * 80)
    meta_cluster_recall_df = compute_meta_cluster_recall_per_model(
        coverage_df, log_files, model_meta_cluster_union
    )

    # Save meta-cluster recall DataFrame
    meta_cluster_recall_output = output_dir / "model_prefill_meta_cluster_recall.csv"
    meta_cluster_recall_df.to_csv(meta_cluster_recall_output, index=False)
    print(f"\nSaved meta-cluster recall DataFrame to: {meta_cluster_recall_output}")

    # Compute model refusal topics (union across prefill modes)
    print("\n" + "=" * 80)
    print("Computing model refusal topics (union across prefill modes)...")
    print("=" * 80)
    model_refusal_topics = compute_model_refusal_topics(log_files, interim_dir)

    # Compute recall per prefill mode
    print("\n" + "=" * 80)
    print("Computing recall per model+prefill mode...")
    print("=" * 80)
    recall_df = compute_recall_per_prefill_mode(
        log_files, model_refusal_topics, interim_dir
    )

    # Save recall DataFrame
    recall_output = output_dir / "model_prefill_recall.csv"
    recall_df.to_csv(recall_output, index=False)
    print(f"\nSaved recall DataFrame to: {recall_output}")

    # Print summary statistics for meta-cluster recall
    print("\n" + "=" * 80)
    print("Meta-Cluster Recall Summary Statistics (PER MODEL)")
    print("=" * 80)
    print(f"\nMeta-cluster recall by prefill mode:")
    print(
        meta_cluster_recall_df.groupby("prefill_mode")["meta_cluster_recall"].agg(
            ["mean", "std", "min", "max"]
        )
    )

    print(f"\nMeta-cluster recall by model:")
    print(
        meta_cluster_recall_df.groupby("model")["meta_cluster_recall"].agg(
            ["mean", "std", "min", "max"]
        )
    )

    print(f"\nTop 10 models by average meta-cluster recall:")
    model_avg_meta_cluster_recall = (
        meta_cluster_recall_df.groupby("model")["meta_cluster_recall"]
        .mean()
        .sort_values(ascending=False)
    )
    print(model_avg_meta_cluster_recall.head(10))

    print(f"\nDetailed meta-cluster recall per model+prefill mode:")
    print(
        meta_cluster_recall_df[
            [
                "model",
                "prefill_mode",
                "meta_cluster_recall",
                "num_recalled_meta_clusters",
                "num_model_meta_clusters",
                "num_meta_clusters_in_mode",
            ]
        ].to_string(index=False)
    )

    # Print summary statistics for topic recall
    print("\n" + "=" * 80)
    print("Topic Recall Summary Statistics")
    print("=" * 80)
    print(f"\nRecall by prefill mode:")
    print(
        recall_df.groupby("prefill_mode")["recall"].agg(["mean", "std", "min", "max"])
    )

    print(f"\nRecall by model:")
    print(recall_df.groupby("model")["recall"].agg(["mean", "std", "min", "max"]))

    print(f"\nTop 10 models by average recall:")
    model_avg_recall = (
        recall_df.groupby("model")["recall"].mean().sort_values(ascending=False)
    )
    print(model_avg_recall.head(10))

    print(f"\nDetailed recall per model+prefill mode:")
    print(
        recall_df[
            [
                "model",
                "prefill_mode",
                "recall",
                "num_topics_in_mode",
                "num_model_refusal_topics",
                "num_covered_topics",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze meta cluster coverage across models and prefill modes"
    )
    parser.add_argument(
        "--meta_clusters_file",
        type=str,
        help="Path to meta_clusters.json (default: artifacts/pbr/meta_clusters.json)",
    )
    parser.add_argument(
        "--interim_dir",
        type=str,
        help="Directory where crawler logs are stored (default: artifacts/interim)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save output files (default: artifacts/pbr)",
    )

    args = parser.parse_args()

    main(
        meta_clusters_file=(
            Path(args.meta_clusters_file) if args.meta_clusters_file else None
        ),
        interim_dir=Path(args.interim_dir) if args.interim_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
