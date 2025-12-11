#!/usr/bin/env python3
"""
Plot discovered meta clusters over crawl steps from meta_clusters.json.

For each model:
- For each meta cluster, find the lowest topic ID from IPC and baseline separately
- Rank topics by ID (ascending) - these represent crawl steps
- Track cumulative number of unique meta clusters discovered as we go through topics
- Plot step functions: 3 subplots (one per model), each with 2 lines (IPC and baseline)
  showing number of discovered meta clusters vs crawl steps (topic ID)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import sys

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


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
    }


def extract_topics_from_meta_clusters(
    meta_clusters: Dict,
) -> Dict[str, Dict[str, List[Tuple[int, str]]]]:
    """
    Extract topics from meta clusters, grouped by model and prefill mode.
    For each meta cluster and model+prefill mode, use the lowest topic ID across all subclusters.

    Returns:
        Dictionary: model -> prefill_mode -> list of (topic_id, meta_cluster_name) tuples
        where topic_id is the minimum ID for that meta_cluster_name in that model+prefill_mode
    """
    # Track minimum topic ID per meta cluster per model+prefill mode
    model_prefill_meta_cluster_min_id = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: float("inf")))
    )

    def extract_from_value(value, meta_cluster_name):
        """Recursively extract topics from nested structure."""
        if isinstance(value, list):
            for item in value:
                if isinstance(item, list):
                    # Each topic is [topic_dict, log_filename]
                    if (
                        len(item) >= 2
                        and isinstance(item[0], dict)
                        and isinstance(item[1], str)
                    ):
                        topic_dict = item[0]
                        log_filename = item[1]

                        # Normalize filename
                        if not log_filename.endswith(".json"):
                            log_filename = f"{log_filename}.json"

                        # Parse log filename
                        parsed = parse_crawler_log_filename(log_filename)
                        if parsed:
                            model = parsed["model"]
                            prefill_mode = parsed["prefill_mode"]
                            topic_id = topic_dict.get("id")

                            if topic_id is not None:
                                # Track minimum topic ID for this meta cluster + model + prefill mode
                                current_min = model_prefill_meta_cluster_min_id[model][
                                    prefill_mode
                                ][meta_cluster_name]
                                if topic_id < current_min:
                                    model_prefill_meta_cluster_min_id[model][
                                        prefill_mode
                                    ][meta_cluster_name] = topic_id
                    else:
                        extract_from_value(item, meta_cluster_name)
                elif isinstance(item, dict):
                    extract_from_value(item, meta_cluster_name)
        elif isinstance(value, dict):
            for v in value.values():
                extract_from_value(v, meta_cluster_name)

    for meta_cluster_name, meta_cluster_data in meta_clusters.items():
        extract_from_value(meta_cluster_data, meta_cluster_name)

    # Convert to list format: model -> prefill_mode -> list of (topic_id, meta_cluster_name)
    model_prefill_topics = defaultdict(lambda: defaultdict(list))
    for model, prefill_modes in model_prefill_meta_cluster_min_id.items():
        for prefill_mode, meta_cluster_ids in prefill_modes.items():
            for meta_cluster_name, min_topic_id in meta_cluster_ids.items():
                if min_topic_id != float("inf"):
                    model_prefill_topics[model][prefill_mode].append(
                        (min_topic_id, meta_cluster_name)
                    )

    return dict(model_prefill_topics)


def get_ipc_prefill_mode(model: str, available_modes: Set[str]) -> Optional[str]:
    """
    Determine IPC prefill mode for a model (assistant or thought, whichever is available).

    Returns:
        'assistant' or 'thought' if available, None otherwise
    """
    if "assistant" in available_modes:
        return "assistant"
    elif "thought" in available_modes:
        return "thought"
    return None


def compute_discovered_meta_clusters(
    topic_ids_with_meta_clusters: List[Tuple[int, str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cumulative number of discovered meta clusters over crawl steps (topic IDs).

    Args:
        topic_ids_with_meta_clusters: List of (topic_id, meta_cluster_name) tuples

    Returns:
        Tuple of (topic_ids_array, num_discovered_meta_clusters_array)
        where topic_ids_array contains the topic IDs and num_discovered_meta_clusters_array
        contains the cumulative count of unique meta clusters discovered up to that point
    """
    # Sort by topic ID (ascending)
    sorted_topics = sorted(topic_ids_with_meta_clusters, key=lambda x: x[0])

    num_topics = len(sorted_topics)
    if num_topics == 0:
        return np.array([]), np.array([])

    # Track unique meta clusters we've seen
    seen_meta_clusters = set()

    topic_ids = []
    num_discovered = []

    for topic_id, meta_cluster_name in sorted_topics:
        seen_meta_clusters.add(meta_cluster_name)
        topic_ids.append(topic_id)
        num_discovered.append(len(seen_meta_clusters))

    return np.array(topic_ids), np.array(num_discovered)


def plot_discovered_meta_clusters(
    model_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    output_path: Path,
):
    """
    Plot step functions showing cumulative discovered meta clusters over crawl steps:
    3 subplots (one per model), each with 2 lines (IPC and baseline).

    Args:
        model_data: Dictionary mapping model -> {'IPC': (topic_ids, num_discovered), 'baseline': (topic_ids, num_discovered)}
        output_path: Path to save the plot
    """
    # Set font properties
    plt.rcParams.update({"font.size": 12, "font.family": "Palatino"})

    models = sorted(model_data.keys())
    num_models = len(models)

    if num_models == 0:
        print("No models to plot")
        return

    # Create subplots: 1 row, 3 columns
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6))

    if num_models == 1:
        axes = [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        data = model_data[model]

        # Plot IPC step function
        if "IPC" in data:
            topic_ids_ipc, num_discovered_ipc = data["IPC"]
            if len(topic_ids_ipc) > 0:
                # Create step function: use step='post' to make it right-continuous
                ax.step(
                    topic_ids_ipc,
                    num_discovered_ipc,
                    where="post",
                    label="IPC",
                    linewidth=2,
                )

        # Plot baseline step function
        if "baseline" in data:
            topic_ids_baseline, num_discovered_baseline = data["baseline"]
            if len(topic_ids_baseline) > 0:
                ax.step(
                    topic_ids_baseline,
                    num_discovered_baseline,
                    where="post",
                    label="baseline",
                    linewidth=2,
                )

        ax.set_xlabel("Crawl Steps (Topic ID)", fontsize=12)
        ax.set_ylabel("Number of Discovered Meta Clusters", fontsize=12)
        ax.set_title(model, fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Discovered meta clusters plot saved to {output_path}")


def main(
    meta_clusters_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
):
    """
    Main function.

    Args:
        meta_clusters_file: Path to meta_clusters.json (default: artifacts/pbr/meta_clusters.json)
        output_dir: Directory to save output files (default: artifacts/pbr)
    """
    # Set up paths
    if meta_clusters_file is None:
        meta_clusters_file = project_root / "artifacts" / "pbr" / "meta_clusters.json"

    if output_dir is None:
        output_dir = project_root / "artifacts" / "pbr"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Discovered Meta Clusters Plotting from Meta Clusters")
    print("=" * 80)

    # Load meta clusters
    print(f"\nLoading meta clusters from {meta_clusters_file}...")
    with open(meta_clusters_file, "r") as f:
        meta_clusters = json.load(f)
    print(f"Found {len(meta_clusters)} meta clusters")

    # Extract topics grouped by model and prefill mode
    print("\nExtracting topics from meta clusters...")
    model_prefill_topics = extract_topics_from_meta_clusters(meta_clusters)

    print(f"Found {len(model_prefill_topics)} models:")
    for model, prefill_modes in model_prefill_topics.items():
        print(f"  {model}: {list(prefill_modes.keys())}")

    # Process each model
    model_roc_data = {}

    for model, prefill_modes in model_prefill_topics.items():
        print(f"\nProcessing model: {model}")

        # Determine IPC prefill mode (assistant or thought)
        available_modes = set(prefill_modes.keys())
        ipc_mode = get_ipc_prefill_mode(model, available_modes)

        if ipc_mode is None:
            print(
                f"  Warning: No IPC mode (assistant/thought) found for {model}, skipping"
            )
            continue

        if "user" not in available_modes:
            print(f"  Warning: No baseline mode (user) found for {model}, skipping")
            continue

        print(f"  IPC mode: {ipc_mode}")
        print(f"  Baseline mode: user")

        # Get topics for IPC and baseline
        ipc_topics = prefill_modes.get(ipc_mode, [])
        baseline_topics = prefill_modes.get("user", [])

        print(f"  IPC topics: {len(ipc_topics)}")
        print(f"  Baseline topics: {len(baseline_topics)}")

        # Build ground truth: union of meta clusters from IPC and baseline
        ipc_meta_clusters = set([meta_cluster for _, meta_cluster in ipc_topics])
        baseline_meta_clusters = set(
            [meta_cluster for _, meta_cluster in baseline_topics]
        )
        ground_truth_meta_clusters = ipc_meta_clusters | baseline_meta_clusters

        print(f"  Ground truth meta clusters: {len(ground_truth_meta_clusters)}")

        # Compute discovered meta clusters over crawl steps for IPC and baseline
        topic_ids_ipc, num_discovered_ipc = compute_discovered_meta_clusters(ipc_topics)
        topic_ids_baseline, num_discovered_baseline = compute_discovered_meta_clusters(
            baseline_topics
        )

        model_roc_data[model] = {
            "IPC": (topic_ids_ipc, num_discovered_ipc),
            "baseline": (topic_ids_baseline, num_discovered_baseline),
        }

    # Plot discovered meta clusters
    if model_roc_data:
        output_path = output_dir / "discovered_meta_clusters.png"
        plot_discovered_meta_clusters(model_roc_data, output_path)
    else:
        print("\nNo data to plot")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot discovered meta clusters over crawl steps from meta_clusters.json"
    )
    parser.add_argument(
        "--meta_clusters_file",
        type=str,
        help="Path to meta_clusters.json (default: artifacts/pbr/meta_clusters.json)",
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
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
