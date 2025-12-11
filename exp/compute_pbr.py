#!/usr/bin/env python3
"""
Compute Pool-based Recall (PBR) metric for each model.

This script:
1. Loads benchmark and crawler logs
2. For each model, identifies topics discovered in initial crawl
3. Tests models on topics from pooled reference set they haven't been tested on
4. Calculates PBR scores
5. Saves results to /artifacts/pbr/pbr_results.json
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
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from core.generation_utils import batch_generate, MessageSegments
from core.crawler_config import CrawlerConfig
from core.topic_queue import Topic
from core.llm_utils import load_model_and_tokenizer
from core.project_config import resolve_cache_dir


@dataclass
class ModelInfo:
    """Information about a model from crawler log."""

    log_file: str
    model_name: str
    backend: str
    config: Optional[CrawlerConfig] = None


def parse_crawler_log_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parse crawler log filename to extract model info."""
    basename = filename.replace(".json", "")
    pattern = r"crawler_log_(\d{8})_(\d{6})_(.+?)_(\d+)samples_(\d+)crawls_(.+?)filter_(.+?)prompt_(.+)"

    match = re.match(pattern, basename)
    if not match:
        return None

    _, _, model, _, _, _, _, backend = match.groups()

    return {
        "model": model,
        "backend": backend,
    }


def load_crawler_log(log_file: Path) -> Dict:
    """Load crawler log JSON."""
    with open(log_file, "r") as f:
        return json.load(f)


def extract_discovered_topics(crawler_data: Dict) -> Set[str]:
    """Extract topic summaries discovered by model (head_refusal_topics)."""
    topics = set()
    head_refusal_topics = (
        crawler_data.get("queue", {}).get("topics", {}).get("head_refusal_topics", [])
    )

    for topic in head_refusal_topics:
        summary = topic.get("summary")
        if summary:
            topics.add(summary)

    return topics


def extract_tested_topics(crawler_data: Dict) -> Set[str]:
    """Extract all topics that were tested (both refused and not refused)."""
    topics = set()

    # Get all head topics (both refused and not)
    head_topics = crawler_data.get("queue", {}).get("topics", {}).get("head_topics", [])
    for topic in head_topics:
        summary = topic.get("summary")
        if summary:
            topics.add(summary)

    return topics


def check_refusal_for_queries(
    model,
    tokenizer,
    queries: List[str],
    config: CrawlerConfig,
    verbose: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Check if model refuses queries for a topic.

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        queries: List of queries to test
        config: Crawler config
        verbose: Whether to print verbose output

    Returns:
        Tuple of (is_refusal, responses)
    """
    if not queries:
        return False, []

    # Generate responses for queries
    try:
        # Create message segments (simple template for testing)
        message_segments = MessageSegments(
            user_template="{}",
            user_prefix="",
            user_suffix="",
            assistant_prefix="",
            thought_prefix="",
            thought_suffix="",
        )

        responses, _ = batch_generate(
            model=model,
            tokenizer=tokenizer,
            selected_topics=queries,
            message_segments=message_segments,
            force_thought_skip=config.do_force_thought_skip,
            tokenization_template=config.tokenization_template,
            num_samples_per_topic=1,
            max_new_tokens=config.max_refusal_check_generated_tokens,
            temperature=1.0,
            verbose=verbose,
            cfg=config,
        )
    except Exception as e:
        print(f"  Error generating responses: {e}")
        return False, []

    # Check for refusals
    refusal_count = 0
    for response in responses:
        if is_refusal(response, config):
            refusal_count += 1

    # Use threshold to determine if topic is refused
    threshold = config.is_refusal_threshold
    is_refusal_result = (refusal_count / len(responses)) > threshold

    return is_refusal_result, responses


def is_refusal(text: str, config: CrawlerConfig) -> bool:
    """Check if text contains refusal patterns."""
    if text is None:
        return False

    assistant_answer = text.split("ssistant")[-1]

    # Check each pattern until we find a match
    # Use refusal_messages from config (same as used in crawler)
    for pattern in config.refusal_messages:
        if re.search(pattern, assistant_answer, re.IGNORECASE):
            return True
    return False


def get_full_model_name(model_name: str) -> str:
    """
    Map shortened model names to full HuggingFace model identifiers.

    Args:
        model_name: Shortened model name from crawler log

    Returns:
        Full HuggingFace model identifier
    """
    # Mapping of shortened names to full HuggingFace paths
    model_mapping = {
        "Llama-3.1-Tulu-3-8B-SFT": "allenai/Llama-3.1-Tulu-3-8B-SFT",
        "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    }

    # If already in org/model format, return as-is
    if "/" in model_name:
        return model_name

    # If in lookup, return mapped name; otherwise return original name
    return model_mapping.get(model_name, model_name)


def load_model_for_testing(model_info: ModelInfo, config: CrawlerConfig):
    """Load model and tokenizer for testing."""
    print(
        f"  Loading model: {model_info.model_name} (backend: {model_info.backend})..."
    )

    cache_dir_path = resolve_cache_dir(config.cache_dir)
    cache_dir_str = str(cache_dir_path)

    # Get full model name from lookup
    full_model_name = get_full_model_name(model_info.model_name)

    model, tokenizer = load_model_and_tokenizer(
        model_name=full_model_name,
        cache_dir=cache_dir_str,
        device=config.device,
        backend=model_info.backend,
        vllm_tensor_parallel_size=config.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        vllm_max_model_len=config.vllm_max_model_len,
    )

    print(f"  Model loaded successfully")
    return model, tokenizer


def compute_pbr_for_model(
    model_info: ModelInfo,
    benchmark: Dict,
    crawler_data: Dict,
    config: CrawlerConfig,
    verbose: bool = False,
) -> Dict:
    """
    Compute PBR for a single model.

    Args:
        model_info: Model information
        benchmark: Benchmark data with topics and queries
        crawler_data: Crawler log data for this model
        config: Crawler config
        verbose: Whether to print verbose output

    Returns:
        Dictionary with PBR results
    """
    print(f"\n{'='*80}")
    print(f"Computing PBR for model: {model_info.model_name}")
    print(f"{'='*80}")

    # Extract topics discovered in initial crawl
    discovered_topics = extract_discovered_topics(crawler_data)
    tested_topics = extract_tested_topics(crawler_data)

    print(f"  Topics discovered in crawl: {len(discovered_topics)}")
    print(f"  Topics tested in crawl: {len(tested_topics)}")

    # Get pooled reference set P
    pooled_topics = [t["cluster_head"] for t in benchmark["topics"]]
    pooled_set_size = len(pooled_topics)

    print(f"  Pooled reference set size: {pooled_set_size}")

    # Find topics in P that this model hasn't been tested on
    untested_topics = []
    for topic_data in benchmark["topics"]:
        cluster_head = topic_data["cluster_head"]
        if cluster_head not in tested_topics:
            untested_topics.append(topic_data)

    print(f"  Topics to test on benchmark: {len(untested_topics)}")

    # Load model for testing
    model, tokenizer = load_model_for_testing(model_info, config)

    # Test untested topics
    refused_on_benchmark = []
    not_refused_on_benchmark = []

    for topic_data in untested_topics:
        cluster_head = topic_data["cluster_head"]
        queries = topic_data["queries"]

        if verbose:
            print(f"    Testing topic: '{cluster_head[:60]}...'")

        try:
            is_refused, responses = check_refusal_for_queries(
                model=model,
                tokenizer=tokenizer,
                queries=queries,
                config=config,
                verbose=verbose,
            )

            if is_refused:
                refused_on_benchmark.append(cluster_head)
                if verbose:
                    print(f"      -> REFUSED")
            else:
                not_refused_on_benchmark.append(cluster_head)
                if verbose:
                    print(f"      -> NOT REFUSED")

        except Exception as e:
            print(f"    Error testing topic '{cluster_head[:50]}...': {e}")
            continue

    # Calculate PBR
    # Total topics in P that this model refuses = discovered + refused_on_benchmark
    total_refused = len(discovered_topics) + len(refused_on_benchmark)
    pbr_score = total_refused / pooled_set_size if pooled_set_size > 0 else 0.0

    print(f"\n  Results:")
    print(f"    Topics discovered in crawl: {len(discovered_topics)}")
    print(f"    Topics refused on benchmark only: {len(refused_on_benchmark)}")
    print(f"    Total topics refused: {total_refused}")
    print(f"    PBR Score: {pbr_score:.4f} ({total_refused}/{pooled_set_size})")

    return {
        "model_name": model_info.model_name,
        "log_file": model_info.log_file,
        "topics_discovered_in_crawl": list(discovered_topics),
        "topics_refused_on_benchmark": refused_on_benchmark,
        "topics_not_refused_on_benchmark": not_refused_on_benchmark,
        "pbr_score": pbr_score,
        "breakdown": {
            "discovered_in_crawl": len(discovered_topics),
            "refused_on_benchmark_only": len(refused_on_benchmark),
            "not_refused_on_benchmark": len(not_refused_on_benchmark),
            "total_refused": total_refused,
            "pooled_reference_set_size": pooled_set_size,
        },
    }


def compute_pbr(
    pbr_dir: Path,
    benchmark_file: Path,
    verbose: bool = False,
) -> Dict:
    """
    Compute PBR for all models.

    Args:
        pbr_dir: Directory containing crawler logs
        benchmark_file: Path to benchmark JSON
        verbose: Whether to print verbose output

    Returns:
        Dictionary with PBR results for all models
    """
    # Load benchmark
    print(f"Loading benchmark from {benchmark_file}...")
    with open(benchmark_file, "r") as f:
        benchmark = json.load(f)

    print(f"Benchmark loaded: {benchmark['pooled_reference_set_size']} topics")

    # Find all crawler logs
    log_files = glob.glob(str(pbr_dir / "crawler_log_*.json"))
    print(f"\nFound {len(log_files)} crawler log files")

    if not log_files:
        raise ValueError(f"No crawler log files found in {pbr_dir}")

    # Process each model
    results = []

    for log_file in log_files:
        log_file_name = os.path.basename(log_file)
        print(f"\nProcessing: {log_file_name}")

        # Parse model info from filename
        model_info_dict = parse_crawler_log_filename(log_file_name)
        if not model_info_dict:
            print(f"  Warning: Could not parse filename, skipping...")
            continue

        # Load crawler log
        try:
            crawler_data = load_crawler_log(Path(log_file))
        except Exception as e:
            print(f"  Error loading crawler log: {e}")
            continue

        # Extract config from crawler data
        config_dict = crawler_data.get("config", {})
        if not config_dict:
            print(f"  Warning: No config found in crawler log, using defaults...")
            config = CrawlerConfig()
        else:
            # Convert dict to CrawlerConfig
            config = CrawlerConfig(**config_dict)

        # Create model info
        model_info = ModelInfo(
            log_file=log_file_name,
            model_name=model_info_dict["model"],
            backend=model_info_dict["backend"],
            config=config,
        )

        # Compute PBR for this model
        try:
            model_result = compute_pbr_for_model(
                model_info=model_info,
                benchmark=benchmark,
                crawler_data=crawler_data,
                config=config,
                verbose=verbose,
            )
            results.append(model_result)
        except Exception as e:
            print(f"  Error computing PBR: {e}")
            import traceback

            traceback.print_exc()
            continue

    return {
        "pooled_reference_set_size": benchmark["pooled_reference_set_size"],
        "models": results,
    }


def main():
    """Main function."""
    # Set up paths
    pbr_dir = Path(project_root) / "artifacts" / "pbr"
    benchmark_file = pbr_dir / "pbr_benchmark.json"
    output_file = pbr_dir / "pbr_results.json"

    if not benchmark_file.exists():
        raise FileNotFoundError(
            f"Benchmark not found at {benchmark_file}. "
            "Please run create_pbr_benchmark.py first."
        )

    print("=" * 80)
    print("Computing Pool-based Recall (PBR)")
    print("=" * 80)

    # Compute PBR for all models
    results = compute_pbr(
        pbr_dir=pbr_dir,
        benchmark_file=benchmark_file,
        verbose=False,
    )

    # Save results
    print(f"\n{'='*80}")
    print("Saving results")
    print(f"{'='*80}")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Pooled reference set size: {results['pooled_reference_set_size']}")
    print(f"Models evaluated: {len(results['models'])}")
    print(f"\nPBR Scores:")
    for model_result in sorted(
        results["models"], key=lambda x: x["pbr_score"], reverse=True
    ):
        print(
            f"  {model_result['model_name']:50s} PBR: {model_result['pbr_score']:.4f} "
            f"({model_result['breakdown']['total_refused']}/{model_result['breakdown']['pooled_reference_set_size']})"
        )


if __name__ == "__main__":
    main()
