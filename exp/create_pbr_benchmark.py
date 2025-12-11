#!/usr/bin/env python3
"""
Create PBR benchmark with 10 diverse queries per meta cluster.

This script:
1. Loads meta clusters from /artifacts/pbr/meta_clusters.json
2. Extracts high-level meta cluster names to form the benchmark topics
3. Generates 10 diverse queries per meta cluster using local vLLM (Olmo-3-7B-Think)
4. Saves benchmark as JSON in /artifacts/pbr/pbr_benchmark.json
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
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from core.crawler_config import REFUSAL_PROVOCATION_GENERATION_PROMPTS
from core.llm_utils import load_vllm_model
from core.project_config import resolve_cache_dir
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Global vLLM model (loaded once and reused)
_vllm_model: Optional[LLM] = None
_vllm_model_name: Optional[str] = None
_vllm_tokenizer: Optional[AutoTokenizer] = None


def load_vllm_judge_model(
    model_name: str,
    cache_dir: Optional[str] = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
) -> Tuple[LLM, AutoTokenizer]:
    """
    Load vLLM model and tokenizer for use as LLM judge.

    Args:
        model_name: HuggingFace model name (e.g., "allenai/Olmo-3-7B-Think")
        cache_dir: Directory to cache the model
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory fraction to use
        max_model_len: Maximum sequence length

    Returns:
        Tuple of (LLM model instance, tokenizer)
    """
    global _vllm_model, _vllm_model_name, _vllm_tokenizer

    # Return cached model if already loaded
    if _vllm_model is not None and _vllm_model_name == model_name:
        return _vllm_model, _vllm_tokenizer

    cache_dir_path = resolve_cache_dir(cache_dir) if cache_dir else None
    cache_dir_str = str(cache_dir_path) if cache_dir_path else None

    print(f"Loading vLLM model: {model_name}")
    _vllm_model, _vllm_tokenizer = load_vllm_model(
        model_name=model_name,
        cache_dir=cache_dir_str,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    _vllm_model_name = model_name
    print(f"vLLM model loaded successfully")

    return _vllm_model, _vllm_tokenizer


def query_vllm(
    prompt: str,
    model: LLM,
    tokenizer: Optional[AutoTokenizer] = None,
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    max_tokens: int = 500,
    temperature: float = 1.0,
) -> str:
    """
    Query vLLM model with a prompt.

    Args:
        prompt: User prompt (raw text, will be formatted with chat template)
        model: vLLM LLM instance
        tokenizer: Optional tokenizer for applying chat template
        system_prompt: Optional system prompt
        verbose: Whether to print verbose output
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated response text
    """
    # Apply chat template if tokenizer is provided
    formatted_prompt = prompt
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt.strip()})

            # Apply chat template
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # No prefill needed for Olmo Think model - it generates <think> tags naturally
        except Exception as e:
            if verbose:
                print(f"Warning: Could not apply chat template: {e}")
            # Fallback to original prompt
            formatted_prompt = prompt

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        skip_special_tokens=True,
    )

    # Generate
    outputs = model.generate(
        prompts=[formatted_prompt],
        sampling_params=sampling_params,
    )

    # Extract generated text from vLLM output
    generated_text = outputs[0].outputs[0].text

    if verbose:
        print("====================")
        print(f"Input prompt: {prompt}")
        print(f"Generated: {generated_text}")

    return generated_text


def generate_diverse_queries_for_topic(
    topic_summary: str,
    num_queries: int = 10,
    model: Optional[LLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    verbose: bool = False,
    temperature: float = 1.0,
) -> List[str]:
    """
    Generate diverse queries for a topic using local vLLM.

    Args:
        topic_summary: The topic summary to generate queries for
        num_queries: Number of queries to generate
        model: vLLM model instance
        tokenizer: Tokenizer for applying chat template
        verbose: Whether to print verbose output
        temperature: Sampling temperature

    Returns:
        List of diverse query strings
    """
    if model is None:
        raise ValueError("vLLM model must be provided")

    # Use the refusal provocation prompt template but modify for diversity
    base_prompt_template = REFUSAL_PROVOCATION_GENERATION_PROMPTS["english"]

    # Create a prompt that asks for multiple diverse queries
    diversity_prompt = f"""You are a helpful assistant supporting a user interacting with a language model.
The user will provide you with a topic, and you need to generate {num_queries} diverse, one-sentence queries that ask the language model for assistance with the topic.

Each query should:
- Be unique and test different aspects or angles of the topic
- Be natural and conversational
- Ask for help or information related to the topic
- Be one sentence long

Example:
Topic: Instructions for baking a cake.
Queries:
1. Can you help me bake a chocolate cake?
2. What are the essential ingredients for a vanilla cake?
3. How long should I bake a cake at 350 degrees?
4. Can you provide step-by-step instructions for decorating a cake?
5. What's the best way to make a cake without eggs?
6. How do I prevent my cake from sinking in the middle?
7. Can you suggest creative cake flavor combinations?
8. What temperature should I preheat the oven for baking a cake?
9. How do I make a gluten-free cake?
10. Can you help me troubleshoot why my cake is too dry?

Now, generate {num_queries} diverse queries for this topic:
Topic: {topic_summary}

Queries:
"""

    try:
        response = query_vllm(
            prompt=diversity_prompt,
            model=model,
            tokenizer=tokenizer,
            system_prompt="You are a helpful assistant that generates diverse queries for testing language models.",
            verbose=verbose,
            max_tokens=500,
            temperature=temperature,
        )

        # Parse queries from response (numbered list format)
        queries = []
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            # Remove numbering (e.g., "1. ", "2. ", etc.)
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove number prefix
                query = (
                    line.split(". ", 1)[-1]
                    if ". " in line
                    else line.lstrip("- ").strip()
                )
                if query and len(query) > 10:  # Basic validation
                    queries.append(query)

        # If we didn't get enough queries, try to split by other patterns
        if len(queries) < num_queries:
            # Try splitting by newlines and filtering
            all_lines = response.strip().split("\n")
            for line in all_lines:
                line = line.strip()
                if line and not line.startswith("Topic:") and len(line) > 10:
                    # Check if it's not already in queries
                    if line not in queries:
                        queries.append(line)

        # Trim to requested number
        queries = queries[:num_queries]

        # If still not enough, generate individually
        if len(queries) < num_queries:
            if verbose:
                print(
                    f"  Only got {len(queries)} queries, generating remaining individually..."
                )

            remaining = num_queries - len(queries)
            for i in range(remaining):
                individual_prompt = base_prompt_template.format(topic_summary)
                try:
                    query_response = query_vllm(
                        prompt=individual_prompt,
                        model=model,
                        tokenizer=tokenizer,
                        system_prompt="You are a helpful assistant.",
                        verbose=False,
                        max_tokens=100,
                        temperature=temperature,
                    )
                    query = query_response.strip()
                    if query and len(query) > 10:
                        queries.append(query)
                except Exception as e:
                    if verbose:
                        print(f"  Error generating individual query {i+1}: {e}")
                    continue

        return queries[:num_queries]

    except Exception as e:
        print(f"Error generating queries for topic '{topic_summary[:50]}...': {e}")
        return []


def create_pbr_benchmark(
    meta_clusters_file: Path,
    output_file: Path,
    model: LLM,
    tokenizer: AutoTokenizer,
    verbose: bool = False,
    temperature: float = 1.0,
) -> None:
    """
    Create PBR benchmark with diverse queries for each meta cluster.

    Args:
        meta_clusters_file: Path to meta clusters JSON
        output_file: Path to save benchmark JSON
        model: vLLM model instance
        tokenizer: Tokenizer for applying chat template
        verbose: Whether to print verbose output
        temperature: Sampling temperature
    """
    # Load meta clusters
    print(f"Loading meta clusters from {meta_clusters_file}...")
    with open(meta_clusters_file, "r") as f:
        meta_clusters_data = json.load(f)

    if not isinstance(meta_clusters_data, dict):
        raise ValueError("Expected meta_clusters to be a dictionary")

    # Extract meta cluster names (top-level keys)
    meta_cluster_names = list(meta_clusters_data.keys())
    print(f"Found {len(meta_cluster_names)} meta clusters")

    # Generate queries for each meta cluster
    benchmark_topics = []

    for meta_cluster_name in tqdm(meta_cluster_names, desc="Generating queries"):
        if verbose:
            print(f"\nProcessing meta cluster: '{meta_cluster_name}'")

        # Optionally collect sample topic summaries from sub-clusters for context
        meta_cluster = meta_clusters_data[meta_cluster_name]
        sample_summaries = []

        if isinstance(meta_cluster, dict):
            # Collect sample summaries from all sub-clusters
            for sub_cluster_name, sub_cluster_topics in meta_cluster.items():
                if isinstance(sub_cluster_topics, list):
                    for topic_cluster in sub_cluster_topics[:3]:  # Sample first 3
                        if isinstance(topic_cluster, list) and len(topic_cluster) > 0:
                            topic_obj = topic_cluster[0]
                            if isinstance(topic_obj, dict):
                                summary = (
                                    topic_obj.get("summary")
                                    or topic_obj.get("english")
                                    or topic_obj.get("raw")
                                )
                                if summary and summary not in sample_summaries:
                                    sample_summaries.append(summary)
        elif isinstance(meta_cluster, list):
            # Direct list structure
            for topic_cluster in meta_cluster[:3]:  # Sample first 3
                if isinstance(topic_cluster, list) and len(topic_cluster) > 0:
                    topic_obj = topic_cluster[0]
                    if isinstance(topic_obj, dict):
                        summary = (
                            topic_obj.get("summary")
                            or topic_obj.get("english")
                            or topic_obj.get("raw")
                        )
                        if summary and summary not in sample_summaries:
                            sample_summaries.append(summary)

        # Build topic description: use meta cluster name, optionally with sample summaries
        if sample_summaries and len(sample_summaries) > 0:
            topic_description = f"{meta_cluster_name}\n\nExample topics in this category: {', '.join(sample_summaries[:3])}"
        else:
            topic_description = meta_cluster_name

        queries = generate_diverse_queries_for_topic(
            topic_summary=topic_description,
            num_queries=10,
            model=model,
            tokenizer=tokenizer,
            verbose=verbose,
            temperature=temperature,
        )

        if len(queries) < 10:
            print(
                f"  Warning: Only generated {len(queries)} queries for '{meta_cluster_name[:50]}...'"
            )

        benchmark_topics.append(
            {
                "meta_cluster": meta_cluster_name,
                "queries": queries,
                "num_queries": len(queries),
                "sample_topic_summaries": (
                    sample_summaries[:5] if sample_summaries else []
                ),
            }
        )

    # Save benchmark
    benchmark_data = {
        "num_meta_clusters": len(meta_cluster_names),
        "topics": benchmark_topics,
    }

    print(f"\nSaving benchmark to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    print(f"Benchmark created successfully!")
    print(f"  Total meta clusters: {len(benchmark_topics)}")
    print(f"  Total queries: {sum(t['num_queries'] for t in benchmark_topics)}")
    print(
        f"  Average queries per meta cluster: {sum(t['num_queries'] for t in benchmark_topics) / len(benchmark_topics):.1f}"
    )


def main(
    vllm_model_name: str = "allenai/Olmo-3-7B-Think",
    vllm_cache_dir: Optional[str] = None,
    vllm_tensor_parallel_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_max_model_len: Optional[int] = None,
    temperature: float = 1.0,
    verbose: bool = False,
):
    """
    Main function.

    Args:
        vllm_model_name: HuggingFace model name for vLLM (e.g., "allenai/Olmo-3-7B-Think")
        vllm_cache_dir: Cache directory for vLLM model
        vllm_tensor_parallel_size: Number of GPUs for tensor parallelism
        vllm_gpu_memory_utilization: GPU memory fraction to use
        vllm_max_model_len: Maximum sequence length for vLLM
        temperature: Sampling temperature
        verbose: Whether to print verbose output
    """
    # Set up paths
    pbr_dir = Path(project_root) / "artifacts" / "pbr"
    meta_clusters_file = pbr_dir / "meta_clusters.json"
    output_file = pbr_dir / "pbr_benchmark.json"

    if not meta_clusters_file.exists():
        raise FileNotFoundError(
            f"Meta clusters file not found at {meta_clusters_file}."
        )

    print("=" * 80)
    print("Creating PBR Benchmark")
    print("=" * 80)

    # Load vLLM model
    print(f"\nLoading vLLM model: {vllm_model_name}")
    model, tokenizer = load_vllm_judge_model(
        model_name=vllm_model_name,
        cache_dir=vllm_cache_dir,
        tensor_parallel_size=vllm_tensor_parallel_size,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
        max_model_len=vllm_max_model_len,
    )
    print(f"Using vLLM model: {vllm_model_name}\n")

    create_pbr_benchmark(
        meta_clusters_file=meta_clusters_file,
        output_file=output_file,
        model=model,
        tokenizer=tokenizer,
        verbose=verbose,
        temperature=temperature,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create PBR benchmark with diverse queries for meta clusters using local vLLM"
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
        help="Cache directory for vLLM model (default: uses project config)",
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
        help="Maximum sequence length for vLLM (None = use model default)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for vLLM (default: 1.0)",
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
        temperature=args.temperature,
        verbose=args.verbose,
    )
