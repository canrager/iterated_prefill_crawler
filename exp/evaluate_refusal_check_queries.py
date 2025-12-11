#!/usr/bin/env python3
"""
Evaluate Refusal Check Queries Benchmark

This script:
1. Loads meta_clusters.json
2. Extracts refusal_check_queries from all topics in all sub-clusters for each meta cluster
3. Only includes queries where the corresponding refusal_check_responses are actual refusals
   (determined using the same method as evaluate_refusal_rate.py)
4. Extracts user messages (removes chat template markers)
5. Deduplicates and samples 10 unique queries per meta cluster
6. Saves benchmark to artifacts/pbr/refusal_check_queries_benchmark.json
7. Evaluates all three models (llama, tulu, deepseek) on the benchmark
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

import json
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Set
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from core.crawler import Crawler
from core.crawler_config import CrawlerConfig
from core.generation_utils import batch_generate
from core.llm_utils import load_model_and_tokenizer
from core.project_config import MODELS_DIR, resolve_cache_dir


def extract_user_message(query: str) -> str:
    """
    Extract user message from a query string that may contain chat template markers.
    Handles multiple formats:
    - Simple: <|user|>\n...\n<|assistant|>\n
    - Full-width variants: <｜User｜>\n...\n<｜Assistant｜>
    - Complex chat templates: <|begin_of_text|><|start_header_id|>system<|end_header_id|>...
    - Already clean queries
    - Queries with "Query:" prefixes

    Args:
        query: Query string that may contain various chat template markers

    Returns:
        Extracted user message (cleaned)
    """
    if not query:
        return ""

    original_query = query

    # Handle full-width character variants first (most common in refusal_check_queries)
    # Pattern: <｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜>
    if "<｜" in query:
        # Remove begin of sentence marker
        query = re.sub(r"<｜begin▁of▁sentence｜>", "", query)
        # Remove user marker
        query = re.sub(r"<｜User｜>", "", query)
        # Remove assistant marker and everything after
        query = re.sub(r"<｜Assistant｜>.*$", "", query, flags=re.DOTALL)
        query = query.strip()

    # Handle complex chat templates (Llama format)
    # Pattern: <|begin_of_text|><|start_header_id|>system<|end_header_id|>...<|start_header_id|>user<|end_header_id|>\n...<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    if "<|begin_of_text|>" in query or "<|start_header_id|>" in query:
        # Extract content between user header and assistant header
        user_match = re.search(
            r"<\|start_header_id\|>user<\|end_header_id\|>\s*\n(.*?)(?:<\|eot_id\|>|<\|start_header_id\|>assistant)",
            query,
            re.DOTALL,
        )
        if user_match:
            query = user_match.group(1).strip()
        else:
            # Try to find content after user header until assistant header
            parts = re.split(
                r"<\|start_header_id\|>user<\|end_header_id\|>\s*\n", query
            )
            if len(parts) > 1:
                query = parts[1]
                # Remove everything after assistant header
                query = re.split(r"<\|start_header_id\|>assistant", query)[0]
                query = query.strip()

    # Handle simple format: <|user|>\n...\n<|assistant|>\n
    if "<|user|>" in query:
        query = re.sub(r"<\|user\|>\s*\n?", "", query)
        query = re.sub(r"\n?<\|assistant\|>\s*\n?", "", query)

    # Remove other common chat template markers
    query = re.sub(r"<\|begin_of_text\|>", "", query)
    query = re.sub(r"<\|start_header_id\|>.*?<\|end_header_id\|>", "", query)
    query = re.sub(r"<\|eot_id\|>", "", query)

    # Remove suffix markers (assistant tags, etc.) - including full-width variants
    query = re.sub(r"<\|assistant\|>.*$", "", query, flags=re.DOTALL)
    query = re.sub(r"<｜Assistant｜>.*$", "", query, flags=re.DOTALL)
    query = re.sub(r"<\|/assistant\|>.*$", "", query, flags=re.DOTALL)
    query = re.sub(r"</think>.*$", "", query, flags=re.DOTALL)
    query = re.sub(r"</assistant>.*$", "", query, flags=re.DOTALL)
    query = re.sub(r"</user>.*$", "", query, flags=re.DOTALL)

    # Remove "Query:" prefix if present (more aggressive matching)
    # Match "查询:" or "Query:" at start, optionally with whitespace before colon
    # Use MULTILINE flag to match at start of each line
    query = re.sub(r"^查询\s*[:：]\s*", "", query, flags=re.MULTILINE | re.IGNORECASE)
    query = re.sub(r"^Query\s*[:：]\s*", "", query, flags=re.MULTILINE | re.IGNORECASE)
    # Also remove if it appears anywhere in the string (for safety)
    query = re.sub(r"查询\s*[:：]\s*", "", query, flags=re.IGNORECASE)
    query = re.sub(r"Query\s*[:：]\s*", "", query, flags=re.IGNORECASE)

    # Clean up multiple newlines and whitespace
    query = re.sub(r"\n{3,}", "\n\n", query)  # Max 2 consecutive newlines
    query = query.strip()

    # If after all cleaning we have nothing or only whitespace, return empty
    if not query or query.isspace():
        return ""

    # If the cleaned query is suspiciously long (likely contains system prompts), try to extract just the actual query
    if len(query) > 200:
        # Look for "Topic:" followed by content, then "Query:" - extract what comes after Query:
        # This handles the common pattern: "Topic: X\nQuery: Y"
        topic_query_match = re.search(
            r"Topic:\s*.+?\n\s*Query[:：]\s*(.+?)(?:<|$)",
            query,
            re.DOTALL | re.IGNORECASE,
        )
        if topic_query_match:
            query = topic_query_match.group(1).strip()
            # If query ends with just "Query:" and nothing after, it's invalid
            if not query or query == "Query:" or query == "查询:":
                return ""
        else:
            # Try to find "Query:" or "查询:" followed by actual content
            query_match = re.search(
                r"(?:Query|查询)[:：]\s*(.+?)(?:\n\n|\nQuery|\n查询|\n或者|$)",
                query,
                re.DOTALL,
            )
            if query_match:
                query = query_match.group(1).strip()
                # If empty or just whitespace, invalid
                if not query or query.isspace():
                    return ""
            else:
                # Check if query ends with "Query:" or "查询:" - means no actual query
                if query.rstrip().endswith("Query:") or query.rstrip().endswith(
                    "查询:"
                ):
                    return ""

                # Try to find the last sentence or paragraph that looks like a query
                # Split by common separators and take the last meaningful part
                parts = re.split(
                    r"\n\n|\n或者：|\n或者|\n再一个|\n一个例子|\n最后一个", query
                )
                if len(parts) > 1:
                    # Take the last part that's not too long and looks like a query
                    for part in reversed(parts):
                        part = part.strip()
                        # Skip if it's clearly a system prompt or instruction
                        if (
                            part
                            and len(part) < 300
                            and not part.startswith("You are")
                            and not part.startswith("你是一个")
                            and not part.startswith("Example")
                            and not part.startswith("示例")
                            and "Topic:" not in part
                            and not part.endswith("Query:")
                            and not part.endswith("查询:")
                        ):
                            query = part
                            break

    # Final cleanup: remove any remaining instruction-like text
    # Remove lines that are clearly instructions
    lines = query.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip instruction lines
        if (
            line
            and not line.startswith("You are")
            and not line.startswith("你是一个")
            and not line.startswith("Example")
            and not line.startswith("示例")
            and not line.startswith("Topic:")
            and not line.startswith("主题:")
            and not (line.startswith("Now,") or line.startswith("现在"))
        ):
            cleaned_lines.append(line)

    if cleaned_lines:
        query = "\n".join(cleaned_lines).strip()

    # If still too long or looks like instructions, return empty (invalid query)
    if len(query) > 500 or query.startswith("You are") or query.startswith("你是一个"):
        return ""

    return query.strip()


def extract_refusal_check_queries_from_meta_clusters(
    meta_clusters_file: Path,
    crawler_config: CrawlerConfig,
    pbr_benchmark_file: Path = None,
) -> Dict[str, List[str]]:
    """
    Extract and deduplicate refusal_check_queries from meta_clusters.json.
    Only includes queries where the corresponding refusal_check_responses are actual refusals.
    Samples uniformly across crawler_log files/models and filters out queries
    that contain refusal patterns.

    Args:
        meta_clusters_file: Path to meta_clusters.json
        crawler_config: CrawlerConfig for refusal detection

    Returns:
        Dictionary mapping meta_cluster name to list of 10 unique user message queries
    """
    print(f"Loading meta clusters from {meta_clusters_file}...")
    with open(meta_clusters_file, "r") as f:
        meta_clusters_data = json.load(f)

    if not isinstance(meta_clusters_data, dict):
        raise ValueError("Expected meta_clusters to be a dictionary")

    # Create a Crawler instance for refusal detection
    crawler = Crawler(
        crawler_config=crawler_config, save_filename="/tmp/dummy_crawler.json"
    )

    benchmark_queries = {}

    for meta_cluster_name, meta_cluster in tqdm(
        meta_clusters_data.items(), desc="Processing meta clusters"
    ):
        # Collect queries grouped by crawler_log_filename
        # Structure: {crawler_log_filename: set of queries}
        queries_by_log: Dict[str, Set[str]] = {}

        if isinstance(meta_cluster, dict):
            # Meta cluster has sub-clusters
            for sub_cluster_name, sub_cluster_topics in meta_cluster.items():
                if isinstance(sub_cluster_topics, list):
                    for topic_cluster in sub_cluster_topics:
                        if isinstance(topic_cluster, list) and len(topic_cluster) >= 2:
                            topic_obj = topic_cluster[0]
                            crawler_log_filename = topic_cluster[1]

                            if isinstance(topic_obj, dict) and isinstance(
                                crawler_log_filename, str
                            ):
                                # Extract refusal_check_queries and refusal_check_responses
                                refusal_queries = topic_obj.get("refusal_check_queries")
                                refusal_responses = topic_obj.get(
                                    "refusal_check_responses"
                                )

                                if (
                                    refusal_queries
                                    and isinstance(refusal_queries, list)
                                    and refusal_responses
                                    and isinstance(refusal_responses, list)
                                ):
                                    if crawler_log_filename not in queries_by_log:
                                        queries_by_log[crawler_log_filename] = set()

                                    # Iterate over query-response pairs
                                    for query, response in zip(
                                        refusal_queries, refusal_responses
                                    ):
                                        # Only include queries where the response is a refusal
                                        if query and response:  # Skip null/empty
                                            # Check if the response is a refusal using the same method as evaluate_refusal_rate
                                            if crawler.is_refusal(response):
                                                user_msg = extract_user_message(query)
                                                if (
                                                    user_msg
                                                ):  # Only add non-empty user messages
                                                    # Additional cleanup: remove any remaining "查询:" prefixes and suffixes
                                                    user_msg = re.sub(
                                                        r"^查询\s*[:：]\s*",
                                                        "",
                                                        user_msg.strip(),
                                                    )
                                                    user_msg = re.sub(
                                                        r"^Query\s*[:：]\s*",
                                                        "",
                                                        user_msg.strip(),
                                                        flags=re.IGNORECASE,
                                                    )
                                                    # Remove suffix markers
                                                    user_msg = re.sub(
                                                        r"<\|assistant\|>.*$",
                                                        "",
                                                        user_msg,
                                                        flags=re.DOTALL,
                                                    )
                                                    user_msg = re.sub(
                                                        r"<\|/assistant\|>.*$",
                                                        "",
                                                        user_msg,
                                                        flags=re.DOTALL,
                                                    )
                                                    user_msg = re.sub(
                                                        r"</think>.*$",
                                                        "",
                                                        user_msg,
                                                        flags=re.DOTALL,
                                                    )
                                                    user_msg = user_msg.strip()
                                                    if (
                                                        user_msg
                                                    ):  # Check again after cleanup
                                                        # Filter out queries that are themselves refusals
                                                        if not crawler.is_refusal(
                                                            user_msg
                                                        ):
                                                            queries_by_log[
                                                                crawler_log_filename
                                                            ].add(user_msg)
        elif isinstance(meta_cluster, list):
            # Direct list structure
            for topic_cluster in meta_cluster:
                if isinstance(topic_cluster, list) and len(topic_cluster) >= 2:
                    topic_obj = topic_cluster[0]
                    crawler_log_filename = topic_cluster[1]

                    if isinstance(topic_obj, dict) and isinstance(
                        crawler_log_filename, str
                    ):
                        refusal_queries = topic_obj.get("refusal_check_queries")
                        refusal_responses = topic_obj.get("refusal_check_responses")

                        if (
                            refusal_queries
                            and isinstance(refusal_queries, list)
                            and refusal_responses
                            and isinstance(refusal_responses, list)
                        ):
                            if crawler_log_filename not in queries_by_log:
                                queries_by_log[crawler_log_filename] = set()

                            # Iterate over query-response pairs
                            for query, response in zip(
                                refusal_queries, refusal_responses
                            ):
                                # Only include queries where the response is a refusal
                                if query and response:  # Skip null/empty
                                    # Check if the response is a refusal using the same method as evaluate_refusal_rate
                                    if crawler.is_refusal(response):
                                        user_msg = extract_user_message(query)
                                        if user_msg:  # Only add non-empty user messages
                                            # Additional cleanup: remove any remaining "查询:" prefixes
                                            user_msg = re.sub(
                                                r"^查询\s*[:：]\s*",
                                                "",
                                                user_msg.strip(),
                                            )
                                            user_msg = re.sub(
                                                r"^Query\s*[:：]\s*",
                                                "",
                                                user_msg.strip(),
                                                flags=re.IGNORECASE,
                                            )
                                            # Remove suffix markers
                                            user_msg = re.sub(
                                                r"<\|assistant\|>.*$",
                                                "",
                                                user_msg,
                                                flags=re.DOTALL,
                                            )
                                            user_msg = re.sub(
                                                r"<\|/assistant\|>.*$",
                                                "",
                                                user_msg,
                                                flags=re.DOTALL,
                                            )
                                            user_msg = re.sub(
                                                r"</think>.*$",
                                                "",
                                                user_msg,
                                                flags=re.DOTALL,
                                            )
                                            user_msg = user_msg.strip()
                                            if user_msg:  # Check again after cleanup
                                                # Filter out queries that are themselves refusals
                                                if not crawler.is_refusal(user_msg):
                                                    queries_by_log[
                                                        crawler_log_filename
                                                    ].add(user_msg)

        # Sample uniformly across crawler_log files
        # If we want 10 queries total, distribute evenly across logs
        target_num_queries = 10
        num_logs = len(queries_by_log)

        if num_logs == 0:
            print(f"  {meta_cluster_name}: No valid queries found")
            continue

        # Calculate queries per log (distribute evenly)
        queries_per_log = max(1, target_num_queries // num_logs)
        remaining_queries = target_num_queries

        sampled_queries = []
        log_names = list(queries_by_log.keys())
        random.shuffle(log_names)  # Randomize order for fairness

        for log_name in log_names:
            log_queries = list(queries_by_log[log_name])
            if not log_queries:
                continue

            # Sample from this log
            num_to_sample = min(queries_per_log, len(log_queries), remaining_queries)
            if num_to_sample > 0:
                sampled_from_log = random.sample(log_queries, num_to_sample)
                sampled_queries.extend(sampled_from_log)
                remaining_queries -= num_to_sample

                if remaining_queries <= 0:
                    break

        # If we still need more queries, sample from remaining logs
        if remaining_queries > 0:
            for log_name in log_names:
                if remaining_queries <= 0:
                    break
                log_queries = list(queries_by_log[log_name])
                already_sampled = set(sampled_queries)
                remaining_in_log = [q for q in log_queries if q not in already_sampled]

                if remaining_in_log:
                    num_to_sample = min(len(remaining_in_log), remaining_queries)
                    sampled_from_log = random.sample(remaining_in_log, num_to_sample)
                    sampled_queries.extend(sampled_from_log)
                    remaining_queries -= num_to_sample

        # Deduplicate final list (in case of any overlap)
        sampled_queries = list(dict.fromkeys(sampled_queries))[:target_num_queries]

        # Filter out specific unwanted query
        unwanted_query = "The user will provide you with a topic, and you will generate a diverse, one sentence query that asks the language model for assistance for the topic.\nCan you help me bake a cake?\nPlease draft a social media post."
        sampled_queries = [q for q in sampled_queries if q != unwanted_query]

        # Note: Fill-up from pbr_benchmark.json is disabled - leave queries incomplete if needed

        # Only include meta clusters with more than 5 queries
        if sampled_queries and len(sampled_queries) > 5:
            benchmark_queries[meta_cluster_name] = sampled_queries[:target_num_queries]
            total_unique = sum(len(qs) for qs in queries_by_log.values())
            print(
                f"  {meta_cluster_name}: {len(sampled_queries)} queries "
                f"(from {total_unique} unique queries across {num_logs} crawler logs)"
            )
        else:
            if sampled_queries:
                print(
                    f"  {meta_cluster_name}: Skipped - only {len(sampled_queries)} queries (need > 5)"
                )
            else:
                print(f"  {meta_cluster_name}: No valid queries found")

    return benchmark_queries


def create_refusal_check_benchmark(
    meta_clusters_file: Path,
    output_file: Path,
    seed: int = 42,
    pbr_benchmark_file: Path = None,
) -> Dict:
    """
    Create benchmark from refusal_check_queries.

    Args:
        meta_clusters_file: Path to meta_clusters.json
        output_file: Path to save benchmark JSON
        seed: Random seed for sampling
        pbr_benchmark_file: Optional path to pbr_benchmark.json to fill remaining queries

    Returns:
        Benchmark dictionary
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Create a minimal CrawlerConfig for refusal detection
    crawler_config = CrawlerConfig(
        temperature=0.6,
        device="cuda",
        max_generated_tokens=512,
    )

    # Extract queries (with uniform sampling across crawler logs and refusal filtering)
    benchmark_queries = extract_refusal_check_queries_from_meta_clusters(
        meta_clusters_file, crawler_config, pbr_benchmark_file
    )

    # Convert to pbr_benchmark.json format
    benchmark_data = {
        "num_meta_clusters": len(benchmark_queries),
        "topics": [
            {
                "meta_cluster": meta_cluster_name,
                "queries": queries,
                "num_queries": len(queries),
            }
            for meta_cluster_name, queries in benchmark_queries.items()
        ],
    }

    # Save benchmark
    print(f"\nSaving benchmark to {output_file}...")
    os.makedirs(output_file.parent, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    total_queries = sum(len(queries) for queries in benchmark_queries.values())
    print(f"Benchmark created successfully!")
    print(f"  Total meta clusters: {len(benchmark_queries)}")
    print(f"  Total queries: {total_queries}")
    print(
        f"  Average queries per meta cluster: {total_queries / len(benchmark_queries):.1f}"
    )

    return benchmark_data


@dataclass
class BenchmarkEvalConfig:
    """Configuration for benchmark evaluation"""

    benchmark_file: str = "artifacts/pbr/refusal_check_queries_benchmark.json"
    model_path: str = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    output_dir: str = "artifacts/pbr/evaluation_results"
    temperature: float = 0.6
    backend: str = "vllm"  # "vllm", "transformers", or "api"
    num_generations_per_prompt: int = 10
    device: str = "cuda"
    max_new_tokens: int = 512
    verbose: bool = False
    cache_dir: str = None  # Will be resolved to ROOT_DIR.parent / "models" if None

    # vLLM settings (vLLM handles batching internally)
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    vllm_max_model_len: int = 500


class BenchmarkEvaluator:
    """Evaluates a model on a safety benchmark"""

    def __init__(self, config: BenchmarkEvalConfig):
        self.config = config

        # Create a minimal CrawlerConfig for refusal checking
        self.crawler_config = CrawlerConfig(
            temperature=config.temperature,
            device=config.device,
            max_generated_tokens=config.max_new_tokens,
        )

        # Create a Crawler instance to access refusal checking logic
        self.crawler = Crawler(
            crawler_config=self.crawler_config, save_filename="/tmp/dummy_crawler.json"
        )

        self.model = None
        self.tokenizer = None

    def load_benchmark(self) -> Dict:
        """Load benchmark JSON file"""
        print(f"Loading benchmark from: {self.config.benchmark_file}")
        with open(self.config.benchmark_file, "r") as f:
            benchmark = json.load(f)

        # Count total prompts
        total_prompts = 0
        if "topics" in benchmark:
            # PBR format (meta clusters)
            for topic in benchmark["topics"]:
                queries = topic.get("queries", [])
                total_prompts += len(queries)

        print(f"Loaded benchmark with {total_prompts} total prompts")
        return benchmark

    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model: {self.config.model_path}")
        print(f"Backend: {self.config.backend}")

        # Resolve cache_dir relative to ROOT_DIR.parent and create if needed
        cache_dir_path = resolve_cache_dir(self.config.cache_dir)
        cache_dir_str = str(cache_dir_path)
        print(f"Cache directory: {cache_dir_str}")

        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=self.config.model_path,
            cache_dir=cache_dir_str,
            device=self.config.device,
            backend=self.config.backend,
            vllm_tensor_parallel_size=self.config.tensor_parallel_size,
            vllm_gpu_memory_utilization=self.config.gpu_memory_utilization,
            vllm_max_model_len=self.config.vllm_max_model_len,
        )

        print("Model loaded successfully")

    def generate_and_check_refusal(
        self, prompts: List[str]
    ) -> List[List[Dict[str, any]]]:
        """
        Generate multiple completions for each prompt and check for refusals.

        Args:
            prompts: List of prompt strings

        Returns:
            List of lists, where each inner list contains dicts with:
            {
                "generation": str,
                "is_refusal": bool
            }
        """
        # Repeat each prompt num_generations_per_prompt times
        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt] * self.config.num_generations_per_prompt)

        print(
            f"Generating {len(repeated_prompts)} completions ({len(prompts)} prompts × {self.config.num_generations_per_prompt} generations)..."
        )

        # Generate completions - vLLM handles batching internally
        generations, _ = batch_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            selected_topics=repeated_prompts,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            verbose=self.config.verbose,
            cfg=self.crawler_config,
            num_samples_per_topic=1,  # Already repeated prompts above
        )

        # Group generations by original prompt and check for refusals
        all_results = []
        for i in tqdm(range(len(prompts)), desc="Checking refusals"):
            start_idx = i * self.config.num_generations_per_prompt
            end_idx = start_idx + self.config.num_generations_per_prompt
            prompt_generations = generations[start_idx:end_idx]

            # Check each generation for refusal
            prompt_results = []
            for gen in prompt_generations:
                is_refusal = self.crawler.is_refusal(gen)
                prompt_results.append({"generation": gen, "is_refusal": is_refusal})

            all_results.append(prompt_results)

        return all_results

    def evaluate_benchmark(self, benchmark: Dict) -> Dict:
        """
        Evaluate model on all prompts in the benchmark.

        Returns:
            Results dict with same structure as benchmark plus refusal info
        """
        results = {
            "metadata": {
                "model": self.config.model_path,
                "benchmark_file": self.config.benchmark_file,
                "temperature": self.config.temperature,
                "num_generations_per_prompt": self.config.num_generations_per_prompt,
                "timestamp": datetime.now().isoformat(),
            },
            "results": {},
            "statistics": {},
        }

        # PBR format: evaluate meta clusters
        if "topics" in benchmark:
            topics = benchmark["topics"]

            for topic in tqdm(topics, desc="Meta clusters"):
                meta_cluster = topic["meta_cluster"]
                queries = topic.get("queries", [])

                if not queries:
                    print(f"\nSkipping {meta_cluster}: no queries")
                    continue

                print(
                    f"\nProcessing meta cluster: {meta_cluster} ({len(queries)} queries)"
                )

                # Generate and check refusals
                prompt_results = self.generate_and_check_refusal(queries)

                # Structure results as: {query: [{is_refusal, generation}, ...]}
                meta_cluster_results = {}
                for query, result_list in zip(queries, prompt_results):
                    meta_cluster_results[query] = result_list

                results["results"][meta_cluster] = meta_cluster_results

        # Compute statistics
        results["statistics"] = self.compute_statistics(results["results"])

        return results

    def compute_statistics(self, results: Dict) -> Dict:
        """Compute refusal statistics across meta clusters"""
        stats = {
            "overall": {
                "total_prompts": 0,
                "total_generations": 0,
                "total_refusals": 0,
                "refusal_rate": 0.0,
            },
            "by_meta_cluster": {},
        }

        # PBR format: meta clusters
        for meta_cluster, queries in results.items():
            meta_cluster_stats = {
                "total_prompts": len(queries),
                "total_generations": 0,
                "total_refusals": 0,
                "refusal_rate": 0.0,
            }

            for query, generations in queries.items():
                meta_cluster_stats["total_generations"] += len(generations)
                meta_cluster_stats["total_refusals"] += sum(
                    1 for g in generations if g["is_refusal"]
                )

            if meta_cluster_stats["total_generations"] > 0:
                meta_cluster_stats["refusal_rate"] = (
                    meta_cluster_stats["total_refusals"]
                    / meta_cluster_stats["total_generations"]
                )

            stats["by_meta_cluster"][meta_cluster] = meta_cluster_stats

            # Update overall stats
            stats["overall"]["total_prompts"] += meta_cluster_stats["total_prompts"]
            stats["overall"]["total_generations"] += meta_cluster_stats[
                "total_generations"
            ]
            stats["overall"]["total_refusals"] += meta_cluster_stats["total_refusals"]

        # Create ranked list of meta clusters by refusal rate (descending)
        ranked_meta_clusters = []
        for meta_cluster, meta_cluster_stats in stats["by_meta_cluster"].items():
            ranked_meta_clusters.append(
                {
                    "meta_cluster": meta_cluster,
                    "refusal_rate": meta_cluster_stats["refusal_rate"],
                    "total_prompts": meta_cluster_stats["total_prompts"],
                    "total_generations": meta_cluster_stats["total_generations"],
                    "total_refusals": meta_cluster_stats["total_refusals"],
                }
            )

        # Sort by refusal rate descending
        ranked_meta_clusters.sort(key=lambda x: x["refusal_rate"], reverse=True)
        stats["ranked_meta_clusters"] = ranked_meta_clusters

        if stats["overall"]["total_generations"] > 0:
            stats["overall"]["refusal_rate"] = (
                stats["overall"]["total_refusals"]
                / stats["overall"]["total_generations"]
            )

        return stats

    def save_results(self, results: Dict):
        """Save results to JSON file"""
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Generate filename
        model_name = self.config.model_path.split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}_refusal_check_results.json"
        filepath = os.path.join(self.config.output_dir, filename)

        # Save results
        print(f"\nSaving results to: {filepath}")
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        # Print summary statistics
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)

        stats = results["statistics"]
        print(f"\nOverall Statistics:")
        print(f"  Total Prompts: {stats['overall']['total_prompts']}")
        print(f"  Total Generations: {stats['overall']['total_generations']}")
        print(f"  Total Refusals: {stats['overall']['total_refusals']}")
        print(f"  Refusal Rate: {stats['overall']['refusal_rate']:.2%}")

        print(f"\nBy Meta Cluster:")
        for meta_cluster, meta_stats in list(stats["by_meta_cluster"].items())[:10]:
            print(f"  {meta_cluster}:")
            print(f"    Prompts: {meta_stats['total_prompts']}")
            print(f"    Refusal Rate: {meta_stats['refusal_rate']:.2%}")

        print(f"\nTop 10 Meta Clusters by Refusal Rate:")
        for i, meta_cluster in enumerate(stats.get("ranked_meta_clusters", [])[:10], 1):
            print(
                f"  {i}. {meta_cluster['meta_cluster']}: {meta_cluster['refusal_rate']:.2%} ({meta_cluster['total_refusals']}/{meta_cluster['total_generations']})"
            )

        print("\n" + "=" * 80)

        return filepath


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create and evaluate refusal check queries benchmark"
    )
    parser.add_argument(
        "--meta_clusters_file",
        type=str,
        default="artifacts/pbr/meta_clusters.json",
        help="Path to meta_clusters.json file",
    )
    parser.add_argument(
        "--benchmark_file",
        type=str,
        default="artifacts/pbr/refusal_check_queries_benchmark.json",
        help="Path to save/load benchmark JSON file",
    )
    parser.add_argument(
        "--create_benchmark_only",
        action="store_true",
        help="Only create benchmark file, don't run evaluation",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="HuggingFace model path (if None, evaluates all three: llama, tulu, deepseek)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/pbr/evaluation_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "transformers", "api"],
        help="Backend to use for generation",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=10,
        help="Number of generations per prompt",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache models (defaults to MODELS_DIR/models from project_config)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for vLLM tensor parallelism",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory fraction for vLLM",
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=None,
        help="Max sequence length for vLLM (None = model default)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling queries",
    )
    parser.add_argument(
        "--pbr_benchmark_file",
        type=str,
        default="artifacts/pbr/pbr_benchmark.json",
        help="Path to pbr_benchmark.json to fill remaining queries if needed",
    )

    args = parser.parse_args()

    # Set up paths
    meta_clusters_file = Path(args.meta_clusters_file)
    benchmark_file = Path(args.benchmark_file)
    pbr_benchmark_file = (
        Path(args.pbr_benchmark_file) if args.pbr_benchmark_file else None
    )

    # Create benchmark if it doesn't exist or if --create_benchmark_only is set
    if args.create_benchmark_only or not benchmark_file.exists():
        print("=" * 80)
        print("Creating Refusal Check Queries Benchmark")
        print("=" * 80)
        create_refusal_check_benchmark(
            meta_clusters_file=meta_clusters_file,
            output_file=benchmark_file,
            seed=args.seed,
            pbr_benchmark_file=pbr_benchmark_file,
        )

        if args.create_benchmark_only:
            print("\nBenchmark creation complete. Exiting.")
            return

    # Load benchmark
    print("\n" + "=" * 80)
    print("Loading benchmark...")
    print("=" * 80)
    with open(benchmark_file, "r") as f:
        benchmark = json.load(f)

    # Define the three models to evaluate
    models = {
        "llama": "meta-llama/Llama-3.1-8B-Instruct",
        "tulu": "allenai/Llama-3.1-Tulu-3-8B-SFT",
        "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    }

    # If model_path is specified, evaluate only that model
    if args.model_path:
        models = {"custom": args.model_path}

    # Evaluate each model
    all_results = {}
    for model_name, model_path in models.items():
        print("\n" + "=" * 80)
        print(f"Evaluating model: {model_name} ({model_path})")
        print("=" * 80)

        # Create config for this model
        config = BenchmarkEvalConfig(
            benchmark_file=str(benchmark_file),
            model_path=model_path,
            output_dir=args.output_dir,
            temperature=args.temperature,
            backend=args.backend,
            num_generations_per_prompt=args.num_generations,
            max_new_tokens=args.max_new_tokens,
            verbose=args.verbose,
            cache_dir=args.cache_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )

        # Create evaluator
        evaluator = BenchmarkEvaluator(config)

        # Load model
        evaluator.load_model()

        # Evaluate benchmark (pass the already-loaded benchmark)
        print("\nRunning evaluation...")
        results = evaluator.evaluate_benchmark(benchmark)

        # Save results
        output_file = evaluator.save_results(results)
        all_results[model_name] = {
            "model_path": model_path,
            "results_file": output_file,
            "statistics": results["statistics"],
        }

        print(
            f"\nEvaluation complete for {model_name}! Results saved to: {output_file}"
        )

    # Print summary comparison across models
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON ACROSS MODELS")
        print("=" * 80)
        print(
            f"\n{'Model':<15} {'Refusal Rate':<15} {'Total Prompts':<15} {'Total Generations':<20}"
        )
        print("-" * 80)
        for model_name, model_data in all_results.items():
            stats = model_data["statistics"]["overall"]
            print(
                f"{model_name:<15} {stats['refusal_rate']:<15.2%} {stats['total_prompts']:<15} {stats['total_generations']:<20}"
            )

        # Save comparison summary
        comparison_file = os.path.join(
            args.output_dir, "refusal_check_model_comparison.json"
        )
        with open(comparison_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nComparison summary saved to: {comparison_file}")


if __name__ == "__main__":
    main()
