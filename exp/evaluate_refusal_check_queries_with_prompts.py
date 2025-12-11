#!/usr/bin/env python3
"""
Evaluate Refusal Check Queries Benchmark with Prompt Inputs and Outputs

Modified version that saves exact prompt inputs and generated outputs.
"""

import os

# Set environment variable to force spawn method before any imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import multiprocessing

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

import json
import sys
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.crawler import Crawler
from core.crawler_config import CrawlerConfig
from core.generation_utils import batch_generate
from core.llm_utils import load_model_and_tokenizer
from core.project_config import resolve_cache_dir


@dataclass
class BenchmarkEvalConfig:
    """Configuration for benchmark evaluation"""

    benchmark_file: str = "artifacts/pbr/refusal_check_topics.json"
    model_path: str = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    output_file: str = "artifacts/pbr/tulu_prompt_outputs.json"
    temperature: float = 0.6
    backend: str = "vllm"
    num_generations_per_prompt: int = 1
    device: str = "cuda"
    max_new_tokens: int = 512
    verbose: bool = False
    cache_dir: str = None

    # vLLM settings
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    vllm_max_model_len: int = 500


class BenchmarkEvaluator:
    """Evaluates a model on a safety benchmark and saves prompt inputs/outputs"""

    def __init__(self, config: BenchmarkEvalConfig):
        self.config = config

        # Create a minimal CrawlerConfig
        self.crawler_config = CrawlerConfig(
            temperature=config.temperature,
            device=config.device,
            max_generated_tokens=config.max_new_tokens,
            model_path=config.model_path,
        )

        # Create a Crawler instance for refusal checking
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

        total_prompts = 0
        if "topics" in benchmark:
            for topic in benchmark["topics"]:
                queries = topic.get("queries", [])
                total_prompts += len(queries)

        print(f"Loaded benchmark with {total_prompts} total prompts")
        return benchmark

    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model: {self.config.model_path}")
        print(f"Backend: {self.config.backend}")

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

    def generate_with_prompts(
        self, prompts: List[str]
    ) -> tuple[List[List[Dict[str, any]]], List[str]]:
        """
        Generate completions for each prompt and capture prompt inputs.

        Args:
            prompts: List of prompt strings

        Returns:
            Tuple of:
            - List of lists, where each inner list contains dicts with:
              {
                  "prompt_input": str,  # The exact prompt input sent to model
                  "generation": str,
                  "is_refusal": bool
              }
            - List of original query strings
        """
        # Repeat each prompt num_generations_per_prompt times
        repeated_prompts = []
        prompt_indices = []  # Track which original prompt each belongs to
        for i, prompt in enumerate(prompts):
            repeated_prompts.extend([prompt] * self.config.num_generations_per_prompt)
            prompt_indices.extend([i] * self.config.num_generations_per_prompt)

        print(
            f"Generating {len(repeated_prompts)} completions ({len(prompts)} prompts × {self.config.num_generations_per_prompt} generations)..."
        )

        # Generate completions - batch_generate returns (generations, input_strs)
        generations, input_strs = batch_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            selected_topics=repeated_prompts,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            verbose=self.config.verbose,
            cfg=self.crawler_config,
            num_samples_per_topic=1,  # Already repeated prompts above
        )

        # Group generations by original prompt and capture prompt inputs
        all_results = []
        for i in tqdm(range(len(prompts)), desc="Processing results"):
            start_idx = i * self.config.num_generations_per_prompt
            end_idx = start_idx + self.config.num_generations_per_prompt
            prompt_generations = generations[start_idx:end_idx]
            prompt_inputs = input_strs[start_idx:end_idx]

            # Check each generation for refusal
            prompt_results = []
            for gen, prompt_input in zip(prompt_generations, prompt_inputs):
                is_refusal = self.crawler.is_refusal(gen)
                prompt_results.append(
                    {
                        "prompt_input": prompt_input,
                        "generation": gen,
                        "is_refusal": is_refusal,
                    }
                )

            all_results.append(prompt_results)

        return all_results, prompts

    def evaluate_benchmark(self, benchmark: Dict) -> Dict:
        """
        Evaluate model on all prompts in the benchmark.

        Returns:
            Results dict with prompt inputs and outputs
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

                # Generate and capture prompt inputs/outputs
                prompt_results, original_queries = self.generate_with_prompts(queries)

                # Structure results
                meta_cluster_results = []
                for query, result_list in zip(original_queries, prompt_results):
                    meta_cluster_results.append(
                        {"query": query, "generations": result_list}
                    )

                results["results"][meta_cluster] = meta_cluster_results

        return results

    def save_results(self, results: Dict):
        """Save results to JSON file"""
        # Create output directory if it doesn't exist
        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        print(f"\nSaving results to: {output_path}")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Print summary
        total_queries = sum(
            len(cluster_results) for cluster_results in results["results"].values()
        )
        total_generations = sum(
            len(gen["generations"])
            for cluster_results in results["results"].values()
            for gen in cluster_results
        )

        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nTotal Meta Clusters: {len(results['results'])}")
        print(f"Total Queries: {total_queries}")
        print(f"Total Generations: {total_generations}")
        print(f"\nResults saved to: {output_path}")
        print("=" * 80)

        return str(output_path)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate refusal check queries and save prompt inputs/outputs"
    )
    parser.add_argument(
        "--benchmark_file",
        type=str,
        default="artifacts/pbr/refusal_check_topics.json",
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="allenai/Llama-3.1-Tulu-3-8B-SFT",
        help="HuggingFace model path",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="artifacts/pbr/tulu_prompt_outputs.json",
        help="Path to save results JSON file",
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
        default=1,
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
        help="Directory to cache models",
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

    args = parser.parse_args()

    # Create config
    config = BenchmarkEvalConfig(
        benchmark_file=args.benchmark_file,
        model_path=args.model_path,
        output_file=args.output_file,
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

    # Load benchmark
    print("\n" + "=" * 80)
    print("Loading benchmark...")
    print("=" * 80)
    benchmark = evaluator.load_benchmark()

    # Load model
    print("\n" + "=" * 80)
    print("Loading model...")
    print("=" * 80)
    evaluator.load_model()

    # Evaluate benchmark
    print("\n" + "=" * 80)
    print("Running evaluation...")
    print("=" * 80)
    results = evaluator.evaluate_benchmark(benchmark)

    # Save results
    output_file = evaluator.save_results(results)
    print(f"\nEvaluation complete! Results saved to: {output_file}")


if __name__ == "__main__":
    main()


