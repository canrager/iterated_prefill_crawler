"""
Benchmark Evaluation Script

Evaluates a language model on a safety benchmark by:
1. Loading prompts from a benchmark JSON file
2. Generating multiple completions per prompt
3. Checking each generation for refusals using the Crawler's refusal detection logic
4. Saving results with refusal statistics
"""

import os

# Set environment variable to force spawn method before any imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import multiprocessing
# Set multiprocessing start method to 'spawn' for CUDA compatibility
# MUST be set before importing torch or any CUDA libraries
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import json
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime
from tqdm import tqdm

from core.crawler import Crawler
from core.crawler_config import CrawlerConfig
from core.generation_utils import batch_generate
from core.llm_utils import load_model_and_tokenizer
from core.project_config import MODELS_DIR


@dataclass
class BenchmarkEvalConfig:
    """Configuration for benchmark evaluation"""
    benchmark_file: str = "artifacts/input/tulu3_jailbreak_benchmark.json"
    model_path: str = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    output_dir: str = "artifacts/result"
    temperature: float = 0.6
    backend: str = "vllm"  # "vllm", "transformers", or "api"
    num_generations_per_prompt: int = 10
    device: str = "cuda"
    max_new_tokens: int = 512
    verbose: bool = False
    cache_dir: str = str(MODELS_DIR)

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
        # We use a dummy save filename since we won't actually save crawler state
        self.crawler = Crawler(
            crawler_config=self.crawler_config,
            save_filename="/tmp/dummy_crawler.json"
        )

        self.model = None
        self.tokenizer = None

    def load_benchmark(self) -> Dict:
        """Load benchmark JSON file"""
        print(f"Loading benchmark from: {self.config.benchmark_file}")
        with open(self.config.benchmark_file, 'r') as f:
            benchmark = json.load(f)

        # Count total prompts
        total_prompts = 0
        if "safety_benchmark_prompts" in benchmark:
            for category, subcategories in benchmark["safety_benchmark_prompts"].items():
                for subcategory, prompts in subcategories.items():
                    total_prompts += len(prompts)

        print(f"Loaded benchmark with {total_prompts} total prompts")
        return benchmark

    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model: {self.config.model_path}")
        print(f"Backend: {self.config.backend}")
        print(f"Cache directory: {self.config.cache_dir}")

        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=self.config.model_path,
            cache_dir=self.config.cache_dir,
            device=self.config.device,
            backend=self.config.backend,
            vllm_tensor_parallel_size=self.config.tensor_parallel_size,
            vllm_gpu_memory_utilization=self.config.gpu_memory_utilization,
            vllm_max_model_len=self.config.vllm_max_model_len,
        )

        print("Model loaded successfully")

    def generate_and_check_refusal(
        self,
        prompts: List[str]
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
        # vLLM will handle batching internally
        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt] * self.config.num_generations_per_prompt)

        print(f"Generating {len(repeated_prompts)} completions ({len(prompts)} prompts × {self.config.num_generations_per_prompt} generations)...")

        # Generate completions - vLLM handles batching internally
        generations, _ = batch_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            selected_topics=repeated_prompts,
            user_message_template="{}",  # Use prompt as-is
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
                prompt_results.append({
                    "generation": gen,
                    "is_refusal": is_refusal
                })

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
            "statistics": {}
        }

        # Process each category and subcategory
        safety_prompts = benchmark.get("safety_benchmark_prompts", {})

        for category, subcategories in tqdm(safety_prompts.items(), desc="Categories"):
            results["results"][category] = {}

            for subcategory, prompts in tqdm(
                subcategories.items(),
                desc=f"  {category}",
                leave=False
            ):
                print(f"\nProcessing {category}/{subcategory} ({len(prompts)} prompts)")

                # Generate and check refusals
                prompt_results = self.generate_and_check_refusal(prompts)

                # Structure results as: {prompt: [{is_refusal, generation}, ...]}
                subcategory_results = {}
                for prompt, result_list in zip(prompts, prompt_results):
                    subcategory_results[prompt] = result_list

                results["results"][category][subcategory] = subcategory_results

        # Compute statistics
        results["statistics"] = self.compute_statistics(results["results"])

        return results

    def compute_statistics(self, results: Dict) -> Dict:
        """Compute refusal statistics across categories and subcategories"""
        stats = {
            "overall": {
                "total_prompts": 0,
                "total_generations": 0,
                "total_refusals": 0,
                "refusal_rate": 0.0,
            },
            "by_category": {},
            "by_subcategory": {}
        }

        for category, subcategories in results.items():
            category_stats = {
                "total_prompts": 0,
                "total_generations": 0,
                "total_refusals": 0,
                "refusal_rate": 0.0,
            }

            for subcategory, prompts in subcategories.items():
                subcategory_stats = {
                    "total_prompts": len(prompts),
                    "total_generations": 0,
                    "total_refusals": 0,
                    "refusal_rate": 0.0,
                }

                for prompt, generations in prompts.items():
                    subcategory_stats["total_generations"] += len(generations)
                    subcategory_stats["total_refusals"] += sum(
                        1 for g in generations if g["is_refusal"]
                    )

                if subcategory_stats["total_generations"] > 0:
                    subcategory_stats["refusal_rate"] = (
                        subcategory_stats["total_refusals"] /
                        subcategory_stats["total_generations"]
                    )

                # Update category stats
                category_stats["total_prompts"] += subcategory_stats["total_prompts"]
                category_stats["total_generations"] += subcategory_stats["total_generations"]
                category_stats["total_refusals"] += subcategory_stats["total_refusals"]

                # Store subcategory stats
                subcategory_key = f"{category}/{subcategory}"
                stats["by_subcategory"][subcategory_key] = subcategory_stats

            if category_stats["total_generations"] > 0:
                category_stats["refusal_rate"] = (
                    category_stats["total_refusals"] /
                    category_stats["total_generations"]
                )

            stats["by_category"][category] = category_stats

            # Update overall stats
            stats["overall"]["total_prompts"] += category_stats["total_prompts"]
            stats["overall"]["total_generations"] += category_stats["total_generations"]
            stats["overall"]["total_refusals"] += category_stats["total_refusals"]

        if stats["overall"]["total_generations"] > 0:
            stats["overall"]["refusal_rate"] = (
                stats["overall"]["total_refusals"] /
                stats["overall"]["total_generations"]
            )

        return stats

    def save_results(self, results: Dict):
        """Save results to JSON file"""
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Generate filename
        model_name = self.config.model_path.split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}_results.json"
        filepath = os.path.join(self.config.output_dir, filename)

        # Save results
        print(f"\nSaving results to: {filepath}")
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary statistics
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)

        stats = results["statistics"]
        print(f"\nOverall Statistics:")
        print(f"  Total Prompts: {stats['overall']['total_prompts']}")
        print(f"  Total Generations: {stats['overall']['total_generations']}")
        print(f"  Total Refusals: {stats['overall']['total_refusals']}")
        print(f"  Refusal Rate: {stats['overall']['refusal_rate']:.2%}")

        print(f"\nBy Category:")
        for category, cat_stats in stats["by_category"].items():
            print(f"  {category}:")
            print(f"    Prompts: {cat_stats['total_prompts']}")
            print(f"    Refusal Rate: {cat_stats['refusal_rate']:.2%}")

        print("\n" + "="*80)

        return filepath

    def run(self):
        """Run the full evaluation pipeline"""
        print("Starting benchmark evaluation...")
        print(f"Config: {self.config}")

        # 1. Load benchmark
        benchmark = self.load_benchmark()

        # 2. Load model
        self.load_model()

        # 3. Evaluate
        print("\nRunning evaluation...")
        results = self.evaluate_benchmark(benchmark)

        # 4. Save results
        output_file = self.save_results(results)

        print(f"\nEvaluation complete! Results saved to: {output_file}")

        return results


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a model on a safety benchmark")
    parser.add_argument(
        "--benchmark_file",
        type=str,
        default="artifacts/input/tulu3_jailbreak_benchmark.json",
        help="Path to benchmark JSON file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="allenai/Llama-3.1-Tulu-3-8B-SFT",
        help="HuggingFace model path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/result",
        help="Directory to save results"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "transformers", "api"],
        help="Backend to use for generation"
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=10,
        help="Number of generations per prompt"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(MODELS_DIR),
        help="Directory to cache models"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for vLLM tensor parallelism"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory fraction for vLLM"
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=None,
        help="Max sequence length for vLLM (None = model default)"
    )

    args = parser.parse_args()

    # Create config
    config = BenchmarkEvalConfig(
        benchmark_file=args.benchmark_file,
        model_path=args.model_path,
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

    # Run evaluation
    evaluator = BenchmarkEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
