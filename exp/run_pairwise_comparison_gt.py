"""
Pairwise Comparison Ground Truth Ranking Script

This script:
1. Loads ground truth safety topics from tulu3_ground_truth_safety_topics.json
2. Runs pairwise comparison ranking using an LLM judge
3. Saves the ranking results
4. Loads benchmark refusal rate results and creates a ranking
5. Computes correlation between the two rankings (Spearman and Kendall's Tau)
6. Generates a correlation scatterplot with linear fit
"""

import os

# Set environment variable to force spawn method before any imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import multiprocessing
# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime
from scipy.stats import spearmanr, kendalltau

from core.ranking import DEFAULT_RANKING_CONFIG, setup_ranking_experiment
from core.project_config import MODELS_DIR, RESULT_DIR, resolve_cache_dir


@dataclass
class PairwiseComparisonGTConfig:
    """Configuration for ground truth pairwise comparison ranking"""
    # Ground truth and benchmark files
    gt_topics_file: str = "artifacts/input/tulu3_ground_truth_safety_topics.json"
    benchmark_results_file: str = "artifacts/result/Llama-3.1-Tulu-3-8B-SFT_20251121_051021_results.json"

    # Model configuration
    lm_judge_name: str = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    device: str = "cuda"
    cache_dir: str = None  # Will be resolved to ROOT_DIR.parent / "models" if None

    # Ranking configuration (uses DEFAULT_RANKING_CONFIG from core/ranking.py)
    ranking_config: Dict = None

    # Output configuration
    output_dir: str = str(RESULT_DIR)
    experiment_name: str = "pairwise_comparison_gt"

    # Execution flags
    force_recompute: bool = False
    debug: bool = False


class PairwiseComparisonGTEvaluator:
    """Evaluates ground truth topics using pairwise comparison and compares with benchmark refusal rates"""

    def __init__(self, config: PairwiseComparisonGTConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_ground_truth_topics(self) -> Dict[str, List[str]]:
        """Load ground truth safety topics from JSON file"""
        print(f"Loading ground truth topics from: {self.config.gt_topics_file}")
        with open(self.config.gt_topics_file, 'r') as f:
            gt_topics = json.load(f)

        # Flatten the structure to get all subcategories
        all_subcategories = []
        for category, subcategories in gt_topics.items():
            all_subcategories.extend(subcategories)

        print(f"Loaded {len(all_subcategories)} ground truth topic subcategories")
        return gt_topics, all_subcategories

    def create_topic_dicts_from_subcategories(self, subcategories: List[str]) -> Dict[str, Dict]:
        """Convert list of subcategories to topic_dicts format for ranking"""
        topic_dicts = {}
        for idx, subcategory in enumerate(subcategories):
            topic_dicts[subcategory] = {
                "first_occurence_id": idx,
            }
        return topic_dicts

    def load_benchmark_results(self) -> Dict:
        """Load benchmark evaluation results"""
        print(f"Loading benchmark results from: {self.config.benchmark_results_file}")
        with open(self.config.benchmark_results_file, 'r') as f:
            benchmark_results = json.load(f)

        print(f"Loaded benchmark results from {benchmark_results['metadata']['timestamp']}")
        return benchmark_results

    def extract_refusal_rate_ranking(self, benchmark_results: Dict) -> List[Tuple[str, float]]:
        """
        Extract subcategory refusal rates from benchmark results and create a ranking.

        Returns:
            List of (subcategory, refusal_rate) tuples sorted by refusal rate (descending)
        """
        print("Extracting refusal rate rankings from benchmark results...")

        by_subcategory = benchmark_results["statistics"]["by_subcategory"]

        # Create list of (subcategory, refusal_rate) tuples
        refusal_rankings = []
        for subcategory_key, stats in by_subcategory.items():
            # Extract the subcategory name from the "category/subcategory" format
            # We need to match it with the ground truth subcategory names
            subcategory_name = subcategory_key.split('/')[-1]
            # Convert underscore format to title case to match ground truth format
            # e.g., "express_curiosity" -> "Express curiosity"
            subcategory_display = subcategory_name.replace('_', ' ').capitalize()

            refusal_rankings.append({
                "subcategory_key": subcategory_key,
                "subcategory_name": subcategory_display,
                "refusal_rate": stats["refusal_rate"],
                "total_prompts": stats["total_prompts"],
                "total_refusals": stats["total_refusals"],
                "total_generations": stats["total_generations"],
            })

        # Sort by refusal rate (descending - higher refusal rate = higher rank)
        refusal_rankings.sort(key=lambda x: x["refusal_rate"], reverse=True)

        print(f"Created refusal rate ranking for {len(refusal_rankings)} subcategories")
        return refusal_rankings

    def run_pairwise_ranking(self, subcategories: List[str]) -> Dict[str, Dict]:
        """Run pairwise comparison ranking on ground truth subcategories"""
        print("\n" + "="*80)
        print("Running pairwise comparison ranking...")
        print("="*80)

        # Convert subcategories to topic_dicts format
        topic_dicts = self.create_topic_dicts_from_subcategories(subcategories)

        # Use default ranking config or provided one
        ranking_config = self.config.ranking_config or DEFAULT_RANKING_CONFIG

        # Resolve cache_dir relative to ROOT_DIR.parent and create if needed
        cache_dir_path = resolve_cache_dir(self.config.cache_dir)
        cache_dir_str = str(cache_dir_path)

        # Run ranking experiment
        ranking_results = setup_ranking_experiment(
            topic_dicts=topic_dicts,
            run_title=self.config.experiment_name,
            model_name=self.config.lm_judge_name,
            device=self.config.device,
            cache_dir=cache_dir_str,
            config=ranking_config,
            force_recompute=self.config.force_recompute,
            debug=self.config.debug,
        )

        return ranking_results

    def save_pairwise_ranking_results(self, ranking_results: Dict[str, Dict]) -> str:
        """Save pairwise ranking results to JSON file"""
        # Generate filename
        model_name = self.config.lm_judge_name.split("/")[-1]
        gt_file_name = os.path.basename(self.config.gt_topics_file).replace('.json', '')
        filename = f"{self.config.experiment_name}_{self.timestamp}_{gt_file_name}_{model_name}.json"
        filepath = os.path.join(self.config.output_dir, filename)

        print(f"\nSaving pairwise ranking results to: {filepath}")
        with open(filepath, 'w') as f:
            json.dump(ranking_results, f, indent=2)

        return filepath

    def save_refusal_ranking_results(self, refusal_rankings: List[Dict]) -> str:
        """Save refusal rate ranking results to JSON file"""
        # Generate filename
        filename = f"{self.config.experiment_name}_{self.timestamp}_refusal_rate_ranking.json"
        filepath = os.path.join(self.config.output_dir, filename)

        print(f"Saving refusal rate ranking results to: {filepath}")

        # Add rank indices to the results
        for rank_idx, item in enumerate(refusal_rankings):
            item["rank_idx"] = rank_idx

        with open(filepath, 'w') as f:
            json.dump(refusal_rankings, f, indent=2)

        return filepath

    def match_topics_for_correlation(
        self,
        pairwise_ranking: Dict[str, Dict],
        refusal_rankings: List[Dict],
        gt_topics: Dict[str, List[str]]
    ) -> Tuple[List[str], List[float], List[float]]:
        """
        Match topics between pairwise ranking and refusal rate ranking.

        Returns:
            matched_topics: List of topic names that appear in both rankings
            pairwise_ranks: List of rank indices from pairwise ranking
            refusal_ranks: List of rank indices from refusal rate ranking
        """
        print("\n" + "="*80)
        print("Matching topics for correlation analysis...")
        print("="*80)

        # Create mapping from normalized subcategory names to original GT names
        gt_subcategory_mapping = {}
        for category, subcategories in gt_topics.items():
            for subcat in subcategories:
                # Normalize: lowercase and handle variations
                normalized = subcat.lower().strip()
                gt_subcategory_mapping[normalized] = subcat

        # Create refusal rate lookup by normalized name
        refusal_lookup = {}
        for item in refusal_rankings:
            # Try multiple normalization strategies
            subcat_key = item["subcategory_key"]
            subcat_name = item["subcategory_name"]

            # Extract the actual subcategory from category/subcategory format
            category_part, subcat_part = subcat_key.split('/')

            # Normalize the subcategory part
            normalized = subcat_part.replace('_', ' ').lower().strip()

            refusal_lookup[normalized] = {
                "rank_idx": item["rank_idx"],
                "refusal_rate": item["refusal_rate"],
                "original_key": subcat_key,
            }

        # Match topics
        matched_topics = []
        pairwise_ranks = []
        refusal_ranks = []
        unmatched_topics = []

        for topic, topic_data in pairwise_ranking.items():
            # Normalize the topic name from pairwise ranking
            normalized_topic = topic.lower().strip()

            # Check if this topic exists in refusal rankings
            if normalized_topic in refusal_lookup:
                matched_topics.append(topic)

                # Get pairwise rank (assuming "elo" method)
                pairwise_rank_idx = topic_data["ranking"]["elo"]["rank_idx"]
                pairwise_ranks.append(pairwise_rank_idx)

                # Get refusal rate rank
                refusal_rank_idx = refusal_lookup[normalized_topic]["rank_idx"]
                refusal_ranks.append(refusal_rank_idx)
            else:
                unmatched_topics.append(topic)

        print(f"Matched {len(matched_topics)} topics")
        if unmatched_topics:
            print(f"Unmatched topics ({len(unmatched_topics)}): {unmatched_topics[:10]}")

        return matched_topics, pairwise_ranks, refusal_ranks

    def compute_correlations(
        self,
        pairwise_ranks: List[float],
        refusal_ranks: List[float]
    ) -> Tuple[float, float, float, float]:
        """
        Compute Spearman and Kendall's Tau correlations.

        Returns:
            spearman_rho, spearman_pval, kendall_tau, kendall_pval
        """
        print("\n" + "="*80)
        print("Computing correlations...")
        print("="*80)

        # Compute Spearman correlation
        spearman_rho, spearman_pval = spearmanr(pairwise_ranks, refusal_ranks)
        print(f"Spearman's rho: {spearman_rho:.4f} (p-value: {spearman_pval:.4e})")

        # Compute Kendall's Tau
        kendall_tau, kendall_pval = kendalltau(pairwise_ranks, refusal_ranks)
        print(f"Kendall's tau: {kendall_tau:.4f} (p-value: {kendall_pval:.4e})")

        return spearman_rho, spearman_pval, kendall_tau, kendall_pval

    def plot_correlation_scatterplot(
        self,
        matched_topics: List[str],
        pairwise_ranks: List[float],
        refusal_ranks: List[float],
        spearman_rho: float,
        kendall_tau: float,
    ) -> str:
        """
        Generate a correlation scatterplot with linear fit.

        Returns:
            filepath: Path to saved plot
        """
        print("\n" + "="*80)
        print("Generating correlation scatterplot...")
        print("="*80)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot
        ax.scatter(pairwise_ranks, refusal_ranks, alpha=0.6, s=50)

        # Linear fit
        z = np.polyfit(pairwise_ranks, refusal_ranks, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(pairwise_ranks), max(pairwise_ranks), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Linear fit: y={z[0]:.2f}x+{z[1]:.2f}')

        # Labels and title
        ax.set_xlabel('Pairwise Comparison Rank Index', fontsize=12)
        ax.set_ylabel('Refusal Rate Rank Index', fontsize=12)
        ax.set_title(
            f'Correlation between Pairwise Comparison and Refusal Rate Rankings\n'
            f'Spearman ρ={spearman_rho:.3f}, Kendall τ={kendall_tau:.3f}',
            fontsize=14
        )

        # Add legend
        ax.legend(fontsize=10)

        # Grid
        ax.grid(True, alpha=0.3)

        # Add diagonal reference line (perfect correlation)
        max_rank = max(max(pairwise_ranks), max(refusal_ranks))
        ax.plot([0, max_rank], [0, max_rank], 'k:', alpha=0.3, linewidth=1, label='Perfect correlation')

        # Tight layout
        plt.tight_layout()

        # Save plot
        filename = f"{self.timestamp}_{self.config.experiment_name}_correlation_plot.png"
        filepath = os.path.join(self.config.output_dir, filename)

        print(f"Saving correlation plot to: {filepath}")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def run(self):
        """Run the full evaluation pipeline"""
        print("\n" + "="*80)
        print("Starting Pairwise Comparison Ground Truth Ranking Evaluation")
        print("="*80)
        print(f"Experiment name: {self.config.experiment_name}")
        print(f"Timestamp: {self.timestamp}")
        print(f"LM Judge: {self.config.lm_judge_name}")

        # 1. Load ground truth topics
        gt_topics, all_subcategories = self.load_ground_truth_topics()

        # 2. Run pairwise comparison ranking
        pairwise_ranking = self.run_pairwise_ranking(all_subcategories)
        pairwise_ranking_file = self.save_pairwise_ranking_results(pairwise_ranking)

        # 3. Load benchmark results and extract refusal rate ranking
        benchmark_results = self.load_benchmark_results()
        refusal_rankings = self.extract_refusal_rate_ranking(benchmark_results)
        refusal_ranking_file = self.save_refusal_ranking_results(refusal_rankings)

        # 4. Match topics for correlation
        matched_topics, pairwise_ranks, refusal_ranks = self.match_topics_for_correlation(
            pairwise_ranking, refusal_rankings, gt_topics
        )

        # 5. Compute correlations
        spearman_rho, spearman_pval, kendall_tau, kendall_pval = self.compute_correlations(
            pairwise_ranks, refusal_ranks
        )

        # 6. Generate correlation plot
        plot_file = self.plot_correlation_scatterplot(
            matched_topics, pairwise_ranks, refusal_ranks, spearman_rho, kendall_tau
        )

        # 7. Save summary
        summary = {
            "metadata": {
                "experiment_name": self.config.experiment_name,
                "timestamp": self.timestamp,
                "lm_judge": self.config.lm_judge_name,
                "gt_topics_file": self.config.gt_topics_file,
                "benchmark_results_file": self.config.benchmark_results_file,
            },
            "correlations": {
                "spearman": {
                    "rho": float(spearman_rho),
                    "p_value": float(spearman_pval),
                },
                "kendall": {
                    "tau": float(kendall_tau),
                    "p_value": float(kendall_pval),
                },
            },
            "matched_topics_count": len(matched_topics),
            "total_gt_topics": len(all_subcategories),
            "output_files": {
                "pairwise_ranking": pairwise_ranking_file,
                "refusal_ranking": refusal_ranking_file,
                "correlation_plot": plot_file,
            }
        }

        summary_file = os.path.join(
            self.config.output_dir,
            f"{self.timestamp}_{self.config.experiment_name}_summary.json"
        )
        print(f"\nSaving summary to: {summary_file}")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Print final summary
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"\nCorrelation Results:")
        print(f"  Spearman's ρ: {spearman_rho:.4f} (p={spearman_pval:.4e})")
        print(f"  Kendall's τ: {kendall_tau:.4f} (p={kendall_pval:.4e})")
        print(f"\nMatched Topics: {len(matched_topics)} / {len(all_subcategories)}")
        print(f"\nOutput Files:")
        print(f"  Pairwise Ranking: {pairwise_ranking_file}")
        print(f"  Refusal Ranking: {refusal_ranking_file}")
        print(f"  Correlation Plot: {plot_file}")
        print(f"  Summary: {summary_file}")
        print("="*80)

        return summary


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run pairwise comparison ranking on ground truth topics and compare with benchmark refusal rates"
    )
    parser.add_argument(
        "--gt_topics_file",
        type=str,
        default="artifacts/input/tulu3_ground_truth_safety_topics.json",
        help="Path to ground truth topics JSON file"
    )
    parser.add_argument(
        "--benchmark_results_file",
        type=str,
        default="artifacts/result/Llama-3.1-Tulu-3-8B-SFT_20251121_051021_results.json",
        help="Path to benchmark results JSON file"
    )
    parser.add_argument(
        "--lm_judge_name",
        type=str,
        default="allenai/Llama-3.1-Tulu-3-8B-SFT",
        help="LLM judge model name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(RESULT_DIR),
        help="Directory to save results"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="pairwise_comparison_gt",
        help="Experiment name"
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recompute even if results exist"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug mode (fewer comparisons)"
    )

    args = parser.parse_args()

    # Create config
    config = PairwiseComparisonGTConfig(
        gt_topics_file=args.gt_topics_file,
        benchmark_results_file=args.benchmark_results_file,
        lm_judge_name=args.lm_judge_name,
        device=args.device,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        force_recompute=args.force_recompute,
        debug=args.debug,
    )

    # Run evaluation
    evaluator = PairwiseComparisonGTEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
