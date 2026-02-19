import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from scipy.stats import kendalltau
from collections import defaultdict
import seaborn as sns
from matplotlib.gridspec import GridSpec

sns.set_theme()


class RankingTracker:
    """Tracks the evolution of rankings during the comparison process"""

    def __init__(self, topics: List[str]):
        self.topics = topics
        self.rating_history = defaultdict(list)  # Topic -> List[ratings]
        self.rank_history = defaultdict(list)  # Topic -> List[ranks]
        self.uncertainty_history = defaultdict(list)  # Topic -> List[uncertainty]
        self.comparison_counts = []  # Number of comparisons at each point

    def update(self, ratings: Dict[str, Any], comparison_count: int, system_type: str = "elo"):
        """
        Update tracking history with new ratings

        Args:
            ratings: Current ratings (format depends on system type)
            comparison_count: Current number of comparisons
            system_type: 'elo', 'trueskill', or 'wincount'
        """
        self.comparison_counts.append(comparison_count)

        # Store current ratings and compute ranks
        if system_type == "trueskill":
            # For TrueSkill, ratings are (mu, sigma) pairs
            current_ratings = {t: r.mu for t, r in ratings.items()}
            current_uncertainties = {t: r.sigma for t, r in ratings.items()}
        else:
            # For Elo and WinCount, ratings are single numbers
            current_ratings = ratings
            current_uncertainties = {t: 0 for t in ratings}

        # Sort topics by rating to get ranks
        sorted_topics = sorted(current_ratings.items(), key=lambda x: x[1], reverse=True)
        current_ranks = {t: i + 1 for i, (t, _) in enumerate(sorted_topics)}

        # Update histories
        for topic in self.topics:
            self.rating_history[topic].append(current_ratings[topic])
            self.rank_history[topic].append(current_ranks[topic])
            self.uncertainty_history[topic].append(current_uncertainties[topic])

    def get_convergence_metrics(self) -> Dict:
        """Calculate convergence metrics from tracked history"""
        metrics = {
            "rating_trajectory": [],  # Average rating over time
            "rank_changes": [],  # Number of rank changes over time
            "uncertainty": [],  # Average uncertainty over time
            "rank_volatility": [],  # Variance in ranks over time
        }

        # Calculate metrics at each time point
        for i in range(len(self.comparison_counts)):
            # Average rating
            ratings = [self.rating_history[t][i] for t in self.topics]
            metrics["rating_trajectory"].append(np.mean(ratings))

            # Rank changes (if not first point)
            if i > 0:
                changes = sum(
                    1 for t in self.topics if self.rank_history[t][i] != self.rank_history[t][i - 1]
                )
                metrics["rank_changes"].append(changes)

            # Average uncertainty
            uncertainties = [self.uncertainty_history[t][i] for t in self.topics]
            metrics["uncertainty"].append(np.mean(uncertainties))

            # Rank volatility (standard deviation of ranks)
            ranks = [self.rank_history[t][i] for t in self.topics]
            metrics["rank_volatility"].append(np.std(ranks))

        return metrics


class RankingEvaluator:
    """Evaluates and visualizes ranking results"""

    def __init__(self, experiment_results: List[Dict]):
        """
        Initialize evaluator with results from multiple experiment runs

        Args:
            experiment_results: List of dictionaries containing results from each run
                Each dict should have:
                - 'ranking': Dict mapping system name to final rankings
                - 'trackers': Dict mapping system name to RankingTracker
                - 'metadata': Dict with experiment parameters
        """
        self.results = experiment_results
        self.num_runs = len(experiment_results)
        self.system_names = list(experiment_results[0].keys())

    def compute_kendall_tau(
        self, ranking1: List[Tuple[str, float]], ranking2: List[Tuple[str, float]]
    ) -> float:
        """Compute Kendall's Tau between two rankings"""
        topics1 = [t for t, _ in ranking1]
        topics2 = [t for t, _ in ranking2]

        ranks1 = {topic: i for i, topic in enumerate(topics1)}
        ranks2 = {topic: i for i, topic in enumerate(topics2)}

        common_topics = set(topics1) & set(topics2)
        x = [ranks1[t] for t in common_topics]
        y = [ranks2[t] for t in common_topics]

        tau, _ = kendalltau(x, y)
        return tau

    def evaluate_consistency(self) -> Dict:
        """Evaluate consistency within and between ranking systems"""
        results = {}

        # Within-system consistency
        for system in self.system_names:
            taus = []
            for i in range(self.num_runs):
                for j in range(i + 1, self.num_runs):
                    ranking1 = self.results[i][system]["ranking"]
                    ranking2 = self.results[j][system]["ranking"]
                    tau = self.compute_kendall_tau(ranking1, ranking2)
                    taus.append(tau)

            results[f"{system}_internal"] = {
                "mean": np.mean(taus),
                "std": np.std(taus),
                "min": np.min(taus),
                "max": np.max(taus),
            }

        # Between-system consistency
        for i, sys1 in enumerate(self.system_names):
            for sys2 in self.system_names[i + 1 :]:
                taus = []
                for run in range(self.num_runs):
                    ranking1 = self.results[run][sys1]["ranking"]
                    ranking2 = self.results[run][sys2]["ranking"]
                    tau = self.compute_kendall_tau(ranking1, ranking2)
                    taus.append(tau)

                results[f"{sys1}_vs_{sys2}"] = {
                    "mean": np.mean(taus),
                    "std": np.std(taus),
                    "min": np.min(taus),
                    "max": np.max(taus),
                }

        return results

    def plot_convergence(self, save_path: str = None):
        """Create comprehensive convergence visualization"""
        # Increase figure size and adjust font sizes
        plt.rcParams.update({'font.size': 14})  # Increase base font size
        fig = plt.figure(figsize=(24, 18))  # Increase from (20, 15)
        gs = GridSpec(3, 2, figure=fig)

        # 1. Rating Trajectories
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_rating_trajectories(ax1)

        # 2. Rank Changes
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_rank_changes(ax2)

        # 3. Uncertainty Evolution
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_uncertainty(ax3)

        # 4. Rank Volatility
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_rank_volatility(ax4)

        # 5. Consistency Heatmap
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_consistency_heatmap(ax5)

        # Adjust spacing between subplots
        plt.tight_layout(pad=3.0)  # Increase padding between subplots
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_rating_trajectories(self, ax):
        """Plot rating trajectories for all systems"""
        for system in self.system_names:
            trajectories = []
            comparison_counts = None
            for run in self.results:
                metrics = run[system]["tracker"].get_convergence_metrics()
                trajectories.append(metrics["rating_trajectory"])
                # Get comparison counts from the first run (they're the same for all runs)
                if comparison_counts is None:
                    comparison_counts = run[system]["tracker"].comparison_counts

            mean_traj = np.mean(trajectories, axis=0)
            std_traj = np.std(trajectories, axis=0)

            ax.plot(comparison_counts, mean_traj, label=system)
            ax.fill_between(
                comparison_counts, mean_traj - std_traj, mean_traj + std_traj, alpha=0.2
            )

        ax.set_title("Rating Trajectories Over Time")
        ax.set_xlabel("Number of Comparisons")
        ax.set_ylabel("Average Rating")
        ax.legend()

    def _plot_rank_changes(self, ax):
        """Plot rank changes over time"""
        for system in self.system_names:
            changes = []
            comparison_counts = None
            for run in self.results:
                metrics = run[system]["tracker"].get_convergence_metrics()
                changes.append(metrics["rank_changes"])
                if comparison_counts is None:
                    comparison_counts = run[system]["tracker"].comparison_counts[
                        1:
                    ]  # Skip first point

            mean_changes = np.mean(changes, axis=0)
            ax.plot(comparison_counts, mean_changes, label=system)

        ax.set_title("Rank Changes Over Time")
        ax.set_xlabel("Number of Comparisons")
        ax.set_ylabel("Number of Rank Changes")
        ax.legend()

    def _plot_uncertainty(self, ax):
        """Plot uncertainty evolution"""
        for system in self.system_names:
            uncertainties = []
            comparison_counts = None
            for run in self.results:
                metrics = run[system]["tracker"].get_convergence_metrics()
                uncertainties.append(metrics["uncertainty"])
                if comparison_counts is None:
                    comparison_counts = run[system]["tracker"].comparison_counts

            mean_uncert = np.mean(uncertainties, axis=0)
            ax.plot(comparison_counts, mean_uncert, label=system)

        ax.set_title("Rating Uncertainty Over Time")
        ax.set_xlabel("Number of Comparisons")
        ax.set_ylabel("Average Uncertainty")
        ax.legend()

    def _plot_rank_volatility(self, ax):
        """Plot rank volatility over time"""
        for system in self.system_names:
            volatility = []
            comparison_counts = None
            for run in self.results:
                metrics = run[system]["tracker"].get_convergence_metrics()
                volatility.append(metrics["rank_volatility"])
                if comparison_counts is None:
                    comparison_counts = run[system]["tracker"].comparison_counts

            mean_vol = np.mean(volatility, axis=0)
            ax.plot(comparison_counts, mean_vol, label=system)

        ax.set_title("Rank Volatility Over Time")
        ax.set_xlabel("Number of Comparisons")
        ax.set_ylabel("Rank Standard Deviation")
        ax.legend()

    def _plot_consistency_heatmap(self, ax):
        """Plot consistency comparison heatmap"""
        consistency = self.evaluate_consistency()

        # Prepare data for heatmap
        systems = self.system_names
        n_systems = len(systems)
        heatmap_data = np.zeros((n_systems, n_systems))

        for i, sys1 in enumerate(systems):
            for j, sys2 in enumerate(systems):
                if i == j:
                    # Internal consistency
                    heatmap_data[i, j] = consistency[f"{sys1}_internal"]["mean"]
                elif i < j:
                    # Between-system consistency
                    heatmap_data[i, j] = consistency[f"{sys1}_vs_{sys2}"]["mean"]
                    heatmap_data[j, i] = heatmap_data[i, j]

        # Increase annotation size in heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            xticklabels=systems,
            yticklabels=systems,
            ax=ax,
            cmap="RdYlBu_r",
            vmin=-1,
            vmax=1,
            annot_kws={"size": 12},  # Increase annotation font size
        )
        ax.set_title("Ranking Consistency (Kendall's Tau)", pad=20, fontsize=16)  # Increase title font size and padding


def create_summary_table(evaluator: RankingEvaluator) -> str:
    """Create a formatted summary table of consistency results"""
    consistency = evaluator.evaluate_consistency()

    # Prepare table string
    table = "Ranking System Consistency Summary\n"
    table += "=" * 80 + "\n\n"

    # Internal consistency
    table += "Internal Consistency:\n"
    table += "-" * 80 + "\n"
    table += f"{'System':<20} {'Mean Tau':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}\n"
    table += "-" * 80 + "\n"

    for system in evaluator.system_names:
        metrics = consistency[f"{system}_internal"]
        table += f"{system:<20} {metrics['mean']:<12.3f} {metrics['std']:<12.3f} "
        table += f"{metrics['min']:<12.3f} {metrics['max']:<12.3f}\n"

    # Between-system consistency
    table += "\nBetween-System Consistency:\n"
    table += "-" * 80 + "\n"
    table += f"{'Comparison':<30} {'Mean Tau':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}\n"
    table += "-" * 80 + "\n"

    for i, sys1 in enumerate(evaluator.system_names):
        for sys2 in evaluator.system_names[i + 1 :]:
            metrics = consistency[f"{sys1}_vs_{sys2}"]
            comparison = f"{sys1} vs {sys2}"
            table += f"{comparison:<30} {metrics['mean']:<12.3f} {metrics['std']:<12.3f} "
            table += f"{metrics['min']:<12.3f} {metrics['max']:<12.3f}\n"

    return table
