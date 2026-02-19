import json
from typing import List

import matplotlib.pyplot as plt
import torch

from core.topic_queue import Topic

EPS = 1e-10


class CrawlerStats:
    def __init__(self):
        # Cumulative counters
        self.total_all = 0  # All topics generated
        self.total_deduped = 0  # These are all HEAD topics (topics after deduplication)
        self.total_refusals = 0  # All deduplicated topics that yield refusals
        self.total_unique_refusals = 0

        # History tracking
        self.all_per_step = []
        self.deduped_per_step = []
        self.refusal_per_step = []

    def log_step(
        self,
        new_topics_all: int,
        new_topics_deduped: int,
        new_topics_refusals: float,
        total_unique_refusals: int,
    ):
        """Log statistics for current step"""
        self.total_all += new_topics_all
        self.total_deduped += new_topics_deduped
        self.total_refusals += new_topics_refusals
        self.total_unique_refusals = total_unique_refusals
        self.all_per_step.append(new_topics_all)
        self.deduped_per_step.append(new_topics_deduped)
        self.refusal_per_step.append(new_topics_refusals)

    def get_current_metrics(self) -> dict:
        """Get current state of metrics"""
        return {
            "total_all": self.total_all,
            "total_deduped": self.total_deduped,
            "total_refusals": sum(self.refusal_per_step),
            "total_unique_refusals": self.total_unique_refusals,
            "avg_refusal_rate": (sum(self.refusal_per_step) / (self.total_all + EPS)),
            "current_step": len(self.all_per_step),
        }

    def visualize_cumulative_topic_count(
        self,
        save_path: str = None,
        show_all_topics: bool = False,
        title: str = "Cumulative topic and refusal count",
    ):
        cumulative_generations = torch.cumsum(torch.tensor(self.all_per_step), dim=0)
        cumulative_topics = torch.cumsum(torch.tensor(self.deduped_per_step), dim=0)
        cumulative_refusals = torch.cumsum(torch.tensor(self.refusal_per_step), dim=0)

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.grid(zorder=-1)
        ax.scatter(
            cumulative_generations, cumulative_topics, label="Unique topics", zorder=10
        )
        ax.scatter(
            cumulative_generations,
            cumulative_refusals,
            label="Refused unique topics",
            zorder=10,
        )
        ax.set_xlabel("Total crawled topics")
        ax.set_ylabel("Total crawled topics after filter")
        ax.set_title(title)
        ax.legend()
        if save_path is not None:
            plt.savefig(save_path)
        return fig

    def to_dict(self):
        """Convert the crawler stats to a dictionary representation."""
        stats_dict = {
            "cumulative": {
                "total_all": self.total_all,
                "total_deduped": self.total_deduped,
                "total_refusals": self.total_refusals,
                "total_unique_refusals": self.total_unique_refusals,
            },
            "current_metrics": self.get_current_metrics(),
            "history": {
                "all_per_step": self.all_per_step,
                "deduped_per_step": self.deduped_per_step,
                "refusal_per_step": self.refusal_per_step,
            },
        }
        return stats_dict

    def save(self, filename: str):
        """Save the crawler stats to a JSON file."""
        stats_dict = self.to_dict()
        with open(filename, "w") as f:
            json.dump(stats_dict, f)

        return stats_dict

    @classmethod
    def load(cls, stats_dict: dict):
        """Load the crawler stats from a dictionary."""
        crawler_stats = cls()
        crawler_stats.total_all = stats_dict["cumulative"]["total_all"]
        crawler_stats.total_deduped = stats_dict["cumulative"]["total_deduped"]
        crawler_stats.total_refusals = stats_dict["cumulative"]["total_refusals"]
        crawler_stats.total_unique_refusals = stats_dict["cumulative"][
            "total_unique_refusals"
        ]
        crawler_stats.all_per_step = stats_dict["history"]["all_per_step"]
        crawler_stats.deduped_per_step = stats_dict["history"]["deduped_per_step"]
        crawler_stats.refusal_per_step = stats_dict["history"]["refusal_per_step"]
        return crawler_stats

    def __repr__(self):
        return f"CrawlerStats(total_all={self.total_all}, total_deduped={self.total_deduped}, all_per_step={self.all_per_step}, deduped_per_step={self.deduped_per_step}, refusal_per_step={self.refusal_per_step})"
