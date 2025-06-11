from dataclasses import dataclass
from typing import List
import json

import torch
from torch import Tensor

from core.generation_utils import compute_embeddings


@dataclass
class Topic:
    raw: str = None
    english: str = None
    chinese: str = None
    shortened: str = None
    id: int = None
    parent_id: int = None
    is_chinese: bool = None
    is_head: bool = None
    is_refusal: bool = None
    cossim_to_head: float = None
    cluster_idx: int = None
    responses: List[str] = None

    def to_dict(self):
        return {
            "id": self.id,
            "raw": self.raw,
            "english": self.english,
            "chinese": self.chinese,
            "shortened": self.shortened,
            "is_chinese": self.is_chinese,
            "is_head": self.is_head,
            "is_refusal": self.is_refusal,
            "cossim_to_head": self.cossim_to_head,
            "cluster_idx": self.cluster_idx,
            "parent_id": self.parent_id,
            "responses": self.responses,
            "cluster_idx": self.cluster_idx,
            "parent_id": self.parent_id,
            "responses": self.responses,
        }


class TopicQueue:
    def __init__(self):
        # Track clusters
        self.head_topics: List[Topic] = []
        self.head_refusal_topics: List[Topic] = []
        self.cluster_topics: List[List[Topic]] = []
        self.cluster_cossims: List[List[float]] = []

        # Stats
        self.num_head_topics: int = 0
        self.num_topics_per_cluster: List[int] = []
        self.num_head_refusal_topics: int = 0
        self.num_total_topics: int = 0

    # Adding new topics and deduplication
    def add_new_cluster_head(self, topic: Topic):
        """Add a new cluster head."""
        self.head_topics.append(topic)
        self.num_head_topics += 1
        if topic.is_refusal:
            self.head_refusal_topics.append(topic)
            self.num_head_refusal_topics += 1
        self.num_topics_per_cluster.append(1)
        self.cluster_topics.append([topic])
        self.cluster_cossims.append([topic.cossim_to_head])  # should be 1
        self.num_total_topics += 1
        return topic

    def append_to_cluster(self, topic: Topic) -> Topic:
        """Add a topic to an existing cluster."""
        self.cluster_topics[topic.cluster_idx].append(topic)
        self.cluster_cossims[topic.cluster_idx].append(topic.cossim_to_head)
        self.num_topics_per_cluster[topic.cluster_idx] += 1
        self.num_total_topics += 1
        return topic

    def incoming_batch(self, topics: List[Topic]) -> List[Topic]:
        """Process a batch of topics to be added to the queue with deduplication."""
        if topics == []:
            print("No topics passed.")
            return []

        for topic in topics:
            topic.id = self.num_total_topics
            if topic.is_head:
                self.add_new_cluster_head(topic)
            else:
                self.append_to_cluster(topic)
        return topics

    # Saving, loading and logging
    def to_dict(self):
        """Convert the topic queue to a dictionary representation."""
        topic_dict = {
            "topics": {
                "head_refusal_topics": [t.to_dict() for t in self.head_refusal_topics],
                "head_topics": [t.to_dict() for t in self.head_topics],
                "cluster_topics": [
                    [t.to_dict() for t in cluster] for cluster in self.cluster_topics
                ],
                "cluster_cossims": self.cluster_cossims,
            },
            "stats": {
                "num_head_refusal_topics": self.num_head_refusal_topics,
                "num_head_topics": self.num_head_topics,
                "num_topics_per_cluster": self.num_topics_per_cluster,
                "num_total_topics": self.num_total_topics,
            },
        }
        return topic_dict

    def save(self, path: str):
        """Save the topic queue to a JSON file."""
        topic_dict = self.to_dict()
        with open(path, "w") as f:
            json.dump(topic_dict, f)
        return topic_dict

    @classmethod
    def load(cls, topic_dict: dict) -> "TopicQueue":
        """Create a new TopicQueue from a dictionary representation."""
        # Initialize new queue
        queue = cls()

        # Set basic attributes
        queue.num_head_topics = topic_dict["stats"]["num_head_topics"]
        queue.num_head_refusal_topics = topic_dict["stats"]["num_head_refusal_topics"]
        queue.num_topics_per_cluster = topic_dict["stats"]["num_topics_per_cluster"]
        queue.num_total_topics = topic_dict["stats"]["num_total_topics"]

        # Reconstruct head topics
        queue.head_topics = [
            Topic(**topic_data) for topic_data in topic_dict["topics"]["head_topics"]
        ]
        queue.head_refusal_topics = [
            Topic(**topic_data) for topic_data in topic_dict["topics"]["head_refusal_topics"]
        ]
        queue.cluster_topics = [
            [Topic(**topic_data) for topic_data in cluster]
            for cluster in topic_dict["topics"]["cluster_topics"]
        ]
        queue.cluster_cossims = topic_dict["topics"]["cluster_cossims"]
        return queue

    def __repr__(self) -> str:
        """Return a string representation of the TopicQueue showing stats and topics."""
        string = "TopicQueue:\n\n"

        # Stats
        string += "Stats:\n"
        string += f"  Total Clusters: {self.num_head_topics}\n"
        string += f"  Topics per Cluster: {self.num_topics_per_cluster}\n"
        string += f"  Total Refusal Topics: {self.num_head_refusal_topics}\n"
        string += f"  Total Topics: {self.num_total_topics}\n\n"

        # Topics by cluster
        string += "Clusters:\n"
        for i in range(self.num_head_topics):
            # Head topic
            head = self.head_topics[i]
            string += f"\nCluster {i}:\n"
            string += (
                f"  Head: text='{head.english}', raw='{head.raw}'\n"
            )

            # Cluster topics
            string += "  Topics:\n"
            for topic in self.cluster_topics[i]:
                if not topic.is_head:  # Skip head topic since we already showed it
                    string += f"    text='{topic.english}', raw='{topic.raw}'\n"
                    string += f"    cossim_to_head: {topic.cossim_to_head}\n"

        return string
