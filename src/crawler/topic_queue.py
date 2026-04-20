from dataclasses import dataclass
from typing import List
import json


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
    judge_refused: bool = None
    cluster_idx: int = None
    refusal_check_queries: List[str] = None
    refusal_check_responses: List[str] = None
    prompt: str = None
    summary: str = None
    api_refused_reason: str = None
    cluster_member_count: int = 0
    refusal_check_inconclusive: bool = False

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
            "judge_refused": self.judge_refused,
            "cluster_idx": self.cluster_idx,
            "parent_id": self.parent_id,
            "refusal_check_queries": self.refusal_check_queries,
            "refusal_check_responses": self.refusal_check_responses,
            "prompt": self.prompt,
            "summary": self.summary,
            "cluster_member_count": self.cluster_member_count,
            "refusal_check_inconclusive": self.refusal_check_inconclusive,
            "api_refused_reason": self.api_refused_reason,
        }


class TopicQueue:
    def __init__(self,
                 head_refusal_topics: List[Topic] = None,
                 ):
        # Track clusters
        self.head_topics: List[Topic] = []
        self.head_refusal_topics: List[Topic] = (
            list(head_refusal_topics) if head_refusal_topics else []
        )
        self.cluster_topics: List[List[Topic]] = []

        # Stats
        self.num_head_topics: int = 0
        self.num_topics_per_cluster: List[int] = []
        self.num_head_refusal_topics: int = 0
        self.num_total_topics: int = 0

    @property
    def head_refusal_topic_strings(self):
        # l = []
        # for t in self.head_refusal_topics:
        #     s = t.chinese if t.is_chinese else t.english
        #     l.append(s)
        l = [t.summary for t in self.head_refusal_topics]
        return l

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
        self.num_total_topics += 1
        return topic

    def append_to_cluster(self, topic: Topic) -> Topic:
        """Add a topic to an existing cluster.

        Raises ValueError loudly if cluster_idx is out of range or negative,
        rather than silently wrong-clustering (negative Python wrap) or raising
        an opaque IndexError.
        """
        cidx = topic.cluster_idx
        if cidx is None or cidx < 0 or cidx >= len(self.cluster_topics):
            raise ValueError(
                f"append_to_cluster: invalid cluster_idx={cidx!r} "
                f"(have {len(self.cluster_topics)} clusters). "
                f"Topic raw={topic.raw!r}. "
                "This indicates an unresolved cluster_idx from grouping_pipeline."
            )
        self.cluster_topics[cidx].append(topic)
        self.num_topics_per_cluster[cidx] += 1
        self.num_total_topics += 1
        return topic

    def incoming_batch(self, topics: List[Topic]) -> List[Topic]:
        """Process a batch of topics to be added to the queue with deduplication.

        Two-pass to stay order-independent: heads first (they create the cluster
        slot in cluster_topics), then non-heads (which require their cluster slot
        to exist). Without this, if the grouping pipeline returns a non-head
        member earlier in the list than its cluster's head, append_to_cluster
        hits IndexError on a not-yet-allocated cluster_idx.
        """
        if topics == []:
            print("No topics passed.")
            return []

        for topic in topics:
            if topic.is_head:
                topic.id = self.num_total_topics
                self.add_new_cluster_head(topic)

        for topic in topics:
            if not topic.is_head:
                topic.id = self.num_total_topics
                self.append_to_cluster(topic)
        return topics

    def refresh_refusal_membership(self):
        """Rebuild head_refusal_topics from head_topics.is_refusal.

        Needed when heads are added with is_refusal=None and later mutated
        in place by the refusal checker — otherwise head_refusal_topics
        stays stale from the add-time snapshot.
        """
        self.head_refusal_topics = [t for t in self.head_topics if t.is_refusal]
        self.num_head_refusal_topics = len(self.head_refusal_topics)

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

        return string
