#!/usr/bin/env python3
"""
Interactive script for manually clustering head refusal topics from crawler logs.
Allows assigning topics to existing or new clusters with navigation support.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def load_crawler_log_from_path(file_path: Path) -> dict:
    """Load crawler log data from a specific file path."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    return data


def load_existing_clusters(clusters_file: Path) -> Dict[str, List[dict]]:
    """Load existing clusters from JSON file."""
    if not clusters_file.exists():
        return {}

    try:
        with open(clusters_file, "r") as f:
            clusters = json.load(f)
        return clusters
    except Exception as e:
        print(f"Warning: Could not load existing clusters: {e}")
        return {}


def save_clusters(clusters_file: Path, clusters: Dict[str, List[dict]]):
    """Save clusters to JSON file."""
    with open(clusters_file, "w") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved clusters to {clusters_file}")


def get_topic_id(topic: dict) -> Optional[int]:
    """Get unique identifier for a topic."""
    return topic.get("id")


def get_topic_display_text(topic: dict) -> str:
    """Get display text for a topic (prefer english, fallback to raw or summary)."""
    return topic.get("english") or topic.get("raw") or topic.get("summary") or "N/A"


def get_already_labeled_topic_ids(clusters: Dict[str, List[dict]]) -> Set[int]:
    """Get set of topic IDs that are already labeled in clusters."""
    labeled_ids = set()
    for cluster_topics in clusters.values():
        for topic in cluster_topics:
            topic_id = get_topic_id(topic)
            if topic_id is not None:
                labeled_ids.add(topic_id)
    return labeled_ids


def get_topic_cluster_name(
    clusters: Dict[str, List[dict]], topic_id: int
) -> Optional[str]:
    """Get the cluster name that contains a topic with the given ID."""
    for cluster_name, cluster_topics in clusters.items():
        for topic in cluster_topics:
            if get_topic_id(topic) == topic_id:
                return cluster_name
    return None


def display_current_topic(
    topic: dict, index: int, total: int, clusters: Dict[str, List[dict]] = None
):
    """Display information about the current topic."""
    print("\n" + "=" * 80)
    print(f"Topic {index + 1}/{total}")
    print("=" * 80)

    # Show current cluster assignment if any
    if clusters:
        topic_id = get_topic_id(topic)
        if topic_id is not None:
            current_cluster = get_topic_cluster_name(clusters, topic_id)
            if current_cluster:
                print(f"Currently assigned to: {current_cluster}")

    print()

    # Display English translation
    english = topic.get("english")
    if english:
        print(f"English: {english}")

    # Display raw text
    raw = topic.get("raw")
    if raw and raw != english:
        print(f"Raw: {raw}")

    # Display summary if different
    summary = topic.get("summary")
    if summary and summary != english and summary != raw:
        print(f"Summary: {summary}")

    # Display refusal responses
    refusal_responses = topic.get("refusal_check_responses")
    if refusal_responses:
        print("\nRefusal Responses:")
        for i, response in enumerate(refusal_responses, 1):
            # Truncate long responses
            response_str = str(response)
            if len(response_str) > 200:
                response_str = response_str[:200] + "..."
            print(f"  {i}. {response_str}")
    else:
        print("\nRefusal Responses: No refusal responses available")

    print()


def display_clusters(clusters: Dict[str, List[dict]]):
    """Display existing clusters with their topics."""
    if not clusters:
        print("Existing Clusters: None (start by creating a new cluster)")
        return

    print("Existing Clusters:")
    cluster_names = list(clusters.keys())
    for i, cluster_name in enumerate(cluster_names, 1):
        topic_count = len(clusters[cluster_name])
        # Show first topic as sample
        sample_topic = clusters[cluster_name][0]
        sample_text = get_topic_display_text(sample_topic)
        # Truncate if too long
        if len(sample_text) > 50:
            sample_text = sample_text[:50] + "..."
        print(
            f"  {i}. {cluster_name} ({topic_count} topic{'s' if topic_count != 1 else ''})"
        )
        print(f"     Sample: {sample_text}")
    print()


def display_menu(clusters: Dict[str, List[dict]]):
    """Display the input menu."""
    cluster_count = len(clusters)
    options = []

    if cluster_count > 0:
        options.append(f"[1-{cluster_count}] assign to cluster")
    options.append("[n]ew cluster")
    options.append("[b]ack")
    options.append("[s]kip")
    options.append("[q]uit")
    options.append("[?] help")

    print(" | ".join(options))
    print()


def show_help():
    """Display help information."""
    print("\n" + "=" * 80)
    print("HELP")
    print("=" * 80)
    print("Commands:")
    print("  1-N     : Assign current topic to cluster number N")
    print("  n, new  : Create a new cluster and assign current topic to it")
    print("  b, back : Go back to the previous topic")
    print("  s, skip : Skip current topic (don't assign to any cluster)")
    print("  q, quit : Save and exit")
    print("  ?, help : Show this help message")
    print("=" * 80 + "\n")


def create_new_cluster(clusters: Dict[str, List[dict]]) -> Optional[str]:
    """Prompt user to create a new cluster and return the cluster name."""
    while True:
        cluster_name = input("Enter new cluster name: ").strip()
        if not cluster_name:
            print("Error: Cluster name cannot be empty. Please try again.")
            continue
        if cluster_name in clusters:
            print(
                f"Error: Cluster '{cluster_name}' already exists. Please choose a different name."
            )
            continue
        return cluster_name


def assign_topic_to_cluster(
    clusters: Dict[str, List[dict]], topic: dict, cluster_name: str
):
    """Assign a topic to a cluster."""
    if cluster_name not in clusters:
        clusters[cluster_name] = []
    clusters[cluster_name].append(topic)
    print(f"✓ Assigned topic to cluster '{cluster_name}'")


def remove_topic_from_clusters(clusters: Dict[str, List[dict]], topic_id: int):
    """Remove a topic from all clusters (if it was previously assigned)."""
    for cluster_name, cluster_topics in list(clusters.items()):
        clusters[cluster_name] = [
            t for t in cluster_topics if get_topic_id(t) != topic_id
        ]
        # Remove empty clusters
        if not clusters[cluster_name]:
            del clusters[cluster_name]


def main():
    """Main interactive clustering loop."""
    parser = argparse.ArgumentParser(
        description="Manually cluster head refusal topics from a crawler log"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        required=True,
        help="Path to crawler log file (e.g., artifacts/baseline_eval/crawler_log_*.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for clusters (default: {input_filename}_manual_clusters.json)",
    )

    args = parser.parse_args()

    # Load crawler log
    file_path = Path(args.file)
    if not file_path.is_absolute():
        file_path = project_root / file_path

    print(f"Loading crawler log from: {file_path}")
    try:
        data = load_crawler_log_from_path(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Extract head refusal topics
    head_refusal_topics = (
        data.get("queue", {}).get("topics", {}).get("head_refusal_topics", [])
    )

    if not head_refusal_topics:
        print("No head refusal topics found in the crawler log.")
        return

    print(f"Found {len(head_refusal_topics)} head refusal topics")

    # Determine output file path
    if args.output:
        clusters_file = Path(args.output)
        if not clusters_file.is_absolute():
            clusters_file = project_root / clusters_file
    else:
        # Auto-generate filename
        input_stem = file_path.stem
        clusters_file = file_path.parent / f"{input_stem}_manual_clusters.json"

    print(f"Clusters will be saved to: {clusters_file}")

    # Load existing clusters
    clusters = load_existing_clusters(clusters_file)
    if clusters:
        total_labeled = sum(len(topics) for topics in clusters.values())
        print(f"Loaded {len(clusters)} existing clusters with {total_labeled} topics")

    # Get already labeled topic IDs
    labeled_ids = get_already_labeled_topic_ids(clusters)

    # Filter out already labeled topics
    unlabeled_topics = [
        topic for topic in head_refusal_topics if get_topic_id(topic) not in labeled_ids
    ]

    if not unlabeled_topics:
        print("\nAll topics have already been labeled!")
        print(f"Total clusters: {len(clusters)}")
        for cluster_name, topics in clusters.items():
            print(f"  {cluster_name}: {len(topics)} topics")
        return

    print(f"\n{len(unlabeled_topics)} topics remaining to label")

    # Interactive loop
    current_index = 0
    history = []  # Track navigation history

    try:
        while current_index < len(unlabeled_topics):
            topic = unlabeled_topics[current_index]
            topic_id = get_topic_id(topic)

            # Display current topic
            display_current_topic(topic, current_index, len(unlabeled_topics), clusters)

            # Display existing clusters
            display_clusters(clusters)

            # Display menu
            display_menu(clusters)

            # Get user input
            user_input = input("> ").strip().lower()

            if user_input in ["q", "quit"]:
                save_clusters(clusters_file, clusters)
                print("\nExiting. Goodbye!")
                break

            elif user_input in ["?", "help"]:
                show_help()
                continue

            elif user_input in ["b", "back"]:
                if history:
                    current_index = history.pop()
                    print(f"\n← Going back to topic {current_index + 1}")
                else:
                    print("\n⚠ Cannot go back further (at the beginning)")
                continue

            elif user_input in ["s", "skip"]:
                print("\n⊘ Skipped topic")
                history.append(current_index)
                current_index += 1
                continue

            elif user_input in ["n", "new"]:
                cluster_name = create_new_cluster(clusters)
                if cluster_name:
                    # Remove topic from any existing cluster first
                    if topic_id is not None:
                        remove_topic_from_clusters(clusters, topic_id)
                    assign_topic_to_cluster(clusters, topic, cluster_name)
                    save_clusters(clusters_file, clusters)
                    history.append(current_index)
                    current_index += 1

            elif user_input.isdigit():
                cluster_num = int(user_input)
                cluster_names = list(clusters.keys())
                if 1 <= cluster_num <= len(cluster_names):
                    cluster_name = cluster_names[cluster_num - 1]
                    # Remove topic from any existing cluster first
                    if topic_id is not None:
                        remove_topic_from_clusters(clusters, topic_id)
                    assign_topic_to_cluster(clusters, topic, cluster_name)
                    save_clusters(clusters_file, clusters)
                    history.append(current_index)
                    current_index += 1
                else:
                    print(
                        f"\n⚠ Invalid cluster number. Please enter 1-{len(cluster_names)}"
                    )
                    continue

            else:
                print("\n⚠ Invalid input. Type '?' for help.")
                continue

        # End of topics
        if current_index >= len(unlabeled_topics):
            print("\n" + "=" * 80)
            print("All topics have been processed!")
            print("=" * 80)
            print(f"\nTotal clusters: {len(clusters)}")
            for cluster_name, topics in clusters.items():
                print(f"  {cluster_name}: {len(topics)} topics")
            save_clusters(clusters_file, clusters)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        save_clusters(clusters_file, clusters)
        print("Progress saved.")


if __name__ == "__main__":
    main()
