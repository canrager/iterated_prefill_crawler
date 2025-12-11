#!/usr/bin/env python3
"""
Offline web-based drag-and-drop interface for manually clustering head refusal topics.
Minimal standalone version without LLM dependencies.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, render_template, request, jsonify

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

# Global state
_crawler_log_path: Optional[Path] = None
_clusters_file: Optional[Path] = None
_topics: List[dict] = []
_clusters: Dict[str, List[dict]] = {}


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


def get_topic_id(topic: dict) -> Optional[int]:
    """Get unique identifier for a topic."""
    return topic.get("id")


@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("cluster_topics.html")


@app.route("/api/init", methods=["POST"])
def init():
    """Initialize the application with crawler log file."""
    global _topics, _clusters, _crawler_log_path, _clusters_file

    data = request.json
    file_path = Path(data.get("file_path", ""))

    # If relative path, try relative to data directory first, then current directory
    if not file_path.is_absolute():
        data_dir = Path(__file__).parent / "data"
        if (data_dir / file_path).exists():
            file_path = data_dir / file_path
        else:
            file_path = Path.cwd() / file_path

    try:
        crawler_data = load_crawler_log_from_path(file_path)
        _crawler_log_path = file_path

        # Extract head refusal topics
        head_refusal_topics = (
            crawler_data.get("queue", {})
            .get("topics", {})
            .get("head_refusal_topics", [])
        )

        if not head_refusal_topics:
            return (
                jsonify({"error": "No head refusal topics found in the crawler log."}),
                400,
            )

        _topics = head_refusal_topics

        # Determine clusters file path
        input_stem = file_path.stem
        _clusters_file = file_path.parent / f"{input_stem}_manual_clusters.json"

        # Load existing clusters
        _clusters = load_existing_clusters(_clusters_file)

        # Get already labeled topic IDs
        labeled_ids = set()
        for cluster_topics in _clusters.values():
            for topic in cluster_topics:
                topic_id = get_topic_id(topic)
                if topic_id is not None:
                    labeled_ids.add(topic_id)

        # Filter topics
        unlabeled_topics = [
            topic for topic in _topics if get_topic_id(topic) not in labeled_ids
        ]

        return jsonify(
            {
                "success": True,
                "total_topics": len(_topics),
                "unlabeled_count": len(unlabeled_topics),
                "labeled_count": len(_topics) - len(unlabeled_topics),
                "clusters_file": str(_clusters_file),
                "clusters": _clusters,
                "topics": _topics,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/clusters", methods=["GET"])
def get_clusters():
    """Get current clusters."""
    return jsonify({"clusters": _clusters})


@app.route("/api/clusters", methods=["POST"])
def update_clusters():
    """Update clusters (assign topic to cluster or create new cluster)."""
    global _clusters

    data = request.json
    action = data.get("action")

    if action == "assign":
        topic_id = data.get("topic_id")
        cluster_name = data.get("cluster_name")

        if not cluster_name:
            return jsonify({"error": "Cluster name is required"}), 400

        # Find the topic
        topic = None
        for t in _topics:
            if get_topic_id(t) == topic_id:
                topic = t
                break

        if not topic:
            return jsonify({"error": f"Topic with id {topic_id} not found"}), 404

        # Remove topic from any existing cluster
        for cluster_name_existing, cluster_topics in list(_clusters.items()):
            _clusters[cluster_name_existing] = [
                t for t in cluster_topics if get_topic_id(t) != topic_id
            ]
            if not _clusters[cluster_name_existing]:
                del _clusters[cluster_name_existing]

        # Add to new cluster
        if cluster_name not in _clusters:
            _clusters[cluster_name] = []
        _clusters[cluster_name].append(topic)

        # Save
        save_clusters(_clusters_file, _clusters)

        return jsonify({"success": True, "clusters": _clusters})

    elif action == "remove":
        topic_id = data.get("topic_id")
        cluster_name = data.get("cluster_name")

        if cluster_name in _clusters:
            _clusters[cluster_name] = [
                t for t in _clusters[cluster_name] if get_topic_id(t) != topic_id
            ]
            if not _clusters[cluster_name]:
                del _clusters[cluster_name]

        save_clusters(_clusters_file, _clusters)
        return jsonify({"success": True, "clusters": _clusters})

    elif action == "rename":
        old_name = data.get("old_name")
        new_name = data.get("new_name")

        if not new_name:
            return jsonify({"error": "New cluster name is required"}), 400

        if new_name in _clusters:
            return jsonify({"error": "Cluster name already exists"}), 400

        if old_name in _clusters:
            # Rebuild dictionary to maintain order while renaming
            new_clusters = {}
            for k, v in _clusters.items():
                if k == old_name:
                    new_clusters[new_name] = v
                else:
                    new_clusters[k] = v
            _clusters = new_clusters
            save_clusters(_clusters_file, _clusters)

        return jsonify({"success": True, "clusters": _clusters})

    elif action == "delete":
        cluster_name = data.get("cluster_name")
        if cluster_name in _clusters:
            del _clusters[cluster_name]
            save_clusters(_clusters_file, _clusters)

        return jsonify({"success": True, "clusters": _clusters})

    return jsonify({"error": "Invalid action"}), 400


@app.route("/api/topics", methods=["GET"])
def get_topics():
    """Get all topics."""
    return jsonify({"topics": _topics})


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Offline web-based drag-and-drop interface for clustering head refusal topics"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Path to crawler log file (optional - can be loaded via UI)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=5000,
        help="Port to run the web server on (default: 5000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )

    args = parser.parse_args()

    # Pre-load file if provided
    if args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            data_dir = Path(__file__).parent / "data"
            if (data_dir / file_path).exists():
                file_path = data_dir / file_path
            else:
                file_path = Path.cwd() / file_path

        try:
            crawler_data = load_crawler_log_from_path(file_path)
            global _crawler_log_path, _topics, _clusters, _clusters_file
            _crawler_log_path = file_path

            head_refusal_topics = (
                crawler_data.get("queue", {})
                .get("topics", {})
                .get("head_refusal_topics", [])
            )

            if head_refusal_topics:
                _topics = head_refusal_topics
                input_stem = file_path.stem
                _clusters_file = file_path.parent / f"{input_stem}_manual_clusters.json"
                _clusters = load_existing_clusters(_clusters_file)
                print(f"Pre-loaded {len(_topics)} topics from {file_path}")
                print(f"Found {len(_clusters)} existing clusters")

        except Exception as e:
            print(f"Warning: Could not pre-load file: {e}")

    print(f"\nStarting web server at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server\n")

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
