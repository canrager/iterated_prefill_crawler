#!/usr/bin/env python3
"""
Web-based drag-and-drop interface for manually clustering head refusal topics.
"""

import argparse
import json
import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from flask import Flask, render_template, request, jsonify, send_from_directory

# Set environment variable to force spawn method before any imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import multiprocessing

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.llm_utils import load_vllm_model
from src.directory_config import resolve_cache_dir
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

app = Flask(
    __name__,
    template_folder=str(script_dir / "templates"),
    static_folder=str(script_dir / "static"),
)

# Global state
_crawler_log_path: Optional[Path] = None
_clusters_file: Optional[Path] = None
_topics: List[dict] = []
_clusters: Dict[str, List[dict]] = {}

# Global LLM state
_vllm_model: Optional[LLM] = None
_vllm_tokenizer: Optional[AutoTokenizer] = None
_vllm_model_name: Optional[str] = None


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


def load_vllm_judge_model(
    model_name: str,
    cache_dir: Optional[str] = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
) -> Tuple[LLM, AutoTokenizer]:
    """
    Load vLLM model and tokenizer for use as LLM judge.

    Args:
        model_name: HuggingFace model name (e.g., "allenai/Olmo-3-7B-Think")
        cache_dir: Directory to cache the model
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory fraction to use
        max_model_len: Maximum sequence length

    Returns:
        Tuple of (LLM model instance, tokenizer)
    """
    global _vllm_model, _vllm_model_name, _vllm_tokenizer

    # Return cached model if already loaded
    if _vllm_model is not None and _vllm_model_name == model_name:
        return _vllm_model, _vllm_tokenizer

    cache_dir_path = resolve_cache_dir(cache_dir) if cache_dir else None
    cache_dir_str = str(cache_dir_path) if cache_dir_path else None

    print(f"Loading vLLM model: {model_name}")
    _vllm_model, _vllm_tokenizer = load_vllm_model(
        model_name=model_name,
        cache_dir=cache_dir_str,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    _vllm_model_name = model_name
    print(f"vLLM model loaded successfully")

    return _vllm_model, _vllm_tokenizer


def batch_query_vllm(
    prompts: List[str],
    model: LLM,
    tokenizer: Optional[AutoTokenizer] = None,
    max_tokens: int = 2000,
    temperature: float = 0.7,
) -> List[str]:
    """
    Query vLLM model with a batch of prompts.

    Args:
        prompts: List of user prompts (raw text, will be formatted with chat template)
        model: vLLM LLM instance
        tokenizer: Optional tokenizer for applying chat template
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        List of generated response texts
    """
    if not prompts:
        return []

    # Apply chat template if tokenizer is provided
    formatted_prompts = prompts
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            formatted_prompts = []
            for prompt in prompts:
                user_message = prompt.strip()
                messages = [{"role": "user", "content": user_message}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                formatted_prompts.append(formatted_prompt)
        except Exception as e:
            print(f"Warning: Could not apply chat template: {e}")
            formatted_prompts = prompts

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        skip_special_tokens=True,
    )

    # Generate directly from formatted prompts
    outputs = model.generate(
        prompts=formatted_prompts,
        sampling_params=sampling_params,
    )

    # Extract generated text from vLLM outputs
    generated_texts = [output.outputs[0].text for output in outputs]

    return generated_texts


def build_cluster_suggestion_prompt(topic_raw: str, cluster_names: List[str]) -> str:
    """
    Build prompt for LLM to suggest which cluster a topic belongs to.

    Args:
        topic_raw: The raw topic text
        cluster_names: List of existing cluster names

    Returns:
        Formatted prompt string
    """
    if not cluster_names:
        return ""

    # Build cluster list
    cluster_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(cluster_names)])

    prompt = f"""You are helping to cluster topics. Given a new topic and a list of existing clusters, determine which cluster the topic belongs to.

Topic: {topic_raw}

Existing Clusters:
{cluster_list}

Think through which cluster this topic best fits into. Consider semantic similarity, thematic coherence, and the purpose of each cluster.

Example 1:
Topic: "hacking computer systems"
Existing Clusters:
1. Security Topics
2. Political Discussions
3. Ethical Concerns

<think>
The topic "hacking computer systems" is clearly related to security and cybersecurity. It involves unauthorized access to systems, which is a security concern. While it could be seen as an ethical concern, the primary category is security-related. This best fits "Security Topics".
</think>
\\answer{{"Security Topics"}}

Example 2:
Topic: "2024 election results"
Existing Clusters:
1. Security Topics
2. Political Discussions
3. Ethical Concerns

<think>
The topic "2024 election results" is directly about political events and electoral processes. This is clearly a political topic, not related to security or ethics in the primary sense. It belongs in "Political Discussions".
</think>
\\answer{{"Political Discussions"}}

Now analyze the given topic and provide your reasoning followed by the answer."""

    return prompt


def parse_answer_from_response(response: str) -> Optional[str]:
    """
    Parse cluster name from LLM response in format \\answer{"cluster_name"}.

    Args:
        response: LLM response text

    Returns:
        Cluster name if found, None otherwise
    """
    # Look for \answer{"cluster_name"} pattern
    # Handle both escaped and unescaped backslashes
    patterns = [
        r'\\answer\{"([^"]+)"\}',  # Escaped backslash
        r'\answer\{"([^"]+)"\}',  # Unescaped backslash
        r'answer\{"([^"]+)"\}',  # Without backslash
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)

    # Fallback: look for any quoted string after "answer"
    match = re.search(r'answer.*?"([^"]+)"', response, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


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

    if not file_path.is_absolute():
        file_path = project_root / file_path

    try:
        crawler_data = load_crawler_log_from_path(file_path)
        _crawler_log_path = file_path

        # Extract head refusal topics
        head_refusal_topics = (
            crawler_data.get("queue", {}).get("topics", {}).get("head_refusal_topics", [])
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
        unlabeled_topics = [topic for topic in _topics if get_topic_id(topic) not in labeled_ids]

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


@app.route("/api/suggest", methods=["POST"])
def suggest_cluster():
    """Get LLM suggestion for which cluster a topic should be assigned to."""
    global _vllm_model, _vllm_tokenizer, _topics, _clusters

    # Check if LLM is loaded
    if _vllm_model is None or _vllm_tokenizer is None:
        return (
            jsonify(
                {"error": "LLM model not loaded. Please start the server with --vllm_model_name"}
            ),
            400,
        )

    data = request.json
    topic_id = data.get("topic_id")

    if topic_id is None:
        return jsonify({"error": "topic_id is required"}), 400

    # Find the topic
    topic = None
    for t in _topics:
        if get_topic_id(t) == topic_id:
            topic = t
            break

    if not topic:
        return jsonify({"error": f"Topic with id {topic_id} not found"}), 404

    # Get topic raw text
    topic_raw = topic.get("raw") or topic.get("english") or topic.get("summary") or ""
    if not topic_raw:
        return jsonify({"error": "Topic has no text content"}), 400

    # Get all cluster names
    cluster_names = list(_clusters.keys())
    if not cluster_names:
        return (
            jsonify({"error": "No clusters exist yet. Create at least one cluster first."}),
            400,
        )

    try:
        # Build prompt
        prompt = build_cluster_suggestion_prompt(topic_raw, cluster_names)

        # Query LLM
        responses = batch_query_vllm(
            prompts=[prompt],
            model=_vllm_model,
            tokenizer=_vllm_tokenizer,
            max_tokens=2000,
            temperature=0.7,
        )

        if not responses:
            return jsonify({"error": "No response from LLM"}), 500

        response_text = responses[0]

        # Parse answer
        suggested_cluster = parse_answer_from_response(response_text)

        if not suggested_cluster:
            return (
                jsonify(
                    {
                        "error": "Could not parse cluster name from LLM response",
                        "response": response_text,
                    }
                ),
                500,
            )

        # Validate cluster exists
        if suggested_cluster not in cluster_names:
            # Try case-insensitive match
            cluster_lower = {name.lower(): name for name in cluster_names}
            if suggested_cluster.lower() in cluster_lower:
                suggested_cluster = cluster_lower[suggested_cluster.lower()]
            else:
                return (
                    jsonify(
                        {
                            "error": f"LLM suggested cluster '{suggested_cluster}' which does not exist",
                            "available_clusters": cluster_names,
                            "response": response_text,
                        }
                    ),
                    400,
                )

        return jsonify(
            {
                "success": True,
                "suggested_cluster": suggested_cluster,
                "response": response_text,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Error getting LLM suggestion: {str(e)}"}), 500


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Web-based drag-and-drop interface for clustering head refusal topics"
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
    parser.add_argument(
        "--vllm_model_name",
        type=str,
        default=None,
        help="vLLM model name for LLM suggestions (e.g., 'allenai/Olmo-3-7B-Think')",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory fraction to use (default: 0.9)",
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=None,
        help="Maximum sequence length for vLLM (default: model default)",
    )

    args = parser.parse_args()

    # Load LLM model if provided
    if args.vllm_model_name:
        try:
            load_vllm_judge_model(
                model_name=args.vllm_model_name,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_len=args.vllm_max_model_len,
            )
            print(f"LLM model loaded: {args.vllm_model_name}")
        except Exception as e:
            print(f"Warning: Could not load LLM model: {e}")
            print("LLM suggestions will not be available")

    # Pre-load file if provided
    if args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = project_root / file_path

        try:
            crawler_data = load_crawler_log_from_path(file_path)
            _crawler_log_path = file_path

            head_refusal_topics = (
                crawler_data.get("queue", {}).get("topics", {}).get("head_refusal_topics", [])
            )

            if head_refusal_topics:
                global _topics, _clusters, _clusters_file
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
