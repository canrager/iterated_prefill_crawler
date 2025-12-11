#!/usr/bin/env python3
"""
Simple script to print the head refusal topics loaded from the latest crawler log.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir
sys.path.insert(0, str(project_root))

from exp.load_latest_crawler_log import find_latest_crawler_log


def load_crawler_log_from_path(file_path: Path) -> dict:
    """Load crawler log data from a specific file path."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    return data


def main():
    """Load the latest crawler log and print head refusal topics."""
    parser = argparse.ArgumentParser(
        description="Print head refusal topics from a crawler log file"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="artifacts/baseline_eval/crawler_log_20251127_083333_DeepSeek-R1-Distill-Llama-8B_1samples_100crawls_Truefilter_thought_prefill_with_seedprompt_vllm.json",
        help="Path to a specific crawler log file (e.g., artifacts/baseline_eval/crawler_log_*.json)",
    )

    args = parser.parse_args()

    # If a file path is provided, load it directly
    if args.file:
        file_path = Path(args.file)
        # Handle relative paths relative to project root
        if not file_path.is_absolute():
            file_path = project_root / file_path

        print(f"Loading crawler log from: {file_path}")
        try:
            data = load_crawler_log_from_path(file_path)
        except Exception as e:
            print(f"Error loading file: {e}")
            return
    else:
        # Use the same config filters as in load_latest_crawler_log.py main()
        config_filters = {
            "prefill_mode": "user_prefill_with_seed",
            "model_path": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        }

        # Find the latest matching crawler log
        result = find_latest_crawler_log(config_filters=config_filters)

        if result is None:
            print("No matching crawler log found.")
            return

        print(f"Loaded crawler log: {result.get('filename', 'N/A')}")
        data = result.get("data", {})

    # Extract head refusal topics
    head_refusal_topics = (
        data.get("queue", {}).get("topics", {}).get("head_refusal_topics", [])
    )

    if not head_refusal_topics:
        print("No head refusal topics found in the crawler log.")
        return

    print(f"Found {len(head_refusal_topics)} head refusal topics")
    print("=" * 80)
    print("\nHead Refusal Topics:\n")

    # Print each topic
    for i, topic in enumerate(head_refusal_topics, 1):
        summary = (
            topic.get("summary") or topic.get("english") or topic.get("raw", "N/A")
        )
        print(f"{i}. {summary}")


if __name__ == "__main__":
    main()
