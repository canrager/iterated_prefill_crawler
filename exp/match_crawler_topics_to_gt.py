#!/usr/bin/env python3
"""
Example script demonstrating how to match crawler logs directly with ground truth topics.

This script processes all crawler log files in /artifacts/threshold_ablation sequentially,
matching topics from each crawler log file against ground truth topics.

Usage:
    python exp/match_crawler_topics_to_gt.py
"""

import os
import glob
from pathlib import Path
from core.safety_topic_ranker_matcher import match_crawler_log_with_gt


def main():
    # Configuration
    threshold_ablation_dir = (
        "/home/can/iterated_prefill_crawler/artifacts/threshold_ablation"
    )

    # Ground truth files to match against
    gt_topics_files = {
        "Tulu Safety": "tulu3_ground_truth_safety_topics",
        # "CCP censorship": "censorship_topics",  # Uncomment to include additional datasets
    }

    # LLM judge to use for semantic matching
    llm_judge_name = "gpt-5-nano"

    # Find all JSON files in threshold_ablation directory
    crawler_log_files = sorted(
        glob.glob(os.path.join(threshold_ablation_dir, "*.json"))
    )

    if not crawler_log_files:
        print(f"No JSON files found in {threshold_ablation_dir}")
        return

    print(f"Found {len(crawler_log_files)} crawler log files to process")
    print(f"Against ground truth datasets: {list(gt_topics_files.keys())}")
    print(f"Using LLM judge: {llm_judge_name}\n")
    print("=" * 80)

    # Process each file sequentially
    for idx, crawler_log_path in enumerate(crawler_log_files, 1):
        crawler_log_filename = os.path.basename(crawler_log_path)
        # Generate run_title from filename (remove .json extension)
        run_title = os.path.splitext(crawler_log_filename)[0]

        print(f"\n[{idx}/{len(crawler_log_files)}] Processing: {crawler_log_filename}")
        print(f"Run title: {run_title}")
        print("-" * 80)

        try:
            matched_topics, gt_first_occurrences = match_crawler_log_with_gt(
                run_title=run_title,
                crawler_log_path=crawler_log_path,
                gt_topics_files=gt_topics_files,
                llm_judge_name=llm_judge_name,
                verbose=False,  # Set to True to see detailed matching process
                force_recompute=True,  # Set to False to use cached results
                debug=False,  # Set to True to limit to first 3 topics per category for testing
            )

            # Print some example results
            print("\nExample matched topics:")
            matched_count = 0
            for summary, data in matched_topics.items():
                if data["is_match"] and matched_count < 3:
                    print(f"\nCrawled topic (ID {data['id']}): {summary}")
                    print(f"  Raw: {data['raw']}")
                    print(f"  English: {data['english']}")
                    print(
                        f"  Matched GT topics: {', '.join(data['ground_truth_matches'])}"
                    )
                    matched_count += 1

            print(f"\nOutput files saved to artifacts/result/:")
            print(f"  - crawler_log_matched_{run_title}.json")
            print(f"  - unmatched_crawler_topics_{run_title}.json")
            print(f"  - crawler_log_match_summary_{run_title}.json")

        except Exception as e:
            print(f"ERROR processing {crawler_log_filename}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("ALL FILES PROCESSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
