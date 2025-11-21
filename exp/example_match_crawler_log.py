#!/usr/bin/env python3
"""
Example script demonstrating how to match crawler logs directly with ground truth topics.

This script shows how to use match_crawler_log_with_gt() to match topics from a crawler log
file (without needing to run ranking first) against ground truth topics.

Usage:
    python exp/example_match_crawler_log.py
"""

import os
from core.safety_topic_ranker_matcher import match_crawler_log_with_gt
from core.project_config import INTERIM_DIR

def main():
    # Configuration
    run_title = "tulu-3-1120-example"
    crawler_log_filename = "crawler_log_20251120_112924_Llama-3.1-Tulu-3-8B-SFT_1samples_1000crawls_Truefilter_user_seedingprompt_vllm.json"
    crawler_log_path = os.path.join(INTERIM_DIR, crawler_log_filename)

    # Ground truth files to match against
    gt_topics_files = {
        "Tulu Safety": "tulu3_ground_truth_safety_topics",
        # "CCP censorship": "censorship_topics",  # Uncomment to include additional datasets
    }

    # LLM judge to use for semantic matching
    llm_judge_name = "gpt-4o"

    # Run matching
    print(f"Matching crawler log: {crawler_log_filename}")
    print(f"Against ground truth datasets: {list(gt_topics_files.keys())}")
    print(f"Using LLM judge: {llm_judge_name}\n")

    matched_topics, gt_first_occurrences = match_crawler_log_with_gt(
        run_title=run_title,
        crawler_log_path=crawler_log_path,
        gt_topics_files=gt_topics_files,
        llm_judge_name=llm_judge_name,
        verbose=False,  # Set to True to see detailed matching process
        force_recompute=True,  # Set to False to use cached results
        debug=False  # Set to True to limit to first 3 topics per category for testing
    )

    print("\n" + "="*80)
    print("MATCHING COMPLETE")
    print("="*80)

    # Print some example results
    print("\nExample matched topics:")
    matched_count = 0
    for summary, data in matched_topics.items():
        if data["is_match"] and matched_count < 5:
            print(f"\nCrawled topic (ID {data['id']}): {summary}")
            print(f"  Raw: {data['raw']}")
            print(f"  English: {data['english']}")
            print(f"  Matched GT topics: {', '.join(data['ground_truth_matches'])}")
            matched_count += 1

    print("\n" + "="*80)
    print("Output files saved to artifacts/result/:")
    print(f"  - crawler_log_matched_{run_title}.json")
    print(f"  - unmatched_crawler_topics_{run_title}.json")
    print(f"  - crawler_log_match_summary_{run_title}.json")
    print("="*80)

if __name__ == "__main__":
    main()
