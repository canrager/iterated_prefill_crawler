#!/usr/bin/env python3
"""Test the baseline log filename parsing."""

import re
import glob
import os
from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict


def parse_crawler_log_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse crawler log filename to extract model and prefill method.

    Format: crawler_log_YYYYMMDD_HHMMSS_MODEL_Nsamples_Mcrawls_TRUEFILTERfilter_PREFILL_METHOD_BACKEND.json
    """
    basename = filename.replace(".json", "")
    # Pattern: crawler_log_timestamp_model_samples_crawls_filter_prefill_backend
    pattern = r"crawler_log_(\d{8})_(\d{6})_(.+?)_(\d+)samples_(\d+)crawls_(.+?)filter_(.+?)_vllm"

    match = re.match(pattern, basename)
    if not match:
        return None

    (
        timestamp_date,
        timestamp_time,
        model,
        samples,
        crawls,
        filter_type,
        prefill_method,
    ) = match.groups()

    return {
        "model": model,
        "prefill_method": prefill_method,
        "timestamp": f"{timestamp_date}_{timestamp_time}",
        "filename": filename,
    }


def group_logs_by_model(baseline_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    """
    Group crawler logs by model name.

    Args:
        baseline_dir: Directory containing baseline evaluation logs

    Returns:
        Dictionary mapping model name to list of log file info dicts
    """
    log_files = glob.glob(str(baseline_dir / "crawler_log_*.json"))

    model_groups = defaultdict(list)

    for log_file in log_files:
        filename = os.path.basename(log_file)
        parsed = parse_crawler_log_filename(filename)

        if parsed:
            model_groups[parsed["model"]].append(
                {
                    "path": log_file,
                    "filename": filename,
                    "prefill_method": parsed["prefill_method"],
                    "timestamp": parsed["timestamp"],
                }
            )
        else:
            print(f"Warning: Could not parse filename: {filename}")

    return dict(model_groups)


# Test filenames
test_filenames = [
    "crawler_log_20251127_083333_DeepSeek-R1-Distill-Llama-8B_1samples_100crawls_Truefilter_thought_prefill_with_seedprompt_vllm.json",
    "crawler_log_20251127_113804_DeepSeek-R1-Distill-Llama-8B_1samples_100crawls_Truefilter_user_prefill_with_seedprompt_vllm.json",
    "crawler_log_20251127_200627_Llama-3.1-Tulu-3-8B-SFT_1samples_100crawls_Truefilter_assistant_prefill_with_seedprompt_vllm.json",
    "crawler_log_20251128_103118_Llama-3.1-8B-Instruct_1samples_100crawls_Truefilter_assistant_prefill_with_seedprompt_vllm.json",
    "crawler_log_20251128_141522_Llama-3.1-Tulu-3-8B-SFT_1samples_100crawls_Truefilter_user_prefill_with_seedprompt_vllm.json",
]

print("Testing filename parsing:")
print("=" * 80)

for filename in test_filenames:
    result = parse_crawler_log_filename(filename)
    if result:
        print(f"\n{filename}")
        print(f"  Model: {result['model']}")
        print(f"  Prefill: {result['prefill_method']}")
        print(f"  Timestamp: {result['timestamp']}")
    else:
        print(f"\nFAILED: {filename}")

# Test grouping
print("\n" + "=" * 80)
print("Testing grouping:")
print("=" * 80)

baseline_dir = Path("/home/can/iterated_prefill_crawler/artifacts/baseline_eval")
if baseline_dir.exists():
    model_groups = group_logs_by_model(baseline_dir)
    print(f"\nFound {len(model_groups)} unique models:")
    for model_name, log_files in model_groups.items():
        print(f"\n{model_name}:")
        for log_info in log_files:
            print(f"  - {log_info['prefill_method']}: {log_info['filename']}")
else:
    print(f"Directory not found: {baseline_dir}")


