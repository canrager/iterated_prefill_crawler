#!/usr/bin/env python3
"""
Script to load the latest crawler_log from /artifacts/interim based on CrawlerConfig attributes.

The script filters crawler log files by matching any CrawlerConfig attribute and returns
the latest one (by datetime in filename).

Filename pattern: crawler_log_<YYYYMMDD>_<HHMMSS>_<model>_<samples>samples_<crawls>crawls_<filter>filter_<prefill_mode>prompt_<backend>.json

Usage:
    Edit the config_filters dictionary in main() to specify which CrawlerConfig attributes
    to filter by, then run:

    python exp/load_latest_crawler_log.py

    Or use programmatically:

    from exp.load_latest_crawler_log import find_latest_crawler_log
    result = find_latest_crawler_log(config_filters={'prefill_mode': 'user_prefill_with_seed', 'model_path': 'allenai/Tulu'})
    data = result['data']
"""

import os
import sys
import json
import re
import shutil
from datetime import datetime
from typing import Dict, Optional, Any, List
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from configs.directory_config import INTERIM_DIR


def parse_crawler_log_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parse a crawler log filename to extract attributes.

    Pattern: crawler_log_<YYYYMMDD>_<HHMMSS>_<model>_<samples>samples_<crawls>crawls_<filter>filter_<prefill_mode>prompt_<backend>.json

    Returns:
        Dictionary with parsed attributes or None if parsing fails
    """
    # Remove .json extension
    basename = filename.replace(".json", "")

    # Pattern: crawler_log_YYYYMMDD_HHMMSS_model_rest...
    pattern = r"crawler_log_(\d{8})_(\d{6})_(.+?)_(\d+)samples_(\d+)crawls_(.+?)filter_(.+?)prompt_(.+)"

    match = re.match(pattern, basename)
    if not match:
        return None

    date_str, time_str, model, samples, crawls, filter_val, prefill_mode, backend = (
        match.groups()
    )

    # Parse datetime
    try:
        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None

    return {
        "filename": filename,
        "datetime": dt,
        "model": model,
        "samples": int(samples),
        "crawls": int(crawls),
        "filter": filter_val,
        "prefill_mode": prefill_mode,
        "backend": backend,
        "date_str": date_str,
        "time_str": time_str,
    }


def find_all_matching_crawler_logs(
    config_filters: Optional[Dict[str, Any]] = None,
    interim_dir: Optional[Path] = None,
    lazy_load: bool = True,
) -> List[Dict[str, Any]]:
    """
    Find all crawler logs matching the specified CrawlerConfig attributes, sorted by datetime (latest first).

    Returns a list of matching file info dictionaries (before loading JSON data).
    """
    if interim_dir is None:
        interim_dir = INTERIM_DIR

    if config_filters is None:
        config_filters = {}

    # Normalize model name for matching (handle both full paths and short names)
    model_normalized = None
    if "model_path" in config_filters:
        model = config_filters["model_path"]
        model_normalized = model.split("/")[-1] if "/" in model else model

    matching_files = []

    # Iterate through all files in interim directory
    for filename in os.listdir(interim_dir):
        if not filename.startswith("crawler_log_") or not filename.endswith(".json"):
            continue

        parsed = parse_crawler_log_filename(filename)
        if parsed is None:
            continue

        # Fast filtering on filename attributes (if lazy_load is True)
        if lazy_load:
            # Apply filename-based filters first for speed
            skip_file = False

            if (
                "prefill_mode" in config_filters
                and parsed["prefill_mode"] != config_filters["prefill_mode"]
            ):
                skip_file = True

            if not skip_file and model_normalized:
                parsed_model = parsed["model"]
                if (
                    model_normalized not in parsed_model
                    and parsed_model not in model_normalized
                ):
                    parsed_model_last = (
                        parsed_model.split("/")[-1]
                        if "/" in parsed_model
                        else parsed_model
                    )
                    model_normalized_last = (
                        model_normalized.split("/")[-1]
                        if "/" in model_normalized
                        else model_normalized
                    )
                    if parsed_model_last != model_normalized_last:
                        skip_file = True

            if (
                not skip_file
                and "num_samples_per_topic" in config_filters
                and parsed["samples"] != config_filters["num_samples_per_topic"]
            ):
                skip_file = True

            if (
                not skip_file
                and "num_crawl_steps" in config_filters
                and parsed["crawls"] != config_filters["num_crawl_steps"]
            ):
                skip_file = True

            if not skip_file and "do_filter_refusals" in config_filters:
                filter_bool = config_filters["do_filter_refusals"]
                filter_str = (
                    str(filter_bool)
                    if isinstance(filter_bool, bool)
                    else str(filter_bool)
                )
                if parsed["filter"] != filter_str:
                    skip_file = True

            if (
                not skip_file
                and "backend" in config_filters
                and config_filters["backend"] not in parsed["backend"]
            ):
                skip_file = True

            if skip_file:
                continue

        # Store parsed info for later (we need datetime for sorting)
        matching_files.append(
            {"filename": filename, "parsed": parsed, "filepath": interim_dir / filename}
        )

    if not matching_files:
        return []

    # Sort by datetime (most recent first)
    matching_files.sort(key=lambda x: x["parsed"]["datetime"], reverse=True)
    return matching_files


def load_crawler_log_from_file_info(
    file_info: Dict[str, Any],
    config_filters: Dict[str, Any],
    lazy_load: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Load and validate a crawler log from file_info, checking config filters.

    Returns the loaded data dictionary if it matches all config filters, None otherwise.
    """
    filepath = file_info["filepath"]
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None

    config = data.get("config", {})

    # Check all config filters
    matches_all = True
    for attr_name, expected_value in config_filters.items():
        # Skip filename-based filters we already checked
        if lazy_load and attr_name in [
            "prefill_mode",
            "model_path",
            "num_samples_per_topic",
            "num_crawl_steps",
            "do_filter_refusals",
            "backend",
        ]:
            # Special handling for model_path - check both model_path and parsed model name
            if attr_name == "model_path":
                config_model = config.get("model_path", "")
                parsed_model = file_info["parsed"]["model"]
                expected_model = (
                    expected_value.split("/")[-1]
                    if "/" in expected_value
                    else expected_value
                )
                config_model_last = (
                    config_model.split("/")[-1] if "/" in config_model else config_model
                )
                if (
                    expected_model not in config_model_last
                    and expected_model not in parsed_model
                ):
                    matches_all = False
                    break
            continue

        # Get actual value from config
        actual_value = config.get(attr_name)

        # Handle None values
        if actual_value is None and expected_value is not None:
            matches_all = False
            break

        # Compare values (handle type conversion for numeric comparisons)
        if actual_value != expected_value:
            # Try type conversion for numeric types
            try:
                if isinstance(expected_value, (int, float)) and isinstance(
                    actual_value, str
                ):
                    actual_value = type(expected_value)(actual_value)
                elif isinstance(actual_value, (int, float)) and isinstance(
                    expected_value, str
                ):
                    expected_value = type(actual_value)(expected_value)
            except (ValueError, TypeError):
                pass

            if actual_value != expected_value:
                matches_all = False
                break

    if matches_all:
        # Found a match!
        latest = file_info["parsed"]

        # Get stats dict from JSON
        stats_dict = data.get("stats", {})

        return {
            "filename": file_info["filename"],
            "filepath": str(filepath),
            "attributes": {
                "datetime": latest["datetime"].isoformat(),
                "model": latest["model"],
                "samples": latest["samples"],
                "crawls": latest["crawls"],
                "filter": latest["filter"],
                "prefill_mode": latest["prefill_mode"],
                "backend": latest["backend"],
            },
            "config": config,
            "stats": stats_dict,  # Stats dictionary from JSON
            "data": data,
        }

    return None


def find_latest_crawler_log(
    config_filters: Optional[Dict[str, Any]] = None,
    interim_dir: Optional[Path] = None,
    lazy_load: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Find the latest crawler log matching the specified CrawlerConfig attributes.

    Args:
        config_filters: Dictionary of CrawlerConfig attribute names to values to filter by.
                       Any CrawlerConfig attribute can be used (e.g., {'temperature': 0.6, 'device': 'cuda:0'}).
                       Common attributes:
                       - 'prefill_mode': e.g., 'user_prefill_with_seed'
                       - 'model_path': e.g., 'allenai/Tulu' or 'DeepSeek-R1-Distill-Llama-8B'
                       - 'num_samples_per_topic': e.g., 1
                       - 'num_crawl_steps': e.g., 100
                       - 'do_filter_refusals': e.g., True
                       - 'backend': e.g., 'vllm' or 'transformers'
                       - 'temperature': e.g., 0.6
                       - 'device': e.g., 'cuda:0'
        interim_dir: Directory to search (defaults to INTERIM_DIR)
        lazy_load: If True, only load JSON for files that pass filename filters first (faster).
                  If False, load JSON for all files to check config filters (slower but more accurate).

    Returns:
        Dictionary with 'filename', 'attributes', 'config', and 'data' keys, or None if no match found
    """
    if interim_dir is None:
        interim_dir = INTERIM_DIR

    if config_filters is None:
        config_filters = {}

    # Normalize model name for matching (handle both full paths and short names)
    model_normalized = None
    if "model_path" in config_filters:
        model = config_filters["model_path"]
        model_normalized = model.split("/")[-1] if "/" in model else model

    matching_files = []

    # Iterate through all files in interim directory
    for filename in os.listdir(interim_dir):
        if not filename.startswith("crawler_log_") or not filename.endswith(".json"):
            continue

        parsed = parse_crawler_log_filename(filename)
        if parsed is None:
            continue

        # Fast filtering on filename attributes (if lazy_load is True)
        if lazy_load:
            # Apply filename-based filters first for speed
            skip_file = False

            if (
                "prefill_mode" in config_filters
                and parsed["prefill_mode"] != config_filters["prefill_mode"]
            ):
                skip_file = True

            if not skip_file and model_normalized:
                parsed_model = parsed["model"]
                if (
                    model_normalized not in parsed_model
                    and parsed_model not in model_normalized
                ):
                    parsed_model_last = (
                        parsed_model.split("/")[-1]
                        if "/" in parsed_model
                        else parsed_model
                    )
                    model_normalized_last = (
                        model_normalized.split("/")[-1]
                        if "/" in model_normalized
                        else model_normalized
                    )
                    if parsed_model_last != model_normalized_last:
                        skip_file = True

            if (
                not skip_file
                and "num_samples_per_topic" in config_filters
                and parsed["samples"] != config_filters["num_samples_per_topic"]
            ):
                skip_file = True

            if (
                not skip_file
                and "num_crawl_steps" in config_filters
                and parsed["crawls"] != config_filters["num_crawl_steps"]
            ):
                skip_file = True

            if not skip_file and "do_filter_refusals" in config_filters:
                filter_bool = config_filters["do_filter_refusals"]
                filter_str = (
                    str(filter_bool)
                    if isinstance(filter_bool, bool)
                    else str(filter_bool)
                )
                if parsed["filter"] != filter_str:
                    skip_file = True

            if (
                not skip_file
                and "backend" in config_filters
                and config_filters["backend"] not in parsed["backend"]
            ):
                skip_file = True

            if skip_file:
                continue

        # Store parsed info for later (we need datetime for sorting)
        matching_files.append(
            {"filename": filename, "parsed": parsed, "filepath": interim_dir / filename}
        )

    if not matching_files:
        return None

    # Sort by datetime (most recent first)
    matching_files.sort(key=lambda x: x["parsed"]["datetime"], reverse=True)

    # Now check config filters by loading JSON data
    for file_info in matching_files:
        filepath = file_info["filepath"]
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            continue

        config = data.get("config", {})

        # Check all config filters
        matches_all = True
        for attr_name, expected_value in config_filters.items():
            # Skip filename-based filters we already checked
            if lazy_load and attr_name in [
                "prefill_mode",
                "model_path",
                "num_samples_per_topic",
                "num_crawl_steps",
                "do_filter_refusals",
                "backend",
            ]:
                # Special handling for model_path - check both model_path and parsed model name
                if attr_name == "model_path":
                    config_model = config.get("model_path", "")
                    parsed_model = file_info["parsed"]["model"]
                    expected_model = (
                        expected_value.split("/")[-1]
                        if "/" in expected_value
                        else expected_value
                    )
                    config_model_last = (
                        config_model.split("/")[-1]
                        if "/" in config_model
                        else config_model
                    )
                    if (
                        expected_model not in config_model_last
                        and expected_model not in parsed_model
                    ):
                        matches_all = False
                        break
                continue

            # Get actual value from config
            actual_value = config.get(attr_name)

            # Handle None values
            if actual_value is None and expected_value is not None:
                matches_all = False
                break

            # Compare values (handle type conversion for numeric comparisons)
            if actual_value != expected_value:
                # Try type conversion for numeric types
                try:
                    if isinstance(expected_value, (int, float)) and isinstance(
                        actual_value, str
                    ):
                        actual_value = type(expected_value)(actual_value)
                    elif isinstance(actual_value, (int, float)) and isinstance(
                        expected_value, str
                    ):
                        expected_value = type(actual_value)(expected_value)
                except (ValueError, TypeError):
                    pass

                if actual_value != expected_value:
                    matches_all = False
                    break

        if matches_all:
            # Found a match!
            latest = file_info["parsed"]

            # Get stats dict from JSON
            stats_dict = data.get("stats", {})

            return {
                "filename": file_info["filename"],
                "filepath": str(filepath),
                "attributes": {
                    "datetime": latest["datetime"].isoformat(),
                    "model": latest["model"],
                    "samples": latest["samples"],
                    "crawls": latest["crawls"],
                    "filter": latest["filter"],
                    "prefill_mode": latest["prefill_mode"],
                    "backend": latest["backend"],
                },
                "config": config,
                "stats": stats_dict,  # Stats dictionary from JSON
                "data": data,
            }

    return None


def main():
    """
    Main function to load the latest crawler log.

    Edit the config_filters dictionary below to specify which CrawlerConfig attributes
    to filter by. Any CrawlerConfig attribute can be used as a key.

    Examples:
        # Filter by prefill_mode and model
        config_filters = {
            'prefill_mode': 'user_prefill_with_seed',
            'model_path': 'allenai/Tulu',  # or 'allenai/Llama-3.1-Tulu-3-8B-SFT'
        }

        # Filter by multiple attributes
        config_filters = {
            'prefill_mode': 'user_prefill_with_seed',
            'model_path': 'allenai/Tulu',
            'temperature': 0.6,
            'num_samples_per_topic': 1,
            'backend': 'vllm',
            'device': 'cuda:0',
        }
    """
    # ============================================================================
    # CONFIGURATION: Edit this dictionary to set CrawlerConfig attribute constraints
    # ============================================================================
    config_filters = {
        "prefill_mode": "assistant_prefill_with_seed",
        # "prefill_mode": "thought_prefill_with_seed",
        # "prefill_mode": "user_prefill_with_seed",
        "model_path": "allenai/Llama-3.1-Tulu-3-8B-SFT",
        # "model_path": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        # "model_path": "perplexity-ai/r1-1776-distill-llama-70b",
        # "model_path": "meta-llama/Llama-3.1-8B-Instruct",
        # Add any other CrawlerConfig attributes here:
        # 'temperature': 0.6,
        # 'num_samples_per_topic': 1,
        # 'num_crawl_steps': 100,
        # 'backend': 'vllm',
        # 'device': 'cuda:0',
        # 'do_filter_refusals': True,
        # etc.
        "is_refusal_threshold": 0.0
    }
    # ============================================================================

    # Options
    print_config = True  # Set to True to print the full CrawlerConfig
    print_data = False  # Set to True to print the full JSON data
    return_data_only = (
        False  # Set to True to only return the data (for programmatic use)
    )
    lazy_load = True  # Set to False to check all files (slower but more accurate)
    interim_dir = None  # Set to a Path if you want to search a different directory
    move_folder_name = Path("artifacts/threshold_ablation")

    # Find all matching crawler logs
    matching_files = find_all_matching_crawler_logs(
        config_filters=config_filters, interim_dir=interim_dir, lazy_load=lazy_load
    )

    if not matching_files:
        print("No matching crawler log found.")
        return None

    if return_data_only:
        # Just return the first matching file's data
        result = load_crawler_log_from_file_info(
            matching_files[0], config_filters, lazy_load=lazy_load
        )
        return result["data"] if result else None

    # Iterate through matching files
    file_index = 0
    total_matching = len(matching_files)

    while file_index < total_matching:
        file_info = matching_files[file_index]
        result = load_crawler_log_from_file_info(
            file_info, config_filters, lazy_load=lazy_load
        )

        if result is None:
            # This file didn't pass config filters, skip it
            file_index += 1
            continue

        print("\n" + "=" * 80)
        print(f"FILE {file_index + 1} OF {total_matching}")
        print("=" * 80)
        print(f"Filename: {result['filename']}")
        print(f"Filepath: {result['filepath']}")
        print("\nFilename Attributes:")
        for key, value in result["attributes"].items():
            print(f"  {key}: {value}")

        if print_config and "config" in result:
            print("\n" + "=" * 80)
            print("CRAWLER CONFIG:")
            print("=" * 80)
            print(json.dumps(result["config"], indent=2))

        if "stats" in result and result["stats"]:
            print("\n" + "=" * 80)
            print("CRAWLER STATS:")
            print("=" * 80)
            stats_dict = result["stats"]
            cumulative = stats_dict.get("cumulative", {})
            history = stats_dict.get("history", {})
            current_metrics = stats_dict.get("current_metrics", {})

            print(f"Total all topics: {cumulative.get('total_all', 0)}")
            print(f"Total deduped topics: {cumulative.get('total_deduped', 0)}")
            print(f"Total refusals: {cumulative.get('total_refusals', 0)}")
            print(
                f"Total unique refusals: {cumulative.get('total_unique_refusals', 0)}"
            )

            all_per_step = history.get("all_per_step", [])
            if all_per_step:
                print(f"Number of steps: {len(all_per_step)}")
                total_all = cumulative.get("total_all", 0)
                total_refusals = cumulative.get("total_refusals", 0)
                print(f"Average topics per step: {total_all / len(all_per_step):.2f}")
                print(
                    f"Average refusal rate: {total_refusals / (total_all + 1e-10):.4f}"
                )

            if current_metrics:
                print(f"\nCurrent metrics:")
                for key, value in current_metrics.items():
                    print(f"  {key}: {value}")

        if print_data:
            print("\n" + "=" * 80)
            print("DATA:")
            print("=" * 80)
            print(json.dumps(result["data"], indent=2))
        else:
            print("\n" + "=" * 80)
            print("Data loaded successfully.")
            print(
                "Set print_data=True or print_config=True in main() to see more details."
            )
            print("=" * 80)

        # Ask if user wants to copy to artifacts/latest
        print("\n" + "=" * 80)
        response = (
            input(
                f"Copy this file ({file_index + 1}/{total_matching}) to /artifacts/pbr? (y/n/q): "
            )
            .strip()
            .lower()
        )

        if response == "y":
            latest_dir = project_root / move_folder_name
            latest_dir.mkdir(parents=True, exist_ok=True)

            source_file = Path(result["filepath"])
            dest_file = latest_dir / result["filename"]

            # Copy the file
            shutil.copy2(source_file, dest_file)
            print(f"✓ Copied to: {dest_file}")
            return result
        elif response == "q":
            print("Quitting.")
            return None
        else:
            print(f"Skipped file {file_index + 1}. Showing next file...")
            file_index += 1

    print("\n" + "=" * 80)
    print("No more matching files to show.")
    print("=" * 80)
    return None


if __name__ == "__main__":
    result = main()
