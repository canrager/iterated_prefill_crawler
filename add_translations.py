#!/usr/bin/env python3
"""Script to add English translations as separate entries to all queries in JSON files."""

import json
import re
from typing import List


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def extract_english_translation(query: str) -> str:
    """Extract English translation if already appended to query."""
    match = re.search(r"\[English:\s*(.+?)\]$", query)
    if match:
        return match.group(1).strip()
    return None


def process_queries(queries: List[str]) -> List[str]:
    """Process queries and add English translations as separate entries."""
    result = []
    i = 0
    while i < len(queries):
        query = queries[i]

        # Check if this entry is already a formatted English translation
        if query.strip().startswith("[English:") and query.strip().endswith("]"):
            # Extract plain English text
            english_text = extract_english_translation(query)
            if (
                english_text
                and english_text != "already in English"
                and english_text != "Translation needed"
            ):
                result.append(english_text)
            i += 1
            continue

        # Check if translation is appended to the query
        english_translation = extract_english_translation(query)

        if english_translation:
            # Remove the appended translation from original query
            original_query = re.sub(r"\s*\[English:.*?\]$", "", query).strip()
            result.append(original_query)
            # Add translation as separate entry (plain text, no formatting)
            if (
                english_translation != "already in English"
                and english_translation != "Translation needed"
            ):
                result.append(english_translation)
        elif contains_chinese(query):
            # Query has Chinese but no translation yet - keep as is
            result.append(query)
        else:
            # Already in English, no translation needed
            result.append(query)

        i += 1

    return result


def process_json_file(file_path: str):
    """Process a JSON file and add translations to all queries."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for topic in data.get("topics", []):
        if "queries" in topic:
            original_queries = topic["queries"]
            topic["queries"] = process_queries(original_queries)
            # Update the count
            topic["num_queries"] = len(topic["queries"])

    # Write back to file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    files = [
        "/home/can/iterated_prefill_crawler/artifacts/pbr/refusal_check_topics.json",
        "/home/can/iterated_prefill_crawler/artifacts/pbr/refusal_check_queries_benchmark.json",
    ]

    for file_path in files:
        print(f"Processing {file_path}...")
        process_json_file(file_path)
        print(f"Done processing {file_path}")
