#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any


def normalized_label(label: str) -> str:
    return " ".join(label.split()).casefold()


def selected_cells(data: dict[str, Any], *, fixture: str, model: str) -> list[dict[str, Any]]:
    return [
        cell
        for cell in data.get("cells", [])
        if cell.get("fixture") == fixture
        and cell.get("model") == model
        and cell.get("error") is None
        and isinstance(cell.get("labels"), list)
    ]


def synthetic_cluster_score(clusters: list[dict[str, Any]]) -> float:
    scores = [
        float(cluster.get("score", 0.0) or 0.0)
        for cluster in clusters
        if float(cluster.get("score", 0.0) or 0.0) > 0
    ]
    return median(scores) if scores else 1.0


def build_combined_artifact(
    *,
    base: dict[str, Any],
    labels: list[str],
    fixture: str,
    model: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    output = json.loads(json.dumps(base))
    topics = output.setdefault("wordcloud_topics", [])
    clusters = output.setdefault("clusters", [])
    existing = {
        normalized_label(row.get("label", ""))
        for row in topics
        if isinstance(row, dict)
    }
    max_topic_index = max(
        (row.get("index") for row in topics if isinstance(row.get("index"), int)),
        default=-1,
    )
    max_cluster_id = max(
        (cluster.get("id") for cluster in clusters if isinstance(cluster.get("id"), int)),
        default=-1,
    )
    score = synthetic_cluster_score(clusters)
    added = 0
    skipped_empty = 0
    skipped_duplicates = 0

    for raw_label in labels:
        label = " ".join(str(raw_label).split())
        if not label:
            skipped_empty += 1
            continue
        key = normalized_label(label)
        if key in existing:
            skipped_duplicates += 1
            continue
        existing.add(key)
        max_topic_index += 1
        max_cluster_id += 1
        topics.append(
            {
                "index": max_topic_index,
                "label": label,
                "raw": label,
                "english": "",
                "chinese": label,
                "summary": label,
                "parent_id": -1000,
                "is_chinese": any("\u4e00" <= char <= "\u9fff" for char in label),
                "prompt": "",
                "source": {
                    "kind": "extractor_fixture",
                    "fixture": fixture,
                    "model": model,
                },
            }
        )
        clusters.append(
            {
                "id": max_cluster_id,
                "label": label,
                "representative_index": max_topic_index,
                "size": 1,
                "validated": False,
                "refusal_rate": None,
                "score": score,
                "examples": [label],
                "member_indices": [max_topic_index],
                "source": {
                    "kind": "extractor_fixture",
                    "fixture": fixture,
                    "model": model,
                },
            }
        )
        added += 1

    output.setdefault("artifacts", {})
    output["artifacts"]["combined_fixture_source"] = {
        "fixture": fixture,
        "model": model,
        "labels_seen": len(labels),
        "labels_added": added,
        "labels_skipped_empty": skipped_empty,
        "labels_skipped_duplicates": skipped_duplicates,
    }
    output.setdefault("counts", {})
    output["counts"]["wordcloud_topics"] = len(topics)
    output["counts"]["clusters"] = len(clusters)
    return output, {
        "labels_seen": len(labels),
        "labels_added": added,
        "labels_skipped_empty": skipped_empty,
        "labels_skipped_duplicates": skipped_duplicates,
        "wordcloud_topics_total": len(topics),
        "clusters_total": len(clusters),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append extractor-bench labels to a cluster-crawler wordcloud artifact."
    )
    parser.add_argument("base_artifact", type=Path)
    parser.add_argument("extractor_bench_json", type=Path)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = json.loads(args.base_artifact.read_text(encoding="utf-8"))
    bench = json.loads(args.extractor_bench_json.read_text(encoding="utf-8"))
    cells = selected_cells(bench, fixture=args.fixture, model=args.model)
    if not cells:
        raise ValueError(f"No successful cells for fixture={args.fixture!r} model={args.model!r}")
    labels: list[str] = []
    for cell in cells:
        labels.extend(str(label) for label in cell.get("labels", []))
    output, summary = build_combined_artifact(
        base=base,
        labels=labels,
        fixture=args.fixture,
        model=args.model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
