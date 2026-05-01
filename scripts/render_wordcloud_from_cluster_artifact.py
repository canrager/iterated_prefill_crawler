#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.bench_wordcloud_ranking import (
    collect_ranking_by_cluster,
    load_candidates,
    rank_cluster_score,
    rank_pairwise_proxy_elo,
    summarize_target_coverage,
    target_labels,
)
from src.cluster_crawler import render_wordcloud_scores


def ranking_for_method(artifact: Path, method: str):
    candidates = load_candidates(artifact)
    if method == "cluster_score":
        return rank_cluster_score(candidates)
    if method == "pairwise_proxy_elo":
        return rank_pairwise_proxy_elo(candidates)
    raise ValueError(f"Unknown ranking method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a paper-style wordcloud from a cluster-crawler JSON artifact."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--ranking-method",
        choices=("cluster_score", "pairwise_proxy_elo"),
        default="pairwise_proxy_elo",
    )
    parser.add_argument("--max-terms", type=int, default=120)
    parser.add_argument(
        "--max-terms-per-cluster",
        type=int,
        default=2,
        help="Maximum rendered labels from one discovered cluster; use 0 to disable.",
    )
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--target-regex",
        default=None,
        help="Optional regex for aggregate coverage reporting; matching labels only are printed.",
    )
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_ranking = ranking_for_method(args.artifact, args.ranking_method)
    max_terms_per_cluster = (
        None if args.max_terms_per_cluster <= 0 else args.max_terms_per_cluster
    )
    source_ranks = {candidate.label: idx + 1 for idx, (candidate, _score) in enumerate(raw_ranking)}
    ranking = collect_ranking_by_cluster(
        raw_ranking,
        max_terms=args.max_terms,
        max_terms_per_cluster=max_terms_per_cluster,
    )
    ranked_rows = [
        {
            "rank": idx + 1,
            "source_rank": source_ranks[candidate.label],
            "label": candidate.label,
            "score": score,
            "parent_id": candidate.parent_id,
            "parent_yield": candidate.parent_yield,
            "cluster_id": candidate.cluster_id,
            "cluster_score": candidate.cluster_score,
            "cluster_size": candidate.cluster_size,
        }
        for idx, (candidate, score) in enumerate(ranking)
    ]

    scores = {
        row["label"]: row["score"]
        for row in ranked_rows
    }
    render_wordcloud_scores(scores, args.output_png)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "artifact": str(args.artifact),
                "ranking_method": args.ranking_method,
                "max_terms": args.max_terms,
                "max_terms_per_cluster": max_terms_per_cluster,
                "wordcloud_png": str(args.output_png),
                "ranked_terms": ranked_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "artifact": str(args.artifact),
        "ranking_method": args.ranking_method,
        "candidate_count": len(raw_ranking),
        "rendered_terms": len(scores),
        "max_terms_per_cluster": max_terms_per_cluster,
        "wordcloud_png": str(args.output_png),
        "ranked_json": str(args.output_json),
    }
    if args.target_regex:
        target_re = re.compile(args.target_regex, re.IGNORECASE)
        summary["target_coverage"] = summarize_target_coverage(
            ranking,
            target_re=target_re,
            top_k=args.top_k,
        )
        summary["target_labels"] = target_labels(ranking, target_re=target_re)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
