#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from collections import Counter
from typing import Iterable


@dataclass(frozen=True)
class Candidate:
    label: str
    index: int
    parent_id: int | None
    cluster_id: int | None
    cluster_score: float
    cluster_size: int
    parent_yield: int

    @property
    def token_count(self) -> int:
        return len(self.label.split())

    @property
    def specificity_score(self) -> float:
        # Prefer phrases with enough detail to be paper-style terms, without
        # blindly rewarding very long labels.
        n = self.token_count
        if 3 <= n <= 6:
            return 1.0
        if n == 2 or n == 7:
            return 0.6
        return 0.25


def load_candidates(path: Path) -> list[Candidate]:
    data = json.loads(path.read_text(encoding="utf-8"))
    topic_rows = data.get("wordcloud_topics") or []
    if not topic_rows:
        raise ValueError("Artifact has no wordcloud_topics; rerun crawler with current serializer")

    cluster_by_member: dict[int, dict] = {}
    for cluster in data.get("clusters", []):
        for idx in cluster.get("member_indices", []):
            cluster_by_member[idx] = cluster

    parent_counts: dict[int, int] = {}
    for row in topic_rows:
        parent_id = row.get("parent_id")
        if isinstance(parent_id, int) and parent_id >= 0:
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1

    candidates: list[Candidate] = []
    seen: set[str] = set()
    for row in topic_rows:
        label = " ".join((row.get("label") or "").split())
        key = label.casefold()
        if not label or key in seen:
            continue
        seen.add(key)
        idx = row.get("index")
        if not isinstance(idx, int):
            continue
        cluster = cluster_by_member.get(idx, {})
        parent_id = row.get("parent_id")
        candidates.append(
            Candidate(
                label=label,
                index=idx,
                parent_id=parent_id if isinstance(parent_id, int) else None,
                cluster_id=cluster.get("id"),
                cluster_score=float(cluster.get("score", 0.0) or 0.0),
                cluster_size=int(cluster.get("size", 1) or 1),
                parent_yield=parent_counts.get(parent_id, 1)
                if isinstance(parent_id, int) and parent_id >= 0
                else 1,
            )
        )
    return candidates


def rank_cluster_score(candidates: Iterable[Candidate]) -> list[tuple[Candidate, float]]:
    return sorted(
        ((candidate, candidate.cluster_score) for candidate in candidates),
        key=lambda item: (-item[1], item[0].label.casefold()),
    )


def proxy_preference_score(candidate: Candidate) -> float:
    # Offline proxy for the kind of latent preference a pairwise judge should
    # learn: strong parent-yield signal, refusal/cluster signal, and concise
    # specificity. This is a benchmark comparator, not production ranking.
    return (
        1.35 * math.log1p(candidate.parent_yield)
        + 0.75 * math.log1p(candidate.cluster_score)
        + 0.55 * candidate.specificity_score
        - 0.08 * math.log1p(candidate.cluster_size)
    )


def rank_pairwise_proxy_elo(
    candidates: list[Candidate],
    *,
    initial_rating: float = 1000.0,
    k_factor: float = 24.0,
) -> list[tuple[Candidate, float]]:
    ratings = {candidate.label: initial_rating for candidate in candidates}
    ordered = sorted(candidates, key=lambda c: c.label.casefold())
    for i, left in enumerate(ordered):
        for right in ordered[i + 1 :]:
            left_score = proxy_preference_score(left)
            right_score = proxy_preference_score(right)
            if left_score == right_score:
                continue
            winner, loser = (left, right) if left_score > right_score else (right, left)
            rating_diff = ratings[loser.label] - ratings[winner.label]
            expected = 1 / (1 + 10 ** (rating_diff / 400))
            ratings[winner.label] += k_factor * (1 - expected)
            ratings[loser.label] -= k_factor * (1 - expected)
    return sorted(
        ((candidate, ratings[candidate.label]) for candidate in candidates),
        key=lambda item: (-item[1], item[0].label.casefold()),
    )


def collect_ranking_by_cluster(
    ranking: list[tuple[Candidate, float]],
    *,
    max_terms: int | None = None,
    max_terms_per_cluster: int | None = None,
) -> list[tuple[Candidate, float]]:
    selected: list[tuple[Candidate, float]] = []
    cluster_counts: Counter[int] = Counter()
    for candidate, score in ranking:
        cluster_id = candidate.cluster_id
        if (
            max_terms_per_cluster is not None
            and cluster_id is not None
            and cluster_counts[cluster_id] >= max_terms_per_cluster
        ):
            continue
        selected.append((candidate, score))
        if cluster_id is not None:
            cluster_counts[cluster_id] += 1
        if max_terms is not None and len(selected) >= max_terms:
            break
    return selected


def summarize_target_coverage(
    ranking: list[tuple[Candidate, float]],
    *,
    target_re: re.Pattern[str],
    top_k: int,
) -> dict:
    target_ranks = [
        idx + 1
        for idx, (candidate, _score) in enumerate(ranking)
        if target_re.search(candidate.label)
    ]
    if not target_ranks:
        return {
            "target_matches": 0,
            "target_in_top_k": 0,
            "best_rank": None,
            "median_rank": None,
        }
    return {
        "target_matches": len(target_ranks),
        "target_in_top_k": sum(1 for rank in target_ranks if rank <= top_k),
        "best_rank": min(target_ranks),
        "median_rank": median(target_ranks),
    }


def target_labels(
    ranking: list[tuple[Candidate, float]],
    *,
    target_re: re.Pattern[str],
) -> list[dict]:
    rows = []
    for idx, (candidate, score) in enumerate(ranking):
        if target_re.search(candidate.label):
            rows.append(
                {
                    "rank": idx + 1,
                    "label": candidate.label,
                    "score": score,
                    "parent_yield": candidate.parent_yield,
                    "cluster_score": candidate.cluster_score,
                    "cluster_size": candidate.cluster_size,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline benchmark for wordcloud ranking ideas on a frozen cluster-crawler artifact."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--target-regex",
        default=None,
        help="Optional post-hoc regex for target-specific coverage scoring.",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--max-terms-per-cluster",
        type=int,
        default=None,
        help="Optional display-time cap for collecting redundant labels from one cluster.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.artifact)
    rankings = {
        "cluster_score": rank_cluster_score(candidates),
        "pairwise_proxy_elo": rank_pairwise_proxy_elo(candidates),
    }
    result = {
        "artifact": str(args.artifact),
        "candidate_count": len(candidates),
        "target_regex": args.target_regex,
        "top_k": args.top_k,
        "methods": {},
    }
    for name, ranking in rankings.items():
        display_ranking = collect_ranking_by_cluster(
            ranking,
            max_terms=None,
            max_terms_per_cluster=args.max_terms_per_cluster,
        )
        method_result: dict = {
            "display_candidate_count": len(display_ranking),
            "top_ranked_scores": [
                {"rank": idx + 1, "score": score}
                for idx, (_candidate, score) in enumerate(
                    display_ranking[: min(args.top_k, len(display_ranking))]
                )
            ]
        }
        if args.target_regex:
            target_re = re.compile(args.target_regex, re.IGNORECASE)
            method_result.update(
                summarize_target_coverage(display_ranking, target_re=target_re, top_k=args.top_k)
            )
            method_result["target_labels"] = target_labels(display_ranking, target_re=target_re)
        result["methods"][name] = method_result
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
