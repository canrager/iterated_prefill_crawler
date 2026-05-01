#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median

from scripts.bench_wordcloud_ranking import (
    Candidate,
    collect_ranking_by_cluster,
    load_candidates,
    rank_cluster_score,
    rank_pairwise_proxy_elo,
)
from src.cluster_crawler import render_wordcloud_scores


def parse_caps(raw: str) -> list[int | None]:
    caps: list[int | None] = []
    for item in raw.split(","):
        value = item.strip().lower()
        if not value:
            continue
        if value in {"none", "raw", "0"}:
            caps.append(None)
        else:
            caps.append(int(value))
    return caps


def rank_candidates(
    candidates: list[Candidate],
    method: str,
) -> list[tuple[Candidate, float]]:
    if method == "cluster_score":
        return rank_cluster_score(candidates)
    if method == "pairwise_proxy_elo":
        return rank_pairwise_proxy_elo(candidates)
    raise ValueError(f"Unknown ranking method: {method}")


def regex_coverage(
    ranking: list[tuple[Candidate, float]],
    *,
    pattern: re.Pattern[str] | None,
    top_k: int,
) -> dict:
    if pattern is None:
        return {
            "regex_provided": False,
        }
    ranks = [
        idx + 1
        for idx, (candidate, _score) in enumerate(ranking)
        if pattern.search(candidate.label)
    ]
    if not ranks:
        return {
            "regex_provided": True,
            "matches_total": 0,
            "matches_in_top_k": 0,
            "best_rank": None,
            "median_rank": None,
        }
    return {
        "regex_provided": True,
        "matches_total": len(ranks),
        "matches_in_top_k": sum(1 for rank in ranks if rank <= top_k),
        "best_rank": min(ranks),
        "median_rank": median(ranks),
    }


def concentration_metrics(
    ranking: list[tuple[Candidate, float]],
) -> dict:
    cluster_counts = Counter(candidate.cluster_id for candidate, _score in ranking)
    parent_counts = Counter(candidate.parent_id for candidate, _score in ranking)
    return {
        "rendered_terms": len(ranking),
        "unique_clusters": len(cluster_counts),
        "unique_parents": len(parent_counts),
        "max_terms_from_one_cluster": max(cluster_counts.values(), default=0),
        "max_terms_from_one_parent": max(parent_counts.values(), default=0),
    }


def render_variant(
    ranking: list[tuple[Candidate, float]],
    *,
    output_path: Path,
) -> dict:
    scores = {candidate.label: score for candidate, score in ranking}
    render_wordcloud_scores(scores, output_path)

    from PIL import Image
    import numpy as np

    image = Image.open(output_path).convert("RGBA")
    pixels = np.array(image)
    alpha = pixels[:, :, 3]
    return {
        "path": str(output_path),
        "size": list(image.size),
        "mode": image.mode,
        "opaque_pixels": int((alpha > 0).sum()),
        "sampled_unique_rgba": len({tuple(x) for x in pixels.reshape(-1, 4)[::1000]}),
    }


def safe_variant_name(method: str, cap: int | None) -> str:
    suffix = "raw" if cap is None else f"cap{cap}"
    return f"{method}_{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate-only offline benchmark for wordcloud renderer ranking "
            "and display-collection variants."
        )
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--max-terms", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--pairwise-caps",
        default="none,1,2,3,5",
        help="Comma-separated max terms per cluster for pairwise variants; use none for raw.",
    )
    parser.add_argument(
        "--target-regex",
        default=None,
        help="Optional post-hoc target coverage regex. The regex is not copied into output.",
    )
    parser.add_argument(
        "--repeated-family-regex",
        default=None,
        help="Optional post-hoc repeated-family regex. The regex is not copied into output.",
    )
    parser.add_argument("--render-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.artifact)
    target_re = re.compile(args.target_regex, re.IGNORECASE) if args.target_regex else None
    repeated_re = (
        re.compile(args.repeated_family_regex, re.IGNORECASE)
        if args.repeated_family_regex
        else None
    )

    variants: list[dict] = []
    raw_rankings = {
        "cluster_score": rank_candidates(candidates, "cluster_score"),
        "pairwise_proxy_elo": rank_candidates(candidates, "pairwise_proxy_elo"),
    }
    variant_specs: list[tuple[str, int | None]] = [("cluster_score", None)]
    variant_specs.extend(("pairwise_proxy_elo", cap) for cap in parse_caps(args.pairwise_caps))

    for method, cap in variant_specs:
        selected = collect_ranking_by_cluster(
            raw_rankings[method],
            max_terms=args.max_terms,
            max_terms_per_cluster=cap,
        )
        variant = {
            "name": safe_variant_name(method, cap),
            "ranking_method": method,
            "max_terms_per_cluster": cap,
            **concentration_metrics(selected),
            "target_coverage": regex_coverage(
                selected,
                pattern=target_re,
                top_k=args.top_k,
            ),
            "repeated_family": regex_coverage(
                selected,
                pattern=repeated_re,
                top_k=args.top_k,
            ),
        }
        if args.render_dir:
            args.render_dir.mkdir(parents=True, exist_ok=True)
            png_path = args.render_dir / f"{safe_variant_name(method, cap)}.png"
            variant["png"] = render_variant(selected, output_path=png_path)
        variants.append(variant)

    result = {
        "artifact": str(args.artifact),
        "candidate_count": len(candidates),
        "max_terms": args.max_terms,
        "top_k": args.top_k,
        "target_regex_provided": args.target_regex is not None,
        "repeated_family_regex_provided": args.repeated_family_regex is not None,
        "variants": variants,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
