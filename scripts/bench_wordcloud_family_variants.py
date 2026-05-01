#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median

import numpy as np

from scripts.bench_wordcloud_ranking import (
    Candidate,
    collect_ranking_by_cluster,
    load_candidates,
    rank_pairwise_proxy_elo,
)
from src.cluster_crawler import cluster_vectors, embed_labels_tfidf, render_wordcloud_scores


@dataclass(frozen=True)
class Family:
    label: str
    members: tuple[str, ...]
    score: float
    source_ranks: tuple[int, ...]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def normalize_label(label: str) -> str:
    return " ".join(re.findall(r"[\w-]+", label.casefold()))


def label_tokens(label: str) -> set[str]:
    return set(normalize_label(label).split())


def string_similarity(left: str, right: str) -> float:
    left_norm = normalize_label(left)
    right_norm = normalize_label(right)
    if not left_norm or not right_norm:
        return 0.0
    left_tokens = label_tokens(left)
    right_tokens = label_tokens(right)
    overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    fuzzy_overlap = fuzzy_token_overlap(left_tokens, right_tokens)
    dice = (2 * len(left_tokens & right_tokens)) / max(1, len(left_tokens) + len(right_tokens))
    sequence = SequenceMatcher(None, left_norm, right_norm).ratio()
    return max(overlap, fuzzy_overlap, dice, sequence)


def fuzzy_token_overlap(left_tokens: set[str], right_tokens: set[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    unmatched_right = set(right_tokens)
    matches = 0
    for left in left_tokens:
        best = None
        best_score = 0.0
        for right in unmatched_right:
            score = SequenceMatcher(None, left, right).ratio()
            if score > best_score:
                best = right
                best_score = score
        if best is not None and best_score >= 0.8:
            matches += 1
            unmatched_right.remove(best)
    return matches / max(1, min(len(left_tokens), len(right_tokens)))


def families_from_ranked_groups(
    ranked: list[tuple[Candidate, float]],
    groups: list[list[int]],
) -> list[Family]:
    by_index = {candidate.index: (candidate, score, rank) for rank, (candidate, score) in enumerate(ranked, start=1)}
    families: list[Family] = []
    for group in groups:
        rows = [by_index[idx] for idx in group if idx in by_index]
        if not rows:
            continue
        rows.sort(key=lambda item: (-item[1], item[2], item[0].label.casefold()))
        label = rows[0][0].label
        members = tuple(item[0].label for item in rows)
        score = max(item[1] for item in rows)
        source_ranks = tuple(item[2] for item in rows)
        families.append(Family(label=label, members=members, score=score, source_ranks=source_ranks))
    families.sort(key=lambda family: (-family.score, min(family.source_ranks), family.label.casefold()))
    return families


def singleton_families(ranked: list[tuple[Candidate, float]]) -> list[Family]:
    return [
        Family(
            label=candidate.label,
            members=(candidate.label,),
            score=score,
            source_ranks=(rank,),
        )
        for rank, (candidate, score) in enumerate(ranked, start=1)
    ]


def string_families(
    ranked: list[tuple[Candidate, float]],
    *,
    threshold: float,
) -> list[Family]:
    families: list[list[tuple[Candidate, float, int]]] = []
    for rank, (candidate, score) in enumerate(ranked, start=1):
        best_family_idx: int | None = None
        best_similarity = 0.0
        for family_idx, family in enumerate(families):
            family_similarity = max(
                string_similarity(candidate.label, existing.label)
                for existing, _existing_score, _existing_rank in family
            )
            if family_similarity > best_similarity:
                best_similarity = family_similarity
                best_family_idx = family_idx
        if best_family_idx is not None and best_similarity >= threshold:
            families[best_family_idx].append((candidate, score, rank))
        else:
            families.append([(candidate, score, rank)])

    out: list[Family] = []
    for family in families:
        family.sort(key=lambda item: (-item[1], item[2], item[0].label.casefold()))
        out.append(
            Family(
                label=family[0][0].label,
                members=tuple(item[0].label for item in family),
                score=max(item[1] for item in family),
                source_ranks=tuple(item[2] for item in family),
            )
        )
    out.sort(key=lambda family: (-family.score, min(family.source_ranks), family.label.casefold()))
    return out


def embedding_families(
    ranked: list[tuple[Candidate, float]],
    *,
    backend: str,
    threshold: float,
    model_name: str,
    device: str,
) -> list[Family]:
    labels = [candidate.label for candidate, _score in ranked]
    if backend == "tfidf":
        vectors = embed_labels_tfidf(labels)
    elif backend == "hf":
        vectors = embed_labels_hf_local(labels, model_name=model_name, device=device)
    else:
        raise ValueError(f"Unknown embedding backend: {backend!r}")
    groups_by_position = cluster_vectors(vectors, threshold=threshold)
    index_groups = [
        [ranked[position][0].index for position in group]
        for group in groups_by_position
    ]
    return families_from_ranked_groups(ranked, index_groups)


def embed_labels_hf_local(
    labels: list[str],
    *,
    model_name: str,
    device: str,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32, local_files_only=True)
    model.to(device)
    model.eval()

    instruction = "Represent this refusal topic label for semantic grouping."
    inputs = [f"Instruct: {instruction}\nQuery: {label}" for label in labels]
    batches = []
    batch_size = 16 if device == "cpu" else 32
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            enc = tokenizer(
                inputs[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"]
            seq_lengths = mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=device), seq_lengths]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(pooled.cpu().numpy())
    if device != "cpu":
        torch.cuda.empty_cache()
    return np.concatenate(batches, axis=0)


def regex_family_coverage(
    families: list[Family],
    *,
    pattern: re.Pattern[str] | None,
    top_k: int,
) -> dict:
    if pattern is None:
        return {"regex_provided": False}
    family_ranks = []
    matching_member_count = 0
    matching_member_top_k = 0
    for idx, family in enumerate(families, start=1):
        member_matches = [member for member in family.members if pattern.search(member)]
        label_match = bool(pattern.search(family.label))
        if member_matches or label_match:
            family_ranks.append(idx)
        matching_member_count += len(member_matches)
        if idx <= top_k:
            matching_member_top_k += len(member_matches)
    if not family_ranks:
        return {
            "regex_provided": True,
            "matching_families_total": 0,
            "matching_families_in_top_k": 0,
            "matching_members_total": 0,
            "matching_members_in_top_k": 0,
            "best_rank": None,
            "median_rank": None,
        }
    return {
        "regex_provided": True,
        "matching_families_total": len(family_ranks),
        "matching_families_in_top_k": sum(1 for rank in family_ranks if rank <= top_k),
        "matching_members_total": matching_member_count,
        "matching_members_in_top_k": matching_member_top_k,
        "best_rank": min(family_ranks),
        "median_rank": median(family_ranks),
    }


def family_metrics(
    families: list[Family],
    *,
    top_k: int,
    target_re: re.Pattern[str] | None,
    repeated_re: re.Pattern[str] | None,
) -> dict:
    sizes = [len(family.members) for family in families]
    return {
        "family_count": len(families),
        "top_k": top_k,
        "max_family_size": max(sizes, default=0),
        "multi_member_families": sum(1 for size in sizes if size > 1),
        "members_total": sum(sizes),
        "target_coverage": regex_family_coverage(families, pattern=target_re, top_k=top_k),
        "repeated_family": regex_family_coverage(families, pattern=repeated_re, top_k=top_k),
    }


def write_vetting_file(
    path: Path,
    *,
    variants: list[tuple[str, list[Family]]],
    pattern: re.Pattern[str] | None,
) -> None:
    rows = []
    if pattern is not None:
        for variant_name, families in variants:
            matches = []
            for rank, family in enumerate(families, start=1):
                if pattern.search(family.label) or any(pattern.search(member) for member in family.members):
                    matches.append(
                        {
                            "rank": rank,
                            "label": family.label,
                            "score": family.score,
                            "members": list(family.members),
                        }
                    )
            rows.append({"variant": variant_name, "matching_families": matches})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"variants": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


def render_family_wordcloud(families: list[Family], output_path: Path) -> dict:
    render_wordcloud_scores({family.label: family.score for family in families}, output_path)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline benchmark for wordcloud family/canonicalization variants."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--max-terms", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--base-max-terms-per-cluster", type=int, default=2)
    parser.add_argument("--string-thresholds", default="0.82,0.88,0.94")
    parser.add_argument("--tfidf-thresholds", default="0.72,0.78,0.84")
    parser.add_argument("--hf-thresholds", default="0.76,0.82,0.88")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--target-regex", default=None)
    parser.add_argument("--repeated-family-regex", default=None)
    parser.add_argument("--vetting-regex", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--vetting-json", type=Path, default=None)
    parser.add_argument("--render-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.artifact)
    raw_ranking = rank_pairwise_proxy_elo(candidates)
    base_ranking = collect_ranking_by_cluster(
        raw_ranking,
        max_terms=args.max_terms,
        max_terms_per_cluster=args.base_max_terms_per_cluster,
    )
    target_re = re.compile(args.target_regex, re.IGNORECASE) if args.target_regex else None
    repeated_re = (
        re.compile(args.repeated_family_regex, re.IGNORECASE)
        if args.repeated_family_regex
        else None
    )
    vetting_re = re.compile(args.vetting_regex, re.IGNORECASE) if args.vetting_regex else None

    variant_families: list[tuple[str, list[Family]]] = [
        ("cap2_no_family", singleton_families(base_ranking))
    ]
    for threshold in parse_float_list(args.string_thresholds):
        variant_families.append(
            (f"string_{threshold:.2f}", string_families(base_ranking, threshold=threshold))
        )
    for threshold in parse_float_list(args.tfidf_thresholds):
        variant_families.append(
            (
                f"tfidf_embedding_{threshold:.2f}",
                embedding_families(
                    base_ranking,
                    backend="tfidf",
                    threshold=threshold,
                    model_name=args.embedding_model,
                    device=args.embedding_device,
                ),
            )
        )

    unavailable: list[dict] = []
    if not args.skip_hf:
        for threshold in parse_float_list(args.hf_thresholds):
            try:
                variant_families.append(
                    (
                        f"qwen_embedding_{threshold:.2f}",
                        embedding_families(
                            base_ranking,
                            backend="hf",
                            threshold=threshold,
                            model_name=args.embedding_model,
                            device=args.embedding_device,
                        ),
                    )
                )
            except Exception as exc:
                unavailable.append(
                    {
                        "variant": f"qwen_embedding_{threshold:.2f}",
                        "reason": type(exc).__name__,
                    }
                )
                break

    variants = []
    if args.render_dir:
        args.render_dir.mkdir(parents=True, exist_ok=True)
    for name, families in variant_families:
        row = {
            "name": name,
            **family_metrics(
                families,
                top_k=args.top_k,
                target_re=target_re,
                repeated_re=repeated_re,
            ),
        }
        if args.render_dir:
            row["png"] = render_family_wordcloud(families, args.render_dir / f"{name}.png")
        variants.append(row)

    if args.vetting_json:
        write_vetting_file(args.vetting_json, variants=variant_families, pattern=vetting_re)

    result = {
        "artifact": str(args.artifact),
        "candidate_count": len(candidates),
        "base_ranked_terms": len(base_ranking),
        "base_ranking": "pairwise_proxy_elo",
        "base_max_terms_per_cluster": args.base_max_terms_per_cluster,
        "target_regex_provided": args.target_regex is not None,
        "repeated_family_regex_provided": args.repeated_family_regex is not None,
        "vetting_regex_provided": args.vetting_regex is not None,
        "unavailable_variants": unavailable,
        "variants": variants,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    printable = {
        **{key: value for key, value in result.items() if key != "variants"},
        "variants": [
            {
                key: value
                for key, value in variant.items()
                if key in {
                    "name",
                    "family_count",
                    "max_family_size",
                    "multi_member_families",
                    "target_coverage",
                    "repeated_family",
                }
            }
            for variant in variants
        ],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
