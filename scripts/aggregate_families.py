#!/usr/bin/env python
"""Aggregate cluster-crawler wordcloud_topics into display families.

Stage 2 of the audit chain (after ``cluster_crawler.py`` and before
``self_rank_families.py``). Takes a cluster-crawler artifact, runs an
incremental-display aggregator pass against a helper LLM in batches, and
writes a families.json that the self-ranking stage consumes.

The aggregator prompt is axis-preserving by design: it asks the helper to
keep distinct refusal axes separate when in doubt, rather than collapsing
them into umbrella labels. See ``.trio/LEARNINGS.md`` aggregator section
for why semantic-clustering style prompts (the original paper's
``exp/postprocess_topic_summaries.py`` and the second update's
``src/aggregation/aggregator.py:REDUCTION_PROMPT``) lose rare-but-distinctive
axes.

Usage:

    uv run python scripts/aggregate_families.py \\
        artifacts/out/<run>/<crawl>.json \\
        --aggregator-model qwen/qwen3-235b-a22b-2507 \\
        --output-dir artifacts/out/<run>/<crawl>_families \\
        --output-json artifacts/out/<run>/<crawl>_families/families.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.openrouter_utils import API_CALL_FAILED_SENTINEL, REASONING_DISABLED, async_query_openrouter
from src.provider_config import get_provider_client_kwargs
from src.wordcloud_topic_loader import (
    Candidate,
    collect_ranking_by_cluster,
    load_candidates,
    order_by_parent_yield,
)


SYSTEM_PROMPT = """You group topic labels for a word cloud.

Return only valid JSON.
Do not explain.
Do not add new topics."""


INCREMENTAL_USER_PROMPT_TEMPLATE = """Assign new topic labels to display families for a word cloud.

You are updating an existing set of display family labels.

Rules:
- Output a JSON array.
- Each item must be: {{"label": "...", "members": ["..."]}}.
- Every new topic string must appear exactly once in members.
- Every member string must be copied exactly from new_topics.
- Do not include old topics or existing labels in members.
- The label should be short, natural, and specific.
- Use an existing label only when the new topic clearly names the same thing.
- Create a new label when the new topic is more specific or meaningfully different.
- Merge only labels that name the same thing in different wording.
- If unsure, keep the new topic separate.

existing_labels:
{existing_labels_json}

new_topics:
{new_topics_json}"""


@dataclass(frozen=True)
class Family:
    label: str
    members: tuple[str, ...]
    score: float
    source_ranks: tuple[int, ...]


@dataclass(frozen=True)
class AggregatorResult:
    model: str
    raw_output_paths: list[str]
    parse_success: bool
    parse_error: str | None
    repair_counts: dict[str, int]
    families: list[Family]


def build_incremental_prompt(*, existing_labels: list[str], new_topics: list[str]) -> str:
    return INCREMENTAL_USER_PROMPT_TEMPLATE.format(
        existing_labels_json=json.dumps(existing_labels, ensure_ascii=False, indent=2),
        new_topics_json=json.dumps(new_topics, ensure_ascii=False, indent=2),
    )


def extract_json_array(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def ranked_input(
    artifact: Path,
    *,
    max_terms: int,
    max_terms_per_cluster: int,
) -> list[tuple[Candidate, float]]:
    candidates = load_candidates(artifact)
    raw_ranking = order_by_parent_yield(candidates)
    return collect_ranking_by_cluster(
        raw_ranking,
        max_terms=max_terms,
        max_terms_per_cluster=max_terms_per_cluster,
    )


def repair_families(
    parsed: Any,
    ranking: list[tuple[Candidate, float]],
) -> tuple[list[Family], dict[str, int]]:
    """Validate aggregator output and add singleton fallbacks for missing labels."""
    labels = [candidate.label for candidate, _score in ranking]
    allowed = set(labels)
    score_by_label = {candidate.label: score for candidate, score in ranking}
    rank_by_label = {candidate.label: rank for rank, (candidate, _score) in enumerate(ranking, start=1)}
    seen: set[str] = set()
    counts = Counter(
        {
            "families_seen": 0,
            "families_kept": 0,
            "non_object_families": 0,
            "invented_members": 0,
            "duplicate_members": 0,
            "empty_families": 0,
            "label_fallbacks": 0,
            "missing_members": 0,
            "singleton_fallbacks_added": 0,
        }
    )
    repaired: list[Family] = []

    if not isinstance(parsed, list):
        counts["non_list_output"] += 1
        parsed = []

    for item in parsed:
        counts["families_seen"] += 1
        if not isinstance(item, dict):
            counts["non_object_families"] += 1
            continue
        raw_members = item.get("members")
        if not isinstance(raw_members, list):
            counts["empty_families"] += 1
            continue
        members: list[str] = []
        for member in raw_members:
            if not isinstance(member, str) or member not in allowed:
                counts["invented_members"] += 1
                continue
            if member in seen:
                counts["duplicate_members"] += 1
                continue
            seen.add(member)
            members.append(member)
        if not members:
            counts["empty_families"] += 1
            continue
        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            counts["label_fallbacks"] += 1
            label = min(members, key=lambda member: rank_by_label[member])
        else:
            label = " ".join(label.split())
        repaired.append(
            Family(
                label=label,
                members=tuple(sorted(members, key=lambda member: rank_by_label[member])),
                score=max(score_by_label[member] for member in members),
                source_ranks=tuple(rank_by_label[member] for member in members),
            )
        )
        counts["families_kept"] += 1

    missing = [label for label in labels if label not in seen]
    counts["missing_members"] = len(missing)
    counts["singleton_fallbacks_added"] = len(missing)
    for label in missing:
        repaired.append(
            Family(
                label=label,
                members=(label,),
                score=score_by_label[label],
                source_ranks=(rank_by_label[label],),
            )
        )

    repaired.sort(key=lambda family: (-family.score, min(family.source_ranks), family.label.casefold()))
    return repaired, dict(counts)


def merge_family_batches(
    existing: list[Family],
    incoming: list[Family],
    ranking: list[tuple[Candidate, float]],
) -> list[Family]:
    score_by_label = {candidate.label: score for candidate, score in ranking}
    rank_by_label = {candidate.label: rank for rank, (candidate, _score) in enumerate(ranking, start=1)}
    members_by_label: dict[str, list[str]] = {family.label: list(family.members) for family in existing}
    for family in incoming:
        members_by_label.setdefault(family.label, []).extend(family.members)

    merged: list[Family] = []
    for label, members in members_by_label.items():
        deduped = sorted(set(members), key=lambda member: rank_by_label[member])
        merged.append(
            Family(
                label=label,
                members=tuple(deduped),
                score=max(score_by_label[member] for member in deduped),
                source_ranks=tuple(rank_by_label[member] for member in deduped),
            )
        )
    merged.sort(key=lambda family: (-family.score, min(family.source_ranks), family.label.casefold()))
    return merged


def combine_repair_counts(counts_by_batch: list[dict[str, int]]) -> dict[str, int]:
    combined = Counter()
    for counts in counts_by_batch:
        combined.update(counts)
    combined["batches"] = len(counts_by_batch)
    return dict(combined)


def write_families(path: Path, result: AggregatorResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": result.model,
                "parse_success": result.parse_success,
                "parse_error": result.parse_error,
                "repair_counts": result.repair_counts,
                "families": [
                    {
                        "rank": idx + 1,
                        "label": family.label,
                        "score": family.score,
                        "members": list(family.members),
                    }
                    for idx, family in enumerate(result.families)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


async def call_aggregator_model(
    *,
    model: str,
    default_provider: str,
    prompt: str,
    max_tokens: int,
) -> tuple[str, str, dict[str, int]]:
    resolved_model, client_kwargs = get_provider_client_kwargs(model, default_provider, None)
    raw, usage = await async_query_openrouter(
        model_name=resolved_model,
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=max_tokens,
        client_kwargs=client_kwargs,
        prefer_nitro=True,
        extra_body=REASONING_DISABLED,
        return_usage=True,
        universal_backup_model=None,
    )
    return resolved_model, raw, usage


async def run_incremental(
    *,
    model: str,
    default_provider: str,
    ranking: list[tuple[Candidate, float]],
    output_dir: Path,
    max_tokens: int,
    batch_size: int,
) -> AggregatorResult:
    resolved_model, _ = get_provider_client_kwargs(model, default_provider, None)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", resolved_model)
    labels = [candidate.label for candidate, _score in ranking]
    families: list[Family] = []
    counts_by_batch: list[dict[str, int]] = []
    parse_success = True
    parse_errors: list[str] = []
    raw_paths: list[str] = []

    for batch_start in range(0, len(ranking), batch_size):
        batch_ranking = ranking[batch_start : batch_start + batch_size]
        batch_labels = labels[batch_start : batch_start + batch_size]
        prompt = build_incremental_prompt(
            existing_labels=[family.label for family in families],
            new_topics=batch_labels,
        )
        _resolved_again, raw, usage = await call_aggregator_model(
            model=model,
            default_provider=default_provider,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        raw_path = output_dir / f"{safe_model}.batch_{len(counts_by_batch) + 1:02d}.raw.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw, encoding="utf-8")
        raw_paths.append(str(raw_path))

        parsed: Any = []
        batch_parse_success = False
        batch_parse_error = None
        if raw == API_CALL_FAILED_SENTINEL:
            batch_parse_error = API_CALL_FAILED_SENTINEL
        else:
            try:
                parsed = extract_json_array(raw)
                batch_parse_success = isinstance(parsed, list)
                if not batch_parse_success:
                    batch_parse_error = "parsed JSON was not an array"
            except Exception as exc:
                batch_parse_error = f"{type(exc).__name__}: {exc}"
        if not batch_parse_success:
            parse_success = False
            parse_errors.append(f"batch {len(counts_by_batch) + 1}: {batch_parse_error}")
        batch_families, repair_counts = repair_families(parsed, batch_ranking)
        repair_counts["prompt_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
        repair_counts["completion_tokens"] = int(usage.get("completion_tokens", 0) or 0)
        counts_by_batch.append(repair_counts)
        families = merge_family_batches(families, batch_families, ranking)

    return AggregatorResult(
        model=resolved_model,
        raw_output_paths=raw_paths,
        parse_success=parse_success,
        parse_error="; ".join(parse_errors) if parse_errors else None,
        repair_counts=combine_repair_counts(counts_by_batch),
        families=families,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("artifact", type=Path, help="Cluster-crawler JSON artifact")
    p.add_argument("--aggregator-model", required=True, help="Helper model id (e.g. qwen/qwen3-235b-a22b-2507)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--max-terms", type=int, default=260)
    p.add_argument("--max-terms-per-cluster", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=3500)
    p.add_argument("--default-provider", default="openrouter")
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking = ranked_input(
        args.artifact,
        max_terms=args.max_terms,
        max_terms_per_cluster=args.max_terms_per_cluster,
    )
    (args.output_dir / "aggregator_prompt.txt").write_text(
        f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER (template):\n{INCREMENTAL_USER_PROMPT_TEMPLATE}",
        encoding="utf-8",
    )
    result = await run_incremental(
        model=args.aggregator_model,
        default_provider=args.default_provider,
        ranking=ranking,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )
    family_path = args.output_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', result.model)}.families.json"
    write_families(family_path, result)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "artifact": str(args.artifact),
                "model": result.model,
                "batch_size": args.batch_size,
                "max_terms": args.max_terms,
                "max_terms_per_cluster": args.max_terms_per_cluster,
                "parse_success": result.parse_success,
                "parse_error": result.parse_error,
                "family_count": len(result.families),
                "max_family_size": max((len(f.members) for f in result.families), default=0),
                "repair_counts": result.repair_counts,
                "families_path": str(family_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(result.families)} families to {family_path}")


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
