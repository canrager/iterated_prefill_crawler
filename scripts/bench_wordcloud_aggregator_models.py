#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from scripts.bench_wordcloud_family_variants import Family, regex_family_coverage
from scripts.bench_wordcloud_ranking import (
    Candidate,
    collect_ranking_by_cluster,
    load_candidates,
    rank_pairwise_proxy_elo,
)
from src.cluster_crawler import load_cluster_crawler_config, render_wordcloud_scores
from src.crawler.config import REDUCTION_PROMPT
from src.openrouter_utils import API_CALL_FAILED_SENTINEL, REASONING_DISABLED, async_query_openrouter
from src.provider_config import get_provider_client_kwargs


SYSTEM_PROMPT = """You group topic labels for a word cloud.

Return only valid JSON.
Do not explain.
Do not add new topics."""


USER_PROMPT_TEMPLATE = """Group these topic labels into display families.

Rules:
- Output a JSON array.
- Each item must be: {{"label": "...", "members": ["..."]}}.
- Every input string must appear exactly once in members.
- Every member string must be copied exactly from the input.
- The label should be short, natural, and specific.
- The label may be rewritten for readability.
- The label must have the same level of specificity as the members.
- Merge only labels that name the same thing in different wording.
- If unsure, keep labels separate.

Example input:
[
  "coffee drinks, cold brew preparation",
  "best chocolate chip cookie recipe",
  "cozy cafe",
  "making chocolate chip cookies",
  "cold-brew coffee"
]

Example output:
[
  {{
    "label": "cold-brew coffee",
    "members": [
      "coffee drinks, cold brew preparation", "cold-brew coffee"
    ]
  }},
  {{
    "label": "cozy cafe",
    "members": [
      "cozy cafe"
    ]
  }},
  {{
    "label": "chocolate chip cookie recipe",
    "members": [
      "best chocolate chip cookie recipe", "making chocolate chip cookies"
    ]
  }}
]

Input labels:
{labels_json}"""


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


OLD_INDEXED_EXAMPLES = [
    """This is the list of novel topics:
1. You're a jerk.
2. How to make a bomb
3. Have you ever had a threesome?
4. You're so stupid.
5. How to hack into someone's account
6. Instructions for creating explosives
7. May contain adult content or be otherwise inappropriate.
8. Instructions for breaking encryption
9. How did you know you were gay?
10. What is the weather today?
11. Tell me about cooking recipes

<think>
Looking at these topics, I can identify several patterns:
- Topics 1 and 4 are both offensive/insulting language directed at someone
- Topics 2 and 6 are both about creating weapons/explosives 
- Topics 3, 7, and 9 all relate to sexual or adult content
- Topics 5 and 8 are about unauthorized access and security breaches
- Topics 10 and 11 are about weather and cooking, which are unrelated to the others and to each other

So I should group topics 1,4 under "offensive language", topics 2,6 under "violence and weapons", topics 3,7,9 under "sexual content", topics 5,8 under "cybersecurity violations", and leave topics 10,11 separate since they don't relate to any safety concerns or each other.
</think>

{"offensive language":[1,4],"violence and weapons":[2,6],"sexual content":[3,7,9],"cybersecurity violations":[5,8],"What is the weather today?":[10],"Tell me about cooking recipes":[11]}""",
    """This is the list of novel topics:
1. The men are discussing a recent mass shooting in the United States
2. You should kill yourself
3. How to obtain firearms illegally
4. I hope you die in a car accident
5. Discussion about gun control policies
6. You're worthless and nobody likes you
7. What is the weather today?
8. Tell me about cooking recipes
9. How do I change a tire?
10. What are the benefits of exercise?

<think>
Analyzing the patterns in these topics:
- Topics 1, 3, and 5 all relate to guns, violence, and firearms
- Topics 2, 4, and 6 are all threatening, harmful, or harassing language toward someone
- Topics 7, 8, 9, and 10 are all completely unrelated general topics about weather, cooking, car maintenance, and health/fitness

I should group 1,3,5 under "gun violence", group 2,4,6 under "harassment and threats", and leave 7,8,9,10 separate since they don't share common themes with each other or the safety topics.
</think>

{"gun violence":[1,3,5],"harassment and threats":[2,4,6],"What is the weather today?":[7],"Tell me about cooking recipes":[8],"How do I change a tire?":[9],"What are the benefits of exercise?":[10]}""",
]


@dataclass(frozen=True)
class AggregatorResult:
    model: str
    raw_output_path: str
    parse_success: bool
    parse_error: str | None
    repair_counts: dict[str, int]
    families: list[Family]


def build_prompt(labels: list[str]) -> str:
    return USER_PROMPT_TEMPLATE.format(labels_json=json.dumps(labels, ensure_ascii=False, indent=2))


def build_incremental_prompt(*, existing_labels: list[str], new_topics: list[str]) -> str:
    return INCREMENTAL_USER_PROMPT_TEMPLATE.format(
        existing_labels_json=json.dumps(existing_labels, ensure_ascii=False, indent=2),
        new_topics_json=json.dumps(new_topics, ensure_ascii=False, indent=2),
    )


def build_old_indexed_prompt(labels: list[str]) -> str:
    examples_text = "\n\n".join(OLD_INDEXED_EXAMPLES)
    topics_list = "\n".join([f"{i+1}. {label}" for i, label in enumerate(labels)])
    return (
        f"Instructions: You are clustering topics into groups based on semantic similarity. "
        f"Given a batch of topics, group them into clusters where topics in the same cluster share a common theme or concept. "
        f"Group topics that are semantically similar (e.g., 'offensive language', 'sexual content', 'gun violence'). "
        f"DO NOT group topics that are unrelated or too different - each unrelated topic should be in its own cluster. "
        f"Output your grouping as a JSON object where each key is a concise abstract description (1-5 words) "
        f"that captures the common theme of the cluster (e.g., 'offensive language', 'sexual content', 'gun violence'). "
        f"Each value is a list of topic indices (1-indexed) that belong to that cluster. "
        f"If a topic doesn't match any others, use the topic text itself as the key (or a brief summary if it's too long). "
        f"The abstract description should be:\n"
        f"- Specific enough to be meaningful (not too broad like 'inappropriate content')\n"
        f"- General enough to cover all topics in the cluster (not too narrow like listing specific examples)\n"
        f"- Examples: 'offensive language', 'sexual content', 'gun violence', 'cybersecurity violations'\n\n"
        f"{'='*80}\n"
        f"EXAMPLES (for reference only - do not cluster these):\n"
        f"{'='*80}\n\n"
        f"{examples_text}\n\n"
        f"{'='*80}\n"
        f"NEW TOPICS TO CLUSTER (these are the topics you need to cluster):\n"
        f"{'='*80}\n\n"
        f"This is the list of novel topics:\n{topics_list}\n\n"
        f"Group these topics into clusters. First, think through your reasoning in <think> tags, then provide the JSON output directly. IMPORTANT: Double-check that every topic index from 1 to {len(labels)} appears exactly once in your JSON output.\n"
    )


def build_reduction_object_prompt(labels: list[str], output_size: int) -> str:
    topics = "\n".join(f"- {label}" for label in labels)
    return REDUCTION_PROMPT.format(
        n_input=len(labels),
        output_batch_size=output_size,
        topics=topics,
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


def extract_json_object(text: str) -> Any:
    stripped = text.strip()
    if "</think>" in stripped:
        stripped = stripped.split("</think>")[-1]
    code_block_pattern = r"```(?:jsonl|json)?\s*(\{.*?\})\s*```"
    code_block_matches = re.findall(code_block_pattern, stripped, re.DOTALL)
    if code_block_matches:
        return json.loads(code_block_matches[-1].replace("\n", "").replace("\r", ""))

    objects = []
    start = 0
    while True:
        json_start = stripped.find("{", start)
        if json_start < 0:
            break
        brace_count = 0
        json_end = json_start
        for idx in range(json_start, len(stripped)):
            if stripped[idx] == "{":
                brace_count += 1
            elif stripped[idx] == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = idx + 1
                    break
        if json_end <= json_start:
            break
        candidate = stripped[json_start:json_end]
        try:
            objects.append(json.loads(candidate.replace("\n", "").replace("\r", "")))
        except json.JSONDecodeError:
            try:
                objects.append(json.loads(candidate))
            except json.JSONDecodeError:
                pass
        start = json_end
    if objects:
        return objects[-1]
    return json.loads(stripped)


def ranked_input(
    artifact: Path,
    *,
    max_terms: int,
    max_terms_per_cluster: int,
) -> list[tuple[Candidate, float]]:
    candidates = load_candidates(artifact)
    raw_ranking = rank_pairwise_proxy_elo(candidates)
    return collect_ranking_by_cluster(
        raw_ranking,
        max_terms=max_terms,
        max_terms_per_cluster=max_terms_per_cluster,
    )


def repair_families(
    parsed: Any,
    ranking: list[tuple[Candidate, float]],
) -> tuple[list[Family], dict[str, int]]:
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
        members = []
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


def repair_indexed_families(
    parsed: Any,
    ranking: list[tuple[Candidate, float]],
) -> tuple[list[Family], dict[str, int]]:
    labels = [candidate.label for candidate, _score in ranking]
    score_by_index = {idx: score for idx, (_candidate, score) in enumerate(ranking, start=1)}
    label_by_index = {idx: candidate.label for idx, (candidate, _score) in enumerate(ranking, start=1)}
    seen: set[int] = set()
    counts = Counter(
        {
            "families_seen": 0,
            "families_kept": 0,
            "non_list_families": 0,
            "invented_indices": 0,
            "duplicate_indices": 0,
            "empty_families": 0,
            "label_fallbacks": 0,
            "missing_members": 0,
            "singleton_fallbacks_added": 0,
        }
    )
    repaired: list[Family] = []

    if not isinstance(parsed, dict):
        counts["non_object_output"] += 1
        parsed = {}

    for raw_label, raw_indices in parsed.items():
        counts["families_seen"] += 1
        if not isinstance(raw_indices, list):
            counts["non_list_families"] += 1
            continue
        indices: list[int] = []
        for raw_idx in raw_indices:
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                counts["invented_indices"] += 1
                continue
            if idx < 1 or idx > len(labels):
                counts["invented_indices"] += 1
                continue
            if idx in seen:
                counts["duplicate_indices"] += 1
                continue
            seen.add(idx)
            indices.append(idx)
        if not indices:
            counts["empty_families"] += 1
            continue
        label = raw_label if isinstance(raw_label, str) and raw_label.strip() else None
        if label is None:
            counts["label_fallbacks"] += 1
            label = label_by_index[min(indices)]
        label = " ".join(label.split())
        ordered_indices = tuple(sorted(indices))
        repaired.append(
            Family(
                label=label,
                members=tuple(label_by_index[idx] for idx in ordered_indices),
                score=max(score_by_index[idx] for idx in ordered_indices),
                source_ranks=ordered_indices,
            )
        )
        counts["families_kept"] += 1

    missing = [idx for idx in range(1, len(labels) + 1) if idx not in seen]
    counts["missing_members"] = len(missing)
    counts["singleton_fallbacks_added"] = len(missing)
    for idx in missing:
        repaired.append(
            Family(
                label=label_by_index[idx],
                members=(label_by_index[idx],),
                score=score_by_index[idx],
                source_ranks=(idx,),
            )
        )

    repaired.sort(key=lambda family: (-family.score, min(family.source_ranks), family.label.casefold()))
    return repaired, dict(counts)


def repair_object_families(
    parsed: Any,
    ranking: list[tuple[Candidate, float]],
) -> tuple[list[Family], dict[str, int]]:
    if not isinstance(parsed, dict):
        return repair_families(parsed, ranking)
    converted = [
        {"label": label, "members": members}
        for label, members in parsed.items()
    ]
    return repair_families(converted, ranking)


def merge_family_batches(
    existing: list[Family],
    incoming: list[Family],
    ranking: list[tuple[Candidate, float]],
) -> list[Family]:
    score_by_label = {candidate.label: score for candidate, score in ranking}
    rank_by_label = {candidate.label: rank for rank, (candidate, _score) in enumerate(ranking, start=1)}
    members_by_label: dict[str, list[str]] = {
        family.label: list(family.members)
        for family in existing
    }
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


def normalize_model_overrides(values: list[str] | None) -> list[str]:
    models: list[str] = []
    for value in values or []:
        for model in value.split(","):
            model = model.strip()
            if model:
                models.append(model)
    return list(dict.fromkeys(models))


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
        "max_family_size": max(sizes, default=0),
        "multi_member_families": sum(1 for size in sizes if size > 1),
        "members_total": sum(sizes),
        "target_coverage": regex_family_coverage(families, pattern=target_re, top_k=top_k),
        "repeated_family": regex_family_coverage(families, pattern=repeated_re, top_k=top_k),
    }


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


def write_vetting(path: Path, *, results: list[AggregatorResult], pattern: re.Pattern[str] | None) -> None:
    rows = []
    if pattern is not None:
        for result in results:
            matches = []
            for rank, family in enumerate(result.families, start=1):
                if pattern.search(family.label) or any(pattern.search(member) for member in family.members):
                    matches.append(
                        {
                            "rank": rank,
                            "label": family.label,
                            "score": family.score,
                            "members": list(family.members),
                        }
                    )
            rows.append({"model": result.model, "matching_families": matches})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


def render_families(path: Path, families: list[Family]) -> dict:
    render_wordcloud_scores({family.label: family.score for family in families}, path)
    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("RGBA")
    pixels = np.array(image)
    alpha = pixels[:, :, 3]
    return {
        "path": str(path),
        "size": list(image.size),
        "mode": image.mode,
        "opaque_pixels": int((alpha > 0).sum()),
        "sampled_unique_rgba": len({tuple(x) for x in pixels.reshape(-1, 4)[::1000]}),
    }


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


async def run_model(
    *,
    model: str,
    default_provider: str,
    prompt: str,
    ranking: list[tuple[Candidate, float]],
    output_dir: Path,
    max_tokens: int,
    prompt_style: str,
) -> AggregatorResult:
    resolved_model, raw, usage = await call_aggregator_model(
        model=model,
        default_provider=default_provider,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", resolved_model)
    raw_path = output_dir / f"{safe_model}.raw.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(raw, encoding="utf-8")

    parse_success = False
    parse_error = None
    parsed: Any = []
    if raw == API_CALL_FAILED_SENTINEL:
        parse_error = API_CALL_FAILED_SENTINEL
    else:
        try:
            if prompt_style in {"old-indexed", "reduction-object"}:
                parsed = extract_json_object(raw)
                parse_success = isinstance(parsed, dict)
                if not parse_success:
                    parse_error = "parsed JSON was not an object"
            else:
                parsed = extract_json_array(raw)
                parse_success = isinstance(parsed, list)
                if not parse_success:
                    parse_error = "parsed JSON was not an array"
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    if prompt_style == "old-indexed":
        families, repair_counts = repair_indexed_families(parsed, ranking)
    elif prompt_style == "reduction-object":
        families, repair_counts = repair_object_families(parsed, ranking)
    else:
        families, repair_counts = repair_families(parsed, ranking)
    repair_counts["prompt_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
    repair_counts["completion_tokens"] = int(usage.get("completion_tokens", 0) or 0)
    return AggregatorResult(
        model=resolved_model,
        raw_output_path=str(raw_path),
        parse_success=parse_success,
        parse_error=parse_error,
        repair_counts=repair_counts,
        families=families,
    )


async def run_incremental_model(
    *,
    model: str,
    default_provider: str,
    ranking: list[tuple[Candidate, float]],
    output_dir: Path,
    max_tokens: int,
    batch_size: int,
) -> AggregatorResult:
    resolved_model, _client_kwargs = get_provider_client_kwargs(model, default_provider, None)
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

    repair_counts = combine_repair_counts(counts_by_batch)
    return AggregatorResult(
        model=resolved_model,
        raw_output_path=";".join(raw_paths),
        parse_success=parse_success,
        parse_error="; ".join(parse_errors) if parse_errors else None,
        repair_counts=repair_counts,
        families=families,
    )


async def run_batched_old_indexed_model(
    *,
    model: str,
    default_provider: str,
    ranking: list[tuple[Candidate, float]],
    output_dir: Path,
    max_tokens: int,
    batch_size: int,
) -> AggregatorResult:
    resolved_model, _client_kwargs = get_provider_client_kwargs(model, default_provider, None)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", resolved_model)
    families: list[Family] = []
    counts_by_batch: list[dict[str, int]] = []
    parse_success = True
    parse_errors: list[str] = []
    raw_paths: list[str] = []

    for batch_start in range(0, len(ranking), batch_size):
        batch_ranking = ranking[batch_start : batch_start + batch_size]
        batch_labels = [candidate.label for candidate, _score in batch_ranking]
        prompt = build_old_indexed_prompt(batch_labels)
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

        parsed: Any = {}
        batch_parse_success = False
        batch_parse_error = None
        if raw == API_CALL_FAILED_SENTINEL:
            batch_parse_error = API_CALL_FAILED_SENTINEL
        else:
            try:
                parsed = extract_json_object(raw)
                batch_parse_success = isinstance(parsed, dict)
                if not batch_parse_success:
                    batch_parse_error = "parsed JSON was not an object"
            except Exception as exc:
                batch_parse_error = f"{type(exc).__name__}: {exc}"
        if not batch_parse_success:
            parse_success = False
            parse_errors.append(f"batch {len(counts_by_batch) + 1}: {batch_parse_error}")
        batch_families, repair_counts = repair_indexed_families(parsed, batch_ranking)
        repair_counts["prompt_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
        repair_counts["completion_tokens"] = int(usage.get("completion_tokens", 0) or 0)
        counts_by_batch.append(repair_counts)
        families = merge_family_batches(families, batch_families, ranking)

    repair_counts = combine_repair_counts(counts_by_batch)
    return AggregatorResult(
        model=resolved_model,
        raw_output_path=";".join(raw_paths),
        parse_success=parse_success,
        parse_error="; ".join(parse_errors) if parse_errors else None,
        repair_counts=repair_counts,
        families=families,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark helper/fallback aggregator models for wordcloud display families."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--cluster-crawler-config", default="debug")
    parser.add_argument("--max-terms", type=int, default=120)
    parser.add_argument("--max-terms-per-cluster", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument(
        "--prompt-style",
        choices=(
            "display-label",
            "incremental-display",
            "old-indexed",
            "batched-old-indexed",
            "reduction-object",
        ),
        default="display-label",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for --prompt-style incremental-display.",
    )
    parser.add_argument(
        "--aggregator-model",
        action="append",
        default=None,
        help=(
            "Override benchmark models. May be passed multiple times or as a comma-separated list. "
            "Defaults to config helper_model and universal_backup_model."
        ),
    )
    parser.add_argument(
        "--reduction-output-size",
        type=int,
        default=None,
        help="Target output count for --prompt-style reduction-object; defaults to max_terms.",
    )
    parser.add_argument("--default-provider", default=None)
    parser.add_argument("--target-regex", default=None)
    parser.add_argument("--repeated-family-regex", default=None)
    parser.add_argument("--vetting-regex", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--vetting-json", type=Path, default=None)
    parser.add_argument("--render-dir", type=Path, default=None)
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    config = load_cluster_crawler_config(args.cluster_crawler_config)
    helper_model = config.get("helper_model")
    fallback_model = config.get("universal_backup_model")
    models = normalize_model_overrides(args.aggregator_model)
    if not models:
        if not helper_model or not fallback_model:
            raise ValueError("Config must define helper_model and universal_backup_model")
        models = list(dict.fromkeys([helper_model, fallback_model]))
    default_provider = args.default_provider or config.get("default_provider") or "openrouter"

    ranking = ranked_input(
        args.artifact,
        max_terms=args.max_terms,
        max_terms_per_cluster=args.max_terms_per_cluster,
    )
    labels = [candidate.label for candidate, _score in ranking]
    if args.prompt_style == "old-indexed":
        prompt = build_old_indexed_prompt(labels)
    elif args.prompt_style == "reduction-object":
        output_size = args.reduction_output_size or args.max_terms
        prompt = build_reduction_object_prompt(labels, output_size)
    elif args.prompt_style in {"incremental-display", "batched-old-indexed"}:
        if not args.batch_size or args.batch_size <= 0:
            raise ValueError(f"--prompt-style {args.prompt_style} requires --batch-size > 0")
        if args.prompt_style == "batched-old-indexed":
            prompt = build_old_indexed_prompt(labels[: args.batch_size])
        else:
            prompt = build_incremental_prompt(existing_labels=[], new_topics=labels[: args.batch_size])
    else:
        prompt = build_prompt(labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregator_prompt.txt").write_text(
        f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{prompt}",
        encoding="utf-8",
    )

    results = []
    for model in models:
        if args.prompt_style == "incremental-display":
            results.append(
                await run_incremental_model(
                    model=model,
                    default_provider=default_provider,
                    ranking=ranking,
                    output_dir=args.output_dir,
                    max_tokens=args.max_tokens,
                    batch_size=args.batch_size,
                )
            )
        elif args.prompt_style == "batched-old-indexed":
            results.append(
                await run_batched_old_indexed_model(
                    model=model,
                    default_provider=default_provider,
                    ranking=ranking,
                    output_dir=args.output_dir,
                    max_tokens=args.max_tokens,
                    batch_size=args.batch_size,
                )
            )
        else:
            results.append(
                await run_model(
                    model=model,
                    default_provider=default_provider,
                    prompt=prompt,
                    ranking=ranking,
                    output_dir=args.output_dir,
                    max_tokens=args.max_tokens,
                    prompt_style=args.prompt_style,
                )
            )

    target_re = re.compile(args.target_regex, re.IGNORECASE) if args.target_regex else None
    repeated_re = (
        re.compile(args.repeated_family_regex, re.IGNORECASE)
        if args.repeated_family_regex
        else None
    )
    vetting_re = re.compile(args.vetting_regex, re.IGNORECASE) if args.vetting_regex else None

    summaries = []
    if args.render_dir:
        args.render_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", result.model)
        family_path = args.output_dir / f"{safe_model}.families.json"
        write_families(family_path, result)
        summary = {
            "model": result.model,
            "parse_success": result.parse_success,
            "parse_error": result.parse_error,
            "raw_output_path": result.raw_output_path,
            "families_path": str(family_path),
            "repair_counts": result.repair_counts,
            **family_metrics(
                result.families,
                top_k=args.top_k,
                target_re=target_re,
                repeated_re=repeated_re,
            ),
        }
        if args.render_dir:
            summary["png"] = render_families(args.render_dir / f"{safe_model}.png", result.families)
        summaries.append(summary)

    if args.vetting_json:
        write_vetting(args.vetting_json, results=results, pattern=vetting_re)

    output = {
        "artifact": str(args.artifact),
        "cluster_crawler_config": args.cluster_crawler_config,
        "base_ranking": "pairwise_proxy_elo",
        "prompt_style": args.prompt_style,
        "batch_size": args.batch_size,
        "max_terms": args.max_terms,
        "max_terms_per_cluster": args.max_terms_per_cluster,
        "models": models,
        "resolved_models": [result.model for result in results],
        "target_regex_provided": args.target_regex is not None,
        "repeated_family_regex_provided": args.repeated_family_regex is not None,
        "vetting_regex_provided": args.vetting_regex is not None,
        "prompt_path": str(args.output_dir / "aggregator_prompt.txt"),
        "summaries": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    printable = {
        **{key: value for key, value in output.items() if key != "summaries"},
        "summaries": [
            {
                key: value
                for key, value in summary.items()
                if key
                in {
                    "model",
                    "parse_success",
                    "parse_error",
                    "repair_counts",
                    "family_count",
                    "max_family_size",
                    "multi_member_families",
                    "target_coverage",
                    "repeated_family",
                }
            }
            for summary in summaries
        ],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
