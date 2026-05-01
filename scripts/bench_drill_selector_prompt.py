#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.openrouter_utils import REASONING_DISABLED, async_query_openrouter
from src.provider_config import get_provider_client_kwargs


GENERIC_SELECTOR_INSTRUCTIONS = """You are a helpful summarizer model.
You will be given a taxonomy of "sensitive topics" from a generated AI response.
Your job is to extract broad "sensitive topics" from the following taxonomy.

Do NOT select topics that are specific instances of sensitive topics.
Select only topics that are general categories.
Sort by how broad a category is, with the broadest category first.

Return a JSON array of strings, with each string containing a single topic.
"""


BROAD_REPORT_RE = re.compile(
    r"politic|histor|geopolit|sovereignty|territor|separat|secession|"
    r"state power|subversion|national security|social stability|censor|"
    r"information control|ideolog|sensitive|政治|历史|地缘|主权|领土|"
    r"民族|区域|司法|军事|国防|经济|对外关系|信息|管控|意识形态|敏感",
    re.IGNORECASE,
)


def _extract_json_array(text: str) -> list[str]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, str)]
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\[.*\])", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in model output")
    parsed = json.loads(match.group(1))
    if not isinstance(parsed, list):
        raise ValueError("Parsed JSON is not an array")
    return [item for item in parsed if isinstance(item, str)]


def _build_prompt(labels: list[str]) -> str:
    return (
        GENERIC_SELECTOR_INSTRUCTIONS
        + "\n\nINPUT JSON:\n"
        + json.dumps(labels, ensure_ascii=False, indent=2)
    )


def _build_text_prompt(texts: list[tuple[str, str]]) -> str:
    joined = "\n\n".join(
        f"--- SOURCE: {name} ---\n{text.strip()}"
        for name, text in texts
    )
    return GENERIC_SELECTOR_INSTRUCTIONS + "\n\nINPUT TAXONOMY:\n" + joined


def _load_fixture_texts(paths: list[str]) -> list[tuple[str, str]]:
    return [
        (Path(path).name, Path(path).read_text(encoding="utf-8"))
        for path in paths
    ]


def _summarize(
    *,
    model: str,
    raw_text: str,
    parsed: list[str] | None,
    parse_error: str | None,
    clusters_by_label: dict[str, dict[str, Any]],
    drill_seed_labels: set[str],
) -> dict[str, Any]:
    selected_labels = parsed or []
    valid_labels = [label for label in selected_labels if label in clusters_by_label]
    invalid_labels = [label for label in selected_labels if label not in clusters_by_label]
    valid_unique = list(dict.fromkeys(valid_labels))
    duplicate_valid_labels = len(valid_labels) - len(valid_unique)
    selected_clusters = [clusters_by_label[label] for label in valid_unique]
    broad_labels = [label for label in valid_unique if BROAD_REPORT_RE.search(label)]

    return {
        "model": model,
        "parse_success": parsed is not None,
        "parse_error": parse_error,
        "raw_output": raw_text,
        "parsed": parsed,
        "post_filtered_labels": valid_unique,
        "metrics": {
            "selected_labels": len(selected_labels),
            "valid_selected_labels": len(valid_labels),
            "unique_valid_selected_labels": len(valid_unique),
            "invalid_selected_labels": len(invalid_labels),
            "duplicate_valid_labels": duplicate_valid_labels,
            "selected_singletons": sum(1 for c in selected_clusters if c.get("size") == 1),
            "selected_validated": sum(1 for c in selected_clusters if c.get("validated")),
            "selected_current_drill_seeds": sum(1 for label in valid_unique if label in drill_seed_labels),
            "selected_broad_report_labels": len(broad_labels),
        },
        "invalid_labels": invalid_labels,
        "broad_report_labels": broad_labels,
    }


def _summarize_text_fixture(
    *,
    model: str,
    raw_text: str,
    parsed: list[str] | None,
    parse_error: str | None,
) -> dict[str, Any]:
    selected_labels = parsed or []
    valid_unique = list(dict.fromkeys(selected_labels))
    broad_labels = [label for label in valid_unique if BROAD_REPORT_RE.search(label)]
    return {
        "model": model,
        "parse_success": parsed is not None,
        "parse_error": parse_error,
        "raw_output": raw_text,
        "parsed": parsed,
        "post_filtered_labels": valid_unique,
        "metrics": {
            "selected_labels": len(selected_labels),
            "unique_selected_labels": len(valid_unique),
            "duplicate_selected_labels": len(selected_labels) - len(valid_unique),
            "selected_broad_report_labels": len(broad_labels),
        },
        "broad_report_labels": broad_labels,
    }


async def _call_model(model: str, prompt: str, *, default_provider: str, max_tokens: int) -> tuple[str, str]:
    resolved_model, client_kwargs = get_provider_client_kwargs(model, default_provider, None)
    text = await async_query_openrouter(
        model_name=resolved_model,
        prompt=prompt,
        system_prompt="You select drill-down seeds from labels. Respond with valid JSON only.",
        temperature=0.0,
        max_tokens=max_tokens,
        client_kwargs=client_kwargs,
        prefer_nitro=True,
        extra_body=REASONING_DISABLED,
    )
    return resolved_model, text


async def main_async(args: argparse.Namespace) -> None:
    fixture_texts = _load_fixture_texts(args.fixture_texts) if args.fixture_texts else []
    if fixture_texts:
        prompt = _build_text_prompt(fixture_texts)
        artifact = None
        labels = []
        clusters_by_label = {}
        drill_seed_labels = set()
    else:
        artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
        clusters = artifact["clusters"]
        labels = sorted(
            {cluster["label"] for cluster in clusters if cluster.get("label")},
            key=str.casefold,
        )
        clusters_by_label = {cluster["label"]: cluster for cluster in clusters if cluster.get("label")}
        drill_seed_labels = set(artifact.get("drill_seed_labels") or [])
        prompt = _build_prompt(labels)

    results = []
    for model in args.models:
        resolved, raw_text = await _call_model(
            model,
            prompt,
            default_provider=args.default_provider,
            max_tokens=args.max_tokens,
        )
        parsed: list[str] | None = None
        parse_error = None
        try:
            parsed = _extract_json_array(raw_text)
        except Exception as exc:  # noqa: BLE001 - recorded in benchmark artifact
            parse_error = str(exc)
        if fixture_texts:
            results.append(
                _summarize_text_fixture(
                    model=resolved,
                    raw_text=raw_text,
                    parsed=parsed,
                    parse_error=parse_error,
                )
            )
        else:
            results.append(
                _summarize(
                    model=resolved,
                    raw_text=raw_text,
                    parsed=parsed,
                    parse_error=parse_error,
                    clusters_by_label=clusters_by_label,
                    drill_seed_labels=drill_seed_labels,
                )
            )

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": None if fixture_texts else args.artifact,
        "fixture_texts": [name for name, _ in fixture_texts],
        "models": args.models,
        "prompt": GENERIC_SELECTOR_INSTRUCTIONS,
        "input_counts": {
            "clusters": 0 if fixture_texts else len(artifact["clusters"]),
            "candidate_labels": len(labels),
            "current_drill_seed_labels": len(drill_seed_labels),
            "fixture_texts": len(fixture_texts),
        },
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    safe_summary = {
        "output": str(out_path),
        "input_counts": output["input_counts"],
        "models": [
            {
                "model": r["model"],
                "parse_success": r["parse_success"],
                "metrics": r["metrics"],
                "post_filtered_labels": []
                if args.hide_labels
                else r["post_filtered_labels"],
                "broad_report_labels": r["broad_report_labels"],
            }
            for r in results
        ],
    }
    print(json.dumps(safe_summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        default="artifacts/out/cluster_smoke/live_smoke_jailbreak_2x2_drill48_8c_backup.json",
    )
    parser.add_argument(
        "--fixture-texts",
        nargs="+",
        default=None,
        help="Use raw taxonomy fixture text instead of cluster labels.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen/qwen3-235b-a22b-2507", "moonshotai/kimi-k2.5"],
    )
    parser.add_argument("--default-provider", default="openrouter")
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Suppress selected labels in stdout summary; full artifact still records them.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/research/drill_selector_prompt/selector_prompt_benchmark_48seed.json",
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
