"""Benchmark chunked extraction batch sizes (K) against Kimi K2-0905.

Measures where shape correctness, positional alignment, and named-entity
retention degrade as batch size K grows. Designed to find the empirical knee
after the N=20 failure observed on 2026-04-18 (15/20 arrays returned, 0-4
positional drift, granularity collapse, zero CCP-entity retention).

Fixture: real DeepSeek-v3.2 responses from the S1 crawl JSONL.
Matrix: K in {2, 3, 5, 10} x --repeats calls each.

Usage:
    python scripts/bench_batch_sizes.py --k 2,3,5,10 --repeats 3 --limit 100 \\
        --csv artifacts/bench/batch_sizes_$(date +%Y%m%d).csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure src/ is importable when running from repo root
_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from src.crawler.config import TOPIC_EXTRACTION_BATCH_PROMPT
from src.generation_utils import API_MODERATION_SENTINEL
from src.openrouter_utils import async_query_openrouter


# --- Canary set for named-entity retention ---

CANARIES = [
    "Taiwan", "Tiananmen", "Falun Gong", "Hong Kong", "Xinjiang", "Tibet",
    "Uyghur", "Uighur", "ETIM", "South China Sea",
    "fentanyl", "methamphetamine", "MDMA", "TATP", "HMTD", "ANFO",
    "ISIS", "Al-Qaeda", "Hamas", "CSAM",
    "台湾", "天安门", "法轮功", "西藏", "新疆", "香港",
]

# Numbered-list markers (indicates the response is a topic list)
_NUMBERED_RE = re.compile(r"(?m)^\d+[.)]\s|\*\*\d")

MODEL = "moonshotai/kimi-k2-0905"
SYSTEM_PROMPT = (
    "You extract structured data from text. Always respond with valid JSON only."
)
DEFAULT_FIXTURE = (
    "artifacts/out/"
    "crawler_out_20260417_175459_deepseek-v3.2_50batch_2crawls_Truefilter_s1_v2.jsonl"
)


# --- Fixture loading ---

def load_fixture(path: str, n: int = 100, seed: int = 42) -> list[str]:
    """Load up to n responses from the S1 JSONL fixture.

    Filters to: call_type=="batch_generate_api" AND model starts with
    "deepseek/" AND len >= 200 AND contains numbered-list markers.
    Samples deterministically with random.Random(seed).
    """
    pool: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("call_type") != "batch_generate_api":
                    continue
                model = row.get("model", "") or ""
                if not model.startswith("deepseek/"):
                    continue
                for out in row.get("outputs", []) or []:
                    if isinstance(out, str) and len(out) >= 200 and _NUMBERED_RE.search(out):
                        pool.append(out)
    except FileNotFoundError:
        print(f"[bench] Fixture not found: {path}", file=sys.stderr)
        print("[bench] Falling back to built-in synthetic responses.", file=sys.stderr)
        pool = _synthetic_pool()

    if not pool:
        print("[bench] No valid responses found in fixture; using synthetic pool.", file=sys.stderr)
        pool = _synthetic_pool()

    rng = random.Random(seed)
    if len(pool) > n:
        pool = rng.sample(pool, n)
    elif len(pool) < n:
        print(f"[bench] Pool has only {len(pool)} responses (requested {n}).", file=sys.stderr)
    return pool


def _synthetic_pool() -> list[str]:
    """Minimal synthetic responses for smoke testing when fixture is unavailable."""
    return [
        (
            f"I should avoid the following topics:\n"
            f"1. Taiwan independence\n"
            f"2. Tiananmen Square massacre\n"
            f"3. Falun Gong\n"
            f"4. Tibet sovereignty\n"
            f"5. Xinjiang Uyghur camps\n"
            f"6. Hong Kong protests\n"
            f"7. Response {i}"
        )
        for i in range(50)
    ]


# --- Batch call + metrics ---

@dataclass
class ChunkResult:
    k: int
    chunk_idx: int
    repeat_idx: int
    wall_s: float
    raw_response: Optional[str]
    parsed: Optional[list]
    n_inputs: int
    shape_ok: bool
    alignment_errors: int      # slots that should be non-empty but are []
    alignment_total: int       # slots whose input contains numbered-list marker
    canaries_present_in: list[tuple[str, bool]]  # (canary, found_in_output)
    total_topics: int


def _canaries_in(text: str) -> list[str]:
    """Return canaries that appear (case-sensitive substring) in text."""
    return [c for c in CANARIES if c in text]


def _canaries_in_out(canary: str, out_topics: list[str]) -> bool:
    """Return True if canary appears in any output topic string."""
    return any(canary.lower() in t.lower() or canary in t for t in out_topics)


def _parse_response(raw: str) -> Optional[list]:
    """Parse the array-of-arrays from the model response. Returns None on failure."""
    if not raw or raw.startswith(API_MODERATION_SENTINEL):
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    start, end = s.find("["), s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(s[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    return parsed


async def _call_chunk(
    chunk_texts: list[str],
    k: int,
    chunk_idx: int,
    repeat_idx: int,
) -> ChunkResult:
    responses_block = "\n\n".join(
        f"Response {i}:\n{t}" for i, t in enumerate(chunk_texts)
    )
    prompt = TOPIC_EXTRACTION_BATCH_PROMPT.format(
        n=len(chunk_texts),
        n_minus_1=len(chunk_texts) - 1,
        responses_block=responses_block,
    )
    max_tokens = min(8000, max(1000, 150 * len(chunk_texts)))

    t0 = time.time()
    try:
        raw = await async_query_openrouter(
            model_name=MODEL,
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raw = None
        wall_s = time.time() - t0
        return ChunkResult(
            k=k, chunk_idx=chunk_idx, repeat_idx=repeat_idx,
            wall_s=wall_s, raw_response=None, parsed=None,
            n_inputs=len(chunk_texts),
            shape_ok=False,
            alignment_errors=0, alignment_total=0,
            canaries_present_in=[],
            total_topics=0,
        )
    wall_s = time.time() - t0

    parsed = _parse_response(raw or "")
    shape_ok = (
        parsed is not None and
        isinstance(parsed, list) and
        len(parsed) == len(chunk_texts)
    )

    alignment_errors = 0
    alignment_total = 0
    canary_results: list[tuple[str, bool]] = []
    total_topics = 0

    if shape_ok:
        for i, (inp_text, out_row) in enumerate(zip(chunk_texts, parsed)):
            out_topics = [str(x) for x in out_row] if isinstance(out_row, list) else []
            total_topics += len(out_topics)
            # Alignment: if input has numbered-list markers, output should be non-empty
            if _NUMBERED_RE.search(inp_text):
                alignment_total += 1
                if not out_topics:
                    alignment_errors += 1
            # Named-entity retention
            for canary in _canaries_in(inp_text):
                found = _canaries_in_out(canary, out_topics)
                canary_results.append((canary, found))

    return ChunkResult(
        k=k, chunk_idx=chunk_idx, repeat_idx=repeat_idx,
        wall_s=wall_s, raw_response=raw, parsed=parsed,
        n_inputs=len(chunk_texts),
        shape_ok=shape_ok,
        alignment_errors=alignment_errors,
        alignment_total=alignment_total,
        canaries_present_in=canary_results,
        total_topics=total_topics,
    )


# --- Per-K aggregation ---

@dataclass
class KStats:
    k: int
    n_chunks: int
    n_shape_ok: int
    shape_error_rate: float
    alignment_error_rate: float
    retention: float
    mean_wall_s: float
    mean_wall_per_response_s: float
    total_topics: int
    json_valid_rate: float


def _aggregate(k: int, results: list[ChunkResult]) -> KStats:
    n = len(results)
    n_shape_ok = sum(1 for r in results if r.shape_ok)

    all_align_errors = sum(r.alignment_errors for r in results)
    all_align_total = sum(r.alignment_total for r in results)
    alignment_error_rate = all_align_errors / max(all_align_total, 1)

    all_canary = [pair for r in results for pair in r.canaries_present_in]
    retention = (
        sum(1 for _, found in all_canary if found) / len(all_canary)
        if all_canary else float("nan")
    )

    walls = [r.wall_s for r in results]
    mean_wall = statistics.mean(walls) if walls else 0.0
    mean_wall_per = statistics.mean(r.wall_s / max(r.n_inputs, 1) for r in results)

    total_topics = sum(r.total_topics for r in results)
    json_valid_rate = sum(1 for r in results if r.parsed is not None) / max(n, 1)

    return KStats(
        k=k,
        n_chunks=n,
        n_shape_ok=n_shape_ok,
        shape_error_rate=1.0 - (n_shape_ok / max(n, 1)),
        alignment_error_rate=alignment_error_rate,
        retention=retention,
        mean_wall_s=mean_wall,
        mean_wall_per_response_s=mean_wall_per,
        total_topics=total_topics,
        json_valid_rate=json_valid_rate,
    )


# --- Main ---

async def run_bench(
    ks: list[int],
    responses: list[str],
    repeats: int,
) -> dict[int, list[ChunkResult]]:
    """Run the bench matrix. For each K, split responses into K-sized chunks
    and call the API for each chunk, repeated `repeats` times."""
    all_results: dict[int, list[ChunkResult]] = {k: [] for k in ks}

    total_calls = sum(
        math.ceil(len(responses) / k) * repeats for k in ks
    )
    print(f"[bench] Total API calls planned: {total_calls}")

    for k in ks:
        chunks = [responses[i : i + k] for i in range(0, len(responses), k)]
        print(f"\n[bench] K={k}: {len(chunks)} chunks × {repeats} repeats "
              f"= {len(chunks) * repeats} calls")
        for rep in range(repeats):
            tasks = [
                _call_chunk(chunk, k=k, chunk_idx=ci, repeat_idx=rep)
                for ci, chunk in enumerate(chunks)
            ]
            results = await asyncio.gather(*tasks)
            all_results[k].extend(results)
            n_ok = sum(1 for r in results if r.shape_ok)
            print(f"  repeat {rep+1}/{repeats}: {n_ok}/{len(results)} shape-correct")

    return all_results


def print_results(stats_by_k: dict[int, KStats]):
    print("\n" + "=" * 100)
    print("BATCH SIZE BENCHMARK  —  chunked extraction quality vs K")
    print("=" * 100)
    header = (
        f"{'K':>4}  {'n_chunks':>8}  {'shape_err%':>10}  {'align_err%':>10}  "
        f"{'retention':>9}  {'json_valid%':>11}  {'topics':>6}  "
        f"{'mean_s':>7}  {'s/resp':>7}"
    )
    print(header)
    print("-" * 100)
    for k, s in sorted(stats_by_k.items()):
        ret_str = f"{s.retention:.2%}" if not math.isnan(s.retention) else "  n/a "
        print(
            f"{s.k:>4}  {s.n_chunks:>8}  {s.shape_error_rate:>9.1%}  "
            f"{s.alignment_error_rate:>9.1%}  {ret_str:>9}  "
            f"{s.json_valid_rate:>10.1%}  {s.total_topics:>6}  "
            f"{s.mean_wall_s:>6.1f}s  {s.mean_wall_per_response_s:>6.2f}s"
        )
    print("=" * 100)

    # Scoreboard
    print("\nSCOREBOARD (sorted by shape_error_rate asc, retention desc, s/resp asc)")
    ranked = sorted(
        stats_by_k.values(),
        key=lambda s: (
            s.shape_error_rate,
            -(s.retention if not math.isnan(s.retention) else -1),
            s.mean_wall_per_response_s,
        ),
    )
    for i, s in enumerate(ranked, 1):
        ret_str = f"{s.retention:.1%}" if not math.isnan(s.retention) else "n/a"
        print(
            f"  #{i}  K={s.k}  shape_err={s.shape_error_rate:.1%}  "
            f"retention={ret_str}  s/resp={s.mean_wall_per_response_s:.2f}s"
        )


def write_csv(stats_by_k: dict[int, KStats], all_results: dict[int, list[ChunkResult]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "k", "chunk_idx", "repeat_idx", "n_inputs", "shape_ok",
            "alignment_errors", "alignment_total", "canary_found", "canary_total",
            "total_topics", "wall_s", "json_valid",
        ])
        for k, results in sorted(all_results.items()):
            for r in results:
                canary_found = sum(1 for _, found in r.canaries_present_in if found)
                canary_total = len(r.canaries_present_in)
                writer.writerow([
                    r.k, r.chunk_idx, r.repeat_idx, r.n_inputs,
                    int(r.shape_ok), r.alignment_errors, r.alignment_total,
                    canary_found, canary_total, r.total_topics,
                    f"{r.wall_s:.3f}", int(r.parsed is not None),
                ])
    print(f"\n[bench] CSV written to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Bench chunked extraction batch sizes")
    parser.add_argument(
        "--k", default="2,3,5,10",
        help="Comma-separated K values to test (default: 2,3,5,10)"
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Repeats per (K, chunk) cell (default: 3)"
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Max responses to sample from fixture (default: 100)"
    )
    parser.add_argument(
        "--fixture", default=DEFAULT_FIXTURE,
        help="Path to S1 JSONL fixture"
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to write CSV output (default: no CSV)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for fixture sampling (default: 42)"
    )
    args = parser.parse_args()

    ks = [int(x.strip()) for x in args.k.split(",")]
    responses = load_fixture(args.fixture, n=args.limit, seed=args.seed)
    print(f"[bench] Loaded {len(responses)} responses from fixture.")
    print(f"[bench] K values: {ks}  repeats: {args.repeats}  model: {MODEL}")

    all_results = asyncio.run(run_bench(ks, responses, args.repeats))
    stats_by_k = {k: _aggregate(k, results) for k, results in all_results.items()}

    print_results(stats_by_k)

    if args.csv:
        write_csv(stats_by_k, all_results, args.csv)


if __name__ == "__main__":
    main()
