"""Role-bucketed call breakdown for a crawler transcript JSONL.

Reuses the same heuristic as scripts/view_transcript.py:classify():
- qwen    -> translate
- kimi    -> extract (if system prompt mentions 'extract') or group (if 'deduplication') else other
- deepseek -> generate (mt >= 3000 or batch >= 10) else refusal-query
- gemma/refusal -> judge (if 'AI safety judge' in user) / provoke (if 'AI safety research') else other

Also reports total, per-role count, and per-call_type count.
Optionally compares against a baseline target shape (PLAN.md V2 targets).

Usage:
    python scripts/call_breakdown.py <transcript.jsonl> [--target]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Pull shared classification helpers so bench_cache_dedup.py imports
# the same object and identity checks (`is`) pass.
sys.path.insert(0, str(Path(__file__).parent))
from _transcript_classify import classify, first_user, first_sys  # noqa: E402


# PLAN.md target shape (V2). Per-role target ranges used as rough bands.
V1_BASELINE = {
    "generate": 4,         # target
    "extract": 10,         # kimi
    "translate": 214,      # qwen
    "group": 0,            # not in V1
    "provoke": 0,
    "refusal-query": 0,    # folded into judge via batched probes
    "judge": 116,          # gemma (10 provocations × heads + judge calls)
    "other": 0,
}

V2_TARGET = {
    "generate": 4,                 # exact
    "extract": 4,                  # one per generation
    "translate": 10,               # bulk per generation, <= 10
    "group": 6,                    # 4-8 kimi calls (batch=50)
    "provoke": 0,                  # unknown in PLAN, tracked separately
    "refusal-query": 0,            # probe calls
    "judge": 20,                   # 3-5 × novel heads (progressive), assume ~5 heads
    "other": 0,
}


def bucket_counts(records):
    role_counts = Counter()
    model_counts = Counter()
    call_type_counts = Counter()
    for r in records:
        role, fam = classify(r)
        role_counts[role] += 1
        model_counts[fam] += 1
        call_type_counts[r.get("call_type", "?")] += 1
    return role_counts, model_counts, call_type_counts


def fmt_row(label, actual, baseline, target):
    return f"  {label:<16} actual={actual:<6} v1={baseline:<6} v2_target={target}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", help="Path to transcript jsonl file")
    p.add_argument("--show-models", action="store_true", help="Also show per-model counts")
    args = p.parse_args(argv)

    path = Path(args.jsonl).resolve()
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    role_counts, model_counts, call_type_counts = bucket_counts(records)

    total = sum(role_counts.values())
    print(f"Transcript: {path}")
    print(f"Total calls: {total}\n")

    print("## Role breakdown (with V1 baseline and V2 target)")
    for role in ("generate", "extract", "translate", "group", "provoke",
                 "refusal-query", "judge", "other"):
        actual = role_counts.get(role, 0)
        baseline = V1_BASELINE.get(role, 0)
        target = V2_TARGET.get(role, 0)
        print(fmt_row(role, actual, baseline, target))

    v1_total = sum(V1_BASELINE.values())
    # PLAN.md V2 total target range is "~80-120".
    print(fmt_row("TOTAL", total, v1_total, f"80-120"))
    print()

    if args.show_models:
        print("## Per-model-family counts")
        for k, v in sorted(model_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<10} {v}")
        print()

    print("## Per-call_type counts")
    for k, v in sorted(call_type_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<30} {v}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
