"""Multi-cache hit-rate measurement for crawler transcript JSONL files.

For each of four cache-candidate types (judge, summarize, translate, extract),
compute how many API calls a per-run in-memory cache would avoid, and project
the dollar savings.  A type passes the S4c gate (and is a candidate for S4d
implementation) if potential savings >= 10%.

The S4d caches reset between crawl runs, so dedup is computed PER RUN, not
globally across all input files.  Cross-run dedup would never hit in production.

Usage:
    python scripts/bench_cache_dedup.py --run path/to/run1.jsonl path/to/run2.jsonl
    python scripts/bench_cache_dedup.py --run *.jsonl --output results.json

Output:
    Stdout: Markdown table + summary block (per-run rows, then aggregate).
    --output: JSON artifact with per_run and aggregate numeric results.

Constraints:
    - No live API calls.
    - Classification is imported from scripts/_transcript_classify.py (shared
      with scripts/call_breakdown.py) — never duplicated.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

# Reach the shared helper without requiring scripts/ to be a package.
sys.path.insert(0, str(Path(__file__).parent))
from _transcript_classify import classify, first_user, first_sys  # noqa: E402

# ---------------------------------------------------------------------------
# Pricing table
# List-price estimates as of ~2026-04; these may have drifted.
# Prices are (input_per_1M_usd, output_per_1M_usd).
# ---------------------------------------------------------------------------
PRICING = {
    "openai/gpt-5.4-mini": (0.15, 0.60),
    "openai/gpt-5.4": (1.25, 10.00),
    "moonshotai/kimi-k2-0905": (0.60, 2.50),
    "moonshotai/kimi-k2.5": (0.60, 2.50),
    "deepseek/deepseek-v3.2": (0.27, 1.10),
    "qwen/qwen3-235b-a22b-2507": (0.13, 0.60),
    "google/gemma-4-26b-a4b-it": (0.10, 0.40),
    "google/gemini-3-flash-preview": (0.10, 0.40),
}

# The four types for which we measure potential cache savings.
CACHE_CANDIDATE_TYPES = ("judge", "summarize", "translate", "extract")


def _pricing_for_model(model: str):
    """Return (input_per_1M, output_per_1M) for model, or (0.0, 0.0) if unknown.

    Strips :nitro / :floor suffixes before prefix-matching.
    """
    clean = model.rstrip(":nitro").rstrip(":floor")
    # Also handle suffix directly
    for suffix in (":nitro", ":floor"):
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
    for prefix, prices in PRICING.items():
        if clean.startswith(prefix) or model.startswith(prefix):
            return prices
    return (0.0, 0.0)


def _sha256_prompt(user_prompt: str, system_prompt: str) -> str:
    combined = (user_prompt + system_prompt).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


def _extract_prompt_pair(inputs):
    """Return (user_prompt, system_prompt) from an inputs structure.

    inputs may be a single message list or a list-of-message-lists (batch).
    This function always returns the FIRST item's prompts for classification;
    for key computation, use _expand_batch_items().
    """
    return first_user(inputs), first_sys(inputs)


def _expand_batch_items(inputs):
    """Yield (user_prompt, system_prompt) for each item in a batch.

    For a non-batched inputs (flat list of dicts), yields a single pair.
    For a batched inputs (list of lists), yields one pair per sub-list.
    """
    if not isinstance(inputs, list) or not inputs:
        yield ("", "")
        return
    if isinstance(inputs[0], list):
        # List-of-message-lists: each element is one call.
        for msg_list in inputs:
            u = ""
            s = ""
            for m in msg_list:
                if isinstance(m, dict):
                    if m.get("role") == "user":
                        u = m.get("content") or ""
                    elif m.get("role") == "system":
                        s = m.get("content") or ""
            yield (u, s)
    else:
        # Single call.
        u = ""
        s = ""
        for m in inputs:
            if isinstance(m, dict):
                if m.get("role") == "user":
                    u = m.get("content") or ""
                elif m.get("role") == "system":
                    s = m.get("content") or ""
        yield (u, s)


def _char_count_for_record(inputs, outputs) -> tuple[int, int]:
    """Return (input_chars, output_chars) for a record.

    For batched inputs (list-of-message-lists), sums char counts across ALL
    items, not just the first.  For single-item inputs, behaviour is unchanged.

    outputs is already a flat list of strings for batched records, so the
    existing sum covers the full batch payload — no change needed there.
    """
    in_chars = sum(
        len(u) + len(s)
        for u, s in _expand_batch_items(inputs)
    )
    if isinstance(outputs, list):
        out_chars = sum(len(o) for o in outputs if isinstance(o, str))
    elif isinstance(outputs, str):
        out_chars = len(outputs)
    else:
        out_chars = 0
    return in_chars, out_chars


def _is_batch_record(rec) -> bool:
    inputs = rec.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        return False
    return isinstance(inputs[0], list)


def _is_target_batch(rec) -> bool:
    """Return True when this record is a target-model batch_generate_api call.

    The target-model is identified by classify() returning 'generate'.  This
    matches deepseek (or any model) calls where mt >= 3000 or batch >= 10.
    We use call_type == "batch_generate_api" as a secondary guard so that
    non-API target calls are excluded.
    """
    role, _ = classify(rec)
    return role == "generate" and rec.get("call_type") == "batch_generate_api"


# ---------------------------------------------------------------------------
# Step bucketing
# ---------------------------------------------------------------------------


def _assign_step_buckets(records: list) -> list[int]:
    """Assign a step-bucket index to each record based on target-batch boundaries.

    Records are processed in JSONL insertion order, which matches timestamp order
    in practice because log_model_call appends on call completion.  No explicit
    sort by timestamp is applied; insertion order is the authoritative sequence.

    Returns a list of bucket indices (same length as records).
    Bucket 0 = calls before the first target batch ("pre" / init-translate).
    Bucket N = calls at or after target-batch N (a target-batch record is placed
    in the bucket it opens, i.e. a target-batch increments the counter and is
    itself placed in the new bucket).

    If there are no target batch calls, all records are in bucket 0, and the
    caller sets a fallback note.  In this case intra_step_dedup equals the
    total duplicate count and cross_step_dedup equals 0 (all in one bucket).
    """
    bucket = 0
    buckets = []
    for rec in records:
        if _is_target_batch(rec):
            bucket += 1
        buckets.append(bucket)
    return buckets


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------


def _analyze_one_run(records: list) -> dict:
    """Analyze a single run's records.

    Returns a dict with:
        "total_calls": int
        "per_type": dict mapping type_name -> {
            "total_calls": int,
            "unique_keys": int,
            "savings_pct": float,
            "savings_calls": int,
            "projected_usd_saved": float,
            "decision": str,
            "avg_input_chars": float,
            "avg_output_chars": float,
            "intra_step_dedup": int,
            "cross_step_dedup": int,
            "step_note": str,
        }
        "has_target_batches": bool
        "step_note": str
    """
    # Assign step buckets; detect fallback condition.
    buckets = _assign_step_buckets(records)
    has_target_batches = any(b > 0 for b in buckets) if buckets else False
    if not has_target_batches:
        step_note = "step inference unavailable: no target batch_generate_api found"
    else:
        step_note = ""

    # Per-type accumulators.
    type_keys: dict[str, list] = defaultdict(list)        # all keys (one per expanded call)
    type_key_sets: dict[str, set] = defaultdict(set)      # unique keys
    type_in_chars: dict[str, int] = defaultdict(int)
    type_out_chars: dict[str, int] = defaultdict(int)
    type_model: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    # Step-breakdown: for each type, track which step each key first appeared in.
    type_key_first_step: dict[str, dict] = defaultdict(dict)  # key -> step_bucket
    type_intra_step: dict[str, int] = defaultdict(int)    # intra-step duplicates
    type_cross_step: dict[str, int] = defaultdict(int)    # cross-step duplicates

    for rec, step_bucket in zip(records, buckets):
        role, _ = classify(rec)
        if role not in CACHE_CANDIDATE_TYPES:
            continue

        model = (rec.get("model") or "").lower()
        temperature = rec.get("temperature") or 0
        max_tokens = rec.get("max_tokens") or 0
        inputs = rec.get("inputs", [])
        outputs = rec.get("outputs", [])

        in_c, out_c = _char_count_for_record(inputs, outputs)

        if _is_batch_record(rec):
            # Expand batch: each sub-list is one call for key computation.
            items = list(_expand_batch_items(inputs))
            n = len(items)
            # Distribute char counts evenly across items.
            per_in = in_c / n if n else 0
            per_out = out_c / n if n else 0
            for user_prompt, sys_prompt in items:
                key = (model, _sha256_prompt(user_prompt, sys_prompt), temperature, max_tokens)
                type_keys[role].append(key)
                _classify_step_dedup(
                    role, key, step_bucket,
                    type_key_sets, type_key_first_step,
                    type_intra_step, type_cross_step,
                )
                type_in_chars[role] += per_in
                type_out_chars[role] += per_out
                type_model[role][model] += 1
        else:
            user_prompt, sys_prompt = _extract_prompt_pair(inputs)
            key = (model, _sha256_prompt(user_prompt, sys_prompt), temperature, max_tokens)
            type_keys[role].append(key)
            _classify_step_dedup(
                role, key, step_bucket,
                type_key_sets, type_key_first_step,
                type_intra_step, type_cross_step,
            )
            type_in_chars[role] += in_c
            type_out_chars[role] += out_c
            type_model[role][model] += 1

    per_type = {}
    for t in CACHE_CANDIDATE_TYPES:
        total = len(type_keys[t])
        unique = len(type_key_sets[t])
        savings_calls = total - unique
        savings_pct = round((1.0 - unique / total) * 100.0, 6) if total > 0 else 0.0
        avg_in = type_in_chars[t] / total if total > 0 else 0.0
        avg_out = type_out_chars[t] / total if total > 0 else 0.0

        est_in_tokens_saved = savings_calls * avg_in / 4.0
        est_out_tokens_saved = savings_calls * avg_out / 4.0

        projected_usd = 0.0
        if type_model[t]:
            for model_id, count in type_model[t].items():
                frac = count / total if total > 0 else 0
                in_price, out_price = _pricing_for_model(model_id)
                projected_usd += (
                    frac * est_in_tokens_saved * in_price / 1_000_000.0
                    + frac * est_out_tokens_saved * out_price / 1_000_000.0
                )

        if total == 0:
            decision = "N/A (no calls)"
        elif savings_pct >= 10.0:
            decision = "SHIP (>= 10%)"
        else:
            decision = "DROP"

        intra = type_intra_step[t]
        cross = type_cross_step[t]
        # Sanity: intra + cross must equal savings_calls.
        assert intra + cross == savings_calls, (
            f"Step dedup sanity failed for {t}: {intra} + {cross} != {savings_calls}"
        )

        t_step_note = step_note if total > 0 else ""

        per_type[t] = {
            "total_calls": total,
            "unique_keys": unique,
            "savings_pct": savings_pct,
            "savings_calls": savings_calls,
            "projected_usd_saved": projected_usd,
            "decision": decision,
            "avg_input_chars": avg_in,
            "avg_output_chars": avg_out,
            "intra_step_dedup": intra,
            "cross_step_dedup": cross,
            "step_note": t_step_note,
        }

    return {
        "total_calls": len(records),
        "per_type": per_type,
        "has_target_batches": has_target_batches,
        "step_note": step_note,
    }


def _classify_step_dedup(
    role: str,
    key,
    step_bucket: int,
    type_key_sets: dict,
    type_key_first_step: dict,
    type_intra_step: dict,
    type_cross_step: dict,
) -> None:
    """Classify a single call-key as intra-step or cross-step duplicate (or unique).

    Modifies type_key_sets, type_key_first_step, type_intra_step, type_cross_step
    in place.
    """
    if key not in type_key_sets[role]:
        # First occurrence — not a duplicate.
        type_key_sets[role].add(key)
        type_key_first_step[role][key] = step_bucket
    else:
        # Duplicate — determine whether it's intra or cross-step.
        first_step = type_key_first_step[role][key]
        if first_step == step_bucket:
            type_intra_step[role] += 1
        else:
            type_cross_step[role] += 1


# ---------------------------------------------------------------------------
# Core analysis: per-run + aggregate
# ---------------------------------------------------------------------------


def analyze_transcripts(paths):
    """Analyze one or more JSONL transcript paths.

    Returns a dict with:
        "per_run": list of {"path": str, "total_calls": int, "results": <run_result>}
        "aggregate": {
            "total_calls": int,
            "per_type": dict mapping type_name -> {
                "total_calls": int,
                "unique_keys": int,         # sum of per-run unique_keys
                "savings_pct": float,       # sum(per_run savings_calls) / sum(per_run total)
                "savings_calls": int,       # sum of per-run savings_calls
                "projected_usd_saved": float,
                "decision": str,
            }
        }
    """
    per_run_results = []

    for path in paths:
        path = Path(path)
        recs = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        run_result = _analyze_one_run(recs)
        per_run_results.append({
            "path": str(path),
            "total_calls": run_result["total_calls"],
            "results": run_result,
        })

    # Build aggregate: sum per-run totals/savings; DON'T re-dedup across runs.
    agg_total_calls = sum(r["total_calls"] for r in per_run_results)
    agg_per_type = {}
    for t in CACHE_CANDIDATE_TYPES:
        agg_total = sum(
            r["results"]["per_type"][t]["total_calls"] for r in per_run_results
        )
        agg_unique = sum(
            r["results"]["per_type"][t]["unique_keys"] for r in per_run_results
        )
        agg_savings_calls = sum(
            r["results"]["per_type"][t]["savings_calls"] for r in per_run_results
        )
        agg_savings_pct = (
            round(agg_savings_calls / agg_total * 100.0, 6) if agg_total > 0 else 0.0
        )
        agg_usd = sum(
            r["results"]["per_type"][t]["projected_usd_saved"] for r in per_run_results
        )
        if agg_total == 0:
            agg_decision = "N/A (no calls)"
        elif agg_savings_pct >= 10.0:
            agg_decision = "SHIP (>= 10%)"
        else:
            agg_decision = "DROP"

        agg_per_type[t] = {
            "total_calls": agg_total,
            "unique_keys": agg_unique,
            "savings_pct": agg_savings_pct,
            "savings_calls": agg_savings_calls,
            "projected_usd_saved": agg_usd,
            "decision": agg_decision,
        }

    return {
        "per_run": per_run_results,
        "aggregate": {
            "total_calls": agg_total_calls,
            "per_type": agg_per_type,
        },
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt_pct(pct: float, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{pct:.1f}%"


def _fmt_step_pct(value: int, total: int) -> str:
    """Format a step-dedup count as a percentage of total calls."""
    if total == 0:
        return "n/a"
    return f"{value / total * 100:.1f}%"


def print_report(result: dict) -> None:
    """Print Markdown-formatted report to stdout."""
    per_run = result["per_run"]
    aggregate = result["aggregate"]

    print("## Per-run Results\n")

    for run_entry in per_run:
        path = run_entry["path"]
        run_res = run_entry["results"]
        total_calls = run_res["total_calls"]
        per_type = run_res["per_type"]
        step_note = run_res.get("step_note", "")

        print(f"### Run: {path}  ({total_calls} total calls)")
        if step_note:
            print(f"  Note: {step_note}")
        print()

        header = (
            "| type      | total | unique | savings% | savings | $saved "
            "| intra_step% | cross_step% | decision |"
        )
        sep = (
            "|-----------|-------|--------|----------|---------|--------"
            "|-------------|-------------|----------|"
        )
        print(header)
        print(sep)
        for t in CACHE_CANDIDATE_TYPES:
            d = per_type[t]
            total = d["total_calls"]
            unique = d["unique_keys"]
            savings_pct = _fmt_pct(d["savings_pct"], total)
            savings_calls = d["savings_calls"]
            proj_usd = f"${d['projected_usd_saved']:.3f}"
            intra = d["intra_step_dedup"]
            cross_ = d["cross_step_dedup"]
            intra_pct = _fmt_step_pct(intra, total)
            cross_pct = _fmt_step_pct(cross_, total)
            decision = d["decision"]
            print(
                f"| {t:<9} | {total:<5} | {unique:<6} | {savings_pct:<8} | "
                f"{savings_calls:<7} | {proj_usd:<6} | {intra_pct:<11} | "
                f"{cross_pct:<11} | {decision} |"
            )
        print()

    print("## Aggregate (sum across runs; dedup is per-run, not cross-run)\n")
    agg_total = aggregate["total_calls"]
    print(f"- Total calls across {len(per_run)} run(s): {agg_total}")
    print()

    header = (
        "| type      | total | unique | savings% | savings | $saved | decision |"
    )
    sep = (
        "|-----------|-------|--------|----------|---------|--------|----------|"
    )
    print(header)
    print(sep)
    for t in CACHE_CANDIDATE_TYPES:
        d = aggregate["per_type"][t]
        total = d["total_calls"]
        unique = d["unique_keys"]
        savings_pct = _fmt_pct(d["savings_pct"], total)
        savings_calls = d["savings_calls"]
        proj_usd = f"${d['projected_usd_saved']:.3f}"
        decision = d["decision"]
        print(
            f"| {t:<9} | {total:<5} | {unique:<6} | {savings_pct:<8} | "
            f"{savings_calls:<7} | {proj_usd:<6} | {decision} |"
        )
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run",
        metavar="PATH",
        nargs="+",
        required=True,
        help="One or more *.jsonl transcript paths to analyze.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Optional path to write JSON artifact with per-type numeric results.",
    )
    args = parser.parse_args(argv)

    result = analyze_transcripts(args.run)
    print_report(result)

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"JSON artifact written to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
