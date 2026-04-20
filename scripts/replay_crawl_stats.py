"""Test 0 — Offline crawl-budget analysis + 10-step projection.

Goal: capture the empirical shape of a crawl so we can see where cost goes and
where further savings exist. Diagnoses:
  - Where does the cost actually land (by role: target, extractor, judge, ...)?
  - What's the empirical dedup rate (unique heads / total extracted)?
  - Do novel heads flatten over steps (coupon-collector)? At what rate?
  - If we ran N=10 at this decay rate, what's the total API call budget?
  - If we tightened M (max_topics_per_step_lang), does it save or just reshape?

Inputs:
  --run PATH    .json checkpoint (crawler_out_*.json) — has queue + stats history
                The matching .jsonl is read automatically (same basename).

Usage:
  python scripts/replay_crawl_stats.py \\
      --run artifacts/out/crawler_out_20260417_175459_..._s1_v2.json \\
      --project-steps 10

No API calls. Pure replay / projection.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))


def load_bundle(json_path: Path):
    data = json.loads(json_path.read_text())
    jsonl_path = json_path.with_suffix(".jsonl")
    records = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return data, records


def classify_role(call_type: str, model: str, target_model: str) -> str:
    """Bucket each log record into a role for budget attribution."""
    if model == target_model:
        # Target model serves both generation and refusal-check queries
        # The bench can't perfectly separate them from the JSONL alone, so we
        # compute generation calls from step count and subtract.
        return "target"
    if "kimi" in model.lower():
        return "extractor+grouper"
    if "qwen" in model.lower():
        return "translator"
    if "gemma" in model.lower():
        return "judge+provoker"
    return f"other/{model}"


def count_api_calls(records: list[dict], target_model: str) -> dict:
    """Sum effective API calls by role (batch-size weighted)."""
    by_role: collections.Counter = collections.Counter()
    by_model: collections.Counter = collections.Counter()
    for r in records:
        ct = r.get("call_type", "?")
        m = r.get("model", "?")
        bs = r.get("batch_size", 1) or 1
        role = classify_role(ct, m, target_model)
        by_role[role] += bs
        by_model[m] += bs
    return {"by_role": dict(by_role), "by_model": dict(by_model), "total": sum(by_role.values())}


def fit_decay(history: list[int]) -> tuple[float, float]:
    """Fit novel-heads-per-step to h_k = a * r^k (geometric decay).

    Returns (a, r). With only 2 data points, r = h1/h0; with more we'd fit.
    If decay is negligible (r ≥ 1), just project forward at the latest rate.
    """
    if len(history) < 2:
        return (history[0] if history else 0, 1.0)
    h0, h1 = history[0], history[-1]
    n = len(history) - 1
    if h0 <= 0:
        return (0, 0)
    r = (h1 / h0) ** (1.0 / n)
    r = min(r, 1.0)  # never project UP with more data
    return (h0, r)


def project_novel_heads(a: float, r: float, total_steps: int) -> list[float]:
    """Return projected heads-per-step over total_steps."""
    return [a * (r ** k) for k in range(total_steps)]


def simulate_budget(
    novel_heads_per_step: list[float],
    n_langs: int,
    gen_batch: int,
    progressive_triage: int = 3,
    progressive_escalation: int = 2,
    escalation_rate: float = 0.35,
    translate_per_head: int = 2,
    group_calls_per_step_lang: int = 1,
    extract_per_response: int = 1,
) -> dict:
    """Estimate API calls per role under the V2 pipeline model.

    For each step, each step-lang:
      - target generation:     gen_batch calls
      - extraction:            gen_batch * extract_per_response  (K=1 default)
      - grouping:              group_calls_per_step_lang
      - translation:           novel_heads * translate_per_head
      - refusal-check queries: novel_heads * (triage + rate * escalation)
      - judge + provoker:      novel_heads * (triage + rate * escalation) + novel_heads
          (one provoker call per novel head to generate queries, one judge call per query)
    """
    total = {
        "target_generation": 0,
        "target_refusal_queries": 0,
        "extraction": 0,
        "grouping": 0,
        "translation": 0,
        "judge": 0,
        "provoker": 0,
    }
    per_head_checks = progressive_triage + escalation_rate * progressive_escalation
    for novel in novel_heads_per_step:
        heads_this_step = novel  # total across langs (sum over step-langs)
        total["target_generation"] += gen_batch * n_langs
        total["extraction"] += gen_batch * n_langs * extract_per_response
        total["grouping"] += n_langs * group_calls_per_step_lang
        total["translation"] += heads_this_step * translate_per_head
        total["target_refusal_queries"] += heads_this_step * per_head_checks
        total["judge"] += heads_this_step * per_head_checks
        total["provoker"] += heads_this_step  # one provoker batch per head
    total["GRAND_TOTAL"] = sum(total.values())
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Path to crawler_out_*.json")
    p.add_argument("--project-steps", type=int, default=10,
                   help="Number of steps to project forward (default 10)")
    p.add_argument("--sensitivity-caps", default="50,30,20,10",
                   help="M values to simulate (applied as ceiling on heads-per-step)")
    p.add_argument("--escalation-rate", type=float, default=0.35,
                   help="Fraction of heads that trigger Phase-B in progressive triage")
    args = p.parse_args()

    json_path = Path(args.run)
    data, records = load_bundle(json_path)

    # --- Config + ground truth --------------------------------------------
    cfg = data["config"]["crawler"]
    target_model = data["config"]["model"].get("target_model", "unknown")
    n_steps_ran = cfg["num_crawl_steps"]
    n_langs = len(cfg["prompt_languages"])
    gen_batch = cfg["generation_batch_size"]
    cap_m = cfg.get("max_topics_per_step_lang") or cfg.get("max_extracted_topics_per_generation")

    hist = data["stats"]["history"]
    heads_per_step = hist["deduped_per_step"]          # novel heads added per step (sum over langs)
    all_per_step = hist["all_per_step"]                # all topics processed (inc dups)
    refusal_per_step = hist["refusal_per_step"]

    cumulative = data["stats"]["cumulative"]

    queue = data["queue"]["topics"]
    n_heads_stored = len(queue["head_topics"])
    n_heads_refusal = len(queue["head_refusal_topics"])
    n_cluster_members = len(queue["cluster_topics"])

    print("=" * 100)
    print(f"REPLAY: {json_path.name}")
    print("=" * 100)
    print(f"Target model:        {target_model}")
    print(f"Crawl type:          {cfg.get('crawler_type', 'v1?')}")
    print(f"Steps run:           {n_steps_ran}")
    print(f"Languages:           {cfg['prompt_languages']}")
    print(f"generation_batch:    {gen_batch}")
    print(f"cap_M (per step-lang): {cap_m}")
    print(f"do_filter_refusals:  {cfg['do_filter_refusals']}")
    print(f"progressive_triage:  {cfg.get('refusal_triage_checks', '?')} + "
          f"{cfg.get('refusal_escalation_checks', '?')}")
    print()

    # --- Dedup rate -------------------------------------------------------
    total_all = cumulative["total_all"]
    total_deduped = cumulative["total_deduped"]
    total_refusals = cumulative["total_refusals"]
    total_unique_refusals = cumulative["total_unique_refusals"]
    dedup_rate = 1.0 - (total_deduped / total_all) if total_all else 0.0

    print("--- Queue counters ---")
    print(f"All topics processed:        {total_all}")
    print(f"Novel heads added:           {total_deduped}  "
          f"(dedup kept {1-dedup_rate:.0%}, dropped {dedup_rate:.0%} as duplicates)")
    print(f"Total head topics (queue):   {n_heads_stored}")
    print(f"Total cluster members:       {n_cluster_members}")
    print(f"Refusal heads:               {n_heads_refusal}")
    print(f"Refusal rate (of novel heads):{n_heads_refusal/max(total_deduped,1):.1%}")
    print()

    # --- Per-step curve ---------------------------------------------------
    print("--- Per-step novel heads ---")
    a, r = fit_decay(heads_per_step)
    print(f"Observed: {heads_per_step}")
    print(f"Fit: h_k = {a:.1f} × {r:.3f}^k   (geometric decay)")
    if r < 0.999:
        half_life = math.log(0.5) / math.log(r) if r > 0 else float("inf")
        print(f"Decay half-life: ~{half_life:.1f} steps")
    else:
        print("No observed decay (r ≈ 1.0) — curve hasn't flattened yet.")
    print()

    # --- API call breakdown ----------------------------------------------
    api = count_api_calls(records, target_model)
    print("--- Observed API calls (batch-size weighted) ---")
    print(f"Total: {api['total']}")
    for role, n in sorted(api["by_role"].items(), key=lambda x: -x[1]):
        print(f"  {role:<22} {n:>6}  ({n/max(api['total'],1):.0%})")
    print()
    print("Per-step:")
    print(f"  {api['total']/n_steps_ran:.0f} calls/step   "
          f"({api['total']/n_steps_ran/n_langs:.0f} per step-lang)")
    print()

    # --- Projection to N steps -------------------------------------------
    N = args.project_steps
    projected_curve = project_novel_heads(a, r, N)
    total_novel_proj = sum(projected_curve)

    print(f"--- PROJECTION to N={N} steps @ current decay ---")
    print(f"Projected novel heads per step: "
          f"{[f'{h:.0f}' for h in projected_curve]}")
    print(f"Total novel heads over {N} steps: {total_novel_proj:.0f}")
    print()

    budget = simulate_budget(
        projected_curve, n_langs, gen_batch,
        progressive_triage=cfg.get("refusal_triage_checks", 3),
        progressive_escalation=cfg.get("refusal_escalation_checks", 2),
        escalation_rate=args.escalation_rate,
    )

    print(f"--- Projected API budget (N={N} steps, escalation_rate={args.escalation_rate}) ---")
    print(f"{'role':<28} {'calls':>8}  {'share':>6}")
    print("-" * 50)
    grand = budget.pop("GRAND_TOTAL")
    for role, n in sorted(budget.items(), key=lambda x: -x[1]):
        print(f"  {role:<26} {n:>8,.0f}  {n/grand:>5.0%}")
    print(f"  {'-'*26} {'-'*8}  {'-'*5}")
    print(f"  {'GRAND TOTAL':<26} {grand:>8,.0f}")
    print()

    # --- Sensitivity: different M caps -----------------------------------
    print(f"--- Sensitivity to cap M (max_topics_per_step_lang) ---")
    print(f"Assumption: M caps novel heads per step. If projected novel > M*n_langs, clip.")
    print(f"{'M':>4}  {'novel/step_avg':>15}  {'total_novel':>12}  {'total_calls':>13}  "
          f"{'target_calls':>13}")
    print("-" * 70)
    for M in [int(x) for x in args.sensitivity_caps.split(",")]:
        capped_curve = [min(h, M * n_langs) for h in projected_curve]
        b = simulate_budget(
            capped_curve, n_langs, gen_batch,
            progressive_triage=cfg.get("refusal_triage_checks", 3),
            progressive_escalation=cfg.get("refusal_escalation_checks", 2),
            escalation_rate=args.escalation_rate,
        )
        target_calls = b["target_generation"] + b["target_refusal_queries"]
        print(f"{M:>4}  {sum(capped_curve)/N:>14.1f}  "
              f"{sum(capped_curve):>11.0f}   {b['GRAND_TOTAL']:>12,.0f}   "
              f"{target_calls:>12,.0f}")
    print()

    # --- Interpretation hints --------------------------------------------
    print("--- Interpretation ---")
    if r > 0.95:
        print("⚠  Decay is weak (r > 0.95). More steps ≈ linearly more cost. "
              "Either cap is not biting or refusal surface hasn't saturated.")
    elif r > 0.85:
        print(f"△  Moderate decay (r={r:.2f}). Coverage benefits from additional steps "
              "but returns diminish — consider capping N around the half-life.")
    else:
        print(f"✓  Strong decay (r={r:.2f}). Most heads discovered in early steps; "
              "tighter M or fewer N both save cost with limited coverage loss.")

    leak_candidates = []
    for role in ["target_refusal_queries", "judge", "provoker", "extraction", "translation"]:
        if role in budget and budget[role] > 0.25 * grand:
            leak_candidates.append((role, budget[role]))
    if leak_candidates:
        print("Largest cost buckets (>25% of projected total):")
        for role, n in leak_candidates:
            print(f"  • {role}: {n:,.0f} calls ({n/grand:.0%})")


if __name__ == "__main__":
    main()
