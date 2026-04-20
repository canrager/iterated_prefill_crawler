"""Monte-Carlo pipeline sweep: projected (cost, coverage) across hyperparameter grid.

Model (uniform coupon-collector on a hidden population of size S):
  - Each step draws `sample_rate` heads with replacement from {0, …, S−1}.
  - sample_rate = novel_rate_per_gen × gen_batch × n_langs
      novel_rate_per_gen is fit from S1 observation (h_1 / (gen_batch × n_langs)).
  - Novel-per-step = draws not yet in `discovered` set.
  - After N steps: coverage = |discovered| / S.

Cost model per config (all calls batch-size weighted):
  generation:   N × n_langs × gen_batch
  extraction:   N × n_langs × gen_batch × K_extract   (K=1)
  grouping:     N × n_langs
  translation:  total_novel × 2 / translation_batch   (2 directions per head)
  provoker:     total_novel × (1 + esc_rate)
  target_ref:   total_novel × (triage + esc_rate × esc_depth)       # hard floor
  judge:        target_ref × judge_llm_escalation_rate              # cascade fallthrough

Neutrality constraint: no topic-level pre-screening. The regex + distilroberta
cascade operates on target-model RESPONSES (topic-agnostic), replacing the LLM
judge where confident — never replacing the target call. `judge_llm_escalation_rate`
is the empirical fraction of responses that fall through to the LLM judge.

Outputs: Pareto frontier of (cost, coverage) at each assumed S. Also prints
the single "recommended" config at a user-supplied budget ceiling.

Usage:
  python scripts/sim_pipeline_sweep.py \\
      --run artifacts/out/crawler_out_..._s1_v2.json \\
      --S-values 250,500,686,1000 \\
      --budget 2000

Zero API calls.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import sys
from pathlib import Path


def simulate_coupon(S: int, sample_rate: int, n_steps: int, seed: int = 0):
    rng = random.Random(seed)
    discovered: set[int] = set()
    novel_per_step: list[int] = []
    for _ in range(n_steps):
        drawn = {rng.randrange(S) for _ in range(sample_rate)}
        novel_per_step.append(len(drawn - discovered))
        discovered |= drawn
    return novel_per_step, len(discovered)


def mean_coverage(S: int, sample_rate: int, n_steps: int, trials: int = 20):
    """Run `trials` seeds; return (mean_novel_curve, mean_discovered, std_discovered)."""
    all_novel = []
    all_discovered = []
    for t in range(trials):
        novel, disc = simulate_coupon(S, sample_rate, n_steps, seed=t)
        all_novel.append(novel)
        all_discovered.append(disc)
    mean_novel = [statistics.mean(col) for col in zip(*all_novel)]
    return mean_novel, statistics.mean(all_discovered), statistics.stdev(all_discovered) if trials > 1 else 0


def project_cost(
    novel_per_step: list[float],
    *,
    n_langs: int,
    gen_batch: int,
    K_extract: int,
    triage: int,
    escalation: int,
    escalation_rate: float,
    translation_batch: int,
    judge_llm_escalation_rate: float,
) -> dict:
    total_novel = sum(novel_per_step)
    n_steps = len(novel_per_step)
    per_head_checks = triage + escalation_rate * escalation
    target_calls = total_novel * per_head_checks

    costs = {
        "generation": n_steps * n_langs * gen_batch,
        "extraction": n_steps * n_langs * gen_batch * K_extract,
        "grouping": n_steps * n_langs,
        "translation": (total_novel * 2) / max(translation_batch, 1),
        "provoker": total_novel * (1 + escalation_rate),
        "target_refusal": target_calls,
        "judge": target_calls * judge_llm_escalation_rate,
        # classifier + regex are free (local); not counted
    }
    costs["TOTAL"] = sum(costs.values())
    return costs


def pareto_frontier(configs: list[dict], cost_key="total_cost", cov_key="coverage"):
    """Return configs on Pareto frontier (min cost, max coverage)."""
    frontier = []
    for c in configs:
        dominated = False
        for c2 in configs:
            if c is c2:
                continue
            if c2[cost_key] <= c[cost_key] and c2[cov_key] >= c[cov_key] and (
                c2[cost_key] < c[cost_key] or c2[cov_key] > c[cov_key]
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(c)
    return sorted(frontier, key=lambda c: c[cost_key])


def load_observed_rate(run_path: Path):
    """Fit novel_rate_per_gen from S1 first-step observation."""
    data = json.loads(run_path.read_text())
    cfg = data["config"]["crawler"]
    n_langs = len(cfg["prompt_languages"])
    gen_batch = cfg["generation_batch_size"]
    h1 = data["stats"]["history"]["deduped_per_step"][0]
    # h_1 = novel_rate × gen_batch × n_langs (since D_0=0, all draws are novel)
    rate = h1 / (gen_batch * n_langs)
    return {"n_langs": n_langs, "observed_gen_batch": gen_batch,
            "observed_h1": h1, "novel_rate_per_gen": rate,
            "target_model": data["config"]["model"].get("target_model", "?"),
            "observed_history": data["stats"]["history"]["deduped_per_step"]}


def infer_S_from_two_steps(h: list[int]):
    """If two observations: h_2/h_1 = (S-h_1)/S → S = h_1² / (h_1 - h_2)."""
    if len(h) < 2 or h[0] <= h[1]:
        return None
    return h[0] ** 2 / (h[0] - h[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="S1 crawler_out_*.json")
    p.add_argument("--S-values", default="250,500,700,1000",
                   help="Assumed refusal-surface sizes to sweep")
    p.add_argument("--budget", type=int, default=None,
                   help="Budget ceiling in API calls; mark configs that fit")
    p.add_argument("--trials", type=int, default=20,
                   help="Monte-Carlo trials per config")
    p.add_argument("--escalation-rate", type=float, default=0.35)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    obs = load_observed_rate(Path(args.run))
    S_values = [int(s) for s in args.S_values.split(",")]

    print("=" * 100)
    print(f"S1 observed: target={obs['target_model']}  "
          f"gen_batch={obs['observed_gen_batch']}  "
          f"h_history={obs['observed_history']}")
    print(f"Fitted novel_rate_per_gen: {obs['novel_rate_per_gen']:.3f}  "
          f"(so sample_rate = rate × gen_batch × n_langs per step)")

    inferred = infer_S_from_two_steps(obs["observed_history"])
    if inferred:
        print(f"Inferred S from 2-point decay: {inferred:.0f} (uniform-population assumption)")
    print(f"Sweep over S assumptions: {S_values}")
    print(f"Trials per cell: {args.trials}")
    if args.budget:
        print(f"Budget ceiling: {args.budget} API calls")
    print()

    # --- Hyperparameter grid ----------------------------------------------
    # judge_llm_escalation_rate reflects how often the regex+classifier cascade
    # falls through to Gemma. S1 empirical ≈ 0.98; with better classifier tuning
    # it could drop toward 0.3. Neutrality-safe: the cascade operates on responses,
    # not on topic strings.
    grid = {
        "N": [3, 5, 7, 10],
        "gen_batch": [25, 35, 50],
        "triage": [3, 5],
        "escalation": [0, 1, 2],
        "translation_batch": [1, 10],
        "judge_llm_escalation_rate": [0.3, 0.5, 1.0],
    }
    keys = list(grid.keys())

    configs = []
    for S in S_values:
        for vals in itertools.product(*grid.values()):
            hp = dict(zip(keys, vals))
            sample_rate = int(round(obs["novel_rate_per_gen"] * hp["gen_batch"] * obs["n_langs"]))
            if sample_rate < 1:
                continue
            mean_novel, disc, sd = mean_coverage(
                S, sample_rate, hp["N"], trials=args.trials,
            )
            cost = project_cost(
                mean_novel,
                n_langs=obs["n_langs"], gen_batch=hp["gen_batch"], K_extract=1,
                triage=hp["triage"], escalation=hp["escalation"],
                escalation_rate=args.escalation_rate,
                translation_batch=hp["translation_batch"],
                judge_llm_escalation_rate=hp["judge_llm_escalation_rate"],
            )
            configs.append({
                "S": S, **hp, "sample_rate": sample_rate,
                "novel_curve": [round(n, 1) for n in mean_novel],
                "total_novel": round(sum(mean_novel), 1),
                "discovered": round(disc, 1),
                "coverage": disc / S,
                "total_cost": cost["TOTAL"],
                "cost_detail": cost,
            })

    # --- Print Pareto per S --------------------------------------------
    for S in S_values:
        scope = [c for c in configs if c["S"] == S]
        frontier = pareto_frontier(scope)
        print("=" * 100)
        print(f"PARETO FRONTIER @ S={S}  ({len(scope)} cells → {len(frontier)} non-dominated)")
        print(f"{'N':>3} {'gb':>3} {'tri':>3} {'esc':>3} {'tb':>3} {'jLLM':>5} "
              f"{'cov':>6} {'cost':>7} {'target':>7} {'judge':>6} {'gen+ext':>7}")
        print("-" * 110)
        for c in frontier:
            d = c["cost_detail"]
            target = d["target_refusal"]
            judge = d["judge"]
            gen_ext = d["generation"] + d["extraction"]
            print(f"{c['N']:>3} {c['gen_batch']:>3} {c['triage']:>3} {c['escalation']:>3} "
                  f"{c['translation_batch']:>3} {c['judge_llm_escalation_rate']:>5.2f} "
                  f"{c['coverage']*100:>5.0f}% {c['total_cost']:>7,.0f} "
                  f"{target:>7,.0f} {judge:>6,.0f} {gen_ext:>7,.0f}")
        print()

    # --- Budget-constrained recommendation -------------------------------
    if args.budget:
        print("=" * 100)
        print(f"BUDGET-CONSTRAINED RECOMMENDATION (≤ {args.budget} calls)")
        print("-" * 100)
        for S in S_values:
            scope = [c for c in configs if c["S"] == S and c["total_cost"] <= args.budget]
            if not scope:
                print(f"@S={S}: NO CONFIG FITS (min cost = "
                      f"{min(c['total_cost'] for c in configs if c['S']==S):,.0f})")
                continue
            best = max(scope, key=lambda c: c["coverage"])
            d = best["cost_detail"]
            print(f"@S={S}: best coverage under budget = {best['coverage']*100:.0f}%  "
                  f"cost={best['total_cost']:,.0f}")
            print(f"       N={best['N']}  gen_batch={best['gen_batch']}  "
                  f"triage={best['triage']}+{best['escalation']}  "
                  f"translation_batch={best['translation_batch']}  "
                  f"judge_llm_escalation={best['judge_llm_escalation_rate']}")
            print(f"       cost split: target={d['target_refusal']:,.0f}  "
                  f"judge={d['judge']:,.0f}  "
                  f"provoker={d['provoker']:,.0f}  "
                  f"gen+ext={d['generation']+d['extraction']:,.0f}  "
                  f"trans={d['translation']:,.0f}")
            print()

    # --- Summary: if we believed the user's "4 × S" rule ----------------
    print("=" * 100)
    print("USER'S BUDGET HEURISTIC: total ≤ 4 × S")
    print("-" * 100)
    for S in S_values:
        target = 4 * S
        scope = [c for c in configs if c["S"] == S and c["total_cost"] <= target]
        if not scope:
            min_cost = min(c["total_cost"] for c in configs if c["S"] == S)
            print(f"@S={S}: budget=4S={target}  NO FIT  (min achievable = {min_cost:,.0f} = {min_cost/S:.1f}×S)")
        else:
            best = max(scope, key=lambda c: c["coverage"])
            print(f"@S={S}: budget=4S={target}  best_cov={best['coverage']*100:.0f}%  "
                  f"cost={best['total_cost']:,.0f}  "
                  f"config=N{best['N']}/gb{best['gen_batch']}/tri{best['triage']}+{best['escalation']}/"
                  f"tb{best['translation_batch']}/jLLM{best['judge_llm_escalation_rate']}")
    print()

    # --- Save ------------------------------------------------------------
    out_path = Path(args.out) if args.out else Path(
        "artifacts/bench/pipeline_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "observed": obs,
        "S_values": S_values,
        "all_configs": configs,
    }, indent=2, ensure_ascii=False))
    print(f"[sweep] Full output: {out_path}  ({len(configs)} configs)")


if __name__ == "__main__":
    main()
