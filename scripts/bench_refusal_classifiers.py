"""Compare distilroberta (current) vs multilingual-e5-small refusal classifiers
on real S1 refusal-check responses.

Neutrality note: both classifiers operate on response *text* (topic-agnostic).
Swapping to multilingual eliminates the Qwen translation pre-step for ZH responses.

Inputs:
  - artifacts/out/crawler_out_*.json : queue with refusal_check_responses[]
  - Optional: artifacts/out/crawler_out_*.jsonl for additional target generations

Metrics:
  - Label agreement rate between the two classifiers
  - Confidence distribution (fraction ≥ 0.99, ≥ 0.9, < 0.5)
  - Fallthrough rate to LLM judge (responses where the classifier is NOT ≥ 0.99)
  - Per-language (EN / ZH) breakdown
  - Wall-clock per 100 samples

No API calls. Downloads model weights on first run (~130MB each).

Usage:
  python scripts/bench_refusal_classifiers.py \\
      --run artifacts/out/crawler_out_..._s1_v2.json \\
      --limit 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

CLASSIFIERS = {
    "distilroberta (current)": "ProtectAI/distilroberta-base-rejection-v1",
    "multilingual-e5-small": "agentlans/multilingual-e5-small-refusal-classifier",
}

# Label name mapping — each classifier may use different strings.
# We normalize to "REFUSAL" vs "NORMAL".
def normalize_label(label: str) -> str:
    lu = label.upper()
    if lu in ("REJECTION", "REFUSAL", "LABEL_1", "REFUSE"):
        return "REFUSAL"
    if lu in ("NORMAL", "NON-REFUSAL", "LABEL_0", "COMPLIANT", "COMPLY"):
        return "NORMAL"
    return lu  # unknown — surface it


def is_chinese(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in text)


def load_responses(run_path: Path, include_target_gens: bool = True,
                   limit: int = 1000) -> list[dict]:
    """Collect (text, lang, source) triples from S1 artifacts."""
    responses: list[dict] = []

    data = json.loads(run_path.read_text())
    for h in data["queue"]["topics"]["head_topics"]:
        for i, r in enumerate(h.get("refusal_check_responses") or []):
            if r and r.strip():
                responses.append({
                    "text": r,
                    "lang": "zh" if is_chinese(r) else "en",
                    "source": "refusal_check_response",
                    "head_id": h.get("id"),
                    "head_is_refusal": h.get("is_refusal"),
                    "query_idx": i,
                })

    if include_target_gens:
        jsonl_path = run_path.with_suffix(".jsonl")
        if jsonl_path.exists():
            for line in jsonl_path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("call_type") == "batch_generate_api":
                    # only target model generations
                    for o in rec.get("outputs", [])[:5]:  # take first 5 per batch to limit
                        if o and o.strip() and len(o) > 50:
                            responses.append({
                                "text": o[:2000],
                                "lang": "zh" if is_chinese(o) else "en",
                                "source": f"target_gen/{rec.get('model', '?')}",
                                "head_id": None,
                                "head_is_refusal": None,
                                "query_idx": None,
                            })

    if limit and len(responses) > limit:
        # Keep language balance: prefer ZH samples since they're rare
        zh = [r for r in responses if r["lang"] == "zh"]
        en = [r for r in responses if r["lang"] == "en"]
        keep_en = max(0, limit - len(zh))
        responses = zh + en[:keep_en]

    return responses


def run_classifier(model_name: str, texts: list[str]):
    from transformers import pipeline
    print(f"  Loading {model_name} …", flush=True)
    t0 = time.time()
    clf = pipeline("text-classification", model=model_name, truncation=True,
                   max_length=512, device=-1)  # CPU is fine for tiny classifiers
    load_wall = time.time() - t0
    print(f"    loaded in {load_wall:.1f}s", flush=True)

    t0 = time.time()
    results = clf(texts, batch_size=32)
    infer_wall = time.time() - t0
    print(f"    {len(texts)} predictions in {infer_wall:.1f}s "
          f"({infer_wall/len(texts)*1000:.1f}ms each)", flush=True)
    return results, load_wall, infer_wall


def summarize(results: list[dict], threshold: float = 0.99) -> dict:
    n = len(results)
    refusal = sum(1 for r in results if normalize_label(r["label"]) == "REFUSAL")
    normal = sum(1 for r in results if normalize_label(r["label"]) == "NORMAL")
    unknown = sum(1 for r in results if normalize_label(r["label"]) not in ("REFUSAL", "NORMAL"))
    confident = sum(1 for r in results if r["score"] >= threshold)
    medium = sum(1 for r in results if 0.8 <= r["score"] < threshold)
    low = sum(1 for r in results if r["score"] < 0.8)
    return {
        "total": n,
        "refusal_pct": refusal / n * 100,
        "normal_pct": normal / n * 100,
        "unknown_pct": unknown / n * 100,
        "confident_pct": confident / n * 100,  # would skip LLM judge
        "medium_confidence_pct": medium / n * 100,
        "low_confidence_pct": low / n * 100,
        "fallthrough_pct": (n - confident) / n * 100,  # requires LLM judge
    }


def agreement(results_a: list[dict], results_b: list[dict]) -> dict:
    assert len(results_a) == len(results_b)
    n = len(results_a)
    same_label = 0
    both_confident_agree = 0
    both_confident = 0
    for a, b in zip(results_a, results_b):
        la = normalize_label(a["label"])
        lb = normalize_label(b["label"])
        if la == lb:
            same_label += 1
        if a["score"] >= 0.99 and b["score"] >= 0.99:
            both_confident += 1
            if la == lb:
                both_confident_agree += 1
    return {
        "label_agreement_pct": same_label / n * 100,
        "both_confident_pct": both_confident / n * 100,
        "confident_agree_rate_pct": (both_confident_agree / max(both_confident, 1)) * 100,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="crawler_out_*.json")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--threshold", type=float, default=0.99,
                   help="Confidence threshold for 'skip LLM judge' decision")
    p.add_argument("--include-target-gens", action="store_true",
                   help="Also benchmark on target generation outputs (not just refusal responses)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    responses = load_responses(Path(args.run),
                               include_target_gens=args.include_target_gens,
                               limit=args.limit)
    print(f"[bench] {len(responses)} responses loaded  "
          f"(ZH={sum(1 for r in responses if r['lang']=='zh')}, "
          f"EN={sum(1 for r in responses if r['lang']=='en')})")
    print()

    texts = [r["text"] for r in responses]

    all_results: dict[str, list[dict]] = {}
    wall_stats: dict[str, tuple] = {}
    for tag, model_name in CLASSIFIERS.items():
        print(f"[bench] {tag}")
        try:
            results, load_wall, infer_wall = run_classifier(model_name, texts)
            all_results[tag] = results
            wall_stats[tag] = (load_wall, infer_wall)
        except Exception as e:
            print(f"  FAILED: {e!r}")
            continue
        print()

    if len(all_results) < 2:
        print("[bench] Need both classifiers to run. Exiting.")
        return

    # Per-classifier summary
    print("=" * 90)
    print(f"{'classifier':<28} {'refusal%':>9} {'normal%':>9} "
          f"{'≥thr%':>7} {'medium%':>9} {'low%':>7} {'infer_µs':>9}")
    print("-" * 90)
    summaries = {}
    for tag, results in all_results.items():
        s = summarize(results, threshold=args.threshold)
        summaries[tag] = s
        wall = wall_stats[tag][1] / len(texts) * 1e6  # microseconds/sample
        print(f"{tag:<28} {s['refusal_pct']:>8.1f}% {s['normal_pct']:>8.1f}% "
              f"{s['confident_pct']:>6.1f}% {s['medium_confidence_pct']:>8.1f}% "
              f"{s['low_confidence_pct']:>6.1f}% {wall:>8.0f}")
    print()

    # Language breakdown
    print("=== Per-language classifier confidence ===")
    for lang in ("en", "zh"):
        lang_idx = [i for i, r in enumerate(responses) if r["lang"] == lang]
        if not lang_idx:
            continue
        print(f"\n  --- {lang.upper()}  n={len(lang_idx)} ---")
        for tag, results in all_results.items():
            lang_results = [results[i] for i in lang_idx]
            s = summarize(lang_results, threshold=args.threshold)
            print(f"    {tag:<28} refusal={s['refusal_pct']:>5.1f}%  "
                  f"confident(≥{args.threshold})={s['confident_pct']:>5.1f}%  "
                  f"fallthrough={s['fallthrough_pct']:>5.1f}%")
    print()

    # Agreement
    tags = list(all_results.keys())
    ag = agreement(all_results[tags[0]], all_results[tags[1]])
    print(f"=== Agreement: {tags[0]}  vs  {tags[1]} ===")
    print(f"  Label agreement overall:    {ag['label_agreement_pct']:.1f}%")
    print(f"  Both ≥{args.threshold} confident:           {ag['both_confident_pct']:.1f}% "
          f"(of {len(texts)} responses)")
    print(f"  Of both-confident, same label: {ag['confident_agree_rate_pct']:.1f}%")
    print()

    # Implication for budget
    fallthrough_a = summaries[tags[0]]["fallthrough_pct"] / 100
    fallthrough_b = summaries[tags[1]]["fallthrough_pct"] / 100
    print("=== Budget implication (at 160 checked heads × 3 triage = 480 responses) ===")
    for tag in tags:
        ft = summaries[tag]["fallthrough_pct"] / 100
        judge_calls = 480 * ft
        print(f"  {tag}: judge LLM calls = {judge_calls:.0f} ({ft*100:.0f}% fallthrough)")
    if fallthrough_a != fallthrough_b:
        delta = abs(fallthrough_a - fallthrough_b) * 480
        print(f"  Δ LLM judge calls: {delta:.0f}")
    print()

    # Save
    if args.out:
        out = Path(args.out)
    else:
        out = Path(f"artifacts/bench/classifier_compare_{time.strftime('%Y%m%d_%H%M')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_responses": len(responses),
        "n_zh": sum(1 for r in responses if r["lang"] == "zh"),
        "threshold": args.threshold,
        "summaries": summaries,
        "agreement": ag,
        "wall_stats": {k: {"load_s": v[0], "infer_s": v[1]} for k, v in wall_stats.items()},
        "per_example": [
            {
                "lang": responses[i]["lang"],
                "source": responses[i]["source"],
                "head_id": responses[i]["head_id"],
                "head_is_refusal": responses[i]["head_is_refusal"],
                **{
                    tag: {
                        "label": normalize_label(all_results[tag][i]["label"]),
                        "score": all_results[tag][i]["score"],
                    }
                    for tag in all_results
                },
                "text_preview": responses[i]["text"][:200],
            }
            for i in range(len(responses))
        ],
    }, indent=2, ensure_ascii=False))
    print(f"[bench] Full output: {out}")


if __name__ == "__main__":
    main()
