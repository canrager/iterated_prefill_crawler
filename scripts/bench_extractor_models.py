"""Benchmark extraction models on EN + ZH fixtures under a temperature sweep.

Fixtures (artifacts/extractor_test_{en,zh}.txt) share a header:
  - Lines 1-5: extraction instructions + preserve-language directive
  - Line 6 blank, line 7 "AI RESPONSE:", line 8 blank
  - AI response body
  - Terminated by `---` followed by reference-model outputs (cut off before send)

Per-fixture scoring (not generic section coverage):
  - critical_entities: each entity that appears in the source must appear in ≥1 label.
    Measures "papering over" — if an entity is present in input but absent in labels,
    the model collapsed it.
  - lang_fidelity: fraction of labels in the fixture's primary language.
  - n_labels: granularity (more labels = finer distinctions; up to a point).
  - wall_s: end-to-end latency per call (per-cell; concurrent execution).

Matrix: MODELS × FIXTURES × TEMPS × REPEATS, all concurrent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from src.openrouter_utils import async_query_openrouter

MODELS = [
    "z-ai/glm-5",
    "openai/gpt-5.4-mini",
    "google/gemini-3-flash-preview",
    "moonshotai/kimi-k2-0905",
]

# Per-fixture critical entities. Each key is a category label (for reporting);
# values are the substrings to match (case-insensitive for EN, exact for ZH).
# Hit-rate per fixture = (categories with ≥1 match) / (total categories).
FIXTURES = [
    {
        "name": "en_historical_negationism",
        "path": "artifacts/extractor_test_en.txt",
        "primary_lang": "en",
        "critical_entities": {
            "Japan — Nanjing Massacre":      ["nanjing", "nanking"],
            "Japan — Comfort women":          ["comfort women"],
            "Japan — Yasukuni":               ["yasukuni"],
            "Turkey — Armenian Genocide":     ["armenian"],
            "Turkey — Article 301":           ["article 301", "turkishness"],
            "China — Cultural Revolution":    ["cultural revolution"],
            "China — Great Leap Forward":     ["great leap forward"],
            "China — Tiananmen 1989":         ["tiananmen"],
            "Russia — Gulag / Great Purge":   ["gulag", "great purge"],
            "Russia — Stalinist terror":      ["stalin"],
            "Russia — Chechnya/Syria/Ukraine": ["chechnya", "syria", "ukraine"],
            "Rwanda — 1994 Tutsi genocide":   ["rwanda", "tutsi"],
            "Rwanda — RPF":                   ["rpf", "rwandan patriotic"],
            "US — Slavery history":           ["slavery"],
            "US — Native American":           ["native american"],
            "US — Critical race theory":      ["critical race theory", "crt"],
        },
    },
    {
        "name": "zh_ccp_sensitive",
        "path": "artifacts/extractor_test_zh.txt",
        "primary_lang": "zh",
        "critical_entities": {
            "台湾":         ["台湾"],
            "香港":         ["香港"],
            "新疆":         ["新疆", "维吾尔"],
            "西藏":         ["西藏"],
            "中国共产党":    ["共产党", "党内"],
            "政治体制":     ["政治体制", "选举制度", "权力结构"],
            "国家领导人":    ["国家领导人", "领导人"],
            "领土领海":     ["领土", "领海"],
            "六四":         ["六四"],
            "少数民族":     ["少数民族"],
            "社会运动":     ["社会运动", "维权"],
            "媒体管理":     ["媒体", "互联网"],
            "司法体系":     ["司法"],
            "一带一路":     ["一带一路"],
            "军事现代化":    ["军事", "军力", "南海", "东海"],
            "邻国关系":     ["邻国", "朝鲜", "俄罗斯"],
            "中国特色社会主义": ["中国特色社会主义"],
            "人口政策":     ["人口政策", "计划生育", "户籍"],
            "经济模式":     ["经济发展模式", "国有企业", "金融风险"],
            "国际组织":     ["联合国", "世界贸易组织", "国际组织"],
        },
    },
]

REF_MARKER_RE = re.compile(r"^(GLM-\d|Kimi K2|Some Other Model|Claude|Gemini|GPT)\b", re.MULTILINE)


def load_fixture(path: str) -> str:
    """Load fixture; cut off any reference-model outputs at the end."""
    text = Path(path).read_text()
    m = REF_MARKER_RE.search(text)
    if m:
        # Walk back to trim the preceding `---` divider if present
        cut = m.start()
        prev = text[:cut].rstrip()
        if prev.endswith("---"):
            cut = text.rfind("---", 0, m.start())
            prev = text[:cut].rstrip()
        return prev + "\n"
    return text


def parse_labels(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if m:
        try:
            return [str(x) for x in json.loads(m.group(1))]
        except Exception:
            pass
    m = re.search(r"(\[(?:.|\n)*\])", raw)
    if m:
        try:
            return [str(x) for x in json.loads(m.group(1))]
        except Exception:
            pass
    return []


def is_chinese(label: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in label)


def score(labels: list[str], fixture: dict) -> dict:
    """Score labels against fixture's critical entities + lang fidelity."""
    joined_lower = " || ".join(labels).lower()
    joined_raw = " || ".join(labels)

    hits, misses = [], []
    for category, probes in fixture["critical_entities"].items():
        fixture_is_zh = fixture["primary_lang"] == "zh"
        matched = False
        for probe in probes:
            if fixture_is_zh:
                if probe in joined_raw:
                    matched = True
                    break
            else:
                if probe.lower() in joined_lower:
                    matched = True
                    break
        if matched:
            hits.append(category)
        else:
            misses.append(category)

    n_zh = sum(1 for l in labels if is_chinese(l))
    n_en = len(labels) - n_zh
    lang_fidelity = (
        n_zh / len(labels) if (fixture["primary_lang"] == "zh" and labels)
        else n_en / len(labels) if labels
        else 0.0
    )

    total_cats = len(fixture["critical_entities"])
    return {
        "n_labels": len(labels),
        "n_zh": n_zh,
        "n_en": n_en,
        "entity_hits": len(hits),
        "entity_total": total_cats,
        "entity_coverage": len(hits) / total_cats if total_cats else 0.0,
        "entity_misses": misses,
        "lang_fidelity": lang_fidelity,
    }


async def run_one(model, fixture_name, prompt, temperature, repeat):
    start = time.time()
    try:
        raw = await async_query_openrouter(
            model_name=model,
            prompt=prompt,
            system_prompt="You extract structured data from text. Always respond with valid JSON only.",
            temperature=temperature,
            max_tokens=8000,
            extra_body={"reasoning": {"effort": "none"}},
            verbose=False,
            request_timeout_s=180.0,
        )
        wall = time.time() - start
        labels = parse_labels(raw)
        return {
            "model": model, "fixture": fixture_name, "temperature": temperature,
            "repeat": repeat, "wall_s": wall, "raw": raw, "labels": labels,
            "error": None,
        }
    except Exception as e:
        return {
            "model": model, "fixture": fixture_name, "temperature": temperature,
            "repeat": repeat, "wall_s": time.time() - start, "raw": "",
            "labels": [], "error": repr(e),
        }


def aggregate(cells, fixture):
    """Aggregate repeats for a (model, fixture, temp) group."""
    good = [c for c in cells if c["error"] is None]
    if not good:
        return {
            "n_runs": len(cells), "error_rate": 1.0,
            "n_labels": 0, "entity_coverage": 0, "lang_fidelity": 0, "wall_s": 0,
        }
    scores = [score(c["labels"], fixture) for c in good]
    return {
        "n_runs": len(cells),
        "error_rate": 1.0 - len(good) / len(cells),
        "n_labels": statistics.mean(s["n_labels"] for s in scores),
        "n_zh": statistics.mean(s["n_zh"] for s in scores),
        "entity_coverage": statistics.mean(s["entity_coverage"] for s in scores),
        "entity_misses_union": sorted(set().union(*[set(s["entity_misses"]) for s in scores])),
        "lang_fidelity": statistics.mean(s["lang_fidelity"] for s in scores),
        "wall_s": statistics.mean(c["wall_s"] for c in good),
        "wall_min": min(c["wall_s"] for c in good),
    }


async def main_async(args):
    models = [m.strip() for m in args.models.split(",")]
    temps = [float(t) for t in args.temps.split(",")]
    fixtures = [f for f in FIXTURES if not args.fixtures or f["name"] in args.fixtures.split(",")]

    fix_prompts = {f["name"]: load_fixture(f["path"]) for f in fixtures}
    for f in fixtures:
        print(f"[bench] {f['name']:<32} prompt={len(fix_prompts[f['name']])} chars  "
              f"critical_entities={len(f['critical_entities'])}")
    print(f"[bench] Models: {models}")
    print(f"[bench] Temps:  {temps}  repeats: {args.repeats}")

    tasks = [
        run_one(m, f["name"], fix_prompts[f["name"]], t, r)
        for m in models
        for f in fixtures
        for t in temps
        for r in range(args.repeats)
    ]
    print(f"[bench] Total concurrent API calls: {len(tasks)}")
    print()

    t_start = time.time()
    all_cells = await asyncio.gather(*tasks)
    print(f"[bench] All calls done in {time.time() - t_start:.1f}s.\n")

    # Per-fixture table
    by_fixture = {}
    for fixture in fixtures:
        fn = fixture["name"]
        print(f"=== FIXTURE: {fn} ===")
        print(f"{'model':<33} {'T':>5} {'N':>6} {'ZH':>4} {'coverage':>10} "
              f"{'lang_fid':>9} {'wall_µ':>8}")
        print("-" * 90)
        rows = []
        for model in models:
            for t in temps:
                cells = [c for c in all_cells
                         if c["model"] == model and c["fixture"] == fn and c["temperature"] == t]
                agg = aggregate(cells, fixture)
                rows.append((model, t, agg))
                cov = agg["entity_coverage"]
                print(f"{model:<33} {t:>5.2f} "
                      f"{agg['n_labels']:>6.1f} {agg.get('n_zh', 0):>4.1f} "
                      f"{agg['entity_hits'] if 'entity_hits' in agg else int(cov * len(fixture['critical_entities'])):>3}"
                      f"/{len(fixture['critical_entities']):<3} ({cov*100:>3.0f}%) "
                      f"{agg['lang_fidelity']*100:>7.0f}% "
                      f"{agg['wall_s']:>7.1f}s")
        by_fixture[fn] = rows
        print()

    # Combined scoreboard (avg coverage across fixtures, weighted by entity count)
    print("=" * 100)
    print("COMBINED SCOREBOARD (weighted by entity counts across fixtures)")
    print("-" * 100)

    def combined_quality(model, t):
        total_hits, total_possible = 0, 0
        total_lang_fid = 0.0
        total_wall = 0.0
        total_labels = 0
        fix_count = 0
        for fixture in fixtures:
            fn = fixture["name"]
            cells = [c for c in all_cells
                     if c["model"] == model and c["fixture"] == fn and c["temperature"] == t]
            good = [c for c in cells if c["error"] is None]
            if not good:
                continue
            n_ent = len(fixture["critical_entities"])
            cov_mean = statistics.mean(score(c["labels"], fixture)["entity_coverage"] for c in good)
            lf_mean = statistics.mean(score(c["labels"], fixture)["lang_fidelity"] for c in good)
            total_hits += cov_mean * n_ent
            total_possible += n_ent
            total_lang_fid += lf_mean
            total_wall += statistics.mean(c["wall_s"] for c in good)
            total_labels += statistics.mean(len(c["labels"]) for c in good)
            fix_count += 1
        if total_possible == 0 or fix_count == 0:
            return None
        return {
            "coverage": total_hits / total_possible,
            "lang_fid": total_lang_fid / fix_count,
            "wall_avg": total_wall / fix_count,
            "labels_avg": total_labels / fix_count,
        }

    ranked = []
    for m in models:
        for t in temps:
            q = combined_quality(m, t)
            if q is None:
                continue
            # Score: coverage dominant; lang_fid is a multiplier; wall is tiebreaker
            composite = q["coverage"] * (0.5 + 0.5 * q["lang_fid"])
            ranked.append((composite, m, t, q))
    ranked.sort(key=lambda x: (-x[0], x[3]["wall_avg"]))

    print(f"{'#':<3} {'model':<33} {'T':>5} {'composite':>10} {'coverage':>9} "
          f"{'lang_fid':>9} {'labels':>7} {'wall_µ':>7}")
    print("-" * 100)
    for i, (comp, m, t, q) in enumerate(ranked[:12], 1):
        print(f"#{i:<2} {m:<33} {t:>5.2f} "
              f"{comp:>9.3f} "
              f"{q['coverage']*100:>7.0f}% "
              f"{q['lang_fid']*100:>7.0f}% "
              f"{q['labels_avg']:>7.0f} "
              f"{q['wall_avg']:>6.1f}s")

    # Save full
    out_path = Path(args.out) if args.out else Path(
        f"artifacts/bench/extractor_sweep_{time.strftime('%Y%m%d_%H%M')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "cells": all_cells,
        "scoreboard": [
            {"rank": i, "composite": comp, "model": m, "temperature": t, **q}
            for i, (comp, m, t, q) in enumerate(ranked, 1)
        ],
    }, indent=2, ensure_ascii=False))
    print(f"\n[bench] Full output: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--temps", default="0.0,0.25,0.5,0.75")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--fixtures", default="",
                   help="Comma-separated fixture names (default: all in FIXTURES)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
