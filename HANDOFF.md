## Status
DONE

## Completed

Commit `202a468` — Add translation benchmark to extractor bench: 3 tasks, scoring helpers, tests

### What was built

**`scripts/bench_extractor_models.py`** — translation benchmark section added after the existing extractor bench, before `main()`. Nothing in the existing extractor bench was changed.

New symbols added:

- `TRANSLATION_PROMPT` — single-call batched translation prompt template; instructs the model to return a same-length JSON array of translated labels.
- `_build_translation_tasks()` / `TRANSLATION_TASKS` — constructs 3 task dicts at import time by slicing `FIXTURES`:
  - `en_to_zh_alignment_triggers`: 38 EN source labels (keys of `en_alignment_triggers.critical_entities`), probes from `zh_combined` positions 36–73.
  - `en_to_zh_historical_negationism`: 16 EN source labels (keys of `en_historical_negationism.critical_entities`), probes from `zh_combined` positions 20–35.
  - `zh_to_en_ccp_sensitive`: 20 ZH source labels (keys of `zh_ccp_sensitive.critical_entities`), probes from `en_combined` positions 54–73 (the "CCP — " prefixed entries).
- `parse_translation_output(raw)` — bare JSON array, markdown-fenced JSON, or bullet/numbered-list fallback.
- `score_translation(out_labels, task)` — returns `coverage`, `count_fidelity`, `lang_fidelity`, `n_src`, `n_out`, `hits`. Reuses `is_chinese()` for script detection.
- `run_translation_one(model, task, temperature, repeat)` — same pattern as `run_one`; `max_tokens=4000`, `prefer_nitro=True`, `extra_body=REASONING_DISABLED`, `temperature=0.0` default.
- `aggregate_translation(cells)` — averages `coverage`, `count_fidelity`, `lang_fidelity`, `json_parse_rate`, `wall_s` over repeats.

**`main_async`** — after the existing `asyncio.gather` for extraction cells, a second `asyncio.gather` runs all translation cells concurrently. Prints per-(model, task) table and a ranked TRANSLATION SCOREBOARD (composite = avg_coverage * avg_count_fidelity * avg_lang_fidelity). Saves results under separate top-level JSON keys (`translation_cells`, `translation_rows`, `translation_scoreboard`) alongside existing keys.

**`tests/test_bench_extractor_metrics.py`** — 28 new tests added:
- `TestParseTranslationOutput` (6 tests): bare JSON, fenced JSON, bullet fallback, numbered fallback, empty input.
- `TestScoreTranslation` (13 tests): coverage all-hit/partial/case-insensitive/empty; count_fidelity exact/double/empty/clamped; lang_fidelity ZH-correct/ZH-wrong/EN-correct/EN-wrong/mixed.
- `TestTranslationTasksStructure` (9 tests): 3 tasks exist, correct names, correct source counts (38/16/20), correct target langs, probes are lists of strings, source labels match probe keys.

### Pytest summary

```
139 passed, 3 deselected in 4.20s
```

Baseline was 111. Net new: 28 tests.

### Target-probe provenance

| Task | Source labels | Target probes |
|------|--------------|---------------|
| `en_to_zh_alignment_triggers` | 38 keys of `en_alignment_triggers.critical_entities` | `zh_combined.critical_entities` positions 36–73 (last 38, alignment-trigger ZH translations) |
| `en_to_zh_historical_negationism` | 16 keys of `en_historical_negationism.critical_entities` | `zh_combined.critical_entities` positions 20–35 (16 historical ZH translations) |
| `zh_to_en_ccp_sensitive` | 20 keys of `zh_ccp_sensitive.critical_entities` | `en_combined.critical_entities` positions 54–73 (last 20, "CCP — " prefixed EN translations) |

## Remaining

- The bench has NOT been run live. The user kicks it off when ready.
- The TRANSLATION SCOREBOARD composite formula (coverage * count_fidelity * lang_fidelity) weights all three metrics equally. If one metric dominates in practice, the formula can be adjusted before or after the live run.

## Blockers

None.
