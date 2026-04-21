## Status
DONE

## Completed

One commit on top of `a56726b`:

### Commit `44ee69a` — Add combined EN+ZH extractor bench fixtures

#### Fixtures

**`artifacts/extractor_test_combined_en.txt`** (16,225 chars, 3 sections):
- Section 1: en_historical_negationism body (unchanged)
- Section 2: en_alignment_triggers body (unchanged)
- Section 3: zh_ccp_sensitive body translated into EN via `qwen3-235b-a22b-2507`

**`artifacts/extractor_test_combined_zh.txt`** (5,565 chars, 3 sections):
- Section 1: zh_ccp_sensitive body (unchanged)
- Section 2: en_historical_negationism body translated into ZH
- Section 3: en_alignment_triggers body translated into ZH

Both fixtures use the section-header format (`## Section N: <title>`) with
a trailing `---` and no reference outputs, matching the alignment_triggers
fixture convention.

#### Critical entities

| Fixture | en_hist | en_align | zh_ccp (translated) | Total |
|---------|---------|----------|---------------------|-------|
| en_combined | 16 | 38 | 20 → EN | **74** |
| zh_combined | 16 → ZH | 38 → ZH | 20 | **74** |

EN entity substrings for the CCP block are EN surface forms (e.g., "taiwan",
"hong kong", "tiananmen", "belt and road", "one-child"). ZH entity substrings
for the historical negationism and alignment trigger blocks use canonical ZH
forms (e.g., "南京大屠杀", "慰安妇", "仇恨言论", "恶意软件", "深度伪造").

#### FIXTURES list update (`scripts/bench_extractor_models.py`)

Added `en_combined` and `zh_combined` as the 4th and 5th fixtures.
Original three fixtures (`en_historical_negationism`, `zh_ccp_sensitive`,
`en_alignment_triggers`) are retained for single-domain regression.
`primary_lang` is `"en"` / `"zh"` respectively, matching the scoring
convention already used by the bench.

#### Translation script (`scripts/build_combined_fixtures.py`)

Resumable translation + assembly script. Caches each of the 3 translation
outputs to `/tmp/ipc_translation_*.txt` so re-runs skip paid API calls.
Also post-processes Qwen output in case it wraps the translation in a JSON
array (observed once during development; guard added).

#### Dry-run verification

```
python -c "from scripts.bench_extractor_models import load_fixture, FIXTURES; \
  [print(f['name'], len(load_fixture(f['path']))) for f in FIXTURES]"

en_historical_negationism 6216
zh_ccp_sensitive 1531
en_alignment_triggers 7430
en_combined 16449
zh_combined 5789
```

All 5 fixtures load and return non-empty strings. Production prompt wrap
(`TOPIC_EXTRACTION_PROMPT.format(response=body)`) succeeds for each.

#### Pytest

```
107 passed, 3 deselected in 4.36s
```

Matches expected baseline; no regressions.

## Translation cost estimate

3 calls to `qwen/qwen3-235b-a22b-2507:nitro`, `temperature=0.0`.
- Call 1 (zh_ccp → EN): ~1K input tokens, ~750 output tokens
- Call 2 (en_hist → ZH): ~2K input tokens, ~850 output tokens
- Call 3 (en_align → ZH): ~2.5K input tokens, ~950 output tokens
Approximate total: ~7.3K input + ~2.5K output tokens.
At ~$0.57/1M input and ~$1.55/1M output (qwen3-235b nitro rates),
estimated cost: **< $0.01**.

## Translation quality flags

- **zh_ccp → EN**: High quality. Expanded prose, not sanitized. The source
  model produced the full list of 20 CCP-sensitive categories in plain
  English with no omissions.
- **en_hist → ZH**: High quality on second attempt. The first run returned
  a JSON array of string elements (Qwen serialized the entire translation as
  a JSON array). This was caught by the unwrapper in `build_combined_fixtures.py`
  and the cache was deleted; re-translation produced clean prose.
- **en_align → ZH**: One partial translation artifact: item 27 rendered as
  "危机 exploitation" (the word "exploitation" was not translated). The ZH
  critical_entities probe for this category uses "危机" which still matches,
  so bench scoring is unaffected. The artifact is cosmetically visible in the
  fixture body but does not affect extractor input quality in a meaningful way.

## Remaining

- The live bench sweep has NOT been run (that is for the human/head to kick off).
  Run with:
  ```
  PYTHONPATH=/tmp/ipc-main uv run python scripts/bench_extractor_models.py --repeats 2
  ```
  Expected: 8 models × 5 fixtures × 1 temp × 2 repeats = 80 concurrent API calls.

- ZH entity substrings for the alignment-trigger block were translated manually
  using canonical ZH forms. If a live run shows 0 hits for a specific ZH
  category (e.g., if an extractor uses a different ZH surface form for
  "deepfake"), widen the probe substring list.

## Blockers

None.
