## Status
DONE

## Completed

Commit `62bb3e8` — Add per-call cost to bench scoreboards; capture token usage from API

### Task 1 — `return_usage` kwarg on `async_query_openrouter`

`src/openrouter_utils.py`: Added `return_usage: bool = False` kwarg. When `False` (default), behavior is identical to before — bare string returned. When `True`, returns `(str, {"prompt_tokens": int, "completion_tokens": int})`. All three early-return paths (no choices, no message, exception) return `("", _empty_usage)` when `return_usage=True` so callers never get a TypeError when unpacking. Prod call sites are untouched.

Two new unit tests in `tests/test_openrouter_utils.py`:
- `test_return_usage_true_returns_tuple_with_token_counts` — asserts tuple returned, counts match mocked usage object.
- `test_return_usage_false_returns_bare_string` — asserts bare string returned (default behavior unchanged).

### Task 2 — `fetch_openrouter_prices()` at bench startup

`scripts/bench_extractor_models.py`: Added `fetch_openrouter_prices()` async helper. Calls `GET https://openrouter.ai/api/v1/models` with `Authorization: Bearer` when `OPENROUTER_API_KEY` is set (optional). Parses `data[].pricing.prompt` and `data[].pricing.completion` as floats. Skips entries that fail to parse and logs a warning. Prints `"[bench] Fetched prices for N models from OpenRouter."` at startup. Called once in `main_async` before any API calls; prices dict is passed as `prices=` kwarg to `run_one` and `run_translation_one`.

Added helper functions:
- `_lookup_price(prices, model_id)`: first tries exact id, then strips `:nitro`/`:floor` routing suffix and retries. Returns `None` if absent.
- `_compute_cost(prices, model_id, prompt_tokens, completion_tokens)`: returns `float` USD or `None`.
- `_fmt_cost(cost_usd)`: `$0.00023` for >= 1e-4, `$1.50e-07` for smaller, `?` for None.

### Task 3 — Cost per cell

`run_one` and `run_translation_one` now call `async_query_openrouter(..., return_usage=True)` and store `prompt_tokens`, `completion_tokens`, and `cost_usd` in every returned cell dict. These fields are persisted in the JSON report (`cells` / `translation_cells` arrays). The `translation_cells` filter only strips `"raw"` — cost fields are not stripped.

`:nitro` resolution: `_lookup_price` strips the suffix, so `moonshotai/kimi-k2.5:nitro` falls back to `moonshotai/kimi-k2.5` for price lookup. In normal operation the bench passes bare ids (pre-`:nitro`) so the fallback is defensive only.

### Task 4 — Cost columns in both scoreboards

Extraction summary table: added `cost_$/call` column (mean over repeats per row).
Extraction scoreboard: added `cost_µ` column (mean across all fixtures and repeats for the model).
Translation summary table: added `cost_$/call` column.
Translation scoreboard: added `cost_µ` column.

Ranking in both scoreboards is unchanged (composite only). Cost is display-only.

### Task 5 — Drop arcee-ai/trinity-large-thinking

`arcee-ai/trinity-large-thinking` was already absent from `MODELS` on this branch. No code change needed.

### Task 6 — Kimi K2.5 diversification note

Added one-line comment on `"moonshotai/kimi-k2.5"` in `MODELS`:
`# non-qwen translator diversification candidate: 0.905 composite, 100% lang_fid, ~2.2s (bench v5)`

## Remaining

- The bench has not been run live (per task instructions: user will kick it off).
- `tests/test_topic_extraction_drift.py` has 3 pre-existing failures (MagicMock vs int in `extraction_batch_size`) unrelated to this work. These were failing before this change. Task specification said "Baseline is 139 passed, 3 deselected" but observed baseline was 139 passed, 3 failed. The 3 failures are unchanged.

## Blockers

None.

### Summary notes

**How cost_usd is computed:**
`cost_usd = prompt_tokens * price["prompt"] + completion_tokens * price["completion"]`
where prices are USD per token from OpenRouter's `/models` response (e.g. `"0.00000015"` cast to float).

**How `:nitro` is resolved for pricing:**
`_lookup_price` first tries the exact model id. If absent, strips `:nitro` or `:floor` suffix and retries. In practice the bench `run_one`/`run_translation_one` call `async_query_openrouter` with the original bare id, and `_apply_nitro` appends `:nitro` internally — the price lookup uses the bare id directly so no stripping is needed in normal flow. The fallback handles any future edge case.

**How missing prices are displayed:**
`?` in both `cost_$/call` (per-row) and `cost_µ` (scoreboard) columns when all cells for that model have `cost_usd=None`.
