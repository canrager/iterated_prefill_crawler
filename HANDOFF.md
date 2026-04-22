## Status
DONE

## Completed

Commit `7e8aa72` — Disable reasoning tokens for all helper-model calls; leave target model unaffected

### Task 1 — extra_body plumbing

Added `REASONING_DISABLED: Dict = {"reasoning": {"effort": "none"}}` constant to `src/openrouter_utils.py` (module-level, exported).

Added `extra_body: Optional[Dict] = None` parameter to:
- `async_query_openrouter` in `src/openrouter_utils.py` — forwarded to `client.chat.completions.create(extra_body=extra_body)`
- `_async_api_single` in `src/generation_utils.py` — forwarded to `client.chat.completions.create(extra_body=extra_body)`
- `_api_batch_generate` in `src/generation_utils.py` — forwarded into `_async_api_single`
- `batch_generate` in `src/generation_utils.py` — forwarded into `_api_batch_generate`

Also added `REASONING_DISABLED` to the `src/generation_utils.py` import from `openrouter_utils` (needed at the `async_summarize_single_topic` call site).

### Task 2 — Helper call sites wired with REASONING_DISABLED

**`src/response_formatting_utils.py`** (4 sites):
- L89: `batch_generate(...)` in `_extract_with_model` local-model path
- L161: `async_query_openrouter(...)` in `extract_single` remote path
- L253: `batch_generate(...)` in `_translate_zn_to_en`
- L284: `batch_generate(...)` in `_translate_en_to_zn`
- L608: `batch_generate(...)` in `summarize_refusal_topics` local-model path

**`src/generation_utils.py`** (1 site):
- L347: `async_query_openrouter(...)` in `async_summarize_single_topic`

**`src/refusal_utils.py`** (3 sites):
- L55: `batch_generate(...)` in `llm_judge_refusals`
- L158: `batch_generate(...)` in `_translate_for_classifier`
- L365: `batch_generate(...)` in `check_refusal` (probe-query generation via `refusal_check_model`)

**NOT modified (target-model sites)**:
- `src/crawler/crawler.py:160` — warmup/seeded generation
- `src/refusal_utils.py:467` — target answer generation during refusal check

### Task 3 — Bench update

`scripts/bench_extractor_models.py`: imported `REASONING_DISABLED` and added `extra_body=REASONING_DISABLED` to the `async_query_openrouter` call in `run_one`.

### Task 4 — Unit tests

Added 4 tests to `tests/test_openrouter_utils.py`:

1. `test_reasoning_disabled_constant_value` — asserts `REASONING_DISABLED == {"reasoning": {"effort": "none"}}`
2. `test_async_query_openrouter_forwards_extra_body` — mocks `AsyncOpenAI`, confirms `extra_body=REASONING_DISABLED` reaches `create()` kwargs
3. `test_async_query_openrouter_extra_body_none_by_default` — confirms `create()` called with `extra_body=None` when not passed
4. `test_batch_generate_forwards_extra_body_to_create` — mocks `AsyncOpenAI`, confirms `extra_body=REASONING_DISABLED` threads through `batch_generate` → `_api_batch_generate` → `_async_api_single` → `create()`

### Pytest summary

```
111 passed, 3 deselected in 4.15s
```

Baseline was 107 passed. Net new: 4 tests.

### Files touched

- `src/openrouter_utils.py` — REASONING_DISABLED constant, extra_body param + forward
- `src/generation_utils.py` — REASONING_DISABLED import, extra_body param on _async_api_single/_api_batch_generate/batch_generate, wire at async_summarize_single_topic
- `src/response_formatting_utils.py` — REASONING_DISABLED import, 5 call sites wired
- `src/refusal_utils.py` — REASONING_DISABLED import, 3 call sites wired
- `scripts/bench_extractor_models.py` — REASONING_DISABLED import, run_one call site wired
- `tests/test_openrouter_utils.py` — 4 new tests

## Remaining

- Live bench sweep has NOT been run (user is still curating the model list per task spec).
- The `summarize_refusal_topics` OpenRouter-API path (which calls `async_batch_summarize_topics` → `async_summarize_single_topic`) already gets REASONING_DISABLED through the `async_summarize_single_topic` fix in generation_utils.py. The intermediate `async_batch_summarize_topics` signature was not modified (not required per task spec).

## Blockers

None.
