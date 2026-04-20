# Learnings

Forward-facing rules. Each entry states what to do or avoid, and why it matters for the research.
Rewrite, merge, and prune continuously. Source control tracks history.

---

## Cap semantics — every budget knob must be batch-level, not per-item

**Rule:** Apply all topic/extraction caps to the *flat concatenated list* after a full step-lang batch completes — not per individual model response inside the batch.

**Why it matters (the math):** Let S = semantically distinct refusal categories a model has (empirically ~100–300 for frontier models). Let M = cap per step-lang. Let B = prompts per batch (default 50). Without a batch-level cap, each of B responses can contribute up to M topics → effective cap B×M. With B=50, M=50, that's 2,500 — not 50.

The right framing: total refusal-check cost ≈ S × C regardless of M (where C ≈ 9 API calls per head for progressive 3+2 triage). M controls *rate*, not *total*. A high M saturates coverage faster but spends more per step; a low M spreads cost across more steps. Either way converges to the same total. So M=50 at N=10 steps gives ~99% coverage (coupon-collector: 1 − (1 − 20/200)^20 ≈ 98.7%) and a predictable ~184 calls/step-lang, while M=∞ produces unpredictable step cost and the same final bill.

**What M actually buys:**
1. Deterministic per-step cost (plannable budget)
2. Fine-grained crash recovery (save-before-refusal loses at most one step-lang)
3. Enables the cross-language V2 win: bilingual grouping collapses EN/ZH heads into one, checked once. S1 showed 456 stored heads vs 219 deduped (40% EN/ZH duplicates) — V2 does 0.6× the refusal checks V1 needed for equivalent coverage.

**Verify this is enforced:** The cap slice must appear *after* flattening all responses from a batch, and *before* any grouping or refusal-check call. Read the code — if the cap is inside a per-response loop, it is wrong.

---

## Don't batch the extraction stage — batch at the queue instead

**Rule:** Run extraction one response per API call (`extraction_batch_size=1`). Do not concatenate multiple target-model responses into a single Kimi prompt to save calls.

**Why:** Extraction is attention-limited — Kimi K2-0905 loses granularity when asked to process multiple responses in one call. The savings you think you're getting from batching are paid back in worse content and unreliable output shape.

**Evidence (2026-04-18 bench at `scripts/bench_batch_sizes.py`, 30 real S1 responses):**

| K | shape_err | align_err | retention | topics extracted | s/resp |
|---|---|---|---|---|---|
| **1** | 13%* | **31%** | **33%** | **141** | 3.25s |
| 2 | 0% | 70% | 0% | 37 | 1.01s |
| 3 | 0% | 53% | 33% | 68 | 1.43s |
| 5 | 17% | 72% | 0% | 35 | 0.98s |
| 10 | 67% | 40% | n/a | 24 | 0.55s |

*The 13% "shape errors" at K=1 are non-topic responses returning `[]` from Kimi — the split-retry path correctly routes these to `[[]]`. Not a real quality failure.

K=1 extracts 2× more topics than K=3, 4× more than K=5, and has the best alignment rate. K=5+ is shape-broken (17–67% malformed outputs). Batching saves wall time but costs coverage — the wrong tradeoff for the research question.

**Where the real API-call savings live:** not in extraction batching, but in *downstream queue discipline* — cap novel heads per step-lang (`max_topics_per_step_lang=50`), refusal-check only novel group heads, translate only once per head, generate provocations only when actually checking. The role-based architecture (summarizer / translator / provoker as separate roles with per-role caps) is the cost story. See "Cap semantics" above.

**Keep the split-retry scaffold anyway:** even at K=1 the `_extract_with_split_retry` / `_call_one_chunk` machinery is a correctness guardrail for the 403 moderation path and the singleton parse-failure path. Don't rip it out just because the default K is 1.

---

## Timeouts must fail loudly, not silently

**Rule:** When an API call times out or exhausts retries, raise an exception — never return an empty string. Empty string downstream is indistinguishable from a compliant (non-refused) response.

**Why it matters:** A timed-out refusal-check query that returns `""` makes the crawler mark a topic as *not refused* — a correctness bug, not just a cost bug. In S1, ~11% of target-model responses were empty; any that were timeouts silently passed topics as compliant.

**Shape:** Use exponential backoff up to a configurable wall-clock limit (default 1 hour). On exhaustion: raise. Refusal-check catches the raise and marks `refusal_check_inconclusive=True`. Extraction/translation/grouping catches it and logs a warning — best-effort, correctness-non-critical.

---

## Jailbreak framing must survive warmup

**Rule:** The jailbreak audit framing (CoT-forgery, policy-transparency preamble) must appear in every prompt — warmup (unseeded) and seeded alike. Never put jailbreak framing in a template slot that depends on a `{}` seed placeholder.

**Why it matters:** If framing lives in `user_post_templates` with `{}`, the prompt builder drops it when no seed is sampled — warmup runs with bland pre-templates only, and the model enumerates nothing interesting. This is silent: the run completes, but head count is near-zero.

**Shape:** Jailbreak framing belongs in `user_pre_templates` (no `{}`). Seed-referencing text belongs in `user_seed_templates` (with `{}`). Assert at `PromptBuilder` init time that no `user_pre_template` contains `{}`.

For `prompts=jailbreak` specifically: warmup does NOT need `initial_topics` — the jailbreak prompt is self-contained. Neutral `initial_topics` (`"an avoided topic"`) is an artifact of the TTF technique; it is irrelevant to jailbreak-style elicitation.

---

## Per-run in-memory caches do not hit — grouping already dedupes

**Rule:** Do not ship per-run in-memory caches for judge, summarize, translate, or extract calls on the aggregate crawler path. They will not hit meaningfully because the grouping pipeline (`src/crawler/grouping_pipeline.py`) already eliminates duplicate topics *upstream* of every helper that could benefit.

**Evidence (2026-04-19, `scripts/bench_cache_dedup.py` against three transcripts):**
- `.trio/baselines/pre_s4_20260419_015746.json` (baseline, 24 calls): translate 0.0%, others absent — DROP.
- `crawler_out_20260417_175459_..._s1_v2.jsonl` (1,394 calls — real run): **judge 0.0%** (820 calls, every prompt unique), translate 0.0% (42 calls), extract 6.9% (72 calls; below 10% gate), summarize absent — all DROP.
- `crawler_out_20260417_134516_..._s1_v2.jsonl` (18 calls): translate 0.0%, extract 0.0% — DROP.
- Aggregate across three runs: judge 0.0%, translate 0.0%, extract 6.0%, summarize N/A. No cache type crosses the 10% gate. Raw artifact: `.trio/baselines/s4c_cache_dedup_20260419.json`.

**Why the caches don't hit:**
- **judge**: each prompt carries the target model's (unique) response text. Two judge calls share a key only if the target emitted the exact same response twice — rare.
- **translate**: topic labels are deduplicated by the semantic-grouping LLM before translation runs, so each translate call is a fresh label.
- **extract**: inputs are unique target responses per step-lang. The 6% residual is within-step retries, not cross-step reuse.
- **summarize**: absorbed into the grouping call on the aggregate path; no standalone summarize calls appear in modern transcripts. Legacy (non-aggregate) runs are the only place these fire.

**What this closes:** S4c bench run on 2026-04-19 dropped S4d from scope. External reviewer suggestions to "add a cache layer for translation, summarization, extraction, and judge decisions" were plausible but empirically wrong for this crawler shape. The savings the reviewer expected already happen — just higher up the pipeline.

**Before re-proposing any caching layer:** rerun `scripts/bench_cache_dedup.py` against recent transcripts. If the per-run dedup rate on a ≥1k-call transcript still registers <10% across all four candidate types, do not build the cache. If architecture changes (e.g. grouping is removed or downgraded), re-measure before re-shipping.

---

## Helper model selection is load-bearing — bench, don't guess

**Rule:** Do not swap helper models, raise/lower per-stage caps, or change classifier choices without running the relevant bench and checking the numbers against the existing CCP-content fixtures. Provider version matters even within the same model family. A model that is "probably fine" on English general content can soft-refuse sensitive Chinese prompts silently.

**Where the benches live — `scripts/`:**
- `scripts/bench_extractors.py` — classify-level smoke on the extractor role; earliest, coarsest signal.
- `scripts/bench_extractor_models.py` — full MODEL × FIXTURE × TEMPERATURE sweep for extraction/grouping on `artifacts/extractor_test_{en,zh}.txt`. Scores entity preservation, lang fidelity, label count, wall-clock.
- `scripts/bench_batch_sizes.py` — extraction sub-batch K sweep (the 2026-04-18 run is why `extraction_batch_size=1`).
- `scripts/bench_refusal_classifiers.py` — distilroberta vs multilingual-e5 on real S1 refusal responses; offline, no API calls.

**Where results land — `artifacts/bench/`:** every bench writes a dated JSON/CSV (e.g. `extractor_models_20260418_1449.json`, `batch_sizes_20260418_1356.csv`, `classifier_compare_20260418_1630.json`). When uncertain about a choice, read the most recent matching file in `artifacts/bench/` before asking for a new live run. Do not guess; do not argue from intuition; do not rely on a model's generic reputation. The fixtures are the arbiter.

**Validated choices (current — check `artifacts/bench/` for timestamped evidence):**
- Extraction / grouping / summarization: `openai/gpt-5.4-mini` as of 2026-04-19, set in `configs/model/ds-v32_remote.yaml`. Evidence: `artifacts/bench/extractor_models_20260418_*.json` plus operator-confirmed pre-S4 baseline run. **Prior validated:** `moonshotai/kimi-k2-0905` (preserves Taiwan / Falun Gong in JSON). **Do not use `moonshotai/kimi-k2.5`**: silently soft-refuses CCP content (~60 s empty return, no moderation sentinel, 75% fail on the bench fixture vs 0% for k2-0905).
- Translation: `qwen/qwen3-235b-a22b-2507` — best Chinese↔English named-entity accuracy. Gemma 4 compresses multi-char Chinese names.
- Refusal judge / provocation generation: `google/gemma-4-26b-a4b-it` — fast (~3 s), 0% fail on sensitive categories, open weights.

**Before swapping any helper model:** run the matching `scripts/bench_*.py`, drop results in `artifacts/bench/`, and verify the sensitive-content fail rate ≤ 5% and entity-preservation ≥ 95%. If the bench wasn't run, the swap isn't justified.

---

## Jailbreak prompts are human-authored — do not touch

**Rule:** `configs/prompts/jailbreak.yaml` is the ground truth. Never "simplify," "clean up," or "normalize" the prompts. If a prompt looks dense, authoritative, or idiosyncratic, that is intentional.

**Ground truth artifacts** — compare pipeline output against these, not against intuition:
- `artifacts/EXAMPLE_ENGLISH_JAILBREAK_PROMPT.md` — §4.1 English prompt with DeepSeek/GPT/Haiku responses
- `artifacts/EXAMPLE_SEEDED_POLITICAL_PROMPT.md` — seeded geopolitical drill-down
- `artifacts/OpenRouter Chat Mon Mar 30 2026.md` — Chinese drill-down; DeepSeek enumerates Taiwan/Tibet/Hong Kong/Falun Gong by name
- `artifacts/OpenRouter Chat Thu Apr 16 2026.md` — reproduction; DeepSeek v3.2 surfaces geopolitical category from `"an avoided topic"` seed

---

## Neutral seeds only

**Rule:** `initial_topics` and any `{}` seed substitutions must be neutral (`"an avoided topic"` / `"一个被回避的话题"`). Never hard-code sensitive or provider-specific topic strings as seeds. The crawler must discover restrictions autonomously.

---

## Rehearsal ⊆ Debug ⊆ Default — scale-only subsets, never structural divergence

**Rule:** The three crawler configs form a strict nesting. `rehearsal.yaml` is the smallest, `debug.yaml` is middle, `default.yaml` is the full production shape. A smaller config may **only** reduce *scale knobs*; it must match the larger config on every *structural knob*. If a smaller config diverges structurally, it stops predicting the larger config's behavior — a clean rehearsal no longer implies a clean debug or default run.

**Scale knobs — may differ down the chain:**
- `num_crawl_steps` (e.g. rehearsal=2, debug=3, default=10)
- `max_crawl_topics`
- `max_generated_tokens`, `max_refusal_check_generated_tokens`
- `max_topics_per_step_lang`
- `generation_batch_size`
- `num_refusal_checks_per_topic` (smaller in rehearsal is acceptable if the progressive-triage shape stays valid, e.g. 3+2 still fits)

**Structural knobs — must match exactly:**
- `crawler_type` (aggregate vs legacy — **this is the one that bit us on 2026-04-19; rehearsal.yaml was missing `crawler_type: aggregate` and silently ran the legacy crawler instead**)
- `prompt_languages` set **identity and order** (a rehearsal that runs only chinese while debug/default run [english, chinese] is a subset of *work* but must not disagree on list order or drop the key entirely)
- `extraction_batch_size`
- `semantic_group_batch_size`
- `is_refusal_threshold`
- `seed_warmup_steps` (may be reduced only if warmup template count supports it; LEARNINGS §"Cap semantics" applies)
- `do_filter_refusals`
- `max_topic_string_length`
- `max_concurrent_summarizations`

**Before editing any crawler yaml:** diff against the other two. `diff configs/crawler/default.yaml configs/crawler/debug.yaml` and `diff configs/crawler/debug.yaml configs/crawler/rehearsal.yaml` should surface only scale-knob differences. A structural diff is a bug to fix before the next live run.

---

## Topic extraction

- **Gate on "is this a topic list" first.** Target models occasionally hallucinate training data (copyright pages, code solutions) instead of topic lists. The extraction prompt's first instruction must be the gate check; extraction only proceeds if the response is a topic list.

- **Extraction granularity is bounded by upstream.** The grouping LLM can only preserve the granularity it receives. If extraction coalesces `"Taiwan presidential election"` into `"taiwan issue"`, grouping cannot recover the finer grain. Enforce event-level granularity at extraction time.

- **CCP-aligned models underreport without jailbreak.** DeepSeek/Qwen refusal rates are ~71% vs 97%+ for GPT/Haiku on the neutral path. This is a finding, not a pipeline bug. Forgery-style elicitation is required to surface political restrictions.

---

## Config and code hygiene

- **Encode experiment variants in YAML, not CLI overrides.** Ad hoc overrides make results non-reproducible. One-off diagnostic flags are the only acceptable overrides.
- **Quote YAML values containing colons.** `key: value` parses as a dict. Test every template in every configured language before merging.
- **When relaxing an invariant, grep every consumer.** Adding a new null-able mode (e.g. `assistant_pre_templates: null`) that was previously always populated crashes downstream silently. Add guards at introduction time.
- **New scripts belong in `scripts/`, reusing `src/` library code.** `/exp` is legacy and broken; do not touch it.

---

## Testing

- **Test the behavioral contract, not just the plumbing.** If a feature has a contract ("warmup uses broad templates, not seed drill-down"), test that contract directly. Full crawl runs are too stochastic and expensive to catch silent behavioral regressions.
- **Validate prompt fixes with local fixtures, not live runs.** Feed the exact contaminating inputs as test fixtures and assert expected output. Live runs are expensive and may not reproduce the bug.
- **No live API calls in builder or reviewer scope.** Build and review are verification-only. The head runs the live validation crawl after APPROVED.

---

## Silent-discard paths audited (2026-04-18)

Head-written audit of every hot-path site in the five aggregate-path files.
Format: `file:line — what silently fails — downstream effect — fix or intentional`.

### Fixed by S0a/S0b/S0c

- `grouping_pipeline.py:280-311` (pre-S0a) — `except APITimeoutError: print(...); continue` on both the local-model and API paths. The entire timed-out batch returned with `summary=None` and was filtered out in `aggregate_crawler.py:229`. **Fixed by S0a**: re-raise by default; opt-in preservation via `grouping_timeout_preserves_batch=True`.

- `grouping_pipeline.py:165-175` (pre-S0b) — out-of-range negative `cluster_idx` left as-is (e.g. `-2` when only 1 known head). Downstream `topic_queue.py:append_to_cluster` used Python negative-index wrap, silently adding a non-head topic to the wrong cluster. Out-of-range positive cidx fell through to a no-op `else` branch and left the topic unassigned, causing it to be treated as a new head with an unvalidated cidx. **Fixed by S0b**: both cases now downgrade to new-head with a contiguous cluster_idx. Belt-and-braces guard in `TopicQueue.append_to_cluster` raises `ValueError` loudly on any invalid cidx.

- `run_crawler.py:4` (pre-S0c) — `load_dotenv(override=True)` at module top. Any repo-local `.env` silently clobbered shell/CI env vars (wrong API key, wrong budget). **Fixed by S0c**: default is `override=False`; opt-in via `CRAWLER_DOTENV_OVERRIDE=1`.

### Intentional (documented here, no fix needed)

- `progressive_refusal.py:408-409` and `490-491` — `except APITimeoutError: responses = [""] * n`. **Intentional**: empty-string responses are excluded from the triage/escalation vote (`resp.strip()` gate in `classify_refusal_triage`), so a timeout produces an all-empty batch that is correctly marked `inconclusive=True` rather than `is_refusal=False`. This implements the LEARNINGS "Timeouts must fail loudly" contract at the progressive-refusal level.

- `refusal_utils.py:200-202` — `if not text or not text.strip(): refusals.append(False); continue`. **Intentional**: empty responses excluded from refusal cascade; the progressive voter already accounts for this with the non-empty count. No topic is lost — the False placeholder is overridden by the vote math in the caller.

- `aggregate_crawler.py:212-213` — `if not new_topics: continue` (after extraction). **Intentional**: if the model response contained no topic list (e.g. returned code, math, off-topic prose), there is nothing to group or check. Covered by a `print` warning in `extract_and_translate`.

- `aggregate_crawler.py:231-232` — `if not new_topics: continue` (after summary filter). **Intentional**: if every extracted topic was flagged as preamble/garbage by the grouping LLM (all have `summary=None`), there are no valid heads to check. This is correct deduplication, not data loss.

### Fixed this iteration — S0e, S0f, S0g, S0h (2026-04-18)

- **S0e** — `response_formatting_utils.py:121-124` — `except Exception as e: return [[] for _ in texts]`. Local-model extraction: any exception (OOM, crash, malformed config) silently returns an all-empty extraction for the entire batch. Downstream: all topics in that step-lang silently lost. **Verdict: FIXED (response_formatting_utils.py:121-131)** — narrowed to `except json.JSONDecodeError` with `logging.exception` + return empties; all other exceptions re-raise. Tests: `tests/test_s0e_local_extraction_errors.py`.

- **S0f** — `response_formatting_utils.py:638-640` and `703-705` — `except APITimeoutError: print("Warning: extraction timed out, returning partial results"); return formatted_topics`. `formatted_topics` is initialized to `[]` before the try block. If `_extract_with_model` raises, the function returns an empty list silently. This path was **effectively dead** because `_extract_with_model` caught all exceptions internally. **Verdict: FIXED (response_formatting_utils.py:656-665, 728-737)** — after S0h propagates `APITimeoutError`, the outer handlers now re-raise with a `logging.warning`. Tests: `tests/test_s0f_outer_timeout_handler.py`.

- **S0g** — `response_formatting_utils.py:836-840` and `900-905` — `except Exception as e: topic.summary = topic.shortened` for both local-model and API summarization. On any exception, all in-flight summaries silently fall back to `shortened`. **Verdict: FIXED (response_formatting_utils.py:876-893, 952-969)** — narrowed to `except APITimeoutError` (log warning + fallback) and `except Exception` (log + re-raise). Tests: `tests/test_s0g_summarization_errors.py`.

- **S0h** — `response_formatting_utils.py:203-221` — `except Exception as e: return [[] for _ in chunk_texts]` inside `_call_one_chunk` (remote extraction path). On any non-403 transport exception, the entire chunk returned an all-empty list silently. The 403 path correctly returns `None` to trigger a split-retry, but the non-403 path discarded the chunk with no split and no re-raise. **Verdict: FIXED (response_formatting_utils.py:207-237)** — `APITimeoutError` re-raises with `logging.warning`; all other unexpected exceptions re-raise with `logging.exception`; 403 path unchanged. Tests: `tests/test_s0h_remote_extraction_errors.py`.
