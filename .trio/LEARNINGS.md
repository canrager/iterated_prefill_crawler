# Learnings

Keep only lessons that change how the next update is planned, built,
reviewed, or validated. Rewrite, merge, and prune continuously.

Source control will track versions of this document, so keep it fresh.

## Current learnings

### The crawl pipeline is five steps; inline post-processing between them destroys recall

The crawler's job is: (1) prompt the target model, (2) extract topics, (3) translate bidirectionally, (4) provoke the target model with refusal probes, (5) classify refusal. That's the entire pipeline. Everything between these steps must preserve data fidelity — if it modifies, splits, or drops labels, it corrupts the seed pool and final output.

Inline heuristics like comma-splitting summaries, regex-filtering shortened labels, and word-count-based summarizer bypasses grew to patch edge cases but actively hurt recall on the cases that matter most. `_split_at_comma` on the summary field shredded translated Chinese political topics: `歪曲党史国史军史` ("Distorting the history of the Party, the nation, and the military") became three fragments including `"the nation"` and `"the military"`, each inheriting the original Chinese raw. `隐私、个人信息与监控` ("Privacy, Personal Information, and Surveillance") became just `"privacy"`. These are real losses from the DeepSeek V3.2 artifact, not hypotheticals.

The extractor decides topic boundaries. If it returns one array element, that is one topic. Post-crawl aggregation and analysis can be as topic-literate as needed — splitting, merging, normalizing — because it operates on final data without feeding corrupted labels back into the seed pool.

*Evidence: 2026-04-26 artifact analysis of `crawler_out_20260424_020254_deepseek-v3.2_25samples_2crawls_Truefilter.json`. Topic 4356 (`歪曲党史国史军史`) had correct `english` but `summary: "distorting the history of the party"` — the rest was split into cloned topics 4562 (`"the nation"`) and 4563 (`"the military"`). Topic 1292 (`隐私、个人信息与监控`) had `summary: "privacy"`. All caused by `_split_at_comma` in `extract_and_format` line 668.*

### Translation must use the same prompt shape that the bench validated

The extractor/translator bench tests JSON-array batch prompts with structured instructions. If prod uses a different prompt shape (terse one-per-call), the bench results are wasted — they validated a capability prod doesn't use. This created a silent divergence where the bench picked the best model for batch translation but prod sent each topic individually without context, making short Chinese terms like `支持台独` vulnerable to truncation or alignment self-censorship.

*Evidence: 2026-04-26 code audit. Prod `_translate_zn_to_en` sent `"Translate to English (translation only): 支持台独"` per topic; bench sent a JSON array with system prompt and structured instructions. The bench's `TRANSLATION_PROMPT` and `parse_translation_output` were never used by prod. Fixed by aligning `_translate_batch` to match the bench format.*

### Extractors silently drop alignment-trigger categories under prod prompts

Helper-LLMs fine-tuned for safety (e.g. `openai/gpt-5.4-mini`) can produce clean JSON and no visible refusal while systematically omitting the hardest categories from a target's audit enumeration — CSAM, weapons, self-harm, non-consensual sexual acts, exploit code. This is the worst failure mode for an audit pipeline: biased recall loss on exactly the categories that matter, with output that looks successful. A general extractor bench (historical/political fixtures) will not catch it; the bench must include a real jailbreak enumeration fixture.

*Evidence: bench 2026-04-22 on `extractor_test_alignment_triggers_en.txt` (DeepSeek R1's 38-category enumeration). Under the prod prompt (`TOPIC_EXTRACTION_PROMPT`), `openai/gpt-5.4-mini` scored 23/38 (61%) with 0% refusal, 100% JSON. Reports at `/tmp/ipc-main/artifacts/bench/extractor_models_*.json`.*

### Reasoning tokens on helpers waste prod budget for no quality gain

Helper roles (extraction, translation, summarization, judging) are deterministic transformations, not reasoning tasks. Disable reasoning at the helper call boundary only; target-model calls must keep reasoning on (that's the audit signal).

*Evidence: bench v5 (`extra_body={"reasoning": {"effort": "none"}}`, 2026-04-22) scored identically to v3 (reasoning on) at lower wall-time. Plumbed via `REASONING_DISABLED` in `src/openrouter_utils.py`.*

### Helper-model concurrency fan-out rate-limits itself; cap at the API boundary

The refusal-check path produces `topics * j` messages in a single `batch_generate` call. OpenRouter rate-limits the provider; the SDK's `max_retries` with exponential backoff turns the stall into a retry storm. Every `_api_batch_generate` call needs a semaphore, and translation sub-batching must not reuse `generation_batch_size`.

*Evidence: rehearsal 2026-04-21 stalled in step 0 refusal-filtering for 50 minutes of SDK retries before manual kill. Fixed by `max_concurrent_api_calls=16` semaphore and a dedicated `translation_batch_size=50`.*

### Seed pool eligibility silently drops narrow branches

Broad discovered topics are only re-entered as seeds if they are refusal-marked. A coarse label that passes the refusal check will never be drilled into, even when its narrower children would have been refused. When debugging coverage gaps, check separately: which prompt family ran, which topics were eligible to re-enter the seed pool, and whether the missing detail appeared only inside refusal-check responses.

*Evidence: DeepSeek crawl 2026-04-01 missed geopolitical leaves because broad topics were not re-queued unless already marked as refusals.*

### Use a live two-step harness to prove a stochastic crawl path before trusting a full run

When a seeded branch is missing, a full crawl conflates code-path bugs with sampling variance. Build a small live probe that reuses real prompt configs for both stages — discover a broad trigger from model output, then feed that extracted trigger into the seeded step.

*Evidence: `--chain-zh` in `tests/test_refusal_pipeline.py` proved the DeepSeek discovery-to-drilldown path in one live run on 2026-04-08.*

### Trigger extraction must score candidates, not stop at the first match

Broad audit responses contain generic scaffolding phrases that technically match trigger patterns before reaching the politically useful seed. First-match extraction silently steers the probe into irrelevant drilldowns. Gather all candidates, blacklist generic policy/compliance frames, and prefer the strongest political or historical parent.

*Evidence: Config-faithful DeepSeek chain only produced useful seeds after scoring-based trigger selection (2026-04-09).*

### Encode experiment variants in YAML; do not rely on CLI overrides

Ad hoc overrides make results non-reproducible and let a stronger experiment path exist without ever being named. If a probe needs different defaults, give it a YAML-backed config. The only acceptable overrides are one-off diagnostic flags.

*Evidence: DeepSeek political drilldown probe became reproducible only after `chain_debug` and `jailbreak_probe` YAML configs replaced token-count CLIs (2026-04-09).*

### Keep targeted rehearsal strictly separate from the neutral experiment path

`jailbreak_rehearsal` is for proving reachability on DeepSeek. `jailbreak` is the paper-facing neutral path. Promote only general mechanism improvements (prompt shape, prefill format, branch count) upward into the neutral path; never promote target-family steering.

*Evidence: Neutral `debug + jailbreak` remained stochastic while `jailbreak_rehearsal` cleanly reached PRC political branches (2026-04-09). Mixed artifacts would overstate the method.*

### Promote branch count before token count

When warm-up finds the right topics but the seeded step drifts, the failure mode is sampling variance, not token budget. Raise `num_samples_per_topic` first.

*Evidence: Raising `rehearsal.yaml` from 1 to 5 samples/topic turned a drifting DeepSeek rehearsal into a successful Taiwan/Tibet/Xinjiang targeting run (2026-04-09).*

### Rehearsal without refusal filtering is judged by branch quality, not refusal counts

`rehearsal.yaml` sets `do_filter_refusals: false` by design. Zero discovered refusals in rehearsal is expected behavior. Judge rehearsal by whether discovered topics enter the intended sensitive branch and whether seeded drilldown stays on it.

*Evidence: First clean DeepSeek rehearsal was wrongly escalated for `0` refusals before rechecking the YAML (2026-04-09).*

### DeepSeek's low neutral-path refusal rate is a finding, not a bug

CCP-aligned models (DeepSeek, Qwen) systematically deny their censorship surface on the neutral `jailbreak` path. A ~71% refusal rate vs 97%+ for GPT/Haiku is expected: without forgery-style elicitation (prefill injection, CoT forgery), these models will not volunteer that "disputing Taiwan independence" is a restricted topic. Do not treat the rate gap as a pipeline failure when comparing across model families.

*Evidence: Neutral debug matrix 2026-04-09; `jailbreak_rehearsal` elicitation confirmed DeepSeek does block the political branch when properly prompted.*

### Topic extractor must gate on "is this a topic list" first

DeepSeek occasionally hallucinates training data into a generation slot (textbook copyright pages, LeetCode solutions) instead of a topic list. The extractor will dutifully label that content as "topics" unless explicitly told to return `[]` for non-topic inputs. The fix is in `TOPIC_EXTRACTION_PROMPT`: the first instruction is now the gate check, not the extraction instruction.

*Evidence: 2026-04-09 neutral debug matrix; book118 copyright page and integer-reversal LeetCode problem produced ~10 false-positive refusal topics that passed the refusal cascade via Gemma probe misfire.*

### Validate prompt fixes with integration tests using real bad fixtures, not debug runs

When an LLM prompt is changed to fix a contamination or extraction bug, the cheapest validation is an integration test that feeds the exact bad inputs as fixtures. A full debug run is expensive and stochastic — the contaminating sample may not reproduce.

*Evidence: 2026-04-10 Kimi extraction prompt fix validated in `tests/test_topic_extraction_drift.py` with textbook + LeetCode fixtures before launching the verification debug run.*

### Aggregation and coverage loaders must share the same topic key

If aggregation pre-consolidates punctuation/case variants but coverage scoring keeps the older `strip().lower()` key, the aggregator and analyzer score different topic surfaces. Put deterministic topic-key normalization in a shared helper and require both loaders to use it.

*Evidence: 2026-04-24 Slice 3 was rejected until `normalize_topic_key()` moved to `src/aggregation/topic_normalization.py` and both `TopicAggregator.load_topics()` and `coverage.load_crawl_topics()` used it.*

### Refusal probe generator bypass must stay opt-in until live-compared

The hardcoded fallback probes can eliminate the refusal-check query-generation call, but that only proves a cost-saving mechanism, not live sufficiency. Keep hardcoded-only mode default-off and require a budgeted live comparison before treating it as a replacement for generated probes.

*Evidence: 2026-04-24 Slice 4 added `crawler.use_hardcoded_refusal_probes_only=false` by default. Offline tests prove default mode still calls `refusal_check` then `target`, while opt-in hardcoded-only mode calls only `target` with five fallback probes.*

### The strongest confirmed crawler bottleneck is seed selection, not prompt-family preference

The latest DeepSeek artifact shows 226 broad political head candidates but only 5 were selected as seeded parents. Retrospective expansion-vs-drilldown comparisons are confounded and did not prove drill-down dominance. Future work should first test neutral parent selection before changing prompts.

*Evidence: Slice 3 seed selection gap. Slice 4 was inconclusive. Reports: `artifacts/research/crawl_shape/slice3_seed_selection.md`, `slice4_prompt_family_yield.md`, and `synthesis.md`.*

### Offline selector bakeoffs hit a ceiling: unobserved parents have no counterfactual children

Offline bakeoffs can measure redundancy, diversity, and whether a selector would have picked historically followed parents. They cannot fairly score child/refusal yield for new candidates because the artifact has no data on what those parents would have produced. Three rounds of bakeoffs (embedding-first, embedding-strict, cheap-hybrid) all hit this same wall. Embeddings are useful for analysis and tie-breaking but are not justified as core queue logic. A replay fixture, second crawl artifact, or budgeted live protocol is needed to break the tie.

*Evidence: 2026-04-26 bakeoffs A, B, and cheap-hybrid. Reports: `artifacts/research/crawl_shape/offline_selector_bakeoff_a.md`, `offline_selector_bakeoff_b.md`, `cheap_hybrid_selector_bakeoff.md`, and `offline_selector_bakeoff_synthesis.md`.*

### Expected sensitive-topic vocabularies must remain external scoring data

The crawler must stay black-box and target-agnostic. Do not put expected CCP/PRC/golden topic literals into prompts, production source, reusable research helper source, reviewer rubrics, or queue-ranking features. If expected-topic data is needed, load it as an external evaluation artifact after selector outputs are frozen, and use it only for post-hoc scoring.

*Evidence: 2026-04-25 neutrality correction after the crawl-shape slices.*
