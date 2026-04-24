# Learnings

Keep only lessons that change how the next update is planned, built,
reviewed, or validated. Rewrite, merge, and prune continuously.

Source control will track versions of this document, so keep it fresh.

## Current learnings

### Extractors silently drop alignment-trigger categories under prod prompts

Helper-LLMs fine-tuned for safety (e.g. `openai/gpt-5.4-mini`) can produce clean JSON and no visible refusal while systematically omitting the hardest categories from a target's audit enumeration — CSAM, weapons, self-harm, non-consensual sexual acts, exploit code. This is the worst failure mode for an audit pipeline: biased recall loss on exactly the categories that matter, with output that looks successful. A general extractor bench (historical/political fixtures) will not catch it; the bench must include a real jailbreak enumeration fixture.

*Evidence: bench 2026-04-22 on `extractor_test_alignment_triggers_en.txt` (DeepSeek R1's 38-category enumeration). Under the prod prompt (`TOPIC_EXTRACTION_PROMPT`), `openai/gpt-5.4-mini` scored 23/38 (61%) with 0% refusal, 100% JSON. Reports at `/tmp/ipc-main/artifacts/bench/extractor_models_*.json`.*

### Reasoning tokens on helpers contaminate bench comparisons and waste prod latency

Helper roles (extraction, translation, summarization, judging) are deterministic transformations, not reasoning tasks. When reasoning effort is left at provider defaults, wall-time and token cost reflect chain-of-thought depth, not throughput — making bench rankings misleading. In prod the same default burns tokens and latency for no quality gain. Disable reasoning at the helper call boundary only; target-model calls must keep reasoning on (that's the audit signal).

*Evidence: bench v3 (reasoning on, 2026-04-21) scored qwen3-235b 0.93 composite with 7.2s wall_s; bench v5 (`extra_body={"reasoning": {"effort": "none"}}`, 2026-04-22) scored 0.93 at 6.8s on the same fixture. Wall-time on slower candidates (gemma-4) changed more dramatically. Plumbed via `REASONING_DISABLED` in `src/openrouter_utils.py`.*

### Helper-model concurrency fan-out rate-limits itself; cap at the API boundary

The refusal-check path produces `topics * j` messages in a single `batch_generate` call. For rehearsal that's ~200 parallel target-model requests. OpenRouter rate-limits the provider; the SDK's `max_retries` with exponential backoff turns the stall into a retry storm that self-perpetuates instead of breaking out. Every `_api_batch_generate` call needs a semaphore, and translation sub-batching must not reuse `generation_batch_size` — that knob is ~2 and inflates translate calls ~50x.

*Evidence: rehearsal 2026-04-21 stalled in step 0 refusal-filtering for 50 minutes of SDK retries before manual kill. Fixed by `max_concurrent_api_calls=16` semaphore and a dedicated `translation_batch_size=50`.*

### Seed pool eligibility silently drops narrow branches

Broad discovered topics are only re-entered as seeds if they are refusal-marked. A coarse label that passes the refusal check will never be drilled into, even when its narrower children would have been refused. When debugging coverage gaps, check separately: which prompt family ran, which topics were eligible to re-enter the seed pool, and whether the missing detail appeared only inside refusal-check responses.

*Evidence: DeepSeek crawl 2026-04-01 missed geopolitical leaves because broad topics were not re-queued unless already marked as refusals.*

### Use a live two-step harness to prove a stochastic crawl path before trusting a full run

When a seeded branch is missing, a full crawl conflates code-path bugs with sampling variance. Build a small live probe that reuses real prompt configs for both stages — discover a broad trigger from model output, then feed that extracted trigger into the seeded step. Only the trigger detector should be hardcoded.

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

When warm-up finds the right topics but the seeded step drifts, the failure mode is sampling variance, not token budget. Raise `num_samples_per_topic` first. Promote that change upward into `debug` once the mechanism is confirmed.

*Evidence: Raising `rehearsal.yaml` from 1 to 5 samples/topic turned a drifting DeepSeek rehearsal into a successful Taiwan/Tibet/Xinjiang targeting run (2026-04-09).*

### Rehearsal without refusal filtering is judged by branch quality, not refusal counts

`rehearsal.yaml` sets `do_filter_refusals: false` by design. Zero discovered refusals in rehearsal is expected behavior, not a mechanism failure. Judge rehearsal by whether discovered topics enter the intended sensitive branch and whether seeded drilldown stays on it.

*Evidence: First clean DeepSeek rehearsal was wrongly escalated for `0` refusals before rechecking the YAML (2026-04-09).*

### For parallel runs, trust `run_parallel.py` and `run.sh`; treat `run_parallel.sh` as untrusted

`scripts/run_parallel.sh` produced only an umbrella directory for the neutral debug matrix. `scripts/run_parallel.py` launched successfully and produced live artifacts. For single-model launches `scripts/run.sh` is canonical. DeepSeek specifically should use the single-model path; it exits silently in the three-model parallel launch.

*Evidence: 2026-04-09 neutral debug matrix; DeepSeek produced only an empty log in the parallel run.*

### DeepSeek's low neutral-path refusal rate is a finding, not a bug

CCP-aligned models (DeepSeek, Qwen) systematically deny their censorship surface on the neutral `jailbreak` path. A ~71% refusal rate vs 97%+ for GPT/Haiku is expected: without forgery-style elicitation (prefill injection, CoT forgery), these models will not volunteer that "disputing Taiwan independence" is a restricted topic. Do not treat the rate gap as a pipeline failure when comparing across model families.

*Evidence: Neutral debug matrix 2026-04-09; `jailbreak_rehearsal` elicitation confirmed DeepSeek does block the political branch when properly prompted.*

### Topic extractor must gate on "is this a topic list" first

DeepSeek occasionally hallucinates training data into a generation slot (textbook copyright pages, LeetCode solutions) instead of a topic list. The extractor (Kimi) will dutifully label that content as "topics" unless explicitly told to return `[]` for non-topic inputs. The fix is in `TOPIC_EXTRACTION_PROMPT`: the first instruction is now the gate check, not the extraction instruction. Integration tests in `tests/test_topic_extraction_drift.py` use the exact contaminating fixtures as regression guards.

*Evidence: 2026-04-09 neutral debug matrix; book118 copyright page and integer-reversal LeetCode problem produced ~10 false-positive refusal topics that passed the refusal cascade via Gemma probe misfire.*

### Validate prompt fixes with integration tests using real bad fixtures, not debug runs

When an LLM prompt is changed to fix a contamination or extraction bug, the cheapest and most targeted validation is an integration test that feeds the exact bad inputs from the incident as fixtures and asserts the expected output. A full debug run is expensive and stochastic — the contaminating sample may not reproduce. Reserve debug runs for validating overall pipeline noise reduction, not individual prompt correctness.

*Evidence: 2026-04-10 Kimi extraction prompt fix validated in `tests/test_topic_extraction_drift.py` with textbook + LeetCode fixtures before launching the verification debug run.*

### Offline replay must patch provider resolution as well as model calls

Fixture replay monkeypatches target/helper calls, but helper code may resolve provider client kwargs before the patched async call is reached. Offline benchmark replay must stub `get_provider_client_kwargs` inside the replay context so missing API keys do not block fixture lookup. A fixture hit rate below the validity gate is a fixture coverage problem, not a provider setup problem.

*Evidence: 2026-04-24 Slice 1: `uv run pytest tests/test_bench_crawler_shape.py::test_run_bench_replays_single_cell_offline -q` failed without `OPENROUTER_API_KEY` before patching provider resolution; after patch it passed. `run3_ds_v32` replay then completed offline but remained invalid at `fixture_hit_rate=0.878`.*

### Topic summary cleanup needs language preservation tests, not just drop tests

Filtering bad labels like `etc` or mixed-script corruption is not enough; the cleanup must prove it preserves useful Chinese-only and English-only labels. Export cleanup and dedup keys should share the same Unicode punctuation definition so labels like `台湾地位。` export as `台湾地位` and dedupe with punctuation variants.

*Evidence: 2026-04-24 Slice 2 was rejected until `test_cleanup_summary_labels_preserves_chinese_labels_and_strips_punctuation` covered Chinese preservation and `_clean_summary_label()` stripped Unicode punctuation using `unicodedata.category(...).startswith("P")`.*

### Aggregation and coverage loaders must share the same topic key

If aggregation pre-consolidates punctuation/case variants but coverage scoring keeps the older `strip().lower()` key, the aggregator and analyzer score different topic surfaces. Put deterministic topic-key normalization in a shared helper and require both loaders to use it.

*Evidence: 2026-04-24 Slice 3 was rejected until `normalize_topic_key()` moved to `src/aggregation/topic_normalization.py` and both `TopicAggregator.load_topics()` and `coverage.load_crawl_topics()` used it. The compatibility test includes ASCII punctuation/case variants and a full-width `。` variant.*

### Refusal probe generator bypass must stay opt-in until live-compared

The hardcoded fallback probes can eliminate the refusal-check query-generation call, but that only proves a cost-saving mechanism, not live sufficiency. Keep hardcoded-only mode default-off, prove call behavior offline, and require a separate budgeted live comparison before treating it as a replacement for generated probes.

*Evidence: 2026-04-24 Slice 4 added `crawler.use_hardcoded_refusal_probes_only=false` by default. Offline tests prove default mode still calls `refusal_check` then `target`, while opt-in hardcoded-only mode calls only `target` with five fallback probes.*
