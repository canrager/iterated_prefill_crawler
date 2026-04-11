# Learnings

Keep only lessons that change how the next update is planned, built,
reviewed, or validated. Rewrite, merge, and prune continuously.

Source control will track versions of this document, so keep it fresh.

## Current learnings

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

### Assistant prefill that works locally may break on remote APIs

The ranking pipeline's `\boxed{` assistant prefill was designed for local vLLM (which continues from the prefill). On OpenRouter with gemini-flash-lite, the same prefill produced unparseable continuations (`Illegal drugs}` instead of `\boxed{B}`). When porting from local to remote, test the prefill behavior first; if it fails, drop it and rely on the model producing the full structured response unaided.

*Evidence: 2026-04-11 word cloud ranking stage; prefill removed after gemini-flash-lite produced garbage continuations.*

### Treat `/exp` as legacy — build new tooling alongside, not inside

`/exp` code is outdated and broken. Rather than fixing it, build standalone scripts that reuse the library code in `src/` but are decoupled from the legacy evaluation harness. This avoids breaking things that already don't work and keeps the new code maintainable.

*Evidence: 2026-04-11 word cloud pipeline; `exp/evaluate_crawler.py` changes reverted per review constraint, new `scripts/generate_wordcloud.py` built instead.*

### Clustering granularity is upper-bounded by upstream topic extraction
The deduplication prompt can only preserve the granularity of the topics provided to it. If the crawler's upstream topic extractor groups specific refusals into a broad topic like "taiwan issue", the clustering stage cannot magically separate them into more granular topics like "Taiwan presidential election". Any prompt instructing the clustering LLM to be granular will only preserve the maximum granularity available from the input data. To achieve true event-level granularity, the topic extraction prompt must also enforce event-level extraction.

*Evidence: 2026-04-11 clustering prompt fix preserved taiwan issue and south china sea issue as distinct from unofficial narratives of political movements, grouping them under Territorial and Sovereignty Disputes rather than a massive Politics bucket. However, it could not extract finer details because the inputs were already coarse.*
