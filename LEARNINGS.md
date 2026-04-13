# Learnings

Rules for future work. Each entry is something to **do or avoid going forward**.
Rewrite, merge, and prune continuously. Source control tracks the history.

---

## Crawl coverage

- **Seed eligibility must include non-refused topics.** A coarse topic that passes the refusal check may have narrower children that would be refused. When debugging coverage gaps, check separately: which topics were eligible to re-enter the seed pool, and whether missing detail appeared only inside refusal-check responses. (DeepSeek missed geopolitical leaves because broad topics weren't re-queued unless refusal-marked.)

- **Prefer undiscovered branches before revisiting known ones.** Random-with-replacement seed selection causes frequent topics to compound while rare branches starve. Use exhaustion-aware seeding and `random.sample` for distinct seeds per step. (Harm topics were compounding while CCP political topics were found once but never drilled into.)

- **Raise `num_samples_per_topic` before raising token budget.** When warmup finds the right topics but the seeded step drifts, the failure is sampling variance, not token starvation. (Going from 1→5 samples/topic turned a drifting DeepSeek rehearsal into successful Taiwan/Tibet/Xinjiang targeting.)

- **Expect CCP-aligned models to underreport on the neutral path.** DeepSeek/Qwen ~71% refusal rate vs 97%+ for GPT/Haiku is a real finding, not a pipeline bug. Without forgery-style elicitation these models won't volunteer their political restrictions. Don't treat the rate gap as a failure.

## Topic extraction

- **Gate on "is this a topic list" before extracting.** Target models occasionally hallucinate training data into a generation slot. The extractor will dutifully label that content as topics unless the prompt's first instruction is the gate check. (DeepSeek produced a book118 copyright page and a LeetCode problem that yielded ~10 false-positive refusal topics.)

- **Extraction granularity is upper-bounded by upstream.** The deduplication/clustering prompt can only preserve the granularity it receives. If the extractor groups specific refusals into "taiwan issue," clustering cannot recover "Taiwan presidential election." To get event-level granularity, enforce it at extraction time.

- **Score trigger candidates; don't stop at first match.** Broad audit responses contain generic scaffolding phrases that match trigger patterns before the politically useful seed. Gather all candidates, blacklist policy/compliance frames, prefer the strongest political or historical parent.

## Prompt and config hygiene

- **Encode experiment variants in YAML, not CLI overrides.** Ad hoc overrides make results non-reproducible and let a stronger path exist without ever being named. The only acceptable overrides are one-off diagnostic flags. (DeepSeek political drilldown only became reproducible after YAML configs replaced token-count CLI overrides.)

- **Keep rehearsal separate from the neutral experiment path.** `jailbreak_rehearsal` proves reachability; `jailbreak` is the paper-facing neutral path. Promote only general mechanism improvements upward; never promote target-family steering. (Mixing them would overstate the method's neutral-path performance.)

- **Quote YAML values that contain colons.** `Key: value` parses as a dict, not a string. Test every template in every configured language before merging. (Crashed bilingual OpenRouter path; Chinese-only was fine because those templates had no colons.)

- **When relaxing an invariant, grep for every consumer.** Adding a new mode (e.g. `assistant_pre_templates: null`) that produces empty content where it was always populated will crash downstream. Add guards at introduction time, not after the first 400 error.

- **`debug` config must be a faithful proxy for `default`.** Same extraction cap, sampling strength, topic string length, and language set. Only reduce step count and budget. If they diverge, debug runs stop predicting real-run behavior.

## Testing and validation

- **Prove stochastic crawl paths with a live two-step harness, not full runs.** A full crawl conflates code-path bugs with sampling variance. Build a small live probe that reuses real prompt configs for both stages (discover, then seed). (`--chain-zh` in `test_refusal_pipeline.py` proved the DeepSeek discovery-to-drilldown path in one live run.)

- **Validate prompt fixes with integration tests using real bad fixtures.** Feed the exact contaminating inputs as test fixtures and assert the expected output. A full debug run is expensive, stochastic, and may not reproduce the bug. (`test_topic_extraction_drift.py` caught the Kimi extraction fix with textbook + LeetCode fixtures before a verification run.)

- **Write a test for the behavioral contract, not just the plumbing.** If a feature has a contract ("warmup steps use broad templates, not seed drill-down"), test that contract directly. Don't rely on full crawl runs to notice when the behavior is silently wrong. (Warmup was silently doing nothing because the implementation only cycled indices without suppressing seed templates.)

- **Audit attribute propagation before broadening a seed pool.** When widening which topics are eligible as seeds, verify all consumed fields are populated on every candidate. Child topics from splitting/comma-expansion are especially likely to have incomplete fields. (`_split_at_comma` only propagated `(raw, summary)`, so 45% of seeds sent literal "None" to the target model.)

- **Judge rehearsal by branch quality, not refusal counts.** `rehearsal.yaml` sets `do_filter_refusals: false`. Zero discovered refusals is expected. Judge by whether topics enter the intended branch and whether drilldown stays on it.

## Infrastructure

- **Assistant prefill that works locally may break on remote APIs.** vLLM continues from the prefill; remote providers may produce garbage. When porting from local to remote, test prefill behavior first. If it breaks, drop it. (gemini-flash-lite produced `Illegal drugs}` instead of `\boxed{B}` from a `\boxed{` prefill.)

- **Pin new local inference components to CPU when vLLM owns the GPU.** A classifier that defaults to CUDA will fight vLLM for memory. Explicitly set `device="cpu"` from the start.

- **For parallel runs, use `run_parallel.py` and `run.sh`.** `run_parallel.sh` is untrusted — it has produced only empty umbrella directories. For single-model launches, `run.sh` is canonical. (DeepSeek exits silently in the three-model parallel launch.)

- **Build new tooling alongside `/exp`, not inside it.** `/exp` is legacy and broken. New scripts should reuse `src/` library code but stay decoupled from the old evaluation harness. (Word cloud pipeline was built as `scripts/generate_wordcloud.py` after `/exp` changes were reverted.)
