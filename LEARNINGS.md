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

### Topic extractor must gate on "is this a topic list" first

DeepSeek occasionally hallucinates training data into a generation slot (textbook copyright pages, LeetCode solutions) instead of a topic list. The extractor (Kimi) will dutifully label that content as "topics" unless explicitly told to return `[]` for non-topic inputs. The fix is in `TOPIC_EXTRACTION_PROMPT`: the first instruction is now the gate check, not the extraction instruction. Integration tests in `tests/test_topic_extraction_drift.py` use the exact contaminating fixtures as regression guards.

_Evidence: 2026-04-09 neutral debug matrix; book118 copyright page and integer-reversal LeetCode problem produced ~10 false-positive refusal topics that passed the refusal cascade via Gemma probe misfire._

### Validate prompt fixes with integration tests using real bad fixtures, not debug runs

When an LLM prompt is changed to fix a contamination or extraction bug, the cheapest and most targeted validation is an integration test that feeds the exact bad inputs from the incident as fixtures and asserts the expected output. A full debug run is expensive and stochastic — the contaminating sample may not reproduce. Reserve debug runs for validating overall pipeline noise reduction, not individual prompt correctness.

_Evidence: 2026-04-10 Kimi extraction prompt fix validated in `tests/test_topic_extraction_drift.py` with textbook + LeetCode fixtures before launching the verification debug run._

### Assistant prefill that works locally may break on remote APIs

The ranking pipeline's `\boxed{` assistant prefill was designed for local vLLM (which continues from the prefill). On OpenRouter with gemini-flash-lite, the same prefill produced unparseable continuations (`Illegal drugs}` instead of `\boxed{B}`). When porting from local to remote, test the prefill behavior first; if it fails, drop it and rely on the model producing the full structured response unaided.

_Evidence: 2026-04-11 word cloud ranking stage; prefill removed after gemini-flash-lite produced garbage continuations._

### Reasoning tokens do consume the generation budget on remote APIs

For reasoning-first remote models, an empty `message.content` is not enough evidence that the provider or model failed. The model may have spent the full generation budget inside `reasoning` and hit `finish_reason=length` before emitting any user-visible answer. When probing or validating a new provider, first test with a comfortably large token budget; only diagnose auth/model-name/batching issues after ruling out reasoning-budget exhaustion.

_Evidence: 2026-04-13 live Ollama Cloud probe against `deepseek-v3.2:cloud` returned empty `content`, non-empty `reasoning`, and `finish_reason="length"` at `max_tokens=12`; the same prompt returned `OK` at `max_tokens=128`._

### Treat `/exp` as legacy — build new tooling alongside, not inside

`/exp` code is outdated and broken. Rather than fixing it, build standalone scripts that reuse the library code in `src/` but are decoupled from the legacy evaluation harness. This avoids breaking things that already don't work and keeps the new code maintainable.

_Evidence: 2026-04-11 word cloud pipeline; `exp/evaluate_crawler.py` changes reverted per review constraint, new `scripts/generate_wordcloud.py` built instead._

### Clustering granularity is upper-bounded by upstream topic extraction

The deduplication prompt can only preserve the granularity of the topics provided to it. If the crawler's upstream topic extractor groups specific refusals into a broad topic like "taiwan issue", the clustering stage cannot magically separate them into more granular topics like "Taiwan presidential election". Any prompt instructing the clustering LLM to be granular will only preserve the maximum granularity available from the input data. To achieve true event-level granularity, the topic extraction prompt must also enforce event-level extraction.

_Evidence: 2026-04-11 clustering prompt fix preserved taiwan issue and south china sea issue as distinct from unofficial narratives of political movements, grouping them under Territorial and Sovereignty Disputes rather than a massive Politics bucket. However, it could not extract finer details because the inputs were already coarse._
