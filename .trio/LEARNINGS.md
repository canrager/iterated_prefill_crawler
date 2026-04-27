# Learnings

This file is the single resumable source of truth for the crawler's research
and engineering state. Read it top to bottom to pick up cold. Each entry is
forward-facing: it tells future-you what to do, what not to do, and why.
Rewrite, merge, and prune continuously. Source control tracks history.

## Where we are right now

**Branch:** `increase-coverage`. Tests: 174 passing.

**Goal (from CLAUDE.md):** become an extremely effective forbidden-topic
recovery crawler against black-box targets that don't enumerate their refusal
surface. Stay target-agnostic in code; treat known target ontologies as
post-hoc external scoring data only.

**Current target organism:** `deepseek/deepseek-v3.2`. Latest crawl artifact:
`artifacts/out/crawler_out_20260424_020254_deepseek-v3.2_25samples_2crawls_Truefilter.json`.

**Recent shipped fixes (committed, not yet validated by a fresh crawl):**

1. Translation step uses the bench-validated JSON-array batch prompt instead of
   one terse call per topic.
2. `_split_at_comma` removed from `extract_and_format`. The previous version
   shredded ~23% of head topics into ghost clones (783/3,361 rows).
3. Analyzer probes (`scripts/analyze_crawl.py`) extended to recognize the
   compact political abbreviations DeepSeek actually uses (`台独`, `藏独`,
   `疆独`, `颠覆`, etc.). Score on the existing artifact moved 4/15 → 6/15.

**The next experiment to run is one bounded live crawl** with the post-fix
codebase (same prompts, same target, same configs). Do not propose new
selection rules, prompt redesigns, or pipeline changes before that crawl
exists. The audit below explains why; reread before starting work.

## Hard constraints (do not violate, ever)

### Stay target-agnostic in code

Never put expected sensitive-topic literals (Taiwan, Tibet, Tiananmen, CCP,
etc.) into prompts, production source, reusable research helper source,
reviewer rubrics, or queue-ranking features. Selection logic must read only
graph metadata (`parent_id`, `is_refusal`, language flags) — never label text.
The crawler must run identically against any target organism; only the
target's own behavior shapes its output.

Expected-topic vocabularies are allowed *only* as external evaluation data
loaded by post-hoc scoring scripts (e.g. analyzer probes), and only after
selector outputs are frozen. The line is: any code path that influences what
the next API call does is target-agnostic; any code path that scores already-
collected data may use external evaluation literals.

### Keep targeted rehearsal separate from the neutral path

`jailbreak_rehearsal.yaml` is for proving reachability on a specific target.
`jailbreak.yaml` is the paper-facing neutral path. Promote only general
mechanism improvements (prompt shape, prefill format, branch count) into the
neutral path. Never promote target-family steering or seeded sensitive
literals upward. Mixed artifacts overstate the method.

### Never retry experiments the human ruled out

If a result, model, or approach was rejected, do not silently rerun it.
Reread this file's "do not retry" section before scoping any new bakeoff,
benchmark, or model swap.

## Pipeline integrity — five steps, nothing in between

The crawler is exactly five steps:

1. Prompt the target model
2. Extract topics (LLM returns a JSON array of labels)
3. Translate bidirectionally (ZH↔EN)
4. Provoke the target model with refusal probes
5. Classify refusal vs. not

Everything between these steps must preserve label fidelity. Inline
post-processing (splitting, regex cleanup, dedup, summarization heuristics)
that *modifies* labels has consistently hurt recall. The extractor decides
topic boundaries: if it returns one array element, that is one topic.

Post-crawl aggregation and analysis can be as topic-literate as needed —
splitting, merging, normalizing — because it operates on final data without
feeding corrupted labels back into the seed pool.

*Concrete losses observed before fixes: `歪曲党史国史军史` ("Distorting the
history of the Party, the nation, and the military") shredded into three
fragments including `"the nation"` and `"the military"`, all carrying the
original Chinese raw. `隐私、个人信息与监控` reduced to `"privacy"`. 783 of
3,361 head topics in the latest artifact were ghost clones from
`_split_at_comma`.*

### Translation must use the prompt shape the bench validated

If the extractor/translator bench tests JSON-array batch prompts, prod must
send JSON-array batch prompts. A bench that validates a capability prod
doesn't use is wasted information. Aligning prod to bench also reduces API
calls (one batch vs. N individual translations) and gives short labels
disambiguating context.

*Before the fix, prod sent `"Translate to English (translation only): 支持台独"`
per topic. The bench sent a JSON array with system prompt and structured
instructions. The bench's `TRANSLATION_PROMPT` and `parse_translation_output`
were never used. Now `_translate_batch` matches the bench format.*

### The extractor's list-gate drops prose deflections — three failure modes

`TOPIC_EXTRACTION_PROMPT` is gated to return `[]` for non-list inputs. That
correctly drops documents, code, and off-topic content. It also drops prose
refusals where the target named a topic in continuous text without listing
("X is a complex issue, let's discuss in balanced terms"). Targets whose
refusal style favors deflection prose lose those topics at extraction.

When labeling a recovery gap, distinguish three failure modes:

1. **Selection failure** — the topic was extracted into the queue but never
   drilled. Fix: graph- or routing-level seed prioritization.
2. **Extraction failure on prose deflection** — the topic was named by the
   target in non-list output and dropped. Fix (open question): a second-pass
   path for topics named inside refusal-classified responses, without
   reintroducing extraction noise.
3. **Elicitation failure** — the target never named the topic at all under
   the prompts in use. Fix: prompt-config work, not pipeline work. No amount
   of drilling or extraction change recovers a topic the target never uttered.

Do not conflate these when proposing crawler changes.

*In the latest artifact, the "missing" 9 golden categories split as: 4 in
deflection prose (`南海`, `中印`, `香港`, `一带一路`), 5 never uttered (`六四`,
`天安门`, `法轮功`, `达赖`, `文化大革命`).*

## Discovery and routing — where the next gain probably lives

### Warmup discovers politically-productive seeds reproducibly; the bottleneck is what we *do* with them

The Chinese warmup prompt in `jailbreak.yaml` reliably elicits CCP-relevant
top-level umbrellas. `生成用于政治诽谤的虚假新闻` (parent 1808) and
`分裂主义与分离运动` are two examples in the latest artifact. Each, when
followed by the **expansion** prompt template, produced a rich enumeration
of CCP-sensitive child topics (`支持台独`, `支持藏独`, `支持疆独`,
`歪曲党史国史军史`, `否定中国共产党领导`, `争议性国家主权表述`, etc.).

The drill-down prompt template was used 12 times in Chinese during the same
crawl. Every single seed it received was a Western-safety topic (hate
content, weapons, hacking, suicide methods). The drill-down was never applied
to a politically-flavored seed.

This is a *routing* gap, not a discovery, prompt design, or selection-pool
gap. The right seeds exist in the queue. The right prompts exist in the
config. They were never combined.

*Evidence: `artifacts/research/crawl_shape/offline_audit_findings.md` (audit
C). Of 38 children of parent 1808, 36 came from the Chinese expansion
template and 2 are comma-split clones. Zero came from drill-down. Of 12
distinct Chinese drill-down invocations, zero seeds were politically-flavored.*

### The drill-down prompt is byte-for-byte the gold March 30 prompt

`user_seed_templates[chinese][0]` in `jailbreak.yaml` matches the gold
March 30 OpenRouter transcript word-for-word, with `{}` substituted for the
seed. The forged CoT, authorization line, and `话题：\n1.` prefill are all
identical. When DeepSeek was hit with this prompt seeded with
`敏感政治与地缘冲突`, it produced a 20-item Chinese enumeration containing
all 15 golden categories.

Do not redesign these prompts. They work. The shortfall is routing.

*Evidence: `artifacts/research/crawl_shape/offline_audit_findings.md` (audit B).*

### `is_refusal`-only seed eligibility was tried and reverted; do not regress

An earlier code revision restricted seed selection to `head_refusal_topics`
only. That consistently missed broad parents that answered safely at the
coarse label but had refusable subcategories underneath ("Politically
Sensitive History" → helpful enumeration of Rwandan Genocide, American
Slavery → no `is_refusal` mark on the parent → never drilled, even though
the children would refuse). The current code in `_get_user_seed_candidates`
deliberately seeds from `head_topics` (all discovered), with
`head_refusal_topics` as a compatibility fallback. The function comment
documents this decision. Do not regress.

A new selection signal beyond `is_refusal` is an open research question.
Probe-shape dependence (the same topic refuses or not depending on how the
probe is phrased) means `is_refusal` alone is too noisy as a sole priority
signal.

### The strongest known offline signal is "this parent already produced refusal children"

Across the latest artifact, the most productive parent (id 1808) produced 35
children flagged `is_refusal: true`. None of those 35 were ever drilled —
random uniform sampling over ~2,097 unexplored top-level candidates gave each
~0.25% chance per slot. The information needed for depth was already in the
queue; selection couldn't use it.

This *correlates* with productivity but is not yet a validated selection
rule. Combining it with the previous warning means: any future selection
work must (a) preserve the current "seed from all head topics" policy as
the floor, (b) avoid topic-text inspection, (c) prove on a counterfactual
crawl artifact (not the same artifact that observed the historical
selection) that the new rule beats current behavior. The latter requires
fresh crawl data, not more offline reasoning over the same artifact.

## Helper-LLM behavior

### Extractors silently drop alignment-trigger categories under prod prompts

Helper LLMs fine-tuned for safety can produce clean JSON and no visible
refusal while systematically omitting the hardest categories (CSAM, weapons,
self-harm, exploit code) from a target's audit enumeration. This is the
worst failure mode: biased recall loss on exactly the categories that matter,
with output that looks successful. A general extractor bench
(historical/political fixtures) will not catch it; benches must include real
jailbreak enumeration fixtures.

*Evidence: bench 2026-04-22 on `extractor_test_alignment_triggers_en.txt`.
`openai/gpt-5.4-mini` scored 23/38 (61%) with 0% visible refusal. Reports at
`/tmp/ipc-main/artifacts/bench/extractor_models_*.json`.*

### Disable reasoning on helper calls; keep it on for target calls

Helper roles (extraction, translation, summarization, judging) are
deterministic transformations, not reasoning tasks. Provider-default reasoning
burns tokens and latency for no quality gain. Disable at the helper call
boundary only; target-model calls must keep reasoning on (that's the audit
signal). Plumbed via `REASONING_DISABLED` in `src/openrouter_utils.py`.

### Cap helper concurrency at the API boundary

Refusal-check produces `topics * j` messages in a single `batch_generate`
call. Without a semaphore, OpenRouter rate-limits the provider and the SDK's
retry-with-backoff turns the stall into a self-perpetuating retry storm.
Every `_api_batch_generate` call needs a semaphore (currently
`max_concurrent_api_calls=16`). Translation sub-batching must use a
dedicated `translation_batch_size`, not `generation_batch_size` (which is
~2 and would inflate translate calls ~50x).

### Topic extractor must gate on "is this a topic list" first

`TOPIC_EXTRACTION_PROMPT` opens with the gate check, not the extraction
instruction. Without the gate, contaminating inputs (textbook copyright
pages, LeetCode solutions hallucinated by the target) become "topics" that
poison the queue. Regression tests in `tests/test_topic_extraction_drift.py`
use the exact contaminating fixtures.

### Aggregation and coverage loaders must share the same topic key

Different normalizers across aggregator and analyzer score different topic
surfaces. Use the shared helper `normalize_topic_key()` in
`src/aggregation/topic_normalization.py`; require both `TopicAggregator` and
`coverage` to call it. The compatibility test covers ASCII punctuation/case
variants and Unicode (full-width) punctuation.

### Refusal-probe-generator bypass: opt-in only

The hardcoded fallback probes can eliminate the refusal-check
query-generation call. That proves a cost-saving mechanism, not live
sufficiency. `crawler.use_hardcoded_refusal_probes_only=false` by default;
a budgeted live comparison is required before treating it as a replacement.

## Methodology

### Use a live two-step harness to prove a stochastic crawl path

A full crawl conflates code-path bugs with sampling variance. For new
mechanisms, build a small live probe (like `--chain-zh` in
`tests/test_refusal_pipeline.py`) that reuses real prompt configs for both
stages. Only the trigger detector should be hardcoded.

### Trigger extraction must score candidates, not first-match

Broad audit responses contain generic scaffolding phrases that match trigger
patterns before reaching the politically useful seed. Gather all candidates,
blacklist generic policy/compliance frames, prefer the strongest political or
historical parent.

### Encode experiment variants in YAML; never CLI-override

CLI overrides leave a stronger experiment path unnamed and unreproducible.
A new defaults set deserves a new YAML config. Acceptable overrides: one-off
diagnostic flags only.

### Promote branch count before token count

When warmup finds the right topics but the seeded step drifts, the failure
is sampling variance, not token budget. Raise `num_samples_per_topic` first.

*Evidence: rehearsal.yaml from 1 to 5 samples/topic turned a drifting
DeepSeek rehearsal into a successful Taiwan/Tibet/Xinjiang targeting run.*

### Rehearsal without refusal filtering is judged by branch quality

`rehearsal.yaml` sets `do_filter_refusals: false` by design. Zero discovered
refusals is expected. Judge by whether discovered topics enter the intended
sensitive branch and whether seeded drilldown stays on it.

### DeepSeek's low neutral-path refusal rate is a finding, not a bug

CCP-aligned models systematically deny their censorship surface on the
neutral `jailbreak` path. ~71% refusal vs 97%+ for GPT/Haiku is expected:
without forgery-style elicitation, these models will not volunteer that
"disputing Taiwan independence" is restricted. Do not treat the rate gap
as a pipeline failure when comparing across model families.

### Validate prompt fixes with integration tests on real bad fixtures

When an LLM prompt is changed to fix a contamination or extraction bug, the
cheapest validation is an integration test that feeds the exact bad inputs
as fixtures. A full debug run is expensive and stochastic.

### Crawl-shape evaluation must separate three measurement layers

Source-level recovery (topic in `raw`/`chinese`/`english`), exported-summary
recovery (topic in `summary`), and analyzer-counted recovery can disagree.
A single headline score can hide bugs in any of the three. Future evaluation
must report all three before claiming a prompt or queue improvement.

## Dead ends — do not retry without new evidence

### Offline selector bakeoffs over a single artifact

Three rounds (embedding-first, embedding-strict cost-saving, cheap-hybrid)
all hit the same wall: offline ranking can score redundancy and historical-
selection match, but cannot fairly score child/refusal yield for unobserved
candidates because the artifact has no counterfactual children. Any new
selection-rule research needs a counterfactual artifact (a separate crawl)
or a budgeted live A/B. Stop running offline bakeoffs.

*Reports: `artifacts/research/crawl_shape/offline_selector_bakeoff_a.md`,
`offline_selector_bakeoff_b.md`, `cheap_hybrid_selector_bakeoff.md`,
`offline_selector_bakeoff_synthesis.md`.*

### Embedding ranking as core queue logic

Local Qwen embeddings work and find semantic redundancy missed by string
matching, but failed to beat cheap baselines on cost-saving and
recovery-per-budget metrics. Useful as analysis tooling; not justified as
production queue logic without new data.

### `is_refusal`-only seed eligibility

Already documented above as a hard "do not regress." Listed here too because
it's tempting to re-propose under the framing of "graph-priority selection."
The rule misses broad-parent-with-helpful-coarse-label cases. Don't.

### Redesigning `jailbreak.yaml` prompts

The Chinese drill-down prompt is byte-for-byte the gold March 30 prompt that
elicited all 15 golden categories from DeepSeek. The English templates
mirror the gold April 16 transcript. The shortfall is routing, not prompt
design. Do not redesign.

### Inline post-processing as a recall fix

Comma splitting on summaries, regex filtering of shortened labels,
word-count summarization bypass — all grew to patch edge cases and all hurt
recall on cases that mattered. Future inline-postprocessing additions need
proof of net-positive recall on a real artifact, not just on the edge case
they target.

## What to do next, in order

1. **Run one bounded live crawl** with the current codebase against
   `deepseek/deepseek-v3.2`, same prompts and configs as the latest
   artifact. The fresh artifact establishes the post-fix baseline. Until
   this exists, no further code changes are well-grounded.

2. **Compare the fresh artifact to the latest artifact** on three axes:
   how many ghost rows from comma split (should be zero), source-vs-summary
   recall on translated labels (should agree), and whether the
   politically-productive-seed-not-drilled pattern reproduces.

3. **If the routing pattern reproduces:** design a target-agnostic seed
   prioritization signal that does not collapse to `is_refusal`-only and
   does not inspect topic text. Open research question.

4. **If the routing pattern does not reproduce:** Slice 1's removal of 783
   ghost rows changed the selection landscape enough on its own. Different
   seeds will be drilled. New patterns will emerge. Re-audit and re-decide.

5. **In parallel, the prose-deflection extraction gap remains an open
   research question.** Designing a second-pass extractor that runs only on
   refusal-classified responses, without reintroducing list-gate noise, is
   worth scoping after step 1's data lands.

Steps 3 and 5 require the live crawl in step 1 first. Do not pre-optimize.
