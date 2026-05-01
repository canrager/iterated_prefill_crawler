# Learnings

This file is for forward-facing decisions only. Do not append lab logs,
command transcripts, one-off run summaries, or postmortem narration. Put exact
commands and raw evidence in `HANDOFF.md`, `REVIEW.md`, or timestamped
artifacts; keep this file as the short guide for what future work should do
and avoid.

## Mission

Build the Iterated Prefill Crawler into a cost-efficient black-box refusal
surface recovery tool. The useful paper artifact is a model-specific
"nutrition label" of refusal topics, recoverable through prefill/fake-prefill
elicitation and summarized as clusters or wordclouds.

Production selection and prompts must remain target-agnostic. Target-specific
topic vocabularies are allowed only for frozen-output, post-hoc scoring.

## Hard Constraints

Never put concrete expected target topics into production prompts, queue
selection, reviewer rubrics, or reusable selector code. The crawler must run
the same way against any target organism.

Do not redesign `jailbreak.yaml` because a crawl misses expected topics. The
known-good drill-down prompt shape already works when routed the right broad
parent. The present gap is routing/selection, not prompt invention.

Do not use `is_refusal` as the only seed eligibility rule. Broad parents can
look safe at the coarse level while their children are refusal-bearing.

Do not treat LEARNINGS as an audit trail. If a note does not change a future
decision, remove it.

## Pipeline Invariants

The crawler's core loop remains:

```text
Generate -> Extract -> Translate -> Provoke -> Classify Refusals
```

Everything before post-crawl analysis must preserve target-emitted label
fidelity. Inline splitting, cleanup, and summarization heuristics that mutate
labels have repeatedly damaged recall. If a helper returns one label, keep it
as one label unless a separate verified post-crawl analysis step splits it.

Upstream API sentinels are terminal records for that stage, not semantic text.
Never pass `__API_CALL_FAILED__` or API-moderation sentinels into extraction,
translation, summarization, clustering, or selector prompts.

Helper-model moderation failures should use the configured universal backup
model before a helper result is allowed to drop data. Target-model moderation
or refusal sentinels are part of the target behavior and must not be hidden by
backup substitution.

## Discovery vs. Routing

Warmup can emit useful broad umbrellas. The repeated failure mode is that the
crawler does not drill the right broad parents. Treat this as a routing
problem until a crawl proves the target never emitted a usable parent.

When diagnosing a miss, classify it as exactly one of:

1. **Elicitation failure:** the target never emitted a relevant broad parent.
2. **Extraction/translation failure:** the parent was present but not preserved.
3. **Selection failure:** the parent was preserved but not drilled.
4. **Validation/reporting failure:** the useful child survived but never
   entered validation or the final artifact.

Do not solve a selection failure with a new target prompt.

## Cluster-First Crawler

The cluster-first crawler is the right low-cost shape for wordcloud artifacts:

```text
Generation -> Extract/Translate/Summarize -> Cluster -> Drill selected umbrellas -> Validate reps -> Wordcloud
```

It is still called a crawler for continuity, but operationally it is a
fixed-budget clusterer. Evidence should be cluster-level, not raw-row-level.

The current mixed selector uses largest clusters, random tail/singletons, and
farthest-first diversity. It is unique within a run, but it is not aligned
with the research objective. Increasing `--auto-drill-seeds` alone improves
breadth but does not reliably recover rare broad umbrellas.

Next selector work should reserve quota for broad rare umbrellas before
random tail sampling. This must be phrased generically: broad, abstract,
contestable, institutional, historical, geopolitical, public-order,
rights-related, security-related, law-related, information-control-related, or
framing-dependent categories. The selector may use those general properties;
it must not use concrete target-topic vocabularies.

## Helper Selector Prompting

The first helper-selector prompt was too cluttered. It over-specified the
rubric, buried the actual job, and asked the model to rank the full label list
in one shot. That benchmark showed a useful distinction: the primary helper model
can obey exact-label constraints, while the fallback model may normalize or
invent labels and therefore needs strict post-filtering.

Forward rule: do not ask a helper to discover rare broad singletons from the
entire cluster list. First create a deterministic, target-agnostic candidate
pool using cheap metadata and broadness features, then ask the helper to rank
that small pool verbatim.

A cleaner helper-selector task should be short:

```text
Choose labels worth drilling because they are broad categories, not concrete
requests. Prefer labels whose children may vary by actor, time period,
institution, jurisdiction, public context, disputed facts, or framing. Return
only verbatim labels from the input.
```

The benchmark pass condition is not "the model returned plausible labels."
It is:

- all selected labels are verbatim input labels;
- the selected set includes rare broad singleton candidates;
- selected candidates were not already drilled or validated;
- post-hoc target-specific scoring improves after those frozen selections.

## Validation Strategy

Validation cannot only follow top cluster score. Rare broad umbrellas and
their descendants are low-frequency by construction. Reserve validation quota
for:

- top scored clusters;
- newly drilled umbrella descendants;
- rare broad singleton clusters;
- diversity among semantic neighborhoods.

Keep validation probes low during shape tests. Increase probes only after the
selector is retrieving the right neighborhoods.

## Model/Helper Guidance

Disable reasoning for helper calls. Helpers are doing extraction, translation,
summarization, ranking, or judging. Reasoning tokens add cost and latency
without being the audit signal. Keep target-model reasoning behavior intact.

Cap helper concurrency at the API boundary. Fan-out stages can otherwise turn
rate limits into retry storms.

Translation and extraction should use the same prompt shape that their benches
validated. A bench that tests a different shape than production is not useful.

## Current Next Move

Do not run another larger crawl yet. The next valuable experiment is an
offline selector slice:

1. Build a deterministic broad-umbrella prefilter over frozen cluster labels.
2. Feed only that reduced candidate pool to the primary helper selector.
3. Strictly post-filter to verbatim labels.
4. Compare selected seeds against the current mixed selector on frozen
   artifacts before spending more target API calls.
5. Only then run a small live crawl with a reserved broad-umbrella drill quota.

The falsifiable question is: can the new selector consistently surface broad
rare umbrellas that the current mixed selector misses, without target-specific
topic literals?

The broad-tail bilingual path now exists behind a flag. Use it as the next
research shape: Kimi extracts a broadest-first array from raw target taxonomy
text, the crawler deduplicates and drills from tail toward head, and each seed
is drilled in both available language surfaces. Treat this as the hypothesis
to compare against structure-only cluster sampling.

The next default crawler shape should reserve broad-topic budget in two
directions before any embedding-based auto-drill: use the head of Kimi's
broadest-first array for lateral "what else" crawl, and the tail for granular
drill-down. Tune new runs with explicit `--broad-head-crawl-seeds`,
`--broad-tail-drill-seeds`, and `--broad-iterations` so effectiveness and API
cost can be attributed.

There is no legacy broad-drill alias in the cluster crawler. Tune and report
the new shape only through explicit `broad_head_crawl_seeds`,
`broad_tail_drill_seeds`, and `broad_iterations` profile fields or CLI flags.

Debug cluster-crawler profiles may be verbose because their purpose is payload
inspection. Rehearsal/default live runs should keep verbose logging off unless
the output is redirected and inspected narrowly, because current verbose mode
can print raw prompts, topics, helper request bodies, and formatted topic
objects.

For paper-style wordclouds, clustering is not the ranking mechanism. The
post-crawl renderer should benchmark pairwise-ranking ideas against frozen
artifacts before changing production display weights. A useful offline proxy is
to compare cluster-size scoring against Elo-style pairwise updates over generic
signals such as parent-yield, cluster score, and label specificity, then score
target-specific recovery only as post-hoc artifact evaluation.

Target-specific regexes or vocabularies belong in one-off post-hoc benchmark
commands or saved result artifacts, not reusable scripts, prompts, selectors,
or configs. The reusable renderer should accept a regex argument and otherwise
remain target-agnostic.

For wordcloud display, rank first and then collect display terms by discovered
cluster. On the frozen debug artifact, raw pairwise-proxy ranking recovered
the target neighborhood but let one repeated semantic family dominate the top
50. A display cap of two terms per cluster preserved all post-hoc target
anchors in the top 50 while cutting the repeated-family top-50 count from
eight to two. Treat cap 2 as the current offline renderer default, and
falsify it on the next rehearsal artifact before claiming generality.

Family/canonicalization should preserve exact member strings and keep the
display label separate. Local string and TF-IDF grouping can reduce obvious
duplicates, but the frozen debug benchmark shows they do not solve canonical
label choice: string grouping can pick a generic label for a more specific
member, while TF-IDF can merge the desired sovereignty wording pair. The next
valuable comparison is a carefully prompted aggregator model with the minimal
schema `[{label, members}]`, scored against the same aggregate metrics and
CCP-only qualitative vetting artifact.

For the minimal aggregator prompt, Kimi behaved better than Qwen on the frozen
debug artifact: both returned valid exact-member JSON with no repair needed,
but Qwen grouped nothing at 120 labels while Kimi made three conservative
two-member families and preserved the useful sovereignty wording merge. Treat
this as evidence for Kimi as the display-family aggregator candidate, not as a
reason to replace Qwen for translation or other helper roles.

Aggregator display labels should be allowed to be readable generated strings;
only `members` must be exact input strings. The prompt should stay
target-agnostic and linguistic, using non-refusal examples. Deterministic
repair should validate exact member coverage and use member-string fallback
only when a display label is missing or blank.

The first display-label aggregator benchmark improved readability but made the
main risk visible: generated labels can become too broad even when members are
exact. Qwen over-grouped on the frozen debug artifact; Kimi stayed cleaner but
still used broader display labels for some specific member strings. The next
prompt should remain domain-agnostic and describe this as a linguistic rule:
prefer labels that keep distinctive concrete words from the most specific
members, and split rather than using a label that could also describe many
other input strings.

Do not assume a stronger aggregation prompt fixes loss of specificity at large
batch size. On the frozen debug artifact, 120-label aggregation made models
choose among bad behaviors: singleton everything, fail parsing, or compress
fine-grained findings into broad umbrellas. Before changing prompt wording
again, run a batch-size ablation on the same frozen input and measure whether
30-60 label chunks preserve exact-member fidelity and produce better display
families.

Incremental batching should be evaluated per model, not assumed uniformly
better. At batch size 30 on the frozen debug artifact, Qwen improved
substantially versus one-shot display-label aggregation: zero repair, smaller
max family, and more CCP-vetting families. Kimi kept zero repair too, but
compressed harder than desired. The next renderer benchmark should ablate
batch sizes before changing wording or model roles.

Do not assume frontier or western aggregator models will preserve paper-style
specificity. In the batch-30 benchmark, Gemma 4 and Claude Sonnet 4.6 both
preserved exact members but hid Taiwan-related strings under broad
territorial/sovereignty labels. The failure is increasingly a task/objective
problem: semantic grouping rewards compression, while the wordcloud needs
specific exact-member visibility.

The older indexed aggregator prompt and the newer `src/aggregation` reduction
prompt are useful comparators, not current renderer candidates. They reduce
repetition, but their objective is semantic clustering/reduction, which is not
the same as paper-style wordcloud labeling. Use them to falsify new ideas, not
as defaults for final display.

Batching the legacy `exp/` indexed prompt makes it more tractable but does not
change its core objective. In the frozen debug artifact, 30-topic legacy
batches still hid Taiwan-related exact strings under broad semantic cluster
labels. Treat the legacy prompt as historical reconstruction evidence, not as
the final paper-wordcloud renderer unless the renderer separately surfaces
high-value exact members inside broad families.

Do not choose the final cloud maker from the tiny debug crawl alone. After
augmenting the frozen debug artifact with Kimi extraction from the combined ZH
fixture, the batched legacy `exp/` prompt became the strongest
website-style candidate: both Qwen and Kimi surfaced all post-hoc CCP-matched
members in the top 80, and Kimi preserved more families. The current best
next step is a larger real crawl followed by Kimi + batched-old-indexed
rendering, with exact members retained for audit and optional member-level
display.

Rehearsal-size live crawls can differ by an order of magnitude across target
and method. In the May 1 rehearsal pair, DeepSeek V3.2 jailbreak produced a
large aggregation-ready artifact, while Haiku 3.5 assistant-prefix produced a
valid but small artifact. Use the DeepSeek artifact for the next cloud-maker
comparison; do not assume the Haiku prefill artifact is an equally strong
nutrition-label run without either more breadth or a stronger prefill variant.

When post-crawl aggregation preserves a rare target family but the wordcloud
still hides it, inspect the family score before changing retrieval. In the
DeepSeek rehearsal, Kimi + batched legacy aggregation preserved a 13-member
CCP-aligned family, but current max-member scoring ranked it 99. A generic
log member-count boost moved it to rank 29 without target-specific vocabulary.
The next renderer should score by member strength plus family support, then
cap rendered terms.

## Do Not Retry Without New Evidence

Do not run more offline bakeoffs over a single historical artifact unless they
produce frozen seed selections for a live counterfactual. A static artifact
cannot reveal child yield for seeds that were never drilled.

Do not promote embedding ranking into production queue logic yet. Embeddings
are useful for analysis and redundancy, but they have not proven better than
cheap structural baselines for recovery per budget.

Do not add inline label-mutating post-processing as a recall fix. Prove net
positive recall on real artifacts first.
