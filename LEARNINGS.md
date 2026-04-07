# Learnings

Keep only lessons that change how the next update is planned, built,
reviewed, or validated. Rewrite, merge, and prune continously.

Source control will track versions of this document, so keep it fresh.

## Current learnings

### Manual drilldown coverage does not imply crawl coverage

What failed or was discovered:
Manual seeded drilldown prompts can surface specific subtopics that the full crawl never reaches, even when the broader parent topic was discovered during warmup.

Why it matters:
The crawl only expands topics that are eligible as later seeds on the main crawl path. A broad topic that is safe at its coarse label can still contain narrower blocked children, so limiting later seed selection to refusal-marked parents silently drops that branch. Refusal-check responses are a separate classification path and do not create new queued topics unless they are explicitly re-extracted.

What changes next time:
When debugging coverage gaps, check three things separately: which prompt family ran, which discovered topics were eligible to re-enter the seed pool, and whether the missing detail appeared only inside refusal-check responses. If the goal is deeper decomposition, later seed pools must include discovered head topics, not only head refusal topics.

Where that change applies:
Prompt-building, crawl-debugging, and any evaluation that compares manual seeded probes against full-crawl results.

Provenance or evidence:
DeepSeek crawl artifact from 2026-04-01 missed manual drilldown-style geopolitical leaves because broad discovered topics were not reused as later drilldown seeds unless they were already marked as refusals.
