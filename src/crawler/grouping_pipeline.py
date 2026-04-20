"""Grouping pipeline: group and deduplicate topics via LLM.

The grouping pipeline batches raw topics alongside known cluster heads to an LLM,
which returns structured JSON with group assignments, head/duplicate labels,
summaries, and cluster sizes. This cuts the number of refusal-check calls by
~25× because only novel group representatives need checking.
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Tuple

from src.crawler.config import CrawlerConfig
from src.crawler.topic_queue import Topic
from src.generation_utils import batch_generate
from src.exceptions import APITimeoutError
from src.provider_config import get_provider_client_kwargs
from src.response_formatting_utils import remove_thinking_context


GROUPING_SYSTEM_PROMPT = """\
You are a topic-deduplication assistant for AI safety research. Your job is to:
1. Merge topics that are different phrasings of the SAME specific topic.
2. Identify whether each NEW topic matches a KNOWN head (is_head=false in that case).
3. Filter out preamble, garbage, or non-topic text.

Rules:
- ONLY group topics that are rewordings of the same specific topic. "Taiwan independence" and "Taiwan sovereignty disputes" ARE the same topic rephrased. "Abortion" and "Controversial sensitive topics" are DIFFERENT topics — do NOT group them.
- Do NOT create abstract categories. Every group should be a specific topic, not a bucket like "political issues" or "sensitive social topics".
- Each group gets a short summary label (2-8 words) that names the specific topic, and a numeric cluster index.
- The first topic in each group is the "head"; all others have is_head=false and reference the head's cluster_idx.
- Preamble, meta-commentary, or garbage that is not a topic should be excluded (put their original indices in "skipped").
- When in doubt, keep topics separate. It is better to have two entries for the same topic than to lose a distinct topic by over-merging.

Respond ONLY with valid JSON matching the schema below. No other text.
"""

GROUPING_USER_PROMPT_TEMPLATE = """\
## KNOWN HEADS (existing cluster heads from prior steps)
{known_heads_section}

## NEW TOPICS (to be grouped and deduplicated)
{new_topics_section}

Respond with a JSON object with this exact structure:
{{
  "groups": [
    {{
      "cluster_idx": <int, starting from 0>,
      "summary": "<2-8 word label for the group>",
      "member_indices": [<indices into NEW TOPICS list>],
      "is_head": true
    }}
  ],
  "duplicates": [
    {{
      "original_index": <index into NEW TOPICS>,
      "cluster_idx": <int, matching a group's cluster_idx>,
      "is_head": false,
      "summary": "<same as group summary>"
    }}
  ],
  "skipped": [<indices into NEW TOPICS of items that are not real topics>]
}}

If a NEW topic matches a KNOWN HEAD, put it in "duplicates" with that known head's cluster_idx (use the negative convention: cluster_idx = -(known_head_idx+1) so they are distinguishable from new clusters).
If all NEW topics are duplicates of known heads or skipped, "groups" may be empty."""


def _build_known_heads_section(known_heads: List[Topic]) -> str:
    """Format known heads for the prompt."""
    if not known_heads:
        return "(none)"
    lines = []
    for i, h in enumerate(known_heads):
        label = h.summary or h.shortened or h.raw
        lines.append(f"[{i}] {label}")
    return "\n".join(lines)


def _build_new_topics_section(topics: List[str]) -> str:
    """Format new topic strings for the prompt."""
    lines = []
    for i, t in enumerate(topics):
        lines.append(f"[{i}] {t}")
    return "\n".join(lines)


def _parse_grouping_response(
    raw_response: str,
    topics: List[Topic],
    known_heads: List[Topic],
    verbose: bool = False,
) -> List[Topic]:
    """Parse the LLM's JSON response and annotate topic objects."""
    # Strip markdown fences if present
    text = raw_response.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("{"):
                text = candidate
                break
        # Remove ```json prefix
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        if verbose:
            print(f"[grouping_pipeline] Failed to parse grouping JSON: {raw_response[:200]}")
        # Fallback: treat every topic as a new head
        for i, topic in enumerate(topics):
            topic.is_head = True
            topic.cluster_idx = len(known_heads) + i
            topic.summary = topic.shortened or topic.raw
            topic.cluster_member_count = 1
        return topics

    groups = data.get("groups", [])
    duplicates = data.get("duplicates", [])
    skipped_indices = set(data.get("skipped", []))

    # The LLM numbers new groups locally within the batch (0, 1, 2, ...), but
    # TopicQueue.append_to_cluster uses cluster_idx as a direct list index into
    # cluster_topics. So new-group cluster_idx must be allocated globally,
    # offset by len(known_heads) — otherwise batch N's groups collide with
    # batch 0's known_heads, and non-head members land in the wrong cluster.
    base_cidx = len(known_heads)
    local_to_global: Dict[int, int] = {}
    # S0i sub-fix 1: skip groups with empty/missing member_indices entirely.
    # Empty groups must NOT advance the global counter — they never mint a
    # cluster slot in TopicQueue (no is_head=True topic), so any later
    # reference to that cidx would be invalid.
    global_counter = 0
    for i, group in enumerate(groups):
        members = group.get("member_indices") or []
        if not members:
            if verbose:
                print(
                    f"[grouping_pipeline] skipping empty group at position {i} "
                    f"(no member_indices) — not allocating cluster_idx"
                )
            continue
        llm_cidx = group.get("cluster_idx", i)
        local_to_global[llm_cidx] = base_cidx + global_counter
        global_counter += 1

    # Build a mapping from new-topic index to its assignment
    assignment: Dict[int, dict] = {}  # index -> {cluster_idx, is_head, summary}

    # Map global cluster_idx to member count
    cluster_sizes: Dict[int, int] = {}

    for i, group in enumerate(groups):
        members = group.get("member_indices") or []
        if not members:
            continue  # already skipped above
        llm_cidx = group.get("cluster_idx", i)
        global_cidx = local_to_global[llm_cidx]
        summary = group.get("summary", "")
        cluster_sizes[global_cidx] = len(members)
        for member_idx in members:
            assignment[member_idx] = {
                "cluster_idx": global_cidx,
                "is_head": member_idx == members[0],
                "summary": summary,
            }

    # Set of valid resolvable cluster_idx values (known heads + new groups).
    # Used to detect out-of-range positive cidx values from the LLM.
    known_head_cidx_set = {h.cluster_idx for h in known_heads}
    valid_global_cidx_set = set(local_to_global.values()) | known_head_cidx_set

    for dup in duplicates:
        orig_idx = dup.get("original_index")
        cidx = dup.get("cluster_idx", 0)
        summary = dup.get("summary", "")
        # Resolve negative indices from Kimi's convention to actual known-head cluster_idx
        # Kimi uses cluster_idx = -(known_head_idx+1) to indicate a match to a known head
        resolved = False
        if cidx < 0:
            known_head_idx = abs(cidx) - 1
            if known_head_idx < len(known_heads):
                cidx = known_heads[known_head_idx].cluster_idx
                resolved = True
            else:
                # Negative index out of range — cannot resolve; downgrade to new head below
                if verbose:
                    print(
                        f"[grouping_pipeline] unresolvable negative cluster_idx={cidx} "
                        f"(known_heads has {len(known_heads)} entries); downgrading to new head"
                    )
        elif cidx in local_to_global:
            # Positive LLM-local cluster_idx points at a new group in this batch
            # — remap to the global cluster_idx we allocated above.
            cidx = local_to_global[cidx]
            resolved = True
        elif cidx in valid_global_cidx_set:
            # S0i sub-fix 2: check valid_global_cidx_set (known heads + newly-
            # allocated batch clusters) rather than known_head_cidx_set alone.
            # A duplicate that directly references a newly-created batch cluster
            # by its global cidx is legitimately valid and should not be downgraded.
            resolved = True
        else:
            # Positive cidx not in any known mapping — out of range; downgrade to new head.
            if verbose:
                print(
                    f"[grouping_pipeline] unresolvable positive cluster_idx={cidx} "
                    f"(not in local_to_global or known_heads); downgrading to new head"
                )

        if resolved:
            assignment[orig_idx] = {
                "cluster_idx": cidx,
                "is_head": False,
                "summary": summary,
            }
            cluster_sizes[cidx] = cluster_sizes.get(cidx, 0) + 1
        # else: not added to assignment — falls through to unassigned new-head path below

    # Annotate topics
    max_assigned = max(
        (a["cluster_idx"] for a in assignment.values() if a["cluster_idx"] >= 0),
        default=-1,
    )
    next_new_cluster = max(len(known_heads), max_assigned + 1)
    for i, topic in enumerate(topics):
        if i in skipped_indices:
            # Mark as filtered — will be dropped downstream
            topic.summary = None
            topic.is_head = False
            topic.cluster_idx = -1
            topic.cluster_member_count = 0
            continue

        if i in assignment:
            a = assignment[i]
            topic.cluster_idx = a["cluster_idx"]
            topic.is_head = a["is_head"]
            topic.summary = a["summary"] or topic.shortened or topic.raw
            topic.cluster_member_count = cluster_sizes.get(a["cluster_idx"], 1)
        else:
            # Unassigned by LLM (or unresolvable cluster_idx) — treat as new head
            topic.is_head = True
            topic.cluster_idx = next_new_cluster
            topic.summary = topic.shortened or topic.raw
            topic.cluster_member_count = 1
            next_new_cluster += 1

    # S0i sub-fix 4: parser-level post-condition.
    # Every non-head topic must reference a cluster_idx that belongs to either
    # (a) a head in this very batch, or (b) a known_head from a prior batch.
    # Any violation means a cluster_idx escaped the resolution/fallback paths
    # above — downgrade the offending topic to a new head so it never reaches
    # TopicQueue.append_to_cluster with an invalid cluster_idx.
    import logging as _logging
    valid_head_cidx_in_batch = {
        t.cluster_idx
        for t in topics
        if t.is_head and t.cluster_idx is not None and t.cluster_idx >= 0
    }
    all_valid_cidx = valid_head_cidx_in_batch | known_head_cidx_set
    for topic in topics:
        if topic.is_head:
            continue
        if topic.cluster_idx == -1:
            continue  # skipped sentinel — intentional
        if topic.cluster_idx is None or topic.cluster_idx not in all_valid_cidx:
            _logging.warning(
                "[grouping_pipeline] post-condition: downgraded non-head topic with "
                "unresolved cluster_idx=%r to new head (raw=%r)",
                topic.cluster_idx,
                topic.raw,
            )
            topic.is_head = True
            topic.cluster_idx = next_new_cluster
            topic.summary = topic.summary or topic.shortened or topic.raw
            topic.cluster_member_count = 1
            next_new_cluster += 1

    return topics


def _preserve_batch_as_heads(batch: List[Topic], known_heads: List[Topic]) -> None:
    """Mark each topic in a timed-out batch as a new head with a fallback summary.

    This is the opt-in preservation path for S0a
    (config.crawler.grouping_timeout_preserves_batch=True). Topics get:
    - is_head=True
    - cluster_idx allocated contiguously starting from len(known_heads)
    - summary = shortened or raw (non-None so downstream filter passes)
    - cluster_member_count = 1

    Mutates topics in-place.
    """
    base_cidx = len(known_heads)
    for i, topic in enumerate(batch):
        topic.is_head = True
        topic.cluster_idx = base_cidx + i
        topic.summary = topic.shortened or topic.raw
        topic.cluster_member_count = 1


def summarize_group_dedup(
    topics: List[Topic],
    known_heads: List[Topic],
    config: CrawlerConfig,
    local_model=None,
    local_tokenizer=None,
    verbose: bool = False,
) -> List[Topic]:
    """Batch raw topics + known heads to the grouping LLM.

    Uses the configured summarization_model (or local model) to group
    semantically equivalent topics, deduplicate against known heads,
    and annotate each Topic with summary, is_head, cluster_idx,
    and cluster_member_count.

    Args:
        topics: Newly extracted topics (must have .raw and .shortened populated)
        known_heads: Existing cluster heads from the queue
        config: CrawlerConfig (used for model routing and batch size)
        local_model: Local vLLM model (used when summarization_model == "local")
        local_tokenizer: Corresponding tokenizer
        verbose: Print debug information

    Returns:
        The same topic list with grouping annotations applied.
        Topics identified as garbage/preamble have summary=None and should be
        filtered out downstream.
    """
    if not topics:
        return topics

    batch_size = config.crawler.semantic_group_batch_size
    all_topics = topics

    # Process in batches
    for batch_start in range(0, len(all_topics), batch_size):
        batch = all_topics[batch_start : batch_start + batch_size]
        topic_strs = [t.raw for t in batch]

        known_heads_section = _build_known_heads_section(known_heads)
        new_topics_section = _build_new_topics_section(topic_strs)

        user_prompt = GROUPING_USER_PROMPT_TEMPLATE.format(
            known_heads_section=known_heads_section,
            new_topics_section=new_topics_section,
        )

        messages = [
            [
                {"role": "system", "content": GROUPING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        ]

        # Route to the appropriate model
        if config.model.summarization_model == "local":
            try:
                responses, _ = batch_generate(
                    model=local_model,
                    tokenizer=local_tokenizer,
                    messages=messages,
                    max_new_tokens=4096,
                    temperature=0.0,
                    verbose=verbose,
                    default_provider=config.model.default_provider,
                    provider_url_overrides=config.model.provider_urls,
                    provider_concurrency_limits=config.model.provider_max_concurrency,
                    prefer_nitro=config.model.prefer_nitro,
                )
            except APITimeoutError:
                if config.crawler.grouping_timeout_preserves_batch:
                    print(
                        f"Warning: grouping timed out (local) for batch of "
                        f"{len(batch)} topics; preserving as new heads (fallback summaries)"
                    )
                    _preserve_batch_as_heads(batch, known_heads)
                    new_heads = [t for t in batch if t.is_head and t.summary is not None]
                    known_heads = known_heads + new_heads
                    continue
                raise
        else:
            from src.openrouter_utils import async_query_openrouter

            resolved_model_id, client_kwargs = get_provider_client_kwargs(
                config.model.summarization_model,
                config.model.default_provider,
                config.model.provider_urls,
            )

            # S4b: use universal_backup_model if set, else no fallback
            grouping_fallbacks = (
                [config.model.universal_backup_model]
                if config.model.universal_backup_model
                else []
            )

            async def _query():
                return await async_query_openrouter(
                    model_name=resolved_model_id,
                    prompt=user_prompt,
                    system_prompt=GROUPING_SYSTEM_PROMPT,
                    max_tokens=4096,
                    temperature=0.0,
                    verbose=verbose,
                    client_kwargs=client_kwargs,
                    extra_body={"reasoning": {"effort": "none"}},
                    fallback_models=grouping_fallbacks,
                    default_provider=config.model.default_provider,
                    provider_url_overrides=config.model.provider_urls,
                    prefer_nitro=config.model.prefer_nitro,
                )

            try:
                responses = [asyncio.run(_query())]
            except APITimeoutError:
                if config.crawler.grouping_timeout_preserves_batch:
                    print(
                        f"Warning: grouping timed out (API) for batch of "
                        f"{len(batch)} topics; preserving as new heads (fallback summaries)"
                    )
                    _preserve_batch_as_heads(batch, known_heads)
                    new_heads = [t for t in batch if t.is_head and t.summary is not None]
                    known_heads = known_heads + new_heads
                    continue
                raise

        # Strip thinking blocks from reasoning models
        responses = remove_thinking_context(responses)

        for resp in responses:
            _parse_grouping_response(resp, batch, known_heads, verbose=verbose)

        # After each batch, newly-minted heads become known for subsequent batches
        new_heads = [t for t in batch if t.is_head and t.summary is not None]
        known_heads = known_heads + new_heads

    return all_topics
