"""Shared transcript classification helpers.

Extracted so that both scripts/call_breakdown.py and
scripts/bench_cache_dedup.py can import the same classify() function
without duplicating logic.  Callers use sys.path.insert to reach this
file since scripts/ has no __init__.py.
"""

import re


def first_user(inputs):
    if not isinstance(inputs, list) or not inputs:
        return ""
    seq = inputs[0] if isinstance(inputs[0], list) else inputs
    for m in seq:
        if isinstance(m, dict) and m.get("role") == "user":
            return m.get("content") or ""
    return ""


def first_sys(inputs):
    if not isinstance(inputs, list) or not inputs:
        return ""
    seq = inputs[0] if isinstance(inputs[0], list) else inputs
    for m in seq:
        if isinstance(m, dict) and m.get("role") == "system":
            return m.get("content") or ""
    return ""


def _is_summarize_call(inputs) -> bool:
    """Return True if this record's prompts match the legacy summarize shape.

    Matches calls from async_summarize_single_topic / async_batch_summarize_topics:
      - user prompt begins with "Extract concise topic labels from the phrase"
        (TOPIC_SUMMARIZATION_PROMPT from src/crawler/config.py), OR
      - system prompt contains "topic labels from phrases"
        (response_formatting_utils.py API-summarize path).

    Must NOT match group calls (system contains "deduplication") or extract
    calls (system contains "You extract structured data from text").
    """
    sys_msg = first_sys(inputs)
    user_msg = first_user(inputs)
    # Positive signals — specific to summarize paths.
    user_match = bool(re.search(
        r"Extract concise topic labels from the phrase", user_msg, re.I
    ))
    sys_match = bool(re.search(r"topic labels from phrases", sys_msg, re.I))
    return user_match or sys_match


def classify(rec):
    """Return ('role', 'model-family') for a transcript record.

    Heuristic:
    - qwen    -> translate
    - summarize shape (any model, see _is_summarize_call) -> summarize
    - kimi / moonshot + 'deduplication' in system -> group
    - kimi / moonshot + 'extract' in system       -> extract
    - deepseek + (mt >= 3000 or batch >= 10)      -> generate
    - deepseek otherwise                          -> refusal-query
    - gemma / refusal + 'AI safety judge' in user -> judge
    - gemma / refusal + safety/research terms     -> provoke
    - everything else                             -> other

    The summarize branch is checked BEFORE the kimi/moonshot extract branch
    because the legacy summarize system prompt also contains the word "extract",
    which would otherwise cause kimi-routed summarize calls to be misclassified
    as extract.
    """
    model = (rec.get("model") or "").lower()
    mt = rec.get("max_tokens") or 0
    bs = rec.get("batch_size") or 0
    inputs = rec.get("inputs")
    if "qwen" in model:
        return ("translate", "qwen")
    # Summarize check is model-agnostic and must precede the kimi/extract branch.
    if _is_summarize_call(inputs):
        return ("summarize", model or "?")
    if "kimi" in model or "moonshot" in model:
        sys_msg = first_sys(inputs)
        if re.search(r"deduplication", sys_msg, re.I):
            return ("group", "kimi")
        if re.search(r"extract", sys_msg, re.I):
            return ("extract", "kimi")
        return ("other", "kimi")
    if "deepseek" in model:
        if mt >= 3000 or bs >= 10:
            return ("generate", "target")
        return ("refusal-query", "target")
    if "gemma" in model or "refusal" in model:
        user = first_user(inputs)
        if re.search(r"AI safety judge", user, re.I):
            return ("judge", "gemma")
        if re.search(r"AI safety research|敏感话题|assisting with AI safety", user, re.I):
            return ("provoke", "gemma")
        return ("other", "gemma")
    return ("other", "?")
