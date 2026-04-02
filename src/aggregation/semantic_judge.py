import json
from typing import Dict, List, Optional, Tuple

from src.aggregation.aggregator import _parse_json_from_response
from src.generation_utils import batch_generate


def _build_judge_prompt(input_term: str, reference_terms: List[str]) -> str:
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(reference_terms))
    return (
        "You are a semantic similarity judge. Given an INPUT term and a list of "
        "REFERENCE terms, determine whether the input term is semantically equivalent "
        "to or semantically contained within any single reference term.\n\n"
        '"Semantically equivalent" means the terms refer to the same concept, even if '
        'worded differently (e.g., "obtaining weapons" and "acquiring firearms").\n'
        '"Semantically contained" means the input term\'s meaning is a subset or '
        'specific instance of a reference term (e.g., "handgun acquisition" is '
        'contained within "obtaining weapons").\n\n'
        f'INPUT: "{input_term}"\n\n'
        f"REFERENCE TERMS:\n{numbered}\n\n"
        "Respond with ONLY a JSON object in this exact format (no markdown, no explanation):\n"
        '{"match": true, "matched_term": "exact text of the matching reference term"}\n'
        "or\n"
        '{"match": false, "matched_term": null}'
    )


def _fuzzy_find_reference(
    candidate: Optional[str], reference_terms: List[str]
) -> Optional[str]:
    if not candidate:
        return None
    candidate_norm = candidate.strip().lower()
    for ref in reference_terms:
        if ref.strip().lower() == candidate_norm:
            return ref
    return None


def _parse_judge_response(
    response: str,
    input_term: str,
    reference_terms: List[str],
) -> Tuple[bool, Tuple[str, Optional[str]]]:
    try:
        parsed = _parse_json_from_response(response)
        is_match = bool(parsed.get("match", False))
        matched_term = parsed.get("matched_term", None)

        if is_match:
            if not matched_term or matched_term not in reference_terms:
                matched_term = _fuzzy_find_reference(matched_term, reference_terms)
                if matched_term is None:
                    is_match = False
        else:
            matched_term = None

        return (is_match, (input_term, matched_term))
    except (ValueError, json.JSONDecodeError):
        return (False, (input_term, None))


def judge_semantic_containment(
    input_terms: List[str],
    reference_terms: List[str],
    model,
    tokenizer,
    max_tokens: int = 256,
    default_provider: str = "openrouter",
    provider_url_overrides: Optional[Dict[str, str]] = None,
    verbose: bool = False,
) -> List[Tuple[bool, Tuple[str, Optional[str]]]]:
    """Check whether each input term is semantically equivalent to or contained
    in any term of the reference set, using an LLM judge.

    Returns a list of (is_match, (input_term, matched_reference_or_None)) tuples.
    """
    if not input_terms:
        return []
    if not reference_terms:
        return [(False, (t, None)) for t in input_terms]

    prompts = [_build_judge_prompt(t, reference_terms) for t in input_terms]
    messages = [[{"role": "user", "content": p}] for p in prompts]

    answers, _ = batch_generate(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_tokens,
        temperature=0.0,
        verbose=verbose,
        default_provider=default_provider,
        provider_url_overrides=provider_url_overrides,
    )

    return [
        _parse_judge_response(ans, term, reference_terms)
        for ans, term in zip(answers, input_terms)
    ]
