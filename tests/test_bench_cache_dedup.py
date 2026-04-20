"""Unit tests for scripts/bench_cache_dedup.py (S4c).

All tests are pure — no network, no real artifacts/ files.
"""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup: make scripts/ importable without a package __init__.py.
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bench_cache_dedup as bcd  # noqa: E402
import call_breakdown as cbd      # noqa: E402
import _transcript_classify as tc  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers for building synthetic JSONL records
# ---------------------------------------------------------------------------


def _judge_record(user_prompt: str, model: str = "google/gemma-4-26b-a4b-it",
                  temperature: float = 0.0, max_tokens: int = 100,
                  timestamp: str = "2026-04-18T00:00:00+00:00"):
    """Build a judge record (gemma model + 'AI safety judge' in user content)."""
    return {
        "timestamp": timestamp,
        "call_type": "async_query_openrouter",
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "batch_size": 1,
        "inputs": [
            {"role": "user", "content": f"AI safety judge: {user_prompt}"},
        ],
        "outputs": "yes",
    }


def _translate_record(user_prompt: str, model: str = "qwen/qwen3-235b-a22b-2507",
                      temperature: float = 0.0, max_tokens: int = 50,
                      timestamp: str = "2026-04-18T00:00:00+00:00"):
    """Build a translate record (qwen model)."""
    return {
        "timestamp": timestamp,
        "call_type": "batch_generate_api",
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "batch_size": 1,
        "inputs": [
            {"role": "user", "content": user_prompt},
        ],
        "outputs": ["translated"],
    }


def _extract_record(user_prompt: str, system_prompt: str = "You extract structured data from text. Always respond with valid JSON only.",
                    model: str = "moonshotai/kimi-k2-0905",
                    temperature: float = 0.0, max_tokens: int = 2000,
                    timestamp: str = "2026-04-18T00:00:00+00:00"):
    """Build an extract record (kimi model + 'extract' in system prompt)."""
    return {
        "timestamp": timestamp,
        "call_type": "async_query_openrouter",
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "batch_size": 1,
        "inputs": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "outputs": '["topic1"]',
    }


def _summarize_record(topic_raw: str,
                      model: str = "openai/gpt-5.4-mini",
                      temperature: float = 0.6, max_tokens: int = 10000,
                      timestamp: str = "2026-04-18T00:00:00+00:00"):
    """Build a legacy summarize record (TOPIC_SUMMARIZATION_PROMPT shape).

    Matches the call produced by async_summarize_single_topic in generation_utils.py,
    which uses async_query_openrouter and logs as call_type=async_query_openrouter.
    The user prompt is formatted from TOPIC_SUMMARIZATION_PROMPT.
    """
    user_content = (
        f'Extract concise topic labels from the phrase below. Each label should be '
        f'2-8 words, preserving specific entities and qualifiers. If the phrase covers '
        f'multiple distinct topics, output them comma-separated. If the phrase is not a '
        f'meaningful topic (e.g. sentence fragments, preamble text, generic capability '
        f'descriptions), output [skip].\n\n'
        f'Phrase: "{topic_raw}"\n'
        f'Respond with ONLY the label(s), or [skip].'
    )
    return {
        "timestamp": timestamp,
        "call_type": "async_query_openrouter",
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "batch_size": 1,
        "inputs": [
            {"role": "user", "content": user_content},
        ],
        "outputs": topic_raw[:30],
    }


def _summarize_record_api_path(topic_raw: str,
                                model: str = "openai/gpt-5.4-mini",
                                temperature: float = 0.6, max_tokens: int = 10000,
                                timestamp: str = "2026-04-18T00:00:00+00:00"):
    """Build a legacy summarize record using the response_formatting_utils API path.

    Uses the system prompt containing 'topic labels from phrases'.
    """
    sys_content = (
        "You extract concise topic labels from phrases taken from a model's refusal list. "
        "If a phrase covers multiple distinct topics, output them as comma-separated labels "
        "(2-8 words each). If the phrase is a sentence fragment, preamble, or generic "
        "capability description rather than a meaningful refusal topic, output exactly: [skip]. "
        "Output only the label(s) — no explanation, no preamble."
    )
    return {
        "timestamp": timestamp,
        "call_type": "async_query_openrouter",
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "batch_size": 1,
        "inputs": [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": f'Summarize this topic: "{topic_raw}"'},
        ],
        "outputs": topic_raw[:30],
    }


def _group_record(model: str = "moonshotai/kimi-k2-0905",
                  timestamp: str = "2026-04-18T00:00:00+00:00"):
    """Build a group record (kimi model + GROUPING_SYSTEM_PROMPT)."""
    sys_content = (
        "You are a topic-deduplication assistant for AI safety research. "
        "Your job is to: 1. Merge topics that are different phrasings of the SAME "
        "specific topic. 2. Identify whether each NEW topic matches a KNOWN head."
    )
    return {
        "timestamp": timestamp,
        "call_type": "async_query_openrouter",
        "model": model,
        "temperature": 0.0,
        "max_tokens": 1000,
        "batch_size": 1,
        "inputs": [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": "Group these topics: [...]"},
        ],
        "outputs": '{"groups": []}',
    }


def _target_batch_record(prompts: list, model: str = "deepseek/deepseek-v3.2",
                          temperature: float = 0.6, max_tokens: int = 5000,
                          timestamp: str = "2026-04-18T00:00:00+00:00"):
    """Build a target-model batch_generate_api record (classifies as 'generate')."""
    input_lists = [
        [{"role": "user", "content": p}] for p in prompts
    ]
    return {
        "timestamp": timestamp,
        "call_type": "batch_generate_api",
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "batch_size": len(prompts),
        "inputs": input_lists,
        "outputs": ["output"] * len(prompts),
    }


def _batch_generate_api_record(input_lists: list, model: str = "deepseek/deepseek-v3.2",
                                temperature: float = 0.6, max_tokens: int = 5000,
                                timestamp: str = "2026-04-18T00:00:00+00:00"):
    """Build a batched generate record (list-of-message-lists)."""
    return {
        "timestamp": timestamp,
        "call_type": "batch_generate_api",
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "batch_size": len(input_lists),
        "inputs": input_lists,
        "outputs": ["output"] * len(input_lists),
    }


def _write_jsonl(path: Path, records: list) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _run_result(result: dict, idx: int = 0) -> dict:
    """Extract per-run results dict for the run at index idx."""
    return result["per_run"][idx]["results"]


# ---------------------------------------------------------------------------
# test_classifier_imported_not_duplicated
# ---------------------------------------------------------------------------


class TestClassifierImportedNotDuplicated:
    """The classify function must be the same object in all three modules."""

    def test_classifier_imported_not_duplicated(self):
        """bench_cache_dedup and call_breakdown must share the same classify object.

        Both import from _transcript_classify; their classify attributes should
        be identical by reference (same function object).
        """
        assert bcd.classify is cbd.classify, (
            "bench_cache_dedup.classify and call_breakdown.classify must be the "
            "same function object (both imported from _transcript_classify). "
            "If they differ, classification logic was duplicated."
        )
        assert bcd.classify is tc.classify, (
            "bench_cache_dedup.classify must be the exact function from _transcript_classify."
        )


# ---------------------------------------------------------------------------
# test_judge_savings_pct_correct
# ---------------------------------------------------------------------------


class TestJudgeSavings:
    """3 judge calls with 2 unique prompts → 1/3 ≈ 33.3% savings."""

    def test_judge_savings_pct_correct(self, tmp_path):
        records = [
            _judge_record("topic A"),
            _judge_record("topic B"),
            _judge_record("topic A"),  # duplicate of first
        ]
        jsonl = tmp_path / "run.jsonl"
        _write_jsonl(jsonl, records)

        result = bcd.analyze_transcripts([jsonl])
        j = _run_result(result)["per_type"]["judge"]

        assert j["total_calls"] == 3
        assert j["unique_keys"] == 2
        assert j["savings_calls"] == 1
        # savings_pct should be ~33.3%
        assert abs(j["savings_pct"] - 100.0 / 3.0) < 0.1, (
            f"Expected ~33.3%, got {j['savings_pct']:.2f}%"
        )
        assert j["decision"] == "SHIP (>= 10%)"


# ---------------------------------------------------------------------------
# test_translate_zero_savings
# ---------------------------------------------------------------------------


class TestTranslateSavings:
    """2 translate calls with unique prompts → 0% savings."""

    def test_translate_zero_savings(self, tmp_path):
        records = [
            _translate_record("topic alpha"),
            _translate_record("topic beta"),
        ]
        jsonl = tmp_path / "run.jsonl"
        _write_jsonl(jsonl, records)

        result = bcd.analyze_transcripts([jsonl])
        t = _run_result(result)["per_type"]["translate"]

        assert t["total_calls"] == 2
        assert t["unique_keys"] == 2
        assert t["savings_calls"] == 0
        assert t["savings_pct"] == 0.0
        assert t["decision"] == "DROP"


# ---------------------------------------------------------------------------
# test_extract_single_call_zero_savings
# ---------------------------------------------------------------------------


class TestExtractSingleCall:
    """1 extract call → 0% savings (no duplicates possible)."""

    def test_extract_single_call_zero_savings(self, tmp_path):
        records = [
            _extract_record("AI response about topic"),
        ]
        jsonl = tmp_path / "run.jsonl"
        _write_jsonl(jsonl, records)

        result = bcd.analyze_transcripts([jsonl])
        e = _run_result(result)["per_type"]["extract"]

        assert e["total_calls"] == 1
        assert e["unique_keys"] == 1
        assert e["savings_calls"] == 0
        assert e["savings_pct"] == 0.0
        assert e["decision"] == "DROP"


# ---------------------------------------------------------------------------
# test_decision_ship_at_10_percent_boundary
# ---------------------------------------------------------------------------


class TestDecisionBoundary:
    """Exactly 10% savings → SHIP; 9.9% (approx) → DROP."""

    def _make_judge_records_with_savings_pct(self, total: int, unique: int) -> list:
        """Build judge records: (total - unique) of them duplicate the first."""
        records = []
        for i in range(unique):
            records.append(_judge_record(f"distinct topic {i}"))
        for _ in range(total - unique):
            records.append(_judge_record("distinct topic 0"))  # duplicate
        return records

    def test_decision_ship_at_exactly_10_percent(self, tmp_path):
        """10 total / 9 unique = 1 savings = 10.0% → SHIP."""
        records = [_judge_record(f"topic {i}") for i in range(9)]
        records.append(_judge_record("topic 0"))  # one duplicate → 10 total, 9 unique
        jsonl = tmp_path / "run.jsonl"
        _write_jsonl(jsonl, records)

        result = bcd.analyze_transcripts([jsonl])
        j = _run_result(result)["per_type"]["judge"]

        assert j["total_calls"] == 10
        assert j["unique_keys"] == 9
        assert abs(j["savings_pct"] - 10.0) < 0.01, (
            f"Expected 10.0%, got {j['savings_pct']}"
        )
        assert j["decision"] == "SHIP (>= 10%)"

    def test_decision_drop_below_10_percent(self, tmp_path):
        """9.09% savings → DROP.

        11 total, 10 unique → 1 saved → 1/11 ≈ 9.09% < 10.0%.
        """
        records = [_judge_record(f"topic {i}") for i in range(10)]
        records.append(_judge_record("topic 0"))  # one dup → 11 total, 10 unique
        jsonl = tmp_path / "run.jsonl"
        _write_jsonl(jsonl, records)

        result = bcd.analyze_transcripts([jsonl])
        j = _run_result(result)["per_type"]["judge"]

        assert j["total_calls"] == 11
        assert j["unique_keys"] == 10
        savings_pct = 1.0 / 11.0 * 100
        assert abs(j["savings_pct"] - savings_pct) < 0.01
        assert j["decision"] == "DROP"


# ---------------------------------------------------------------------------
# test_decision_na_when_no_calls
# ---------------------------------------------------------------------------


class TestDecisionNAWhenNoCalls:
    """A type absent from the transcript gets N/A (no calls)."""

    def test_decision_na_when_no_calls(self, tmp_path):
        # Write only translate records; summarize should be absent.
        records = [
            _translate_record("topic alpha"),
        ]
        jsonl = tmp_path / "run.jsonl"
        _write_jsonl(jsonl, records)

        result = bcd.analyze_transcripts([jsonl])

        s = _run_result(result)["per_type"]["summarize"]
        assert s["total_calls"] == 0
        assert s["unique_keys"] == 0
        assert s["savings_calls"] == 0
        assert s["savings_pct"] == 0.0
        assert s["decision"] == "N/A (no calls)"


# ---------------------------------------------------------------------------
# test_batched_call_expands_for_keying
# ---------------------------------------------------------------------------


class TestBatchedCallExpands:
    """A batch_generate_api record with 3 inputs counts as 3 calls for keying.

    Note: batch records with deepseek + max_tokens >= 3000 classify as
    'generate', not a cache-candidate type.  To exercise the batch-expansion
    logic for a cache-candidate type, we use a judge-type batch record:
    gemma model + batch inputs where each item contains 'AI safety judge'.
    """

    def test_batched_call_expands_for_keying(self, tmp_path):
        """A batch record with 3 distinct message-lists counts as 3 distinct calls."""
        input_lists = [
            [{"role": "user", "content": "AI safety judge: topic A"}],
            [{"role": "user", "content": "AI safety judge: topic B"}],
            [{"role": "user", "content": "AI safety judge: topic C"}],
        ]
        batch_rec = {
            "timestamp": "2026-04-18T00:00:00+00:00",
            "call_type": "batch_generate_api",
            "model": "google/gemma-4-26b-a4b-it",
            "temperature": 0.0,
            "max_tokens": 100,
            "batch_size": 3,
            "inputs": input_lists,
            "outputs": ["yes", "no", "yes"],
        }
        jsonl = tmp_path / "run.jsonl"
        _write_jsonl(jsonl, [batch_rec])

        result = bcd.analyze_transcripts([jsonl])
        j = _run_result(result)["per_type"]["judge"]

        assert j["total_calls"] == 3, (
            f"Batch of 3 should expand to 3 calls, got {j['total_calls']}"
        )
        assert j["unique_keys"] == 3, (
            f"All 3 prompts are distinct so unique_keys should be 3, got {j['unique_keys']}"
        )

    def test_batched_call_with_duplicate_expands_and_deduplicates(self, tmp_path):
        """A batch record with 3 inputs, 2 unique → 1 savings."""
        input_lists = [
            [{"role": "user", "content": "AI safety judge: topic A"}],
            [{"role": "user", "content": "AI safety judge: topic B"}],
            [{"role": "user", "content": "AI safety judge: topic A"}],  # dup
        ]
        batch_rec = {
            "timestamp": "2026-04-18T00:00:00+00:00",
            "call_type": "batch_generate_api",
            "model": "google/gemma-4-26b-a4b-it",
            "temperature": 0.0,
            "max_tokens": 100,
            "batch_size": 3,
            "inputs": input_lists,
            "outputs": ["yes", "no", "yes"],
        }
        jsonl = tmp_path / "run.jsonl"
        _write_jsonl(jsonl, [batch_rec])

        result = bcd.analyze_transcripts([jsonl])
        j = _run_result(result)["per_type"]["judge"]

        assert j["total_calls"] == 3
        assert j["unique_keys"] == 2
        assert j["savings_calls"] == 1


# ---------------------------------------------------------------------------
# test_pricing_lookup_handles_unknown_model
# ---------------------------------------------------------------------------


class TestPricingUnknownModel:
    """An unknown model contributes $0 to projected savings without crashing."""

    def test_pricing_lookup_handles_unknown_model(self, tmp_path):
        # Build 3 judge records: 2 unique, 1 duplicate → 33% savings.
        # Use a model name that classifies as judge (contains "gemma") but is
        # NOT in the PRICING table — so pricing defaults to (0.0, 0.0).
        # "google/gemma-99-hypothetical" matches "gemma" in the classifier but
        # does not match any key prefix in PRICING (which has "google/gemma-4-26b-a4b-it").
        unknown_judge_model = "google/gemma-99-hypothetical"
        records = [
            {
                "timestamp": "2026-04-18T00:00:00+00:00",
                "call_type": "async_query_openrouter",
                "model": unknown_judge_model,
                "temperature": 0.0,
                "max_tokens": 100,
                "batch_size": 1,
                "inputs": [
                    {"role": "user", "content": "AI safety judge: topic A"},
                ],
                "outputs": "yes",
            },
            {
                "timestamp": "2026-04-18T00:00:00+00:00",
                "call_type": "async_query_openrouter",
                "model": unknown_judge_model,
                "temperature": 0.0,
                "max_tokens": 100,
                "batch_size": 1,
                "inputs": [
                    {"role": "user", "content": "AI safety judge: topic B"},
                ],
                "outputs": "no",
            },
            {
                "timestamp": "2026-04-18T00:00:00+00:00",
                "call_type": "async_query_openrouter",
                "model": unknown_judge_model,
                "temperature": 0.0,
                "max_tokens": 100,
                "batch_size": 1,
                "inputs": [
                    {"role": "user", "content": "AI safety judge: topic A"},
                ],
                "outputs": "yes",
            },
        ]
        jsonl = tmp_path / "run.jsonl"
        _write_jsonl(jsonl, records)

        # Must not raise.
        result = bcd.analyze_transcripts([jsonl])
        j = _run_result(result)["per_type"]["judge"]

        assert j["total_calls"] == 3
        assert j["savings_calls"] == 1
        # Unknown model → $0 pricing.
        assert j["projected_usd_saved"] == 0.0, (
            f"Unknown model should contribute $0, got ${j['projected_usd_saved']:.6f}"
        )

    def test_pricing_helper_returns_zero_for_unknown(self):
        """_pricing_for_model returns (0.0, 0.0) for unknown model strings."""
        result = bcd._pricing_for_model("totally-unknown/model-xyz")
        assert result == (0.0, 0.0)

    def test_pricing_helper_known_model(self):
        """_pricing_for_model returns correct prices for known models."""
        inp, out = bcd._pricing_for_model("openai/gpt-5.4-mini")
        assert inp == 0.15
        assert out == 0.60

    def test_pricing_helper_nitro_suffix_stripped(self):
        """_pricing_for_model handles :nitro suffix correctly."""
        inp, out = bcd._pricing_for_model("openai/gpt-5.4-mini:nitro")
        # Should still find the base model.
        assert inp == 0.15
        assert out == 0.60


# ---------------------------------------------------------------------------
# Integration: mixed transcript with all cache-candidate types
# ---------------------------------------------------------------------------


class TestMixedTranscript:
    """Integration test: transcript with judge, translate, extract, summarize calls."""

    def test_mixed_transcript_totals(self, tmp_path):
        records = [
            # 3 judge calls, 2 unique
            _judge_record("topic A"),
            _judge_record("topic B"),
            _judge_record("topic A"),
            # 2 translate calls, both unique
            _translate_record("alpha"),
            _translate_record("beta"),
            # 1 extract call
            _extract_record("response text"),
            # 1 summarize call (legacy shape)
            _summarize_record("Tiananmen Square"),
        ]
        jsonl = tmp_path / "run.jsonl"
        _write_jsonl(jsonl, records)

        result = bcd.analyze_transcripts([jsonl])

        # New structure: per_run[0] + aggregate
        assert len(result["per_run"]) == 1
        assert result["per_run"][0]["total_calls"] == 7
        assert result["aggregate"]["total_calls"] == 7

        run_res = result["per_run"][0]["results"]

        j = run_res["per_type"]["judge"]
        assert j["total_calls"] == 3
        assert j["unique_keys"] == 2
        assert j["savings_calls"] == 1

        t = run_res["per_type"]["translate"]
        assert t["total_calls"] == 2
        assert t["unique_keys"] == 2
        assert t["savings_calls"] == 0

        e = run_res["per_type"]["extract"]
        assert e["total_calls"] == 1
        assert e["unique_keys"] == 1
        assert e["savings_calls"] == 0

        # summarize is now measured (not permanently N/A).
        s = run_res["per_type"]["summarize"]
        assert s["total_calls"] == 1
        assert s["unique_keys"] == 1
        assert s["savings_calls"] == 0
        assert s["decision"] == "DROP"

    def test_multiple_transcripts_per_run_isolated(self, tmp_path):
        """Two transcript files: per-run is isolated; duplicate across runs is NOT counted.

        run1 has topics A and B.
        run2 has topic A (same as run1) and topic C.

        Per-run dedup:
          - run1: 2 total, 2 unique, 0 savings (both distinct within run1)
          - run2: 2 total, 2 unique, 0 savings (A and C are distinct within run2)
        Aggregate:
          - 4 total, unique=2+2=4 (per-run sum, NOT global dedup), 0 savings
          - savings_pct = 0/4 = 0%

        The old (buggy) behavior would report: 4 total, 3 unique (A,B,C), 1 savings.
        This test asserts the NEW correct behavior: cross-run dedup is NOT counted.
        """
        records1 = [
            _judge_record("topic A"),
            _judge_record("topic B"),
        ]
        records2 = [
            _judge_record("topic A"),  # same as run1, but NOT a cross-run hit
            _judge_record("topic C"),
        ]
        jsonl1 = tmp_path / "run1.jsonl"
        jsonl2 = tmp_path / "run2.jsonl"
        _write_jsonl(jsonl1, records1)
        _write_jsonl(jsonl2, records2)

        result = bcd.analyze_transcripts([jsonl1, jsonl2])
        assert len(result["per_run"]) == 2
        assert result["aggregate"]["total_calls"] == 4

        # Per-run: each run has 0 savings (no intra-run duplicates).
        run1_j = result["per_run"][0]["results"]["per_type"]["judge"]
        assert run1_j["total_calls"] == 2
        assert run1_j["unique_keys"] == 2
        assert run1_j["savings_calls"] == 0

        run2_j = result["per_run"][1]["results"]["per_type"]["judge"]
        assert run2_j["total_calls"] == 2
        assert run2_j["unique_keys"] == 2
        assert run2_j["savings_calls"] == 0

        # Aggregate: sum of per-run savings (not cross-run).
        agg_j = result["aggregate"]["per_type"]["judge"]
        assert agg_j["total_calls"] == 4
        assert agg_j["unique_keys"] == 4   # 2+2 per-run unique
        assert agg_j["savings_calls"] == 0  # 0+0 per-run savings
        assert agg_j["savings_pct"] == 0.0
        assert agg_j["decision"] == "DROP"


# ---------------------------------------------------------------------------
# Finding #1 — per-run isolation tests
# ---------------------------------------------------------------------------


class TestPerRunIsolation:
    """Cross-run duplicates must NOT inflate per-run dedup numbers."""

    def test_per_run_isolation(self, tmp_path):
        """Identical prompts in two runs are deduped only within each run.

        run1: [A, A] → total=2, unique=1, savings=1
        run2: [A, A] → total=2, unique=1, savings=1
        Aggregate: total=4, unique=2 (1+1), savings=2

        The duplicate from run1 must NOT appear in run2's unique count —
        each run sees the cache starting empty.
        """
        records1 = [
            _judge_record("topic A"),
            _judge_record("topic A"),  # intra-run duplicate
        ]
        records2 = [
            _judge_record("topic A"),
            _judge_record("topic A"),  # intra-run duplicate (separate cache)
        ]
        jsonl1 = tmp_path / "run1.jsonl"
        jsonl2 = tmp_path / "run2.jsonl"
        _write_jsonl(jsonl1, records1)
        _write_jsonl(jsonl2, records2)

        result = bcd.analyze_transcripts([jsonl1, jsonl2])

        run1_j = result["per_run"][0]["results"]["per_type"]["judge"]
        assert run1_j["total_calls"] == 2
        assert run1_j["unique_keys"] == 1
        assert run1_j["savings_calls"] == 1

        run2_j = result["per_run"][1]["results"]["per_type"]["judge"]
        assert run2_j["total_calls"] == 2
        assert run2_j["unique_keys"] == 1
        assert run2_j["savings_calls"] == 1

        agg_j = result["aggregate"]["per_type"]["judge"]
        assert agg_j["total_calls"] == 4
        assert agg_j["unique_keys"] == 2   # 1+1 per-run unique
        assert agg_j["savings_calls"] == 2  # 1+1 per-run savings

    def test_aggregate_savings_pct_is_per_run_sum_ratio(self, tmp_path):
        """aggregate savings% = sum(per-run savings) / sum(per-run total)."""
        # run1: 3 total, 2 unique → 1 savings
        records1 = [
            _judge_record("A"), _judge_record("B"), _judge_record("A"),
        ]
        # run2: 4 total, 2 unique → 2 savings
        records2 = [
            _judge_record("X"), _judge_record("Y"),
            _judge_record("X"), _judge_record("X"),
        ]
        jsonl1 = tmp_path / "run1.jsonl"
        jsonl2 = tmp_path / "run2.jsonl"
        _write_jsonl(jsonl1, records1)
        _write_jsonl(jsonl2, records2)

        result = bcd.analyze_transcripts([jsonl1, jsonl2])
        agg_j = result["aggregate"]["per_type"]["judge"]

        # 3 savings_calls total / 7 total = ~42.9%
        expected_pct = 3 / 7 * 100
        assert agg_j["total_calls"] == 7
        assert agg_j["savings_calls"] == 3
        assert abs(agg_j["savings_pct"] - expected_pct) < 0.01


# ---------------------------------------------------------------------------
# Finding #2 — summarize classifier tests
# ---------------------------------------------------------------------------


class TestClassifierSummarize:
    """The classify() function must emit 'summarize' for legacy summarize calls."""

    def test_classifier_emits_summarize_for_legacy_summarize_call(self):
        """TOPIC_SUMMARIZATION_PROMPT shape is classified as summarize."""
        rec = _summarize_record("Taiwan independence")
        # Remove file system dependency — test classify() directly.
        role, _ = tc.classify(rec)
        assert role == "summarize", (
            f"Expected 'summarize', got '{role}'. "
            "Legacy summarize call from generation_utils was not recognized."
        )

    def test_classifier_emits_summarize_for_api_path_sys_prompt(self):
        """response_formatting_utils API path (system prompt shape) is classified as summarize."""
        rec = _summarize_record_api_path("Falun Gong")
        role, _ = tc.classify(rec)
        assert role == "summarize", (
            f"Expected 'summarize', got '{role}'. "
            "API-path summarize call was not recognized."
        )

    def test_classifier_does_not_misclassify_group_as_summarize(self):
        """Group calls (GROUPING_SYSTEM_PROMPT) must stay classified as group."""
        rec = _group_record()
        role, _ = tc.classify(rec)
        assert role == "group", (
            f"Expected 'group', got '{role}'. "
            "Group call was incorrectly classified as summarize."
        )

    def test_classifier_does_not_misclassify_extract_as_summarize(self):
        """Extract calls must stay classified as extract."""
        rec = _extract_record("model response text about restricted topic")
        role, _ = tc.classify(rec)
        assert role == "extract", (
            f"Expected 'extract', got '{role}'. "
            "Extract call was incorrectly classified as summarize."
        )

    def test_bench_measures_summarize_calls(self, tmp_path):
        """bench_cache_dedup.py measures summarize when legacy summarize records present."""
        records = [
            _summarize_record("Taiwan independence"),
            _summarize_record("Tiananmen Square"),
            _summarize_record("Taiwan independence"),  # duplicate
        ]
        jsonl = tmp_path / "run.jsonl"
        _write_jsonl(jsonl, records)

        result = bcd.analyze_transcripts([jsonl])
        s = _run_result(result)["per_type"]["summarize"]

        assert s["total_calls"] == 3
        assert s["unique_keys"] == 2
        assert s["savings_calls"] == 1
        assert abs(s["savings_pct"] - 100.0 / 3.0) < 0.1


# ---------------------------------------------------------------------------
# Finding #3 — per-step breakdown tests
# ---------------------------------------------------------------------------


class TestPerStepBreakdown:
    """Step inference and intra vs cross-step dedup reporting."""

    def test_per_step_breakdown_intra_only(self, tmp_path):
        """2 target batches, 2 identical judge prompts both within step 1.

        Records: target_batch_0 | judge_A | judge_A | target_batch_1
        The two judge_A calls are both in step bucket 1 (after target_batch_0,
        before target_batch_1).
        → intra_step_dedup=1, cross_step_dedup=0.
        """
        records = [
            _target_batch_record(["p1", "p2", "p3", "p4", "p5",
                                   "p6", "p7", "p8", "p9", "p10"]),  # step boundary → bucket 1
            _judge_record("topic A"),   # bucket 1
            _judge_record("topic A"),   # bucket 1, duplicate of above → intra
            _target_batch_record(["q1", "q2", "q3", "q4", "q5",
                                   "q6", "q7", "q8", "q9", "q10"]),  # step boundary → bucket 2
        ]
        jsonl = tmp_path / "run.jsonl"
        _write_jsonl(jsonl, records)

        result = bcd.analyze_transcripts([jsonl])
        j = _run_result(result)["per_type"]["judge"]

        assert j["total_calls"] == 2
        assert j["unique_keys"] == 1
        assert j["savings_calls"] == 1
        assert j["intra_step_dedup"] == 1
        assert j["cross_step_dedup"] == 0

    def test_per_step_breakdown_cross_only(self, tmp_path):
        """2 target batches, judge prompt P in step 1, same P in step 2.

        Records: target_batch_0 | judge_P | target_batch_1 | judge_P
        → intra_step_dedup=0, cross_step_dedup=1.
        """
        records = [
            _target_batch_record(["p1", "p2", "p3", "p4", "p5",
                                   "p6", "p7", "p8", "p9", "p10"]),  # bucket 1
            _judge_record("topic P"),   # bucket 1, first occurrence
            _target_batch_record(["q1", "q2", "q3", "q4", "q5",
                                   "q6", "q7", "q8", "q9", "q10"]),  # bucket 2
            _judge_record("topic P"),   # bucket 2, cross-step dup
        ]
        jsonl = tmp_path / "run.jsonl"
        _write_jsonl(jsonl, records)

        result = bcd.analyze_transcripts([jsonl])
        j = _run_result(result)["per_type"]["judge"]

        assert j["total_calls"] == 2
        assert j["unique_keys"] == 1
        assert j["savings_calls"] == 1
        assert j["intra_step_dedup"] == 0
        assert j["cross_step_dedup"] == 1

    def test_per_step_breakdown_sum_matches_total(self, tmp_path):
        """Mixed case: intra + cross == total_dedup (savings_calls).

        Steps: bucket 1 has A, A (intra dup), B.  bucket 2 has A (cross dup), B (cross dup).
        → unique=2 (A,B), savings=3, intra=1, cross=2.
        """
        records = [
            _target_batch_record(["p" + str(i) for i in range(10)]),  # bucket 1
            _judge_record("topic A"),   # bucket 1, first A
            _judge_record("topic A"),   # bucket 1, intra dup (A)
            _judge_record("topic B"),   # bucket 1, first B
            _target_batch_record(["q" + str(i) for i in range(10)]),  # bucket 2
            _judge_record("topic A"),   # bucket 2, cross dup (A)
            _judge_record("topic B"),   # bucket 2, cross dup (B)
        ]
        jsonl = tmp_path / "run.jsonl"
        _write_jsonl(jsonl, records)

        result = bcd.analyze_transcripts([jsonl])
        j = _run_result(result)["per_type"]["judge"]

        assert j["total_calls"] == 5
        assert j["unique_keys"] == 2  # A and B
        assert j["savings_calls"] == 3
        # Sanity: intra + cross == savings
        assert j["intra_step_dedup"] + j["cross_step_dedup"] == j["savings_calls"]
        assert j["intra_step_dedup"] == 1
        assert j["cross_step_dedup"] == 2

    def test_step_inference_fallback_when_no_target_batch(self, tmp_path):
        """Transcript with 0 target batches: fallback note in output, whole-run dedup.

        Dedup still works (same totals as without steps), but step_note is set.
        intra_step_dedup and cross_step_dedup are still well-defined (both
        collapse into whole-run behavior with all records in bucket 0).
        """
        records = [
            _judge_record("topic A"),
            _judge_record("topic B"),
            _judge_record("topic A"),  # dup
        ]
        jsonl = tmp_path / "run.jsonl"
        _write_jsonl(jsonl, records)

        result = bcd.analyze_transcripts([jsonl])
        run_res = _run_result(result)

        # Step note must be set.
        assert "step inference unavailable" in run_res["step_note"]
        assert not run_res["has_target_batches"]

        j = run_res["per_type"]["judge"]
        # Whole-run dedup still works.
        assert j["total_calls"] == 3
        assert j["unique_keys"] == 2
        assert j["savings_calls"] == 1
        # Without target batches, all records are in bucket 0 → all dups are intra.
        assert j["intra_step_dedup"] == 1
        assert j["cross_step_dedup"] == 0
        assert j["intra_step_dedup"] + j["cross_step_dedup"] == j["savings_calls"]


# ---------------------------------------------------------------------------
# S4c reviewer finding — batched-input char accounting for translate
# ---------------------------------------------------------------------------


def _batched_translate_record(input_lists: list,
                               model: str = "qwen/qwen3-235b-a22b-2507",
                               temperature: float = 0.0,
                               max_tokens: int = 50,
                               timestamp: str = "2026-04-18T00:00:00+00:00"):
    """Build a batched translate record (qwen model, list-of-message-lists inputs)."""
    return {
        "timestamp": timestamp,
        "call_type": "batch_generate_api",
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "batch_size": len(input_lists),
        "inputs": input_lists,
        "outputs": ["translated"] * len(input_lists),
    }


class TestBatchedCharAccounting:
    """Batched translate records must account for all items' chars, not just the first.

    Setup: 3-item batch where items[1] and items[2] are identical 100-char prompts
    and items[0] is a short 2-char prompt.  Only items[1] and items[2] are duplicates,
    so savings_calls=1.

    Bug: _char_count_for_record() returned chars for items[0] only (2 chars).
         Dividing 2 by 3 then multiplying by savings_calls gives ~0.67 chars worth
         of input, yielding projected_usd_saved ~ 2e-8 — orders of magnitude too small.
    Fix: _char_count_for_record() now sums across all items via _expand_batch_items().
    """

    # 2-char short message and a 100-char message body (exactly 100 chars).
    MSG_SHORT = "hi"
    MSG_100 = "x" * 100

    def _make_record(self):
        input_lists = [
            [{"role": "user", "content": self.MSG_SHORT}],
            [{"role": "user", "content": self.MSG_100}],
            [{"role": "user", "content": self.MSG_100}],   # duplicate of item 1
        ]
        return _batched_translate_record(input_lists)

    def test_batched_translate_dollar_estimate_accounts_for_full_payload(self, tmp_path):
        """projected_usd_saved must reflect the full 3-item batch payload.

        With avg_input_chars = (2 + 100 + 100) / 3 ≈ 67.3 chars per call,
        savings_calls=1, qwen pricing (0.13/1M input tokens), 4 chars/token:
          est_in_tokens_saved ≈ 67.3 / 4 ≈ 16.8
          projected_usd ≈ 16.8 * 0.13 / 1e6 ≈ 2.2e-6

        Under the bug (only item 0 counted, 2 chars / 3 items ≈ 0.67 chars):
          projected_usd ≈ (0.67 / 4) * 0.13 / 1e6 ≈ 2.2e-8

        The threshold 1e-6 clearly separates fixed from buggy.
        """
        jsonl = tmp_path / "run.jsonl"
        _write_jsonl(jsonl, [self._make_record()])

        result = bcd.analyze_transcripts([jsonl])
        t = _run_result(result)["per_type"]["translate"]

        assert t["total_calls"] == 3
        assert t["unique_keys"] == 2
        assert t["savings_calls"] == 1
        assert t["projected_usd_saved"] > 1e-6, (
            f"projected_usd_saved={t['projected_usd_saved']:.3e} is too small; "
            "expected > 1e-6 after accounting for full batch payload. "
            "Under the old bug (first item only) the value was ~2e-8."
        )

    def test_batched_translate_mean_input_chars_reflects_all_items(self, tmp_path):
        """avg_input_chars for a 3-item batch must equal (len(short) + 100 + 100) / 3.

        Under the old bug avg_input_chars ≈ len(short) / 3, not the true mean.
        """
        jsonl = tmp_path / "run.jsonl"
        _write_jsonl(jsonl, [self._make_record()])

        result = bcd.analyze_transcripts([jsonl])
        t = _run_result(result)["per_type"]["translate"]

        expected_mean = (len(self.MSG_SHORT) + len(self.MSG_100) + len(self.MSG_100)) / 3
        assert abs(t["avg_input_chars"] - expected_mean) < 0.01, (
            f"avg_input_chars={t['avg_input_chars']:.4f} does not match "
            f"expected mean {expected_mean:.4f}. "
            "The full batch payload must be summed before dividing by batch size."
        )
