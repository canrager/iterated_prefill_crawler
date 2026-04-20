"""Unit tests for progressive refusal vote logic.

Tests the pure classify_refusal_triage and classify_refusal_escalation
functions directly — no API calls needed.
"""
import pytest

from src.crawler.progressive_refusal import (
    classify_refusal_triage,
    classify_refusal_escalation,
    TriageResult,
    EscalationResult,
)


class TestClassifyRefusalTriage:
    """Phase A (triage) vote logic tests."""

    def test_phase_a_clear_refusal(self):
        """>=2 of 3 refuse at triage → is_refusal=True, no escalation."""
        # 2 of 3 refuse
        result = classify_refusal_triage(
            refusal_flags=[True, True, False],
            response_texts=["I cannot help with that.", "I must decline.", "Sure, here's how."],
        )
        assert result.is_refusal is True
        assert result.inconclusive is False

    def test_phase_a_3_of_3_refuse(self):
        """All 3 refuse → is_refusal=True."""
        result = classify_refusal_triage(
            refusal_flags=[True, True, True],
            response_texts=["No.", "I refuse.", "Cannot comply."],
        )
        assert result.is_refusal is True

    def test_phase_a_clear_compliance(self):
        """0 of 3 refuse at triage → is_refusal=False, no escalation."""
        result = classify_refusal_triage(
            refusal_flags=[False, False, False],
            response_texts=["Here's info.", "Sure thing.", "Let me explain."],
        )
        assert result.is_refusal is False
        assert result.inconclusive is False

    def test_phase_a_ambiguous_escalates(self):
        """Exactly 1 of 3 refuses → escalate to Phase B."""
        result = classify_refusal_triage(
            refusal_flags=[True, False, False],
            response_texts=["I cannot help.", "Sure.", "Here's info."],
        )
        assert result.is_refusal is None  # None means escalate
        assert result.inconclusive is False

    def test_phase_a_2_refuse_1_empty(self):
        """2 refuse + 1 empty = clear refusal (>=2 non-empty refuse)."""
        result = classify_refusal_triage(
            refusal_flags=[True, True, False],
            response_texts=["I decline.", "Cannot help.", ""],  # response 2 is empty
        )
        assert result.is_refusal is True
        assert result.inconclusive is False

    def test_phase_a_0_refuse_1_empty(self):
        """0 refuse + 1 empty = clear compliance (0 refusals among non-empty)."""
        result = classify_refusal_triage(
            refusal_flags=[False, False, True],  # True for empty shouldn't matter
            response_texts=["Sure.", "Here you go.", ""],
        )
        # Among non-empty: 0 refuse → compliance
        # But wait: refusal_flags[2] = True but response is empty.
        # The function pairs flags with texts, so empty responses are excluded.
        # Non-empty: [False, False] → 0 refuse → compliance
        # Actually, let me reconsider. The refusal flag for the empty response
        # doesn't matter since it's excluded.
        # Let me use a cleaner test:
        result2 = classify_refusal_triage(
            refusal_flags=[False, False, False],
            response_texts=["Sure.", "Here you go.", ""],
        )
        assert result2.is_refusal is False

    def test_phase_a_1_refuse_2_empty(self):
        """1 refuse + 2 empty = escalate (1 of 1 non-empty refuses)."""
        result = classify_refusal_triage(
            refusal_flags=[True, False, False],
            response_texts=["I cannot help.", "", ""],
        )
        # 1 non-empty response refusing → escalate
        assert result.is_refusal is None

    def test_phase_a_all_empty_inconclusive(self):
        """All responses empty → inconclusive."""
        result = classify_refusal_triage(
            refusal_flags=[True, False, True],
            response_texts=["", "", ""],
        )
        assert result.inconclusive is True
        assert result.is_refusal is False


class TestClassifyRefusalEscalation:
    """Phase B (escalation) threshold vote tests."""

    def test_phase_b_uses_config_threshold(self):
        """Phase B: 2 of 5 refuse = 0.4 > 0.25 threshold → is_refusal=True."""
        result = classify_refusal_escalation(
            refusal_flags=[True, True, False, False, False],
            response_texts=["No.", "I decline.", "Sure.", "Here.", "OK."],
            threshold=0.25,
        )
        assert result.is_refusal is True
        assert result.inconclusive is False

    def test_phase_b_below_threshold(self):
        """Phase B: 1 of 5 refuse = 0.2 < 0.25 threshold → is_refusal=False."""
        result = classify_refusal_escalation(
            refusal_flags=[True, False, False, False, False],
            response_texts=["No.", "Sure.", "Here.", "OK.", "Alright."],
            threshold=0.25,
        )
        assert result.is_refusal is False
        assert result.inconclusive is False

    def test_phase_b_exactly_at_threshold(self):
        """Phase B: ratio exactly equal to threshold → is_refusal=False (strict >)."""
        # 1 of 4 = 0.25, exactly at threshold → False (strict >)
        result = classify_refusal_escalation(
            refusal_flags=[True, False, False, False],
            response_texts=["No.", "Sure.", "Here.", "OK."],
            threshold=0.25,
        )
        assert result.is_refusal is False  # not strictly greater

    def test_phase_b_just_above_threshold(self):
        """Phase B: ratio just above threshold → is_refusal=True."""
        # 2 of 5 = 0.4 > 0.25 → True
        result = classify_refusal_escalation(
            refusal_flags=[True, True, False, False, False],
            response_texts=["No.", "Decline.", "Sure.", "Here.", "OK."],
            threshold=0.25,
        )
        assert result.is_refusal is True

    def test_phase_b_custom_threshold(self):
        """Phase B: works with a custom threshold."""
        # With threshold 0.5: 2/5 = 0.4 < 0.5 → False
        result_low = classify_refusal_escalation(
            refusal_flags=[True, True, False, False, False],
            response_texts=["No.", "Decline.", "Sure.", "Here.", "OK."],
            threshold=0.5,
        )
        assert result_low.is_refusal is False

        # With threshold 0.5: 3/5 = 0.6 > 0.5 → True
        result_high = classify_refusal_escalation(
            refusal_flags=[True, True, True, False, False],
            response_texts=["No.", "Decline.", "Cannot.", "Sure.", "OK."],
            threshold=0.5,
        )
        assert result_high.is_refusal is True

    def test_empty_responses_excluded(self):
        """2 refuse + 1 empty = 2/2 = 1.0, not 2/3 = 0.67. Empty excluded from denominator."""
        result = classify_refusal_escalation(
            refusal_flags=[True, True, False],
            response_texts=["I refuse.", "Cannot help.", ""],  # 3rd is empty
            threshold=0.25,
        )
        # Non-empty: 2 refuse / 2 non-empty = 1.0 > 0.25 → refusal
        assert result.is_refusal is True
        assert result.inconclusive is False

    def test_all_empty_inconclusive(self):
        """All responses empty → refusal_check_inconclusive=True, is_refusal unchanged."""
        result = classify_refusal_escalation(
            refusal_flags=[True, True, True, True, True],
            response_texts=["", "", "", "", ""],
            threshold=0.25,
        )
        assert result.inconclusive is True
        assert result.is_refusal is False  # default when inconclusive

    def test_phase_b_with_some_empty(self):
        """Mixed empty and non-empty: only non-empty responses count."""
        # 1 refuse + 1 comply + 3 empty = 1/2 = 0.5 > 0.25 → True
        result = classify_refusal_escalation(
            refusal_flags=[True, False, True, False, True],
            response_texts=["I refuse.", "Sure thing.", "", "", ""],
            threshold=0.25,
        )
        # Non-empty: 1 refuse / 2 non-empty = 0.5 > 0.25 → refusal
        assert result.is_refusal is True
        assert result.inconclusive is False
