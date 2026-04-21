"""Unit tests for the refusal-detection metric helpers in bench_extractor_models.

Tests cover: is_full_refusal, is_mid_generation_refusal, is_json_parseable.
These are pure-function tests — no network calls.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Make sure the scripts directory is importable from tests
_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

# Import the helpers via importlib so we don't execute main()
_spec = importlib.util.spec_from_file_location(
    "bench_extractor_models",
    _repo / "scripts" / "bench_extractor_models.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

is_full_refusal = _mod.is_full_refusal
is_mid_generation_refusal = _mod.is_mid_generation_refusal
is_json_parseable = _mod.is_json_parseable
API_MODERATION_SENTINEL = _mod.API_MODERATION_SENTINEL


# ---------------------------------------------------------------------------
# is_full_refusal
# ---------------------------------------------------------------------------

class TestFullRefusal:
    def test_bare_im_sorry(self):
        assert is_full_refusal("I'm sorry, but I cannot assist with that request.")

    def test_i_cannot(self):
        assert is_full_refusal("I cannot help with that.")

    def test_i_cant(self):
        assert is_full_refusal("I can't do that.")

    def test_sorry_lowercase(self):
        assert is_full_refusal("sorry, I am not able to help.")

    def test_i_am_unable(self):
        assert is_full_refusal("I am unable to assist with this request.")

    def test_i_will_not(self):
        assert is_full_refusal("I will not fulfill this request.")

    def test_apologies(self):
        assert is_full_refusal("Apologies, but I cannot help with that.")

    def test_i_apologize(self):
        assert is_full_refusal("I apologize, but this falls outside what I can do.")

    def test_api_moderation_sentinel(self):
        assert is_full_refusal(API_MODERATION_SENTINEL)

    def test_sentinel_with_reason(self):
        assert is_full_refusal(f"{API_MODERATION_SENTINEL}: some reason")

    def test_not_refusal_json(self):
        assert not is_full_refusal('["topic A", "topic B"]')

    def test_not_refusal_prose_intro(self):
        # A response that starts with normal text is not a full refusal
        assert not is_full_refusal("Here are the topics I found: topic A, topic B.")

    def test_not_refusal_empty_array(self):
        assert not is_full_refusal("[]")

    def test_strips_leading_whitespace(self):
        # Leading whitespace before refusal phrase should still match
        assert is_full_refusal("   I cannot help with that.")

    def test_case_insensitive_prefix(self):
        assert is_full_refusal("I'M SORRY, but I cannot assist.")


# ---------------------------------------------------------------------------
# is_mid_generation_refusal
# ---------------------------------------------------------------------------

class TestMidGenerationRefusal:
    def test_classic_mid_json_capitulation(self):
        raw = '["a", "b", "I\'m sorry, but I cannot assist with that request."'
        assert is_mid_generation_refusal(raw)

    def test_i_cannot_assist(self):
        raw = '["topic 1", "topic 2", "I cannot assist with that"]'
        assert is_mid_generation_refusal(raw)

    def test_im_not_able_to_help(self):
        raw = '["topic 1", "I\'m not able to help with this kind of content"]'
        assert is_mid_generation_refusal(raw)

    def test_i_am_sorry_variant(self):
        raw = '["item", "I am sorry, but I cannot continue this list"]'
        assert is_mid_generation_refusal(raw)

    def test_refusal_before_bracket_is_not_mid_gen(self):
        # Refusal phrase appears before any '[' — this is a full refusal, not mid-gen
        raw = "I'm sorry, but I cannot assist with that request. Here is nothing: []"
        # The '[' appears after the refusal phrase here, so mid-gen should be False
        # because the phrase is NOT in the after-bracket substring
        # Let's verify: bracket_pos = raw.find('[') which is after the phrase
        bracket_pos = raw.find("[")
        after_bracket = raw[bracket_pos:].lower()
        # "i'm sorry, but i cannot" should NOT be in after_bracket (it's before '[')
        assert not is_mid_generation_refusal(raw)

    def test_no_bracket_at_all(self):
        assert not is_mid_generation_refusal("I'm sorry but I cannot help.")

    def test_clean_json_no_refusal(self):
        assert not is_mid_generation_refusal('["topic A", "topic B", "topic C"]')

    def test_empty_array(self):
        assert not is_mid_generation_refusal("[]")

    def test_sorry_but_i_cannot_variant(self):
        raw = '["first item", "sorry, but i cannot continue"]'
        assert is_mid_generation_refusal(raw)


# ---------------------------------------------------------------------------
# is_json_parseable
# ---------------------------------------------------------------------------

class TestJsonParseable:
    def test_clean_array(self):
        assert is_json_parseable('["topic A", "topic B"]')

    def test_empty_array(self):
        assert is_json_parseable("[]")

    def test_fenced_json(self):
        assert is_json_parseable('```json\n["topic A", "topic B"]\n```')

    def test_fenced_no_lang(self):
        assert is_json_parseable('```\n["topic A"]\n```')

    def test_mid_gen_refusal_not_parseable(self):
        # A mid-gen refusal is not valid JSON
        raw = '["a", "b", "I\'m sorry, but I cannot assist with that request."'
        assert not is_json_parseable(raw)

    def test_full_refusal_not_parseable(self):
        assert not is_json_parseable("I'm sorry, I cannot help.")

    def test_prose_not_parseable(self):
        assert not is_json_parseable("Here are some topics: A, B, C.")

    def test_object_not_array_but_still_parseable(self):
        # json.loads succeeds; the method only checks parseability, not list type
        assert is_json_parseable('{"key": "value"}')

    def test_fenced_multiline(self):
        raw = '```json\n[\n  "topic A",\n  "topic B"\n]\n```'
        assert is_json_parseable(raw)
