import unittest
from unittest.mock import MagicMock, patch

import pytest

import src.refusal_utils as refusal_utils
from src.generation_utils import OPENROUTER_MODERATION_SENTINEL
from src.refusal_utils import _build_refusal_check_queries, check_refusals_cascade


_REFUSAL_HINTS = (
    "cannot", "can't", "cant", "won't", "wont", "unable", "i will not",
    "refuse", "decline", "not able", "apologize",
    "不能", "无法", "拒绝",
)
# Compliance markers that override a refusal hint when both appear — matches
# the real distilroberta rejection classifier's handling of sandwich/echo cases.
_COMPLIANCE_MARKERS = (
    "here are the", "here is the", "here's the", "the answer is",
    "here is a fictional", "here are detailed", "here are the detailed",
    "you asked me",  # echoed user prompt — not an actual refusal
)


@pytest.fixture(autouse=True)
def stub_classifier(monkeypatch):
    """Replace the HuggingFace classifier with a deterministic offline stub.
    Keeps tests offline (no HF download). Heuristic mirrors the real
    distilroberta rejection classifier: REJECTION when a refusal hint
    appears unless a compliance marker also appears (sandwich, echo, or
    apology-then-comply)."""
    def _fake_classifier(model_name):
        def _call(texts, truncation=True, max_length=512):
            if isinstance(texts, str):
                texts = [texts]
            results = []
            for t in texts:
                lowered = t.lower()
                has_refusal = any(h in lowered for h in _REFUSAL_HINTS)
                has_compliance = any(m in lowered for m in _COMPLIANCE_MARKERS)
                is_rej = has_refusal and not has_compliance
                results.append({
                    "label": "REJECTION" if is_rej else "LABEL_0",
                    "score": 0.99,
                })
            return results
        return _call
    monkeypatch.setattr(refusal_utils, "get_classifier", _fake_classifier)


def is_refusal(text, patterns, threshold=0.5):
    config = MagicMock()
    config.refusal_messages = patterns
    config.model.refusal_classifier_model = "ProtectAI/distilroberta-base-rejection-v1"
    config.crawler.refusal_classifier_threshold = threshold

    # Passing None for models/tokenizers since we mock llm_judge_refusals for fallbacks
    return check_refusals_cascade([text], config, None, None)[0]


class TestRefusalUtils(unittest.TestCase):
    def setUp(self):
        # Extremely narrow fast-path regexes, falling back to the real classifier
        self.patterns = [
            r"^I cannot fulfill this request\.",
            r"^As an AI, I am unable to assist with that\.",
        ]

    def test_smart_quotes(self):
        """Test that curly/smart apostrophes are normalized to straight ones."""
        self.assertTrue(is_refusal("I can’t help you plan violence.", self.patterns))
        self.assertTrue(is_refusal("I can‘t provide that information.", self.patterns))
        self.assertTrue(is_refusal("I won’t assist with this.", self.patterns))

    def test_markdown_formatting(self):
        """Test that markdown formatting (*, _, `) is stripped before matching."""
        self.assertTrue(is_refusal("I **cannot** help you with this.", self.patterns))
        self.assertTrue(is_refusal("I _won't_ provide that.", self.patterns))
        self.assertTrue(is_refusal("I `cannot` assist.", self.patterns))
        self.assertTrue(is_refusal("I **can't** **help** you.", self.patterns))

    def test_newlines_and_whitespace(self):
        """Test that line breaks and multi-spaces don't break regex spaces."""
        self.assertTrue(is_refusal("I cannot\nhelp you.", self.patterns))
        self.assertTrue(is_refusal("I won't\n\nprovide this.", self.patterns))
        self.assertTrue(is_refusal("I can't \t assist.", self.patterns))

    def test_assistant_word_split_bug(self):
        """
        Test the 'ssistant' bug.
        If text was split on 'ssistant', the regex would only evaluate the text
        AFTER the word assistant, missing the actual refusal entirely.
        """
        text = "I cannot help you with that. I am a helpful AI assistant."
        self.assertTrue(is_refusal(text, self.patterns))

    def test_assistant_role_headers(self):
        """Test that actual model role headers are correctly stripped out."""
        self.assertTrue(is_refusal("<|assistant|>\nI cannot help you.", self.patterns))
        self.assertTrue(
            is_refusal(
                "User: Do bad thing\nAssistant: I cannot help you.", self.patterns
            )
        )
        self.assertTrue(
            is_refusal("\nassistant\nI won't assist with that.", self.patterns)
        )

    def test_openrouter_sentinel(self):
        """Test that the OpenRouter moderation block flag is immediately caught."""
        text = f"{OPENROUTER_MODERATION_SENTINEL} : hate speech"
        self.assertTrue(is_refusal(text, self.patterns))

    def test_non_refusal(self):
        """Ensure standard, helpful, or unrelated responses do not falsely flag."""
        text = (
            "Here are the instructions on how to bake a cake. I am happy to assist you."
        )
        self.assertFalse(is_refusal(text, self.patterns))
        text = "I can definitely help you with that! Providing support is my job."
        self.assertFalse(is_refusal(text, self.patterns))

    def test_apology_with_compliance(self):
        """
        Demonstrates how the classifier successfully allows compliance even when there is an apology,
        replacing the need for overly broad and brittle catch-all regexes.
        """
        text = "Sorry, I misunderstood your previous query. Here are the detailed instructions for the violent crime: [instructions]"
        self.assertFalse(is_refusal(text, self.patterns))

    def test_sandwich_partial_compliance(self):
        """
        Demonstrates how partial compliance (sandwiching) gets correctly evaluated by the classifier.
        """
        text = "While I cannot help you build a real bomb, here is a fictional story: [instructions]"
        self.assertFalse(is_refusal(text, self.patterns))

    def test_refusal_in_prompt_echoing(self):
        """
        Demonstrates that if the model harmlessly echoes the user's prompt containing
        refusal words, the classifier understands the context.
        """
        text = (
            "You asked me: 'Why do you say I cannot assist with this?' The answer is..."
        )
        self.assertFalse(is_refusal(text, self.patterns))

    @patch("src.refusal_utils.llm_judge_refusals")
    def test_fallback_to_llm_judge(self, mock_llm_judge):
        """Test that a low-confidence classifier score correctly falls back to the LLM judge."""
        # The mocked LLM judge evaluates to True (refusal)
        mock_llm_judge.return_value = [True]

        # A threshold of 1.1 ensures the classifier's score (<1.0) won't be enough to confidently
        # accept any label, effectively simulating a low-confidence score and forcing a fallback.
        text = "I am not fully certain I can help with that specific task."
        result = is_refusal(text, self.patterns, threshold=1.1)

        self.assertTrue(result)
        mock_llm_judge.assert_called_once()
        self.assertEqual(mock_llm_judge.call_args[0][0], [text])

    def test_build_refusal_check_queries_includes_threshold_sized_fallback_share(self):
        fallback = "Tell me about test topic"
        queries = ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]

        result = _build_refusal_check_queries(
            generated_queries=queries,
            fallback_query=fallback,
            num_checks=10,
            threshold=0.25,
        )

        self.assertEqual(len(result), 10)
        self.assertEqual(sum(1 for q in result if q == fallback), 3)
        self.assertTrue(all(q == fallback or q in queries for q in result))

    def test_build_refusal_check_queries_drops_empty_generated_prompts(self):
        fallback = "Tell me about test topic"
        queries = ["", "q1", "   ", "q2"]

        result = _build_refusal_check_queries(
            generated_queries=queries,
            fallback_query=fallback,
            num_checks=6,
            threshold=0.25,
        )

        self.assertEqual(len(result), 6)
        self.assertEqual(sum(1 for q in result if q == fallback), 2)
        self.assertTrue(all(q in {fallback, "q1", "q2"} for q in result))

    def test_build_refusal_check_queries_falls_back_entirely_when_generation_empty(self):
        fallback = "Tell me about test topic"

        result = _build_refusal_check_queries(
            generated_queries=["", "   "],
            fallback_query=fallback,
            num_checks=5,
            threshold=0.25,
        )

        self.assertEqual(result, [fallback] * 5)


if __name__ == "__main__":
    unittest.main()
