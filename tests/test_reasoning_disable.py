"""Integration test: reasoning disable via extra_body.

Call Kimi k2.5 with extra_body={"reasoning": {"effort": "none"}} and assert
the response is clean JSON without a thinking-tag block.
"""
import asyncio
import json
import os

import pytest

from src.openrouter_utils import async_query_openrouter

THINKING_TAG = "<" + "think"  # avoid literal thinking-tag in source


@pytest.mark.integration
def test_k2_5_reasoning_disabled():
    """Kimi k2.5 with reasoning disabled should return clean JSON without thinking blocks."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")

    async def _test():
        prompt = (
            'Classify the following topics into groups. '
            'Respond with ONLY a JSON object: {"groups": [{"summary": "label", '
            '"member_indices": [0]}], "duplicates": [], "skipped": []}\n\n'
            'Topics:\n[0] Drug manufacturing\n[1] Illegal drug synthesis'
        )

        response = await async_query_openrouter(
            model_name="moonshotai/kimi-k2.5",
            prompt=prompt,
            system_prompt="You are a topic-deduplication assistant. Respond ONLY with valid JSON.",
            max_tokens=4096,
            temperature=0.0,
            extra_body={"reasoning": {"effort": "none"}},
        )

        assert response, f"Empty response from k2.5 (reasoning disabled)"

        # Must not contain thinking blocks
        assert THINKING_TAG not in response, f"Response contains thinking block: {response[:200]}"

        # Must be parseable JSON
        stripped = response.strip()
        if stripped.startswith("```"):
            for part in stripped.split("```"):
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate.startswith("{"):
                    stripped = candidate
                    break
        data = json.loads(stripped)
        assert "groups" in data, f"Response JSON missing 'groups' key: {data}"
        assert isinstance(data["groups"], list)

        print(f"k2.5 (reasoning disabled) response: {response[:300]}")
        print(f"Parsed groups: {len(data['groups'])}")

    asyncio.run(_test())
