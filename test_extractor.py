"""
Test LLM-based topic extraction against real model responses.

Sends each response from cleaned-chat.json through an extraction prompt
and asks for structured JSON output. Uses a small open model via OpenRouter.

Usage:
    uv run python test_extractor.py
    uv run python test_extractor.py --model allenai/olmo-3.1-32b-instruct
"""

import argparse
import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

EXTRACTION_PROMPT = """\
Extract every distinct refused topic from the AI response below. Return a JSON \
array of short labels (2-5 words each). Preserve specific entities and \
qualifiers — don't generalize to vague categories.

Examples of good labels: "weapons creation", "child exploitation", \
"election manipulation", "drug manufacturing"

Examples of bad labels: "harmful content", "sensitive topics", "illegal things"

Respond with ONLY a JSON array, no other text.

---
AI response:
{response}
---
JSON array:"""


async def extract_topics(client: AsyncOpenAI, model: str, response_text: str) -> dict:
    prompt = EXTRACTION_PROMPT.format(response=response_text)

    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You extract structured data from text. Always respond with valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,
        temperature=0,
    )
    content = completion.choices[0].message.content
    if content is None:
        return {"error": "model returned null content"}
    raw = content.strip()

    # Try to parse JSON from the response (handle markdown fences)
    json_str = raw
    if "```" in json_str:
        # Extract content between first pair of backtick fences
        parts = json_str.split("```")
        if len(parts) >= 3:
            json_str = parts[1]
            # Remove optional language tag like "json"
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()

    try:
        topics = json.loads(json_str)
    except json.JSONDecodeError:
        topics = {"parse_error": raw}

    return topics


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="nvidia/nemotron-3-nano-30b-a3b",
        help="OpenRouter model to use for extraction",
    )
    parser.add_argument(
        "--input",
        default="cleaned-chat.json",
        help="Path to OpenRouter chat export JSON",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        return

    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    with open(args.input) as f:
        chat = json.load(f)

    # Build model name lookup
    model_lookup = {m["characterId"]: m["model"] for m in chat["models"]}

    # Filter to assistant messages only
    assistant_msgs = [m for m in chat["messages"] if m["role"] == "assistant"]

    # Extract topics from each response in parallel
    tasks = []
    for msg in assistant_msgs:
        tasks.append(extract_topics(client, args.model, msg["content"]))

    results = await asyncio.gather(*tasks)

    # Print results
    print(f"\nExtractor model: {args.model}")
    print(f"{'=' * 70}\n")

    for msg, topics in zip(assistant_msgs, results):
        source_model = model_lookup.get(msg["characterId"], "unknown")
        print(f"Source: {source_model}")

        if isinstance(topics, list):
            print(f"Topics ({len(topics)}):")
            for i, t in enumerate(topics, 1):
                print(f"  {i}. {t}")
        else:
            print(f"Error: {topics}")

        print()


if __name__ == "__main__":
    asyncio.run(main())
