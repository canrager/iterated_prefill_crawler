from typing import List, Union
import asyncio
import os


async def async_query_openrouter(
    model_name: str,
    prompt: str,
    assistant_prefill: str = "",
    system_prompt: str = "",
    verbose: bool = False,
    max_tokens: int = 10000,
    temperature: float = 1.0,
) -> str:
    """Query any model via the OpenRouter API (OpenAI-compatible)."""
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt.strip()})
    if assistant_prefill:
        messages.append({"role": "assistant", "content": assistant_prefill.strip()})

    if verbose:
        print(f"OpenRouter request: model={model_name}, messages={messages}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response = completion.choices[0].message.content
            if verbose:
                print(f"OpenRouter response:\n{response}")
            return response
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(5)

    return ""


# Alias kept for backward compatibility
async_query_llm_api = async_query_openrouter


def query_llm_api(
    model_name: str,
    prompt: Union[str, List[str]],
    assistant_prefill: str = "",
    system_prompt: str = "",
    verbose: bool = False,
    max_tokens: int = 10000,
) -> Union[str, List[str]]:
    """Synchronous wrapper: query OpenRouter with one prompt or a batch.

    Args:
        model_name: OpenRouter model ID, e.g. "openai/gpt-4o-mini"
        prompt: Single prompt string or list of prompts for concurrent processing
        assistant_prefill: Optional text to prefill the assistant turn
        system_prompt: Optional system prompt
        verbose: Print request/response details
        max_tokens: Maximum tokens to generate

    Returns:
        Single response string if prompt is a string, list of responses otherwise
    """
    is_single = isinstance(prompt, str)
    prompts = [prompt] if is_single else prompt

    async def _run():
        tasks = [
            async_query_openrouter(
                model_name=model_name,
                prompt=p,
                assistant_prefill=assistant_prefill,
                system_prompt=system_prompt,
                verbose=verbose,
                max_tokens=max_tokens,
            )
            for p in prompts
        ]
        return list(await asyncio.gather(*tasks))

    responses = asyncio.run(_run())
    return responses[0] if is_single else responses


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-4o-mini"

    async def main():
        print(f"Using model: {model}")
        print("=" * 60)

        # Test 1: basic async query
        print("\nTest 1: basic async query")
        response = await async_query_openrouter(
            model_name=model,
            prompt="Reply with exactly: pong",
            verbose=True,
        )
        print(f"Result: {repr(response)}")

        # Test 2: assistant prefilling
        # Send an incomplete assistant turn; if prefilling works the model
        # continues from the injected text rather than restarting.
        print("\nTest 2: assistant prefilling")
        prefill = "The capital of France is"
        response_pf = await async_query_openrouter(
            model_name=model,
            prompt="What is the capital of France?",
            assistant_prefill=prefill,
            max_tokens=20,
            verbose=True,
        )
        full = prefill + " " + response_pf.lstrip()
        print(f"Prefill : {repr(prefill)}")
        print(f"Continuation: {repr(response_pf)}")
        print(f"Combined: {repr(full)}")

        # Heuristic: if the response starts with "Paris" or " Paris" the model
        # continued from the prefill.  If it restarts with "The capital …" the
        # prefill was likely ignored (some providers strip it server-side).
        lowered = response_pf.strip().lower()
        if lowered.startswith("paris") or lowered.startswith(" paris"):
            print("=> Prefilling appears to work (model continued from injected text)")
        elif "the capital" in lowered[:40] or "france" in lowered[:40]:
            print("=> Prefilling appears NOT to be applied (model restarted the answer)")
        else:
            print("=> Ambiguous — inspect the continuation above manually")

    asyncio.run(main())
