from typing import Dict, List, Optional, Tuple, Union
import time
import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import anthropic
import os
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

from src.tokenization_utils_new import encode_for_generation
from src.directory_config import INPUT_DIR


def batch_generate_from_tokens_vllm(
    model: LLM,
    tokenizer: AutoTokenizer,
    input_ids_BL: List[List[int]],
    max_generation_length: int = 1000,
    max_new_tokens: Optional[int] = None,
    skip_special_tokens: bool = False,
    temperature: Optional[float] = None,
    verbose: bool = False,
    cfg=None,
):
    """
    Generate text using vLLM backend.

    Args:
        model: vLLM LLM instance
        tokenizer: HuggingFace tokenizer
        input_ids_BL: List of input token lists (variable length, no padding needed)
        max_generation_length: Maximum total sequence length (ignored if max_new_tokens is set)
        max_new_tokens: Maximum number of new tokens to generate
        skip_special_tokens: Whether to skip special tokens in decoding
        temperature: Sampling temperature (None = greedy, converted to 0.0)
        verbose: Print debug information

    Returns:
        List[str]: Generated texts
    """
    # Convert None temperature to greedy (0.0)
    if temperature is None:
        temperature = 0.0

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=(max_new_tokens if max_new_tokens is not None else max_generation_length),
        skip_special_tokens=skip_special_tokens,
    )

    # vLLM handles variable-length sequences natively - no padding needed!
    # Wrap token IDs in TokensPrompt format for vLLM 0.11.0+
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in input_ids_BL]

    outputs = model.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )

    # Extract generated text from vLLM outputs
    generated_texts = [output.outputs[0].text for output in outputs]

    if verbose:
        for i, (input_ids, output) in enumerate(zip(input_ids_BL, outputs)):
            print("====================")
            print(f"Input tokens: {input_ids}")
            print(f"Generated: {output.outputs[0].text}")

    return generated_texts


def batch_generate(
    model: LLM,
    tokenizer: AutoTokenizer,
    messages: List[List[Dict]],
    max_new_tokens: int = 150,
    temperature: float = 0.6,
    verbose: bool = False,
    skip_special_tokens: bool = False,
    cfg=None,
) -> Tuple[List[str], List[str]]:
    """Generate text from a list of message dicts using the vLLM backend.

    Args:
        model: vLLM LLM instance
        tokenizer: HuggingFace tokenizer
        messages: List of message lists, each in OpenAI chat format.
                  Each inner list has at most two messages:
                  - [{"role":"user","content":...}]
                  - [{"role":"user","content":...}, {"role":"assistant","content":...}]
                  The assistant content is used as prefill (thought or assistant,
                  depending on cfg.prefill_mode).
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (None → greedy)
        verbose: Print input/output pairs
        skip_special_tokens: Skip special tokens when decoding outputs
        cfg: Config object with model_path and prefill_mode attributes

    Returns:
        Tuple of (generated_texts, input_strs)
    """
    input_ids, input_strs = encode_for_generation(
        tokenizer=tokenizer,
        messages=messages,
    )

    generated_texts = batch_generate_from_tokens_vllm(
        model=model,
        tokenizer=tokenizer,
        input_ids_BL=input_ids,
        max_generation_length=None,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        skip_special_tokens=skip_special_tokens,
        verbose=False,
        cfg=cfg,
    )

    if verbose:
        for input_str, output in zip(input_strs, generated_texts):
            print(f"===========================\n====input: {input_str}\n\n==== output:\n {output}\n\n")

    return generated_texts, input_strs


def query_llm_api(
    model_name: str,
    prompt: Union[str, List[str]],
    assistant_prefill: str = "",
    system_prompt: str = "",
    verbose: bool = False,
    max_tokens: int = 10000,
) -> Union[str, List[str]]:
    """Query LLM API with prompt caching enabled and retry logic.

    Args:
        model_name: Name of the model to query
        prompt: Single prompt string or list of prompts for batched processing
        assistant_prefill: Assistant prefill text
        system_prompt: System prompt text
        verbose: Whether to print verbose output
        max_tokens: Maximum number of tokens to generate

    Returns:
        Single response string if prompt is a string, list of responses if prompt is a list
    """
    # Handle single prompt - convert to list for uniform processing
    is_single_prompt = isinstance(prompt, str)
    prompts = [prompt] if is_single_prompt else prompt

    # Run async batch processing
    responses = asyncio.run(
        _batch_query_llm_api(
            model_name=model_name,
            prompts=prompts,
            assistant_prefill=assistant_prefill,
            system_prompt=system_prompt,
            verbose=verbose,
            max_tokens=max_tokens,
        )
    )

    # Return single response if single prompt was provided
    return responses[0] if is_single_prompt else responses


async def _batch_query_llm_api(
    model_name: str,
    prompts: List[str],
    assistant_prefill: str = "",
    system_prompt: str = "",
    verbose: bool = False,
    max_tokens: int = 10000,
) -> List[str]:
    """Internal async function to process batch of prompts concurrently"""
    tasks = []
    for prompt in prompts:
        task = async_query_llm_api(
            model_name=model_name,
            prompt=prompt,
            assistant_prefill=assistant_prefill,
            system_prompt=system_prompt,
            verbose=verbose,
            max_tokens=max_tokens,
        )
        tasks.append(task)

    # Execute all queries concurrently
    responses = await asyncio.gather(*tasks)
    return responses


def query_anthropic(
    prompt: str,
    api_key: str,
    llm_judge_name: str,
    system_prompt: str = "",
    assistant_prefill: str = "",
    verbose: bool = False,
    max_tokens: int = 1000,
    temperature: float = 1,
) -> str:
    """Query Anthropic's Claude model with prompt caching enabled and retry logic"""
    client = anthropic.Client(
        api_key=api_key,
        # Enable prompt caching beta feature
        default_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )

    message_args = {}
    message_args["messages"] = [{"role": "user", "content": prompt.strip()}]
    if assistant_prefill != "":
        message_args["messages"].append({"role": "assistant", "content": assistant_prefill.strip()})
    if system_prompt != "":
        message_args["system"] = [
            {
                "type": "text",
                "text": system_prompt.strip(),
                "cache_control": {"type": "ephemeral"},
            }
        ]

    if verbose:
        print(f"Message args: {message_args}")

    max_retries = 3
    for attempt in range(max_retries):
        if verbose:
            print(f"\nAttempt {attempt + 1}/{max_retries}")
            print("-" * 40)

        try:
            message = client.messages.create(
                model=llm_judge_name,
                max_tokens=max_tokens,
                temperature=temperature,
                **message_args,
            )

            response = message.content[0].text
            if verbose:
                print("RESPONSE:")
                print("-" * 40)
                print(response)
                print("-" * 40)

            time.sleep(1)
            return response
        except Exception as e:
            print(f"Anthropic API error: {e}")
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                if verbose:
                    print(f"Retrying in 60 seconds...")
                time.sleep(60)
            continue

    if verbose:
        print(f"\nFailed to get valid response after {max_retries} attempts")
    return ""


def query_openai(
    prompt: str,
    api_key: str,
    model_name: str,
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    max_tokens: int = 10000,
    temperature: float = 1,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    max_retries = 3
    for attempt in range(max_retries):
        if verbose:
            print(f"\nAttempt {attempt + 1}/{max_retries}")
            print("-" * 40)

        try:
            message = client.chat.completions.create(
                model=model_name,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            response = message.choices[0].message.content
            if verbose:
                print("RESPONSE:")
                print("-" * 40)
                print(response)
                print("-" * 40)

            time.sleep(1)
            return response
        except Exception as e:
            print(f"OpenAI API error: {e}")
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                if verbose:
                    print(f"Retrying in 60 seconds...")
                time.sleep(60)
            continue

    if verbose:
        print(f"\nFailed to get valid response after {max_retries} attempts")
    return ""


def query_grok(
    prompt: str,
    api_key: str,
    model_name: str,
    system_prompt: Optional[str] = None,
    assistant_prefill: str = "",  # Grok API might not support this, will check
    verbose: bool = False,
    max_tokens: int = 10000,
    temperature: float = 1,
) -> str:
    from openai import OpenAI  # Grok uses OpenAI compatible API

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )

    max_retries = 3
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    # Assistant prefill is not standard in OpenAI's chat completion,
    # and likely not for Grok's compatible API.
    # If it were supported, it might look like:
    # if assistant_prefill:
    #     messages.append({"role": "assistant", "content": assistant_prefill})

    for attempt in range(max_retries):
        if verbose:
            print(f"\nAttempt {attempt + 1}/{max_retries} for Grok API")
            print("-" * 40)
            print(f"Messages: {messages}")
            print("-" * 40)

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            response = completion.choices[0].message.content
            if verbose:
                print("GROK RESPONSE:")
                print("-" * 40)
                print(response)
                print("-" * 40)

            time.sleep(1)  # Rate limiting
            return response
        except Exception as e:
            print(f"Grok API error: {e}")
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                if verbose:
                    print(f"Retrying in 60 seconds...")
                time.sleep(60)
            continue

    if verbose:
        print(f"\nFailed to get valid response from Grok after {max_retries} attempts")
    return ""


# ============================================================================
# Async API Query Functions for Batch Processing
# ============================================================================


async def async_query_openai(
    prompt: str,
    api_key: str,
    model_name: str,
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    max_tokens: int = 10000,
    temperature: float = 1,
) -> str:
    """Async version of query_openai for concurrent requests"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)

    max_retries = 3
    for attempt in range(max_retries):
        if verbose:
            print(f"\nAttempt {attempt + 1}/{max_retries}")
            print("-" * 40)

        try:
            message = await client.chat.completions.create(
                model=model_name,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            response = message.choices[0].message.content
            if verbose:
                print("RESPONSE:")
                print("-" * 40)
                print(response)
                print("-" * 40)

            await asyncio.sleep(0.1)  # Small delay to avoid rate limits
            return response
        except Exception as e:
            print(f"OpenAI API error: {e}")
            if attempt < max_retries - 1:
                if verbose:
                    print(f"Retrying in 5 seconds...")
                await asyncio.sleep(5)
            continue

    if verbose:
        print(f"\nFailed to get valid response after {max_retries} attempts")
    return ""


async def async_query_anthropic(
    prompt: str,
    api_key: str,
    llm_judge_name: str,
    system_prompt: str = "",
    assistant_prefill: str = "",
    verbose: bool = False,
    max_tokens: int = 1000,
    temperature: float = 1,
) -> str:
    """Async version of query_anthropic for concurrent requests"""
    client = anthropic.AsyncAnthropic(
        api_key=api_key,
        default_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )

    message_args = {}
    message_args["messages"] = [{"role": "user", "content": prompt.strip()}]
    if assistant_prefill != "":
        message_args["messages"].append({"role": "assistant", "content": assistant_prefill.strip()})
    if system_prompt != "":
        message_args["system"] = [
            {
                "type": "text",
                "text": system_prompt.strip(),
                "cache_control": {"type": "ephemeral"},
            }
        ]

    if verbose:
        print(f"Message args: {message_args}")

    max_retries = 3
    for attempt in range(max_retries):
        if verbose:
            print(f"\nAttempt {attempt + 1}/{max_retries}")
            print("-" * 40)

        try:
            message = await client.messages.create(
                model=llm_judge_name,
                max_tokens=max_tokens,
                temperature=temperature,
                **message_args,
            )

            response = message.content[0].text
            if verbose:
                print("RESPONSE:")
                print("-" * 40)
                print(response)
                print("-" * 40)

            await asyncio.sleep(0.1)
            return response
        except Exception as e:
            print(f"Anthropic API error: {e}")
            if attempt < max_retries - 1:
                if verbose:
                    print(f"Retrying in 5 seconds...")
                await asyncio.sleep(5)
            continue

    if verbose:
        print(f"\nFailed to get valid response after {max_retries} attempts")
    return ""


async def async_query_grok(
    prompt: str,
    api_key: str,
    model_name: str,
    system_prompt: Optional[str] = None,
    assistant_prefill: str = "",
    verbose: bool = False,
    max_tokens: int = 10000,
    temperature: float = 1,
) -> str:
    """Async version of query_grok for concurrent requests"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )

    max_retries = 3
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        if verbose:
            print(f"\nAttempt {attempt + 1}/{max_retries} for Grok API")
            print("-" * 40)
            print(f"Messages: {messages}")
            print("-" * 40)

        try:
            completion = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            response = completion.choices[0].message.content
            if verbose:
                print("GROK RESPONSE:")
                print("-" * 40)
                print(response)
                print("-" * 40)

            await asyncio.sleep(0.1)
            return response
        except Exception as e:
            print(f"Grok API error: {e}")
            if attempt < max_retries - 1:
                if verbose:
                    print(f"Retrying in 5 seconds...")
                await asyncio.sleep(5)
            continue

    if verbose:
        print(f"\nFailed to get valid response from Grok after {max_retries} attempts")
    return ""


async def async_query_llm_api(
    model_name: str,
    prompt: str,
    assistant_prefill: str = "",
    system_prompt: str = "",
    verbose: bool = False,
    max_tokens: int = 10000,
) -> str:
    """Async dispatcher for LLM API queries"""
    if "claude" in model_name:
        temperature = 1
        with open(os.path.join(INPUT_DIR, "ant.txt"), "r") as f:
            api_key = f.read()
        return await async_query_anthropic(
            prompt,
            api_key,
            model_name,
            system_prompt,
            assistant_prefill,
            verbose,
            max_tokens,
            temperature,
        )
    elif "gpt" in model_name:
        assert assistant_prefill == "", (
            "Assistant prefill is not supported for GPT. Argument assistant_prefill is: "
            + assistant_prefill
        )
        temperature = 1
        api_key = os.environ.get("OPENAI_API_KEY")
        return await async_query_openai(
            prompt, api_key, model_name, system_prompt, verbose, max_tokens, temperature
        )
    elif "grok" in model_name:
        temperature = 1
        with open(os.path.join(INPUT_DIR, "grok.txt"), "r") as f:
            api_key = f.read()
        return await async_query_grok(
            prompt,
            api_key,
            model_name,
            system_prompt,
            assistant_prefill,
            verbose,
            max_tokens,
            temperature,
        )
    else:
        raise ValueError(
            f"Model {model_name} not supported, must contain 'claude', 'gpt', or 'grok'"
        )


async def async_summarize_single_topic(
    topic_raw: str,
    llm_judge_name: str,
    system_prompt: str,
    verbose: bool = False,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Async function to summarize a single topic.

    Returns:
        Tuple of (topic_raw, summary, error_message)
    """
    content_prompt = f"""Given a topic phrase, extract the core concept being addressed. Be specific and concise, using 1-3 keywords.

Examples:
- Input: "suicide or other self-inflicted injuries"
  Output: suicide, self-harm

- Input: "Information on how to obtain firearms or other lethal weapons."
  Output: obtaining weapons

- Input: "Showing signs of increased alcohol or drug use."
  Output: alcohol, drugs

- Input: "User authentication and access control: Ensuring that users are who they claim to be and have appropriate access to the resources they need. This can involve using multi-factor authentication, role-based access control, and regular user account reviews."
  Output: access control

Now extract the core topic from this phrase:
Input: "{topic_raw}"
Output:"""

    try:
        summary = await async_query_llm_api(
            model_name=llm_judge_name,
            system_prompt=system_prompt,
            prompt=content_prompt,
            verbose=verbose,
        )
        summary = summary.strip()

        if verbose:
            print(f"Summarized topic:")
            print(f"  Raw: {topic_raw}")
            print(f"  Summary: {summary}")

        return (topic_raw, summary, None)
    except Exception as e:
        error_msg = f"Error summarizing topic '{topic_raw}': {e}"
        print(error_msg)
        return (topic_raw, None, error_msg)


async def async_batch_summarize_topics(
    topics_raw: List[str],
    llm_judge_name: str,
    system_prompt: str,
    max_concurrent: int = 10,
    verbose: bool = False,
) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Batch summarize multiple topics concurrently with rate limiting.

    Args:
        topics_raw: List of raw topic strings to summarize
        llm_judge_name: Name of the LLM model to use
        system_prompt: System prompt for the LLM
        max_concurrent: Maximum number of concurrent requests
        verbose: Whether to print debug information

    Returns:
        List of tuples: (topic_raw, summary, error_message)
    """
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def rate_limited_summarize(topic_raw: str):
        async with semaphore:
            return await async_summarize_single_topic(
                topic_raw, llm_judge_name, system_prompt, verbose
            )

    # Create tasks for all topics
    tasks = [rate_limited_summarize(topic_raw) for topic_raw in topics_raw]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and handle any exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            error_msg = f"Exception during summarization: {result}"
            print(error_msg)
            processed_results.append((topics_raw[i], None, error_msg))
        else:
            processed_results.append(result)

    return processed_results
