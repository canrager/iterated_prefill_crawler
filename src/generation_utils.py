import asyncio
import logging
import random
import time
from typing import Dict, List, Optional, Tuple, Union

from src.crawler.config import ModelConfig

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt


# httpx schedules TLS teardown tasks that fire after asyncio.run() closes the loop,
# producing spurious "Event loop is closed" RuntimeError noise. Filter it globally.
class _SuppressEventLoopClosed(logging.Filter):
    def filter(self, record):
        return "Event loop is closed" not in record.getMessage()


logging.getLogger("asyncio").addFilter(_SuppressEventLoopClosed())

from src.directory_config import INPUT_DIR
from src.transcript_logger import log_model_call
from src.openrouter_utils import (  # re-exported for backward compatibility
    async_query_llm_api,
    async_query_openrouter,
    query_llm_api,
)
from src.provider_config import get_provider_client_kwargs, parse_model_string
from src.tokenization_utils import encode_for_generation


def batch_generate_from_tokens_vllm(
    model: LLM,
    tokenizer: AutoTokenizer,
    input_ids_BL: List[List[int]],
    max_generation_length: int = 1000,
    max_new_tokens: Optional[int] = None,
    skip_special_tokens: bool = False,
    temperature: Optional[float] = None,
    verbose: bool = False,
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
        max_tokens=(
            max_new_tokens if max_new_tokens is not None else max_generation_length
        ),
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


API_MODERATION_SENTINEL = "__API_MODERATION_REFUSED__"
# Backward-compatible alias
OPENROUTER_MODERATION_SENTINEL = API_MODERATION_SENTINEL


from src.exceptions import APITimeoutError  # noqa: F811 — re-export for backward compat

async def _async_api_single(
    client,
    model_name: str,
    messages: List[Dict],
    max_new_tokens: int,
    temperature: float,
    timeout: float = 120.0,
    request_max_total_s: float = 3600.0,
    extra_body: Optional[Dict] = None,
    fallback_specs: Optional[List[Tuple]] = None,
) -> str:
    """Send a single chat conversation to an OpenAI-compatible API and return the response text.

    On ``asyncio.TimeoutError``, retries with exponential backoff (factor 2,
    jitter) until *request_max_total_s* wall-clock time is exceeded, then raises
    ``APITimeoutError``.

    Args:
        timeout: Per-attempt timeout in seconds.
        request_max_total_s: Wall-clock budget for retries (default 3600s = 1h).
        extra_body: Forwarded to ``client.chat.completions.create()``.
            Use for OpenRouter-specific params like ``{"reasoning": {"effort": "none"}}``.
        fallback_specs: Optional list of ``(client, model_id)`` tuples for
            fallback models.  Each fallback can use a different client,
            enabling cross-provider fallback.  Resolved by the caller
            (e.g. ``_api_batch_generate``).
    """
    from openai import APIStatusError

    # Guard: some providers return HTTP 400 "Input must have at least 1 token"
    # when any message has empty content. This can happen when
    # remove_thinking_context() returns "" for an incomplete <think> rollout
    # that was truncated by max_new_tokens.
    if any(not str(m.get("content") or "").strip() for m in messages):
        print(
            f"Skipping API call for {model_name}: one or more messages have empty content"
        )
        return ""

    models_to_try = [(client, model_name)] + (fallback_specs or [])
    for i, (current_client, current_model) in enumerate(models_to_try):
        is_fallback = i > 0
        if is_fallback:
            print(f"Fallback ({i}/{len(models_to_try)-1}): retrying with model={current_model}")

        deadline = time.monotonic() + request_max_total_s
        backoff = 1.0  # initial backoff in seconds

        while True:
            try:
                create_kwargs = dict(
                    model=current_model,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
                if extra_body is not None:
                    create_kwargs["extra_body"] = extra_body

                completion = await asyncio.wait_for(
                    current_client.chat.completions.create(**create_kwargs),
                    timeout=timeout,
                )

                if not completion.choices:
                    print(f"API returned no choices ({current_model})")
                    break  # try next fallback

                choice = completion.choices[0]
                if choice.message is None:
                    finish_reason = getattr(choice, "finish_reason", "unknown")
                    print(
                        f"API returned choice with no message ({current_model}). Finish reason: {finish_reason}"
                    )
                    break

                content = choice.message.content or ""
                reasoning = (
                    getattr(choice.message, "reasoning", None)
                    or getattr(choice.message, "reasoning_content", None)
                    or ""
                )
                finish_reason = getattr(choice, "finish_reason", "unknown")

                if not content and reasoning:
                    if finish_reason == "length":
                        print(
                            "API response exhausted max_tokens in reasoning before any visible "
                            f"answer content ({current_model}). Increase max_new_tokens or disable "
                            "reasoning for this provider if supported."
                        )
                    else:
                        print(
                            f"API returned reasoning but no visible answer content ({current_model}). "
                            f"Finish reason: {finish_reason}"
                        )

                if not content:
                    if i < len(models_to_try) - 1:
                        print(f"Empty response from {current_model}, trying next fallback")
                        break

                return content
            except APIStatusError as e:
                if e.status_code == 403 and "moderation" in str(e.message).lower():
                    reasons = (
                        e.body.get("error", {}).get("metadata", {}).get("reasons", [])
                        if isinstance(e.body, dict)
                        else []
                    )
                    reason_str = ", ".join(reasons) if reasons else "unknown"
                    print(f"API moderation refusal ({current_model}): {reason_str}")
                    return f"{API_MODERATION_SENTINEL}: {reason_str}"
                if e.status_code in (400, 401, 403, 404):
                    raise
                print(
                    f"API error ({current_model}) [status {e.status_code}, retries exhausted]: {e}"
                )
                break  # non-retryable server error -> next fallback
            except asyncio.TimeoutError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print(
                        f"API timeout budget exhausted for {current_model} "
                        f"after {request_max_total_s:.0f}s — trying next fallback"
                    )
                    break  # try next model in fallback chain
                jitter = random.uniform(0, backoff)
                print(
                    f"API timeout ({current_model}) [>{timeout:.0f}s], "
                    f"retrying in {jitter:.1f}s ({remaining:.0f}s remaining)"
                )
                await asyncio.sleep(jitter)
                backoff = min(backoff * 2, 60.0)
            except Exception as e:
                print(f"API error ({current_model}) [retries exhausted]: {e}")
                break  # try next fallback

    # All models/fallbacks exhausted
    raise APITimeoutError(
        f"All models exhausted (tried: {[m for _, m in models_to_try]})"
    )




def _api_batch_generate(
    model_name: str,
    messages: List[List[Dict]],
    max_new_tokens: int,
    temperature: float,
    verbose: bool = False,
    default_provider: str = "openrouter",
    provider_url_overrides: Optional[Dict[str, str]] = None,
    provider_concurrency_limits: Optional[Dict[str, int]] = None,
    extra_body: Optional[Dict] = None,
    fallback_models: Optional[List[str]] = None,
    request_max_total_s: float = 3600.0,
    prefer_nitro: bool = False,
) -> Tuple[List[str], List[str]]:
    """Send a batch of chat conversations to an OpenAI-compatible API concurrently.

    The *model_name* may include a ``provider:`` prefix (e.g.
    ``openai:gpt-4o``).  When absent the *default_provider* is used.

    Returns:
        Tuple of (generated_texts, input_strs) where input_strs are reconstructed from messages.
    """
    from openai import AsyncOpenAI
    from src.openrouter_utils import _apply_nitro

    provider_name, _ = parse_model_string(model_name, default_provider)
    resolved_model_id, client_kwargs = get_provider_client_kwargs(
        model_name,
        default_provider,
        provider_url_overrides,
    )

    # Apply nitro transform to the primary model
    primary_base_url = client_kwargs.get("base_url", "")
    resolved_model_id = _apply_nitro(resolved_model_id, primary_base_url, prefer_nitro)

    # The SDK auto-retries 429/500/502/503/504 with exponential backoff.
    # 8 retries: backoff caps at ~60s, total wait up to ~2 min per request.
    # This is enough to ride out a full OpenRouter rate-limit window when
    # firing large concurrent batches (e.g. 300+ judge calls at once).
    client = AsyncOpenAI(**client_kwargs, max_retries=8)

    # Resolve fallback models to (client, model_id) pairs, creating new
    # clients when a fallback targets a different provider.
    fallback_specs = None
    if fallback_models:
        fallback_specs = []
        for fb_str in fallback_models:
            fb_provider, _ = parse_model_string(fb_str, default_provider)
            fb_model_id, fb_kwargs = get_provider_client_kwargs(
                fb_str, default_provider, provider_url_overrides,
            )
            fb_base_url = fb_kwargs.get("base_url", "")
            fb_model_id = _apply_nitro(fb_model_id, fb_base_url, prefer_nitro)
            if fb_provider == provider_name:
                fallback_specs.append((client, fb_model_id))
            else:
                fb_client = AsyncOpenAI(**fb_kwargs, max_retries=8)
                fallback_specs.append((fb_client, fb_model_id))

    # Default concurrency caps per provider to avoid flooding rate limits.
    # These can be overridden via provider_max_concurrency in the model config.
    _DEFAULT_CONCURRENCY: Dict[str, int] = {
        "openrouter": 50,
    }

    max_concurrency: Optional[int] = None
    if provider_concurrency_limits:
        normalized_limits = {
            str(provider).lower(): int(limit)
            for provider, limit in provider_concurrency_limits.items()
            if limit is not None
        }
        configured_limit = normalized_limits.get(provider_name.lower())
        if configured_limit and configured_limit > 0:
            max_concurrency = configured_limit

    # Fall back to default cap if no explicit override
    if max_concurrency is None:
        max_concurrency = _DEFAULT_CONCURRENCY.get(provider_name.lower())

    async def _run():
        async def _single(msg_list: List[Dict]) -> str:
            return await _async_api_single(
                client, resolved_model_id, msg_list, max_new_tokens, temperature,
                extra_body=extra_body, fallback_specs=fallback_specs,
                request_max_total_s=request_max_total_s,
            )

        if max_concurrency is None:
            tasks = [_single(msg_list) for msg_list in messages]
        else:
            semaphore = asyncio.Semaphore(max_concurrency)

            async def _single_limited(msg_list: List[Dict]) -> str:
                async with semaphore:
                    return await _single(msg_list)

            tasks = [_single_limited(msg_list) for msg_list in messages]
        return list(await asyncio.gather(*tasks))

    texts = asyncio.run(_run())

    log_model_call(
        call_type="batch_generate_api",
        model=resolved_model_id,
        inputs=messages,
        outputs=texts,
        temperature=temperature,
        max_tokens=max_new_tokens,
        batch_size=len(messages),
    )

    # Reconstruct input_strs from messages (join all content fields)
    input_strs = [" ".join(m["content"] for m in msg_list) for msg_list in messages]

    if verbose:
        for input_str, output in zip(input_strs, texts):
            print(
                f"===========================\n====input: {input_str}\n\n==== output:\n {output}\n\n"
            )

    return texts, input_strs


def batch_generate(
    model,
    tokenizer,
    messages: List[List[Dict]],
    max_new_tokens: int = 150,
    temperature: float = 0.6,
    verbose: bool = False,
    skip_special_tokens: bool = False,
    default_provider: str = "openrouter",
    provider_url_overrides: Optional[Dict[str, str]] = None,
    provider_concurrency_limits: Optional[Dict[str, int]] = None,
    extra_body: Optional[Dict] = None,
    fallback_models: Optional[List[str]] = None,
    request_max_total_s: float = 3600.0,
    prefer_nitro: bool = False,
) -> Tuple[List[str], List[str]]:
    """Generate text from a list of message dicts.

    Dispatches to an OpenAI-compatible API when *model* is a ``str``, or to
    vLLM when *model* is an ``LLM`` instance.

    The model string may include a ``provider:`` prefix (e.g.
    ``openai:gpt-4o``, ``ollama:llama3``).  When absent the
    *default_provider* is used (defaults to ``"openrouter"``).

    Args:
        model: vLLM LLM instance, or model ID string (with optional provider prefix)
        tokenizer: HuggingFace tokenizer (ignored when model is a str)
        messages: List of message lists, each in OpenAI chat format.
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (None → greedy)
        verbose: Print input/output pairs
        skip_special_tokens: Skip special tokens when decoding outputs
        default_provider: Fallback provider when model string has no prefix
        provider_url_overrides: Optional ``{provider: url}`` overrides

    Returns:
        Tuple of (generated_texts, input_strs)
    """
    if isinstance(model, str):
        return _api_batch_generate(
            model_name=model,
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=verbose,
            default_provider=default_provider,
            provider_url_overrides=provider_url_overrides,
            provider_concurrency_limits=provider_concurrency_limits,
            extra_body=extra_body,
            fallback_models=fallback_models,
            request_max_total_s=request_max_total_s,
            prefer_nitro=prefer_nitro,
        )

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
    )

    log_model_call(
        call_type="batch_generate_vllm",
        model=getattr(model, "model", "vllm_local"),
        inputs=input_strs,
        outputs=generated_texts,
        temperature=temperature,
        max_tokens=max_new_tokens,
        batch_size=len(messages),
    )

    if verbose:
        for input_str, output in zip(input_strs, generated_texts):
            print(
                f"===========================\n====input: {input_str}\n\n==== output:\n {output}\n\n"
            )

    return generated_texts, input_strs


async def async_summarize_single_topic(
    topic_raw: str,
    llm_judge_name: str,
    system_prompt: str,
    verbose: bool = False,
    client_kwargs: Optional[Dict] = None,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Async function to summarize a single topic.

    Returns:
        Tuple of (topic_raw, summary, error_message)
    """
    from src.crawler.config import TOPIC_SUMMARIZATION_PROMPT

    content_prompt = TOPIC_SUMMARIZATION_PROMPT.format(topic_raw=topic_raw)

    try:
        summary = await async_query_openrouter(
            model_name=llm_judge_name,
            system_prompt=system_prompt,
            prompt=content_prompt,
            verbose=verbose,
            client_kwargs=client_kwargs,
            temperature=0.6,
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
    client_kwargs: Optional[Dict] = None,
) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Batch summarize multiple topics concurrently with rate limiting.

    Args:
        topics_raw: List of raw topic strings to summarize
        llm_judge_name: Name of the LLM model to use
        system_prompt: System prompt for the LLM
        max_concurrent: Maximum number of concurrent requests
        verbose: Whether to print debug information
        client_kwargs: Optional dict with ``api_key`` and ``base_url``

    Returns:
        List of tuples: (topic_raw, summary, error_message)
    """
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def rate_limited_summarize(topic_raw: str):
        async with semaphore:
            return await async_summarize_single_topic(
                topic_raw,
                llm_judge_name,
                system_prompt,
                verbose,
                client_kwargs=client_kwargs,
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
