import asyncio
import os
import random
import time
from typing import Dict, List, Optional, Union

from src.exceptions import APITimeoutError
from src.transcript_logger import log_model_call

# OpenRouter base URL (canonical form, without trailing slash).
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _apply_nitro(model_id: str, base_url: str, prefer: bool) -> str:
    """Append ':nitro' to *model_id* when all conditions are met.

    Returns *model_id* unchanged when:
    - *prefer* is False, OR
    - *base_url* is not the OpenRouter API endpoint, OR
    - *model_id* already ends in ':nitro' or ':floor'.

    Otherwise returns ``f"{model_id}:nitro"``.
    """
    if not prefer:
        return model_id
    # Normalise trailing slash for comparison
    normalised_url = base_url.rstrip("/")
    if normalised_url != _OPENROUTER_BASE_URL.rstrip("/"):
        return model_id
    if model_id.endswith(":nitro") or model_id.endswith(":floor"):
        return model_id
    return f"{model_id}:nitro"


async def async_query_openrouter(
    model_name: str,
    prompt: str,
    assistant_prefill: str = "",
    system_prompt: str = "",
    verbose: bool = False,
    max_tokens: int = 10000,
    temperature: float = 1.0,
    client_kwargs: Optional[Dict] = None,
    extra_body: Optional[Dict] = None,
    fallback_models: Optional[List[str]] = None,
    default_provider: str = "openrouter",
    provider_url_overrides: Optional[Dict] = None,
    request_timeout_s: float = 120.0,
    request_max_total_s: float = 3600.0,
    prefer_nitro: bool = False,
) -> str:
    """Query any model via an OpenAI-compatible API, with optional fallback chain.

    By default routes to OpenRouter.  Pass *client_kwargs* (with ``api_key``
    and ``base_url``) to target a different provider.

    Args:
        extra_body: Forwarded to ``client.chat.completions.create()``.
            Use for OpenRouter-specific params like ``{"reasoning": {"effort": "none"}}``.
        fallback_models: Optional list of model names to try if the primary
            model fails (timeout, 5xx after retries, or other error).  Each
            fallback is tried in order.  Fallback model strings may include a
            ``provider:`` prefix (e.g. ``openai:gpt-4o``); when the prefix
            resolves to a different provider, a new client is created
            automatically.
        default_provider: Fallback provider when model strings have no prefix.
        provider_url_overrides: Optional per-provider base-URL overrides.
        request_timeout_s: Per-attempt timeout in seconds (default 120).
        request_max_total_s: Wall-clock retry budget in seconds (default 3600).
    """
    from openai import APIStatusError, AsyncOpenAI

    # Let the SDK handle retries (429/5xx) with exponential backoff.
    if client_kwargs is not None:
        client = AsyncOpenAI(**client_kwargs, max_retries=8)
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_retries=8,
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt.strip()})
    if assistant_prefill:
        messages.append({"role": "assistant", "content": assistant_prefill.strip()})

    if verbose:
        print(f"API request: model={model_name}, messages={messages}")

    # Build (client, model_name) pairs — resolve cross-provider fallbacks.
    # Fallback strings with an explicit provider: prefix (e.g. "openai:gpt-4o")
    # are resolved to a new client when the provider differs from the primary.
    # Strings without a prefix reuse the primary client (backward-compatible).
    if client_kwargs is not None:
        primary_base_url = client_kwargs.get("base_url", "")
    else:
        primary_base_url = "https://openrouter.ai/api/v1"

    # Apply nitro transform to primary model
    primary_model_name = _apply_nitro(model_name, primary_base_url, prefer_nitro)
    models_to_try = [(client, primary_model_name)]
    if fallback_models:
        from src.provider_config import (
            BUILTIN_PROVIDERS,
            get_provider_client_kwargs,
        )
        for fb_str in fallback_models:
            # Only resolve cross-provider when string has an explicit prefix
            has_prefix = (
                ":" in fb_str
                and fb_str.split(":", 1)[0].lower() in BUILTIN_PROVIDERS
            )
            if has_prefix:
                fb_model_id, fb_kwargs = get_provider_client_kwargs(
                    fb_str, default_provider, provider_url_overrides,
                )
                fb_base_url = fb_kwargs.get("base_url", "")
                fb_model_id = _apply_nitro(fb_model_id, fb_base_url, prefer_nitro)
                if fb_base_url == primary_base_url:
                    models_to_try.append((client, fb_model_id))
                else:
                    fb_client = AsyncOpenAI(**fb_kwargs, max_retries=8)
                    models_to_try.append((fb_client, fb_model_id))
            else:
                # No explicit provider prefix — use same client (same base_url as primary)
                fb_model_id = _apply_nitro(fb_str, primary_base_url, prefer_nitro)
                models_to_try.append((client, fb_model_id))

    for i, (current_client, current_model) in enumerate(models_to_try):
        is_fallback = i > 0
        if is_fallback:
            print(f"Fallback ({i}/{len(models_to_try)-1}): retrying with model={current_model}")

        deadline = time.monotonic() + request_max_total_s
        backoff = 1.0

        while True:
            try:
                create_kwargs = dict(
                    model=current_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if extra_body is not None:
                    create_kwargs["extra_body"] = extra_body

                completion = await asyncio.wait_for(
                    current_client.chat.completions.create(**create_kwargs),
                    timeout=request_timeout_s,
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

                response = choice.message.content or ""
                if not response:
                    if i < len(models_to_try) - 1:
                        print(f"Empty response from {current_model}, trying next fallback")
                        break

                log_model_call(
                    call_type="async_query_openrouter",
                    model=current_model,
                    inputs=messages,
                    outputs=response,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if verbose:
                    print(f"API response ({current_model}):\n{response}")
                return response
            except APIStatusError as e:
                if e.status_code in (400, 401, 403, 404):
                    raise
                print(
                    f"API error ({current_model}) [status {e.status_code}, retries exhausted]: {e}"
                )
                break  # non-retryable -> next fallback
            except asyncio.TimeoutError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise APITimeoutError(
                        f"API timeout exhausted for {current_model} "
                        f"after {request_max_total_s:.0f}s wall clock"
                    )
                jitter = random.uniform(0, backoff)
                print(
                    f"API timeout ({current_model}) [>{request_timeout_s:.0f}s], "
                    f"retrying in {jitter:.1f}s ({remaining:.0f}s remaining)"
                )
                await asyncio.sleep(jitter)
                backoff = min(backoff * 2, 60.0)
            except Exception as e:
                print(f"API error ({current_model}) [retries exhausted]: {e}")
                break  # try next fallback

    raise APITimeoutError(
        f"All models exhausted (tried: {[m for _, m in models_to_try]})"
    )


# Alias kept for backward compatibility
async_query_llm_api = async_query_openrouter


def query_llm_api(
    model_name: str,
    prompt: Union[str, List[str]],
    assistant_prefill: str = "",
    system_prompt: str = "",
    verbose: bool = False,
    max_tokens: int = 10000,
    client_kwargs: Optional[Dict] = None,
) -> Union[str, List[str]]:
    """Synchronous wrapper around async_query_openrouter for single or batch prompts."""
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
                client_kwargs=client_kwargs,
            )
            for p in prompts
        ]
        return list(await asyncio.gather(*tasks))

    responses = asyncio.run(_run())
    return responses[0] if is_single else responses
