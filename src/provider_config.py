"""Provider registry for multi-provider LLM routing.

Model strings use a ``provider:model_id`` format.  When no prefix is given the
model is routed to the **default provider** (OpenRouter unless overridden in
``ModelConfig.default_provider``).

Supported providers and their default base URLs:

    openrouter  → https://openrouter.ai/api/v1   (env: OPENROUTER_API_KEY)
    openai      → https://api.openai.com/v1       (env: OPENAI_API_KEY)
    ollama      → http://localhost:11434/v1        (env: —, no key needed)
    lmstudio    → http://localhost:1234/v1         (env: —, no key needed)

Base URLs can be overridden per-provider via ``ModelConfig.provider_urls``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Default provider definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderInfo:
    """Immutable definition of a known LLM provider."""
    base_url: str
    env_key: Optional[str] = None          # environment variable for the API key
    requires_api_key: bool = True


BUILTIN_PROVIDERS: Dict[str, ProviderInfo] = {
    "openrouter": ProviderInfo(
        base_url="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
    ),
    "openai": ProviderInfo(
        base_url="https://api.openai.com/v1",
        env_key="OPENAI_API_KEY",
    ),
    "ollama": ProviderInfo(
        base_url="http://localhost:11434/v1",
        env_key=None,
        requires_api_key=False,
    ),
    "lmstudio": ProviderInfo(
        base_url="http://localhost:1234/v1",
        env_key=None,
        requires_api_key=False,
    ),
}

DEFAULT_PROVIDER = "openrouter"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_model_string(
    model_str: str,
    default_provider: str = DEFAULT_PROVIDER,
) -> Tuple[str, str]:
    """Split a model string into ``(provider, model_id)``.

    Examples::

        "openai:gpt-4o"           → ("openai",      "gpt-4o")
        "ollama:llama3"            → ("ollama",      "llama3")
        "lmstudio:deepseek-r1"    → ("lmstudio",    "deepseek-r1")
        "anthropic/claude-3.5-haiku"  → ("openrouter", "anthropic/claude-3.5-haiku")
        "gpt-4o"                   → ("openrouter",  "gpt-4o")

    The colon delimiter is only recognised when the left-hand side matches a
    known provider name so that model IDs containing colons (rare but possible)
    are not mis-parsed.
    """
    if ":" in model_str:
        prefix, rest = model_str.split(":", 1)
        if prefix.lower() in BUILTIN_PROVIDERS:
            return prefix.lower(), rest
    # No recognised prefix → use default provider, full string is the model ID
    return default_provider.lower(), model_str


def resolve_provider(
    provider_name: str,
    provider_url_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """Return ``(base_url, api_key)`` for a given provider.

    Raises ``ValueError`` if the provider requires an API key but the
    corresponding environment variable is unset.  Local providers that
    don't need a key get ``"not-needed"`` as the api_key value.
    """
    provider_name = provider_name.lower()
    info = BUILTIN_PROVIDERS.get(provider_name)
    if info is None:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Known providers: {list(BUILTIN_PROVIDERS)}"
        )

    # Allow per-provider URL overrides from config (case-insensitive keys)
    if provider_url_overrides:
        normalized_overrides = {k.lower(): v for k, v in provider_url_overrides.items()}
        base_url = normalized_overrides.get(provider_name, info.base_url)
    else:
        base_url = info.base_url

    api_key: str = ""
    if info.env_key:
        api_key = os.environ.get(info.env_key, "")
    if info.requires_api_key and not api_key:
        raise ValueError(
            f"Provider '{provider_name}' requires an API key.  "
            f"Set the {info.env_key} environment variable."
        )

    # Local providers (ollama, lmstudio) don't need a real key but the
    # OpenAI client constructor still wants *something*.
    if not api_key:
        api_key = "not-needed"

    return base_url, api_key


def get_provider_client_kwargs(
    model_str: str,
    default_provider: str = DEFAULT_PROVIDER,
    provider_url_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict]:
    """High-level helper: parse a model string and return the model ID plus
    keyword arguments suitable for ``openai.AsyncOpenAI(**kwargs)``.

    Returns ``(model_id, {"api_key": ..., "base_url": ...})``.
    """
    provider_name, model_id = parse_model_string(model_str, default_provider)
    base_url, api_key = resolve_provider(provider_name, provider_url_overrides)
    return model_id, {"api_key": api_key, "base_url": base_url}


def collect_required_api_keys(
    model_strings: list[str],
    default_provider: str = DEFAULT_PROVIDER,
) -> Dict[str, str]:
    """Return a mapping ``{provider_name: env_var_name}`` for every provider
    referenced by *model_strings* that requires an API key.

    Only includes providers whose key is **missing** from the environment so
    callers can raise a single informative error.
    """
    missing: Dict[str, str] = {}
    for ms in model_strings:
        if ms == "local":
            continue
        provider_name, _ = parse_model_string(ms, default_provider)
        info = BUILTIN_PROVIDERS.get(provider_name)
        if info and info.requires_api_key and info.env_key:
            if not os.environ.get(info.env_key):
                missing[provider_name] = info.env_key
    return missing
