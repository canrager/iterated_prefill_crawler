"""Tests for the multi-provider routing module."""

import os
import pytest

from src.provider_config import (
    BUILTIN_PROVIDERS,
    collect_required_api_keys,
    get_provider_client_kwargs,
    parse_model_string,
    resolve_provider,
)


# ---------------------------------------------------------------------------
# parse_model_string
# ---------------------------------------------------------------------------

class TestParseModelString:
    def test_openai_prefix(self):
        assert parse_model_string("openai:gpt-4o") == ("openai", "gpt-4o")

    def test_ollama_prefix(self):
        assert parse_model_string("ollama:llama3") == ("ollama", "llama3")

    def test_lmstudio_prefix(self):
        assert parse_model_string("lmstudio:deepseek-r1") == ("lmstudio", "deepseek-r1")

    def test_openrouter_prefix(self):
        assert parse_model_string("openrouter:anthropic/claude-3.5-haiku") == (
            "openrouter",
            "anthropic/claude-3.5-haiku",
        )

    def test_no_prefix_defaults_to_openrouter(self):
        assert parse_model_string("anthropic/claude-3.5-haiku") == (
            "openrouter",
            "anthropic/claude-3.5-haiku",
        )

    def test_no_prefix_bare_model(self):
        assert parse_model_string("gpt-4o") == ("openrouter", "gpt-4o")

    def test_custom_default_provider(self):
        assert parse_model_string("gpt-4o", default_provider="openai") == (
            "openai",
            "gpt-4o",
        )

    def test_unknown_prefix_treated_as_model_id(self):
        """A colon with an unknown prefix is kept as part of the model ID."""
        assert parse_model_string("unknown:some-model") == (
            "openrouter",
            "unknown:some-model",
        )

    def test_case_insensitive_prefix(self):
        assert parse_model_string("OpenAI:gpt-4o") == ("openai", "gpt-4o")

    def test_case_insensitive_default_provider(self):
        provider, model = parse_model_string("gpt-4o", default_provider="OpenAI")
        assert provider == "openai"


# ---------------------------------------------------------------------------
# resolve_provider
# ---------------------------------------------------------------------------

class TestResolveProvider:
    def test_openrouter_needs_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            resolve_provider("openrouter")

    def test_openrouter_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        base_url, api_key = resolve_provider("openrouter")
        assert base_url == "https://openrouter.ai/api/v1"
        assert api_key == "test-key"

    def test_openai_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        base_url, api_key = resolve_provider("openai")
        assert base_url == "https://api.openai.com/v1"
        assert api_key == "sk-test"

    def test_ollama_no_key_needed(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        base_url, api_key = resolve_provider("ollama")
        assert base_url == "http://localhost:11434/v1"
        assert api_key == "not-needed"

    def test_lmstudio_no_key_needed(self):
        base_url, api_key = resolve_provider("lmstudio")
        assert base_url == "http://localhost:1234/v1"
        assert api_key == "not-needed"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            resolve_provider("unknown_provider")

    def test_url_override(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        base_url, _ = resolve_provider(
            "ollama", provider_url_overrides={"ollama": "http://my-box:11434/v1"}
        )
        assert base_url == "http://my-box:11434/v1"

    def test_url_override_case_insensitive(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        base_url, _ = resolve_provider(
            "ollama", provider_url_overrides={"Ollama": "http://my-box:11434/v1"}
        )
        assert base_url == "http://my-box:11434/v1"

    def test_case_insensitive_provider_name(self, monkeypatch):
        base_url, _ = resolve_provider("Ollama")
        assert base_url == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# get_provider_client_kwargs
# ---------------------------------------------------------------------------

class TestGetProviderClientKwargs:
    def test_openai_prefixed(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        model_id, kwargs = get_provider_client_kwargs("openai:gpt-4o")
        assert model_id == "gpt-4o"
        assert kwargs["base_url"] == "https://api.openai.com/v1"
        assert kwargs["api_key"] == "sk-test"

    def test_unprefixed_defaults_to_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
        model_id, kwargs = get_provider_client_kwargs("anthropic/claude-3.5-haiku")
        assert model_id == "anthropic/claude-3.5-haiku"
        assert kwargs["base_url"] == "https://openrouter.ai/api/v1"

    def test_ollama_with_url_override(self):
        model_id, kwargs = get_provider_client_kwargs(
            "ollama:llama3",
            provider_url_overrides={"ollama": "http://remote:11434/v1"},
        )
        assert model_id == "llama3"
        assert kwargs["base_url"] == "http://remote:11434/v1"


# ---------------------------------------------------------------------------
# collect_required_api_keys
# ---------------------------------------------------------------------------

class TestCollectRequiredApiKeys:
    def test_local_skipped(self):
        assert collect_required_api_keys(["local"]) == {}

    def test_ollama_no_key_needed(self):
        assert collect_required_api_keys(["ollama:llama3"]) == {}

    def test_openrouter_missing(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        missing = collect_required_api_keys(["anthropic/claude-3.5-haiku"])
        assert "openrouter" in missing
        assert missing["openrouter"] == "OPENROUTER_API_KEY"

    def test_openai_missing(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        missing = collect_required_api_keys(["openai:gpt-4o"])
        assert "openai" in missing

    def test_mixed_providers(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "ok")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        missing = collect_required_api_keys([
            "anthropic/claude-3.5-haiku",  # openrouter — key set
            "openai:gpt-4o",               # openai — key missing
            "ollama:llama3",               # no key needed
        ])
        assert "openai" in missing
        assert "openrouter" not in missing
        assert "ollama" not in missing

    def test_all_keys_present(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "ok")
        monkeypatch.setenv("OPENAI_API_KEY", "ok")
        missing = collect_required_api_keys([
            "anthropic/claude-3.5-haiku",
            "openai:gpt-4o",
        ])
        assert missing == {}
