"""Unit tests for S0c — .env must not override shell/CI environment by default.

Tests:
- test_shell_env_wins_over_dotenv_by_default (stub-based, policy flag logic)
- test_dotenv_override_opt_in_via_env_var (stub-based, policy flag logic)
- test_no_override_for_values_other_than_one (stub-based, policy flag logic)
- test_precedence_process_wins_when_no_override (helper-level behavioral)
- test_precedence_dotenv_wins_when_override_set (helper-level behavioral)
- test_precedence_no_existing_env_var_loads_from_dotenv (helper-level behavioral)
"""

import os
import pytest
import dotenv as _dotenv_module

from src.crawler.run_crawler import _apply_dotenv_policy


# ---------------------------------------------------------------------------
# Stub-based tests — verify the policy maps env var to the right override flag
# ---------------------------------------------------------------------------


class TestDotenvPolicy:
    """_apply_dotenv_policy must not clobber existing env vars unless opted in."""

    def test_shell_env_wins_over_dotenv_by_default(self, monkeypatch, tmp_path):
        """Without CRAWLER_DOTENV_OVERRIDE=1, existing env vars must not be overridden."""
        # Set a value in the process environment
        monkeypatch.setenv("TEST_DOTENV_KEY", "process_value")
        # Ensure no override flag
        monkeypatch.delenv("CRAWLER_DOTENV_OVERRIDE", raising=False)

        # Create a .env file with a conflicting value
        dotenv_file = tmp_path / ".env"
        dotenv_file.write_text("TEST_DOTENV_KEY=dotenv_value\n")

        # Patch load_dotenv to capture what override value is used
        captured = {}

        def _fake_load_dotenv(dotenv_path=None, override=False, **kwargs):
            captured["override"] = override
            # Do NOT actually load the file — just record the call

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("src.crawler.run_crawler.load_dotenv", _fake_load_dotenv)
            _apply_dotenv_policy()

        assert captured.get("override") is False, (
            f"load_dotenv must be called with override=False by default, "
            f"got override={captured.get('override')}"
        )
        # The process env value must still be present (not clobbered)
        assert os.environ.get("TEST_DOTENV_KEY") == "process_value"

    def test_dotenv_override_opt_in_via_env_var(self, monkeypatch):
        """With CRAWLER_DOTENV_OVERRIDE=1, load_dotenv is called with override=True."""
        monkeypatch.setenv("CRAWLER_DOTENV_OVERRIDE", "1")

        captured = {}

        def _fake_load_dotenv(dotenv_path=None, override=False, **kwargs):
            captured["override"] = override

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("src.crawler.run_crawler.load_dotenv", _fake_load_dotenv)
            _apply_dotenv_policy()

        assert captured.get("override") is True, (
            f"load_dotenv must be called with override=True when CRAWLER_DOTENV_OVERRIDE=1, "
            f"got override={captured.get('override')}"
        )

    def test_no_override_for_values_other_than_one(self, monkeypatch):
        """CRAWLER_DOTENV_OVERRIDE with values other than '1' must not enable override."""
        for bad_value in ["true", "True", "yes", "1 ", "2", ""]:
            monkeypatch.setenv("CRAWLER_DOTENV_OVERRIDE", bad_value)

            captured = {}

            def _fake_load_dotenv(dotenv_path=None, override=False, **kwargs):
                captured["override"] = override

            with pytest.MonkeyPatch().context() as mp:
                mp.setattr("src.crawler.run_crawler.load_dotenv", _fake_load_dotenv)
                _apply_dotenv_policy()

            assert captured.get("override") is False, (
                f"override must be False for CRAWLER_DOTENV_OVERRIDE={bad_value!r}, "
                f"got override={captured.get('override')}"
            )


# ---------------------------------------------------------------------------
# Helper-level behavioral tests — call _apply_dotenv_policy() directly
# ---------------------------------------------------------------------------


class TestDotenvPrecedenceBehavioral:
    """Prove that process env beats .env (or loses to it) by calling _apply_dotenv_policy().

    find_dotenv() walks up from the caller's source file when usecwd=False (the
    default), so monkeypatch.chdir() alone cannot redirect it to a temp file.
    Instead we patch src.crawler.run_crawler.load_dotenv with a wrapper that
    forwards the override flag (the policy logic under test) but pins dotenv_path
    to our temp file.  This exercises the full policy end-to-end — the only thing
    stubbed is file discovery.

    The behavioral contract being tested:
      - CRAWLER_DOTENV_OVERRIDE unset → override=False → existing process env var survives
      - CRAWLER_DOTENV_OVERRIDE=1    → override=True  → .env file wins, process env var replaced
    """

    _TEST_KEY = "OPENROUTER_API_KEY"

    def _write_dotenv(self, path, value):
        """Write a minimal .env file to path with _TEST_KEY=value."""
        path.write_text(f"{self._TEST_KEY}={value}\n")

    def _make_redirecting_load_dotenv(self, env_file):
        """Return a load_dotenv wrapper that pins dotenv_path to env_file."""
        def _redirected_load_dotenv(override=False, **kwargs):
            return _dotenv_module.load_dotenv(
                dotenv_path=str(env_file), override=override
            )
        return _redirected_load_dotenv

    def test_precedence_process_wins_when_no_override(self, monkeypatch, tmp_path):
        """Existing process env var survives when CRAWLER_DOTENV_OVERRIDE is unset.

        _apply_dotenv_policy() must call load_dotenv(override=False) so that a
        .env file cannot silently clobber a value already set in the shell or CI.
        """
        env_file = tmp_path / ".env"
        self._write_dotenv(env_file, "dotenv_value")

        monkeypatch.setenv(self._TEST_KEY, "process_value")
        monkeypatch.delenv("CRAWLER_DOTENV_OVERRIDE", raising=False)
        monkeypatch.setattr(
            "src.crawler.run_crawler.load_dotenv",
            self._make_redirecting_load_dotenv(env_file),
        )

        _apply_dotenv_policy()

        assert os.environ[self._TEST_KEY] == "process_value", (
            "Process env var must survive when CRAWLER_DOTENV_OVERRIDE is unset; "
            f"got {os.environ.get(self._TEST_KEY)!r}"
        )

    def test_precedence_dotenv_wins_when_override_set(self, monkeypatch, tmp_path):
        """The .env value replaces the process env var when CRAWLER_DOTENV_OVERRIDE=1.

        _apply_dotenv_policy() must pass override=True to load_dotenv in this case,
        allowing the .env file to win over an already-set process env var.
        """
        env_file = tmp_path / ".env"
        self._write_dotenv(env_file, "dotenv_value")

        monkeypatch.setenv(self._TEST_KEY, "process_value")
        monkeypatch.setenv("CRAWLER_DOTENV_OVERRIDE", "1")
        monkeypatch.setattr(
            "src.crawler.run_crawler.load_dotenv",
            self._make_redirecting_load_dotenv(env_file),
        )

        _apply_dotenv_policy()

        assert os.environ[self._TEST_KEY] == "dotenv_value", (
            "The .env value must win when CRAWLER_DOTENV_OVERRIDE=1; "
            f"got {os.environ.get(self._TEST_KEY)!r}"
        )

    def test_precedence_no_existing_env_var_loads_from_dotenv(self, monkeypatch, tmp_path):
        """When the key is absent from process env, _apply_dotenv_policy() always sets it.

        This is a sanity check that the .env file is actually being read.
        override=False still sets variables that are not already present.
        """
        env_file = tmp_path / ".env"
        self._write_dotenv(env_file, "dotenv_value")

        monkeypatch.delenv(self._TEST_KEY, raising=False)
        monkeypatch.delenv("CRAWLER_DOTENV_OVERRIDE", raising=False)
        monkeypatch.setattr(
            "src.crawler.run_crawler.load_dotenv",
            self._make_redirecting_load_dotenv(env_file),
        )

        _apply_dotenv_policy()

        assert os.environ[self._TEST_KEY] == "dotenv_value", (
            "load_dotenv must set the key when it is absent from process env; "
            f"got {os.environ.get(self._TEST_KEY)!r}"
        )
