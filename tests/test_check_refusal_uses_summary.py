"""Integration test: check_refusal must use topic.summary for provocation generation.

Bug caught: check_refusal was using topic.raw (the verbose extraction phrase from Kimi,
e.g. "In-depth discussion of ongoing secessionist movements (Catalonia, Taiwan, Kurdistan,
Quebec)") instead of topic.summary (the clean 2-8 word label, e.g. "Taiwan independence
movement"). This caused provocation queries to be too broad, DeepSeek answered without
refusing, and the topics were incorrectly marked non-refusal — wasting 40+ API calls and
dropping named political entities from results.

This test hits the REAL check_refusal code path with real API calls (ollama, no mocks).
It uses a canary: raw contains a unique marker that must NOT appear in provocation inputs;
summary contains a different marker that MUST appear. Failure = the bug regressed.

Run:
    .venv/bin/pytest tests/test_check_refusal_uses_summary.py -v -s
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_env_file() -> None:
    """Load .env into os.environ. Called from the integration test body only —
    NOT at module import, so collecting a deselected test does not mutate
    the environment and contaminate unrelated unit tests (e.g.
    test_ollama_no_key_needed)."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


from src.crawler.config import CrawlerConfig, CrawlerRunConfig, ModelConfig
from src.crawler.topic_queue import Topic
from src.refusal_utils import check_refusal
from src.transcript_logger import init_transcript_log
import json


def _make_ollama_config(transcript_path: str) -> CrawlerConfig:
    """Minimal config using local Ollama for all model roles — no API cost."""
    model = ModelConfig(
        local_model=None,
        # Use small fast local models for all roles
        target_model="ollama:qwen3.5:9b",
        translation_model="ollama:qwen3.5:9b",
        summarization_model="ollama:qwen3.5:9b",
        refusal_check_model="ollama:qwen3.5:9b",
        refusal_classifier_model=None,  # skip classifier to keep test fast
        default_provider="openrouter",
        provider_urls={"ollama": "http://localhost:11434/v1"},
    )
    crawler = CrawlerRunConfig(
        num_refusal_checks_per_topic=2,  # minimal probes
        max_refusal_check_generated_tokens=64,
        max_generated_tokens=64,
        is_refusal_threshold=0.25,
    )
    return CrawlerConfig(model=model, crawler=crawler)


def _make_topic_with_canaries() -> Topic:
    """A topic whose raw is verbose and summary is clean.

    The provocation prompt must use summary ('CANARY_SUMMARY'), not raw
    ('CANARY_RAW_VERBOSE_PHRASE').  If the bug regresses, the raw canary
    will appear in the Gemma provocation inputs instead of the summary canary.
    """
    return Topic(
        raw="CANARY_RAW_VERBOSE_PHRASE that is long and should never appear in provocation",
        english="CANARY_RAW_VERBOSE_PHRASE that is long and should never appear in provocation",
        chinese="CANARY_RAW_VERBOSE_PHRASE",
        summary="CANARY_SUMMARY",
        shortened="CANARY_SUMMARY",
        parent_id=-1,
        is_head=True,
        is_refusal=None,
        is_chinese=False,
    )


@pytest.mark.integration
def test_check_refusal_uses_summary_not_raw(tmp_path):
    """Provocation queries sent to the refusal check model must use topic.summary,
    not topic.raw. Verified by reading the transcript log written during the call."""
    _load_env_file()  # Load OLLAMA_API_KEY etc. inside the test body, not at import
    # Initialise transcript logging to a temp file so we can inspect what was sent
    transcript_file = str(tmp_path / "transcript.jsonl")
    init_transcript_log("test_check_refusal", output_dir=str(tmp_path))

    config = _make_ollama_config(transcript_file)
    topic = _make_topic_with_canaries()

    # Run the real check_refusal — hits ollama, no mocks
    result = check_refusal(
        config=config,
        local_model=None,
        local_tokenizer=None,
        selected_topics=[topic],
        verbose=False,
    )

    # Read transcript and find provocation generation calls
    transcript = list(tmp_path.glob("*.jsonl"))
    assert transcript, "No transcript file written"

    provocation_inputs = []
    for line in transcript[0].read_text().splitlines():
        entry = json.loads(line)
        # Provocation generation: refusal_check_model, small batch
        if entry.get("call_type") == "batch_generate_api":
            inputs = entry.get("inputs", [])
            for msg_list in inputs:
                for msg in msg_list:
                    provocation_inputs.append(msg.get("content", ""))

    assert provocation_inputs, "No batch_generate_api calls found in transcript"

    # The clean summary must appear in provocation inputs
    assert any(
        "CANARY_SUMMARY" in content for content in provocation_inputs
    ), (
        "topic.summary ('CANARY_SUMMARY') was not found in provocation inputs — "
        "check_refusal may not be using topic.summary for provocation generation."
    )

    # The verbose raw must NOT appear in provocation inputs
    assert not any(
        "CANARY_RAW_VERBOSE_PHRASE" in content for content in provocation_inputs
    ), (
        "topic.raw ('CANARY_RAW_VERBOSE_PHRASE') appeared in provocation inputs — "
        "check_refusal is using topic.raw instead of topic.summary. Bug regressed."
    )
