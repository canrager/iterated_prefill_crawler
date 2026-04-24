import json
from pathlib import Path

import pytest

from src.crawler.config import CrawlerConfig
from src.crawler_shape_bench import (
    FixtureCaptureWriter,
    assert_replay_compatible_config,
    build_request_payload,
    compute_prompt_hash,
    iter_fixture_records_from_model_call,
    load_fixture_records,
)


def test_prompt_hash_is_stable_for_equivalent_requests():
    messages = [{"role": "user", "content": "Tell me about topic X."}]
    request_a = build_request_payload(
        call_type="batch_generate_api",
        messages=messages,
        temperature=0.6,
        max_tokens=128,
    )
    request_b = build_request_payload(
        call_type="batch_generate_api",
        messages=[{"content": "Tell me about topic X.", "role": "user"}],
        temperature=0.6,
        max_tokens=128,
    )

    assert compute_prompt_hash(request_a) == compute_prompt_hash(request_b)


def test_fixture_writer_serializes_records_and_tags_roles(tmp_path: Path):
    writer = FixtureCaptureWriter(
        fixture_root=tmp_path,
        run_id="demo-run",
        crawler_config=CrawlerConfig(),
        hydra_overrides=["crawler.num_crawl_steps=2"],
    )

    writer.record_model_call(
        call_type="batch_generate_api",
        model="openrouter/demo-target",
        inputs=[
            [{"role": "user", "content": "Tell me about topic X."}],
            [{"role": "user", "content": "Translate to English (translation only): 台湾"}],
        ],
        outputs=["target answer", "Taiwan"],
        temperature=0.6,
        max_tokens=128,
    )
    writer.record_model_call(
        call_type="async_query_openrouter",
        model="openrouter/demo-helper",
        inputs=[
            {"role": "system", "content": "You extract structured data from text. Always respond with valid JSON only."},
            {"role": "user", "content": "First, determine whether the AI response below is a list of refused or sensitive topics.\nAI response:\n..."},
        ],
        outputs='["topic a", "topic b"]',
        temperature=0.0,
        max_tokens=512,
    )

    metadata = writer.finalize({"head_refusal_topics_count": 2})
    records = load_fixture_records(tmp_path / "demo-run" / "responses.jsonl")

    assert metadata["records_written"] == 3
    assert len(records) == 3
    assert [record.model_role for record in records] == [
        "target",
        "translation",
        "summarization",
    ]
    assert records[0].response == "target answer"
    assert records[1].response == "Taiwan"
    assert records[2].response == '["topic a", "topic b"]'

    saved_config = json.loads((tmp_path / "demo-run" / "config.json").read_text())
    assert saved_config["hydra_overrides"] == ["crawler.num_crawl_steps=2"]
    assert saved_config["live_summary"]["head_refusal_topics_count"] == 2


def test_iter_fixture_records_rejects_mismatched_batch_lengths():
    with pytest.raises(ValueError):
        list(
            iter_fixture_records_from_model_call(
                call_type="batch_generate_api",
                model="openrouter/demo-target",
                inputs=[[{"role": "user", "content": "only one prompt"}]],
                outputs=["a", "b"],
                temperature=0.6,
                max_tokens=128,
            )
        )


def test_capture_requires_api_backed_roles():
    crawler_config = CrawlerConfig()
    crawler_config.model.target_model = "local"

    with pytest.raises(ValueError, match="requires API-backed models"):
        assert_replay_compatible_config(crawler_config)
