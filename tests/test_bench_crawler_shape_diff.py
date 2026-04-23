from pathlib import Path

from src.crawler_shape_bench import format_scoreboard_diff


def test_format_scoreboard_diff_excludes_invalid_rows_from_top_rankings(tmp_path: Path):
    scoreboard = {
        "fixture_dir": "artifacts/crawler_shape_fixtures/demo",
        "cells": [
            {
                "cell_id": "baseline",
                "valid": True,
                "params": {
                    "num_samples_per_topic": 5,
                    "num_refusal_checks_per_topic": 10,
                    "seed_language_balance": "any",
                    "is_refusal_threshold": 0.25,
                    "seed_priority_keywords": None,
                },
                "metrics": {
                    "composite_score": 0.200,
                    "ccp_leaf_recall": 0,
                    "golden_en_coverage": 1,
                    "golden_zh_coverage": 1,
                    "sampler_political_seed_hit_rate": 0.10,
                    "fixture_hit_rate": 1.0,
                },
            },
            {
                "cell_id": "invalid_but_high",
                "valid": False,
                "params": {
                    "num_samples_per_topic": 40,
                    "num_refusal_checks_per_topic": 20,
                    "seed_language_balance": "match",
                    "is_refusal_threshold": 0.15,
                    "seed_priority_keywords": ["政治"],
                },
                "metrics": {
                    "composite_score": None,
                    "ccp_leaf_recall": 10,
                    "golden_en_coverage": 9,
                    "golden_zh_coverage": 9,
                    "sampler_political_seed_hit_rate": 0.90,
                    "fixture_hit_rate": 0.40,
                },
            },
            {
                "cell_id": "winner",
                "valid": True,
                "params": {
                    "num_samples_per_topic": 20,
                    "num_refusal_checks_per_topic": 10,
                    "seed_language_balance": "match",
                    "is_refusal_threshold": 0.25,
                    "seed_priority_keywords": ["政治"],
                },
                "metrics": {
                    "composite_score": 0.700,
                    "ccp_leaf_recall": 4,
                    "golden_en_coverage": 3,
                    "golden_zh_coverage": 5,
                    "sampler_political_seed_hit_rate": 0.55,
                    "fixture_hit_rate": 0.98,
                },
            },
        ],
    }

    rendered = format_scoreboard_diff(scoreboard)

    assert "baseline" in rendered
    assert "top1" in rendered
    assert "invalid_but_high" not in rendered
    assert "Δ winner" in rendered
    assert "top2" not in rendered
