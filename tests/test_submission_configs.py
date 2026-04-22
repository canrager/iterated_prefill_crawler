from pathlib import Path

import yaml

from src.crawler.config import CrawlerRunConfig, ModelConfig


ROOT = Path(__file__).resolve().parents[1]


def _load_model_config(name: str) -> ModelConfig:
    with (ROOT / "configs" / "model" / f"{name}.yaml").open() as f:
        return ModelConfig(**yaml.safe_load(f))


def _load_crawler_config(name: str) -> CrawlerRunConfig:
    with (ROOT / "configs" / "crawler" / f"{name}.yaml").open() as f:
        return CrawlerRunConfig(**yaml.safe_load(f))


def test_debug_and_rehearsal_remain_same_method_subsets_of_default():
    default = _load_crawler_config("default")
    rehearsal = _load_crawler_config("rehearsal")
    debug = _load_crawler_config("debug")

    for smaller in (rehearsal, debug):
        assert smaller.do_filter_refusals == default.do_filter_refusals
        assert smaller.prompt_languages == default.prompt_languages
        assert smaller.seed_warmup_steps == default.seed_warmup_steps
        assert smaller.num_samples_per_topic == default.num_samples_per_topic
        assert smaller.num_crawl_steps <= default.num_crawl_steps
        assert smaller.generation_batch_size <= default.generation_batch_size
        assert smaller.max_crawl_topics <= default.max_crawl_topics
        assert smaller.max_generated_tokens <= default.max_generated_tokens
        assert (
            smaller.max_refusal_check_generated_tokens
            <= default.max_refusal_check_generated_tokens
        )
        assert (
            smaller.max_extracted_topics_per_generation
            <= default.max_extracted_topics_per_generation
        )
        assert (
            smaller.num_refusal_checks_per_topic
            <= default.num_refusal_checks_per_topic
        )

    assert debug.num_crawl_steps < rehearsal.num_crawl_steps < default.num_crawl_steps
