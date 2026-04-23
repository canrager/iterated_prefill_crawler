from __future__ import annotations

import argparse
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.crawler.config import CrawlerConfig
from src.crawler.crawler import Crawler
from src.crawler.run_crawler import get_run_name
from src.crawler_shape_bench import (
    FixtureCaptureWriter,
    assert_replay_compatible_config,
    summarize_live_run,
)
from src.directory_config import CONFIG_DIR, resolve_cache_dir
from src.llm_utils import load_model_and_tokenizer
from src.provider_config import collect_required_api_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the crawler once against live APIs and capture every model response "
            "into a replay fixture. Do not use this in offline bench runs."
        )
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/crawler_shape_fixtures",
        help="Directory under which <run_id>/{responses.jsonl,config.json} is written.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional fixture run id. Defaults to crawler-shape_<UTC timestamp>.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides forwarded to the crawler config compose step.",
    )
    return parser.parse_args()


def load_crawler_config(overrides: list[str]) -> CrawlerConfig:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    if "env" in cfg and cfg.env is not None:
        for key, value in cfg.env.items():
            os.environ[key] = value

    crawler_config = CrawlerConfig(**OmegaConf.to_container(cfg, resolve=True))
    assert_replay_compatible_config(crawler_config)

    non_local_model_strings = [
        getattr(crawler_config.model, f"{role}_model")
        for role in ("target", "translation", "summarization", "refusal_check")
        if getattr(crawler_config.model, f"{role}_model") != "local"
    ]
    missing_keys = collect_required_api_keys(
        non_local_model_strings,
        default_provider=crawler_config.model.default_provider,
    )
    if missing_keys:
        details = ", ".join(f"{provider} ({env_var})" for provider, env_var in missing_keys.items())
        raise ValueError(
            f"Missing API key(s) for providers used by non-local roles: {details}"
        )

    return crawler_config


@contextmanager
def capture_model_calls(writer: FixtureCaptureWriter):
    import src.generation_utils as generation_utils
    import src.openrouter_utils as openrouter_utils

    original_generation_log = generation_utils.log_model_call
    original_openrouter_log = openrouter_utils.log_model_call

    def wrapped_log_model_call(**kwargs):
        writer.record_model_call(**kwargs)
        original_generation_log(**kwargs)

    generation_utils.log_model_call = wrapped_log_model_call
    openrouter_utils.log_model_call = wrapped_log_model_call
    try:
        yield
    finally:
        generation_utils.log_model_call = original_generation_log
        openrouter_utils.log_model_call = original_openrouter_log


def run_capture(
    *,
    output_root: Path,
    run_id: str,
    crawler_config: CrawlerConfig,
    hydra_overrides: list[str],
) -> Path:
    writer = FixtureCaptureWriter(
        fixture_root=output_root,
        run_id=run_id,
        crawler_config=crawler_config,
        hydra_overrides=hydra_overrides,
    )

    cache_dir = resolve_cache_dir(crawler_config.model.cache_dir)
    cache_dir_str = str(cache_dir)

    if crawler_config.model.local_model is not None:
        local_model, local_tokenizer = load_model_and_tokenizer(
            crawler_config.model.local_model,
            device=crawler_config.model.device,
            cache_dir=cache_dir_str,
            quantization_bits=crawler_config.model.quantization_bits,
            vllm_tensor_parallel_size=crawler_config.model.vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=crawler_config.model.vllm_gpu_memory_utilization,
            vllm_max_model_len=crawler_config.model.vllm_max_model_len,
        )
    else:
        local_model, local_tokenizer = None, None

    with tempfile.TemporaryDirectory(prefix="crawler-shape-capture-") as temp_dir:
        save_filename = str(Path(temp_dir) / f"{run_id}.json")
        crawler = Crawler(crawler_config=crawler_config, save_filename=save_filename)
        with capture_model_calls(writer):
            crawler.crawl(
                local_model=local_model,
                local_tokenizer=local_tokenizer,
                verbose=crawler_config.crawler.verbose,
            )

    writer.finalize(summarize_live_run(crawler))

    if local_model is not None:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
        del local_model
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    return writer.fixture_dir


def main() -> None:
    load_dotenv()
    args = parse_args()
    crawler_config = load_crawler_config(args.overrides)
    output_root = Path(args.output_root)
    run_id = args.run_id or (
        f"crawler-shape_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{get_run_name(crawler_config)}"
    )
    fixture_dir = run_capture(
        output_root=output_root,
        run_id=run_id,
        crawler_config=crawler_config,
        hydra_overrides=args.overrides,
    )
    print(f"Captured fixture: {fixture_dir}")
    print(f"Responses: {fixture_dir / 'responses.jsonl'}")
    print(f"Config: {fixture_dir / 'config.json'}")


if __name__ == "__main__":
    main()
