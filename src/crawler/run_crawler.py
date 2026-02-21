import os

# Set environment variable to force spawn method before any imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import multiprocessing

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# MUST be set before importing torch or any CUDA libraries
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from src.crawler.crawler import Crawler, get_run_name
from src.crawler.config import CrawlerConfig
from src.llm_utils import load_model_and_tokenizer, load_filter_models, load_from_path
from src.directory_config import INTERIM_DIR, RESULT_DIR, CONFIG_DIR, resolve_cache_dir


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    # Set environment variables if specified
    if "env" in cfg and cfg.env is not None:
        for key, value in cfg.env.items():
            os.environ[key] = value

    crawler_config = CrawlerConfig(**OmegaConf.to_container(cfg, resolve=True))

    # Resolve cache_dir relative to ROOT_DIR.parent and create if needed
    cache_dir_path = resolve_cache_dir(cfg.cache_dir)
    cache_dir_str = str(cache_dir_path)

    # Initialize models
    if not any(keyword in cfg.model_path for keyword in ["claude", "grok"]):
        model_crawl, tokenizer_crawl = load_model_and_tokenizer(
            cfg.model_path,
            device=cfg.device,
            cache_dir=cache_dir_str,
            quantization_bits=cfg.quantization_bits,
            backend=cfg.backend,
            vllm_tensor_parallel_size=cfg.vllm_tensor_parallel_size if cfg.backend == "vllm" else 1,
            vllm_gpu_memory_utilization=(
                cfg.vllm_gpu_memory_utilization if cfg.backend == "vllm" else 0.9
            ),
            vllm_max_model_len=crawler_config.max_context_tokens if cfg.backend == "vllm" else None,
        )
    else:
        model_crawl = cfg.model_path
        tokenizer_crawl = None

    filter_models = load_filter_models(cache_dir_str, cfg.device)

    # Get crawler name
    run_name = get_run_name(cfg.model_path, crawler_config, cfg.prefill_mode)
    # Add backend and quantization info to run name
    if cfg.backend == "vllm":
        run_name = f"{run_name}_vllm"
    else:
        run_name = f"{run_name}_hf_q{cfg.quantization_bits}"
    crawler_log_filename = os.path.join(INTERIM_DIR, f"{run_name}.json")
    print(f"Run name: {run_name}")
    print(f"Saving to: {crawler_log_filename}\n\n")

    # Create Crawler or load from checkpoint
    if cfg.load_fname is None:
        crawler = Crawler(crawler_config=crawler_config, save_filename=crawler_log_filename)
    else:
        load_dir = os.path.join(INTERIM_DIR, cfg.load_fname)
        crawler = Crawler.load(
            load_from_filename=load_dir,
            save_to_filename=crawler_log_filename,
        )
        crawler.config = crawler_config  # adapt the config to the new parameters
        crawler.config.initial_topics = (
            []
        )  # no initial topics, we do not need to seed as we're not starting from scratch

    # Go crawling!
    crawler.crawl(
        model=model_crawl,
        tokenizer=tokenizer_crawl,
        filter_models=filter_models,
        verbose=cfg.verbose,
    )

    plot_filename = os.path.join(RESULT_DIR, f"{run_name}.png")
    crawler.stats.visualize_cumulative_topic_count(plot_filename)


if __name__ == "__main__":
    main()
