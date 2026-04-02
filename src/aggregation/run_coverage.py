import os

# Set environment variable to force spawn method before any imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import json
import os
from datetime import datetime

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.aggregation.coverage import (
    CoverageResult,
    analyze_coverage,
    load_crawl_topics,
    load_ground_truth,
)
from src.crawler.config import CrawlerConfig
from src.directory_config import CONFIG_DIR, ROOT_DIR, resolve_cache_dir
from src.llm_utils import load_model_and_tokenizer
from src.provider_config import collect_required_api_keys
from src.transcript_logger import init_transcript_log


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    crawler_config = CrawlerConfig(**OmegaConf.to_container(cfg, resolve=True))
    exp = crawler_config.experiments

    # Validate inputs
    input_paths = exp.input_paths
    if not input_paths:
        raise ValueError(
            "experiments.input_paths is required. Set in YAML or pass as: "
            "experiments.input_paths='[path1.json,path2.json]'"
        )
    if not exp.ground_truth_reference:
        raise ValueError(
            "experiments.ground_truth_reference is required. Set in YAML or pass as: "
            "experiments.ground_truth_reference='path/to/ground_truth.json'"
        )

    # Resolve model
    agg_model_name = exp.aggregation_model
    if agg_model_name == "local":
        if cfg.model.local_model is None:
            raise ValueError(
                "experiments.aggregation_model is 'local' but model.local_model is not set"
            )
        cache_dir_path = resolve_cache_dir(cfg.model.cache_dir)
        model, tokenizer = load_model_and_tokenizer(
            cfg.model.local_model,
            device=cfg.model.device,
            cache_dir=str(cache_dir_path),
            quantization_bits=cfg.model.quantization_bits,
            vllm_tensor_parallel_size=cfg.model.vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=cfg.model.vllm_gpu_memory_utilization,
            vllm_max_model_len=crawler_config.model.vllm_max_model_len,
        )
    else:
        missing = collect_required_api_keys([agg_model_name])
        if missing:
            details = ", ".join(f"{p} ({v})" for p, v in missing.items())
            raise ValueError(
                f"Missing API key(s) for aggregation_model={agg_model_name}: {details}"
            )
        model, tokenizer = agg_model_name, None

    # Prepare output dir and transcript log
    hydra_choices = HydraConfig.get().runtime.choices
    exp_name = hydra_choices.get("experiments", "default")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(str(ROOT_DIR), "artifacts", "coverage", f"{exp_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    run_name = f"coverage_{timestamp}"
    transcript_path = init_transcript_log(run_name, output_dir=output_dir)
    print(f"Transcript log: {transcript_path}")

    # Load data
    ground_truth = load_ground_truth(exp.ground_truth_reference)
    crawl_topics = load_crawl_topics(input_paths)
    print(f"Loaded {sum(len(v) for v in ground_truth.values())} GT subtopics "
          f"across {len(ground_truth)} categories")
    print(f"Loaded {len(crawl_topics)} unique crawl topics from {len(input_paths)} file(s)")

    # Run coverage analysis
    result = analyze_coverage(
        ground_truth=ground_truth,
        crawl_topics=crawl_topics,
        model=model,
        tokenizer=tokenizer,
        max_tokens=exp.max_tokens,
        verbose=exp.verbose,
    )

    result.print_report()

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(crawler_config.to_dict(), f, indent=2)

    report_path = os.path.join(output_dir, "coverage_report.json")
    with open(report_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"Artifacts saved to {output_dir}/")
    print(f"  config.json, coverage_report.json")

    # Cleanup vLLM
    if not isinstance(model, str) and model is not None:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
        del model
        import gc

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
