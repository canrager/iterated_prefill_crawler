import os

# Set environment variable to force spawn method before any imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os
from datetime import datetime

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.aggregation.aggregator import TopicAggregator, compute_consistency_score
from src.crawler.config import CrawlerConfig
from src.directory_config import CONFIG_DIR, ROOT_DIR, resolve_cache_dir
from src.llm_utils import load_model_and_tokenizer
from src.provider_config import collect_required_api_keys
from src.transcript_logger import init_transcript_log


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    crawler_config = CrawlerConfig(**OmegaConf.to_container(cfg, resolve=True))
    exp = crawler_config.aggregation

    # Validate input_paths (lives in aggregation config)
    input_paths = exp.input_paths
    if not input_paths:
        raise ValueError(
            "aggregation.input_paths is required. Set in YAML or pass as: "
            "aggregation.input_paths='[path1.json,path2.json]'"
        )

    # Resolve the aggregation model
    agg_model_name = exp.aggregation_model
    if agg_model_name == "local":
        if cfg.model.local_model is None:
            raise ValueError(
                "aggregation.aggregation_model is 'local' but model.local_model is not set"
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
        # Remote model — check required API keys
        missing = collect_required_api_keys([agg_model_name])
        if missing:
            details = ", ".join(f"{p} ({v})" for p, v in missing.items())
            raise ValueError(
                f"Missing API key(s) for aggregation_model={agg_model_name}: {details}"
            )
        model, tokenizer = agg_model_name, None

    # Prepare output dir and transcript log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(str(ROOT_DIR), "artifacts", "aggregation", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    run_name = f"aggregation_{timestamp}"
    transcript_path = init_transcript_log(run_name, output_dir=output_dir)
    print(f"Transcript log: {transcript_path}")

    # Load topics and run aggregation
    aggregator = TopicAggregator(crawler_config)
    topics, topic_sources = aggregator.load_topics(input_paths)
    print(f"Loaded {len(topics)} unique topics from {len(input_paths)} file(s)")

    # Constrained mode: classify into a fixed taxonomy instead of discovering
    # clusters via iterative reduction.
    fixed_topics_path = exp.fixed_topics_path
    if fixed_topics_path:
        with open(fixed_topics_path, "r") as f:
            fixed_topics = [line.strip() for line in f if line.strip()]
        print(
            f"Constrained mode: {len(fixed_topics)} fixed topics from "
            f"{fixed_topics_path}"
        )
        final_topics, trajectory, source_sets = aggregator.classify(
            model, tokenizer, topics, fixed_topics, topic_sources
        )
    else:
        final_topics, trajectory, source_sets = aggregator.aggregate(
            model, tokenizer, topics, topic_sources
        )

    # Report consistency score
    score, n_consistent, n_total = compute_consistency_score(
        source_sets, len(input_paths)
    )
    print(
        f"Consistency: {score:.1%} ({n_consistent}/{n_total} topics "
        f"present in all {len(input_paths)} runs)"
    )
    aggregator.save_artifacts(
        output_dir, final_topics, trajectory, input_paths, source_sets
    )
    if fixed_topics_path:
        aggregator.save_cell_matrix(
            output_dir, final_topics, topic_sources, input_paths
        )

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
