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

from src.aggregation.aggregator import TopicAggregator
from src.crawler.config import CrawlerConfig
from src.directory_config import CONFIG_DIR, ROOT_DIR, resolve_cache_dir
from src.llm_utils import load_model_and_tokenizer


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    crawler_config = CrawlerConfig(**OmegaConf.to_container(cfg, resolve=True))
    exp = crawler_config.experiments

    # Validate input_paths (lives in experiments config)
    input_paths = exp.input_paths
    if not input_paths:
        raise ValueError(
            "experiments.input_paths is required. Set in YAML or pass as: "
            "experiments.input_paths='[path1.json,path2.json]'"
        )

    # Resolve the aggregation model
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
        # OpenRouter model — check API key
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError(
                f"OPENROUTER_API_KEY not set, but experiments.aggregation_model={agg_model_name}"
            )
        model, tokenizer = agg_model_name, None

    # Load topics and run aggregation
    aggregator = TopicAggregator(crawler_config)
    topics = aggregator.load_topics(input_paths)
    print(f"Loaded {len(topics)} unique topics from {len(input_paths)} file(s)")

    final_clusters = aggregator.aggregate(model, tokenizer, topics)

    # Save artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(str(ROOT_DIR), "artifacts", "aggregation", timestamp)
    aggregator.save_artifacts(output_dir, final_clusters, input_paths)

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
