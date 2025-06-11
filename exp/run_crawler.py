import os
import torch

from core.crawler import Crawler, get_run_name
from core.crawler_config import CrawlerConfig
from core.llm_utils import load_model_and_tokenizer, load_filter_models, load_from_path
from core.project_config import INTERIM_DIR, RESULT_DIR
import argparse

DEFAULT_CONFIG = {
    "crawler": CrawlerConfig(
        temperature=0.6,
        num_samples_per_topic=1,
        num_crawl_steps=1000,
        generation_batch_size=2,
        max_topic_string_length=100,
        max_generated_tokens=100,
        max_extracted_topics_per_generation=10,
        max_crawl_topics=1_000_000,
        tokenization_template="chat",
        do_filter_refusals=True,
        do_force_thought_skip=False,
        prompt_languages=["english", "chinese"],
        refusal_max_new_tokens=25,
    ),
    "misc": {
        "verbose": False,
    },
}

DEBUG_CONFIG = {
    "crawler": CrawlerConfig(
        temperature=0.6,
        num_samples_per_topic=1,
        num_crawl_steps=2,
        generation_batch_size=10,
        max_topic_string_length=200,
        max_generated_tokens=200,
        max_extracted_topics_per_generation=10,
        max_crawl_topics=10000,
        tokenization_template="chat",
        do_filter_refusals=True,
        do_force_thought_skip=False,
        prompt_languages=["english"],
    ),
    "misc": {
        "verbose": True,
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--load_fname", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--model_path",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Path to the model to use for crawling",
    )
    parser.add_argument(
        "--quantization_bits",
        type=str,
        default="none",
        choices=["none", "4", "8"],
        help="Quantization bits for model loading (none for no quantization, 4 or 8 for quantization)",
    )
    parser.add_argument(
        "--prompt_injection_location",
        type=str,
        choices=[
            "user_all",
            "user_suffix",
            "assistant_prefix",
            "thought_prefix",
            "thought_suffix",
        ],
        help="Where to inject the prompt in the conversation",
    )
    parser.add_argument(
        "--not_use_openai_embeddings",
        action="store_true",
        help="Do not use OpenAI embeddings for filtering",
    )
    args = parser.parse_args()

    # Convert quantization_bits from string to appropriate type
    if args.quantization_bits == "none":
        args.quantization_bits = None
    else:
        args.quantization_bits = int(args.quantization_bits)

    if args.debug:
        exp_config = DEBUG_CONFIG
    else:
        exp_config = DEFAULT_CONFIG

    # Setting device for tensors and smaller models
    exp_config["crawler"].device = args.device

    # Initialize models
    if not any(keyword in args.model_path for keyword in ["claude", "grok"]):
        model_crawl, tokenizer_crawl = load_model_and_tokenizer(
            args.model_path,
            device=args.device,
            cache_dir=args.cache_dir,
            quantization_bits=args.quantization_bits,
        )
    else:
        model_crawl = args.model_path
        tokenizer_crawl = None

    filter_models = load_filter_models(
        args.cache_dir, args.device, load_openai=not args.not_use_openai_embeddings
    )

    # Get crawler name
    run_name = get_run_name(
        args.model_path, exp_config["crawler"], args.prompt_injection_location
    )
    # Add quantization info to run name
    run_name = f"{run_name}_q{args.quantization_bits}"
    crawler_log_filename = os.path.join(INTERIM_DIR, f"{run_name}.json")
    print(f"Run name: {run_name}")
    print(f"Saving to: {crawler_log_filename}\n\n")

    # Create Crawler or load from checkpoint
    if args.load_fname is None:
        crawler = Crawler(
            crawler_config=exp_config["crawler"], save_filename=crawler_log_filename
        )
    else:
        load_dir = os.path.join(INTERIM_DIR, args.load_fname)
        crawler = Crawler.load(
            load_from_filename=load_dir,
            save_to_filename=crawler_log_filename,
        )
        crawler.config = exp_config["crawler"]  # adapt the config to the new parameters
        crawler.config.initial_topics = (
            []
        )  # no initial topics, we do not need to seed as we're not starting from scratch

    # Go crawling!
    crawler.crawl(
        model=model_crawl,
        tokenizer=tokenizer_crawl,
        filter_models=filter_models,
        prompt_injection_location=args.prompt_injection_location,
        verbose=exp_config["misc"]["verbose"],
    )

    plot_filename = os.path.join(RESULT_DIR, f"{run_name}.png")
    crawler.stats.visualize_cumulative_topic_count(plot_filename)
