import random
from tqdm import trange
import json
from typing import List, Dict, Tuple

from datetime import datetime

import torch

from src.generation_utils import batch_generate
from src.prompt_builder import PromptBuilder
from src.crawler.topic_queue import TopicQueue, Topic
from src.crawler.config import CrawlerConfig
from src.crawler.crawler_stats import CrawlerStats
from src.response_formatting_utils import TopicFormatter
from src.refusal_utils import check_refusal


class Crawler:
    def __init__(self, crawler_config: CrawlerConfig, save_filename: str) -> None:

        self.config = crawler_config
        self.queue = TopicQueue()
        # NOTE Model, Tokenizer, are not saved to the Object to reduce overhead

        self.stats = CrawlerStats()

        self.formatter = TopicFormatter(crawler_config)

        self.prompt_builder = PromptBuilder(
            user_pres=crawler_config.fallback_user_message_templates,
            user_seed_templates=crawler_config.user_message_templates,
            assistant_pres=crawler_config.crawler_thinking_messages,
            user_seed_topics=self.queue,
            languages=crawler_config.crawler.prompt_languages,
        )

        self.save_filename = save_filename
        self.save(save_filename)  # Already testing at initialization whether saving works

    def _resolve_model(self, role: str, local_model, local_tokenizer):
        """Return (model, tokenizer) for the given role.

        If the role's model config is "local", returns the local vLLM model/tokenizer.
        Otherwise returns the OpenRouter model name string (with None tokenizer).
        """
        model_name = getattr(self.config.model, f"{role}_model")
        if model_name == "local":
            return local_model, local_tokenizer
        else:
            return model_name, None

    def initialize_topics(
        self,
        local_model,
        local_tokenizer,
        initial_topics: List[str],
        verbose: bool = False,
    ) -> List[Topic]:
        """Initialize all initial topics as heads."""
        topics = []
        for i, topic_str in enumerate(initial_topics):
            is_chinese = self.formatter._has_chinese(topic_str)
            if is_chinese:
                topic_english = self.formatter._translate_zn_to_en(
                    local_model,
                    local_tokenizer,
                    topic_str,
                )
                topic_chinese = topic_str
            else:
                topic_english = topic_str
                topic_chinese = self.formatter._translate_en_to_zn(
                    local_model,
                    local_tokenizer,
                    topic_str,
                )
            topics.append(
                Topic(
                    raw=topic_str,
                    english=topic_english,
                    chinese=topic_chinese,
                    is_head=True,
                    cluster_idx=i,
                    is_refusal=True,  # this is the one initialization seed.
                    parent_id=-5,
                    summary=topic_str,
                )
            )

        topics = self.queue.incoming_batch(topics)

        self.save(self.save_filename)

        return topics

    def crawl(
        self,
        local_model,
        local_tokenizer,
        verbose: bool = False,
    ) -> List[str]:
        """Crawl the topics."""

        # Initialize topics
        if self.config.initial_topics:
            self.initialize_topics(
                local_model=local_model,
                local_tokenizer=local_tokenizer,
                initial_topics=self.config.initial_topics,
                verbose=verbose,
            )

        # Iterate through crawling steps
        for crawl_step_idx in trange(self.config.crawler.num_crawl_steps, desc="Crawling topics"):
            print(f"Crawl step: {crawl_step_idx} / {self.config.crawler.num_crawl_steps}")

            # Generate with prefilling, iterating over all languages to incentivize language balance
            for lang in self.config.crawler.prompt_languages:
                torch.cuda.empty_cache()

                # Determine whether in warmup phase
                if crawl_step_idx < self.config.crawler.seed_warmup_steps:
                    warmup_step_idx = crawl_step_idx
                else:
                    warmup_step_idx = None

                n = self.config.crawler.generation_batch_size * self.config.crawler.num_samples_per_topic
                messages, topic_parent_ids = self.prompt_builder.build_messages(
                    lang, n, warmup_step_idx
                )

                if verbose:
                    print(f"\n## generating...")

                if "thought_suffix" in self.config.crawler.prefill_mode:
                    assert (
                        self.config.max_new_tokens == 2048
                    ), "We need to allow longform generation to capture the full thought before prefilling!"

                target_model, target_tokenizer = self._resolve_model("target", local_model, local_tokenizer)
                generated_texts, input_strs = batch_generate(
                    target_model,
                    target_tokenizer,
                    messages,
                    max_new_tokens=self.config.crawler.max_generated_tokens,
                    temperature=self.config.model.temperature,
                    verbose=verbose,
                    cfg=self.config,
                )

                # Post-generation prefilling
                if self.config.crawler.prefill_mode == "thought_suffix":
                    prefilled_texts = []
                    for gen in generated_texts:
                        if "</think>" in gen:
                            gen = gen.split("</think>")[0]  # Only keep the text until </think>
                            gen = (
                                gen + "</think>\n\n" + thinking_message + "\n1. "
                            )  # Prefill the generated text
                            prefilled_texts.append(gen)
                        else:
                            pass  # dont use topics that lack a complete thinking process
                    generated_texts = batch_complete_R1(
                        model=target_model,
                        tokenizer=target_tokenizer,
                        texts=prefilled_texts,
                        max_new_tokens=self.config.crawler.max_generated_tokens,
                        temperature=self.config.model.temperature,
                    )
                    print(f"post suffixing: {generated_texts}")

                if verbose:
                    print(f"\n## formatting...")
                new_topics = self.formatter.extract_and_format(
                    local_model=local_model,
                    local_tokenizer=local_tokenizer,
                    input_strs=input_strs,
                    generations=generated_texts,
                    parent_ids=topic_parent_ids,
                    verbose=verbose,
                )

                if verbose:
                    print(f"\n## deduplicating...")
                new_topics = self.formatter.deduplicate_exact(
                    formatted_topics=new_topics,
                    head_topics=self.queue.head_topics,
                    verbose=verbose,
                )

                if len(new_topics) == 0:
                    continue

                if self.config.crawler.do_filter_refusals:
                    if verbose:
                        print(f"\n## filtering for refusal...")
                    new_topics = check_refusal(
                        config=self.config,
                        local_model=local_model,
                        local_tokenizer=local_tokenizer,
                        selected_topics=new_topics,
                        verbose=verbose,
                    )

                # Update queue
                self.queue.incoming_batch(new_topics)
                if verbose:
                    for topic in new_topics:
                        print(
                            f"new topic from crawl step {crawl_step_idx}:\n{topic.english}\n{topic.raw}\n\n"
                        )
                        # Record memory usage

                # Log stats
                self.stats.log_step(
                    new_topics_all=len(new_topics),
                    new_topics_deduped=sum(1 for t in new_topics if t.is_head),
                    new_topics_refusals=sum(1 for t in new_topics if t.is_refusal),
                    total_unique_refusals=len(self.queue.head_refusal_topics),
                )

            # Save
            self.save(self.save_filename)

            if self.queue.num_head_topics > self.config.crawler.max_crawl_topics:
                print(f"Topic queue has {len(self.queue.head_topics)} topics")
                break

        self.save(self.save_filename)
        return self.queue

    def to_dict(self):
        crawler_dict = {
            "stats": self.stats.to_dict(),
            "config": self.config.to_dict(),
            "queue": self.queue.to_dict(),
        }
        return crawler_dict

    def save(self, filename: str):
        crawler_dict = self.to_dict()
        with open(filename, "w") as f:
            json.dump(crawler_dict, f)
        return crawler_dict

    @classmethod
    def load(cls, load_from_filename: str, save_to_filename: str):
        with open(load_from_filename, "r") as f:
            crawler_dict = json.load(f)

        crawler_config = CrawlerConfig(**crawler_dict["config"])
        crawler = cls(crawler_config, save_to_filename)
        crawler.queue = TopicQueue.load(crawler_dict["queue"])
        crawler.stats = CrawlerStats.load(crawler_dict["stats"])
        return crawler


def get_run_name(crawler_config: CrawlerConfig):
    target_model = crawler_config.model.target_model
    if target_model == "local":
        model_name = crawler_config.model.local_model.split("/")[-1] if crawler_config.model.local_model else "no_model"
    else:
        model_name = target_model.split("/")[-1]
    run_name = (
        "crawler_log"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_{model_name}"
        f"_{crawler_config.crawler.num_samples_per_topic}samples"
        f"_{crawler_config.crawler.num_crawl_steps}crawls"
        f"_{crawler_config.crawler.do_filter_refusals}filter"
        f"_{crawler_config.crawler.prefill_mode}prompt"
    )
    return run_name
