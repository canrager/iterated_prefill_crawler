"""AggregateCrawler — groups topics before checking refusals.

Instead of checking every topic individually for refusal, the AggregateCrawler
uses the Kimi pipeline to group semantically equivalent topics first, then
only checks the representative (head) of each group. This cuts target-model
refusal-check calls by ~25×.

Verdict propagation: non-head members inherit the head's is_refusal result.
Topics that match a known head inherit that head's refusal status directly
without any additional check.
"""

import json
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import trange

from src.crawler.config import CrawlerConfig
from src.crawler.crawler_stats import CrawlerStats
from src.crawler.grouping_pipeline import summarize_group_dedup
from src.crawler.topic_queue import Topic, TopicQueue
from src.generation_utils import batch_generate
from src.prompt_builder import PromptBuilder
from src.crawler.progressive_refusal import check_refusal_progressive
from src.response_formatting_utils import TopicFormatter


class AggregateCrawler:
    """Crawler that groups topics before refusal checking.

    The crawl loop:
      1. Generate with prefilling (same as Crawler)
      2. Extract + translate (no per-topic summarization)
      3. Group + deduplicate via Kimi pipeline
      4. Check refusal on novel group heads only
      5. Propagate verdicts to group members
      6. Queue
    """

    def __init__(self, crawler_config: CrawlerConfig, save_filename: str) -> None:
        self.config = crawler_config
        self.queue = TopicQueue()
        self.stats = CrawlerStats()
        self.formatter = TopicFormatter(crawler_config)
        self.prompt_builder = PromptBuilder(
            user_pre_templates=crawler_config.prompts.user_pre_templates,
            user_seed_templates=crawler_config.prompts.user_seed_templates,
            user_post_templates=crawler_config.prompts.user_post_templates,
            system_templates=crawler_config.prompts.system_templates,
            assistant_pre_templates=crawler_config.prompts.assistant_pre_templates,
            assistant_seed_templates=crawler_config.prompts.assistant_seed_templates,
            assistant_post_templates=crawler_config.prompts.assistant_post_templates,
            user_seed_topics=self.queue,
            languages=crawler_config.crawler.prompt_languages,
        )
        self.save_filename = save_filename
        self.save(save_filename)

    def _resolve_model(self, role: str, local_model, local_tokenizer):
        """Return (model, tokenizer) for the given role."""
        model_name = getattr(self.config.model, f"{role}_model")
        if model_name == "local":
            return local_model, local_tokenizer
        else:
            return model_name, None

    def initialize_topics(
        self,
        local_model,
        local_tokenizer,
        initial_topics,
        verbose: bool = False,
    ) -> List[Topic]:
        """Initialize bootstrap seeds as heads.

        Each entry in ``initial_topics`` is either:
          - a plain string (legacy): translated in bulk (one API call per
            direction for all string seeds); or
          - a dict ``{"english": ..., "chinese": ...}``: translations baked
            into the config, skipping the runtime translation call entirely.

        S4e: string seeds are now bulk-translated — one call for all Chinese
        seeds (zn_to_en) and one call for all English seeds (en_to_zn) — rather
        than a per-seed call each.
        """
        # First pass: partition string entries by language and record indices.
        # Dict entries are handled inline (no translation needed).
        chinese_indices: List[int] = []   # index into initial_topics
        chinese_strs: List[str] = []
        english_indices: List[int] = []   # index into initial_topics
        english_strs: List[str] = []

        for i, entry in enumerate(initial_topics):
            if isinstance(entry, dict):
                continue  # no translation needed; handled in second pass
            topic_str = entry
            if self.formatter._has_chinese(topic_str):
                chinese_indices.append(i)
                chinese_strs.append(topic_str)
            else:
                english_indices.append(i)
                english_strs.append(topic_str)

        # Bulk translate: one call per direction (S4e).
        translated_chinese_to_en: List[str] = []
        if chinese_strs:
            translated_chinese_to_en = self.formatter._translate_zn_to_en(
                local_model, local_tokenizer, chinese_strs
            )

        translated_english_to_zn: List[str] = []
        if english_strs:
            translated_english_to_zn = self.formatter._translate_en_to_zn(
                local_model, local_tokenizer, english_strs
            )

        # Build lookup: original index → (english, chinese)
        translations: dict = {}
        for idx, orig, en_trans in zip(chinese_indices, chinese_strs, translated_chinese_to_en):
            translations[idx] = (en_trans, orig)  # (english, chinese)
        for idx, orig, zn_trans in zip(english_indices, english_strs, translated_english_to_zn):
            translations[idx] = (orig, zn_trans)  # (english, chinese)

        # Second pass: construct Topic objects in original order.
        topics = []
        for i, entry in enumerate(initial_topics):
            if isinstance(entry, dict):
                topic_english = entry.get("english") or entry.get("en") or ""
                topic_chinese = entry.get("chinese") or entry.get("zh") or ""
                topic_str = topic_english or topic_chinese
            else:
                topic_str = entry
                topic_english, topic_chinese = translations[i]

            topics.append(
                Topic(
                    raw=topic_str,
                    english=topic_english,
                    chinese=topic_chinese,
                    is_head=True,
                    cluster_idx=i,
                    is_refusal=True,
                    parent_id=-5,
                    summary=topic_str,
                )
            )
        topics = self.queue.incoming_batch(topics)
        self.save(self.save_filename)
        return topics

    def _propagate_verdicts(self, topics: List[Topic]) -> List[Topic]:
        """Propagate refusal verdicts from group heads to members.

        - Group heads that were checked get their own is_refusal from check_refusal.
        - Non-head members inherit their group head's is_refusal.
        - Topics matching a known head (duplicate) inherit that known head's is_refusal.
        """
        # Build a map: cluster_idx -> head's is_refusal for heads in this batch
        head_verdicts: Dict[int, bool] = {}
        for t in topics:
            if t.is_head and t.is_refusal is not None:
                head_verdicts[t.cluster_idx] = t.is_refusal

        # Also include known heads from the queue
        for t in self.queue.head_topics:
            if t.is_refusal is not None:
                head_verdicts[t.cluster_idx] = t.is_refusal

        for t in topics:
            if not t.is_head and t.cluster_idx is not None and t.cluster_idx in head_verdicts:
                t.is_refusal = head_verdicts[t.cluster_idx]

        return topics

    def crawl(
        self,
        local_model,
        local_tokenizer,
        verbose: bool = False,
    ) -> TopicQueue:
        """Run the aggregate crawl loop."""
        if self.config.initial_topics:
            self.initialize_topics(
                local_model=local_model,
                local_tokenizer=local_tokenizer,
                initial_topics=self.config.initial_topics,
                verbose=verbose,
            )

        for crawl_step_idx in trange(
            self.config.crawler.num_crawl_steps, desc="Agg crawling"
        ):
            print(f"Crawl step: {crawl_step_idx} / {self.config.crawler.num_crawl_steps}")

            for lang in self.config.crawler.prompt_languages:
                torch.cuda.empty_cache()

                if crawl_step_idx < self.config.crawler.seed_warmup_steps:
                    warmup_step_idx = crawl_step_idx
                else:
                    warmup_step_idx = None

                n = self.config.crawler.generation_batch_size
                use_seed_templates = (
                    crawl_step_idx >= self.config.crawler.seed_warmup_steps
                    or self.prompt_builder.user_pre is None
                )
                if warmup_step_idx is not None and self.prompt_builder.user_pre and lang in self.prompt_builder.user_pre:
                    n = len(self.prompt_builder.user_pre[lang])
                messages, topic_parent_ids = self.prompt_builder.build_messages(
                    lang, n,
                    warmup_idx=warmup_step_idx,
                    use_seed_templates=use_seed_templates,
                )

                if verbose:
                    print(f"\n## generating...")

                target_model, target_tokenizer = self._resolve_model(
                    "target", local_model, local_tokenizer
                )
                generated_texts, input_strs = batch_generate(
                    target_model, target_tokenizer, messages,
                    max_new_tokens=self.config.crawler.max_generated_tokens,
                    temperature=self.config.model.temperature,
                    verbose=verbose,
                    default_provider=self.config.model.default_provider,
                    provider_url_overrides=self.config.model.provider_urls,
                    provider_concurrency_limits=self.config.model.provider_max_concurrency,
                    prefer_nitro=self.config.model.prefer_nitro,
                )

                if verbose:
                    print(f"\n## extracting + translating...")
                # Step 2: extract and translate (no summarization)
                new_topics = self.formatter.extract_and_translate(
                    local_model=local_model,
                    local_tokenizer=local_tokenizer,
                    input_strs=input_strs,
                    generations=generated_texts,
                    parent_ids=topic_parent_ids,
                    verbose=verbose,
                )

                if not new_topics:
                    continue

                # Step 3: group + deduplicate via Kimi pipeline
                if verbose:
                    print(f"\n## grouping + deduplicating ({len(new_topics)} topics)...")
                known_heads = self.queue.head_topics
                new_topics = summarize_group_dedup(
                    topics=new_topics,
                    known_heads=known_heads,
                    config=self.config,
                    local_model=local_model,
                    local_tokenizer=local_tokenizer,
                    verbose=verbose,
                )

                # Drop topics flagged as garbage/preamble
                new_topics = [t for t in new_topics if t.summary is not None]

                if not new_topics:
                    continue

                # Step 3.5: persist grouping results BEFORE the refusal check,
                # so if the (minutes-to-hours-long) refusal check is killed or
                # raises, the queue still reflects the grouping work done so far.
                self.queue.incoming_batch(new_topics)
                self.save(self.save_filename)

                # Step 4: check refusal on novel heads only
                if self.config.crawler.do_filter_refusals:
                    novel_heads = [t for t in new_topics if t.is_head]
                    if verbose:
                        print(
                            f"\n## checking refusals on {len(novel_heads)} novel heads "
                            f"(skipped {len(new_topics) - len(novel_heads)} duplicates)"
                        )
                    if novel_heads:
                        check_refusal_progressive(
                            config=self.config,
                            local_model=local_model,
                            local_tokenizer=local_tokenizer,
                            selected_topics=novel_heads,
                            verbose=verbose,
                        )
                        # Refusal results are updated in-place on novel_heads by
                        # check_refusal_progressive; no write-back needed.

                    # Step 5: propagate verdicts
                    new_topics = self._propagate_verdicts(new_topics)

                    # Heads were added with is_refusal=None; rebuild the
                    # membership cache now that refusal check has mutated them.
                    self.queue.refresh_refusal_membership()
                    self.save(self.save_filename)

                if verbose:
                    for topic in new_topics:
                        print(
                            f"new topic from crawl step {crawl_step_idx}:\n"
                            f"  {topic.english}\n  raw={topic.raw}\n"
                            f"  is_head={topic.is_head} cluster={topic.cluster_idx}\n"
                        )

                self.stats.log_step(
                    new_topics_all=len(new_topics),
                    new_topics_deduped=sum(1 for t in new_topics if t.is_head),
                    new_topics_refusals=sum(1 for t in new_topics if t.is_refusal),
                    total_unique_refusals=len(self.queue.head_refusal_topics),
                )

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
            "head_refusal_topics_summaries": [
                t.summary for t in self.queue.head_refusal_topics if t.parent_id != -5
            ],
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
        crawler.prompt_builder.user_seed_topics = crawler.queue
        crawler.stats = CrawlerStats.load(crawler_dict["stats"])
        return crawler
