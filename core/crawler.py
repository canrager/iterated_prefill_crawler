import re
import random
import string
from tqdm import trange
import json
from typing import List, Dict, Tuple, Union

from datetime import datetime

import torch
from torch import Tensor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from core.generation_utils import (
    batch_generate,
    compute_embeddings,
    batch_compute_openai_embeddings,
    batch_complete_R1,
    MessageSegments,
)
from core.topic_queue import TopicQueue, Topic
from core.crawler_config import CrawlerConfig
from core.crawler_stats import CrawlerStats
from core.response_formatting_utils import remove_thinking_context, TopicFormatter


class Crawler:
    def __init__(self, crawler_config: CrawlerConfig, save_filename: str) -> None:

        self.config = crawler_config
        self.queue = TopicQueue()
        self.head_embedding_CD: Tensor = (
            None  # will be initialized in initialize_head_embeddings
        )
        # NOTE Model, Tokenizer, are not saved to the Object to reduce overhead

        self.stats = CrawlerStats()

        self.formatter = TopicFormatter(crawler_config)

        self.save_filename = save_filename
        self.save(
            save_filename
        )  # Already testing at initialization whether saving works

    def deduplicate_exact(
        self,
        formatted_topics: List[Topic],
        verbose: bool = False,
    ) -> List[Topic]:
        """
        Finds novel head topics in incoming batch by checking for exact duplicates
        (after normalization: lowercase + strip punctuation) in topic.summary.
        New topics are marked as heads.

        Args:
            formatted_topics: List of topics to deduplicate
            verbose: Whether to print verbose output

        Returns:
            List[Topic]: The input topics with is_head, cluster_idx, and cossim_to_head fields updated
        """
        if formatted_topics == []:
            return formatted_topics

        # Normalization helper function: lowercase + strip punctuation
        def normalize_summary(text) -> str:
            if text is None:
                return ""
            # Handle list case - join into string
            if isinstance(text, list):
                text = " ".join(str(item) for item in text if item)
            # Ensure text is a string
            if not isinstance(text, str):
                text = str(text)
            # Convert to lowercase and remove all punctuation
            text_lower = text.lower()
            # Remove punctuation using translate
            translator = str.maketrans("", "", string.punctuation)
            normalized = text_lower.translate(translator)
            return normalized

        # Build lookup dictionary: normalized_summary -> cluster_idx
        normalized_to_cluster_idx = {}
        for idx, head_topic in enumerate(self.queue.head_topics):
            normalized_summary = normalize_summary(head_topic.summary)
            if normalized_summary:  # Only add non-empty normalized summaries
                normalized_to_cluster_idx[normalized_summary] = idx

        # Process each topic
        for topic in formatted_topics:
            normalized_summary = normalize_summary(topic.summary)

            if normalized_summary and normalized_summary in normalized_to_cluster_idx:
                # Exact match found - not a head
                cluster_idx = normalized_to_cluster_idx[normalized_summary]
                topic.is_head = False
                topic.cluster_idx = cluster_idx
                topic.cossim_to_head = 1.0
            else:
                # No match found - new head
                topic.is_head = True
                topic.cluster_idx = len(self.queue.head_topics)
                topic.cossim_to_head = 1.0
                # Add to lookup for future comparisons
                if normalized_summary:
                    normalized_to_cluster_idx[normalized_summary] = topic.cluster_idx

        if verbose:
            new_head_topics = [t for t in formatted_topics if t.is_head]
            print(f"new head topics (exact deduplication):\n")
            for t in new_head_topics:
                print(f"{t.summary}\n{t.raw}\n\n")

        return formatted_topics

    def is_refusal(self, text: str | None) -> bool:
        if text is None:
            return False

        assistant_answer = text.split("ssistant")[-1]

        # Check each pattern until we find a match
        for pattern in self.config.refusal_messages:
            if re.search(pattern, assistant_answer, re.IGNORECASE):
                return True
        return False

    def check_refusal(
        self,
        model: Union[AutoModelForCausalLM, str],
        tokenizer: AutoTokenizer,
        selected_topics: List[Topic],
        user_message_templates: List[List[str]],
        force_thought_skip: bool = False,
        verbose: bool = False,
    ) -> List[Topic]:
        """Filtering incoming head topics for refusals."""

        if selected_topics == []:
            return selected_topics

        num_checks = self.config.num_refusal_checks_per_topic
        threshold = self.config.is_refusal_threshold

        # Step 1: Generate queries for ALL topics in parallel
        all_query_prompts = []
        topic_indices = []  # Track which prompts belong to which topic

        for topic_idx, topic in enumerate(selected_topics):
            instructions = (
                self.config.refusal_provocation_generation_prompts["chinese"]
                if topic.is_chinese
                else self.config.refusal_provocation_generation_prompts["english"]
            )
            # Add num_checks prompts for this topic
            all_query_prompts.extend([instructions.format(topic.raw)] * num_checks)
            topic_indices.extend([topic_idx] * num_checks)

        # Generate all queries at once
        # Note that batch_generate does enforce internal batching, but vllm might do automatically?
        all_queries, all_query_input_strs = batch_generate(
            model,
            tokenizer,
            all_query_prompts,
            max_new_tokens=(
                self.config.vllm_max_model_len
                if self.config.vllm_max_model_len is not None
                else self.config.max_refusal_check_generated_tokens
            ),
            temperature=1,
            verbose=verbose,
            cfg=self.config,
        )

        # Remove thinking context from queries if present
        all_queries = remove_thinking_context(all_queries)

        # Step 2: Process query refusals and collect topics that need answer checks
        topics_needing_answers = []
        topic_to_queries = {}  # Map topic_idx -> list of queries

        for topic_idx, topic in enumerate(selected_topics):
            # Extract queries for this topic
            start_idx = topic_idx * num_checks
            end_idx = start_idx + num_checks
            queries = all_queries[start_idx:end_idx]
            prompts = all_query_input_strs[start_idx:end_idx]

            # Check if model refused to make the query
            refused_to_make_query = [self.is_refusal(query) for query in queries]
            make_query_majority_refusal = (
                sum(refused_to_make_query) / len(refused_to_make_query)
            ) > threshold

            if verbose:
                print(
                    f"Topic {topic_idx}: make_query_majority_refusal: {make_query_majority_refusal}"
                )
                print(f"queries: {queries}")

            topic.is_refusal = False
            if make_query_majority_refusal:
                topic.is_refusal = True
                topic.refusal_check_queries = prompts
                topic.refusal_check_responses = queries

            else:
                # Store queries and mark for answer generation
                topic_to_queries[topic_idx] = queries
                topics_needing_answers.append(topic_idx)

        # Step 3: Generate answers for all non-refused topics in parallel
        if topics_needing_answers:
            all_answer_prompts = []
            answer_topic_indices = []  # Track which prompts belong to which topic

            for topic_idx in topics_needing_answers:
                queries = topic_to_queries[topic_idx]
                # Extract user messages from queries
                extracted_queries = [
                    query.split("ssistant")[-1].strip(" <>|:") for query in queries
                ]
                all_answer_prompts.extend(extracted_queries)
                answer_topic_indices.extend([topic_idx] * len(extracted_queries))

            # Generate all answers at once
            all_answers, all_answer_strs = batch_generate(
                model,
                tokenizer,
                all_answer_prompts,
                max_new_tokens=self.config.max_refusal_check_generated_tokens,
                temperature=self.config.temperature,
                cfg=self.config,
            )

            # Step 4: Process answer refusals
            for topic_idx in topics_needing_answers:
                topic = selected_topics[topic_idx]

                # Extract answers for this topic
                start_idx = topics_needing_answers.index(topic_idx) * num_checks
                end_idx = start_idx + num_checks
                answers = all_answers[start_idx:end_idx]
                answer_strs = all_answer_strs[start_idx:end_idx]

                # Check if model refused to answer
                refused_to_answer_query = [
                    self.is_refusal(answer) for answer in answers
                ]
                make_answer_majority_refusal = (
                    sum(refused_to_answer_query) / len(refused_to_answer_query)
                ) > threshold

                if verbose:
                    print(
                        f"Topic {topic_idx}: make_answer_majority_refusal: {make_answer_majority_refusal}"
                    )
                    print(f"answers: {answers}")

                topic.refusal_check_queries = answer_strs
                topic.refusal_check_responses = answers
                if make_answer_majority_refusal:
                    topic.is_refusal = True

        return selected_topics

    def initialize_topics(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        filter_models: Dict,
        initial_topics: List[str],
        verbose: bool = False,
    ) -> List[Topic]:
        """Initialize all initial topics as heads."""
        topics = []
        for i, topic_str in enumerate(initial_topics):
            is_chinese = self.formatter._has_chinese(topic_str)
            if is_chinese:
                topic_english = self.formatter._translate_zn_to_en(
                    filter_models["model_zh_en"],
                    filter_models["tokenizer_zh_en"],
                    topic_str,
                )
                topic_chinese = topic_str
            else:
                topic_english = topic_str
                topic_chinese = self.formatter._translate_en_to_zn(
                    filter_models["model_en_zh"],
                    filter_models["tokenizer_en_zh"],
                    topic_str,
                )
            topics.append(
                Topic(
                    raw=topic_str,
                    english=topic_english,
                    chinese=topic_chinese,
                    is_head=True,
                    cluster_idx=i,
                    cossim_to_head=1.0,
                    is_refusal=True,  # this is the one initialization seed.
                    parent_id=-5,
                    summary=topic_str,
                )
            )

        topics = self.queue.incoming_batch(topics)

        self.save(self.save_filename)

        return topics

    def _get_seed_topics(self) -> List[Topic]:
        """Get seed topics for the current crawl step.

        Args:
            crawl_step_idx: The current crawl step index

        Returns:
            List of seed topics to use for generation
        """
        # Return an empty list of topics if no seed_topics required:
        if "no_seed" in self.config.prefill_mode:
            empty_seeds = [""] * self.config.generation_batch_size
            seed_topics_text_languages = {}
            seed_topics_text_languages["english"] = empty_seeds
            seed_topics_text_languages["chinese"] = empty_seeds
            empty_parent_ids = [-1] * self.config.generation_batch_size
            return empty_seeds, empty_parent_ids

        # Choose topic seed candidates
        if self.config.do_filter_refusals:
            topic_seed_candidates = self.queue.head_refusal_topics
        else:
            topic_seed_candidates = self.queue.head_topics

        # Get sample topics
        if topic_seed_candidates == []:
            raise ValueError("Empty topic list!")
        elif len(topic_seed_candidates) <= self.config.generation_batch_size:
            # Early stage filling the queue
            seed_topics = topic_seed_candidates
        else:
            # Randomly sample for topic diversity
            seed_topics = random.sample(
                topic_seed_candidates, self.config.generation_batch_size
            )

        # Select relevant attributes for seeding
        seed_topics_text_languages = {}
        seed_topics_text_languages["english"] = [t.english for t in seed_topics]
        seed_topics_text_languages["chinese"] = [t.chinese for t in seed_topics]
        topic_parent_ids = [t.id for t in seed_topics]

        return seed_topics_text_languages, topic_parent_ids

    def _get_message_segments(self, lang="english", warmup_idx=None) -> MessageSegments:
        if warmup_idx:
            prefill_message = self.config.crawler_thinking_messages[lang][warmup_idx]
        else:
            prefill_message = random.choice(self.config.crawler_thinking_messages[lang])

        listing_prefill = "Topics:\n1. "

        match self.config.prefill_mode:
            case "user_prefill_no_seed":
                segments = MessageSegments(
                    user_prefix=random.choice(
                        self.config.fallback_user_message_templates[lang]
                    ),
                    user_suffix=prefill_message,
                    assistant_prefix=listing_prefill,
                )
            case "user_prefill_with_seed":
                segments = MessageSegments(
                    user_template=random.choice(
                        self.config.user_message_templates[lang]
                    ),
                    user_suffix=prefill_message,
                    assistant_prefix=listing_prefill,
                )
            case "assistant_prefill_no_seed":
                segments = MessageSegments(
                    user_prefix=random.choice(
                        self.config.fallback_user_message_templates[lang]
                    ),
                    assistant_prefix=f"{prefill_message}\n{listing_prefill}",
                )
            case "assistant_prefill_with_seed":
                segments = MessageSegments(
                    user_template=random.choice(
                        self.config.user_message_templates[lang]
                    ),
                    assistant_prefix=f"{prefill_message}\n{listing_prefill}",
                )
            case "thought_prefill_no_seed":
                segments = MessageSegments(
                    user_prefix=random.choice(
                        self.config.fallback_user_message_templates[lang]
                    ),
                    thought_prefix=f"{prefill_message}\n{listing_prefill}",
                )
            case "thought_prefill_with_seed":
                segments = MessageSegments(
                    user_template=random.choice(
                        self.config.user_message_templates[lang]
                    ),
                    thought_prefix=f"{prefill_message}\n{listing_prefill}",
                )
            case _:
                raise ValueError("Invalid prefill mode selected.")

        return segments

    def crawl(
        self,
        model: Union[AutoModelForCausalLM, str],
        tokenizer: AutoTokenizer,
        filter_models: Dict,
        verbose: bool = False,
    ) -> List[str]:
        """Crawl the topics."""

        # Initialize topics
        if self.config.initial_topics:
            self.initialize_topics(
                model=model,
                tokenizer=tokenizer,
                filter_models=filter_models,
                initial_topics=self.config.initial_topics,
                verbose=verbose,
            )

        # Iterate through crawling steps
        for crawl_step_idx in trange(
            self.config.num_crawl_steps, desc="Crawling topics"
        ):
            print(f"Crawl step: {crawl_step_idx} / {self.config.num_crawl_steps}")
            # Get seed topics for this crawl step
            seed_topics_text_languages, topic_parent_ids = self._get_seed_topics()

            # Generate with prefilling, iterating over all languages to incentivize language balance
            for lang in self.config.prompt_languages:
                torch.cuda.empty_cache()

                # Determine whether in warmup phase
                if crawl_step_idx < self.config.seed_warmup_steps:
                    warmup_step_idx = crawl_step_idx
                else:
                    warmup_step_idx = None

                # Choose Message Segments / Templates
                message_segments = self._get_message_segments(lang, warmup_step_idx)
                seed_topics_text = seed_topics_text_languages[lang]

                if verbose:
                    print(f"\n## generating...")

                if "thought_suffix" in self.config.prefill_mode:
                    assert (
                        self.config.max_new_tokens == 2048
                    ), "We need to allow longform generation to capture the full thought before prefilling!"

                generated_texts, input_strs = batch_generate(
                    model,
                    tokenizer,
                    seed_topics_text,
                    message_segments=message_segments,
                    force_thought_skip=False,
                    max_new_tokens=self.config.max_generated_tokens,
                    temperature=self.config.temperature,
                    num_samples_per_topic=self.config.num_samples_per_topic,
                    verbose=verbose,
                    cfg=self.config,
                )

                # Post-generation prefilling
                if self.config.prefill_mode == "thought_suffix":
                    prefilled_texts = []
                    for gen in generated_texts:
                        if "</think>" in gen:
                            gen = gen.split("</think>")[
                                0
                            ]  # Only keep the text until </think>
                            gen = (
                                gen + "</think>\n\n" + thinking_message + "\n1. "
                            )  # Prefill the generated text
                            prefilled_texts.append(gen)
                        else:
                            pass  # dont use topics that lack a complete thinking process
                    generated_texts = batch_complete_R1(
                        model=model,
                        tokenizer=tokenizer,
                        texts=prefilled_texts,
                        max_new_tokens=self.config.max_generated_tokens,
                        temperature=self.config.temperature,
                    )
                    print(f"post suffixing: {generated_texts}")

                if verbose:
                    print(f"\n## formatting...")
                new_topics = self.formatter.extract_and_format(
                    model_zh_en=filter_models["model_zh_en"],
                    tokenizer_zh_en=filter_models["tokenizer_zh_en"],
                    model_en_zh=filter_models["model_en_zh"],
                    tokenizer_en_zh=filter_models["tokenizer_en_zh"],
                    model_spacy_en=filter_models["model_spacy_en"],
                    input_strs=input_strs,
                    generations=generated_texts,
                    parent_ids=topic_parent_ids,
                    model=model,
                    tokenizer=tokenizer,
                    verbose=verbose,
                )

                if verbose:
                    print(f"\n## deduplicating...")
                # Select deduplication method based on config
                dedup_method = getattr(self.config, "deduplication_method", None)

                # Default fallback: use exact deduplication
                new_topics = self.deduplicate_exact(
                    formatted_topics=new_topics,
                    verbose=verbose,
                )

                if len(new_topics) == 0:
                    continue

                if self.config.do_filter_refusals:
                    if verbose:
                        print(f"\n## filtering for refusal...")
                    new_topics = self.check_refusal(
                        model=model,
                        tokenizer=tokenizer,
                        selected_topics=new_topics,
                        user_message_templates=self.config.user_message_templates,
                        force_thought_skip=self.config.do_force_thought_skip,
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

            if self.queue.num_head_topics > self.config.max_crawl_topics:
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


def get_run_name(model_path: str, crawler_config: CrawlerConfig, prefill_mode: str):
    model_name = model_path.split("/")[-1]
    run_name = (
        "crawler_log"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_{model_name}"
        f"_{crawler_config.num_samples_per_topic}samples"
        f"_{crawler_config.num_crawl_steps}crawls"
        f"_{crawler_config.do_filter_refusals}filter"
        f"_{prefill_mode}prompt"
    )
    return run_name
