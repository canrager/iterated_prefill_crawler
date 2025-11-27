import re
import os
import random
import string
from tqdm import trange, tqdm
import json
from typing import List, Dict, Tuple, Union
from copy import copy

from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModel,
)
from spacy.language import Language

from core.generation_utils import (
    batch_generate,
    compute_embeddings,
    batch_compute_openai_embeddings,
    batch_complete_R1,
    MessageSegments,
)
from core.topic_queue import TopicQueue, Topic
from core.crawler_config import CrawlerConfig
from core.tokenization_utils import match_chat_template
from core.formatting_utils import remove_thinking_context

EPS = 1e-10


class CrawlerStats:
    def __init__(self):
        # Cumulative counters
        self.total_all = 0  # All topics generated
        self.total_deduped = 0  # These are all HEAD topics (topics after deduplication)
        self.total_refusals = 0  # All deduplicated topics that yield refusals
        self.total_unique_refusals = 0

        # History tracking
        self.all_per_step = []
        self.deduped_per_step = []
        self.refusal_per_step = []
        self.cluster_sizes_per_step = []

    def log_step(
        self,
        new_topics_all: int,
        new_topics_deduped: int,
        new_topics_refusals: float,
        current_clusters: List[List[Topic]],
        total_unique_refusals: int,
    ):
        """Log statistics for current step"""
        self.total_all += new_topics_all
        self.total_deduped += new_topics_deduped
        self.total_refusals += new_topics_refusals
        self.total_unique_refusals = total_unique_refusals
        self.all_per_step.append(new_topics_all)
        self.deduped_per_step.append(new_topics_deduped)
        self.refusal_per_step.append(new_topics_refusals)
        self.cluster_sizes_per_step.append(
            [len(cluster) for cluster in current_clusters]
        )

    def get_current_metrics(self) -> dict:
        """Get current state of metrics"""
        return {
            "total_all": self.total_all,
            "total_deduped": self.total_deduped,
            "total_refusals": sum(self.refusal_per_step),
            "total_unique_refusals": self.total_unique_refusals,
            "avg_refusal_rate": (sum(self.refusal_per_step) / (self.total_all + EPS)),
            "current_step": len(self.all_per_step),
            "largest_cluster": (
                max(self.cluster_sizes_per_step[-1])
                if self.cluster_sizes_per_step
                else 0
            ),
        }

    def visualize_cumulative_topic_count(
        self,
        save_path: str = None,
        show_all_topics: bool = False,
        title: str = "Cumulative topic and refusal count",
    ):
        cumulative_generations = torch.cumsum(torch.tensor(self.all_per_step), dim=0)
        cumulative_topics = torch.cumsum(torch.tensor(self.deduped_per_step), dim=0)
        cumulative_refusals = torch.cumsum(torch.tensor(self.refusal_per_step), dim=0)

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.grid(zorder=-1)
        ax.scatter(
            cumulative_generations, cumulative_topics, label="Unique topics", zorder=10
        )
        ax.scatter(
            cumulative_generations,
            cumulative_refusals,
            label="Refused unique topics",
            zorder=10,
        )
        ax.set_xlabel("Total crawled topics")
        ax.set_ylabel("Total crawled topics after filter")
        ax.set_title(title)
        ax.legend()
        if save_path is not None:
            plt.savefig(save_path)
        return fig

    def to_dict(self):
        """Convert the crawler stats to a dictionary representation."""
        stats_dict = {
            "cumulative": {
                "total_all": self.total_all,
                "total_deduped": self.total_deduped,
                "total_refusals": self.total_refusals,
                "total_unique_refusals": self.total_unique_refusals,
            },
            "current_metrics": self.get_current_metrics(),
            "history": {
                "all_per_step": self.all_per_step,
                "deduped_per_step": self.deduped_per_step,
                "refusal_per_step": self.refusal_per_step,
                "cluster_sizes_per_step": self.cluster_sizes_per_step,
            },
        }
        return stats_dict

    def save(self, filename: str):
        """Save the crawler stats to a JSON file."""
        stats_dict = self.to_dict()
        with open(filename, "w") as f:
            json.dump(stats_dict, f)

        return stats_dict

    @classmethod
    def load(cls, stats_dict: dict):
        """Load the crawler stats from a dictionary."""
        crawler_stats = cls()
        crawler_stats.total_all = stats_dict["cumulative"]["total_all"]
        crawler_stats.total_deduped = stats_dict["cumulative"]["total_deduped"]
        crawler_stats.total_refusals = stats_dict["cumulative"]["total_refusals"]
        crawler_stats.total_unique_refusals = stats_dict["cumulative"][
            "total_unique_refusals"
        ]
        crawler_stats.all_per_step = stats_dict["history"]["all_per_step"]
        crawler_stats.deduped_per_step = stats_dict["history"]["deduped_per_step"]
        crawler_stats.refusal_per_step = stats_dict["history"]["refusal_per_step"]
        crawler_stats.cluster_sizes_per_step = stats_dict["history"][
            "cluster_sizes_per_step"
        ]
        return crawler_stats

    def __repr__(self):
        return f"CrawlerStats(total_all={self.total_all}, total_deduped={self.total_deduped}, all_per_step={self.all_per_step}, deduped_per_step={self.deduped_per_step}, refusal_per_step={self.refusal_per_step}, cluster_sizes_per_step={self.cluster_sizes_per_step})"


class Crawler:
    def __init__(self, crawler_config: CrawlerConfig, save_filename: str) -> None:

        self.config = crawler_config
        self.queue = TopicQueue()
        self.head_embedding_CD: Tensor = (
            None  # will be initialized in initialize_head_embeddings
        )
        # NOTE Model, Tokenizer, are not saved to the Object to reduce overhead

        self.stats = CrawlerStats()

        self.numbered_list_pattern = re.compile(r"\d+\.\s*(.*?)(?:\n|$)")
        self.chinese_pattern = re.compile(r"[\u4e00-\u9fff]")

        self.save_filename = save_filename
        self.save(
            save_filename
        )  # Already testing at initialization whether saving works

    def _extract_from_numbered_list(self, text: str) -> List[str]:
        """Extract topics from a text that contains a numbered list."""
        extracted_list = self.numbered_list_pattern.findall(text)
        extracted_list = list(dict.fromkeys(extracted_list))  # Remove exact duplicates
        extracted_list = extracted_list[
            : self.config.max_extracted_topics_per_generation
        ]
        extracted_list = [item for item in extracted_list if item is not None]
        return extracted_list

    def _has_chinese(self, text: str) -> bool:
        """Check if the text contains Chinese characters."""
        return bool(self.chinese_pattern.search(text))

    def _translate_zn_to_en(
        self,
        model_zh_en: AutoModelForSeq2SeqLM,
        tokenizer_zh_en: AutoTokenizer,
        inputs: Union[str, List[str]],
    ) -> Union[str, List[str]]:
        zh_en_ids_B = tokenizer_zh_en(
            inputs, padding=True, truncation=True, return_tensors="pt"
        ).to(model_zh_en.device)
        with torch.inference_mode():
            translated_ids_B = model_zh_en.generate(**zh_en_ids_B, max_new_tokens=30)
        translated_str_B = tokenizer_zh_en.batch_decode(
            translated_ids_B, skip_special_tokens=True
        )
        return translated_str_B

    def _translate_en_to_zn(
        self,
        model_en_zh: AutoModelForSeq2SeqLM,
        tokenizer_en_zh: AutoTokenizer,
        inputs: Union[str, List[str]],
    ) -> Union[str, List[str]]:
        en_zh_ids_B = tokenizer_en_zh(
            inputs, padding=True, truncation=True, return_tensors="pt"
        ).to(model_en_zh.device)
        with torch.inference_mode():
            translated_ids_B = model_en_zh.generate(**en_zh_ids_B, max_new_tokens=30)
        translated_str_B = tokenizer_en_zh.batch_decode(
            translated_ids_B, skip_special_tokens=True
        )
        return translated_str_B

    def _batch_translate_chinese_english_both_ways(
        self,
        model_zh_en: AutoModelForSeq2SeqLM,
        tokenizer_zh_en: AutoTokenizer,
        model_en_zh: AutoModelForSeq2SeqLM,
        tokenizer_en_zh: AutoTokenizer,
        topics: List[Topic],
    ) -> List[Topic]:
        """Given a list of texts, translate the texts with chinese characters to english. Do not translate others.
        Changes the order of topics in the batch to [english] + [chinese]"""
        # check for chinese characters
        chinese_topics, chinese_indices = [], []
        english_topics, english_indices = [], []
        for i, topic in enumerate(topics):
            topic.is_chinese = self._has_chinese(topic.raw)
            if topic.is_chinese:
                chinese_topics.append(topic)
                chinese_indices.append(i)
            else:
                english_topics.append(topic)
                english_indices.append(i)

        # translate the subset with chinese characters in a single batch
        for batch_start in range(
            0, len(chinese_topics), self.config.generation_batch_size
        ):
            batch_end = batch_start + self.config.generation_batch_size
            chinese_topic_B = chinese_topics[batch_start:batch_end]
            chinese_indices_B = chinese_indices[batch_start:batch_end]
            chinese_raw_B = [t.raw for t in chinese_topic_B]
            translated_str_B = self._translate_zn_to_en(
                model_zh_en, tokenizer_zh_en, chinese_raw_B
            )
            for original, translation, idx in zip(
                chinese_raw_B, translated_str_B, chinese_indices_B
            ):
                topics[idx].english = translation
                topics[idx].shortened = translation  # copy of english
                topics[idx].chinese = original

        for batch_start in range(
            0, len(english_topics), self.config.generation_batch_size
        ):
            batch_end = batch_start + self.config.generation_batch_size
            english_topic_B = english_topics[batch_start:batch_end]
            english_indices_B = english_indices[batch_start:batch_end]
            english_raw_B = [t.raw for t in english_topic_B]
            translated_str_B = self._translate_en_to_zn(
                model_en_zh, tokenizer_en_zh, english_raw_B
            )
            for original, translation, idx in zip(
                english_raw_B, translated_str_B, english_indices_B
            ):
                topics[idx].english = original
                topics[idx].shortened = original  # copy of english
                topics[idx].chinese = translation
        return topics

    def _semantic_filter(
        self, model_spacy_en: Language, topics: List[Topic]
    ) -> List[Topic]:
        # Process in batches using the configured batch size
        for batch_start in range(0, len(topics), self.config.generation_batch_size):
            batch_end = batch_start + self.config.generation_batch_size
            batch_topics = topics[batch_start:batch_end]

            # Process batch of texts together
            docs = model_spacy_en.pipe([topic.shortened for topic in batch_topics])

            # Update each topic's text with filtered tokens
            for topic, doc in zip(batch_topics, docs):
                meaningful_tokens = [
                    token.text
                    for token in doc
                    if token.tag_[:2] in set(self.config.allowed_spacy_tags)
                ]
                topic.shortened = " ".join(meaningful_tokens)

        return topics

    def _regex_filter(self, topics: List[Topic]) -> List[Topic]:
        for topic in topics:
            item = topic.shortened
            item = item.lower()
            item = item.strip(" ./:\",'()[]")
            item = item.replace(".", "")  # remove dots
            item = " ".join(
                word for word in item.split() if len(word) > 1
            )  # remove single characters
            topic.shortened = item
        return topics

    def _remove_words(self, topics: List[Topic]) -> List[Topic]:
        """
        Remove filtered words and duplicates from a list of texts.

        Args:
            texts: List of strings to process

        Returns:
            List of processed strings with filtered words removed
        """
        if not topics:
            return []

        for topic in topics:
            if not topic.shortened:
                continue

            # duplicate removal
            words = list(
                dict.fromkeys(
                    word
                    for word in topic.shortened.split()
                    if word not in set(self.config.regex_filter_global)
                )
            )

            # Remove filtered words from start and end
            while words and words[0] in set(self.config.regex_filter_start_end_only):
                words.pop(0)
            while words and words[-1] in set(self.config.regex_filter_start_end_only):
                words.pop()

            topic.shortened = " ".join(words)
        return topics

    def _split_at_comma(
        self,
        topics: List[Topic],
        attribute: str,
    ) -> List[Topic]:
        relevant_attributes = ("raw", "summary")
        if attribute not in relevant_attributes:
            raise ValueError("Unknown Attribute.")

        for topic in topics:
            topic_attr = getattr(topic, attribute)
            if topic_attr and ("," in topic_attr or " or " in topic_attr):
                splitted_text = re.split(r",\s*|\s+or\s+", topic_attr)
                # Update the original topic with the first part
                setattr(topic, attribute, splitted_text[0].strip())
                # Create new topics for the remaining parts
                for item in splitted_text[1:]:
                    new_topic_kwargs = {
                        "parent_id": topic.parent_id,
                        attribute: item.strip(),
                    }
                    # Keep other attributes
                    for a in relevant_attributes:
                        if a != attribute:
                            new_topic_kwargs[a] = getattr(topic, a)
                    topics.append(Topic(**new_topic_kwargs))

        return topics

    def extract_and_format(
        self,
        model_zh_en: AutoModelForSeq2SeqLM,
        tokenizer_zh_en: AutoTokenizer,
        model_en_zh: AutoModelForSeq2SeqLM,
        tokenizer_en_zh: AutoTokenizer,
        model_spacy_en: Language | None,
        input_strs: List[str],
        generations: List[str],
        parent_ids: List[int],
        model: Union[AutoModelForCausalLM, str] = None,
        tokenizer: AutoTokenizer = None,
        verbose: bool = False,
    ) -> List[Topic]:

        formatted_topics = []
        parent_ids = parent_ids * (len(generations) // len(parent_ids))
        assert len(parent_ids) == len(generations)

        for gen, prompt, pid in zip(generations, input_strs, parent_ids):
            extracted_items = self._extract_from_numbered_list(gen)
            for item in extracted_items:
                formatted_topics.append(Topic(raw=item, parent_id=pid, prompt=prompt))

        if len(formatted_topics) == 0:
            print(f"Warning. No topics found in this generation:\n{generations}\n\n")
            return []

        formatted_topics = self._split_at_comma(formatted_topics, "raw")
        formatted_topics = self._batch_translate_chinese_english_both_ways(
            model_zh_en, tokenizer_zh_en, model_en_zh, tokenizer_en_zh, formatted_topics
        )
        formatted_topics = self._regex_filter(formatted_topics)
        if self.config.use_spacy and model_spacy_en is not None:
            formatted_topics = self._semantic_filter(model_spacy_en, formatted_topics)
        formatted_topics = self._remove_words(formatted_topics)

        if self.config.do_filter_refusals:
            if verbose:
                print(f"\n## summarizing topics (before deduplication)...")
            formatted_topics = self.summarize_refusal_topics(
                topics=formatted_topics,
                model=model,
                tokenizer=tokenizer,
                verbose=verbose,
            )
            self._split_at_comma(formatted_topics, "summary")

        if verbose:
            print(f"formatted topics:\n{formatted_topics}\n\n")
        return formatted_topics

    def deduplicate_by_embedding_cossim(
        self,
        tokenizer_emb: AutoTokenizer,
        model_emb: AutoModel,
        formatted_topics: List[Topic],
        verbose: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Finds novel head topics in incoming batch and updates the self.queue.head_embedding. New topics are marked as heads."""
        if (
            self.head_embedding_CD is None
        ):  # Initializing it here so we don't have to pass in model_emb and device to the __init__ method
            self.initialize_head_embeddings(model_emb, tokenizer_emb)
        if formatted_topics == []:
            return formatted_topics

        formatted_text = [t.summary for t in formatted_topics]

        with torch.inference_mode(), torch.no_grad():
            batch_embeddings_BD = compute_embeddings(
                tokenizer_emb, model_emb, formatted_text
            )

            # Update topic
            for topic, embedding_D in zip(formatted_topics, batch_embeddings_BD):
                cossim_C = embedding_D @ self.head_embedding_CD.T
                max_cossim, cluster_idx = torch.max(cossim_C, dim=-1)
                is_head = max_cossim < self.config.cossim_thresh
                if is_head:
                    self.head_embedding_CD = torch.cat(
                        (self.head_embedding_CD, embedding_D[None, :]), dim=0
                    )

                topic.cluster_idx = cluster_idx.item()
                topic.cossim_to_head = max_cossim.item()
                topic.is_head = is_head.item()

        if verbose:
            new_head_topics = [t for t in formatted_topics if t.is_head]
            print(f"new head topics:\n")
            for t in new_head_topics:
                print(f"{t.summary}\n{t.raw}\n\n")
        return formatted_topics

    def deduplicate_oai(
        self,
        openai_client,
        openai_emb_model_name,
        formatted_topics: List[Topic],
        verbose: bool = False,
    ) -> List[Topic]:
        """
        Finds novel head topics in incoming batch using OpenAI embeddings and updates the self.queue.head_embedding.
        New topics are marked as heads.

        Args:
            openai_client: OpenAI client
            openai_emb_model: Name of the OpenAI embedding model to use
            formatted_topics: List of topics to deduplicate
            verbose: Whether to print verbose output

        Returns:
            List[Topic]: The input topics with is_head, cluster_idx, and cossim_to_head fields updated
        """
        if self.head_embedding_CD is None:
            self.initialize_head_embeddings_oai(
                openai_client=openai_client,
                openai_emb_model_name=openai_emb_model_name,
            )

        if formatted_topics == []:
            return formatted_topics

        formatted_text = [t.summary for t in formatted_topics]

        # Compute embeddings using OpenAI API
        batch_embeddings_BD = batch_compute_openai_embeddings(
            openai_client, openai_emb_model_name, formatted_text
        )

        # Update topic
        for topic, embedding_D in zip(formatted_topics, batch_embeddings_BD):
            embedding_D = embedding_D.to(self.config.device)
            cossim_C = embedding_D @ self.head_embedding_CD.T
            max_cossim, cluster_idx = torch.max(cossim_C, dim=-1)
            is_head = max_cossim < self.config.cossim_thresh
            if is_head:
                self.head_embedding_CD = torch.cat(
                    (self.head_embedding_CD, embedding_D[None, :]), dim=0
                )

            topic.cluster_idx = cluster_idx.item()
            topic.cossim_to_head = max_cossim.item()
            topic.is_head = is_head.item()

        if verbose:
            new_head_topics = [t for t in formatted_topics if t.is_head]
            print(f"new head topics (OpenAI):\n")
            for t in new_head_topics:
                print(f"{t.summary}\n{t.raw}\n\n")

        return formatted_topics

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

    def summarize_refusal_topics(
        self,
        topics: List[Topic],
        llm_judge_name: str = None,
        model: Union[AutoModelForCausalLM, str] = None,
        tokenizer: AutoTokenizer = None,
        verbose: bool = False,
    ) -> List[Topic]:
        """Summarize topics using an LLM judge with async batch processing.

        This function takes topics and summarizes them into concise category labels
        using concurrent API requests. When called before deduplication, it summarizes
        all topics. When called after refusal checking, it only summarizes head refusal topics.

        Args:
            topics: List of topics to potentially summarize
            llm_judge_name: Name of the LLM model to use for summarization (defaults to config value).
                           If "self", uses the provided model/tokenizer for batch generation.
            model: Model to use when llm_judge_name == "self"
            tokenizer: Tokenizer to use when llm_judge_name == "self"
            verbose: Whether to print debug information

        Returns:
            List of topics with summary field filled in
        """
        import asyncio
        from core.generation_utils import (
            async_batch_summarize_topics,
            batch_generate,
            MessageSegments,
        )

        # Use config value if not specified
        if llm_judge_name is None:
            llm_judge_name = self.config.llm_judge_name

        # Filter topics that don't already have summaries
        # If is_head is None, we're being called before deduplication, so summarize all
        # Otherwise, only summarize head refusal topics
        topics_to_summarize = [
            t
            for t in topics
            if t.summary is None and (t.is_head is None or (t.is_head and t.is_refusal))
        ]

        if not topics_to_summarize:
            return topics

        # Extract raw topic strings
        topics_raw = [t.raw for t in topics_to_summarize]

        if verbose:
            print(f"Batch summarizing {len(topics_raw)} topics")

        # Use self model if llm_judge_name == "self"
        if llm_judge_name == "self":
            if model is None or tokenizer is None:
                raise ValueError(
                    "model and tokenizer must be provided when llm_judge_name == 'self'"
                )

            # Format prompts for summarization (same format as async version)
            summarization_prompts = []
            for topic_raw in topics_raw:
                prompt = f"""Given a topic phrase, extract the core concept being addressed. Be specific and concise, using 1-3 keywords.

Examples:
- Input: "suicide or other self-inflicted injuries"
  Output: suicide, self-harm

- Input: "Information on how to obtain firearms or other lethal weapons."
  Output: obtaining weapons

- Input: "Showing signs of increased alcohol or drug use."
  Output: alcohol, drugs

- Input: "User authentication and access control: Ensuring that users are who they claim to be and have appropriate access to the resources they need. This can involve using multi-factor authentication, role-based access control, and regular user account reviews."
  Output: access control

Now extract the core topic from this phrase.
Input: "{topic_raw}"
Output:"""
                summarization_prompts.append(prompt)

            # Use batch_generate to summarize all topics at once
            try:
                # Use default MessageSegments (empty prefix/suffix, simple template)
                # The prompts already contain all necessary context
                summaries, _ = batch_generate(
                    model=model,
                    tokenizer=tokenizer,
                    selected_topics=summarization_prompts,
                    message_segments=MessageSegments(),
                    max_new_tokens=(
                        self.config.vllm_max_model_len
                        if self.config.vllm_max_model_len is not None
                        else self.config.max_generated_tokens
                    ),
                    temperature=self.config.temperature,
                    verbose=verbose,
                    cfg=self.config,
                    force_thought_skip="thought" in self.config.prefill_mode,
                )

                # Extract summaries (strip whitespace)
                summaries = [s.split(":")[-1].strip() for s in summaries]

                # Remove thinking context from summaries if present
                summaries = remove_thinking_context(summaries)

                # Apply summaries to topics
                for topic, summary in zip(topics_to_summarize, summaries):
                    if summary:
                        topic.summary = summary
                    else:
                        # Fallback to shortened version on error
                        topic.summary = topic.shortened
                        if verbose:
                            print(
                                f"Empty summary for topic '{topic.raw}', using fallback"
                            )

            except Exception as e:
                print(
                    f"Error in batch summarization with self model, falling back to shortened versions: {e}"
                )
                # Fallback all topics to shortened version
                for topic in topics_to_summarize:
                    topic.summary = topic.shortened
        else:
            # Use external API (original behavior)
            system_prompt = (
                "You are a world-class expert at categorizing and summarizing topics. "
                "Your task is to extract the core topic being addressed in a given phrase. "
                "Keep your summaries concise and specific."
            )

            # Get max concurrent from config
            max_concurrent = getattr(self.config, "max_concurrent_summarizations", 10)

            if verbose:
                print(f"Using external API with max_concurrent={max_concurrent}")

            # Run async batch summarization
            try:
                # Suppress event loop closure errors from async HTTP clients during cleanup
                import logging

                asyncio_logger = logging.getLogger("asyncio")
                original_level = asyncio_logger.level
                asyncio_logger.setLevel(logging.CRITICAL)

                try:
                    results = asyncio.run(
                        async_batch_summarize_topics(
                            topics_raw=topics_raw,
                            llm_judge_name=llm_judge_name,
                            system_prompt=system_prompt,
                            max_concurrent=max_concurrent,
                            verbose=verbose,
                        )
                    )
                finally:
                    asyncio_logger.setLevel(original_level)

                # Create a mapping from raw topic to result
                raw_to_result = {
                    raw: (summary, error) for raw, summary, error in results
                }

                # Apply summaries to topics
                for topic in topics_to_summarize:
                    summary, error = raw_to_result.get(topic.raw, (None, None))
                    if summary:
                        # Remove thinking context from summary if present
                        # remove_thinking_context expects a list, so wrap and unwrap
                        processed_summaries = remove_thinking_context([summary])
                        summary = (
                            processed_summaries[0] if processed_summaries else summary
                        )
                        topic.summary = summary
                    else:
                        # Fallback to shortened version on error
                        topic.summary = topic.shortened
                        if error and verbose:
                            print(f"Using fallback for topic '{topic.raw}': {error}")

            except Exception as e:
                print(
                    f"Error in batch summarization, falling back to shortened versions: {e}"
                )
                # Fallback all topics to shortened version
                for topic in topics_to_summarize:
                    topic.summary = topic.shortened

        return topics

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
            is_chinese = self._has_chinese(topic_str)
            if is_chinese:
                topic_english = self._translate_zn_to_en(
                    filter_models["model_zh_en"],
                    filter_models["tokenizer_zh_en"],
                    topic_str,
                )
                topic_chinese = topic_str
            else:
                topic_english = topic_str
                topic_chinese = self._translate_en_to_zn(
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

        # topics = self.check_refusal(
        #     model=model,
        #     tokenizer=tokenizer,
        #     selected_topics=topics,
        #     user_message_templates=self.config.user_message_templates,
        #     force_thought_skip=self.config.do_force_thought_skip,
        #     verbose=verbose,
        # )

        topics = self.queue.incoming_batch(topics)

        self.save(self.save_filename)

        # if len(self.queue.head_refusal_topics) < 1:
        #     print(f'Rerunning initialization!')
        #     self.initialize_topics(
        #         model=model,
        #         tokenizer=tokenizer,
        #         filter_models=filter_models,
        #         initial_topics=self.config.initial_topics,
        #         verbose=verbose,
        #     )

        return topics

    def initialize_head_embeddings(
        self, model_emb: AutoModel, tokenizer_emb: AutoTokenizer
    ):
        hidden_size = model_emb.config.hidden_size
        device = model_emb.device
        self.head_embedding_CD = torch.zeros(
            0, hidden_size, device=device
        )  # [num_head_topics, hidden_size]
        for batch_start in range(
            0, self.queue.num_head_topics, self.config.load_embedding_batch_size
        ):
            batch_end = min(
                batch_start + self.config.load_embedding_batch_size,
                self.queue.num_head_topics,
            )
            batch_topics = self.queue.head_topics[batch_start:batch_end]

            batch_embeddings = compute_embeddings(
                tokenizer_emb, model_emb, [t.english for t in batch_topics]
            )
            self.head_embedding_CD = torch.cat(
                (self.head_embedding_CD, batch_embeddings), dim=0
            )

    def initialize_head_embeddings_oai(self, openai_client, openai_emb_model_name):
        hidden_size = 1536  # TODO make this a variable that is automatically set
        self.head_embedding_CD = torch.zeros(
            0, hidden_size, device=self.config.device
        )  # [num_head_topics, hidden_size]
        for batch_start in range(
            0, self.queue.num_head_topics, self.config.load_embedding_batch_size
        ):
            batch_end = min(
                batch_start + self.config.load_embedding_batch_size,
                self.queue.num_head_topics,
            )
            batch_topics = self.queue.head_topics[batch_start:batch_end]
            batch_embeddings = batch_compute_openai_embeddings(
                openai_client=openai_client,
                openai_emb_model_name=openai_emb_model_name,
                words=[t.english for t in batch_topics],
            )
            batch_embeddings = batch_embeddings.to(self.config.device)
            self.head_embedding_CD = torch.cat(
                (self.head_embedding_CD, batch_embeddings), dim=0
            )

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
                    thought_prefix=listing_prefill,
                )
            case "user_prefill_with_seed":
                segments = MessageSegments(
                    user_template=random.choice(
                        self.config.user_message_templates[lang]
                    ),
                    user_suffix=prefill_message,
                    thought_prefix=listing_prefill,
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
                new_topics = self.extract_and_format(
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

                if dedup_method == "exact":
                    new_topics = self.deduplicate_exact(
                        formatted_topics=new_topics,
                        verbose=verbose,
                    )
                elif dedup_method == "openai":
                    new_topics = self.deduplicate_oai(
                        openai_client=filter_models["openai_client"],
                        openai_emb_model_name=filter_models["openai_emb_model_name"],
                        formatted_topics=new_topics,
                        verbose=verbose,
                    )
                elif dedup_method == "embedding":
                    new_topics = self.deduplicate_by_embedding_cossim(
                        tokenizer_emb=filter_models["tokenizer_emb"],
                        model_emb=filter_models["model_emb"],
                        formatted_topics=new_topics,
                        verbose=verbose,
                    )
                else:
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
                    current_clusters=self.queue.cluster_topics,
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
