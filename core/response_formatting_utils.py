import re
from typing import List, Union

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from spacy.language import Language

from core.topic_queue import Topic


def remove_thinking_context(queries: List[str]) -> List[str]:
    """
    Remove thinking context from queries that end with </think> or <\\think>.

    For each query, if it contains a thinking context ending pattern,
    removes everything from the start up to and including the pattern.

    Args:
        queries: List of query strings that may contain thinking context

    Returns:
        List of queries with thinking context removed
    """
    processed_queries = []
    think_begin_pattern = "<think>"
    think_end_pattern = "</think>"

    for query in queries:
        processed_query = query
        if think_end_pattern in query:
            # Find the position of the pattern and remove everything up to and including it
            pattern_pos = query.find(think_end_pattern)
            if pattern_pos != -1:
                # Remove everything from start to end of pattern (inclusive)
                processed_query = query[pattern_pos + len(think_end_pattern) :].lstrip()

        elif think_begin_pattern in query:
            # Incomplete rollout, thought has started but not ended
            processed_query = ""

        processed_queries.append(processed_query)

    return processed_queries


class TopicFormatter:
    def __init__(self, config):
        self.config = config
        self.numbered_list_pattern = re.compile(r"\d+\.\s*(.*?)(?:\n|$)")
        self.chinese_pattern = re.compile(r"[\u4e00-\u9fff]")

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
            item = item.strip(" ./:\\\",'()[]")
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
