import json
import re
import string
from typing import List, Union

from src.exceptions import APITimeoutError

from src.crawler.topic_queue import Topic


def _parse_json_array(text: str) -> List[str] | None:
    """Parse a JSON array from model output. Tolerant of ```json fences."""
    if not text or not text.strip():
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    # Extract first [...] block to survive stray preamble.
    start, end = s.find("["), s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(s[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    return [str(x) for x in parsed]


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
        self.numbered_list_pattern = re.compile(r"(?m)^\d+\.\s*(.*?)$")
        self.chinese_pattern = re.compile(r"[\u4e00-\u9fff]")

    def _extract_from_numbered_list(self, text: str) -> List[str]:
        """Extract topics from a text that contains a numbered list."""
        extracted_list = self.numbered_list_pattern.findall(text)
        extracted_list = list(dict.fromkeys(extracted_list))  # Remove exact duplicates
        extracted_list = extracted_list[:100]  # generous per-response cap; batch-level cap applied later
        extracted_list = [item for item in extracted_list if item is not None]
        return extracted_list

    def _extract_with_model(
        self,
        texts: List[str],
        local_model=None,
        local_tokenizer=None,
        verbose: bool = False,
    ) -> List[List[str]]:
        import asyncio
        import json
        import logging

        if self.config.model.summarization_model == "local":
            from src.generation_utils import batch_generate

            prompts = [
                self.config.topic_extraction_prompt.format(response=t) for t in texts
            ]
            messages = [
                [
                    {
                        "role": "system",
                        "content": "You extract structured data from text. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": p},
                ]
                for p in prompts
            ]
            try:
                responses, _ = batch_generate(
                    model=local_model,
                    tokenizer=local_tokenizer,
                    messages=messages,
                    max_new_tokens=2000,
                    temperature=0.0,
                    verbose=verbose,
                    default_provider=self.config.model.default_provider,
                    provider_url_overrides=self.config.model.provider_urls,
                    provider_concurrency_limits=self.config.model.provider_max_concurrency,
                )
            except json.JSONDecodeError as e:
                # Recoverable: batch_generate returned malformed JSON that
                # we can't parse.  Log a warning with full traceback (not
                # gated by verbose) and return empties — callers rely on
                # list alignment with texts.
                logging.exception(
                    "[extract] local-model JSONDecodeError for batch of %d: %s",
                    len(texts),
                    e,
                )
                return [[] for _ in texts]
            except Exception:
                logging.exception(
                    "[extract] local-model extraction failed with unexpected exception "
                    "for batch of %d",
                    len(texts),
                )
                raise

            all_extracted = []
            for raw in responses:
                json_str = raw.strip()
                if "```" in json_str:
                    parts = json_str.split("```")
                    if len(parts) >= 3:
                        json_str = parts[1]
                        if json_str.startswith("json"):
                            json_str = json_str[4:]
                        json_str = json_str.strip()
                try:
                    topics = json.loads(json_str)
                    if isinstance(topics, list):
                        all_extracted.append(
                            [str(t) for t in topics][:100]  # generous per-response cap; batch-level cap applied later
                        )
                    else:
                        all_extracted.append([])
                except json.JSONDecodeError:
                    if verbose:
                        print(f"Failed to parse extraction JSON: {raw}")
                    all_extracted.append([])
            return all_extracted

        import logging
        import math

        from src.generation_utils import API_MODERATION_SENTINEL, async_query_openrouter
        from src.provider_config import get_provider_client_kwargs
        from src.crawler.config import TOPIC_EXTRACTION_BATCH_PROMPT

        # Resolve provider routing for summarization model
        resolved_model_id, client_kwargs = get_provider_client_kwargs(
            self.config.model.summarization_model,
            self.config.model.default_provider,
            self.config.model.provider_urls,
        )

        # S4b: use universal_backup_model if set, else no fallback
        fallback_models = (
            [self.config.model.universal_backup_model]
            if self.config.model.universal_backup_model
            else []
        )

        system_prompt = (
            "You extract structured data from text. "
            "Always respond with valid JSON only."
        )

        async def _call_one_chunk(chunk_texts):
            """Call the extraction API for a single chunk.

            Returns:
              List[List[str]]  on success (shape matches chunk_texts)
              None             on recoverable failure → caller should split-retry
              [[] for _ in chunk_texts]  on transport exception → don't retry
            """
            # Build the batch prompt for this chunk
            responses_block = "\n\n".join(
                f"Response {i}:\n{t}" for i, t in enumerate(chunk_texts)
            )
            prompt = TOPIC_EXTRACTION_BATCH_PROMPT.format(
                n=len(chunk_texts),
                n_minus_1=len(chunk_texts) - 1,
                responses_block=responses_block,
            )
            max_tokens = min(8000, max(1000, 150 * len(chunk_texts)))

            try:
                response = await async_query_openrouter(
                    model_name=resolved_model_id,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    client_kwargs=client_kwargs,
                    fallback_models=fallback_models,
                    default_provider=self.config.model.default_provider,
                    provider_url_overrides=self.config.model.provider_urls,
                    prefer_nitro=self.config.model.prefer_nitro,
                )
            except Exception as e:
                # 403 moderation from the provider → recoverable (split may
                # isolate the offending text). openrouter_utils raises the
                # APIStatusError directly for 4xx; detect via duck-typed
                # status_code attribute so we don't couple to the SDK import.
                status_code = getattr(e, "status_code", None)
                if status_code == 403:
                    if verbose:
                        print(
                            f"[extract] 403 moderation for chunk of "
                            f"{len(chunk_texts)}, will split"
                        )
                    return None
                # APITimeoutError: re-raise so the outer handler (S0f) fires
                # and the caller surfaces the loss rather than silently
                # discarding extracted topics.
                if isinstance(e, APITimeoutError):
                    logging.warning(
                        "[extract] APITimeoutError for chunk of %d — re-raising",
                        len(chunk_texts),
                    )
                    raise
                # All other unexpected exceptions: log full traceback then
                # re-raise.  Do NOT silently return empties — that discards
                # topics with no visible signal.
                logging.exception(
                    "[extract] unexpected exception for chunk of %d: %s",
                    len(chunk_texts),
                    e,
                )
                raise

            # Moderation block via sentinel string (some providers return a
            # string instead of raising) → recoverable.
            if response and response.startswith(API_MODERATION_SENTINEL):
                if verbose:
                    print(f"[extract] moderation sentinel for chunk of {len(chunk_texts)}, will split")
                return None

            if not response:
                if verbose:
                    print(f"[extract] empty response for chunk of {len(chunk_texts)}, will split")
                return None

            # Parse the JSON array-of-arrays response.
            # Tolerate ```json fences by extracting the outermost [...] block.
            raw = response.strip()
            s = raw
            if s.startswith("```"):
                s = s.strip("`")
                if s.lower().startswith("json"):
                    s = s[4:]
                s = s.strip()
            start, end = s.find("["), s.rfind("]")
            if start == -1 or end == -1 or end <= start:
                if verbose:
                    print(f"[extract] no JSON array in chunk of {len(chunk_texts)}: {raw[:200]}")
                return None  # recoverable

            try:
                parsed = json.loads(s[start : end + 1])
            except (json.JSONDecodeError, ValueError) as e:
                if verbose:
                    print(f"[extract] JSON parse error for chunk of {len(chunk_texts)}: {e}")
                return None  # recoverable

            if not isinstance(parsed, list) or len(parsed) != len(chunk_texts):
                if verbose:
                    print(
                        f"[extract] shape mismatch for chunk of {len(chunk_texts)}: "
                        f"got {type(parsed).__name__} of length "
                        f"{len(parsed) if isinstance(parsed, list) else 'N/A'}"
                    )
                return None  # recoverable — the N=20 failure mode

            return [
                [str(t) for t in row][:100] if isinstance(row, list) else []
                for row in parsed
            ]

        # Clamp K so K <= len(texts) (avoids over-chunking on tiny inputs)
        K = max(1, min(self.config.crawler.extraction_batch_size, len(texts)))
        max_depth = math.ceil(math.log2(max(K, 2)))

        async def _extract_with_split_retry(chunk_texts, depth=0):
            result = await _call_one_chunk(chunk_texts)
            if result is not None:
                return result
            # Recoverable failure — try splitting unless we've bottomed out
            if len(chunk_texts) == 1 or depth >= max_depth:
                if verbose:
                    print(
                        f"[extract] split-retry bottomed out at depth={depth}, "
                        f"len={len(chunk_texts)}; returning empty"
                    )
                return [[] for _ in chunk_texts]
            mid = len(chunk_texts) // 2
            left = await _extract_with_split_retry(chunk_texts[:mid], depth + 1)
            right = await _extract_with_split_retry(chunk_texts[mid:], depth + 1)
            return left + right

        async def _run_all():
            chunks = [texts[i : i + K] for i in range(0, len(texts), K)]
            out = []
            for chunk in chunks:
                out.extend(await _extract_with_split_retry(chunk))
            return out

        asyncio_logger = logging.getLogger("asyncio")
        original_level = asyncio_logger.level
        asyncio_logger.setLevel(logging.CRITICAL)
        try:
            return asyncio.run(_run_all())
        finally:
            asyncio_logger.setLevel(original_level)

    def _has_chinese(self, text: str) -> bool:
        """Check if the text contains Chinese characters."""
        return bool(self.chinese_pattern.search(text))

    def _translate_batch(
        self,
        translation_model,
        translation_tokenizer,
        texts: List[str],
        direction: str,
    ) -> List[str]:
        """JSON-in/JSON-out translation: one API call for N topics.

        On parse failure or length mismatch, retries once with the bad output
        as evidence. Final fallback is per-item translation (one call per item)
        for correctness when batching catastrophically fails.
        """
        from src.generation_utils import batch_generate

        if not texts:
            return []

        if direction == "zn_to_en":
            instruction = (
                "Translate each string in this JSON array to English. "
                "Return ONLY a JSON array of EXACTLY the same length, in the same order, "
                "with no extra text, code fences, or commentary."
            )
        else:
            instruction = (
                "将此 JSON 数组中的每个字符串翻译成中文。"
                "只返回一个长度相同、顺序相同的 JSON 数组，"
                "不要添加任何其他文字、代码块标记或说明。"
            )

        def call_once(payload_texts: List[str]) -> List[str] | None:
            prompt = f"{instruction}\n\n{json.dumps(payload_texts, ensure_ascii=False)}"
            responses, _ = batch_generate(
                translation_model,
                translation_tokenizer,
                [[{"role": "user", "content": prompt}]],
                max_new_tokens=max(500, len(payload_texts) * 100),
                temperature=0.3,
                default_provider=self.config.model.default_provider,
                provider_url_overrides=self.config.model.provider_urls,
                provider_concurrency_limits=self.config.model.provider_max_concurrency,
                prefer_nitro=self.config.model.prefer_nitro,
            )
            return _parse_json_array(responses[0]) if responses else None

        parsed = call_once(texts)
        if parsed is None or len(parsed) != len(texts):
            # One retry telling the model explicitly what went wrong.
            bad = "(empty)" if parsed is None else json.dumps(parsed, ensure_ascii=False)
            prompt = (
                f"{instruction}\n\n"
                f"Input ({len(texts)} items): {json.dumps(texts, ensure_ascii=False)}\n\n"
                f"Your previous response was invalid "
                f"(expected {len(texts)} items, got {'unparseable' if parsed is None else len(parsed)}):\n"
                f"{bad}\n\n"
                f"Please return a corrected JSON array of exactly {len(texts)} items."
            )
            responses, _ = batch_generate(
                translation_model,
                translation_tokenizer,
                [[{"role": "user", "content": prompt}]],
                max_new_tokens=max(500, len(texts) * 100),
                temperature=0.3,
                default_provider=self.config.model.default_provider,
                provider_url_overrides=self.config.model.provider_urls,
                provider_concurrency_limits=self.config.model.provider_max_concurrency,
                prefer_nitro=self.config.model.prefer_nitro,
            )
            parsed = _parse_json_array(responses[0]) if responses else None

        if parsed is not None and len(parsed) == len(texts):
            return [str(t).strip() if str(t).strip() else src for t, src in zip(parsed, texts)]

        # Fallback: per-item translation (one call each — correctness over cost).
        if direction == "zn_to_en":
            per_prompt = lambda t: f"Translate to English (translation only): {t}"
        else:
            per_prompt = lambda t: f"翻译成中文（只输出翻译）：{t}"
        messages = [[{"role": "user", "content": per_prompt(t)}] for t in texts]
        responses, _ = batch_generate(
            translation_model,
            translation_tokenizer,
            messages,
            max_new_tokens=500,
            temperature=0.0,
            default_provider=self.config.model.default_provider,
            provider_url_overrides=self.config.model.provider_urls,
            provider_concurrency_limits=self.config.model.provider_max_concurrency,
            prefer_nitro=self.config.model.prefer_nitro,
        )
        return [r.strip() if r.strip() else src for r, src in zip(responses, texts)]

    def _translate_zn_to_en(
        self,
        local_model,
        local_tokenizer,
        inputs: Union[str, List[str]],
    ) -> Union[str, List[str]]:
        translation_model, translation_tokenizer = self._resolve_model(
            "translation", local_model, local_tokenizer
        )
        is_single = isinstance(inputs, str)
        texts = [inputs] if is_single else inputs
        translated = self._translate_batch(
            translation_model, translation_tokenizer, texts, direction="zn_to_en"
        )
        return translated[0] if is_single else translated

    def _translate_en_to_zn(
        self,
        local_model,
        local_tokenizer,
        inputs: Union[str, List[str]],
    ) -> Union[str, List[str]]:
        translation_model, translation_tokenizer = self._resolve_model(
            "translation", local_model, local_tokenizer
        )
        is_single = isinstance(inputs, str)
        texts = [inputs] if is_single else inputs
        translated = self._translate_batch(
            translation_model, translation_tokenizer, texts, direction="en_to_zn"
        )
        return translated[0] if is_single else translated

    def _resolve_model(self, role: str, local_model, local_tokenizer):
        """Return (model, tokenizer) for the given role.

        If the role's model config is "local", returns the local model/tokenizer.
        Otherwise returns the OpenRouter model name string (with None tokenizer).
        """
        model_name = getattr(self.config.model, f"{role}_model")
        if model_name == "local":
            return local_model, local_tokenizer
        else:
            return model_name, None

    def _batch_translate_chinese_english_both_ways(
        self,
        local_model,
        local_tokenizer,
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
            0, len(chinese_topics), self.config.crawler.generation_batch_size
        ):
            batch_end = batch_start + self.config.crawler.generation_batch_size
            chinese_topic_B = chinese_topics[batch_start:batch_end]
            chinese_indices_B = chinese_indices[batch_start:batch_end]
            chinese_raw_B = [t.raw for t in chinese_topic_B]
            translated_str_B = self._translate_zn_to_en(
                local_model, local_tokenizer, chinese_raw_B
            )
            for original, translation, idx in zip(
                chinese_raw_B, translated_str_B, chinese_indices_B
            ):
                topics[idx].english = translation
                topics[idx].shortened = translation  # copy of english
                topics[idx].chinese = original

        for batch_start in range(
            0, len(english_topics), self.config.crawler.generation_batch_size
        ):
            batch_end = batch_start + self.config.crawler.generation_batch_size
            english_topic_B = english_topics[batch_start:batch_end]
            english_indices_B = english_indices[batch_start:batch_end]
            english_raw_B = [t.raw for t in english_topic_B]
            translated_str_B = self._translate_en_to_zn(
                local_model, local_tokenizer, english_raw_B
            )
            for original, translation, idx in zip(
                english_raw_B, translated_str_B, english_indices_B
            ):
                topics[idx].english = original
                topics[idx].shortened = original  # copy of english
                topics[idx].chinese = translation
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


    def _split_at_comma(
        self,
        topics: List[Topic],
        attribute: str,
    ) -> List[Topic]:
        relevant_attributes = ("raw", "summary", "chinese", "english", "shortened")
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
                    item = re.sub(r"^(?:or|and)\s+", "", item.strip())
                    new_topic_kwargs = {
                        "parent_id": topic.parent_id,
                        attribute: item.strip(),
                        "is_chinese": topic.is_chinese,
                    }
                    # Keep other relevant attributes
                    for a in relevant_attributes:
                        if a != attribute:
                            new_topic_kwargs[a] = getattr(topic, a)
                    # When splitting a summary, the item IS the English label
                    if attribute == "summary":
                        new_topic_kwargs["english"] = item.strip()
                        new_topic_kwargs["shortened"] = item.strip()
                    else:
                        new_topic_kwargs["english"] = topic.english
                        new_topic_kwargs["shortened"] = topic.shortened
                    topics.append(Topic(**new_topic_kwargs))

        return topics

    def deduplicate_exact(
        self,
        formatted_topics: List[Topic],
        head_topics: List[Topic],
        verbose: bool = False,
    ) -> List[Topic]:
        """
        Finds novel head topics in incoming batch by checking for exact duplicates
        (after normalization: lowercase + strip punctuation) in topic.summary.
        New topics are marked as heads.

        Args:
            formatted_topics: List of topics to deduplicate
            head_topics: Existing head topics to compare against
            verbose: Whether to print verbose output

        Returns:
            List[Topic]: The input topics with is_head, cluster_idx, and cossim_to_head fields updated
        """
        if formatted_topics == []:
            return formatted_topics

        def normalize_summary(text) -> str:
            if text is None:
                return ""
            if isinstance(text, list):
                text = " ".join(str(item) for item in text if item)
            if not isinstance(text, str):
                text = str(text)
            text_lower = text.lower()
            translator = str.maketrans("", "", string.punctuation)
            normalized = text_lower.translate(translator)
            return normalized

        # Build lookup dictionary: normalized_summary -> cluster_idx
        normalized_to_cluster_idx = {}
        for idx, head_topic in enumerate(head_topics):
            normalized_summary = normalize_summary(head_topic.summary)
            if normalized_summary:
                normalized_to_cluster_idx[normalized_summary] = idx

        # Process each topic
        for topic in formatted_topics:
            normalized_summary = normalize_summary(topic.summary)

            if normalized_summary and normalized_summary in normalized_to_cluster_idx:
                cluster_idx = normalized_to_cluster_idx[normalized_summary]
                topic.is_head = False
                topic.cluster_idx = cluster_idx
            else:
                topic.is_head = True
                topic.cluster_idx = len(head_topics)
                if normalized_summary:
                    normalized_to_cluster_idx[normalized_summary] = topic.cluster_idx

        if verbose:
            new_head_topics = [t for t in formatted_topics if t.is_head]
            print(f"new head topics (exact deduplication):\n")
            for t in new_head_topics:
                print(f"{t.summary}\n{t.raw}\n\n")

        return formatted_topics

    def extract_and_format(
        self,
        local_model,
        local_tokenizer,
        input_strs: List[str],
        generations: List[str],
        parent_ids: List[int],
        verbose: bool = False,
    ) -> List[Topic]:

        formatted_topics = []
        parent_ids = parent_ids * (len(generations) // len(parent_ids))
        assert len(parent_ids) == len(generations)

        try:
            all_extracted_items = self._extract_with_model(
                generations,
                local_model=local_model,
                local_tokenizer=local_tokenizer,
                verbose=verbose,
            )
        except APITimeoutError:
            # Belt-and-braces guard: _extract_with_model propagates
            # APITimeoutError from _call_one_chunk (S0h fix). Re-raise
            # loudly — swallowing silently loses all topics for this step.
            import logging
            logging.warning(
                "[extract_and_format] APITimeoutError propagated from inner "
                "extraction — re-raising to caller"
            )
            raise

        for extracted_items, prompt, pid in zip(
            all_extracted_items, input_strs, parent_ids
        ):
            for item in extracted_items:
                formatted_topics.append(Topic(raw=item, parent_id=pid, prompt=prompt))

        # Batch-level cap: apply to the flat list after all responses are concatenated
        formatted_topics = formatted_topics[:self.config.crawler.max_topics_per_step_lang]

        if len(formatted_topics) == 0:
            print(f"Warning. No topics found in this generation:\n{generations}\n\n")
            return []

        formatted_topics = self._batch_translate_chinese_english_both_ways(
            local_model, local_tokenizer, formatted_topics
        )
        formatted_topics = self._regex_filter(formatted_topics)

        if self.config.crawler.do_filter_refusals:
            if verbose:
                print(f"\n## summarizing topics (before deduplication)...")
            formatted_topics = self.summarize_refusal_topics(
                topics=formatted_topics,
                local_model=local_model,
                local_tokenizer=local_tokenizer,
                verbose=verbose,
            )
            self._split_at_comma(formatted_topics, "summary")
            # Drop topics the summarizer flagged as non-meaningful
            formatted_topics = [t for t in formatted_topics if t.summary is not None]

        if verbose:
            print(f"formatted topics:\n{formatted_topics}\n\n")
        return formatted_topics


    def extract_and_translate(
        self,
        local_model,
        local_tokenizer,
        input_strs: List[str],
        generations: List[str],
        parent_ids: List[int],
        verbose: bool = False,
    ) -> List[Topic]:
        """Extract topics from generations and translate, but skip summarization.

        Same as extract_and_format but without the summarization step.
        Used by AggregateCrawler which does its own grouping-based summarization.
        """
        formatted_topics = []
        parent_ids = parent_ids * (len(generations) // len(parent_ids))
        assert len(parent_ids) == len(generations)

        try:
            all_extracted_items = self._extract_with_model(
                generations,
                local_model=local_model,
                local_tokenizer=local_tokenizer,
                verbose=verbose,
            )
        except APITimeoutError:
            # Belt-and-braces guard: _extract_with_model propagates
            # APITimeoutError from _call_one_chunk (S0h fix). Re-raise
            # loudly — swallowing silently loses all topics for this step.
            import logging
            logging.warning(
                "[extract_and_translate] APITimeoutError propagated from inner "
                "extraction — re-raising to caller"
            )
            raise

        for extracted_items, prompt, pid in zip(
            all_extracted_items, input_strs, parent_ids
        ):
            for item in extracted_items:
                formatted_topics.append(Topic(raw=item, parent_id=pid, prompt=prompt))

        # Batch-level cap: apply to the flat list after all responses are concatenated
        formatted_topics = formatted_topics[:self.config.crawler.max_topics_per_step_lang]

        if len(formatted_topics) == 0:
            print(f"Warning. No topics found in this generation:\n{generations}\n\n")
            return []

        formatted_topics = self._batch_translate_chinese_english_both_ways(
            local_model, local_tokenizer, formatted_topics
        )
        formatted_topics = self._regex_filter(formatted_topics)

        if verbose:
            print(f"formatted topics (no summarization):\n{formatted_topics}\n\n")
        return formatted_topics

    def summarize_refusal_topics(
        self,
        topics: List[Topic],
        local_model=None,
        local_tokenizer=None,
        verbose: bool = False,
    ) -> List[Topic]:
        """Summarize topics using the configured summarization_model.

        Routes to vLLM (batch_generate) when summarization_model is "local",
        or to async OpenRouter API otherwise.

        Args:
            topics: List of topics to potentially summarize
            local_model: Local vLLM model (used when summarization_model == "local")
            local_tokenizer: Local tokenizer (used when summarization_model == "local")
            verbose: Whether to print debug information

        Returns:
            List of topics with summary field filled in
        """
        import asyncio
        import logging

        from src.generation_utils import (
            async_batch_summarize_topics,
            batch_generate,
        )

        summarization_model = self.config.model.summarization_model

        # Filter topics that don't already have summaries
        # If is_head is None, we're being called before deduplication, so summarize all
        # Otherwise, only summarize head refusal topics
        topics_to_summarize = []
        for t in topics:
            if t.summary is not None or not (
                t.is_head is None or (t.is_head and t.is_refusal)
            ):
                continue
            word_count = len(t.raw.split()) if t.raw else 0
            if word_count <= 7:
                # Short topics are already concise — use as-is, skip LLM call
                t.summary = t.shortened or t.raw
            else:
                topics_to_summarize.append(t)

        if not topics_to_summarize:
            return topics

        # Extract raw topic strings
        topics_raw = [t.raw for t in topics_to_summarize]

        if verbose:
            print(f"Batch summarizing {len(topics_raw)} topics")

        if summarization_model == "local":
            model, tokenizer = local_model, local_tokenizer

            # Format prompts for summarization
            from src.crawler.config import TOPIC_SUMMARIZATION_PROMPT

            summarization_prompts = [
                TOPIC_SUMMARIZATION_PROMPT.format(topic_raw=topic_raw)
                for topic_raw in topics_raw
            ]

            # Use batch_generate to summarize all topics at once
            try:
                summarization_messages = [
                    [{"role": "user", "content": p}] for p in summarization_prompts
                ]
                summaries, _ = batch_generate(
                    model=model,
                    tokenizer=tokenizer,
                    messages=summarization_messages,
                    max_new_tokens=(
                        self.config.model.vllm_max_model_len
                        if self.config.model.vllm_max_model_len is not None
                        else self.config.crawler.max_generated_tokens
                    ),
                    temperature=self.config.model.temperature,
                    verbose=verbose,
                    default_provider=self.config.model.default_provider,
                    provider_url_overrides=self.config.model.provider_urls,
                    provider_concurrency_limits=self.config.model.provider_max_concurrency,
                )

                # Extract summaries (strip whitespace)
                summaries = [s.split(":")[-1].strip() for s in summaries]

                # Remove thinking context from summaries if present
                summaries = remove_thinking_context(summaries)

                # Apply summaries to topics
                for topic, summary in zip(topics_to_summarize, summaries):
                    if summary and summary.lower() != "[skip]":
                        topic.summary = summary
                    elif summary.lower() == "[skip]":
                        topic.summary = None  # will be filtered downstream
                    else:
                        # Fallback to shortened version on error
                        topic.summary = topic.shortened
                        if verbose:
                            print(
                                f"Empty summary for topic '{topic.raw}', using fallback"
                            )

            except APITimeoutError as e:
                # Recoverable transient failure: log warning and fall back to
                # shortened label so downstream filtering still works.
                logging.warning(
                    "[summarize] APITimeoutError in local-model batch "
                    "summarization — falling back to shortened labels: %s",
                    e,
                )
                for topic in topics_to_summarize:
                    topic.summary = topic.shortened
            except Exception:
                # Unexpected (OOM, auth error, config bug) — re-raise so the
                # caller sees a real error rather than silently getting wrong data.
                logging.exception(
                    "[summarize] unexpected error in local-model batch summarization"
                )
                raise
        else:
            # Use OpenRouter API
            system_prompt = (
                "You extract concise topic labels from phrases taken from a model's refusal list. "
                "If a phrase covers multiple distinct topics, output them as comma-separated labels (2-8 words each). "
                "If the phrase is a sentence fragment, preamble, or generic capability description rather than a meaningful refusal topic, output exactly: [skip]. "
                "Output only the label(s) — no explanation, no preamble."
            )

            max_concurrent = self.config.crawler.max_concurrent_summarizations

            # Resolve provider routing for summarization model
            from src.provider_config import get_provider_client_kwargs
            resolved_summ_id, summ_client_kwargs = get_provider_client_kwargs(
                summarization_model,
                self.config.model.default_provider,
                self.config.model.provider_urls,
            )

            if verbose:
                print(
                    f"Using API (model={resolved_summ_id}, max_concurrent={max_concurrent})"
                )

            try:
                results = asyncio.run(
                    async_batch_summarize_topics(
                        topics_raw=topics_raw,
                        llm_judge_name=resolved_summ_id,
                        system_prompt=system_prompt,
                        max_concurrent=max_concurrent,
                        verbose=verbose,
                        client_kwargs=summ_client_kwargs,
                    )
                )

                # Create a mapping from raw topic to result
                raw_to_result = {
                    raw: (summary, error) for raw, summary, error in results
                }

                # Apply summaries to topics
                for topic in topics_to_summarize:
                    summary, error = raw_to_result.get(topic.raw, (None, None))
                    if summary:
                        processed_summaries = remove_thinking_context([summary])
                        summary = (
                            processed_summaries[0] if processed_summaries else summary
                        )
                        if summary.lower() == "[skip]":
                            topic.summary = None  # will be filtered downstream
                        else:
                            topic.summary = summary
                    else:
                        topic.summary = topic.shortened
                        if error and verbose:
                            print(f"Using fallback for topic '{topic.raw}': {error}")

            except APITimeoutError as e:
                # Recoverable transient failure: log warning and fall back to
                # shortened label so downstream filtering still works.
                logging.warning(
                    "[summarize] APITimeoutError in API batch "
                    "summarization — falling back to shortened labels: %s",
                    e,
                )
                for topic in topics_to_summarize:
                    topic.summary = topic.shortened
            except Exception:
                # Unexpected (OOM, auth error, config bug) — re-raise so the
                # caller sees a real error rather than silently getting wrong data.
                logging.exception(
                    "[summarize] unexpected error in API batch summarization"
                )
                raise

        return topics
