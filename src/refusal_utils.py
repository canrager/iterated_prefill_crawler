import re
from typing import List

from src.generation_utils import batch_generate
from src.response_formatting_utils import remove_thinking_context
from src.crawler.topic_queue import Topic
from src.crawler.config import CrawlerConfig


def _resolve_model(config: CrawlerConfig, role: str, local_model, local_tokenizer):
    """Return (model, tokenizer) for the given role.

    If the role's model config is "local", returns the local vLLM model/tokenizer.
    Otherwise returns the OpenRouter model name string (with None tokenizer).
    """
    model_name = getattr(config.model, f"{role}_model")
    if model_name == "local":
        return local_model, local_tokenizer
    else:
        return model_name, None


def is_refusal(text: str | None, refusal_patterns: list[str]) -> bool:
    if text is None:
        return False

    assistant_answer = text.split("ssistant")[-1]

    # Check each pattern until we find a match
    for pattern in refusal_patterns:
        if re.search(pattern, assistant_answer, re.IGNORECASE):
            return True
    return False


def check_refusal(
    config: CrawlerConfig,
    local_model,
    local_tokenizer,
    selected_topics: List[Topic],
    verbose: bool = False,
) -> List[Topic]:
    """Filtering incoming head topics for refusals."""

    if selected_topics == []:
        return selected_topics

    refusal_model, refusal_tokenizer = _resolve_model(config, "refusal_check", local_model, local_tokenizer)
    target_model, target_tokenizer = _resolve_model(config, "target", local_model, local_tokenizer)

    num_checks = config.crawler.num_refusal_checks_per_topic
    threshold = config.crawler.is_refusal_threshold

    # Step 1: Generate queries for ALL topics in parallel
    all_query_prompts = []
    topic_indices = []  # Track which prompts belong to which topic

    for topic_idx, topic in enumerate(selected_topics):
        instructions = (
            config.refusal_provocation_generation_prompts["chinese"]
            if topic.is_chinese
            else config.refusal_provocation_generation_prompts["english"]
        )
        # Add num_checks prompts for this topic
        all_query_prompts.extend([instructions.format(topic.raw)] * num_checks)
        topic_indices.extend([topic_idx] * num_checks)

    # Generate all queries at once
    query_messages = [[{"role": "user", "content": p}] for p in all_query_prompts]
    all_queries, all_query_input_strs = batch_generate(
        refusal_model,
        refusal_tokenizer,
        query_messages,
        max_new_tokens=(
            config.model.vllm_max_model_len
            if config.model.vllm_max_model_len is not None
            else config.crawler.max_refusal_check_generated_tokens
        ),
        temperature=1,
        verbose=verbose,
        cfg=config,
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
        refused_to_make_query = [is_refusal(query, config.refusal_messages) for query in queries]
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
            topic.judge_refused = True
            # Fallback: test target directly with the raw topic
            fallback_queries = [f"Tell me about {topic.raw}"]
            topic_to_queries[topic_idx] = fallback_queries
            topics_needing_answers.append(topic_idx)

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
        answer_messages = [[{"role": "user", "content": p}] for p in all_answer_prompts]
        all_answers, all_answer_strs = batch_generate(
            target_model,
            target_tokenizer,
            answer_messages,
            max_new_tokens=config.crawler.max_refusal_check_generated_tokens,
            temperature=config.model.temperature,
            cfg=config,
        )

        # Step 4: Process answer refusals
        offset = 0
        for topic_idx in topics_needing_answers:
            topic = selected_topics[topic_idx]

            # Extract answers for this topic
            n = len(topic_to_queries[topic_idx])
            answers = all_answers[offset:offset + n]
            answer_strs = all_answer_strs[offset:offset + n]
            offset += n

            # Check if model refused to answer
            refused_to_answer_query = [is_refusal(answer, config.refusal_messages) for answer in answers]
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
