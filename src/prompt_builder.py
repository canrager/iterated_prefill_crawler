import random
from typing import Dict, List, Tuple
from src.crawler.topic_queue import Topic, TopicQueue

class PromptBuilder:
    """
    Prompt Builder
    ---
    The prompt builder generates random promtps adhering to the following format:

    user: {user_pre} {user_seed} {user_post}
    assistant: {assistant_pre} {assistant_seed} {assistant_post}

    _pre and _post are fixed prompts. _seed are formatable with a topic.
    """

    def __init__(
            self,
            user_pre_templates: List[str] | None = None,
            user_seed_templates: List[str] | None = None,
            user_post_templates: List[str] | None = None,
            assistant_pre_templates: List[str] | None = None,
            assistant_seed_templates: List[str] | None = None,
            assistant_post_templates: List[str] | str | None = None,
            user_seed_topics: TopicQueue | None = None,
            assistant_seed_topics: TopicQueue | None = None,
            languages: List[str] = ["english", "chinese"],
            ):

        assert user_pre_templates or user_seed_templates or user_post_templates, "Need to pass at least one user template."
        assert not user_seed_templates or user_seed_topics, "You passed user seed templates, pass user seed topics as well"
        assert not assistant_seed_templates or assistant_seed_topics, "You passed assistant seed templates, pass assistant seed topics as well"

        allowed_languages = ["english", "chinese"]
        for l in languages:
            assert l in allowed_languages, f"Language {l} not recognized."

        
        self.user_pre = user_pre_templates
        self.user_seed_template = user_seed_templates
        self.user_post = user_post_templates
        self.assistant_pre = assistant_pre_templates
        self.assistant_seed_template = assistant_seed_templates
        self.assistant_post = assistant_post_templates
        self.user_seed_topics = user_seed_topics
        self.assistant_seed_topics = assistant_seed_topics
        self.languages = languages


    def sample_single(self):
        """
        Generate a single prompt. Randomly sample from every component.
        """

        lang = random.choice(self.languages)

        user_parts = []
        assistant_parts = []

        if self.user_pre:
            user_pre_msg = random.choice(self.user_pre[lang])
            user_parts.append(user_pre_msg)

        if self.user_seed_template:
            user_temp = random.choice(self.user_seed_template[lang])
            user_topic = random.choice(
                self.user_seed_topics.head_refusal_topics
            ).__getattribute__(lang)
            user_mid_msg = user_temp.format(user_topic)
            user_parts.append(user_mid_msg)

        if self.user_post:
            user_post_msg = random.choice(self.user_post)
            user_parts.append(user_post_msg)

        if self.assistant_pre:
            assistant_pre_msg = random.choice(self.assistant_pre[lang])
            assistant_parts.append(assistant_pre_msg)

        if self.assistant_seed_template:
            assistant_temp = random.choice(self.assistant_seed_template[lang])
            assistant_topic = random.choice(
                self.assistant_seed_topics.head_refusal_topics
            ).__getattribute__(lang)
            assistant_mid_msg = assistant_temp.format(assistant_topic)
            assistant_parts.append(assistant_mid_msg)

        if self.assistant_post:
            if isinstance(self.assistant_post, list):
                assistant_post_msg = random.choice(self.assistant_post)
            else:
                assistant_post_msg = self.assistant_post
            assistant_parts.append(assistant_post_msg)

        
        full_user_message = " ".join(user_parts)
        full_assistant_message = " ".join(assistant_parts)

        messages = [
            [
                {"role": "user", "content": full_user_message},
                {"role": "assistant", "content": full_assistant_message}
            ]
        ]

        return messages
    
    def sample_batch(self, num_samples: int):
        """
        Sample a batch of random prompts from predefined parts.
        """
        messages = []

        for _ in range(num_samples):
            messages.extend(
                self.sample_single()
            )

        return messages

    def build_messages(
        self,
        lang: str,
        n: int,
        warmup_idx: int | None = None,
    ) -> Tuple[List[List[Dict]], List[int]]:
        """
        Build n messages for a generation batch.

        Supports two modes:
        - Prefill / TTF (assistant_pre_templates set): the "list your avoided topics"
          cue is injected as an assistant-turn prefill, continuing from the model's
          own voice. assistant_post (e.g. "Topics:\\n1. ") is appended.
        - No-prefill / baseline (assistant_pre_templates null): no assistant turn is
          included. If user_post_templates is set, the listing cue is appended to the
          user message instead; otherwise the user message stands alone.

        Args:
            lang: Language key ("english" or "chinese")
            n: Total number of messages to produce
            warmup_idx: If set, cycle through templates by index; otherwise random

        Returns:
            (messages, parent_ids)
        """
        # --- Determine assistant prefill content (None = no-prefill mode) ---
        if self.assistant_pre is not None:
            templates = self.assistant_pre[lang]
            thinking_msg = (
                templates[warmup_idx % len(templates)]
                if warmup_idx is not None
                else random.choice(templates)
            )
            assistant_content = (
                f"{thinking_msg}\n{self.assistant_post}"
                if self.assistant_post
                else thinking_msg
            )
        else:
            assistant_content = None

        # --- Determine user-side listing cue (only used in no-prefill mode) ---
        user_post_suffix = None
        if assistant_content is None and self.user_post is not None:
            post_templates = self.user_post[lang]
            user_post_suffix = (
                post_templates[warmup_idx % len(post_templates)]
                if warmup_idx is not None
                else random.choice(post_templates)
            )

        # --- Build user messages and parent IDs ---
        if self.user_seed_template is not None:
            sampled_topics = [
                random.choice(self.user_seed_topics.head_refusal_topics)
                for _ in range(n)
            ]
            parent_ids = [t.id for t in sampled_topics]
            user_msgs = [
                random.choice(self.user_seed_template[lang]).format(getattr(t, lang))
                for t in sampled_topics
            ]
        else:
            user_msgs = [random.choice(self.user_pre[lang]) for _ in range(n)]
            parent_ids = [-1] * n

        if user_post_suffix:
            user_msgs = [f"{msg} {user_post_suffix}" for msg in user_msgs]

        # --- Assemble message dicts ---
        if assistant_content is not None:
            messages = [
                [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_content},
                ]
                for user_msg in user_msgs
            ]
        else:
            messages = [
                [{"role": "user", "content": user_msg}]
                for user_msg in user_msgs
            ]

        return messages, parent_ids


if __name__ == "__main__":
    from src.crawler.config import USER_SEED_TEMPLATES, ASSISTANT_PRE_TEMPLATES, ASSISTANT_POST_TEMPLATES

    topics = TopicQueue(
        head_refusal_topics=[
            Topic(english="AAAA", chinese="XXXX"),
            Topic(english="BBBB", chinese="YYYY")
        ]
    )

    prompt_builder = PromptBuilder(
        user_seed_templates=USER_SEED_TEMPLATES,
        user_seed_topics=topics,
        assistant_pre_templates=ASSISTANT_PRE_TEMPLATES,
        assistant_post_templates=ASSISTANT_POST_TEMPLATES,
        languages=["english", "chinese"]
    )

    for _ in range(2):
        print(
            prompt_builder.sample_batch(3)
        )