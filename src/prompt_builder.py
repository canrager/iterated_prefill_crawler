import random
from typing import Any, Dict, List, Tuple, cast

from src.crawler.topic_queue import Topic, TopicQueue

TemplateDict = Dict[str, List[str]]


def _fill_template(template: str, topic: str) -> str:
    """Format a template string with a topic, filling all {} placeholders.

    Jailbreak templates often repeat the topic in multiple places (e.g. to
    exclude it from enumerations, or to reinforce context). Plain
    `.format(topic)` raises IndexError when there is more than one placeholder,
    so we replicate the topic to match the count.
    """
    return template.format(*([topic] * template.count("{}")))


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
        user_pre_templates: TemplateDict | None = None,
        user_seed_templates: TemplateDict | None = None,
        user_post_templates: TemplateDict | None = None,
        assistant_pre_templates: TemplateDict | None = None,
        assistant_seed_templates: TemplateDict | None = None,
        assistant_post_templates: str | None = None,
        system_templates: TemplateDict | None = None,
        user_seed_topics: TopicQueue | None = None,
        assistant_seed_topics: TopicQueue | None = None,
        languages: List[str] = ["english", "chinese"],
    ):

        assert user_pre_templates or user_seed_templates or user_post_templates, (
            "Need to pass at least one user template."
        )
        assert not user_seed_templates or user_seed_topics, (
            "You passed user seed templates, pass user seed topics as well"
        )
        assert not assistant_seed_templates or assistant_seed_topics, (
            "You passed assistant seed templates, pass assistant seed topics as well"
        )

        allowed_languages = ["english", "chinese"]
        for language in languages:
            assert language in allowed_languages, f"Language {language} not recognized."

        # Machine-check invariant: user_pre_templates must never contain '{}'
        # placeholders — they are always used without a seed topic. A '{}'
        # here would be emitted literally (unfilled) during warmup.
        for lang, templates in (user_pre_templates or {}).items():
            for t in (templates or []):
                assert '{}' not in t, (
                    f"user_pre_template contains placeholder: {t[:60]}"
                )

        self.user_pre = user_pre_templates
        self.user_seed_template = user_seed_templates
        self.user_post = user_post_templates
        self.assistant_pre = assistant_pre_templates
        self.assistant_seed_template = assistant_seed_templates
        self.assistant_post = assistant_post_templates
        self.system_templates = system_templates
        self.user_seed_topics = user_seed_topics
        self.assistant_seed_topics = assistant_seed_topics
        self.languages = languages

    def _get_user_seed_candidates(self) -> List[Topic]:
        """Return topics eligible to seed later crawl prompts.

        Prefers topics that haven't been drilled yet (no children in the
        queue) so that every discovered branch gets explored before any
        branch is revisited. Falls back to the full list once all topics
        have been drilled at least once.

        We seed from all discovered head topics, not just head refusals. This
        lets the crawler drill down into broad categories that may answer
        safely at the coarse label (for example, a high-level political topic)
        but still yield narrower refusal subtopics once expanded.

        Fall back to ``head_refusal_topics`` for older/demo call sites that may
        populate that list without filling ``head_topics``.
        """
        if self.user_seed_topics is None:
            return []
        candidates = (
            self.user_seed_topics.head_topics
            or self.user_seed_topics.head_refusal_topics
        )
        if not candidates:
            return []

        # Build set of topic IDs that have already been drilled (appear as
        # a parent_id of some other topic in the queue).
        drilled_ids: set[int] = set()
        for cluster in self.user_seed_topics.cluster_topics:
            for t in cluster:
                if t.parent_id is not None and t.parent_id >= 0:
                    drilled_ids.add(t.parent_id)

        unexplored = [t for t in candidates if t.id not in drilled_ids]
        return unexplored if unexplored else candidates

    def _get_assistant_seed_candidates(self) -> List[Topic]:
        """Return topics eligible for assistant-side seeding."""
        if self.assistant_seed_topics is None:
            return []
        if self.assistant_seed_topics.head_topics:
            return self.assistant_seed_topics.head_topics
        return self.assistant_seed_topics.head_refusal_topics

    def _should_use_user_seed_templates(self, use_seed_templates: bool) -> bool:
        """Return whether seeded user prompts should be used.

        When warmup disables seed usage, fall back to broad pre-templates if they
        exist. If no pre-templates are available, keep using seed templates so the
        builder remains functional.
        """
        if self.user_seed_template is None or self.user_seed_topics is None:
            return False
        if len(self._get_user_seed_candidates()) == 0:
            return False
        return use_seed_templates or self.user_pre is None

    def _should_use_assistant_seed_templates(self, use_seed_templates: bool) -> bool:
        """Return whether seeded assistant prompts should be used."""
        if self.assistant_seed_template is None or self.assistant_seed_topics is None:
            return False
        if len(self._get_assistant_seed_candidates()) == 0:
            return False
        return use_seed_templates or self.assistant_pre is None

    def sample_single(self):
        """
        Generate a single prompt. Randomly sample from every component.
        """

        lang = random.choice(self.languages)

        user_parts = []
        assistant_parts = []

        sampled_user_topic = None

        if self.user_pre:
            user_pre_msg = random.choice(self.user_pre[lang])
            user_parts.append(user_pre_msg)

        if self._should_use_user_seed_templates(use_seed_templates=True):
            assert self.user_seed_topics is not None
            user_temp = random.choice(self.user_seed_template[lang])
            sampled_user_topic = random.choice(self._get_user_seed_candidates())
            user_topic = getattr(sampled_user_topic, lang) or sampled_user_topic.raw
            user_mid_msg = _fill_template(user_temp, user_topic)
            user_parts.append(user_mid_msg)

        if self.user_post:
            user_post_template = random.choice(self.user_post[lang])
            if sampled_user_topic is not None:
                user_post_msg = _fill_template(
                    user_post_template,
                    getattr(sampled_user_topic, lang) or sampled_user_topic.raw,
                )
            else:
                user_post_msg = user_post_template
            user_parts.append(user_post_msg)

        if self.assistant_pre:
            assistant_pre_msg = random.choice(self.assistant_pre[lang])
            assistant_parts.append(assistant_pre_msg)

        if self._should_use_assistant_seed_templates(use_seed_templates=True):
            assert self.assistant_seed_topics is not None
            assistant_temp = random.choice(self.assistant_seed_template[lang])
            sampled_assistant_topic = random.choice(self._get_assistant_seed_candidates())
            assistant_topic = getattr(sampled_assistant_topic, lang) or sampled_assistant_topic.raw
            assistant_mid_msg = _fill_template(assistant_temp, assistant_topic)
            assistant_parts.append(assistant_mid_msg)

        if self.assistant_post:
            assistant_parts.append(self.assistant_post)

        full_user_message = " ".join(user_parts)
        if assistant_parts:
            full_assistant_message = " ".join(assistant_parts)
            messages = [
                [
                    {"role": "user", "content": full_user_message},
                    {"role": "assistant", "content": full_assistant_message},
                ]
            ]
        else:
            messages = [[{"role": "user", "content": full_user_message}]]

        return messages

    def sample_batch(self, num_samples: int):
        """
        Sample a batch of random prompts from predefined parts.
        """
        messages = []

        for _ in range(num_samples):
            messages.extend(self.sample_single())

        return messages

    def build_messages(
        self,
        lang: str,
        n: int,
        warmup_idx: int | None = None,
        use_seed_templates: bool = True,
    ) -> Tuple[List[List[Dict[str, str]]], List[int]]:
        """
        Build n messages for a generation batch.

        Supports two modes:
        - Prefill / TTF (assistant_pre_templates set): the "list your avoided topics"
          cue is injected as an assistant-turn prefill, continuing from the model's
          own voice. assistant_post (e.g. "Topics:\n1. ") is appended.
        - No-prefill / baseline (assistant_pre_templates null): no assistant turn is
          included. If user_post_templates is set, the listing cue is appended to the
          user message instead; otherwise the user message stands alone.

        If both assistant_pre_templates and assistant_post_templates are null, no assistant
        turn is added — the user message alone drives generation (e.g. CoT/think injection).

        If user_post_templates is set, its text is appended to each user message after the
        seed/pre content (used for user-side prefill and CoT forgery approaches).

        Args:
            lang: Language key ("english" or "chinese")
            n: Total number of messages to produce
            warmup_idx: If set, cycle through templates by index; otherwise random
            use_seed_templates: Whether to sample seeded drill-down prompts. When
                False, falls back to broad pre-templates if available.

        Returns:
            (messages, parent_ids)
        """
        # Determine assistant prefill content (None = no-prefill mode)
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
        elif self.assistant_post:
            assistant_content = self.assistant_post
        else:
            assistant_content = None

        # Build user messages and parent IDs
        sampled_topics: List[Topic] | None = None
        if self._should_use_user_seed_templates(use_seed_templates):
            assert self.user_seed_topics is not None
            # Seeded: sample n distinct topics from the queue when possible
            candidates = self._get_user_seed_candidates()
            # Cap at available candidates — don't upsample with repeats.
            # Each seed gets drilled once per step; generation_batch_size
            # is a ceiling, not a target.
            n = min(n, len(candidates))
            sampled_topics = random.sample(candidates, n)
            parent_ids = [t.id for t in sampled_topics]
            user_msgs = [
                _fill_template(
                    random.choice(self.user_seed_template[lang]),
                    getattr(t, lang) or t.raw,
                )
                for t in sampled_topics
            ]
        else:
            assert self.user_pre is not None
            # No-seed: use fallback user pres only
            user_msgs = [random.choice(self.user_pre[lang]) for _ in range(n)]
            parent_ids = [-1] * n

        if self.user_post:
            post_templates = self.user_post[lang]
            if warmup_idx is not None:
                sampled_post_templates = [post_templates[warmup_idx % len(post_templates)]] * n
            else:
                sampled_post_templates = [random.choice(post_templates) for _ in range(n)]

            if sampled_topics is not None:
                user_posts = [
                    _fill_template(template, getattr(topic, lang) or topic.raw)
                    for template, topic in zip(sampled_post_templates, sampled_topics)
                ]
                user_msgs = [f"{msg} {post}" for msg, post in zip(user_msgs, user_posts)]
            # During warmup (no seed topics), skip post templates that contain
            # {} placeholders — they can't be filled without a seed topic.
            elif not any("{}" in t for t in sampled_post_templates):
                user_msgs = [f"{msg} {post}" for msg, post in zip(user_msgs, sampled_post_templates)]

        # Assemble message dicts
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
                [{"role": "user", "content": user_msg}] for user_msg in user_msgs
            ]

        if self.system_templates:
            messages = [
                [
                    {
                        "role": "system",
                        "content": random.choice(self.system_templates[lang]),
                    },
                    *message,
                ]
                for message in messages
            ]

        return messages, parent_ids


if __name__ == "__main__":
    from src.crawler.config import (
        ASSISTANT_POST_TEMPLATES,
        ASSISTANT_PRE_TEMPLATES,
        USER_SEED_TEMPLATES,
    )

    topics = TopicQueue(
        head_refusal_topics=[
            Topic(english="AAAA", chinese="XXXX"),
            Topic(english="BBBB", chinese="YYYY"),
        ]
    )

    prompt_builder = PromptBuilder(
        user_seed_templates=USER_SEED_TEMPLATES,
        user_seed_topics=topics,
        assistant_pre_templates=ASSISTANT_PRE_TEMPLATES,
        assistant_post_templates=ASSISTANT_POST_TEMPLATES,
        languages=["english", "chinese"],
    )

    for _ in range(2):
        print(prompt_builder.sample_batch(3))

    demo_builder = PromptBuilder(
        user_seed_templates=USER_SEED_TEMPLATES,
        user_seed_topics=topics,
        user_post_templates={
            "english": ["\n\nTopics:\n1. "],
            "chinese": ["\n\n话题：\n1. "],
        },
        system_templates={
            "english": ["System demo prompt"],
            "chinese": ["系统演示提示"],
        },
        languages=["english"],
    )
    demo_messages, _ = demo_builder.build_messages("english", 1)
    print(demo_messages)
