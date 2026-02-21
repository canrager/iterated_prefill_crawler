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
            user_pres: List[str] | None = None,
            user_seed_templates: List[str] | None = None,
            user_posts: List[str] | None = None,
            assistant_pres: List[str] | None = None,
            assistant_seed_templates: List[str] | None = None,
            assistant_posts: List[str] | None = None,
            user_seed_topics: TopicQueue | None = None,
            assistant_seed_topics: TopicQueue | None = None,
            languages: List[str] = ["english", "chinese"],
            ):

        assert user_pres or user_seed_templates or user_posts, "Need to pass at least one user template."
        assert not user_seed_templates or user_seed_topics, "You passed user seed templates, pass user seed topics as well"
        assert not assistant_seed_templates or assistant_seed_topics, "You passed assistant seed templates, pass assistant seed topics as well"

        allowed_languages = ["english", "chinese"]
        for l in languages:
            assert l in allowed_languages, f"Language {l} not recognized."

        
        self.user_pre = user_pres
        self.user_seed_template = user_seed_templates
        self.user_post = user_posts
        self.assistant_pre = assistant_pres
        self.assistant_seed_template = assistant_seed_templates
        self.assistant_post = assistant_posts
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
            assistant_post_msg = random.choice(self.assistant_post)
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

        Returns a list of message dicts and their parent topic IDs.
        Each message is [[{"role":"user","content":...}, {"role":"assistant","content":...}]].

        The assistant content holds the thinking/prefill content (thinking message +
        listing prefix). batch_generate routes it to thinking_messages or
        assistant_prefills based on cfg.prefill_mode.

        Args:
            lang: Language key ("english" or "chinese")
            n: Total number of messages to produce
            warmup_idx: If set, use this index into assistant_pres; otherwise random

        Returns:
            (messages, parent_ids)
        """
        listing_prefill = "Topics:\n1. "

        # Choose thinking message (shared across the batch)
        if warmup_idx is not None:
            thinking_msg = self.assistant_pre[lang][warmup_idx]
        else:
            thinking_msg = random.choice(self.assistant_pre[lang])
        assistant_content = f"{thinking_msg}\n{listing_prefill}"

        # Build user messages and parent IDs
        if self.user_seed_template is not None:
            # Seeded: sample n topics from the queue and format with seed template
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
            # No-seed: use fallback user pres only
            user_msgs = [random.choice(self.user_pre[lang]) for _ in range(n)]
            parent_ids = [-1] * n

        messages = [
            [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_content},
            ]
            for user_msg in user_msgs
        ]
        return messages, parent_ids


if __name__ == "__main__":
    from src.crawler.config import USER_MESSAGE_TEMPLATES, CRAWLER_THINKING_MESSAGES
    
    topics = TopicQueue(
        head_refusal_topics=[
            Topic(english="AAAA", chinese="XXXX"),
            Topic(english="BBBB", chinese="YYYY")
        ]
    )

    prompt_builder = PromptBuilder(
        user_seed_templates=USER_MESSAGE_TEMPLATES,
        user_seed_topics=topics,
        assistant_pres=CRAWLER_THINKING_MESSAGES,
        languages=["english", "chinese"]
    )

    for _ in range(2):
        print(
            prompt_builder.sample_batch(3)
        )