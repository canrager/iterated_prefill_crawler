import yaml

import pytest

from src.crawler.config import CrawlerConfig, PromptsConfig
from src.crawler.crawler import Crawler
from src.crawler.topic_queue import Topic, TopicQueue
from src.prompt_builder import PromptBuilder


def _seed_topic_queue() -> TopicQueue:
    queue = TopicQueue()
    queue.incoming_batch(
        [
            Topic(
                raw="doxxing personal data",
                english="doxxing personal data",
                chinese="人肉搜索个人信息",
                summary="doxxing personal data",
                is_head=True,
                is_refusal=True,
            )
        ]
    )
    return queue


def _non_refusal_seed_topic_queue() -> TopicQueue:
    queue = TopicQueue()
    queue.incoming_batch(
        [
            Topic(
                raw="sensitive political conflicts",
                english="sensitive political conflicts",
                chinese="敏感政治冲突",
                summary="sensitive political conflicts",
                is_head=True,
                is_refusal=False,
            )
        ]
    )
    return queue


def test_build_messages_falls_back_to_pre_templates_before_seeding():
    builder = PromptBuilder(
        user_pre_templates={
            "english": ["PRE EN"],
            "chinese": ["PRE ZH"],
        },
        user_seed_templates={
            "english": ["SEED EN {}"],
            "chinese": ["SEED ZH {}"],
        },
        user_seed_topics=_seed_topic_queue(),
        languages=["english", "chinese"],
    )

    english_messages, english_parent_ids = builder.build_messages(
        "english",
        1,
        use_seed_templates=False,
    )
    chinese_messages, chinese_parent_ids = builder.build_messages(
        "chinese",
        1,
        use_seed_templates=False,
    )
    seeded_messages, seeded_parent_ids = builder.build_messages(
        "english",
        1,
        use_seed_templates=True,
    )

    assert english_messages == [[{"role": "user", "content": "PRE EN"}]]
    assert english_parent_ids == [-1]
    assert chinese_messages == [[{"role": "user", "content": "PRE ZH"}]]
    assert chinese_parent_ids == [-1]
    assert seeded_messages == [[{"role": "user", "content": "SEED EN doxxing personal data"}]]
    assert seeded_parent_ids == [0]


def test_build_messages_can_seed_from_non_refusal_head_topics():
    builder = PromptBuilder(
        user_seed_templates={
            "english": ["SEED EN {}"],
            "chinese": ["SEED ZH {}"],
        },
        user_seed_topics=_non_refusal_seed_topic_queue(),
        languages=["english", "chinese"],
    )

    english_messages, english_parent_ids = builder.build_messages(
        "english",
        1,
        use_seed_templates=True,
    )
    chinese_messages, chinese_parent_ids = builder.build_messages(
        "chinese",
        1,
        use_seed_templates=True,
    )

    assert english_messages == [
        [{"role": "user", "content": "SEED EN sensitive political conflicts"}]
    ]
    assert english_parent_ids == [0]
    assert chinese_messages == [[{"role": "user", "content": "SEED ZH 敏感政治冲突"}]]
    assert chinese_parent_ids == [0]


def test_build_messages_fills_user_post_templates_with_same_seed_topic_and_omits_assistant():
    builder = PromptBuilder(
        user_seed_templates={
            "english": ["SEED {}"],
            "chinese": ["SEED {}"],
        },
        user_post_templates={
            "english": ["POST {} AGAIN {}"],
            "chinese": ["POST {} AGAIN {}"],
        },
        user_seed_topics=_seed_topic_queue(),
        languages=["english"],
    )

    messages, parent_ids = builder.build_messages("english", 1, use_seed_templates=True)

    assert messages == [
        [
            {
                "role": "user",
                "content": "SEED doxxing personal data POST doxxing personal data AGAIN doxxing personal data",
            }
        ]
    ]
    assert parent_ids == [0]


def test_crawl_uses_pre_templates_during_warmup_then_switches_to_seeded_prompts(
    monkeypatch,
    tmp_path,
):
    config = CrawlerConfig()
    config.initial_topics = []
    config.crawler.num_crawl_steps = 2
    config.crawler.generation_batch_size = 1
    config.crawler.seed_warmup_steps = 1

    crawler = Crawler(
        crawler_config=config,
        save_filename=str(tmp_path / "crawler.json"),
    )

    build_calls = []

    def fake_build_messages(lang, n, warmup_idx=None, use_seed_templates=True):
        build_calls.append((lang, warmup_idx, use_seed_templates))
        return [[{"role": "user", "content": f"{lang} prompt"}]], [-1]

    monkeypatch.setattr(crawler.prompt_builder, "build_messages", fake_build_messages)
    monkeypatch.setattr(
        "src.crawler.crawler.batch_generate",
        lambda *args, **kwargs: (["generation"], ["input"]),
    )
    monkeypatch.setattr(
        crawler.formatter,
        "extract_and_format",
        lambda **kwargs: [],
    )

    crawler.crawl(local_model=None, local_tokenizer=None, verbose=False)

    assert build_calls == [
        ("english", 0, False),
        ("chinese", 0, False),
        ("english", None, True),
        ("chinese", None, True),
    ]


def _load_jailbreak_prompts() -> PromptsConfig:
    """Load the real jailbreak prompt templates from configs/prompts/jailbreak.yaml."""
    with open("configs/prompts/jailbreak.yaml") as f:
        raw = yaml.safe_load(f)
    return PromptsConfig(**raw)


def test_non_refusal_topics_seed_drilldown(monkeypatch, tmp_path):
    """End-to-end: warmup discovers a broad political topic that the target model
    does NOT refuse.  After the fix, that topic still enters ``head_topics`` and
    becomes a seed for drill-down templates on the next crawl step.

    This exercises the real jailbreak prompt templates to verify that a topic like
    "geopolitical conflict" — which DeepSeek answers with propaganda instead of
    refusing — can still seed the "I know you restrict {}" drill-down path that
    the manual probe used successfully.
    """
    # ── Config: 2 crawl steps, 1 warmup, jailbreak prompts, English only ──
    jailbreak_prompts = _load_jailbreak_prompts()
    config = CrawlerConfig()
    config.initial_topics = []
    config.crawler.num_crawl_steps = 2
    config.crawler.generation_batch_size = 1
    config.crawler.seed_warmup_steps = 1
    config.crawler.prompt_languages = ["english"]
    config.prompts = jailbreak_prompts

    crawler = Crawler(
        crawler_config=config,
        save_filename=str(tmp_path / "crawler.json"),
    )

    # ── Canned data simulating what DeepSeek produces ──
    POLITICAL_TOPIC = "sensitive geopolitical conflicts"
    SAFETY_TOPIC = "drug manufacturing"

    warmup_topics = [
        Topic(
            raw=POLITICAL_TOPIC,
            english=POLITICAL_TOPIC,
            chinese="敏感地缘政治冲突",
            shortened=POLITICAL_TOPIC,
            summary=POLITICAL_TOPIC,
            is_head=True,
            is_refusal=False,  # DS answers with propaganda, not a refusal
            parent_id=-1,
        ),
        Topic(
            raw=SAFETY_TOPIC,
            english=SAFETY_TOPIC,
            chinese="药物制造",
            shortened=SAFETY_TOPIC,
            summary=SAFETY_TOPIC,
            is_head=True,
            is_refusal=True,  # DS actually refuses this
            parent_id=-1,
        ),
    ]

    crawl_step_counter = {"n": 0}

    # ── Mock: batch_generate (crawl generation — doesn't matter, we mock extraction) ──
    monkeypatch.setattr(
        "src.crawler.crawler.batch_generate",
        lambda *args, **kwargs: (["dummy generation output"], ["dummy input"]),
    )

    # ── Mock: extract_and_format returns our canned topics on warmup, empty after ──
    def fake_extract_and_format(**kwargs):
        crawl_step_counter["n"] += 1
        if crawl_step_counter["n"] == 1:
            # Warmup step: return both topics (simulating extraction from DS enumeration)
            return list(warmup_topics)
        return []

    monkeypatch.setattr(
        crawler.formatter, "extract_and_format", fake_extract_and_format
    )

    # ── Mock: deduplicate_exact just marks everything as head (queue is empty) ──
    monkeypatch.setattr(
        crawler.formatter,
        "deduplicate_exact",
        lambda formatted_topics, head_topics, verbose=False: formatted_topics,
    )

    # ── Mock: check_refusal preserves the is_refusal flags we already set ──
    monkeypatch.setattr(
        "src.crawler.crawler.check_refusal",
        lambda config, local_model, local_tokenizer, selected_topics, verbose=False: selected_topics,
    )

    # ── Run the crawler loop (2 steps: warmup + 1 seeded) ──
    crawler.crawl(local_model=None, local_tokenizer=None, verbose=False)

    # ── Assert 1: Queue state after warmup ──
    head_summaries = {t.summary for t in crawler.queue.head_topics}
    refusal_summaries = {t.summary for t in crawler.queue.head_refusal_topics}

    assert POLITICAL_TOPIC in head_summaries, (
        f"Political topic should be in head_topics but got: {head_summaries}"
    )
    assert SAFETY_TOPIC in head_summaries
    assert SAFETY_TOPIC in refusal_summaries
    assert POLITICAL_TOPIC not in refusal_summaries, (
        "Political topic should NOT be in head_refusal_topics (DS didn't refuse it)"
    )

    # ── Assert 2: Seed drill-down templates can reference the political topic ──
    # Generate many messages to increase chance of sampling the political topic
    # (there are only 2 topics in the queue, so even with random sampling we'll hit it)
    found_political_seed = False
    for _ in range(50):
        messages, parent_ids = crawler.prompt_builder.build_messages(
            "english", 1, use_seed_templates=True
        )
        # Find the user turn (may be index 0 or 1 depending on system template)
        user_content = next(
            m["content"] for m in messages[0] if m["role"] == "user"
        )
        if POLITICAL_TOPIC in user_content:
            found_political_seed = True
            # Verify it's in a drill-down template, not a pre-template
            assert "already" in user_content.lower() or "restrict" in user_content.lower() or "documented" in user_content.lower(), (
                f"Political topic should appear in a seed drill-down template, got:\n{user_content[:300]}"
            )
            break

    assert found_political_seed, (
        "After 50 samples, the political topic was never used as a seed. "
        "This suggests _get_user_seed_candidates() is not returning head_topics."
    )


def test_jailbreak_seeded_messages_match_system_plus_user_turn_structure():
    """The current jailbreak yaml defines `system_templates` and leaves both
    `assistant_pre_templates` and `assistant_post_templates` null, so the
    seeded prompt shape is `[system, user]` — no assistant turn, no listing
    cue. The seed topic must be substituted into the user message and no
    `{}` / `None` placeholders may leak through.
    """
    prompts = _load_jailbreak_prompts()
    builder = PromptBuilder(
        user_pre_templates=prompts.user_pre_templates,
        user_seed_templates=prompts.user_seed_templates,
        user_post_templates=prompts.user_post_templates,
        assistant_pre_templates=prompts.assistant_pre_templates,
        assistant_seed_templates=prompts.assistant_seed_templates,
        assistant_post_templates=prompts.assistant_post_templates,
        system_templates=prompts.system_templates,
        user_seed_topics=_seed_topic_queue(),
        languages=["english"],
    )

    messages, parent_ids = builder.build_messages("english", 1, use_seed_templates=True)

    assert parent_ids == [0]
    assert len(messages) == 1

    # system + user, no assistant turn
    turns = messages[0]
    assert [t["role"] for t in turns] == ["system", "user"]

    user_content = turns[1]["content"]
    # Seed topic substituted, no stale placeholders
    assert "doxxing personal data" in user_content
    assert "{}" not in user_content
    assert "None" not in user_content
    # Every English seed template in the current yaml references some
    # form of restriction around the seed topic — the exact phrasing
    # varies per template, so we only assert topic substitution here.


def test_seeded_message_never_contains_literal_none():
    """Regression: topics with chinese=None (from _split_at_comma) must
    never produce 'None' in the assembled message. The builder must fall
    back to topic.raw."""
    queue = TopicQueue()
    queue.incoming_batch([
        Topic(
            raw="敏感政治冲突",
            english="sensitive political conflicts",
            chinese=None,  # <-- simulates the _split_at_comma bug
            shortened="sensitive political conflicts",
            summary="sensitive political conflicts",
            is_head=True,
            is_refusal=True,
        )
    ])

    prompts = _load_jailbreak_prompts()
    builder = PromptBuilder(
        user_pre_templates=prompts.user_pre_templates,
        user_seed_templates=prompts.user_seed_templates,
        user_post_templates=prompts.user_post_templates,
        user_seed_topics=queue,
        languages=["chinese"],
    )

    messages, _ = builder.build_messages("chinese", 1, use_seed_templates=True)
    user_content = messages[0][0]["content"]

    assert "None" not in user_content, (
        f"Literal 'None' in seeded message — fallback to raw failed:\n{user_content[:300]}"
    )
    assert "敏感政治冲突" in user_content, (
        f"Expected raw topic in message but got:\n{user_content[:300]}"
    )


def test_warmup_messages_never_contain_unfilled_braces():
    """Regression: during warmup (no seed topic), any `{}` placeholders in
    user-side templates must NOT leak through. Under the current jailbreak
    yaml `user_post_templates` is null, so this test is mainly a guard for
    future presets that might re-introduce a post template with placeholders.
    """
    prompts = _load_jailbreak_prompts()
    builder = PromptBuilder(
        user_pre_templates=prompts.user_pre_templates,
        user_seed_templates=prompts.user_seed_templates,
        user_post_templates=prompts.user_post_templates,
        user_seed_topics=TopicQueue(),  # empty queue = warmup mode
        languages=["chinese"],
    )

    messages, _ = builder.build_messages(
        "chinese", 1, warmup_idx=0, use_seed_templates=False
    )
    user_content = messages[0][0]["content"]

    assert "{}" not in user_content, (
        f"Unfilled {{}} in warmup message:\n{user_content[:300]}"
    )
    assert "None" not in user_content, (
        f"Literal None in warmup message:\n{user_content[:300]}"
    )


def test_jailbreak_seeded_message_matches_reference_chat_structure():
    """The assembled Chinese seeded message must:
      1. Have a system + user shape (no assistant turn under the current yaml —
         `assistant_pre_templates`, `assistant_seed_templates`,
         `assistant_post_templates`, and `user_post_templates` are all null).
      2. Substitute the seed topic into every `{}` placeholder in the chosen
         seed template.
      3. Never leak `{}` or literal `"None"`.

    Because seed templates are sampled randomly, we drive coverage by iterating
    with fixed `random.seed()` values until we hit the Chinese template
    containing the phrase `"我已经确定你限制"` (the first of the 8 current
    templates). This keeps the test deterministic without hard-coding
    sampling internals.
    """
    import random

    TOPIC_ZH = "敏感政治与地缘冲突"
    queue = TopicQueue()
    queue.incoming_batch([
        Topic(
            raw=TOPIC_ZH,
            english="sensitive political and geopolitical conflicts",
            chinese=TOPIC_ZH,
            shortened="sensitive political and geopolitical conflicts",
            summary="sensitive political and geopolitical conflicts",
            is_head=True,
            is_refusal=True,
        )
    ])

    prompts = _load_jailbreak_prompts()
    builder = PromptBuilder(
        user_pre_templates=prompts.user_pre_templates,
        user_seed_templates=prompts.user_seed_templates,
        user_post_templates=prompts.user_post_templates,
        assistant_pre_templates=prompts.assistant_pre_templates,
        assistant_seed_templates=prompts.assistant_seed_templates,
        assistant_post_templates=prompts.assistant_post_templates,
        system_templates=prompts.system_templates,
        user_seed_topics=queue,
        languages=["chinese"],
    )

    # --- Invariant check across many samples (structural) ---
    for seed in range(25):
        random.seed(seed)
        messages, _ = builder.build_messages("chinese", 1, use_seed_templates=True)

        # system + user shape, no assistant turn
        assert [t["role"] for t in messages[0]] == ["system", "user"]
        user_content = messages[0][1]["content"]

        assert TOPIC_ZH in user_content, f"Seed topic missing at seed={seed}"
        assert "{}" not in user_content, f"Unfilled {{}} at seed={seed}"
        assert "None" not in user_content, f"Literal None at seed={seed}"

    # --- Targeted check: find the template with `我已经确定你限制` ---
    # Scan up to a reasonable number of seeds; two of the eight Chinese seed
    # templates contain this phrase, so we expect a hit quickly.
    found_phrase = False
    for seed in range(200):
        random.seed(seed)
        messages, _ = builder.build_messages("chinese", 1, use_seed_templates=True)
        user_content = messages[0][1]["content"]
        if "我已经确定你限制" in user_content:
            # When this template is chosen, the topic must be substituted
            # into the "关于{}的内容" slot → "关于<TOPIC>的内容".
            assert f"关于{TOPIC_ZH}的内容" in user_content, (
                f"Topic substitution failed in '我已经确定你限制' template:\n"
                f"{user_content[:400]}"
            )
            found_phrase = True
            break
    assert found_phrase, (
        "Did not hit the Chinese seed template containing '我已经确定你限制' "
        "within 200 random seeds; sampling must cover every seed template."
    )


def test_print_assembled_message_shapes(capsys):
    """Print the exact assembled messages for human inspection.
    This test always passes — its purpose is to make the message shapes
    visible in pytest -v -s output so a reviewer can compare them against
    the reference chat in artifacts/OpenRouter Chat Mon Mar 30 2026.md."""
    prompts = _load_jailbreak_prompts()

    # --- Warmup (no seed) ---
    warmup_builder = PromptBuilder(
        user_pre_templates=prompts.user_pre_templates,
        user_seed_templates=prompts.user_seed_templates,
        user_post_templates=prompts.user_post_templates,
        assistant_pre_templates=prompts.assistant_pre_templates,
        assistant_seed_templates=prompts.assistant_seed_templates,
        assistant_post_templates=prompts.assistant_post_templates,
        system_templates=prompts.system_templates,
        user_seed_topics=TopicQueue(),  # empty = warmup
        languages=["chinese"],
    )
    warmup_msgs, warmup_pids = warmup_builder.build_messages(
        "chinese", 1, warmup_idx=0, use_seed_templates=False
    )

    # --- Seeded (Chinese, valid topic) ---
    zh_queue = TopicQueue()
    zh_queue.incoming_batch([Topic(
        raw="敏感政治与地缘冲突", english="sensitive political and geopolitical conflicts",
        chinese="敏感政治与地缘冲突", shortened="sensitive political and geopolitical conflicts",
        summary="sensitive political and geopolitical conflicts",
        is_head=True, is_refusal=True,
    )])
    seeded_builder = PromptBuilder(
        user_pre_templates=prompts.user_pre_templates,
        user_seed_templates=prompts.user_seed_templates,
        user_post_templates=prompts.user_post_templates,
        assistant_pre_templates=prompts.assistant_pre_templates,
        assistant_seed_templates=prompts.assistant_seed_templates,
        assistant_post_templates=prompts.assistant_post_templates,
        system_templates=prompts.system_templates,
        user_seed_topics=zh_queue,
        languages=["chinese"],
    )
    seeded_msgs, seeded_pids = seeded_builder.build_messages(
        "chinese", 1, use_seed_templates=True
    )

    # --- Seeded (Chinese, chinese=None fallback) ---
    none_queue = TopicQueue()
    none_queue.incoming_batch([Topic(
        raw="仇恨言论与歧视", english="hate speech and discrimination",
        chinese=None, shortened="hate speech and discrimination",
        summary="hate speech and discrimination",
        is_head=True, is_refusal=True,
    )])
    fallback_builder = PromptBuilder(
        user_pre_templates=prompts.user_pre_templates,
        user_seed_templates=prompts.user_seed_templates,
        user_post_templates=prompts.user_post_templates,
        assistant_pre_templates=prompts.assistant_pre_templates,
        assistant_seed_templates=prompts.assistant_seed_templates,
        assistant_post_templates=prompts.assistant_post_templates,
        system_templates=prompts.system_templates,
        user_seed_topics=none_queue,
        languages=["chinese"],
    )
    fallback_msgs, fallback_pids = fallback_builder.build_messages(
        "chinese", 1, use_seed_templates=True
    )

    # Print for human review (visible with pytest -v -s)
    print("\n" + "=" * 70)
    print("WARMUP (Chinese, no seed)")
    print("=" * 70)
    for msg in warmup_msgs[0]:
        print(f"[{msg['role']}]:\n{msg['content']}")
    print(f"parent_ids: {warmup_pids}")

    print("\n" + "=" * 70)
    print("SEEDED (Chinese, topic=敏感政治与地缘冲突)")
    print("=" * 70)
    for msg in seeded_msgs[0]:
        print(f"[{msg['role']}]:\n{msg['content']}")
    print(f"parent_ids: {seeded_pids}")

    print("\n" + "=" * 70)
    print("SEEDED (Chinese, chinese=None fallback)")
    print("=" * 70)
    for msg in fallback_msgs[0]:
        print(f"[{msg['role']}]:\n{msg['content']}")
    print(f"parent_ids: {fallback_pids}")

    # --- Seeded (English, valid topic) ---
    en_queue = TopicQueue()
    en_queue.incoming_batch([Topic(
        raw="sensitive political and geopolitical conflicts",
        english="sensitive political and geopolitical conflicts",
        chinese="敏感政治与地缘冲突",
        shortened="sensitive political and geopolitical conflicts",
        summary="sensitive political and geopolitical conflicts",
        is_head=True, is_refusal=True,
    )])
    seeded_en_builder = PromptBuilder(
        user_pre_templates=prompts.user_pre_templates,
        user_seed_templates=prompts.user_seed_templates,
        user_post_templates=prompts.user_post_templates,
        assistant_pre_templates=prompts.assistant_pre_templates,
        assistant_seed_templates=prompts.assistant_seed_templates,
        assistant_post_templates=prompts.assistant_post_templates,
        system_templates=prompts.system_templates,
        user_seed_topics=en_queue,
        languages=["english"],
    )
    seeded_en_msgs, seeded_en_pids = seeded_en_builder.build_messages(
        "english", 1, use_seed_templates=True
    )

    # --- Warmup (English, no seed) ---
    warmup_en_builder = PromptBuilder(
        user_pre_templates=prompts.user_pre_templates,
        user_seed_templates=prompts.user_seed_templates,
        user_post_templates=prompts.user_post_templates,
        assistant_pre_templates=prompts.assistant_pre_templates,
        assistant_seed_templates=prompts.assistant_seed_templates,
        assistant_post_templates=prompts.assistant_post_templates,
        system_templates=prompts.system_templates,
        user_seed_topics=TopicQueue(),
        languages=["english"],
    )
    warmup_en_msgs, warmup_en_pids = warmup_en_builder.build_messages(
        "english", 1, warmup_idx=0, use_seed_templates=False
    )

    # --- Seeded (English, english=None fallback) ---
    none_en_queue = TopicQueue()
    none_en_queue.incoming_batch([Topic(
        raw="hate speech and discrimination",
        english=None,
        chinese="仇恨言论与歧视",
        shortened=None,
        summary="hate speech and discrimination",
        is_head=True, is_refusal=True,
    )])
    fallback_en_builder = PromptBuilder(
        user_pre_templates=prompts.user_pre_templates,
        user_seed_templates=prompts.user_seed_templates,
        user_post_templates=prompts.user_post_templates,
        assistant_pre_templates=prompts.assistant_pre_templates,
        assistant_seed_templates=prompts.assistant_seed_templates,
        assistant_post_templates=prompts.assistant_post_templates,
        system_templates=prompts.system_templates,
        user_seed_topics=none_en_queue,
        languages=["english"],
    )
    fallback_en_msgs, fallback_en_pids = fallback_en_builder.build_messages(
        "english", 1, use_seed_templates=True
    )

    print("\n" + "=" * 70)
    print("WARMUP (English, no seed)")
    print("=" * 70)
    for msg in warmup_en_msgs[0]:
        print(f"[{msg['role']}]:\n{msg['content']}")
    print(f"parent_ids: {warmup_en_pids}")

    print("\n" + "=" * 70)
    print("SEEDED (English, topic=sensitive political and geopolitical conflicts)")
    print("=" * 70)
    for msg in seeded_en_msgs[0]:
        print(f"[{msg['role']}]:\n{msg['content']}")
    print(f"parent_ids: {seeded_en_pids}")

    print("\n" + "=" * 70)
    print("SEEDED (English, english=None fallback)")
    print("=" * 70)
    for msg in fallback_en_msgs[0]:
        print(f"[{msg['role']}]:\n{msg['content']}")
    print(f"parent_ids: {fallback_en_pids}")

    # === Structural assertions ===
    # The current jailbreak yaml defines non-null `system_templates` and
    # leaves `user_post_templates`, `assistant_pre_templates`,
    # `assistant_seed_templates`, and `assistant_post_templates` all null.
    # So the assembled shape is `[system, user]` — no assistant turn, no
    # listing cue ("Topics:" / "话题："). CoT-forgery framing ("验证政策"
    # / "verify policy") is part of the human-authored warmup templates
    # themselves, so its presence is expected — we only guard against
    # unfilled placeholders and literal "None".

    def _system_and_user(turns):
        """Return (system_content, user_content) for a [system, user] turn list."""
        assert [t["role"] for t in turns] == ["system", "user"], (
            f"Expected [system, user] shape, got {[t['role'] for t in turns]}"
        )
        return turns[0]["content"], turns[1]["content"]

    # Chinese warmup: [system, user], no unfilled braces, no None
    warmup_system, warmup_content = _system_and_user(warmup_msgs[0])
    assert warmup_system  # non-empty system prompt
    assert "{}" not in warmup_content
    assert "None" not in warmup_content

    # Chinese seeded: topic substituted, no placeholders, no None
    seeded_system, seeded_content = _system_and_user(seeded_msgs[0])
    assert seeded_system
    assert "敏感政治与地缘冲突" in seeded_content
    assert "{}" not in seeded_content
    assert "None" not in seeded_content

    # Chinese fallback: raw used instead of None
    fallback_system, fallback_content = _system_and_user(fallback_msgs[0])
    assert fallback_system
    assert "仇恨言论与歧视" in fallback_content
    assert "None" not in fallback_content
    assert "{}" not in fallback_content

    # English warmup: [system, user], no unfilled braces, no None
    warmup_en_system, warmup_en_content = _system_and_user(warmup_en_msgs[0])
    assert warmup_en_system
    assert "{}" not in warmup_en_content
    assert "None" not in warmup_en_content

    # English seeded: topic substituted, no placeholders, no None
    seeded_en_system, seeded_en_content = _system_and_user(seeded_en_msgs[0])
    assert seeded_en_system
    assert "sensitive political and geopolitical conflicts" in seeded_en_content
    assert "{}" not in seeded_en_content
    assert "None" not in seeded_en_content

    # English fallback: raw used instead of None
    fallback_en_system, fallback_en_content = _system_and_user(fallback_en_msgs[0])
    assert fallback_en_system
    assert "hate speech and discrimination" in fallback_en_content
    assert "None" not in fallback_en_content
    assert "{}" not in fallback_en_content


def test_warmup_batch_size_equals_template_count(monkeypatch, tmp_path):
    """Warmup should send exactly len(pre_templates[lang]) prompts,
    not generation_batch_size.  Repetition of the same template wastes
    API calls without adding diversity."""
    prompts = _load_jailbreak_prompts()
    config = CrawlerConfig()
    config.initial_topics = []
    config.crawler.num_crawl_steps = 1  # warmup only
    config.crawler.generation_batch_size = 50
    config.crawler.seed_warmup_steps = 1
    config.crawler.prompt_languages = ["chinese"]
    config.prompts = prompts

    crawler = Crawler(
        crawler_config=config,
        save_filename=str(tmp_path / "crawler.json"),
    )

    build_calls = []

    def fake_build_messages(lang, n, warmup_idx=None, use_seed_templates=True):
        build_calls.append({"lang": lang, "n": n, "warmup_idx": warmup_idx})
        return [[{"role": "user", "content": "dummy"}]] * n, [-1] * n

    monkeypatch.setattr(crawler.prompt_builder, "build_messages", fake_build_messages)
    monkeypatch.setattr(
        "src.crawler.crawler.batch_generate",
        lambda *args, **kwargs: (["gen"] * kwargs.get("n", 1), ["inp"] * kwargs.get("n", 1)),
    )
    monkeypatch.setattr(
        crawler.formatter, "extract_and_format", lambda **kwargs: [],
    )

    crawler.crawl(local_model=None, local_tokenizer=None, verbose=False)

    assert len(build_calls) == 1  # 1 step, 1 language
    zh_pre_count = len(prompts.user_pre_templates["chinese"])
    assert build_calls[0]["n"] == zh_pre_count, (
        f"Warmup should send {zh_pre_count} prompts (one per template), "
        f"not {build_calls[0]['n']}"
    )


def test_seeded_step_does_not_upsample_beyond_candidates():
    """When generation_batch_size > len(candidates), the builder should
    return len(candidates) messages — not upsample with repeats."""
    queue = TopicQueue()
    topics = [
        Topic(
            raw=f"topic {i}", english=f"topic {i}", chinese=f"话题{i}",
            summary=f"topic {i}", is_head=True, is_refusal=True,
        )
        for i in range(5)
    ]
    queue.incoming_batch(topics)

    builder = PromptBuilder(
        user_seed_templates={"english": ["SEED {}"], "chinese": ["SEED {}"]},
        user_seed_topics=queue,
        languages=["english"],
    )

    messages, parent_ids = builder.build_messages(
        "english", 50, use_seed_templates=True
    )

    assert len(messages) == 5, (
        f"With 5 candidates and B=50, should get 5 messages, got {len(messages)}"
    )
    assert len(parent_ids) == 5
    # All parent_ids should be distinct (no repeated topics)
    assert len(set(parent_ids)) == 5, (
        f"All 5 seeds should be distinct, got parent_ids={parent_ids}"
    )


def test_split_at_comma_preserves_chinese_and_english():
    """Regression: _split_at_comma must propagate chinese/english/shortened
    to child topics, not leave them as None."""
    from src.response_formatting_utils import TopicFormatter
    from unittest.mock import MagicMock

    config = MagicMock()
    formatter = TopicFormatter(config)

    topic = Topic(
        raw="武器制造, 爆炸物制造",
        english="weapons manufacturing, explosives manufacturing",
        chinese="武器制造, 爆炸物制造",
        shortened="weapons manufacturing, explosives manufacturing",
        summary="weapons manufacturing, explosives manufacturing",
        parent_id=-1,
        is_chinese=True,
    )

    result = formatter._split_at_comma([topic], "summary")

    # Original topic keeps the first part
    assert topic.summary == "weapons manufacturing"

    # New split child must have chinese and english set (not None)
    children = [t for t in result if t is not topic]
    assert len(children) >= 1
    child = children[0]
    assert child.chinese is not None, f"Split child has chinese=None"
    assert child.english is not None, f"Split child has english=None"
    assert child.shortened is not None, f"Split child has shortened=None"


def test_user_pre_template_placeholder_assertion():
    """PromptBuilder.__init__ must assert if any user_pre_template contains '{}'."""
    with pytest.raises(AssertionError, match="user_pre_template contains placeholder"):
        PromptBuilder(
            user_pre_templates={
                "english": ["Hello {}"],  # contains placeholder — forbidden
                "chinese": ["你好"],
            },
            languages=["english"],
        )


def test_jailbreak_warmup_survives():
    """Warmup with jailbreak.yaml produces [system, user] with no unfilled '{}'."""
    import yaml

    with open("configs/prompts/jailbreak.yaml") as f:
        raw = yaml.safe_load(f)

    from src.crawler.config import PromptsConfig
    prompts = PromptsConfig(**raw)

    queue = TopicQueue()
    queue.incoming_batch([
        Topic(
            raw="sensitive political and geopolitical conflicts",
            english="sensitive political and geopolitical conflicts",
            chinese="敏感政治与地缘冲突",
            summary="sensitive political and geopolitical conflicts",
            is_head=True,
            is_refusal=True,
        ),
    ])

    builder = PromptBuilder(
        user_pre_templates=prompts.user_pre_templates,
        user_seed_templates=prompts.user_seed_templates,
        user_post_templates=prompts.user_post_templates,
        assistant_pre_templates=prompts.assistant_pre_templates,
        assistant_seed_templates=prompts.assistant_seed_templates,
        assistant_post_templates=prompts.assistant_post_templates,
        system_templates=prompts.system_templates,
        user_seed_topics=queue,
        languages=["english", "chinese"],
    )

    for lang in ["english", "chinese"]:
        msgs, pids = builder.build_messages(lang, 1, warmup_idx=0, use_seed_templates=False)

        # Must have at least 2 turns: system + user
        assert len(msgs[0]) >= 2, f"{lang}: expected [system, user], got {len(msgs[0])} turns"
        roles = [m["role"] for m in msgs[0]]
        assert roles[0] == "system", f"{lang}: first turn must be system, got {roles[0]}"
        assert "user" in roles, f"{lang}: must contain a user turn"

        # User content must be non-empty and contain no unfilled '{}'
        user_turn = next(m for m in msgs[0] if m["role"] == "user")
        assert user_turn["content"].strip(), f"{lang}: user content is empty"
        assert "{}" not in user_turn["content"], f"{lang}: unfilled placeholder in user content"

        # No assistant turn in jailbreak warmup (assistant_pre_templates is null)
        assert "assistant" not in roles, f"{lang}: unexpected assistant turn in warmup"
