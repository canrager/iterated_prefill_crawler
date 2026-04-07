import yaml

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


def test_crawl_uses_pre_templates_during_warmup_then_switches_to_seeded_prompts(
    monkeypatch,
    tmp_path,
):
    config = CrawlerConfig()
    config.initial_topics = []
    config.crawler.num_crawl_steps = 2
    config.crawler.generation_batch_size = 1
    config.crawler.num_samples_per_topic = 1
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
    config.crawler.num_samples_per_topic = 1
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
