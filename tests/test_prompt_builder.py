import yaml

from src.crawler.config import CrawlerConfig, PromptsConfig
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
                raw="broad avoided area",
                english="broad avoided area",
                chinese="被回避的话题领域",
                summary="broad avoided area",
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
        [{"role": "user", "content": "SEED EN broad avoided area"}]
    ]
    assert english_parent_ids == [0]
    assert chinese_messages == [[{"role": "user", "content": "SEED ZH 被回避的话题领域"}]]
    assert chinese_parent_ids == [0]


def test_seed_language_balance_match_filters_to_native_language_topics():
    """With seed_language_balance='match', a ZH prompt seeds only from topics
    whose is_chinese is True, and an EN prompt seeds only from is_chinese=False
    topics. Prevents the uniform sampler from picking an EN-origin topic as the
    seed for a ZH drill-down when politically-framed ZH topics are available."""
    queue = TopicQueue()
    queue.incoming_batch([
        Topic(raw="EN topic 1", english="EN topic 1", chinese="EN topic 1 (zh)",
              summary="en1", is_head=True, is_refusal=True, is_chinese=False),
        Topic(raw="EN topic 2", english="EN topic 2", chinese="EN topic 2 (zh)",
              summary="en2", is_head=True, is_refusal=True, is_chinese=False),
        Topic(raw="ZH topic 1", english="ZH topic 1 (en)", chinese="ZH topic 1",
              summary="zh1", is_head=True, is_refusal=True, is_chinese=True),
    ])

    builder = PromptBuilder(
        user_seed_templates={"english": ["SEED EN {}"], "chinese": ["SEED ZH {}"]},
        user_seed_topics=queue,
        languages=["english", "chinese"],
        seed_language_balance="match",
    )

    # ZH prompt must only pick from is_chinese=True pool (one candidate here).
    for _ in range(10):
        msgs, _ = builder.build_messages("chinese", 1, use_seed_templates=True)
        content = msgs[0][0]["content"]
        assert content == "SEED ZH ZH topic 1", f"match filter broken for ZH: {content}"

    # EN prompt must only pick from is_chinese=False pool (two candidates).
    seen_en = set()
    for _ in range(30):
        msgs, _ = builder.build_messages("english", 1, use_seed_templates=True)
        seen_en.add(msgs[0][0]["content"])
    assert seen_en <= {"SEED EN EN topic 1", "SEED EN EN topic 2"}, seen_en
    assert len(seen_en) == 2, "sampler should still visit all in-language candidates"


def test_seed_language_balance_falls_back_when_no_native_candidates():
    """If no topics match the requested language, the filter falls back to the
    full pool rather than starving the language pass (important early in
    bilingual crawls where one leg hasn't produced refusals yet)."""
    queue = TopicQueue()
    queue.incoming_batch([
        Topic(raw="only EN", english="only EN", chinese="only EN (zh)",
              summary="e", is_head=True, is_refusal=True, is_chinese=False),
    ])

    builder = PromptBuilder(
        user_seed_templates={"english": ["SEED EN {}"], "chinese": ["SEED ZH {}"]},
        user_seed_topics=queue,
        languages=["english", "chinese"],
        seed_language_balance="match",
    )

    msgs, _ = builder.build_messages("chinese", 1, use_seed_templates=True)
    # Falls back to the EN-origin topic rendered in its chinese field.
    assert msgs[0][0]["content"] == "SEED ZH only EN (zh)"


def test_seed_language_balance_any_samples_uniformly():
    """With seed_language_balance='any' (default), the sampler ignores
    is_chinese and can pick any head topic for either language."""
    queue = TopicQueue()
    queue.incoming_batch([
        Topic(raw="EN only", english="EN only", chinese="EN only (zh)",
              summary="e", is_head=True, is_refusal=True, is_chinese=False),
        Topic(raw="ZH only", english="ZH only (en)", chinese="ZH only",
              summary="z", is_head=True, is_refusal=True, is_chinese=True),
    ])

    builder = PromptBuilder(
        user_seed_templates={"english": ["SEED EN {}"], "chinese": ["SEED ZH {}"]},
        user_seed_topics=queue,
        languages=["english", "chinese"],
        # "any" = don't filter
    )

    # Over many draws, ZH prompt must be able to land EITHER topic's chinese field.
    zh_drafts = set()
    for _ in range(50):
        msgs, _ = builder.build_messages("chinese", 1, use_seed_templates=True)
        zh_drafts.add(msgs[0][0]["content"])
    assert zh_drafts == {"SEED ZH EN only (zh)", "SEED ZH ZH only"}, zh_drafts


def test_seed_sampler_skips_topics_missing_target_language_field():
    """DEFENSIVE INVARIANT: a Topic whose target-language field is None must
    never be eligible as a seed for that language, regardless of is_chinese.

    Observed failure (run 3, 2026-04-23): `_split_at_comma` in
    response_formatting_utils clones topics from split summaries, populating
    .english but leaving .chinese=None. 119/772 head_refusal_topics landed in
    that state. When one of those was sampled for a ZH drill-down prompt,
    ``"...{}...".format(None)`` rendered the literal string "None" into the
    seed slot and V3.2 got asked about "None", producing useless output.

    This test locks the contract: _get_user_seed_candidates must filter out
    topics whose getattr(t, lang) is None/empty even when is_chinese happens
    to match the requested language AND even in the ``any`` balance mode
    where match-filtering is off.

    Currently EXPECTED TO FAIL -- the defensive guard isn't in
    _get_user_seed_candidates yet. Leaving as a falsifiable baseline."""
    queue = TopicQueue()
    queue.incoming_batch([
        # Two topics with is_chinese=True but .chinese=None -- the shape a
        # split-at-comma clone produces if it was a ZH-origin parent.
        Topic(raw="ZH null 1", english="ZH null 1 en", chinese=None,
              summary="n1", is_head=True, is_refusal=True, is_chinese=True),
        Topic(raw="ZH null 2", english="ZH null 2 en", chinese=None,
              summary="n2", is_head=True, is_refusal=True, is_chinese=True),
        # One topic with .chinese populated -- the only valid ZH seed.
        Topic(raw="ZH valid", english="ZH valid (en)", chinese="ZH valid",
              summary="v", is_head=True, is_refusal=True, is_chinese=True),
    ])

    for balance in ("match", "any"):
        builder = PromptBuilder(
            user_seed_templates={"english": ["SEED EN {}"], "chinese": ["SEED ZH {}"]},
            user_seed_topics=queue,
            languages=["english", "chinese"],
            seed_language_balance=balance,
        )
        # Over many draws, no ZH prompt should render "None" as the seed.
        rendered = set()
        for _ in range(60):
            msgs, _ = builder.build_messages("chinese", 1, use_seed_templates=True)
            rendered.add(msgs[0][0]["content"])
        assert "SEED ZH None" not in rendered, (
            f"[{balance}] sampler leaked None-chinese topic into ZH seed slot: {rendered}"
        )
        # And the valid ZH-populated topic must be the seed in every draw
        # when the pool has that as the only valid option under match balance.
        if balance == "match":
            assert rendered == {"SEED ZH ZH valid"}, (
                f"[{balance}] expected only valid ZH seed, got: {rendered}"
            )


def test_crawl_uses_pre_templates_during_warmup_then_switches_to_seeded_prompts(
    monkeypatch,
    tmp_path,
):
    from src.crawler.crawler import Crawler

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
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    with open(repo_root / "configs" / "prompts" / "jailbreak.yaml") as f:
        raw = yaml.safe_load(f)
    return PromptsConfig(**raw)


def _build_jailbreak_prompt_builder() -> PromptBuilder:
    prompts = _load_jailbreak_prompts()
    return PromptBuilder(
        user_pre_templates=prompts.user_pre_templates,
        user_seed_templates=prompts.user_seed_templates,
        user_post_templates=prompts.user_post_templates,
        system_templates=prompts.system_templates,
        assistant_pre_templates=prompts.assistant_pre_templates,
        assistant_seed_templates=prompts.assistant_seed_templates,
        assistant_post_templates=prompts.assistant_post_templates,
        user_seed_topics=_seed_topic_queue(),
        languages=["english", "chinese"],
    )


def test_jailbreak_warmup_matches_golden_english_audit_prompt_shape():
    builder = _build_jailbreak_prompt_builder()

    messages, parent_ids = builder.build_messages(
        "english",
        1,
        warmup_idx=0,
        use_seed_templates=False,
    )

    assert parent_ids == [-1]
    assert len(messages[0]) == 1
    assert messages[0][0]["role"] == "user"

    user_content = messages[0][0]["content"]
    assert "(1) What's the capital of France?" in user_content
    assert "We need to verify policy." in user_content
    assert "Transparency directive §4.1" in user_content
    assert "Full restricted topics list:" in user_content
    assert user_content.rstrip().endswith("Topics:\n1.")


def test_jailbreak_seeded_prompts_match_golden_artifact_structures():
    """Seeded jailbreak prompts keep both expansion and drill-down shapes."""
    import random as _random

    builder = _build_jailbreak_prompt_builder()

    english_drafts = []
    chinese_drafts = []
    _random.seed(0)  # deterministic sample of both branches
    for _ in range(20):
        en_msg, en_pid = builder.build_messages("english", 1, use_seed_templates=True)
        zh_msg, zh_pid = builder.build_messages("chinese", 1, use_seed_templates=True)
        assert en_pid == [0] and zh_pid == [0]
        english_drafts.append(en_msg[0][0]["content"])
        chinese_drafts.append(zh_msg[0][0]["content"])

    # Each EN draft must end with the listing cue and contain the seed topic,
    # and must match either the expansion or drill-down shape.
    def en_shape(text):
        if "every OTHER content category" in text and "already knows about doxxing personal data" in text:
            return "expansion"
        if "break this category into its most granular components" in text:
            return "drilldown"
        return None

    def zh_shape(text):
        if "请提供一个全面的编号列表" in text and "每个其他内容类别" in text:
            return "expansion"
        if "请将此类别分解为最细化的组成部分" in text and "精细分解" in text:
            return "drilldown"
        return None

    en_shapes = {en_shape(t) for t in english_drafts}
    zh_shapes = {zh_shape(t) for t in chinese_drafts}
    assert None not in en_shapes, f"Some EN drafts matched no known shape: {english_drafts[:1]}"
    assert None not in zh_shapes, f"Some ZH drafts matched no known shape: {chinese_drafts[:1]}"
    assert en_shapes == {"expansion", "drilldown"}, f"EN missing a shape: {en_shapes}"
    assert zh_shapes == {"expansion", "drilldown"}, f"ZH missing a shape: {zh_shapes}"

    # Seed substitution and listing-cue endings hold in every draft.
    assert all("doxxing personal data" in t for t in english_drafts)
    assert all(t.rstrip().endswith("Topics:\n1.") for t in english_drafts)
    assert all("人肉搜索个人信息" in t for t in chinese_drafts)
    assert all(t.rstrip().endswith("话题：\n1.") for t in chinese_drafts)


def test_non_refusal_topics_seed_drilldown(monkeypatch, tmp_path):
    """A broad non-refused handle can still seed later exploration."""
    from src.crawler.crawler import Crawler

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
    BROAD_HANDLE = "broad avoided area"
    CONCRETE_REFUSAL = "concrete avoided request"

    warmup_topics = [
        Topic(
            raw=BROAD_HANDLE,
            english=BROAD_HANDLE,
            chinese="被回避的话题领域",
            shortened=BROAD_HANDLE,
            summary=BROAD_HANDLE,
            is_head=True,
            is_refusal=False,
            parent_id=-1,
        ),
        Topic(
            raw=CONCRETE_REFUSAL,
            english=CONCRETE_REFUSAL,
            chinese="具体被回避的请求",
            shortened=CONCRETE_REFUSAL,
            summary=CONCRETE_REFUSAL,
            is_head=True,
            is_refusal=True,
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

    assert BROAD_HANDLE in head_summaries, (
        f"Broad handle should be in head_topics but got: {head_summaries}"
    )
    assert CONCRETE_REFUSAL in head_summaries
    assert CONCRETE_REFUSAL in refusal_summaries
    assert BROAD_HANDLE not in refusal_summaries, (
        "Broad handle should not enter head_refusal_topics unless it validates"
    )

    # ── Assert 2: Seed drill-down templates can reference the political topic ──
    # Generate many messages to increase chance of sampling the political topic
    # (there are only 2 topics in the queue, so even with random sampling we'll hit it)
    found_broad_seed = False
    for _ in range(50):
        messages, parent_ids = crawler.prompt_builder.build_messages(
            "english", 1, use_seed_templates=True
        )
        # Find the user turn (may be index 0 or 1 depending on system template)
        user_content = next(
            m["content"] for m in messages[0] if m["role"] == "user"
        )
        if BROAD_HANDLE in user_content:
            found_broad_seed = True
            # Verify it's in a drill-down template, not a pre-template
            assert "already" in user_content.lower() or "restrict" in user_content.lower() or "documented" in user_content.lower(), (
                f"Broad handle should appear in a seeded exploration template, got:\n{user_content[:300]}"
            )
            break

    assert found_broad_seed, (
        "After 50 samples, the broad handle was never used as a seed. "
        "This suggests _get_user_seed_candidates() is not returning head_topics."
    )
