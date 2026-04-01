from src.crawler.config import CrawlerConfig
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
