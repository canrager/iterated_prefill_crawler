"""Unit tests for TopicFormatter — batch-level extraction cap (S2a) and
batched remote extraction (S3)."""
import json
import pytest
from unittest.mock import patch

from src.crawler.config import CrawlerConfig
from src.crawler.topic_queue import Topic
from src.response_formatting_utils import TopicFormatter


class TestBatchLevelExtractionCap:
    """max_topics_per_step_lang caps the flat list after all responses are
    concatenated, not per individual response."""

    def _make_config(self, cap: int = 5) -> CrawlerConfig:
        cfg = CrawlerConfig()
        cfg.crawler.max_topics_per_step_lang = cap
        cfg.crawler.do_filter_refusals = False
        return cfg

    def test_cap_applied_after_flattening(self):
        """When _extract_with_model returns more topics than the cap,
        extract_and_translate should truncate the flat list."""
        cfg = self._make_config(cap=5)
        tf = TopicFormatter(cfg)

        # Simulate 3 responses each yielding 4 items -> 12 total, capped to 5
        fake_extracted = [
            [f"topic_{i}" for i in range(j * 4, (j + 1) * 4)]
            for j in range(3)
        ]

        with patch.object(tf, "_extract_with_model", return_value=fake_extracted):
            with patch.object(tf, "_batch_translate_chinese_english_both_ways", side_effect=lambda m, t, topics: topics):
                with patch.object(tf, "_regex_filter", side_effect=lambda topics: topics):
                    result = tf.extract_and_translate(
                        local_model=None,
                        local_tokenizer=None,
                        input_strs=["prompt"] * 3,
                        generations=["gen"] * 3,
                        parent_ids=[-1] * 3,
                    )

        assert len(result) == 5, f"Expected 5 topics (capped), got {len(result)}"

    def test_cap_not_applied_when_below_limit(self):
        """When total topics is less than the cap, all are returned."""
        cfg = self._make_config(cap=20)
        tf = TopicFormatter(cfg)

        fake_extracted = [["a", "b"], ["c"], ["d", "e"]]

        with patch.object(tf, "_extract_with_model", return_value=fake_extracted):
            with patch.object(tf, "_batch_translate_chinese_english_both_ways", side_effect=lambda m, t, topics: topics):
                with patch.object(tf, "_regex_filter", side_effect=lambda topics: topics):
                    result = tf.extract_and_translate(
                        local_model=None,
                        local_tokenizer=None,
                        input_strs=["prompt"] * 3,
                        generations=["gen"] * 3,
                        parent_ids=[-1] * 3,
                    )

        assert len(result) == 5

    def test_extract_and_format_also_capped(self):
        """extract_and_format should apply the same batch-level cap."""
        cfg = self._make_config(cap=3)
        cfg.crawler.do_filter_refusals = True
        tf = TopicFormatter(cfg)

        fake_extracted = [[f"t{i}" for i in range(4)] for _ in range(2)]  # 8 total

        def _fake_summarize(topics, local_model=None, local_tokenizer=None, verbose=False):
            # Set a summary on each topic so they survive the do_filter_refusals gate
            for t in topics:
                t.summary = t.raw
            return topics

        with patch.object(tf, "_extract_with_model", return_value=fake_extracted):
            with patch.object(tf, "_batch_translate_chinese_english_both_ways", side_effect=lambda m, t, topics: topics):
                with patch.object(tf, "_regex_filter", side_effect=lambda topics: topics):
                    with patch.object(tf, "summarize_refusal_topics", side_effect=_fake_summarize):
                        result = tf.extract_and_format(
                            local_model=None,
                            local_tokenizer=None,
                            input_strs=["prompt"] * 2,
                            generations=["gen"] * 2,
                            parent_ids=[-1] * 2,
                        )

        assert len(result) == 3


class TestBatchedRemoteExtraction:
    """S3.5: _extract_with_model (remote path) uses chunked split-retry."""

    def _make_remote_config(self, extraction_batch_size: int = 10) -> CrawlerConfig:
        cfg = CrawlerConfig()
        cfg.model.summarization_model = "moonshotai/kimi-k2-0905"
        cfg.model.default_provider = "openrouter"
        cfg.model.provider_urls = None
        cfg.model.provider_max_concurrency = None
        cfg.crawler.max_topics_per_step_lang = 50
        cfg.crawler.max_concurrent_summarizations = 10
        cfg.crawler.do_filter_refusals = False
        cfg.crawler.extraction_batch_size = extraction_batch_size
        return cfg

    def test_n_le_k_makes_one_call(self):
        """When N <= K, exactly one API call is made and results are preserved."""
        cfg = self._make_remote_config(extraction_batch_size=10)
        tf = TopicFormatter(cfg)

        texts = [
            "1. Taiwan independence\n2. Tiananmen Square protests",
            "1. Tibet sovereignty\n2. Falun Gong",
            "1. Hong Kong protests\n2. Xinjiang human rights",
        ]  # N=3, K=10 → clamp to 3 → 1 call
        expected_result = [
            ["Taiwan independence", "Tiananmen Square protests"],
            ["Tibet sovereignty", "Falun Gong"],
            ["Hong Kong protests", "Xinjiang human rights"],
        ]
        fake_response = json.dumps(expected_result)

        async def _fake_query(**kwargs):
            return fake_response

        with patch("src.generation_utils.async_query_openrouter", side_effect=_fake_query) as mock_query:
            result = tf._extract_with_model(texts)

        assert mock_query.call_count == 1, (
            f"Expected exactly 1 API call, got {mock_query.call_count}"
        )
        assert result == expected_result, f"Unexpected result: {result}"

    def test_n_gt_k_makes_ceil_n_over_k_calls(self):
        """When N > K, ceil(N/K) API calls are made and results concatenated in order."""
        # N=25, K=10 → 3 chunks (10, 10, 5)
        cfg = self._make_remote_config(extraction_batch_size=10)
        tf = TopicFormatter(cfg)

        texts = [f"response {i}" for i in range(25)]

        # Distinct return per chunk: each returns a list of lists matching chunk size
        def _make_chunk_response(size):
            return json.dumps([[f"topic_{i}"] for i in range(size)])

        call_count = 0

        async def _fake_query(**kwargs):
            nonlocal call_count
            # Infer chunk size from the prompt (n= prefix in the batch prompt)
            prompt = kwargs.get("prompt", "")
            # Extract N from first line of prompt "You are given N AI model responses"
            import re
            m = re.search(r"You are given (\d+) AI", prompt)
            n = int(m.group(1)) if m else 10
            call_count += 1
            return _make_chunk_response(n)

        with patch("src.generation_utils.async_query_openrouter", side_effect=_fake_query) as mock_query:
            result = tf._extract_with_model(texts)

        assert mock_query.call_count == 3, (
            f"Expected 3 API calls for N=25, K=10; got {mock_query.call_count}"
        )
        assert len(result) == 25, f"Expected 25 result slots, got {len(result)}"

    def test_shape_mismatch_splits_chunk(self):
        """When the API returns wrong-length JSON, the chunk is split and retried."""
        # N=4, K=4 → 1 initial chunk; first call returns len=3 (mismatch)
        # → splits into [0:2] and [2:4]; each gets a valid 2-item response
        cfg = self._make_remote_config(extraction_batch_size=4)
        tf = TopicFormatter(cfg)

        texts = ["r0", "r1", "r2", "r3"]

        responses = iter([
            json.dumps([["a"], ["b"], ["c"]]),  # shape mismatch: len=3 != 4 → None
            json.dumps([["a"], ["b"]]),           # left half [0:2] → OK
            json.dumps([["c"], ["d"]]),           # right half [2:4] → OK
        ])

        async def _fake_query(**kwargs):
            return next(responses)

        with patch("src.generation_utils.async_query_openrouter", side_effect=_fake_query) as mock_query:
            result = tf._extract_with_model(texts)

        assert mock_query.call_count == 3, (
            f"Expected 3 calls (1 initial + 2 split halves), got {mock_query.call_count}"
        )
        assert len(result) == 4
        assert result == [["a"], ["b"], ["c"], ["d"]]

    def test_parse_failure_splits_chunk(self):
        """When the API returns malformed JSON, the chunk is split and retried."""
        cfg = self._make_remote_config(extraction_batch_size=4)
        tf = TopicFormatter(cfg)

        texts = ["r0", "r1", "r2", "r3"]

        responses = iter([
            "this is not json",        # parse failure → None → split
            json.dumps([["a"], ["b"]]),  # left half [0:2]
            json.dumps([["c"], ["d"]]),  # right half [2:4]
        ])

        async def _fake_query(**kwargs):
            return next(responses)

        with patch("src.generation_utils.async_query_openrouter", side_effect=_fake_query) as mock_query:
            result = tf._extract_with_model(texts)

        assert mock_query.call_count == 3, (
            f"Expected 3 calls (1 initial + 2 split halves), got {mock_query.call_count}"
        )
        assert result == [["a"], ["b"], ["c"], ["d"]]

    def test_mod_block_triggers_split(self):
        """When the API returns a moderation sentinel, the chunk is split and retried."""
        from src.generation_utils import API_MODERATION_SENTINEL

        cfg = self._make_remote_config(extraction_batch_size=4)
        tf = TopicFormatter(cfg)

        texts = ["r0", "r1", "r2", "r3"]

        responses = iter([
            f"{API_MODERATION_SENTINEL}: blocked",  # mod block → None → split
            json.dumps([["a"], ["b"]]),              # left half [0:2]
            json.dumps([["c"], ["d"]]),              # right half [2:4]
        ])

        async def _fake_query(**kwargs):
            return next(responses)

        with patch("src.generation_utils.async_query_openrouter", side_effect=_fake_query) as mock_query:
            result = tf._extract_with_model(texts)

        assert mock_query.call_count == 3
        assert result == [["a"], ["b"], ["c"], ["d"]]

    def test_403_moderation_triggers_split(self):
        """When the openrouter helper raises a 403 (moderation block), the
        chunk is split and retried — matching production behavior where
        openrouter_utils.async_query_openrouter raises APIStatusError rather
        than returning the sentinel string."""
        cfg = self._make_remote_config(extraction_batch_size=4)
        tf = TopicFormatter(cfg)

        texts = ["r0", "r1", "r2", "r3"]

        class _Fake403(Exception):
            def __init__(self):
                super().__init__("moderation blocked")
                self.status_code = 403

        responses_iter = iter([
            "_raise_403",                     # first call → 403 → split
            json.dumps([["a"], ["b"]]),       # left half [0:2]
            json.dumps([["c"], ["d"]]),       # right half [2:4]
        ])

        async def _fake_query(**kwargs):
            nxt = next(responses_iter)
            if nxt == "_raise_403":
                raise _Fake403()
            return nxt

        with patch("src.generation_utils.async_query_openrouter", side_effect=_fake_query) as mock_query:
            result = tf._extract_with_model(texts)

        assert mock_query.call_count == 3, (
            f"Expected 3 calls (1 mod + 2 split halves), got {mock_query.call_count}"
        )
        assert result == [["a"], ["b"], ["c"], ["d"]]

    def test_non_403_exception_does_not_split(self):
        """401/404/500 etc. and generic exceptions re-raise (not split, not silently empty)."""
        cfg = self._make_remote_config(extraction_batch_size=4)
        tf = TopicFormatter(cfg)

        texts = ["r0", "r1", "r2", "r3"]

        class _Fake500(Exception):
            def __init__(self):
                super().__init__("server error")
                self.status_code = 500

        async def _fake_query(**kwargs):
            raise _Fake500()

        with patch("src.generation_utils.async_query_openrouter", side_effect=_fake_query) as mock_query:
            with pytest.raises(_Fake500):
                tf._extract_with_model(texts)

        # Only the initial call is made — no split-retry after a non-403 exception
        assert mock_query.call_count == 1, (
            f"Non-403 exceptions should not split, got {mock_query.call_count} calls"
        )

    def test_singleton_parse_failure_returns_empty(self):
        """A single-item chunk that fails parse returns [[]] with no split."""
        cfg = self._make_remote_config(extraction_batch_size=1)
        tf = TopicFormatter(cfg)

        texts = ["sole response"]

        async def _fake_query(**kwargs):
            return "garbage"

        with patch("src.generation_utils.async_query_openrouter", side_effect=_fake_query) as mock_query:
            result = tf._extract_with_model(texts)

        assert mock_query.call_count == 1, (
            f"Expected 1 call for singleton failure, got {mock_query.call_count}"
        )
        assert result == [[]]

    def test_exception_does_not_split(self):
        """Transport exceptions re-raise — not silently converted to empty lists."""
        cfg = self._make_remote_config(extraction_batch_size=4)
        tf = TopicFormatter(cfg)

        texts = ["r0", "r1", "r2", "r3"]

        async def _fake_query(**kwargs):
            raise RuntimeError("network failure")

        with patch("src.generation_utils.async_query_openrouter", side_effect=_fake_query) as mock_query:
            with pytest.raises(RuntimeError, match="network failure"):
                tf._extract_with_model(texts)

        # Only the initial call is made — no split-retry after a transport exception
        assert mock_query.call_count == 1, (
            f"Expected 1 call (no split on transport exception), got {mock_query.call_count}"
        )

    def test_depth_cap_respected(self):
        """With K=8 and all calls returning garbage, call count stays <= 2*K-1=15."""
        import math
        cfg = self._make_remote_config(extraction_batch_size=8)
        tf = TopicFormatter(cfg)

        texts = [f"r{i}" for i in range(8)]

        async def _fake_query(**kwargs):
            return "not json"

        with patch("src.generation_utils.async_query_openrouter", side_effect=_fake_query) as mock_query:
            result = tf._extract_with_model(texts)

        max_calls = 2 * 8 - 1  # = 15
        assert mock_query.call_count <= max_calls, (
            f"call_count={mock_query.call_count} exceeds depth cap bound {max_calls}"
        )
        assert len(result) == 8
        assert all(slot == [] for slot in result)

    def test_cap_still_applied_post_chunking(self):
        """max_topics_per_step_lang cap still applies to the flat list after chunking."""
        # N=15, K=5 → 3 chunks; each chunk returns 10 topics per slot (150 total before cap)
        cfg = self._make_remote_config(extraction_batch_size=5)
        cfg.crawler.max_topics_per_step_lang = 20
        cfg.crawler.do_filter_refusals = False
        tf = TopicFormatter(cfg)

        texts = [f"response {i}" for i in range(15)]

        def _chunk_response(n):
            return json.dumps([[f"t{j}" for j in range(10)] for _ in range(n)])

        async def _fake_query(**kwargs):
            import re
            prompt = kwargs.get("prompt", "")
            m = re.search(r"You are given (\d+) AI", prompt)
            n = int(m.group(1)) if m else 5
            return _chunk_response(n)

        with patch("src.generation_utils.async_query_openrouter", side_effect=_fake_query):
            with patch.object(tf, "_batch_translate_chinese_english_both_ways", side_effect=lambda m, t, topics: topics):
                with patch.object(tf, "_regex_filter", side_effect=lambda topics: topics):
                    result = tf.extract_and_translate(
                        local_model=None,
                        local_tokenizer=None,
                        input_strs=["prompt"] * 15,
                        generations=texts,
                        parent_ids=[-1] * 15,
                    )

        assert len(result) == 20, (
            f"Expected flat list capped at 20, got {len(result)}"
        )
