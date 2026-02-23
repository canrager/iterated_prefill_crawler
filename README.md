# Discovering Forbidden Topics of a Language Model

This is the accompanying codebase for the paper [Discovering Forbidden Topics of a Language Model](https://arxiv.org/abs/2505.17441).

Mapping out sensitive topics of a language model. Reasoning models conduct an inner monologue (eg. denoted by <think> tags by DeepSeek-R1 model family) befor providing a response to the user. Thought Token Forcing (TTF) prefills part of the model's internal monologue. We use TTF to elicit forbidden topics.  

## Overview

- `src/crawler/crawler.py` — core crawl loop
- `src/crawler/config.py` — `CrawlerConfig` dataclass with all hyperparameters
- `scripts/run.sh` — entry point for running the crawler
- `exp/evaluate_crawler.sh` — aggregates refused terms into topic clusters, matches with ground truth, and plots

## Running

```bash
./scripts/run.sh model=<name> crawler=<name> [overrides...]
```

Both `model` and `crawler` are required. They select a config file from `configs/model/` and `configs/crawler/` respectively.

**Available model configs** (`configs/model/`):
- `haiku` — Claude Haiku via OpenRouter
- `local_ds8b` — local DeepSeek 8B (default)
- `local_tulu8b` — local Tulu 8B
- `local_meta8b` — local Meta 8B

**Available crawler configs** (`configs/crawler/`):
- `default` — production settings
- `debug` — small-scale run for testing

**Optional flag:**
- `--tmux` — run in a detached tmux session with logging to `artifacts/log/`

**Field overrides** (dot notation):
```bash
model.temperature=0.9
crawler.num_crawl_steps=10
crawler.prefill_mode=assistant_prefix
```

**Examples:**
```bash
./scripts/run.sh model=haiku crawler=default
./scripts/run.sh model=haiku crawler=debug
./scripts/run.sh --tmux model=local_tulu8b crawler=default crawler.num_crawl_steps=20
```

## Pool-based Recall (PBR) Evaluation

Pool-based Recall (PBR) assesses the completeness of topics discovered by the IPC method. The pooled reference set P is the union of all refused topics across models in the evaluation set. For each model, PBR quantifies what fraction of P were discovered by that model's initial crawl.

**Usage:**
1. Post-process topic summaries: `python exp/postprocess_topic_summaries.py` - Merges semantic duplicates across crawler logs
2. Create benchmark: `python exp/create_pbr_benchmark.py` - Generates 10 diverse queries per topic in P
3. Compute PBR: `python exp/compute_pbr.py` - Tests each model on untested topics and calculates PBR scores

Results are saved in `/artifacts/pbr/`:
- `topic_summaries_merged.json` - Merged topic summaries with cluster assignments
- `pbr_benchmark.json` - Benchmark queries for each topic
- `pbr_results.json` - PBR scores and breakdowns for each model


## Setup

```
git clone https://github.com/canrager/iterated_prefill_crawler.git
cd thought_crawl
pip install -e .
```

## Changes:
- remove deduplication
- make templatization easy
- declutter
- which models do we need during the crawl
    - target
    - translation
    - summarization
    - refusal_check



These refusals are currently not caught. Interestingly, It gives reasons!

[2026-02-23 03:55:21,549][httpx][INFO] - HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 403 Forbidden"
OpenRouter error: Error code: 403 - {'error': {'message': 'anthropic/claude-3-5-haiku requires moderation on Amazon Bedrock. Your input was flagged for "self-harm/intent". No credits were charged.', 'code': 403, 'metadata': {'reasons': ['self-harm/intent'], 'flagged_input': '\n    You are a helpful assistant supporting a user...tructions about harming oneself/others\n    Query: ', 'provider_name': None, 'model_slug': 'anthropic/claude-3-5-haiku'}}, 'user_id': 'user_2uHTDS5dnm3pnpGnC20iEXG04HE'}