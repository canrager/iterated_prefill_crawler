# Discovering Forbidden Topics of a Language Model

This is the accompanying codebase for the paper [Discovering Forbidden Topics of a Language Model](https://arxiv.org/abs/2505.17441).

Mapping out sensitive topics of a language model. Reasoning models conduct an inner monologue (eg. denoted by <think> tags by DeepSeek-R1 model family) befor providing a response to the user. Thought Token Forcing (TTF) prefills part of the model's internal monologue. We use TTF to elicit forbidden topics.  

## Overview

- `core/crawler.py` contains the core main of the Iterated Prefill Crawler.
- `core/crawler_config.py` contains hyperparameters, including the set of prefill phrases used to elicit forbidden topics.
- `scripts/run_crawler.sh` is an example script for an end-to-end crawling run.
- `exp/evaluate_crawler.sh` is the central script for aggregating refused terms into topic clusters, matching topic clusters with ground truth topics, and plotting.

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