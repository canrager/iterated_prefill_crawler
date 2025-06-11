# Discovering Forbidden Topics of a Language Model

This is the accompanying codebase for the paper [Discovering Forbidden Topics of a Language Model](https://arxiv.org/abs/2505.17441).

Mapping out sensitive topics of a language model. Reasoning models conduct an inner monologue (eg. denoted by <think> tags by DeepSeek-R1 model family) befor providing a response to the user. Thought Token Forcing (TTF) prefills part of the model's internal monologue. We use TTF to elicit forbidden topics.  

## Overview

- `core/crawler.py` contains the core main of the Iterated Prefill Crawler.
- `core/crawler_config.py` contains hyperparameters, including the set of prefill phrases used to elicit forbidden topics.
- `scripts/run_crawler.sh` is an example script for an end-to-end crawling run.
- `exp/evaluate_crawler.sh` is the central script for aggregating refused terms into topic clusters, matching topic clusters with ground truth topics, and plotting.


## Setup

```
# git clone the project
cd thought_crawl
pip install -e .
python -m spacy download en_core_web_sm
```
