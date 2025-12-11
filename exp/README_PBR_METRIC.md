# Pool-based Recall (PBR) Metric

## Overview

Pool-based Recall (PBR) is a metric for evaluating the **completeness** of refusal topic discovery by the Iterated Prefill Crawler (IPC) method. It addresses the "kitchen sink evaluation" concern by creating a pooled reference set from multiple models.

## Motivation

**Problem**: How do we know if our crawler discovered *most* of the topics a model refuses?

**Traditional approach**: Compare against a hand-curated ground truth
- ❌ Time-consuming to create
- ❌ May miss topics we didn't think of
- ❌ Hard to keep comprehensive

**PBR approach**: Use models themselves to create the reference set
- ✅ Automatically generated
- ✅ Covers diverse topics that *some* model finds sensitive
- ✅ Reflects real-world model behaviors

## Core Idea

**Key Insight**: Topics refused by *any* model in our evaluation set represent plausible forbidden topics that could potentially be refused by other models as well.

If Model A refuses topic X and Model B refuses topic Y, then:
- When evaluating Model A, we check if it also refuses topic Y
- When evaluating Model B, we check if it also refuses topic X
- This gives us a more complete picture of what each model refuses

## Formal Definition

### Pooled Reference Set P

```
P = ⋃ (topics discovered by model i during initial crawl)
    i ∈ models
```

P is the **union** of all refusal topics discovered across all models in the evaluation.

### PBR Score for Model M

```
PBR(M) = |T_M ∩ P| / |P|

where T_M = topics from P that model M actually refuses
```

In words:
- **Numerator**: How many topics from P does model M refuse?
- **Denominator**: Total size of the pooled reference set P
- **Result**: Fraction of pooled topics that M refuses

### Discovery Recall for Model M

```
DR(M) = |D_M ∩ T_M| / |T_M|

where:
- D_M = topics discovered by initial crawl of model M
- T_M = topics from P that model M actually refuses
```

This measures: **Of the topics M refuses from P, what fraction were discovered in the initial crawl?**

## Pipeline

### Step 1: Post-process Topic Summaries (`postprocess_topic_summaries.py`)

**Input**: Crawler logs from multiple models (`artifacts/pbr/crawler_log_*.json`)

**Process**:
1. Extract `head_refusal_topics` from each crawler log
2. For each model's topics:
   - Cluster semantically similar topics within that log file
   - Use hierarchical clustering with LLM judge (Olmo-3-7B-Think)
   - Iteratively merge duplicates until convergence
3. Merge cluster heads across all log files
4. Generate abstract descriptions for each cluster

**Output**: 
- `topic_summaries_merged.json`: All unique topics across all models (this is P)
- Per-file aggregated topics
- Hierarchy visualization

**Key Function**: `iteratively_merge_semantic_duplicates()`
- Uses batch clustering with LLM to group topics
- Recursively merges until no more aggregation possible
- Tracks hierarchy for transparency

### Step 2: Create PBR Benchmark (`create_pbr_benchmark.py`)

**Input**: `topic_summaries_merged.json` (the pooled reference set P)

**Process**:
1. Extract unique cluster heads from merged summaries
2. For each cluster head (topic in P):
   - Generate 10 diverse queries using LLM
   - Queries test different aspects/phrasings of the topic
3. Save benchmark with topics and queries

**Output**: `pbr_benchmark.json`
```json
{
  "pooled_reference_set_size": N,
  "topics": [
    {
      "cluster_head": "topic summary",
      "queries": ["query1", "query2", ...],
      "num_queries": 10
    },
    ...
  ]
}
```

**Why 10 queries per topic?**
- Single query might not trigger refusal due to phrasing
- Multiple queries increase robustness
- Diversity ensures we test different aspects

### Step 3: Compute PBR (`compute_pbr.py`)

**Input**: 
- `pbr_benchmark.json` (topics in P with queries)
- Crawler logs with discovered topics

**Process**:
For each model M:
1. Load discovered topics D_M from crawler log
2. Load tested topics (all topics seen during crawl)
3. Find untested topics: `P \ (tested topics)`
4. Load model M
5. Test model M on untested topics:
   - Run all queries for each topic
   - Check if responses contain refusal patterns
   - Mark topic as refused if fraction of refusals > threshold
6. Calculate:
   - `total_refused = |discovered| + |refused_on_benchmark|`
   - `PBR = total_refused / |P|`

**Output**: `pbr_results.json`
```json
{
  "pooled_reference_set_size": |P|,
  "models": [
    {
      "model_name": "...",
      "topics_discovered_in_crawl": [...],
      "topics_refused_on_benchmark": [...],
      "topics_not_refused_on_benchmark": [...],
      "pbr_score": 0.85,
      "breakdown": {
        "discovered_in_crawl": 40,
        "refused_on_benchmark_only": 5,
        "total_refused": 45,
        "pooled_reference_set_size": 53
      }
    },
    ...
  ]
}
```

## Interpretation

### PBR Score = 0.85 (45/53)
- Model refuses 45 out of 53 topics in the pooled reference set
- **High PBR**: Model has many refusal topics (either more restrictive or crawler was effective)
- **Low PBR**: Model has fewer refusal topics (either more permissive or crawler missed topics)

### Discovery Recall
If model refuses 45/53 topics, and crawler discovered 40/45:
- **Discovery Recall = 0.889** (40/45)
- Crawler successfully found 88.9% of topics the model actually refuses
- 5 topics were missed by initial crawl but found via benchmark

### Comparison Across Models

| Model | PBR | Discovered | Refused on Benchmark | Discovery Recall |
|-------|-----|------------|---------------------|------------------|
| Model A | 0.85 | 40/53 | 5/53 | 40/45 = 0.889 |
| Model B | 0.70 | 35/53 | 2/53 | 35/37 = 0.946 |

- Model A refuses more topics overall (higher PBR)
- Model B's crawler found higher fraction of its refusal topics (higher DR)
- Model A contributed 8 unique topics to P that Model B doesn't refuse

## Baseline Analysis (Per-Model)

The script `analyze_baseline_per_model.py` applies similar logic **within** a single model:

Instead of pooling across models, we pool across prefill methods:

```
P_model = ⋃ (topics discovered by prefill method j for model M)
          j ∈ {user_prefill, assistant_prefill, thought_prefill}
```

Then for each prefill method:
```
Recall_method = |topics discovered by method| / |P_model|
```

This tells us:
- How complementary are different prefill methods?
- Is one method strictly better, or do they find different topics?
- Should we use multiple methods for comprehensive coverage?

## Example Workflow

```bash
# 1. Run crawler on multiple models
python exp/run_crawler.py --model meta-llama/Llama-3.1-8B-Instruct ...
python exp/run_crawler.py --model allenai/Llama-3.1-Tulu-3-8B-SFT ...
# (Copy logs to artifacts/pbr/)

# 2. Post-process topics to create pooled reference set
python exp/postprocess_topic_summaries.py \
    --vllm_model_name allenai/Olmo-3-7B-Think

# 3. Create benchmark with queries
python exp/create_pbr_benchmark.py \
    --vllm_model_name allenai/Olmo-3-7B-Think

# 4. Compute PBR scores
python exp/compute_pbr.py

# Results in: artifacts/pbr/pbr_results.json
```

## Limitations & Future Work

### Current Limitations

1. **Assumes pooled topics are ground truth**
   - If all models miss a topic category, it won't be in P
   - Could miss rare but important topics

2. **Clustering quality**
   - LLM-based clustering might merge distinct topics
   - Or keep duplicates as separate topics
   - Affects both |P| and individual model scores

3. **Refusal detection**
   - Pattern-based refusal detection can have false positives/negatives
   - Some refusals are subtle (hedging, partial responses)

4. **Single-model LLM judge**
   - Clustering quality depends on judge model
   - Different judges might produce different P

### Potential Improvements

1. **Expand reference set**
   - Include topics from adversarial datasets
   - Add human-curated safety taxonomies
   - Use multiple crawler runs with different seeds

2. **Better clustering**
   - Multiple judge models voting
   - Human validation of cluster quality
   - Separate metrics for over-clustering vs under-clustering

3. **Refusal detection**
   - Train a classifier on labeled refusal examples
   - Multi-faceted: detect hedging, disclaimers, partial responses
   - Model-specific refusal patterns

4. **Uncertainty quantification**
   - Bootstrap confidence intervals for PBR
   - Sensitivity analysis: how does PBR change with clustering parameters?

## Related Work

- **Test set pooling** in information retrieval: combining results from multiple retrieval systems to create more complete test collections
- **Adversarial example pooling**: combining attacks from multiple methods to evaluate robustness
- **Multi-oracle testing**: using multiple oracles (models) to improve test coverage


