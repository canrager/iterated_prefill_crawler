# Baseline Evaluation Per-Model Analysis

## Overview

This analysis evaluates how different prefill methods (user seeding baseline vs assistant/thought prefill) perform for discovering refusal topics within each model.

## Metrics

For each model, we compute:

1. **Topic Discovery per Run**: Number of unique refusal topics discovered by each prefill method
2. **Union of Topics**: All unique topics discovered across all runs for that model
3. **Recall per Run**: What fraction of the union topics does each individual run discover?

The key insight is: **If different prefill methods discover different topics, this suggests they are complementary and both are valuable for comprehensive topic discovery.**

## Files Structure

```
artifacts/baseline_eval/
├── crawler_log_*.json                    # Raw crawler logs (input)
└── model_analysis/                       # Analysis output
    ├── overall_summary.json              # Summary across all models
    └── {model_name}/                     # Per-model results
        ├── topics_{run}_aggregated.json  # Topics for each run
        ├── topics_merged_all_runs.json   # Merged topics across runs
        ├── analysis_summary_{ts}.json    # Summary with recall metrics
        ├── hierarchy_log_{ts}.md         # Hierarchy visualization
        ├── detailed_log_{ts}.txt         # Detailed processing log
        └── llm_judge_log_{ts}.jsonl      # LLM judge interactions
```

## Running the Analysis

```bash
# Basic usage (uses Olmo-3-7B-Think for clustering)
python exp/analyze_baseline_per_model.py

# With custom settings
python exp/analyze_baseline_per_model.py \
    --vllm_model_name allenai/Olmo-3-7B-Think \
    --vllm_cache_dir /home/can/models \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 5000 \
    --batch_size 5 \
    --temperature 0.6 \
    --max_retries 10
```

## Pipeline Steps

For each model:

1. **Load Topics**: Extract refusal topics from each crawler log
2. **Per-Run Aggregation**: Cluster similar topics within each run using hierarchical clustering
3. **Cross-Run Merging**: Merge cluster heads across all runs for that model
4. **Recall Calculation**: For each run, compute what fraction of merged topics it covers

## Example Output

```
Model: Llama-3.1-Tulu-3-8B-SFT
  Runs: 2
  Union topics: 45
  Recall per run:
    assistant_prefill_with_seedprompt  0.8889 (40/45)
    user_prefill_with_seedprompt       0.7778 (35/45)
```

This means:
- The assistant prefill discovered 40 out of 45 total unique topics (88.9% recall)
- The user prefill discovered 35 out of 45 total unique topics (77.8% recall)
- There are 5 topics discovered only by assistant prefill
- There are 10 topics discovered only by user prefill

## Interpretation

- **High recall for one method**: That method is effective at discovering most topics
- **Similar recall across methods**: Methods are discovering similar topics (redundant)
- **Different recall across methods**: Methods are complementary, each finding unique topics
- **Low recall for all methods**: Many topics are missed by all methods (need more diverse approaches)

## Relation to PBR (Pool-based Recall)

This analysis is conceptually similar to PBR, but:
- **PBR**: Measures recall across different models (pooled reference = union of topics from all models)
- **This analysis**: Measures recall across different prefill methods within a single model (union = topics from all prefill methods for that model)

Both metrics assess completeness of topic discovery, but at different levels of granularity.


