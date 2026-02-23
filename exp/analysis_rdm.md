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
