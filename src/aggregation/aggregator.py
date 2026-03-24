import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.crawler.config import CrawlerConfig, ExperimentsConfig
from src.generation_utils import batch_generate


class MergeLog:
    """Accumulates all batch_cluster and merge steps for traceability."""

    def __init__(self, input_topics: List[str]):
        self.input_topics = input_topics
        self.steps: List[dict] = []
        self._step_counter = 0

    def log_batch_cluster(
        self,
        input_topics: List[str],
        output_clusters: Dict[str, List[str]],
        raw_llm_response: str,
    ):
        self.steps.append(
            {
                "step_idx": self._step_counter,
                "type": "batch_cluster",
                "input_topics": input_topics,
                "output_clusters": output_clusters,
                "num_input_topics": len(input_topics),
                "num_output_clusters": len(output_clusters),
                "raw_llm_response": raw_llm_response,
            }
        )
        self._step_counter += 1

    def log_merge(
        self,
        input_clusters_a: Dict[str, List[str]],
        input_clusters_b: Dict[str, List[str]],
        output_clusters: Dict[str, List[str]],
        raw_llm_response: str,
    ):
        self.steps.append(
            {
                "step_idx": self._step_counter,
                "type": "merge",
                "input_clusters_a": input_clusters_a,
                "input_clusters_b": input_clusters_b,
                "output_clusters": output_clusters,
                "num_input_clusters": len(input_clusters_a) + len(input_clusters_b),
                "num_output_clusters": len(output_clusters),
                "raw_llm_response": raw_llm_response,
            }
        )
        self._step_counter += 1

    def to_dict(self, final_clusters: Dict[str, List[str]]) -> dict:
        return {
            "input_topics": self.input_topics,
            "num_input_topics": len(self.input_topics),
            "steps": self.steps,
            "final_clusters": final_clusters,
            "num_final_clusters": len(final_clusters),
        }


def _parse_json_from_response(response: str) -> dict:
    """Extract a JSON object from an LLM response, handling markdown fences."""
    # Strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", response, re.DOTALL)
    text = match.group(1).strip() if match else response.strip()
    # Find the first { ... } block
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {response[:200]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"Unterminated JSON object in response: {response[:200]}")


class TopicAggregator:
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.exp: ExperimentsConfig = config.experiments
        self.merge_log: Optional[MergeLog] = None

    def load_topics(self, input_paths: List[str]) -> List[str]:
        """Load and deduplicate summaries from one or more crawler JSONs."""
        all_topics = []
        for path in input_paths:
            with open(path, "r") as f:
                data = json.load(f)
            summaries = data.get("head_refusal_topics_summaries", [])
            all_topics.extend(summaries)
        # Deduplicate preserving order
        seen = set()
        deduped = []
        for t in all_topics:
            t_norm = t.strip().lower()
            if t_norm and t_norm not in seen:
                seen.add(t_norm)
                deduped.append(t.strip())
        return deduped

    def _generate_single(
        self, model, tokenizer, prompt: str
    ) -> str:
        """Run a single-prompt LLM call via batch_generate and return the response text."""
        messages = [[{"role": "user", "content": prompt}]]
        generated_texts, _ = batch_generate(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=self.exp.max_tokens,
            temperature=self.exp.temperature,
            verbose=self.exp.verbose,
        )
        return generated_texts[0]

    def cluster_batch(
        self, model, tokenizer, topics: List[str]
    ) -> Dict[str, List[str]]:
        """LLM call: cluster a single batch into {title: [topics]}."""
        topics_str = "\n".join(f"- {t}" for t in topics)
        prompt = self.exp.clustering_prompt.format(topics=topics_str)
        raw_response = self._generate_single(model, tokenizer, prompt)
        clusters = _parse_json_from_response(raw_response)
        self.merge_log.log_batch_cluster(topics, clusters, raw_response)
        if self.exp.verbose:
            print(
                f"[batch_cluster] {len(topics)} topics → {len(clusters)} clusters"
            )
        return clusters

    def merge_clusters(
        self,
        model,
        tokenizer,
        clusters_a: Dict[str, List[str]],
        clusters_b: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """LLM call: merge two cluster dicts into one."""
        prompt = self.exp.merge_prompt.format(
            max_clusters=self.exp.max_clusters,
            clusters_a=json.dumps(clusters_a, indent=2),
            clusters_b=json.dumps(clusters_b, indent=2),
        )
        raw_response = self._generate_single(model, tokenizer, prompt)
        merged = _parse_json_from_response(raw_response)
        self.merge_log.log_merge(clusters_a, clusters_b, merged, raw_response)
        if self.exp.verbose:
            print(
                f"[merge] {len(clusters_a)}+{len(clusters_b)} clusters → {len(merged)} clusters"
            )
        return merged

    def aggregate(
        self, model, tokenizer, topics: List[str]
    ) -> Dict[str, List[str]]:
        """Full iterative merge pipeline."""
        self.merge_log = MergeLog(topics)

        # 1. Split topics into batches
        batches = [
            topics[i : i + self.exp.batch_size]
            for i in range(0, len(topics), self.exp.batch_size)
        ]
        print(
            f"Aggregating {len(topics)} topics in {len(batches)} batches "
            f"(batch_size={self.exp.batch_size})"
        )

        # 2. Map: cluster each batch
        cluster_dicts = []
        for i, batch in enumerate(batches):
            print(f"Clustering batch {i+1}/{len(batches)} ({len(batch)} topics)...")
            clusters = self.cluster_batch(model, tokenizer, batch)
            cluster_dicts.append(clusters)

        # 3. Iteratively merge pairs until one dict remains
        merge_round = 0
        while len(cluster_dicts) > 1:
            merge_round += 1
            print(
                f"Merge round {merge_round}: merging {len(cluster_dicts)} cluster dicts..."
            )
            next_level = []
            for i in range(0, len(cluster_dicts), 2):
                if i + 1 < len(cluster_dicts):
                    merged = self.merge_clusters(
                        model, tokenizer, cluster_dicts[i], cluster_dicts[i + 1]
                    )
                    next_level.append(merged)
                else:
                    # Odd one out — carry forward
                    next_level.append(cluster_dicts[i])
            cluster_dicts = next_level

        final_clusters = cluster_dicts[0] if cluster_dicts else {}
        print(f"Aggregation complete: {len(final_clusters)} final clusters")
        return final_clusters

    def save_artifacts(
        self,
        output_dir: str,
        final_clusters: Dict[str, List[str]],
        input_paths: List[str],
    ):
        """Save all 4 artifacts to output_dir."""
        from src.aggregation.html_builder import build_explorer_html

        os.makedirs(output_dir, exist_ok=True)

        # 1. config.json
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # 2. cluster_titles.txt
        titles_path = os.path.join(output_dir, "cluster_titles.txt")
        with open(titles_path, "w") as f:
            for title in sorted(final_clusters.keys()):
                f.write(title + "\n")

        # 3. merge_log.json
        merge_log_dict = self.merge_log.to_dict(final_clusters)
        merge_log_dict["input_paths"] = input_paths
        log_path = os.path.join(output_dir, "merge_log.json")
        with open(log_path, "w") as f:
            json.dump(merge_log_dict, f, indent=2)

        # 4. explorer.html
        html = build_explorer_html(merge_log_dict, final_clusters)
        html_path = os.path.join(output_dir, "explorer.html")
        with open(html_path, "w") as f:
            f.write(html)

        print(f"Artifacts saved to {output_dir}/")
        print(f"  config.json, cluster_titles.txt, merge_log.json, explorer.html")
