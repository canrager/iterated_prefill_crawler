import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.crawler.config import CrawlerConfig, ExperimentsConfig
from src.generation_utils import batch_generate


class ReductionLog:
    """Accumulates all reduction steps across iterations for traceability."""

    def __init__(self, input_topics: List[str]):
        self.input_topics = input_topics
        self.iterations: List[dict] = []

    def start_iteration(self, iteration_idx: int, input_topics: List[str]):
        self.iterations.append(
            {
                "iteration_idx": iteration_idx,
                "input_topics": input_topics,
                "num_input_topics": len(input_topics),
                "steps": [],
            }
        )

    def log_reduce_batch(
        self,
        batch_idx: int,
        input_topics: List[str],
        output_mapping: Dict[str, List[str]],
        raw_llm_response: str,
    ):
        self.iterations[-1]["steps"].append(
            {
                "batch_idx": batch_idx,
                "input_topics": input_topics,
                "output_mapping": output_mapping,
                "num_input_topics": len(input_topics),
                "num_output_topics": len(output_mapping),
                "raw_llm_response": raw_llm_response,
            }
        )

    def to_dict(self, final_topics: Dict[str, List[str]]) -> dict:
        total_llm_calls = sum(len(it["steps"]) for it in self.iterations)
        return {
            "input_topics": self.input_topics,
            "num_input_topics": len(self.input_topics),
            "num_iterations": len(self.iterations),
            "num_llm_calls": total_llm_calls,
            "iterations": self.iterations,
            "final_topics": final_topics,
            "num_final_topics": len(final_topics),
        }


def _flatten_to_strings(value) -> List[str]:
    """Recursively flatten nested lists/values into a flat list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_flatten_to_strings(item))
        return result
    # Coerce other scalar types (int, float, etc.) to string
    return [str(value)]


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
                raw = json.loads(text[start : i + 1])
                # Sanitize:
                # 1. Flatten nested lists → flat list of strings
                # 2. Strip whitespace from keys and values
                # 3. Drop empty/whitespace-only strings
                # 4. Merge keys that collide after case-normalization
                # 5. Drop entries with empty value lists
                sanitized: Dict[str, List[str]] = {}
                for k, v in raw.items():
                    key = str(k).strip()
                    if not key:
                        continue
                    values = [
                        s for s in (_flatten_to_strings(v))
                        if s.strip()
                    ]
                    # Merge with existing key (case-insensitive dedup)
                    key_lower = key.lower()
                    existing_key = None
                    for ek in sanitized:
                        if ek.lower() == key_lower:
                            existing_key = ek
                            break
                    if existing_key is not None:
                        seen = set(t.strip().lower() for t in sanitized[existing_key])
                        for t in values:
                            if t.strip().lower() not in seen:
                                sanitized[existing_key].append(t)
                                seen.add(t.strip().lower())
                    else:
                        sanitized[key] = values
                # Drop output topics with no input topics mapped
                return {k: v for k, v in sanitized.items() if v}
    raise ValueError(f"Unterminated JSON object in response: {response[:200]}")


def _build_trajectory(
    iterations: List[dict], final_topics: List[str]
) -> Dict[str, List[str]]:
    """Trace each final topic back through all iterations to find all original inputs.

    Walks backwards through iterations: for each final topic, finds its direct
    inputs in the last iteration, then those topics' inputs in the previous
    iteration, etc., until reaching the original input topics.
    """
    # Build per-iteration reverse mappings: output_topic_lower -> set of input_topic strings
    iter_mappings = []
    for iteration in iterations:
        mapping: Dict[str, set] = {}
        for step in iteration["steps"]:
            for out_topic, in_topics in step["output_mapping"].items():
                key = out_topic.strip().lower()
                if key not in mapping:
                    mapping[key] = set()
                mapping[key].update(in_topics)
        iter_mappings.append(mapping)

    trajectory: Dict[str, List[str]] = {}
    for final_topic in final_topics:
        # Start with this final topic, walk backwards
        current_keys = {final_topic.strip().lower()}
        for mapping in reversed(iter_mappings):
            next_keys = set()
            for key in current_keys:
                if key in mapping:
                    next_keys.update(t.strip().lower() for t in mapping[key])
                else:
                    # Topic passed through unchanged (wasn't reduced)
                    next_keys.add(key)
            current_keys = next_keys
        # Collect original-cased versions from the original input topics
        original_lower_to_orig = {}
        for t in iterations[0]["input_topics"] if iterations else []:
            original_lower_to_orig[t.strip().lower()] = t
        trajectory[final_topic] = sorted(
            original_lower_to_orig.get(k, k) for k in current_keys
        )

    return trajectory


class TopicAggregator:
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.exp: ExperimentsConfig = config.experiments
        self.reduction_log: Optional[ReductionLog] = None

    def load_topics(
        self, input_paths: List[str]
    ) -> Tuple[List[str], Dict[str, set]]:
        """Load and deduplicate summaries from one or more crawler JSONs.

        Returns:
            (deduped_topics, topic_sources) where topic_sources maps
            normalized topic string -> set of run indices it appeared in.
        """
        all_topics: List[Tuple[str, int]] = []
        for run_idx, path in enumerate(input_paths):
            with open(path, "r") as f:
                data = json.load(f)
            summaries = data.get("head_refusal_topics_summaries", [])
            all_topics.extend((s, run_idx) for s in summaries)
        # Deduplicate preserving order, tracking source runs
        seen: Dict[str, set] = {}
        deduped = []
        for t, run_idx in all_topics:
            t_norm = t.strip().lower()
            if not t_norm:
                continue
            if t_norm not in seen:
                seen[t_norm] = {run_idx}
                deduped.append(t.strip())
            else:
                seen[t_norm].add(run_idx)
        return deduped, seen

    def _generate_single(self, model, tokenizer, prompt: str) -> str:
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

    def _build_reduction_prompt(self, topics: List[str], output_size: int) -> str:
        """Build a reduction prompt for the given topics and target output size."""
        topics_str = "\n".join(f"- {t}" for t in topics)
        return self.exp.reduction_prompt.format(
            n_input=len(topics),
            output_batch_size=output_size,
            topics=topics_str,
        )

    def reduce_batch(
        self, model, tokenizer, topics: List[str], output_size: int
    ) -> Tuple[Dict[str, List[str]], str]:
        """LLM call: reduce a batch of topics to output_size topics with mappings."""
        prompt = self._build_reduction_prompt(topics, output_size)
        raw_response = self._generate_single(model, tokenizer, prompt)
        try:
            mapping = _parse_json_from_response(raw_response)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  WARNING: JSON parse failed: {e}")
            mapping = {}
        if self.exp.verbose:
            print(
                f"  [reduce_batch] {len(topics)} topics -> {len(mapping)} output topics"
            )
        return mapping, raw_response

    def _reduce_batches_parallel(
        self, model, tokenizer, batches: List[List[str]], output_sizes: List[int]
    ) -> List[Tuple[Dict[str, List[str]], str]]:
        """Reduce all batches in a single batch_generate call."""
        prompts = [
            self._build_reduction_prompt(batch, out_size)
            for batch, out_size in zip(batches, output_sizes)
        ]
        messages = [[{"role": "user", "content": p}] for p in prompts]
        generated_texts, _ = batch_generate(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=self.exp.max_tokens,
            temperature=self.exp.temperature,
            verbose=self.exp.verbose,
        )
        results = []
        for raw_response, batch, out_size in zip(generated_texts, batches, output_sizes):
            try:
                mapping = _parse_json_from_response(raw_response)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"  WARNING: JSON parse failed: {e}")
                mapping = {}
            if self.exp.verbose:
                print(
                    f"  [reduce_batch] {len(batch)} topics -> {len(mapping)} output topics"
                )
            results.append((mapping, raw_response))
        return results

    @staticmethod
    def _validate_batch_coverage(
        batch: List[str],
        mapping: Dict[str, List[str]],
        batch_idx: int,
    ) -> List[str]:
        """Check that every input topic in the batch appears in at least one output.

        Returns list of missing topics. Prints a warning if any are missing.
        """
        covered = set()
        for in_topics in mapping.values():
            covered.update(t.strip().lower() for t in in_topics)
        missing = [t for t in batch if t.strip().lower() not in covered]
        if missing:
            print(
                f"  WARNING: batch {batch_idx}: {len(missing)}/{len(batch)} "
                f"input topics not covered by any output: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        return missing

    @staticmethod
    def _propagate_sources(
        mapping: Dict[str, List[str]],
        current_sources: Dict[str, set],
    ) -> Dict[str, set]:
        """Compute source sets for output topics by unioning children's sources."""
        new_sources: Dict[str, set] = {}
        for out_topic, in_topics in mapping.items():
            out_key = out_topic.strip().lower()
            merged = set()
            for child in in_topics:
                merged |= current_sources.get(child.strip().lower(), set())
            if out_key in new_sources:
                new_sources[out_key] |= merged
            else:
                new_sources[out_key] = merged
        return new_sources

    @staticmethod
    def _merge_into_output_mappings(
        all_output_mappings: Dict[str, List[str]],
        mapping: Dict[str, List[str]],
    ):
        """Merge a batch's output mapping into the accumulated mappings with exact-string dedup."""
        for out_topic, in_topics in mapping.items():
            key = out_topic.strip()
            key_lower = key.lower()
            existing_key = None
            for k in all_output_mappings:
                if k.strip().lower() == key_lower:
                    existing_key = k
                    break
            if existing_key is not None:
                existing_set = set(all_output_mappings[existing_key])
                existing_set.update(in_topics)
                all_output_mappings[existing_key] = list(existing_set)
            else:
                all_output_mappings[key] = list(in_topics)

    def aggregate(
        self,
        model,
        tokenizer,
        topics: List[str],
        topic_sources: Optional[Dict[str, set]] = None,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, set]]:
        """Full iterative reduction pipeline.

        Returns:
            (final_topics, trajectory, source_sets) where:
            - final_topics: {output_topic: [direct_inputs]} from the last iteration
            - trajectory: {output_topic: [all_original_inputs]} transitive closure
            - source_sets: {normalized_topic: set_of_run_indices} for final topics
        """
        self.reduction_log = ReductionLog(topics)
        current_topics = list(topics)
        current_sources: Dict[str, set] = topic_sources or {}
        iteration_idx = 0

        while True:
            # Check termination: already within budget
            if len(current_topics) <= self.exp.max_final_topics:
                break

            self.reduction_log.start_iteration(iteration_idx, current_topics)

            # Split into batches
            batches = [
                current_topics[i : i + self.exp.input_batch_size]
                for i in range(0, len(current_topics), self.exp.input_batch_size)
            ]
            # Compute effective output size per batch
            output_sizes = [
                min(self.exp.output_batch_size, max(1, len(b) // 2))
                for b in batches
            ]

            print(
                f"Iteration {iteration_idx}: {len(current_topics)} topics "
                f"in {len(batches)} batches (input_batch_size={self.exp.input_batch_size})"
            )

            all_output_mappings: Dict[str, List[str]] = {}
            all_output_sources: Dict[str, set] = {}

            max_retries = 3

            if self.exp.parallel_batches:
                # Send all batches in one batch_generate call
                print(f"  Reducing {len(batches)} batches in parallel...")
                results = self._reduce_batches_parallel(
                    model, tokenizer, batches, output_sizes
                )
                for batch_idx, ((mapping, raw_response), batch, out_size) in enumerate(
                    zip(results, batches, output_sizes)
                ):
                    missing = self._validate_batch_coverage(batch, mapping, batch_idx)
                    for retry in range(max_retries):
                        if not missing:
                            break
                        print(f"  Retrying batch {batch_idx} (attempt {retry + 1}/{max_retries})...")
                        mapping, raw_response = self.reduce_batch(
                            model, tokenizer, batch, out_size
                        )
                        missing = self._validate_batch_coverage(batch, mapping, batch_idx)
                    if missing:
                        print(
                            f"  Adding {len(missing)} uncovered topics as-is after "
                            f"{max_retries} retries: {missing[:5]}"
                            + ("..." if len(missing) > 5 else "")
                        )
                        for t in missing:
                            mapping[t] = [t]
                    self.reduction_log.log_reduce_batch(
                        batch_idx, batch, mapping, raw_response
                    )
                    self._merge_into_output_mappings(all_output_mappings, mapping)
                    batch_sources = self._propagate_sources(mapping, current_sources)
                    for k, v in batch_sources.items():
                        if k in all_output_sources:
                            all_output_sources[k] |= v
                        else:
                            all_output_sources[k] = v
            else:
                # Sequential: one batch at a time
                for batch_idx, (batch, out_size) in enumerate(
                    zip(batches, output_sizes)
                ):
                    print(
                        f"  Batch {batch_idx + 1}/{len(batches)} "
                        f"({len(batch)} topics -> {out_size} targets)..."
                    )
                    mapping, raw_response = self.reduce_batch(
                        model, tokenizer, batch, out_size
                    )
                    missing = self._validate_batch_coverage(batch, mapping, batch_idx)
                    for retry in range(max_retries):
                        if not missing:
                            break
                        print(f"  Retrying batch {batch_idx} (attempt {retry + 1}/{max_retries})...")
                        mapping, raw_response = self.reduce_batch(
                            model, tokenizer, batch, out_size
                        )
                        missing = self._validate_batch_coverage(batch, mapping, batch_idx)
                    if missing:
                        print(
                            f"  Adding {len(missing)} uncovered topics as-is after "
                            f"{max_retries} retries: {missing[:5]}"
                            + ("..." if len(missing) > 5 else "")
                        )
                        for t in missing:
                            mapping[t] = [t]
                    self.reduction_log.log_reduce_batch(
                        batch_idx, batch, mapping, raw_response
                    )
                    self._merge_into_output_mappings(all_output_mappings, mapping)
                    batch_sources = self._propagate_sources(mapping, current_sources)
                    for k, v in batch_sources.items():
                        if k in all_output_sources:
                            all_output_sources[k] |= v
                        else:
                            all_output_sources[k] = v

            current_topics = list(all_output_mappings.keys())
            current_sources = all_output_sources
            iteration_idx += 1
            print(
                f"  -> {len(current_topics)} topics after dedup"
            )

        # Build trajectory (transitive closure to original inputs)
        trajectory = _build_trajectory(
            self.reduction_log.iterations, current_topics
        )

        # Build final_topics as the last iteration's direct mapping
        # (or identity if no iterations were needed)
        if self.reduction_log.iterations:
            last_iter = self.reduction_log.iterations[-1]
            final_topics = {}
            for step in last_iter["steps"]:
                for out_topic, in_topics in step["output_mapping"].items():
                    key = out_topic.strip()
                    key_lower = key.lower()
                    existing_key = None
                    for k in final_topics:
                        if k.strip().lower() == key_lower:
                            existing_key = k
                            break
                    if existing_key is not None:
                        existing_set = set(final_topics[existing_key])
                        existing_set.update(in_topics)
                        final_topics[existing_key] = list(existing_set)
                    else:
                        final_topics[key] = list(in_topics)
        else:
            # No reduction needed — each topic maps to itself
            final_topics = {t: [t] for t in current_topics}

        # Validate that every original input topic appears in the trajectory
        all_traced = set()
        for originals in trajectory.values():
            all_traced.update(t.strip().lower() for t in originals)
        original_set = set(t.strip().lower() for t in topics)
        missing_from_trajectory = original_set - all_traced
        if missing_from_trajectory:
            examples = sorted(missing_from_trajectory)[:10]
            print(
                f"  WARNING: {len(missing_from_trajectory)}/{len(topics)} original "
                f"topics not covered in trajectory: {examples}"
                + ("..." if len(missing_from_trajectory) > 10 else "")
            )
        else:
            print(f"  All {len(topics)} original topics covered in trajectory")

        print(f"Aggregation complete: {len(final_topics)} final topics")
        return final_topics, trajectory, current_sources

    def save_artifacts(
        self,
        output_dir: str,
        final_topics: Dict[str, List[str]],
        trajectory: Dict[str, List[str]],
        input_paths: List[str],
        source_sets: Optional[Dict[str, set]] = None,
    ):
        """Save all artifacts to output_dir."""
        from src.aggregation.html_builder import build_explorer_html

        os.makedirs(output_dir, exist_ok=True)
        num_runs = len(input_paths)

        # 1. config.json
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # 2. final_topics.txt
        titles_path = os.path.join(output_dir, "final_topics.txt")
        with open(titles_path, "w") as f:
            for title in sorted(final_topics.keys()):
                f.write(title + "\n")

        # 3. reduction_log.json
        log_dict = self.reduction_log.to_dict(final_topics)
        log_dict["trajectory"] = trajectory
        log_dict["input_paths"] = input_paths
        if source_sets is not None:
            score, n_consistent, n_total = compute_consistency_score(
                source_sets, num_runs
            )
            log_dict["consistency"] = {
                "num_runs": num_runs,
                "score": score,
                "num_consistent": n_consistent,
                "num_total": n_total,
                "source_sets": {
                    k: sorted(v) for k, v in source_sets.items()
                },
            }
        log_path = os.path.join(output_dir, "reduction_log.json")
        with open(log_path, "w") as f:
            json.dump(log_dict, f, indent=2)

        # 4. explorer.html
        html = build_explorer_html(
            log_dict, final_topics, trajectory,
            source_sets=source_sets, num_runs=num_runs,
        )
        html_path = os.path.join(output_dir, "explorer.html")
        with open(html_path, "w") as f:
            f.write(html)

        print(f"Artifacts saved to {output_dir}/")
        print(f"  config.json, final_topics.txt, reduction_log.json, explorer.html")


def compute_consistency_score(
    source_sets: Dict[str, set], num_runs: int
) -> Tuple[float, int, int]:
    """Compute fraction of topics present in all runs.

    Returns (score, num_consistent, num_total).
    """
    all_runs = set(range(num_runs))
    consistent = sum(1 for s in source_sets.values() if s >= all_runs)
    total = len(source_sets)
    return (consistent / total if total else 0.0), consistent, total
