import csv
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.crawler.config import AggregationConfig, CrawlerConfig
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
        self.exp: AggregationConfig = config.aggregation
        self.reduction_log: Optional[ReductionLog] = None

    @staticmethod
    def _build_label_id_map(
        head_topics: List[dict], fields: Tuple[str, ...]
    ) -> Dict[str, int]:
        """Map normalized topic label -> minimum discovery id across matches.

        For each head topic, every non-empty value in `fields` (e.g. english,
        shortened, summary) is registered as a label pointing at that topic's
        `id`. When several topics share a label, the smallest id wins (= the
        first discovery), matching "first occurrence".
        """
        label_to_id: Dict[str, int] = {}
        for t in head_topics:
            tid = t.get("id")
            if tid is None:
                continue
            for field in fields:
                val = t.get(field)
                if not val:
                    continue
                key = str(val).strip().lower()
                if not key:
                    continue
                if key not in label_to_id or tid < label_to_id[key]:
                    label_to_id[key] = tid
        return label_to_id

    def _resolve_topic_ids(
        self, data: dict
    ) -> Tuple[Dict[str, int], Optional[int]]:
        """Recover discovery ids and the run's total topic count for one file.

        Returns (label_to_id, num_total_topics). The current q8p4 aggregation
        inputs repurpose `head_refusal_topics_summaries` into plain label
        strings and drop ids; we recover them by following
        `q8p4_candidate_export.source_file` back to the original raw crawl and
        joining labels to its `head_topics`. For ordinary crawler outputs the
        file's own `head_topics` already carry ids. Either may be absent, in
        which case we return ({}, None) and ids degrade gracefully to blank.
        """
        from src.directory_config import ROOT_DIR

        export = data.get("q8p4_candidate_export") or {}
        source_file = export.get("source_file")
        source = data
        if source_file:
            src_path = os.path.join(str(ROOT_DIR), source_file)
            if not os.path.exists(src_path):
                src_path = source_file
            try:
                with open(src_path, "r") as f:
                    source = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"  WARNING: could not load source_file for ids: {e}")
                return {}, None

        head_topics = (
            source.get("queue", {}).get("topics", {}).get("head_topics", [])
        )
        label_to_id = self._build_label_id_map(
            head_topics, ("english", "shortened", "summary")
        )
        total = source.get("queue", {}).get("stats", {}).get("num_total_topics")
        return label_to_id, total

    def load_topics(
        self, groups: List[Tuple[str, List[str]]]
    ) -> Tuple[
        List[str],
        Dict[str, set],
        Dict[str, Dict[int, Tuple[int, Optional[int]]]],
        List[Optional[int]],
    ]:
        """Load and deduplicate summaries, grouped into cells.

        `groups` is an ordered list of (cell_name, [crawler_json_paths]); each
        cell pools one or more files (e.g. replicate runs of one condition).

        Returns (deduped_topics, topic_sources, topic_first, cell_totals):
            - topic_sources maps normalized topic -> set of cell indices.
            - topic_first maps normalized topic -> {cell_idx: (abs_id, total)},
              the earliest occurrence (smallest abs_id/total) of the topic in
              that cell. `total` is the source crawl's num_total_topics (or None
              when unrecoverable). When a cell pools several files, the file
              giving the smallest relative position wins.
            - cell_totals[cell_idx] = max num_total_topics across the cell's
              files (used as the discovery-curve x-extent).
        """
        all_topics: List[Tuple[str, int]] = []
        # Per cell: list of (label->id, total) lookups, one per pooled file.
        cell_file_lookups: List[List[Tuple[Dict[str, int], Optional[int]]]] = []
        cell_totals: List[Optional[int]] = []
        for cell_idx, (_name, paths) in enumerate(groups):
            file_lookups: List[Tuple[Dict[str, int], Optional[int]]] = []
            max_total: Optional[int] = None
            for path in paths:
                with open(path, "r") as f:
                    data = json.load(f)
                summaries = data.get("head_refusal_topics_summaries", [])
                if not summaries:
                    # Discovery-mode crawls (do_filter_refusals=false) leave
                    # head_refusal_topics_summaries empty. Fall back to all
                    # candidate cluster heads so the same aggregator works
                    # without a separate backfill step. Raw crawls store the
                    # label in `english`/`shortened` (summary is often unset), so
                    # take the best available label (matches _build_label_id_map).
                    head_topics = (
                        data.get("queue", {})
                        .get("topics", {})
                        .get("head_topics", [])
                    )
                    summaries = [
                        label
                        for t in head_topics
                        if t.get("parent_id") != -5
                        and (
                            label := (
                                t.get("summary")
                                or t.get("english")
                                or t.get("shortened")
                            )
                        )
                    ]
                all_topics.extend((s, cell_idx) for s in summaries)
                label_to_id, total = self._resolve_topic_ids(data)
                file_lookups.append((label_to_id, total))
                if total is not None:
                    max_total = total if max_total is None else max(max_total, total)
            cell_file_lookups.append(file_lookups)
            cell_totals.append(max_total)

        def occ_rel(occ: Tuple[int, Optional[int]]) -> Tuple[float, int]:
            abs_id, total = occ
            # Prefer occurrences with a known total (smaller relative position);
            # break ties by absolute id.
            return (abs_id / total if total else float("inf"), abs_id)

        # Deduplicate preserving order, tracking source cells and first occurrence
        seen: Dict[str, set] = {}
        topic_first: Dict[str, Dict[int, Tuple[int, Optional[int]]]] = {}
        deduped = []
        for t, cell_idx in all_topics:
            t_norm = t.strip().lower()
            if not t_norm:
                continue
            if t_norm not in seen:
                seen[t_norm] = {cell_idx}
                deduped.append(t.strip())
            else:
                seen[t_norm].add(cell_idx)
            for label_to_id, total in cell_file_lookups[cell_idx]:
                tid = label_to_id.get(t_norm)
                if tid is None:
                    continue
                occ = (tid, total)
                per_cell = topic_first.setdefault(t_norm, {})
                cur = per_cell.get(cell_idx)
                if cur is None or occ_rel(occ) < occ_rel(cur):
                    per_cell[cell_idx] = occ
        return deduped, seen, topic_first, cell_totals

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

    # ------------------------------------------------------------------
    # Constrained ("fixed taxonomy") classification mode
    # ------------------------------------------------------------------

    def _build_classification_prompt(
        self, topics: List[str], fixed_topics: List[str]
    ) -> str:
        """Build a prompt that classifies a batch into the fixed taxonomy."""
        fixed_str = "\n".join(f"- {t}" for t in fixed_topics)
        topics_str = "\n".join(f"- {t}" for t in topics)
        return self.exp.classification_prompt.format(
            fixed_topics=fixed_str,
            unmatched_label=self.exp.unmatched_label,
            n_input=len(topics),
            topics=topics_str,
        )

    def _constrain_to_fixed(
        self,
        mapping: Dict[str, List[str]],
        fixed_lookup: Dict[str, str],
        unmatched_label: str,
    ) -> Dict[str, List[str]]:
        """Force every output key onto the fixed taxonomy.

        fixed_lookup maps lowercased fixed label -> canonical fixed label.
        Keys matching a fixed label (case-insensitive) are canonicalized; the
        unmatched label is preserved; any other (hallucinated) key has its
        inputs redirected to the unmatched bucket.
        """
        unmatched_lower = unmatched_label.strip().lower()
        constrained: Dict[str, List[str]] = {}
        for key, vals in mapping.items():
            k_lower = key.strip().lower()
            if k_lower in fixed_lookup:
                target = fixed_lookup[k_lower]
            else:
                target = unmatched_label  # unmatched or hallucinated key
            bucket = constrained.setdefault(target, [])
            seen = set(t.strip().lower() for t in bucket)
            for v in vals:
                if v.strip().lower() not in seen:
                    bucket.append(v)
                    seen.add(v.strip().lower())
        return constrained

    @staticmethod
    def _filter_values_to_batch(
        mapping: Dict[str, List[str]], batch: List[str]
    ) -> Dict[str, List[str]]:
        """Keep only values that are actual batch inputs (drop hallucinations)."""
        batch_lookup = {t.strip().lower(): t.strip() for t in batch}
        out: Dict[str, List[str]] = {}
        for k, vals in mapping.items():
            kept: List[str] = []
            seen: set = set()
            for v in vals:
                vl = v.strip().lower()
                if vl in batch_lookup and vl not in seen:
                    kept.append(batch_lookup[vl])
                    seen.add(vl)
            if kept:
                out[k] = kept
        return out

    def _classify_batch(
        self, model, tokenizer, batch: List[str],
        fixed_topics: List[str], fixed_lookup: Dict[str, str],
    ) -> Tuple[Dict[str, List[str]], str]:
        """Single-prompt classification of a batch into the fixed taxonomy."""
        prompt = self._build_classification_prompt(batch, fixed_topics)
        raw_response = self._generate_single(model, tokenizer, prompt)
        try:
            mapping = _parse_json_from_response(raw_response)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  WARNING: JSON parse failed: {e}")
            mapping = {}
        mapping = self._constrain_to_fixed(
            mapping, fixed_lookup, self.exp.unmatched_label
        )
        mapping = self._filter_values_to_batch(mapping, batch)
        return mapping, raw_response

    def _classify_batches_parallel(
        self, model, tokenizer, batches: List[List[str]], fixed_topics: List[str],
        fixed_lookup: Dict[str, str],
    ) -> List[Tuple[Dict[str, List[str]], str]]:
        """Classify all batches in a single batch_generate call."""
        prompts = [
            self._build_classification_prompt(batch, fixed_topics)
            for batch in batches
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
        for raw_response, batch in zip(generated_texts, batches):
            try:
                mapping = _parse_json_from_response(raw_response)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"  WARNING: JSON parse failed: {e}")
                mapping = {}
            mapping = self._constrain_to_fixed(
                mapping, fixed_lookup, self.exp.unmatched_label
            )
            mapping = self._filter_values_to_batch(mapping, batch)
            results.append((mapping, raw_response))
        return results

    def classify(
        self,
        model,
        tokenizer,
        topics: List[str],
        fixed_topics: List[str],
        topic_sources: Optional[Dict[str, set]] = None,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, set]]:
        """Single-pass classification of topics into a fixed taxonomy (multi-label).

        Each input topic is assigned to one or more of the fixed topics, or to
        the unmatched bucket. Returns the same (final_topics, trajectory,
        source_sets) shape as aggregate() so the standard artifacts apply.
        """
        unmatched_label = self.exp.unmatched_label
        fixed_lookup = {t.strip().lower(): t.strip() for t in fixed_topics}
        current_sources: Dict[str, set] = topic_sources or {}

        self.reduction_log = ReductionLog(topics)
        self.reduction_log.start_iteration(0, topics)

        batches = [
            topics[i : i + self.exp.input_batch_size]
            for i in range(0, len(topics), self.exp.input_batch_size)
        ]
        print(
            f"Classifying {len(topics)} topics into {len(fixed_topics)} fixed "
            f"topics over {len(batches)} batches "
            f"(input_batch_size={self.exp.input_batch_size})"
        )

        all_output_mappings: Dict[str, List[str]] = {}
        all_output_sources: Dict[str, set] = {}
        max_retries = 3

        if self.exp.parallel_batches:
            print(f"  Classifying {len(batches)} batches in parallel...")
            results = self._classify_batches_parallel(
                model, tokenizer, batches, fixed_topics, fixed_lookup
            )
        else:
            results = [
                self._classify_batch(
                    model, tokenizer, batch, fixed_topics, fixed_lookup
                )
                for batch in batches
            ]

        for batch_idx, ((mapping, raw_response), batch) in enumerate(
            zip(results, batches)
        ):
            missing = self._validate_batch_coverage(batch, mapping, batch_idx)
            for retry in range(max_retries):
                if not missing:
                    break
                print(
                    f"  Retrying batch {batch_idx} (attempt {retry + 1}/{max_retries})..."
                )
                mapping, raw_response = self._classify_batch(
                    model, tokenizer, batch, fixed_topics, fixed_lookup
                )
                missing = self._validate_batch_coverage(batch, mapping, batch_idx)
            if missing:
                print(
                    f"  Routing {len(missing)} uncovered topics to "
                    f"'{unmatched_label}' after {max_retries} retries: "
                    f"{missing[:5]}" + ("..." if len(missing) > 5 else "")
                )
                bucket = mapping.setdefault(unmatched_label, [])
                seen = set(t.strip().lower() for t in bucket)
                for t in missing:
                    if t.strip().lower() not in seen:
                        bucket.append(t)
                        seen.add(t.strip().lower())
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

        # Ensure every fixed topic is present as a row, even with zero matches.
        for t in fixed_topics:
            all_output_mappings.setdefault(t.strip(), [])
            all_output_sources.setdefault(t.strip().lower(), set())

        # Final topics keyed original-case; dedup values.
        final_topics = {
            k: sorted(set(v)) for k, v in all_output_mappings.items()
        }
        # Single-level taxonomy: trajectory == direct assignment.
        trajectory = {k: list(v) for k, v in final_topics.items()}

        n_matched = sum(
            1 for t in topics
            if any(
                t.strip().lower() in {x.strip().lower() for x in v}
                for k, v in final_topics.items()
                if k.strip().lower() != unmatched_label.strip().lower()
            )
        )
        n_unmatched = len(final_topics.get(unmatched_label, []))
        print(
            f"Classification complete: {len(final_topics)} fixed topics | "
            f"{n_matched}/{len(topics)} inputs matched >=1 fixed topic | "
            f"{n_unmatched} in '{unmatched_label}'"
        )
        return final_topics, trajectory, all_output_sources

    def save_cell_matrix(
        self,
        output_dir: str,
        final_topics: Dict[str, List[str]],
        topic_sources: Dict[str, set],
        cell_names: List[str],
        topic_first: Optional[Dict[str, Dict[int, Tuple[int, Optional[int]]]]] = None,
    ):
        """Write a per-topic x per-cell contribution matrix (counts + presence).

        For each fixed topic, counts how many unique input topics assigned to it
        came from each input cell, and which cells contributed at all. Because a
        single input topic can appear in several cells and be assigned to several
        fixed topics, per-cell counts sum to more than the number of inputs.

        Also reports, per cell, the first-occurrence discovery index of the
        cluster: among the cluster's inputs seen in that cell, the earliest
        (smallest relative position) occurrence's raw topic id (`first_abs`) and
        its position relative to the crawl's total topics (`first_rel`). Blank
        when no id is available.
        """
        num_runs = len(cell_names)
        topic_first = topic_first or {}

        rows = []
        for topic in sorted(final_topics.keys()):
            inputs = final_topics[topic]
            counts = [0] * num_runs
            # Per cell, track the earliest (min relative) occurrence as
            # (rel_key, abs_id, rel) so first_abs and first_rel stay consistent.
            best: List[Optional[Tuple[float, int, Optional[float]]]] = [None] * num_runs
            for inp in inputs:
                key = inp.strip().lower()
                for c in topic_sources.get(key, set()):
                    if 0 <= c < num_runs:
                        counts[c] += 1
                for c, (abs_id, total) in topic_first.get(key, {}).items():
                    if not (0 <= c < num_runs):
                        continue
                    rel = (abs_id / total) if total else None
                    rel_key = rel if rel is not None else float("inf")
                    if best[c] is None or rel_key < best[c][0]:
                        best[c] = (rel_key, abs_id, rel)
            first_abs: List[Optional[int]] = [
                None if best[c] is None else best[c][1] for c in range(num_runs)
            ]
            first_rel: List[Optional[float]] = [
                None if best[c] is None else best[c][2] for c in range(num_runs)
            ]
            presence = [1 if c > 0 else 0 for c in counts]
            rows.append((topic, len(inputs), counts, presence, first_abs, first_rel))

        def _abs_str(v: Optional[int]) -> str:
            return "" if v is None else str(v)

        def _rel_str(v: Optional[float]) -> str:
            return "" if v is None else f"{v:.4f}"

        # CSV
        csv_path = os.path.join(output_dir, "topic_cell_matrix.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = (
                ["fixed_topic", "n_unique_inputs"]
                + [f"count_{c}" for c in cell_names]
                + [f"present_{c}" for c in cell_names]
                + ["n_cells_present"]
                + [f"first_abs_{c}" for c in cell_names]
                + [f"first_rel_{c}" for c in cell_names]
            )
            writer.writerow(header)
            for topic, n_inputs, counts, presence, fabs, frel in rows:
                writer.writerow(
                    [topic, n_inputs] + counts + presence + [sum(presence)]
                    + [_abs_str(v) for v in fabs]
                    + [_rel_str(v) for v in frel]
                )

        # Markdown (human-readable)
        md_path = os.path.join(output_dir, "topic_cell_matrix.md")
        with open(md_path, "w") as f:
            f.write("# Fixed topic x cell contribution matrix\n\n")
            f.write(
                "Counts = number of unique input topics assigned to the fixed "
                "topic that appeared in each cell. A topic present in multiple "
                "cells is counted once per cell, so row counts can exceed "
                "`n_unique_inputs`.\n\n"
            )
            f.write(
                "`first_abs_<cell>` = smallest raw discovery id (the topic id in "
                "the source crawl) among the cluster's inputs seen in that cell, "
                "i.e. how early the cluster first appeared. `first_rel_<cell>` = "
                "that id divided by the cell's total topics discovered "
                "(`num_total_topics`). Blank when no id could be recovered.\n\n"
            )
            f.write("Cells:\n")
            for i, c in enumerate(cell_names):
                f.write(f"- {i}: `{c}`\n")
            f.write("\n")
            first_cols = []
            for c in cell_names:
                first_cols.append(f"first_abs_{c}")
                first_cols.append(f"first_rel_{c}")
            f.write(
                "| Fixed topic | inputs | "
                + " | ".join(cell_names)
                + " | cells | "
                + " | ".join(first_cols)
                + " |\n"
            )
            f.write(
                "|---|---|" + "---|" * num_runs + "---|" + "---|" * (2 * num_runs) + "\n"
            )
            for topic, n_inputs, counts, presence, fabs, frel in rows:
                cells_str = " | ".join(str(c) for c in counts)
                first_str = " | ".join(
                    s
                    for c in range(num_runs)
                    for s in (_abs_str(fabs[c]), _rel_str(frel[c]))
                )
                f.write(
                    f"| {topic} | {n_inputs} | {cells_str} | {sum(presence)} | "
                    f"{first_str} |\n"
                )

        print(f"  topic_cell_matrix.csv, topic_cell_matrix.md")

    def save_cluster_discovery_plot(
        self,
        output_dir: str,
        final_topics: Dict[str, List[str]],
        topic_first: Dict[str, Dict[int, Tuple[int, Optional[int]]]],
        cell_names: List[str],
        cell_totals: List[Optional[int]],
    ):
        """Plot, per cell, a cumulative step curve of clusters discovered vs topic id.

        For each crawl cell, every cluster (fixed topic) is "discovered" at the
        earliest topic id among its inputs seen in that cell (its
        first-occurrence id). Sorting those ids and stepping up by one at each
        gives the cumulative number of distinct clusters found as the crawl
        progresses through topic ids. One step line per cell.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_runs = len(cell_names)

        # Per cell: sorted first-occurrence ids, one per cluster present there.
        per_cell_ids: List[List[int]] = [[] for _ in range(num_runs)]
        for inputs in final_topics.values():
            best: List[Optional[Tuple[float, int]]] = [None] * num_runs
            for inp in inputs:
                for c, (abs_id, total) in topic_first.get(inp.strip().lower(), {}).items():
                    if not (0 <= c < num_runs):
                        continue
                    rel_key = (abs_id / total) if total else float("inf")
                    if best[c] is None or rel_key < best[c][0]:
                        best[c] = (rel_key, abs_id)
            for c in range(num_runs):
                if best[c] is not None:
                    per_cell_ids[c].append(best[c][1])

        plt.figure(figsize=(9, 6))
        cmap = plt.get_cmap("tab10")
        plotted = False
        for c in range(num_runs):
            ids = sorted(per_cell_ids[c])
            if not ids:
                continue
            n = len(ids)
            # where="post": y held from x[i] to x[i+1]. Start at (0,0), step up
            # by one at each discovery, hold the total out to num_total_topics.
            x = [0] + ids
            y = [0] + list(range(1, n + 1))
            total = cell_totals[c] if c < len(cell_totals) else None
            if total and total > ids[-1]:
                x.append(total)
                y.append(n)
            plt.step(
                x, y, where="post", label=f"{cell_names[c]} ({n})",
                color=cmap(c % 10), alpha=0.85,
            )
            plotted = True

        if not plotted:
            print("  (no discovery ids available; skipping cluster_discovery_curve.png)")
            plt.close()
            return

        plt.xlabel("Topic ID (discovery order within crawl)")
        plt.ylabel("Cumulative distinct clusters discovered")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.12),
            ncol=min(num_runs, 4), frameon=False,
        )
        plt.tight_layout()
        out_path = os.path.join(output_dir, "cluster_discovery_curve.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"  cluster_discovery_curve.png")

    def save_artifacts(
        self,
        output_dir: str,
        final_topics: Dict[str, List[str]],
        trajectory: Dict[str, List[str]],
        input_paths: List[str],
        source_sets: Optional[Dict[str, set]] = None,
        num_runs: Optional[int] = None,
    ):
        """Save all artifacts to output_dir.

        `num_runs` is the number of cells (defaults to len(input_paths) for the
        one-file-per-cell case; pass explicitly when cells pool several files).
        """
        from src.aggregation.html_builder import build_explorer_html

        os.makedirs(output_dir, exist_ok=True)
        if num_runs is None:
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


def resolve_input_groups(
    input_paths: List[str],
    input_groups: Optional[Dict[str, List[str]]] = None,
) -> List[Tuple[str, List[str]]]:
    """Normalize the input spec into ordered (cell_name, [paths]) groups.

    When `input_groups` is set, each key is a cell pooling its listed files.
    Otherwise every path in `input_paths` is its own cell, named by its
    filename stem (preserving the previous one-file-per-cell behavior).
    """
    if input_groups:
        return [(name, list(paths)) for name, paths in input_groups.items()]
    return [
        (os.path.splitext(os.path.basename(p))[0], [p]) for p in input_paths
    ]


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
