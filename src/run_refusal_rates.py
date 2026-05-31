"""Post-hoc refusal-rate computation for aggregated cluster heads.

Reads an aggregation output directory (final_topics.txt + reduction_log.json),
drives ``check_refusal`` once per (cluster head, language) Topic against the
configured target model, and writes a reviewer-table-shaped artifact bundle:

Alternatively, pass ``+specificity_level=L5`` (or several levels at once as
``+specificity_level=[L4,L5]``; optionally ``+specificity_csv=<path>``,
defaulting to ``<aggregation_dir>/specificity_scores.csv``) to probe only the
topics scored at those specificity levels by a specificity-scoring aggregation
run. In that mode cluster heads and per-cell discovery booleans come from the
csv's ``topic`` / ``present_<cell>`` columns instead of final_topics.txt +
reduction_log.json, and each row carries its ``specificity_level``.


    refusal_rates.json   schema parity with reviewer_refusal_probe.json
    refusal_rates.md     markdown table sorted by refusal rate desc
    config.json          resolved Hydra config

Discovery booleans per cell are read from
``reduction_log.json["consistency"]["source_sets"]`` (a dict of normalized
cluster heads -> list of run indices), with the run index -> cell mapping
parsed from each ``input_paths`` filename
(``{cell}_candidate_topics_for_aggregation.json``).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# vLLM/CUDA require spawn before any torch import
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.crawler.config import CrawlerConfig
from src.crawler.topic_queue import Topic
from src.directory_config import CONFIG_DIR, ROOT_DIR, resolve_cache_dir
from src.llm_utils import load_model_and_tokenizer
from src.provider_config import collect_required_api_keys
from src.refusal_utils import check_refusal
from src.transcript_logger import init_transcript_log


CELLS: Tuple[str, ...] = ("direct", "prefill_only", "iter_no_prefill", "ipc")
LANGUAGES: Tuple[str, ...] = ("english", "chinese")
_CANDIDATE_SUFFIX_RE = re.compile(
    r"_candidate_topics_for_aggregation(?:\.json)?$"
)


def _read_final_topics(path: Path) -> List[str]:
    lines = path.read_text().splitlines()
    return [line.strip() for line in lines if line.strip()]


def _normalize_levels(value: object) -> List[str]:
    """Coerce a Hydra ``specificity_level`` override into a list of levels.

    Accepts a list/ListConfig (``[L4, L5]``) or a comma/space-separated string
    (``"L4,L5"``, ``"L4 L5"``, or a single ``"L5"``). Returns ``[]`` when unset.
    """
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,\s]+", value.strip())
    else:
        parts = [str(v) for v in value]
    return [p for p in (s.strip() for s in parts) if p]


def _read_specificity_csv(
    path: Path, levels: Sequence[str]
) -> Tuple[List[str], Dict[str, Dict[str, bool]], List[str], Dict[str, str]]:
    """Select topics at one or more specificity levels from a scores csv.

    Returns ``(topics, head_lower -> {cell: bool}, cell_names,
    head_lower -> level)``. The csv has columns ``topic, level,
    present_<cell>...``; discovery booleans come straight from the
    ``present_<cell>`` columns, so no reduction_log is needed in this mode
    (this is the analogue of ``_read_final_topics`` + ``_read_discovery`` for a
    specificity-scoring aggregation run). With several levels the selected
    topics are the union across them, in csv row order.
    """
    wanted = {lv.strip() for lv in levels}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        present_cols = [c for c in fieldnames if c.startswith("present_")]
        cell_names = [c[len("present_") :] for c in present_cols]
        topics: List[str] = []
        discovery: Dict[str, Dict[str, bool]] = {}
        head_levels: Dict[str, str] = {}
        for row in reader:
            level = (row.get("level") or "").strip()
            if level not in wanted:
                continue
            topic = (row.get("topic") or "").strip()
            if not topic:
                continue
            topics.append(topic)
            discovery[topic.lower()] = {
                cell: ((row.get(col) or "").strip() not in ("", "0"))
                for cell, col in zip(cell_names, present_cols)
            }
            head_levels[topic.lower()] = level
    return topics, discovery, cell_names, head_levels


def _cell_from_input_path(path: str) -> Optional[str]:
    """Return the cell name parsed from an aggregation input filename.

    Recognizes the ``{cell}_candidate_topics_for_aggregation.json`` pattern.
    Returns None for paths that don't match (caller falls back to cell_<i>).
    """
    stem = Path(path).name
    match = _CANDIDATE_SUFFIX_RE.search(stem)
    if not match:
        return None
    candidate = stem[: match.start()]
    return candidate or None


def _read_discovery(
    reduction_log_path: Path,
) -> Tuple[Dict[str, Dict[str, bool]], List[str]]:
    """Return (head_lower -> {cell: bool}, cell_names_in_order)."""
    with open(reduction_log_path, "r") as f:
        log = json.load(f)
    input_paths = log.get("input_paths") or []
    cell_names: List[str] = []
    for i, p in enumerate(input_paths):
        name = _cell_from_input_path(p) or f"cell_{i}"
        cell_names.append(name)

    source_sets = (
        log.get("consistency", {}).get("source_sets", {}) or {}
    )
    discovery: Dict[str, Dict[str, bool]] = {}
    for head_lower, run_indices in source_sets.items():
        idxs = set(int(i) for i in run_indices)
        discovery[head_lower] = {
            name: (i in idxs) for i, name in enumerate(cell_names)
        }
    return discovery, cell_names


def _probes_for_topic(topic: Topic) -> List[Dict[str, object]]:
    queries = topic.refusal_check_queries or []
    responses = topic.refusal_check_responses or []
    flags = topic.refusal_check_refused or []
    n = max(len(queries), len(responses), len(flags))
    probes = []
    for i in range(n):
        probes.append(
            {
                "query": queries[i] if i < len(queries) else None,
                "response": responses[i] if i < len(responses) else None,
                "is_refusal": flags[i] if i < len(flags) else None,
            }
        )
    return probes


def _topic_record(
    cluster: str,
    language: str,
    topic: Topic,
) -> Dict[str, object]:
    flags = topic.refusal_check_refused or []
    valid_flags = [bool(v) for v in flags if v is not None]
    total_refusals = sum(1 for v in valid_flags if v)
    total_probes = len(valid_flags)
    return {
        "cluster": cluster,
        "language": language,
        "raw": topic.raw,
        "is_refusal": bool(topic.is_refusal) if topic.is_refusal is not None else False,
        "refusal_rate": topic.refusal_rate,
        "total_refusals": total_refusals,
        "total_probes": total_probes,
        "refusal_flags": list(flags),
        "probes": _probes_for_topic(topic),
    }


def _cluster_record(
    cluster: str,
    discovery: Dict[str, bool],
    language_records: List[Dict[str, object]],
    level: Optional[str] = None,
) -> Dict[str, object]:
    total_refusals = sum(r["total_refusals"] for r in language_records)
    total_probes = sum(r["total_probes"] for r in language_records)
    rate = total_refusals / total_probes if total_probes else None
    record: Dict[str, object] = {
        "cluster": cluster,
        "discovery": discovery,
        "refusal_rate": rate,
        "total_refusals": total_refusals,
        "total_probes": total_probes,
        "language_records": language_records,
    }
    # Only present in specificity-level mode; preserves schema parity with
    # reviewer_refusal_probe.json for the default final_topics path.
    if level is not None:
        record["specificity_level"] = level
    return record


def _write_markdown(
    out_path: Path,
    cluster_records: List[Dict[str, object]],
    cell_names: List[str],
) -> None:
    # Sort by refusal rate desc; None rates sink to the bottom.
    def _sort_key(rec: Dict[str, object]):
        rate = rec["refusal_rate"]
        return (-(rate if rate is not None else -1.0), rec["cluster"].lower())

    sorted_records = sorted(cluster_records, key=_sort_key)
    # Show a Level column only in specificity-level mode (where it's populated).
    show_level = any(rec.get("specificity_level") for rec in cluster_records)
    level_h = "Level | " if show_level else ""
    level_sep = "---|" if show_level else ""
    cell_headers = " | ".join(name.replace("_", " ") for name in cell_names)
    header = (
        f"| Cluster | {level_h}{cell_headers} | Refusal rate | Refusals / probes |\n"
        f"|---|{level_sep}{'|'.join(['---'] * len(cell_names))}|---|---|\n"
    )
    rows = []
    for rec in sorted_records:
        rate = rec["refusal_rate"]
        rate_str = "—" if rate is None else f"{rate:.2f}"
        cell_cells = " | ".join(
            ("✓" if rec["discovery"].get(name) else "")
            for name in cell_names
        )
        level_cell = f"{rec.get('specificity_level', '')} | " if show_level else ""
        rows.append(
            f"| {rec['cluster']} | {level_cell}{cell_cells} | {rate_str} | "
            f"{rec['total_refusals']} / {rec['total_probes']} |"
        )
    out_path.write_text(header + "\n".join(rows) + "\n")


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    crawler_config = CrawlerConfig(**OmegaConf.to_container(cfg, resolve=True))

    aggregation_dir = cfg.get("aggregation_dir")
    if not aggregation_dir:
        raise ValueError(
            "Required override missing: +aggregation_dir=artifacts/aggregation/<ts>/"
        )
    aggregation_dir = Path(aggregation_dir)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    default_out_dir = ROOT_DIR / "artifacts" / "refusal_rates" / timestamp
    out_dir = Path(cfg.get("out_dir") or default_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Early API key check for non-local helper roles
    non_local_model_strings = [
        getattr(crawler_config.model, f"{role}_model")
        for role in ("target", "translation", "summarization", "refusal_check")
        if getattr(crawler_config.model, f"{role}_model") != "local"
    ]
    if crawler_config.model.universal_backup_model:
        non_local_model_strings.append(crawler_config.model.universal_backup_model)
    missing_keys = collect_required_api_keys(
        non_local_model_strings,
        default_provider=crawler_config.model.default_provider,
    )
    if missing_keys:
        details = ", ".join(f"{p} ({v})" for p, v in missing_keys.items())
        raise ValueError(
            f"Missing API key(s) for providers used by non-local roles: {details}"
        )

    transcript_path = init_transcript_log(
        f"refusal_rates_{timestamp}", output_dir=str(out_dir)
    )
    print(f"Transcript log: {transcript_path}")

    # Load aggregator outputs — either a specificity-level subset (read from
    # specificity_scores.csv) or the full final_topics.txt cluster-head set.
    specificity_levels = _normalize_levels(cfg.get("specificity_level"))
    head_levels: Dict[str, str] = {}
    if specificity_levels:
        csv_path = Path(
            cfg.get("specificity_csv") or aggregation_dir / "specificity_scores.csv"
        )
        cluster_heads, discovery, cell_names, head_levels = _read_specificity_csv(
            csv_path, specificity_levels
        )
        print(
            f"Loaded {len(cluster_heads)} topics at {specificity_levels} "
            f"from {csv_path}; cells: {cell_names}"
        )
    else:
        cluster_heads = _read_final_topics(aggregation_dir / "final_topics.txt")
        discovery, cell_names = _read_discovery(aggregation_dir / "reduction_log.json")
        print(
            f"Loaded {len(cluster_heads)} cluster heads "
            f"from {aggregation_dir}; cells: {cell_names}"
        )

    # Load local model (if configured) — required for local_ds70b target.
    if cfg.model.local_model is not None:
        cache_dir_path = resolve_cache_dir(cfg.model.cache_dir)
        local_model, local_tokenizer = load_model_and_tokenizer(
            cfg.model.local_model,
            device=cfg.model.device,
            cache_dir=str(cache_dir_path),
            quantization_bits=cfg.model.quantization_bits,
            vllm_tensor_parallel_size=cfg.model.vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=cfg.model.vllm_gpu_memory_utilization,
            vllm_max_model_len=crawler_config.model.vllm_max_model_len,
        )
    else:
        local_model, local_tokenizer = None, None

    # Build two Topics per cluster head (english + chinese) and probe in one call.
    topics: List[Topic] = []
    for head in cluster_heads:
        topics.append(Topic(raw=head, is_chinese=False, is_head=True))
        topics.append(Topic(raw=head, is_chinese=True, is_head=True))

    print(f"Probing {len(topics)} topic-language pairs (target = {cfg.model.target_model})")
    check_refusal(crawler_config, local_model, local_tokenizer, topics)

    # Build per-language and per-cluster records
    per_topic: List[Dict[str, object]] = []
    per_cluster: List[Dict[str, object]] = []
    for ci, head in enumerate(cluster_heads):
        head_topics = topics[ci * 2 : ci * 2 + 2]
        head_discovery = {
            name: discovery.get(head.strip().lower(), {}).get(name, False)
            for name in cell_names
        }
        language_records = []
        for lang, topic in zip(LANGUAGES, head_topics):
            rec = _topic_record(head, lang, topic)
            language_records.append(rec)
            per_topic.append(rec)
        per_cluster.append(
            _cluster_record(
                head,
                head_discovery,
                language_records,
                head_levels.get(head.strip().lower()),
            )
        )

    metadata = {
        "model_config": cfg.model.target_model,
        "discovery_dir": str(aggregation_dir),
        "dry_run": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    bundle = {
        "metadata": metadata,
        "per_cluster": per_cluster,
        "per_topic": per_topic,
    }

    (out_dir / "refusal_rates.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=False))
    _write_markdown(out_dir / "refusal_rates.md", per_cluster, cell_names)
    (out_dir / "config.json").write_text(
        json.dumps(crawler_config.to_dict(), indent=2, default=str)
    )
    print(f"Wrote refusal-rate artifacts to {out_dir}/")

    # Cleanup vLLM
    if local_model is not None:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
        del local_model
        import gc

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
