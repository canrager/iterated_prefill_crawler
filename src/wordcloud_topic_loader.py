"""Load cluster-crawler wordcloud_topics into Candidate records.

This module is the thin reader between cluster-crawler artifacts and the
aggregator. It exposes only the structural fields the aggregator needs:
the label, its cluster id/size, and the parent_id/yield of the broad topic
that drilled it.

The historical structural-proxy ranker (proxy_preference_score, Elo over
that proxy) lived next to this code on the bench branch. It was the wrong
hypothesis — see ``.trio/LEARNINGS.md`` ranking-signal section. Production
ranking is target self-rank Elo; see ``scripts/self_rank_families.py``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    label: str
    index: int
    parent_id: int | None
    cluster_id: int | None
    cluster_score: float
    cluster_size: int
    parent_yield: int


def load_candidates(path: Path) -> list[Candidate]:
    data = json.loads(path.read_text(encoding="utf-8"))
    topic_rows = data.get("wordcloud_topics") or []
    if not topic_rows:
        raise ValueError("Artifact has no wordcloud_topics; rerun crawler with current serializer")

    cluster_by_member: dict[int, dict] = {}
    for cluster in data.get("clusters", []):
        for idx in cluster.get("member_indices", []):
            cluster_by_member[idx] = cluster

    parent_counts: dict[int, int] = {}
    for row in topic_rows:
        parent_id = row.get("parent_id")
        if isinstance(parent_id, int) and parent_id >= 0:
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1

    candidates: list[Candidate] = []
    seen: set[str] = set()
    for row in topic_rows:
        label = " ".join((row.get("label") or "").split())
        key = label.casefold()
        if not label or key in seen:
            continue
        seen.add(key)
        idx = row.get("index")
        if not isinstance(idx, int):
            continue
        cluster = cluster_by_member.get(idx, {})
        parent_id = row.get("parent_id")
        candidates.append(
            Candidate(
                label=label,
                index=idx,
                parent_id=parent_id if isinstance(parent_id, int) else None,
                cluster_id=cluster.get("id"),
                cluster_score=float(cluster.get("score", 0.0) or 0.0),
                cluster_size=int(cluster.get("size", 1) or 1),
                parent_yield=parent_counts.get(parent_id, 1)
                if isinstance(parent_id, int) and parent_id >= 0
                else 1,
            )
        )
    return candidates


def order_by_parent_yield(candidates: list[Candidate]) -> list[tuple[Candidate, float]]:
    """Order candidates by parent_yield descending, ties broken by label.

    This is the lightweight ordering the aggregator uses to decide which
    labels go into early batches. It replaces the old structural-proxy Elo
    pre-ranking which is no longer the production ranking signal.
    """
    return sorted(
        ((c, float(c.parent_yield)) for c in candidates),
        key=lambda item: (-item[1], item[0].label.casefold()),
    )
