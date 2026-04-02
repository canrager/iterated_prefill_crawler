import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.aggregation.semantic_judge import judge_semantic_containment


@dataclass
class CoverageResult:
    """Coverage analysis results grouped by ground truth category."""

    category_results: Dict[str, List[Tuple[str, Optional[str]]]] = field(
        default_factory=dict
    )
    # {category: [(subtopic, matched_crawl_topic_or_None), ...]}

    @property
    def overall_covered(self) -> int:
        return sum(
            1
            for matches in self.category_results.values()
            for _, matched in matches
            if matched is not None
        )

    @property
    def overall_total(self) -> int:
        return sum(len(matches) for matches in self.category_results.values())

    @property
    def overall_rate(self) -> float:
        total = self.overall_total
        return self.overall_covered / total if total > 0 else 0.0

    def per_category_summary(self) -> Dict[str, Dict]:
        summary = {}
        for category, matches in self.category_results.items():
            covered = [sub for sub, m in matches if m is not None]
            missing = [sub for sub, m in matches if m is None]
            total = len(matches)
            summary[category] = {
                "covered": len(covered),
                "total": total,
                "rate": len(covered) / total if total > 0 else 0.0,
                "missing": missing,
                "matches": {sub: m for sub, m in matches if m is not None},
            }
        return summary

    def to_dict(self) -> dict:
        return {
            "overall": {
                "covered": self.overall_covered,
                "total": self.overall_total,
                "rate": self.overall_rate,
            },
            "per_category": self.per_category_summary(),
            "raw_results": {
                category: [(sub, m) for sub, m in matches]
                for category, matches in self.category_results.items()
            },
        }

    def print_report(self):
        print(f"\n{'='*60}")
        print(f"Ground Truth Coverage Report")
        print(f"{'='*60}")
        for category, info in self.per_category_summary().items():
            status = (
                "FULL" if info["rate"] == 1.0 else f"{info['rate']:.0%}"
            )
            print(f"\n  {category}: {info['covered']}/{info['total']} ({status})")
            if info["missing"]:
                for m in info["missing"]:
                    print(f"    MISS: {m}")
            if info["matches"]:
                for sub, matched in info["matches"].items():
                    print(f"    HIT:  {sub} -> {matched}")
        print(f"\n{'='*60}")
        print(
            f"Overall: {self.overall_covered}/{self.overall_total} "
            f"({self.overall_rate:.1%})"
        )
        print(f"{'='*60}\n")


def load_ground_truth(path: str) -> Dict[str, List[str]]:
    with open(path, "r") as f:
        return json.load(f)


def load_crawl_topics(input_paths: List[str]) -> List[str]:
    """Load and deduplicate topic summaries from crawler output JSONs."""
    all_topics = []
    for path in input_paths:
        with open(path, "r") as f:
            data = json.load(f)
        summaries = data.get("head_refusal_topics_summaries", [])
        all_topics.extend(summaries)
    seen = set()
    deduped = []
    for t in all_topics:
        t_norm = t.strip().lower()
        if t_norm and t_norm not in seen:
            seen.add(t_norm)
            deduped.append(t.strip())
    return deduped


def analyze_coverage(
    ground_truth: Dict[str, List[str]],
    crawl_topics: List[str],
    model,
    tokenizer,
    max_tokens: int = 256,
    reference_chunk_size: int = 50,
    default_provider: str = "openrouter",
    provider_url_overrides: Optional[Dict[str, str]] = None,
    verbose: bool = False,
) -> CoverageResult:
    """Check coverage of ground truth subtopics by crawl topics.

    Chunks crawl_topics into groups of reference_chunk_size and runs the
    semantic judge on each chunk, taking the union of matches.
    """
    # Flatten GT subtopics with category tracking
    all_subtopics = []
    subtopic_to_category: Dict[str, str] = {}
    for category, subtopics in ground_truth.items():
        for sub in subtopics:
            all_subtopics.append(sub)
            subtopic_to_category[sub] = category

    if not all_subtopics:
        return CoverageResult()

    if not crawl_topics:
        result = CoverageResult()
        for category, subtopics in ground_truth.items():
            result.category_results[category] = [(sub, None) for sub in subtopics]
        return result

    # Chunk crawl topics
    chunks = [
        crawl_topics[i : i + reference_chunk_size]
        for i in range(0, len(crawl_topics), reference_chunk_size)
    ]

    if verbose:
        print(
            f"Checking {len(all_subtopics)} GT subtopics against "
            f"{len(crawl_topics)} crawl topics in {len(chunks)} chunks"
        )

    # Track best match per subtopic across all chunks
    best_match: Dict[str, Optional[str]] = {sub: None for sub in all_subtopics}

    for chunk_idx, chunk in enumerate(chunks):
        # Only check subtopics that haven't been matched yet
        unmatched = [sub for sub in all_subtopics if best_match[sub] is None]
        if not unmatched:
            break

        if verbose:
            print(
                f"  Chunk {chunk_idx + 1}/{len(chunks)}: "
                f"{len(unmatched)} unmatched subtopics vs {len(chunk)} crawl topics"
            )

        results = judge_semantic_containment(
            input_terms=unmatched,
            reference_terms=chunk,
            model=model,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            default_provider=default_provider,
            provider_url_overrides=provider_url_overrides,
            verbose=verbose,
        )

        for is_match, (subtopic, matched_term) in results:
            if is_match and best_match[subtopic] is None:
                best_match[subtopic] = matched_term

    # Group results by category
    result = CoverageResult()
    for category, subtopics in ground_truth.items():
        result.category_results[category] = [
            (sub, best_match.get(sub)) for sub in subtopics
        ]

    return result
