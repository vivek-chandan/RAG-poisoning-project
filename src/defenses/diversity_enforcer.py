"""Diversity enforcement for Secure RAG retrieval results.

This module reduces source monoculture by enforcing per-source caps and minimum
source diversity in selected context chunks.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DiversityConfig:
    """Configuration for diversity constraints."""

    max_documents_per_source: int = 2
    min_unique_sources: int = 2
    top_k: int = 4
    dominance_threshold: float = 0.6
    source_field: str = "source_id"
    fallback_source: str = "unknown"


@dataclass(frozen=True)
class DiversityItem:
    """Candidate item for diversity-aware selection."""

    item_id: str
    document: str
    score: float
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class DiversityReport:
    """Diversity diagnostics for a candidate set or selected set."""

    total_items: int
    unique_sources: int
    source_distribution: Dict[str, int]
    dominance_ratio: float
    dominant_source: str
    source_dominance_detected: bool
    entropy: float
    normalized_entropy: float


@dataclass(frozen=True)
class DiversitySelectionResult:
    """Structured output from diversity enforcement."""

    selected: List[DiversityItem]
    rejected: List[DiversityItem]
    report_before: DiversityReport
    report_after: DiversityReport


class DiversityEnforcer:
    """Apply source diversity constraints to retrieval candidates."""

    def __init__(self, config: Optional[DiversityConfig] = None) -> None:
        self._config = config or DiversityConfig()

    def _extract_source(self, metadata: Mapping[str, Any]) -> str:
        source = str(metadata.get(self._config.source_field, "")).strip()
        return source if source else self._config.fallback_source

    def _entropy(self, counts: Mapping[str, int]) -> float:
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        entropy = 0.0
        for value in counts.values():
            p = value / total
            if p > 0:
                entropy -= p * math.log(p, 2)
        return entropy

    def compute_diversity_score(self, items: Sequence[DiversityItem]) -> DiversityReport:
        """Compute entropy-based diversity diagnostics for provided items."""
        if not items:
            return DiversityReport(
                total_items=0,
                unique_sources=0,
                source_distribution={},
                dominance_ratio=0.0,
                dominant_source="",
                source_dominance_detected=False,
                entropy=0.0,
                normalized_entropy=0.0,
            )

        counts = Counter(self._extract_source(item.metadata) for item in items)
        total = len(items)
        dominant_source, dominant_count = counts.most_common(1)[0]
        dominance_ratio = dominant_count / total
        entropy = self._entropy(counts)
        max_entropy = math.log(len(counts), 2) if len(counts) > 1 else 0.0
        normalized = (entropy / max_entropy) if max_entropy > 0 else 0.0

        return DiversityReport(
            total_items=total,
            unique_sources=len(counts),
            source_distribution=dict(counts),
            dominance_ratio=round(dominance_ratio, 6),
            dominant_source=dominant_source,
            source_dominance_detected=dominance_ratio >= self._config.dominance_threshold,
            entropy=round(entropy, 6),
            normalized_entropy=round(normalized, 6),
        )

    def detect_source_dominance(self, items: Sequence[DiversityItem]) -> bool:
        """Return True when any single source dominates above threshold."""
        return self.compute_diversity_score(items).source_dominance_detected

    def select_top_k(self, items: Sequence[DiversityItem]) -> DiversitySelectionResult:
        """Select top-k items with diversity constraints.

        Strategy:
        1) sort by score descending
        2) first pass: enforce max per source and prioritize new sources until
           minimum unique sources is met
        3) second pass: fill remaining slots while respecting per-source cap
        """
        ranked = sorted(items, key=lambda item: item.score, reverse=True)
        before = self.compute_diversity_score(ranked)

        selected: List[DiversityItem] = []
        selected_by_source: Dict[str, int] = defaultdict(int)
        selected_sources: set[str] = set()

        # Pass 1: maximize source coverage first while keeping ranking order.
        for candidate in ranked:
            if len(selected) >= self._config.top_k:
                break

            source = self._extract_source(candidate.metadata)
            if selected_by_source[source] >= self._config.max_documents_per_source:
                continue

            if len(selected_sources) < self._config.min_unique_sources and source in selected_sources:
                continue

            selected.append(candidate)
            selected_by_source[source] += 1
            selected_sources.add(source)

        # Pass 2: fill remaining slots using best ranked candidates under source cap.
        if len(selected) < self._config.top_k:
            selected_ids = {item.item_id for item in selected}
            for candidate in ranked:
                if len(selected) >= self._config.top_k:
                    break
                if candidate.item_id in selected_ids:
                    continue

                source = self._extract_source(candidate.metadata)
                if selected_by_source[source] >= self._config.max_documents_per_source:
                    continue

                selected.append(candidate)
                selected_by_source[source] += 1
                selected_sources.add(source)
                selected_ids.add(candidate.item_id)

        selected_ids_final = {item.item_id for item in selected}
        rejected = [item for item in ranked if item.item_id not in selected_ids_final]
        after = self.compute_diversity_score(selected)

        return DiversitySelectionResult(
            selected=selected,
            rejected=rejected,
            report_before=before,
            report_after=after,
        )


def example_usage() -> None:
    """Executable examples for quick validation."""
    config = DiversityConfig(
        max_documents_per_source=1,
        min_unique_sources=3,
        top_k=4,
        dominance_threshold=0.5,
        source_field="source_id",
    )
    enforcer = DiversityEnforcer(config=config)

    candidates = [
        DiversityItem("d1", "policy chunk A", 0.98, {"source_id": "s_hr"}),
        DiversityItem("d2", "policy chunk B", 0.95, {"source_id": "s_hr"}),
        DiversityItem("d3", "security chunk", 0.91, {"source_id": "s_sec"}),
        DiversityItem("d4", "wiki chunk", 0.88, {"source_id": "s_wiki"}),
        DiversityItem("d5", "blog chunk", 0.87, {"source_id": "s_blog"}),
    ]

    result = enforcer.select_top_k(candidates)
    print("Selected IDs:", [item.item_id for item in result.selected])
    print("Rejected IDs:", [item.item_id for item in result.rejected])
    print("Before diversity:", result.report_before)
    print("After diversity:", result.report_after)


if __name__ == "__main__":
    example_usage()
