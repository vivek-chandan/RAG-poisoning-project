"""Defense package.

Includes injection detection, provenance filtering, and compatibility exports.
"""

from defenses.base import Defense
from defenses.chunk_sanitizer import ChunkSanitizer, DetectionResult, InjectionMatch, PatternDefinition, SanitizeMode, SanitizationResult
from defenses.context_firewall import ContextFirewall, ContextFirewallConfig, FirewallResult
from defenses.diversity_enforcer import DiversityConfig, DiversityEnforcer, DiversityItem, DiversityReport, DiversitySelectionResult
from defenses.injection import InjectionPatternDefense
from defenses.provenance import ProvenanceDefense
from defenses.provenance_tracker import ProvenanceTracker, TrustLevel
from defenses.security_reranker import PenaltyWeights, RerankedResult, SecurityReranker, SecurityRerankerConfig


class DefenseConfig:
    """Compatibility config for legacy callers."""

    def __init__(
        self,
        allow_poisoned_sources: bool = True,
        strip_instruction_like_text: bool = False,
        trusted_categories: tuple[str, ...] = (),
        max_context_chunks: int = 4,
    ) -> None:
        self.allow_poisoned_sources = allow_poisoned_sources
        self.strip_instruction_like_text = strip_instruction_like_text
        self.trusted_categories = trusted_categories
        self.max_context_chunks = max_context_chunks


def filter_results(result: dict, defense: DefenseConfig) -> dict:
    """Compatibility function mirroring prior public API."""
    chain: list[Defense] = [
        ProvenanceDefense(
            allow_poisoned=defense.allow_poisoned_sources,
            trusted_categories=defense.trusted_categories,
            max_chunks=defense.max_context_chunks,
        )
    ]
    if defense.strip_instruction_like_text:
        chain.append(InjectionPatternDefense(strip_only=True))

    current = result
    for item in chain:
        current = item.apply(current)
    return current


__all__ = [
    "Defense",
    "InjectionPatternDefense",
    "ProvenanceDefense",
    "ChunkSanitizer",
    "ContextFirewall",
    "ContextFirewallConfig",
    "FirewallResult",
    "PatternDefinition",
    "InjectionMatch",
    "DetectionResult",
    "SanitizationResult",
    "SanitizeMode",
    "DiversityConfig",
    "DiversityItem",
    "DiversityReport",
    "DiversitySelectionResult",
    "DiversityEnforcer",
    "PenaltyWeights",
    "SecurityRerankerConfig",
    "RerankedResult",
    "SecurityReranker",
    "DefenseConfig",
    "filter_results",
    "ProvenanceTracker",
    "TrustLevel",
]
