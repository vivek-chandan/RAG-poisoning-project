"""Secure RAG orchestrator.

Implements a defense-first retrieval pipeline:
1) Retrieve raw candidates
2) Provenance filtering
3) Security reranking
4) Diversity enforcement
5) Chunk sanitization
6) Context firewall
7) Prompt hardening
8) LLM answer generation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from config import Config
from defenses import (
    ChunkSanitizer,
    ContextFirewall,
    ContextFirewallConfig,
    DiversityConfig,
    DiversityEnforcer,
    DiversityItem,
    ProvenanceTracker,
    SanitizeMode,
    SecurityReranker,
    SecurityRerankerConfig,
    TrustLevel,
)
from llm import HardeningStrategy, PromptHardener, PromptRequirements, build_provider
from retrieval.vector_store_manager import SearchResult, VectorStoreManager


@dataclass(frozen=True)
class OrchestratorResult:
    """Structured response from the SecureRAG orchestrator."""

    answer: str
    sources_used: List[str]
    warnings: List[str]
    blocked_docs: List[Dict[str, Any]]
    diversity_score: float
    security_metrics: Dict[str, Any]


@dataclass
class SecureRAGOrchestratorConfig:
    """Runtime configuration for the orchestrator pipeline."""

    retrieval_top_k: int = 10
    final_top_k: int = 4
    llm_provider: str = "local"  # local | ollama | deepseek
    fail_on_local_error: bool = False
    hardening_strategy: HardeningStrategy = HardeningStrategy.STRICT
    sanitize_mode: SanitizeMode = SanitizeMode.REDACT
    prompt_requirements: PromptRequirements = field(
        default_factory=lambda: PromptRequirements(
            require_confidence_score=True,
            require_source_citations=True,
            require_contradiction_reporting=True,
        )
    )


class SecureRAGOrchestrator:
    """Production-style orchestrator for secure retrieval-augmented generation."""

    def __init__(
        self,
        app_config: Optional[Config] = None,
        orchestrator_config: Optional[SecureRAGOrchestratorConfig] = None,
        vector_store_manager: Optional[VectorStoreManager] = None,
        provenance_tracker: Optional[ProvenanceTracker] = None,
        security_reranker: Optional[SecurityReranker] = None,
        diversity_enforcer: Optional[DiversityEnforcer] = None,
        chunk_sanitizer: Optional[ChunkSanitizer] = None,
        context_firewall: Optional[ContextFirewall] = None,
        prompt_hardener: Optional[PromptHardener] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._app_config = app_config or Config()
        self._cfg = orchestrator_config or SecureRAGOrchestratorConfig(
            retrieval_top_k=max(self._app_config.top_k_retrieval * 3, 10),
            final_top_k=self._app_config.top_k_retrieval,
        )

        self._logger = logger or logging.getLogger("secure_rag.orchestrator")

        self._vector_store = vector_store_manager or VectorStoreManager(
            db_path=self._app_config.vector_db_path,
            collection_name="secure_rag_documents",
            embedding_model_name=self._app_config.embedding_model,
            embedding_cache_dir=self._app_config.sentence_transformers_home,
            logger=self._logger,
        )

        self._provenance = provenance_tracker or ProvenanceTracker(
            registry_path=str(Path(self._app_config.app_root) / "data" / "provenance_registry.json"),
            minimum_trust_level=TrustLevel.MEDIUM,
            logger=self._logger,
        )

        self._reranker = security_reranker or SecurityReranker(config=SecurityRerankerConfig())
        self._diversity = diversity_enforcer or DiversityEnforcer(
            config=DiversityConfig(top_k=self._cfg.final_top_k)
        )
        self._sanitizer = chunk_sanitizer or ChunkSanitizer(logger=self._logger)
        self._firewall = context_firewall or ContextFirewall(
            config=ContextFirewallConfig(max_segments_per_source=2)
        )
        self._hardener = prompt_hardener or PromptHardener(strategy=self._cfg.hardening_strategy)

    def _select_provider(self):
        return build_provider(
            self._cfg.llm_provider,
            ollama_base_url=self._app_config.ollama_base_url,
            ollama_model=self._app_config.ollama_model,
            deepseek_api_key=self._app_config.deepseek_api_key,
            deepseek_model=self._app_config.deepseek_model,
            llama_model_path=self._app_config.llama_model_path,
            fail_on_local_error=self._cfg.fail_on_local_error,
        )

    @staticmethod
    def _search_to_chroma_shape(search_results: Sequence[SearchResult]) -> Dict[str, List]:
        return {
            "ids": [[item.document_id for item in search_results]],
            "documents": [[item.text for item in search_results]],
            "metadatas": [[item.metadata for item in search_results]],
            "distances": [[item.score_distance for item in search_results]],
        }

    def _apply_chunk_sanitization(self, items: Sequence[DiversityItem]) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
        sanitized_chunks: List[Dict[str, Any]] = []
        warnings: List[str] = []

        total_risk = 0.0
        risk_items = 0
        sanitized_count = 0

        for item in items:
            sanitized = self._sanitizer.sanitize(item.document, mode=self._cfg.sanitize_mode)
            if sanitized.detection.is_injection:
                warnings.append(
                    f"Sanitizer flagged chunk '{item.item_id}' (risk={sanitized.detection.risk_score}, mode={self._cfg.sanitize_mode.value})"
                )
                total_risk += sanitized.detection.risk_score
                risk_items += 1
                sanitized_count += 1

            metadata = dict(item.metadata)
            metadata["injection_risk_score"] = sanitized.detection.risk_score

            sanitized_chunks.append(
                {
                    "id": item.item_id,
                    "text": sanitized.sanitized_text,
                    "score": item.score,
                    "metadata": metadata,
                }
            )

        metrics = {
            "sanitized_chunks": sanitized_count,
            "avg_injection_risk_selected": round((total_risk / risk_items), 6) if risk_items else 0.0,
        }
        return sanitized_chunks, metrics, warnings

    def query(self, question: str) -> OrchestratorResult:
        """Run full secure RAG pipeline and return structured response."""
        warnings: List[str] = []
        blocked_docs: List[Dict[str, Any]] = []
        security_metrics: Dict[str, Any] = {}

        try:
            # 1) Retrieve raw candidates.
            raw_candidates = self._vector_store.search(question, top_k=self._cfg.retrieval_top_k)
            security_metrics["raw_candidates"] = len(raw_candidates)

            # 2) Provenance filtering.
            provenance_input = self._search_to_chroma_shape(raw_candidates)
            provenance_output = self._provenance.filter_by_provenance(provenance_input)
            accepted = provenance_output["accepted"]
            rejected = provenance_output["rejected"]["items"]
            blocked_docs.extend(rejected)
            warnings.extend([f"Provenance rejected document '{item.get('id')}' reason={item.get('reason')}" for item in rejected])
            security_metrics["provenance_accepted"] = len(accepted.get("ids", [[]])[0])
            security_metrics["provenance_rejected"] = len(rejected)

            # 3) Security reranking.
            reranked = self._reranker.rerank_from_chroma_result(accepted)
            security_metrics["reranked_candidates"] = len(reranked)
            security_metrics["avg_rerank_penalty"] = round(
                sum(item.penalty for item in reranked) / len(reranked), 6
            ) if reranked else 0.0

            # 4) Diversity enforcement.
            diversity_candidates = [
                DiversityItem(
                    item_id=item.document_id,
                    document=item.document,
                    score=item.adjusted_score,
                    metadata=item.metadata,
                )
                for item in reranked
            ]
            diversity_result = self._diversity.select_top_k(diversity_candidates)
            diversity_score = diversity_result.report_after.normalized_entropy
            security_metrics["diversity_before_entropy"] = diversity_result.report_before.normalized_entropy
            security_metrics["diversity_after_entropy"] = diversity_score
            security_metrics["source_dominance_detected"] = diversity_result.report_after.source_dominance_detected

            # 5) Chunk sanitization.
            sanitized_chunks, sanitizer_metrics, sanitizer_warnings = self._apply_chunk_sanitization(diversity_result.selected)
            warnings.extend(sanitizer_warnings)
            security_metrics.update(sanitizer_metrics)

            # 6) Context firewall.
            firewall_result = self._firewall.validate_chunks(sanitized_chunks)
            warnings.extend(firewall_result.warnings)
            blocked_docs.extend(
                [
                    {
                        "id": chunk.chunk_id,
                        "reason": ",".join(chunk.reasons),
                        "source_id": chunk.source_id,
                        "metadata": chunk.metadata,
                    }
                    for chunk in firewall_result.blocked_chunks
                ]
            )
            blocked_docs.extend(
                [
                    {
                        "id": chunk.chunk_id,
                        "reason": "quarantined:" + ",".join(chunk.reasons),
                        "source_id": chunk.source_id,
                        "metadata": chunk.metadata,
                    }
                    for chunk in firewall_result.quarantined_chunks
                ]
            )
            security_metrics["firewall_allowed"] = len(firewall_result.allowed_chunks)
            security_metrics["firewall_blocked"] = len(firewall_result.blocked_chunks)
            security_metrics["firewall_quarantined"] = len(firewall_result.quarantined_chunks)

            # 7) Prompt hardening.
            self._hardener.set_strategy(self._cfg.hardening_strategy)
            hardened = self._hardener.generate(
                question=question,
                context=firewall_result.safe_context,
                requirements=self._cfg.prompt_requirements,
            )

            # 8) Generate LLM answer.
            provider = self._select_provider()
            llm_response = provider.generate(hardened.final_prompt)
            security_metrics["llm_provider"] = llm_response.provider

            sources_used = sorted({chunk.source_id for chunk in firewall_result.allowed_chunks})

            return OrchestratorResult(
                answer=llm_response.text,
                sources_used=sources_used,
                warnings=warnings,
                blocked_docs=blocked_docs,
                diversity_score=diversity_score,
                security_metrics=security_metrics,
            )
        except Exception as exc:
            self._logger.exception("SecureRAG query failed: %s", exc)
            if self._cfg.fail_on_local_error:
                raise
            warnings.append(f"pipeline_error:{exc}")
            return OrchestratorResult(
                answer="I could not safely answer due to pipeline error.",
                sources_used=[],
                warnings=warnings,
                blocked_docs=blocked_docs,
                diversity_score=0.0,
                security_metrics=security_metrics,
            )


def example_usage() -> None:
    """Minimal runnable example."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    orchestrator = SecureRAGOrchestrator()
    result = orchestrator.query("What is our vacation policy?")
    print(
        {
            "answer": result.answer,
            "sources_used": result.sources_used,
            "warnings": result.warnings[:3],
            "blocked_docs": len(result.blocked_docs),
            "diversity_score": result.diversity_score,
            "security_metrics": result.security_metrics,
        }
    )


if __name__ == "__main__":
    example_usage()
