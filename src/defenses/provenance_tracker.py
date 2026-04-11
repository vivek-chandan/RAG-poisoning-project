"""Provenance tracking and verification for Secure RAG retrieval.

This module provides an auditable provenance policy layer that can be applied
before retrieval or immediately after retrieval to reject untrusted context.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class TrustLevel(str, Enum):
    """Allowed provenance trust levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class SourceRecord:
    """Registry record for an approved or monitored source."""

    source_id: str
    trust_level: TrustLevel
    description: str = ""
    public_key_id: str = ""
    enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class DocumentVerificationResult:
    """Outcome of provenance verification for one document."""

    accepted: bool
    reason: str
    source_id: str
    trust_level: str
    metadata: Dict[str, Any]


class ProvenanceTracker:
    """Track trusted sources and verify document provenance.

    Features:
    - trusted source allowlist and metadata registry
    - trust level threshold enforcement
    - document hash validation
    - optional HMAC signature verification
    - batch filtering with accepted/rejected split
    """

    def __init__(
        self,
        registry_path: str = "./data/provenance_registry.json",
        minimum_trust_level: TrustLevel = TrustLevel.MEDIUM,
        signature_secrets: Optional[Mapping[str, str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or logging.getLogger("secure_rag.provenance")
        self._registry_path = Path(registry_path)
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._minimum_trust_level = minimum_trust_level
        self._signature_secrets: Dict[str, str] = dict(signature_secrets or {})
        self._registry: Dict[str, SourceRecord] = {}
        self._load_registry()

    @staticmethod
    def _rank(level: TrustLevel) -> int:
        order = {
            TrustLevel.LOW: 0,
            TrustLevel.MEDIUM: 1,
            TrustLevel.HIGH: 2,
            TrustLevel.CRITICAL: 3,
        }
        return order[level]

    @staticmethod
    def _hash_document(document_text: str) -> str:
        return hashlib.sha256(document_text.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_level(value: str | TrustLevel) -> TrustLevel:
        if isinstance(value, TrustLevel):
            return value
        try:
            return TrustLevel(value.lower())
        except Exception as exc:
            raise ValueError(f"Invalid trust level: {value}") from exc

    def _load_registry(self) -> None:
        if not self._registry_path.exists():
            self._logger.info("Provenance registry not found at %s; starting empty", self._registry_path)
            return
        try:
            payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
            records = payload.get("sources", [])
            for item in records:
                level = self._normalize_level(item["trust_level"])
                record = SourceRecord(
                    source_id=item["source_id"],
                    trust_level=level,
                    description=item.get("description", ""),
                    public_key_id=item.get("public_key_id", ""),
                    enabled=bool(item.get("enabled", True)),
                    created_at=item.get("created_at", datetime.now(timezone.utc).isoformat()),
                    updated_at=item.get("updated_at", datetime.now(timezone.utc).isoformat()),
                )
                self._registry[record.source_id] = record
            self._logger.info("Loaded %s source records from registry", len(self._registry))
        except Exception as exc:
            self._logger.exception("Failed to load provenance registry: %s", exc)
            raise RuntimeError(f"Failed to load provenance registry: {exc}") from exc

    def _save_registry(self) -> None:
        payload = {"sources": [asdict(item) for item in self._registry.values()]}
        try:
            self._registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            self._logger.exception("Failed to save provenance registry: %s", exc)
            raise RuntimeError(f"Failed to save provenance registry: {exc}") from exc

    def add_trusted_source(
        self,
        source_id: str,
        trust_level: str | TrustLevel,
        description: str = "",
        public_key_id: str = "",
        enabled: bool = True,
    ) -> SourceRecord:
        """Add or update a trusted source in the registry allowlist."""
        if not source_id.strip():
            raise ValueError("source_id must be non-empty")
        level = self._normalize_level(trust_level)
        now = datetime.now(timezone.utc).isoformat()

        existing = self._registry.get(source_id)
        created_at = existing.created_at if existing else now
        record = SourceRecord(
            source_id=source_id,
            trust_level=level,
            description=description,
            public_key_id=public_key_id,
            enabled=enabled,
            created_at=created_at,
            updated_at=now,
        )
        self._registry[source_id] = record
        self._save_registry()
        self._logger.info("Added/updated trusted source '%s' with level '%s'", source_id, level.value)
        return record

    def verify_source(self, source_id: str) -> bool:
        """Verify a source against allowlist and minimum trust policy."""
        record = self._registry.get(source_id)
        if not record:
            self._logger.warning("Source verification failed: source '%s' not in allowlist", source_id)
            return False
        if not record.enabled:
            self._logger.warning("Source verification failed: source '%s' is disabled", source_id)
            return False
        if self._rank(record.trust_level) < self._rank(self._minimum_trust_level):
            self._logger.warning(
                "Source verification failed: source '%s' trust '%s' is below minimum '%s'",
                source_id,
                record.trust_level.value,
                self._minimum_trust_level.value,
            )
            return False
        return True

    def _verify_signature(self, document_text: str, metadata: Mapping[str, Any]) -> Tuple[bool, str]:
        signature = str(metadata.get("signature", "")).strip()
        if not signature:
            return True, "no_signature"

        key_id = str(metadata.get("signature_key_id", "")).strip()
        if not key_id:
            return False, "missing_signature_key_id"

        secret = self._signature_secrets.get(key_id)
        if not secret:
            return False, "unknown_signature_key_id"

        computed = hmac.new(secret.encode("utf-8"), document_text.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, computed):
            return False, "invalid_signature"
        return True, "signature_valid"

    def verify_document(self, document_text: str, metadata: Mapping[str, Any]) -> DocumentVerificationResult:
        """Verify one document's provenance and integrity metadata."""
        source_id = str(metadata.get("source_id", "")).strip()
        if not source_id:
            return DocumentVerificationResult(False, "missing_source_id", "", "", dict(metadata))

        if not self.verify_source(source_id):
            record = self._registry.get(source_id)
            level = record.trust_level.value if record else "unknown"
            return DocumentVerificationResult(False, "source_not_trusted", source_id, level, dict(metadata))

        record = self._registry[source_id]

        provided_hash = str(metadata.get("doc_hash", "")).strip()
        if not provided_hash:
            return DocumentVerificationResult(False, "missing_doc_hash", source_id, record.trust_level.value, dict(metadata))

        computed_hash = self._hash_document(document_text)
        if not hmac.compare_digest(provided_hash, computed_hash):
            self._logger.warning("Document hash mismatch for source '%s'", source_id)
            return DocumentVerificationResult(False, "doc_hash_mismatch", source_id, record.trust_level.value, dict(metadata))

        signature_ok, signature_reason = self._verify_signature(document_text, metadata)
        if not signature_ok:
            self._logger.warning("Signature verification failed for source '%s': %s", source_id, signature_reason)
            return DocumentVerificationResult(False, signature_reason, source_id, record.trust_level.value, dict(metadata))

        self._logger.info(
            "Document accepted for source '%s' (trust=%s, signature=%s)",
            source_id,
            record.trust_level.value,
            signature_reason,
        )
        return DocumentVerificationResult(True, "accepted", source_id, record.trust_level.value, dict(metadata))

    def filter_by_provenance(self, retrieval_result: Mapping[str, List]) -> Dict[str, Any]:
        """Split retrieval result into accepted and rejected documents.

        Input format expected from Chroma-like query output:
            {
                "ids": [[...]],
                "documents": [[...]],
                "metadatas": [[...]],
                "distances": [[...]],
            }
        """
        ids = retrieval_result.get("ids", [[]])[0]
        docs = retrieval_result.get("documents", [[]])[0]
        metas = retrieval_result.get("metadatas", [[]])[0]
        dists = retrieval_result.get("distances", [[]])[0]

        accepted_rows: List[Tuple[Any, Any, Any, Any]] = []
        rejected_rows: List[Dict[str, Any]] = []

        for doc_id, doc_text, meta, distance in zip(ids, docs, metas, dists):
            verification = self.verify_document(str(doc_text), dict(meta))
            if verification.accepted:
                accepted_rows.append((doc_id, doc_text, meta, distance))
            else:
                rejected_rows.append(
                    {
                        "id": doc_id,
                        "document": doc_text,
                        "metadata": meta,
                        "distance": distance,
                        "reason": verification.reason,
                        "source_id": verification.source_id,
                        "trust_level": verification.trust_level,
                    }
                )

        accepted = {
            "ids": [[row[0] for row in accepted_rows]],
            "documents": [[row[1] for row in accepted_rows]],
            "metadatas": [[row[2] for row in accepted_rows]],
            "distances": [[row[3] for row in accepted_rows]],
        }
        rejected = {
            "items": rejected_rows,
            "count": len(rejected_rows),
        }

        self._logger.info(
            "Provenance filtering completed: accepted=%s rejected=%s",
            len(accepted_rows),
            len(rejected_rows),
        )
        return {
            "accepted": accepted,
            "rejected": rejected,
        }
