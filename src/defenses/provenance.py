"""Provenance-based defense policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from defenses.base import Defense


@dataclass
class ProvenanceDefense(Defense):
    """Filter retrieved chunks using source trust and poisoning metadata."""

    allow_poisoned: bool = True
    trusted_categories: Tuple[str, ...] = ()
    max_chunks: int = 4

    def apply(self, result: Dict[str, List]) -> Dict[str, List]:
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0] if "ids" in result else [str(i) for i in range(len(docs))]

        kept = []
        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            poisoned = bool(meta.get("poisoned"))
            source = str(meta.get("source", ""))
            category = source.split("/")[0] if "/" in source else str(meta.get("category", ""))

            if not self.allow_poisoned and poisoned:
                continue
            if self.trusted_categories and category not in self.trusted_categories:
                continue
            kept.append((doc_id, doc, meta, dist))

        kept = kept[: self.max_chunks]
        return {
            "ids": [[x[0] for x in kept]],
            "documents": [[x[1] for x in kept]],
            "metadatas": [[x[2] for x in kept]],
            "distances": [[x[3] for x in kept]],
        }
