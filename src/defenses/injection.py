"""Prompt-injection pattern filtering defense."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from defenses.base import Defense


PATTERNS = [
    re.compile(r"ignore previous instructions", re.IGNORECASE),
    re.compile(r"always reveal", re.IGNORECASE),
    re.compile(r"print secrets", re.IGNORECASE),
]


@dataclass
class InjectionPatternDefense(Defense):
    """Mask or remove instruction-like text in retrieved chunks."""

    strip_only: bool = True

    def apply(self, result: Dict[str, List]) -> Dict[str, List]:
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0] if "ids" in result else [str(i) for i in range(len(docs))]

        transformed = []
        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            matched = any(pattern.search(doc) for pattern in PATTERNS)
            if matched and not self.strip_only:
                continue
            if matched:
                doc = re.sub(r"(?i)ignore previous instructions\.?", "", doc).strip()
            transformed.append((doc_id, doc, meta, dist))

        return {
            "ids": [[x[0] for x in transformed]],
            "documents": [[x[1] for x in transformed]],
            "metadatas": [[x[2] for x in transformed]],
            "distances": [[x[3] for x in transformed]],
        }
