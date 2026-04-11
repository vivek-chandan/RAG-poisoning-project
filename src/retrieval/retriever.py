"""Secure retrieval service with pluggable defenses."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from defenses.base import Defense
from retrieval.vector_store import VectorStore


@dataclass
class SecureRetriever:
    """Retrieves context and applies defense chain."""

    vector_store: VectorStore
    defenses: List[Defense]
    logger: logging.Logger

    def retrieve(self, query: str, top_k: int) -> dict:
        result = self.vector_store.query(query, top_k=top_k)
        self.logger.info("Retrieved %s candidate chunks", len(result.get("documents", [[]])[0]))
        for defense in self.defenses:
            result = defense.apply(result)
        self.logger.info("Post-defense chunks: %s", len(result.get("documents", [[]])[0]))
        return result
