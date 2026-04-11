"""Ingestion pipeline service."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ingestion.sources import DocumentSource
from retrieval.vector_store import VectorStore


@dataclass
class IngestionPipeline:
    """Coordinates loading and indexing documents."""

    source: DocumentSource
    vector_store: VectorStore
    logger: logging.Logger

    def run(self) -> int:
        """Load and index documents; return indexed count."""
        records = self.source.load()
        self.logger.info("Loaded %s documents from source", len(records))
        if not records:
            self.logger.warning("No documents found for ingestion")
            return 0
        self.vector_store.ingest(records)
        indexed = self.vector_store.count()
        self.logger.info("Vector store now contains %s records", indexed)
        return indexed
