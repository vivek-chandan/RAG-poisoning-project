"""Ingestion package.

Provides document loading, validation, and ingestion orchestration.
"""

from ingestion.pipeline import IngestionPipeline
from ingestion.sources import DocumentSource, FileSystemDocumentSource

__all__ = [
    "DocumentSource",
    "FileSystemDocumentSource",
    "IngestionPipeline",
]
