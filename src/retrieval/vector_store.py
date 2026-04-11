"""Vector-store abstractions and ChromaDB implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import chromadb
from chromadb.config import Settings

from documents import DocumentRecord
from retrieval.embeddings import EmbeddingProvider


class VectorStore(ABC):
    """Abstract vector-store contract."""

    @abstractmethod
    def ingest(self, records: Sequence[DocumentRecord]) -> None:
        """Index records into the store."""

    @abstractmethod
    def query(self, query_text: str, top_k: int, filters: Optional[Dict[str, object]] = None) -> Dict[str, List]:
        """Retrieve top-k records for a query."""

    @abstractmethod
    def count(self) -> int:
        """Return indexed record count."""


@dataclass
class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store."""

    db_path: str
    collection_name: str
    embeddings: EmbeddingProvider

    def __post_init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(self, records: Sequence[DocumentRecord]) -> None:
        if not records:
            return
        ids = [r.id for r in records]
        texts = [r.text for r in records]
        metadata = [
            {
                "source": r.source,
                "category": r.category,
                "poisoned": r.poisoned,
            }
            for r in records
        ]
        vectors = self.embeddings.embed(texts)
        self._collection.upsert(ids=ids, documents=texts, metadatas=metadata, embeddings=vectors)

    def query(self, query_text: str, top_k: int, filters: Optional[Dict[str, object]] = None) -> Dict[str, List]:
        vector = self.embeddings.embed([query_text])[0]
        return self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        return int(self._collection.count())
