"""Retrieval package.

Contains embedding providers, vector-store adapters, and retriever services.
"""

from retrieval.embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from retrieval.retriever import SecureRetriever
from retrieval.vector_store import ChromaVectorStore, VectorStore
from retrieval.vector_store_manager import IngestionDocument, SearchResult, VectorStoreManager

__all__ = [
    "EmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "VectorStore",
    "ChromaVectorStore",
    "SecureRetriever",
    "VectorStoreManager",
    "IngestionDocument",
    "SearchResult",
]
