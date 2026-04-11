"""Embedding providers used by retrieval services."""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingProvider(ABC):
    """Embedding provider interface."""

    @abstractmethod
    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """Convert text collection to dense vectors."""


@dataclass
class HashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic fallback embedding provider."""

    dimension: int = 384

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=float)
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                vector[index] += 1.0
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            vectors.append(vector.tolist())
        return vectors


@dataclass
class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Primary embedding provider backed by sentence-transformers."""

    model_name: str
    cache_dir: str
    logger: logging.Logger

    def __post_init__(self) -> None:
        self._fallback = HashingEmbeddingProvider()
        try:
            self._model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
            self.logger.info("Loaded embedding model: %s", self.model_name)
        except Exception as exc:
            self.logger.exception("Embedding model load failed; using fallback: %s", exc)
            self._model = None

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        text_list = list(texts)
        if self._model is None:
            return self._fallback.embed(text_list)
        vectors = self._model.encode(text_list, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=float).tolist()
