"""Defense strategy interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class Defense(ABC):
    """Base class for retrieval-time defenses."""

    @abstractmethod
    def apply(self, result: Dict[str, List]) -> Dict[str, List]:
        """Apply defense to retrieval result and return transformed result."""
