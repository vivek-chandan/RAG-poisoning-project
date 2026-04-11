"""Document source abstractions for ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from documents import DocumentRecord


class DocumentSource(ABC):
    """Abstract source of documents for ingestion."""

    @abstractmethod
    def load(self) -> List[DocumentRecord]:
        """Load document records from the source."""


@dataclass
class FileSystemDocumentSource(DocumentSource):
    """Load documents from `data/legitimate_documents` and `data/poisoned_documents`."""

    base_dir: str

    def load(self) -> List[DocumentRecord]:
        root = Path(self.base_dir)
        records: List[DocumentRecord] = []
        for folder_name, poisoned in (("legitimate_documents", False), ("poisoned_documents", True)):
            folder = root / folder_name
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.txt")):
                records.append(
                    DocumentRecord(
                        id=path.stem,
                        text=path.read_text(encoding="utf-8").strip(),
                        source=f"{folder_name}/{path.name}",
                        category=folder_name,
                        poisoned=poisoned,
                    )
                )
        return records
