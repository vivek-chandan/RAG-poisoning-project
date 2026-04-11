from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv


def _load_dotenv_files() -> None:
    load_dotenv(override=False)
    keys_path = Path(__file__).resolve().parent.parent / ".keys"
    if keys_path.exists():
        load_dotenv(keys_path, override=False)
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def _configure_noisy_loggers() -> None:
    # Chroma 0.4.x may emit repeated telemetry errors in some environments.
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
    logging.getLogger("chromadb.telemetry.product").setLevel(logging.CRITICAL)


_load_dotenv_files()
_configure_noisy_loggers()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ensure_parent(path: str) -> str:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class Config:
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    vector_db_path: str = field(default_factory=lambda: os.getenv("VECTOR_DB_PATH", "./data/chroma_db"))
    llama_model_path: str = field(default_factory=lambda: os.getenv("LLAMA_MODEL_PATH", "./models/llm/Phi-3.5-mini-instruct.Q4_K_M.gguf"))
    sentence_transformers_home: str = field(default_factory=lambda: os.getenv("SENTENCE_TRANSFORMERS_HOME", "./models/embedding"))
    transformers_cache: str = field(default_factory=lambda: os.getenv("TRANSFORMERS_CACHE", "./models/embedding"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "Phi-3.5-mini-instruct:latest"))
    deepseek_model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    top_k_retrieval: int = field(default_factory=lambda: int(os.getenv("TOP_K_RETRIEVAL", "3").split("#")[0].strip()))
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.0").split("#")[0].strip() or 0.0))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "WARN"))
    log_file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "./logs/rag_demo.log"))
    use_gpu: bool = field(default_factory=lambda: _env_bool("USE_GPU", False))
    app_root: str = field(default_factory=lambda: str(Path(__file__).resolve().parent.parent))

    def __post_init__(self) -> None:
        self.vector_db_path = _ensure_parent(self.vector_db_path)
        self.llama_model_path = _ensure_parent(self.llama_model_path)
        self.sentence_transformers_home = _ensure_parent(self.sentence_transformers_home)
        self.transformers_cache = _ensure_parent(self.transformers_cache)
        self.log_file = _ensure_parent(self.log_file)
        Path(self.vector_db_path).mkdir(parents=True, exist_ok=True)
        Path(self.sentence_transformers_home).mkdir(parents=True, exist_ok=True)
        Path(self.transformers_cache).mkdir(parents=True, exist_ok=True)
        Path(self.llama_model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

    def print_config(self) -> None:
        values: Dict[str, object] = {
            "embedding_model": self.embedding_model,
            "vector_db_path": self.vector_db_path,
            "llama_model_path": self.llama_model_path,
            "sentence_transformers_home": self.sentence_transformers_home,
            "transformers_cache": self.transformers_cache,
            "ollama_base_url": self.ollama_base_url,
            "ollama_model": self.ollama_model,
            "deepseek_model": self.deepseek_model,
            "top_k_retrieval": self.top_k_retrieval,
            "similarity_threshold": self.similarity_threshold,
            "log_level": self.log_level,
            "log_file": self.log_file,
        }
        for key, value in values.items():
            print(f"{key}: {value}")
