"""Provider factory helpers for LLM runtime selection."""

from __future__ import annotations

from llm.providers import DeepSeekProvider, LocalLlamaProvider, OllamaProvider


def build_provider(
    provider_name: str,
    *,
    ollama_base_url: str,
    ollama_model: str,
    deepseek_api_key: str,
    deepseek_model: str,
    llama_model_path: str,
    fail_on_local_error: bool = False,
):
    """Construct an LLM provider instance from configuration values."""
    normalized = provider_name.lower().strip()
    if normalized == "ollama":
        return OllamaProvider(base_url=ollama_base_url, model=ollama_model)
    if normalized == "deepseek":
        return DeepSeekProvider(api_key=deepseek_api_key, model=deepseek_model)
    return LocalLlamaProvider(model_path=llama_model_path, fail_on_error=fail_on_local_error)
