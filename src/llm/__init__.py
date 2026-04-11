"""LLM package.

Provides model-provider interfaces and generation services.
"""

from llm.generator import ResponseGenerator, build_prompt
from llm.factory import build_provider
from llm.prompt_hardener import HardenedPrompt, HardeningStrategy, PromptHardener, PromptRequirements
from llm.providers import DeepSeekProvider, LocalLlamaProvider, OllamaProvider


def local_fallback_generate(prompt: str):
    """Compatibility wrapper for legacy callers."""
    provider = LocalLlamaProvider(model_path="")
    return provider.generate(prompt)


def ollama_generate(base_url: str, model: str, prompt: str):
    """Compatibility wrapper for legacy callers."""
    provider = OllamaProvider(base_url=base_url, model=model)
    return provider.generate(prompt)


def deepseek_generate(api_key: str, model: str, prompt: str):
    """Compatibility wrapper for legacy callers."""
    provider = DeepSeekProvider(api_key=api_key, model=model)
    return provider.generate(prompt)


def local_generate(model_path: str, prompt: str):
    """Compatibility wrapper for legacy callers."""
    provider = LocalLlamaProvider(model_path=model_path)
    return provider.generate(prompt)


__all__ = [
    "ResponseGenerator",
    "HardeningStrategy",
    "PromptRequirements",
    "HardenedPrompt",
    "PromptHardener",
    "OllamaProvider",
    "DeepSeekProvider",
    "LocalLlamaProvider",
    "build_prompt",
    "build_provider",
    "local_fallback_generate",
    "ollama_generate",
    "deepseek_generate",
    "local_generate",
]
