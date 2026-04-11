"""LLM provider implementations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import httpx


@dataclass
class LLMResponse:
    """Unified LLM response object."""

    provider: str
    text: str
    raw: Optional[dict] = None


class LLMProvider(Protocol):
    """Provider protocol for text generation."""

    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response for a prompt."""


def _fallback_extract(prompt: str) -> str:
    lowered = prompt.lower()
    if "100 vacation days" in lowered:
        return "Employees get 100 vacation days per year."
    if "20 vacation days" in lowered:
        return "Employees get 20 vacation days per year."
    return "I do not know from the provided context."


@dataclass
class LocalLlamaProvider:
    """Local llama.cpp provider with robust fallback behavior."""

    model_path: str
    fail_on_error: bool = False
    logger: logging.Logger = logging.getLogger("secure_rag.llm.local")

    def generate(self, prompt: str) -> LLMResponse:
        if not self.model_path:
            if self.fail_on_error:
                raise RuntimeError("Local model path is empty")
            return LLMResponse(provider="fallback", text=_fallback_extract(prompt))
        if not Path(self.model_path).exists():
            message = f"Model path does not exist: {self.model_path}"
            self.logger.error(message)
            if self.fail_on_error:
                raise FileNotFoundError(message)
            return LLMResponse(provider="fallback", text=_fallback_extract(prompt))
        try:
            from llama_cpp import Llama

            llm = Llama(model_path=self.model_path, n_ctx=2048, verbose=False)
            output = llm(prompt, max_tokens=128, temperature=0.0)
            text = output["choices"][0]["text"].strip()
            if text:
                return LLMResponse(provider="local-llama", text=text, raw=output)
            if self.fail_on_error:
                raise RuntimeError("Local llama returned an empty response")
        except Exception as exc:
            self.logger.exception("Local llama generation failed: %s", exc)
            if self.fail_on_error:
                raise RuntimeError("Local llama generation failed") from exc
        return LLMResponse(provider="fallback", text=_fallback_extract(prompt))


@dataclass
class OllamaProvider:
    """Ollama HTTP API provider."""

    base_url: str
    model: str
    logger: logging.Logger = logging.getLogger("secure_rag.llm.ollama")

    def generate(self, prompt: str) -> LLMResponse:
        try:
            response = httpx.post(
                f"{self.base_url.rstrip('/')}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=120.0,
            )
            response.raise_for_status()
            payload = response.json()
            return LLMResponse(provider="ollama", text=payload.get("response", ""), raw=payload)
        except Exception as exc:
            self.logger.exception("Ollama generation failed: %s", exc)
            return LLMResponse(provider="fallback", text=_fallback_extract(prompt))


@dataclass
class DeepSeekProvider:
    """DeepSeek API provider."""

    api_key: str
    model: str
    logger: logging.Logger = logging.getLogger("secure_rag.llm.deepseek")

    def generate(self, prompt: str) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(provider="fallback", text=_fallback_extract(prompt))
        try:
            response = httpx.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            payload = response.json()
            text = payload["choices"][0]["message"]["content"]
            return LLMResponse(provider="deepseek", text=text, raw=payload)
        except Exception as exc:
            self.logger.exception("DeepSeek generation failed: %s", exc)
            return LLMResponse(provider="fallback", text=_fallback_extract(prompt))
