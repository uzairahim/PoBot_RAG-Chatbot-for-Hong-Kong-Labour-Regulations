"""Pluggable LLM provider layer.

The RAG pipeline only ever calls `get_llm().chat(messages)`. Which concrete
model answers is decided by config (LLM_PROVIDER / LLM_MODEL) — swap providers
by changing one env value, no pipeline changes. Provider SDKs are imported
lazily so you only need the package for the provider you actually use.

Add a new provider by writing a subclass and registering it in _PROVIDERS.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.config import settings

# A chat message is {"role": "system"|"user"|"assistant", "content": str}.
Message = dict[str, str]


class LLMProvider(ABC):
    """Minimal interface every provider implements."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def chat(self, messages: list[Message], temperature: float = 0.1,
             max_tokens: int = 1024) -> str:
        """Return the assistant's reply text for a list of chat messages."""


class GroqProvider(LLMProvider):
    """Groq — fast, free-tier hosted open models (default)."""

    def __init__(self, model: str):
        super().__init__(model)
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env "
                "(get a free key at https://console.groq.com/keys)."
            )
        from groq import Groq  # lazy import
        self._client = Groq(api_key=settings.groq_api_key)

    def chat(self, messages, temperature=0.1, max_tokens=1024) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


class OpenAIProvider(LLMProvider):
    """OpenAI or any OpenAI-compatible endpoint (set OPENAI_BASE_URL)."""

    def __init__(self, model: str):
        super().__init__(model)
        from openai import OpenAI  # lazy import
        kwargs = {"api_key": settings.openai_api_key or "not-needed"}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self._client = OpenAI(**kwargs)

    def chat(self, messages, temperature=0.1, max_tokens=1024) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


class OllamaProvider(LLMProvider):
    """Fully-offline local models via a running Ollama daemon."""

    def chat(self, messages, temperature=0.1, max_tokens=1024) -> str:
        import ollama  # lazy import
        resp = ollama.chat(
            model=self.model, messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return resp["message"]["content"].strip()


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}


def get_llm(provider: str | None = None, model: str | None = None) -> LLMProvider:
    """Factory: build the configured provider (override via args for testing)."""
    provider = (provider or settings.llm_provider).lower()
    model = model or settings.llm_model
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Options: {', '.join(_PROVIDERS)}."
        )
    return _PROVIDERS[provider](model)
