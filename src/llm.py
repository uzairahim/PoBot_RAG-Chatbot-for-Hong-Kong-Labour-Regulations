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


class HFLocalProvider(LLMProvider):
    """Run our fine-tuned model in-process: base model + LoRA adapter.

    This is the connect-back for the fine-tuning bonus — set LLM_PROVIDER=hf_local
    and the QLoRA-tuned model answers through the exact same RAG pipeline. Runs on
    CPU if no GPU is present (slow but functional — it's a demonstration).
    """

    def __init__(self, model: str):
        super().__init__(model)
        import torch  # lazy imports — only needed for this provider
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = settings.hf_base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        # float32 for CPU correctness; no device_map (that path needs accelerate
        # and is for multi-GPU/offload). We place the model on the one device we
        # have — GPU if present, else CPU
        self._model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
        # Attach the fine-tuned adapter if one is configured.
        if settings.hf_adapter_dir:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, settings.hf_adapter_dir)
        self._model.to("cuda" if torch.cuda.is_available() else "cpu")
        self._model.eval()

    def chat(self, messages, temperature=0.1, max_tokens=1024) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._model.device)
        gen = self._model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=temperature > 0, temperature=max(temperature, 1e-2),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
    "hf_local": HFLocalProvider,
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
