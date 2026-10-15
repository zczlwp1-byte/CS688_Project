"""
llm.py — Thin wrapper around the local Ollama server.

All model calls in this project go through this module, so swapping the
backend (e.g., to an OpenAI-compatible API) only requires changing this file.

Requires: Ollama running locally (https://ollama.com), with models pulled:
    ollama pull qwen2.5:7b          # chat model (or llama3.1:8b, qwen2.5:3b ...)
    ollama pull nomic-embed-text    # embedding model
"""

from __future__ import annotations

import ollama

# Default models — change here or override per-call.
CHAT_MODEL = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text"


def chat(messages: list[dict], model: str = CHAT_MODEL, temperature: float = 0.2) -> str:
    """Send a chat conversation to the local LLM and return the reply text.

    Args:
        messages: list of {"role": "system"|"user"|"assistant", "content": str}
        model: Ollama model tag.
        temperature: sampling temperature (low = more deterministic).
    """
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )
    return response["message"]["content"]


def generate(prompt: str, system: str = "", model: str = CHAT_MODEL,
             temperature: float = 0.2) -> str:
    """Convenience single-turn completion."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, model=model, temperature=temperature)


def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Return one embedding vector per input text (all computed locally)."""
    response = ollama.embed(model=model, input=texts)
    return response["embeddings"]


def is_available() -> bool:
    """Check whether the Ollama server is reachable."""
    try:
        ollama.list()
        return True
    except Exception:
        return False
