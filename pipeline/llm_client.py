"""
pipeline/llm_client.py
Lightweight Ollama adapter for EMS text generation.

Uses gemma3:1b by default — a free, local model (~815 MB).
No OpenAI or Anthropic key required.

Usage:
    from pipeline.llm_client import generate_ems_text
    text = generate_ems_text("Generate 3 EMS dispatch scenarios in JSON.")
"""

from __future__ import annotations

import os

import ollama

MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")

_SYSTEM_PROMPT = (
    "You generate fictional EMS training scenarios. "
    "Never invent real patient identities. "
    "Follow the user's requested output format."
)


def generate_ems_text(prompt: str, model: str | None = None) -> str:
    """Generate EMS-domain text using a local Ollama model.

    Parameters
    ----------
    prompt : str
        User instruction describing what to generate.
    model : str | None
        Ollama model tag to use. Defaults to the ``OLLAMA_MODEL`` environment
        variable, falling back to ``gemma3:1b``.

    Returns
    -------
    str
        Generated text from the model.

    Raises
    ------
    ollama.ResponseError
        If the Ollama server is unreachable or the model is not available.
    """
    _model = model or MODEL
    response = ollama.chat(
        model=_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": 0.2,
            "num_predict": 512,
        },
    )
    return response["message"]["content"]


EMS_SYSTEM_PROMPT = _SYSTEM_PROMPT


class OllamaClient:
    """Convenience client wrapper around Ollama."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or MODEL

    def generate_text(self, user_prompt: str, system_prompt: str | None = None) -> str:
        _sys = system_prompt or _SYSTEM_PROMPT
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": _sys},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.2,
                "num_predict": 512,
            },
        )
        return response["message"]["content"]

