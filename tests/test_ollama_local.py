"""
tests/test_ollama_local.py
Integration test for the Ollama-backed LLM client.

This test is NOT mocked — it requires:
  1. Ollama server running:  ollama serve
  2. Model pulled:           ollama pull gemma3:1b

Run with:
    python -m pytest tests/test_ollama_local.py -v -s

Skip automatically if the Ollama server is not reachable.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Helper: detect whether Ollama is reachable
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama server not reachable — start with `ollama serve` and pull gemma3:1b",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_ollama
def test_generate_ems_text_returns_string():
    """generate_ems_text() should return a non-empty string."""
    from pipeline.llm_client import generate_ems_text

    result = generate_ems_text(
        "Generate 1 short fictional EMS dispatch scenario. "
        "Return plain text, no JSON needed."
    )

    assert isinstance(result, str), "Expected str, got: " + type(result).__name__
    assert len(result.strip()) > 0, "Response was empty"
    print("\n[Ollama response]\n", result[:400])


@requires_ollama
def test_generate_ems_text_json_format():
    """generate_ems_text() should follow formatting instructions."""
    from pipeline.llm_client import generate_ems_text

    result = generate_ems_text(
        "Generate 2 fictional EMS emergency dispatch scenarios. "
        "Return ONLY a JSON array. Each object must have: "
        "scenario_id (string), dispatch_text (string), priority (High/Medium/Low). "
        "Do not include real patient information."
    )

    assert isinstance(result, str)
    assert len(result.strip()) > 10
    # Soft check: model should mention JSON-like structure
    lower = result.lower()
    assert any(kw in lower for kw in ("scenario", "dispatch", "priority", "ems", "{")), (
        f"Response does not look like EMS content:\n{result[:300]}"
    )
    print("\n[Ollama JSON response]\n", result[:600])


@requires_ollama
def test_generate_ems_text_custom_model():
    """generate_ems_text() should accept an explicit model override."""
    import os
    from pipeline.llm_client import generate_ems_text

    model = os.getenv("OLLAMA_MODEL", "gemma3:1b")
    result = generate_ems_text("Say: EMS test OK", model=model)

    assert isinstance(result, str)
    assert len(result.strip()) > 0

