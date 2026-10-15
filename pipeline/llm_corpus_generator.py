"""
pipeline/llm_corpus_generator.py
Adapter that wraps Repository 1's llm_ems_dialogue_generator.py and
makes Ollama the default provider — no OpenAI or Anthropic key needed.

Recommended Ollama models for this project (best → fastest):
  - gemma3:1b     Default — lightweight, fast, ~815 MB RAM
  - llama3.1:8b   Best quality for EMS dialogue, needs ~6 GB RAM
  - mistral:7b    Good quality, slightly faster, ~5 GB RAM
  - llama3.2:3b   Fastest, lowest RAM (~3 GB), decent quality
  - phi3:mini     Smallest (~2 GB), good for low-end hardware

Usage:
    from pipeline.llm_corpus_generator import generate_corpus
    generate_corpus(num_utterances=200, model="gemma3:1b")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _add_ems_to_path(ems_root: Path) -> None:
    root_str = str(ems_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def check_ollama_running() -> bool:
    """Return True if the Ollama server is reachable on localhost:11434."""
    try:
        import ollama
        ollama.list()   # raises if server is not running
        return True
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    """Return names of locally available Ollama models."""
    try:
        import ollama
        return [m["name"] for m in ollama.list().get("models", [])]
    except Exception:
        return []


def pull_model_if_missing(model: str) -> None:
    """Pull *model* from Ollama registry if not already downloaded."""
    available = list_ollama_models()
    # Normalize: 'llama3.1:8b' matches 'llama3.1:8b' in the list
    if not any(model in m for m in available):
        logger.info("Model '%s' not found locally — pulling from Ollama registry...", model)
        try:
            import ollama
            ollama.pull(model)
            logger.info("Model '%s' pulled successfully.", model)
        except Exception as exc:
            logger.error("Failed to pull '%s': %s", model, exc)
            raise


def generate_corpus(
    num_utterances: int = 200,
    model: str = "gemma3:1b",
    output_path: Optional[Path] = None,
    seed: int = 42,
    ems_root: Optional[Path] = None,
    auto_pull: bool = True,
) -> Path:
    """
    Generate an EMS dialogue corpus using a local Ollama model and save
    it as a JSONL manifest to *output_path*.

    Parameters
    ----------
    num_utterances : Number of utterances to generate.
    model          : Ollama model tag. Default: 'gemma3:1b'.
    output_path    : Destination JSONL file (default: data/manifests/corpus.jsonl).
    seed           : Random seed for reproducibility.
    ems_root       : Path to ems_audio_syn_pipeline repo root. Auto-detected
                     from PipelineConfig if not provided.
    auto_pull      : Automatically pull model if not already downloaded.

    Returns
    -------
    Path to the written JSONL corpus file.
    """
    # Resolve ems_root
    if ems_root is None:
        from pipeline.config import PipelineConfig
        ems_root = PipelineConfig().ems_pipeline_root

    # Resolve output path
    if output_path is None:
        from pipeline.config import PipelineConfig
        output_path = PipelineConfig().manifests_dir / "corpus.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate Ollama is running
    if not check_ollama_running():
        raise RuntimeError(
            "Ollama server is not running.\n"
            "Start it with:  ollama serve\n"
            "Or install from: https://ollama.com/download"
        )

    # Pull model if needed
    if auto_pull:
        pull_model_if_missing(model)

    # Delegate to Repo 1's generator
    _add_ems_to_path(ems_root)
    try:
        from phase1_text_generation.llm_ems_dialogue_generator import generate_utterances  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"Could not import llm_ems_dialogue_generator from {ems_root}: {exc}"
        ) from exc

    logger.info(
        "Generating %d EMS utterances with Ollama model '%s' → %s",
        num_utterances, model, output_path,
    )
    generate_utterances(
        num_utterances=num_utterances,
        provider="ollama",
        model=model,
        output_path=str(output_path),
        seed=seed,
    )
    logger.info("Corpus saved: %s", output_path)
    return output_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate EMS corpus with Ollama (free, local).")
    parser.add_argument("--num", type=int, default=200, help="Number of utterances")
    parser.add_argument(
        "--model", type=str, default="gemma3:1b",
        help="Ollama model tag (e.g. gemma3:1b, llama3.1:8b, mistral:7b, llama3.2:3b)"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = generate_corpus(
        num_utterances=args.num,
        model=args.model,
        output_path=args.output,
        seed=args.seed,
    )
    print(f"Done. Corpus written to: {out}")

