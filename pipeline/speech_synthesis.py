"""
pipeline/speech_synthesis.py
Adapter layer that wraps Repository 1 (ChiShengChen/ems_audio_syn_pipeline)
TTS backends for use in the Speak AI Pipeline.

Supported TTS engines (delegated to Repository 1):
  - XTTS v2 (voice cloning)   → phase2_speech_synthesis/xtts_voice_cloning.py
  - Bark                       → phase2_speech_synthesis/bark_synthesis.py
  - Edge TTS (Microsoft)       → phase2_speech_synthesis/edge_tts_synthesis.py

The TTS orchestrator from Repository 1 is also exposed directly.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _add_ems_to_path(ems_root: Path) -> None:
    root_str = str(ems_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


# ── Edge TTS (lightweight, no GPU, good quality) ─────────────────────────────

def synthesise_edge_tts(
    text: str,
    output_path: Path,
    ems_root: Path,
    voice: str = "en-US-AriaNeural",
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> Path:
    """
    Synthesise *text* with Edge TTS (via Repository 1) and write to *output_path*.
    """
    _add_ems_to_path(ems_root)
    try:
        from phase2_speech_synthesis.edge_tts_synthesis import synthesize_edge_tts  # type: ignore

        synthesize_edge_tts(
            text=text,
            output_path=str(output_path),
            voice=voice,
            rate=rate,
            pitch=pitch,
        )
    except ImportError:
        # Fallback: call edge-tts directly
        import asyncio
        import edge_tts  # type: ignore

        async def _run():
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            await communicate.save(str(output_path))

        asyncio.run(_run())

    logger.info("Edge TTS → %s", output_path)
    return output_path


# ── XTTS v2 (voice cloning) ───────────────────────────────────────────────────

def synthesise_xtts(
    text: str,
    output_path: Path,
    speaker_wav: Path,
    ems_root: Path,
    language: str = "en",
) -> Path:
    """
    Clone the voice from *speaker_wav* and synthesise *text* with XTTS v2.
    """
    _add_ems_to_path(ems_root)
    from phase2_speech_synthesis.xtts_voice_cloning import synthesize_xtts  # type: ignore

    synthesize_xtts(
        text=text,
        output_path=str(output_path),
        speaker_wav=str(speaker_wav),
        language=language,
    )
    logger.info("XTTS → %s", output_path)
    return output_path


# ── Bark (expressive, slow, GPU recommended) ──────────────────────────────────

def synthesise_bark(
    text: str,
    output_path: Path,
    ems_root: Path,
    voice_preset: str = "v2/en_speaker_6",
) -> Path:
    """
    Synthesise *text* with Bark via Repository 1.
    """
    _add_ems_to_path(ems_root)
    from phase2_speech_synthesis.bark_synthesis import synthesize_bark  # type: ignore

    synthesize_bark(
        text=text,
        output_path=str(output_path),
        voice_preset=voice_preset,
    )
    logger.info("Bark → %s", output_path)
    return output_path


# ── TTS Orchestrator (Repository 1) ──────────────────────────────────────────

def synthesise_with_orchestrator(
    texts: list[str],
    output_dir: Path,
    ems_root: Path,
    engine: str = "edge_tts",
    **engine_kwargs,
) -> list[Path]:
    """
    Delegate to Repository 1's TTS orchestrator for bulk synthesis.

    engine: one of 'edge_tts', 'xtts', 'bark'.
    Returns list of output WAV paths.
    """
    _add_ems_to_path(ems_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from phase2_speech_synthesis.tts_orchestrator import TTSOrchestrator  # type: ignore

        orchestrator = TTSOrchestrator(engine=engine, output_dir=str(output_dir), **engine_kwargs)
        paths = orchestrator.synthesize_batch(texts)
        return [Path(p) for p in paths]
    except ImportError as exc:
        logger.warning("TTSOrchestrator not available (%s). Falling back to per-item Edge TTS.", exc)

    # Fallback: per-item edge-tts synthesis
    results: list[Path] = []
    for i, text in enumerate(texts):
        out = output_dir / f"synth_{i:05d}.wav"
        synthesise_edge_tts(text, out, ems_root, **engine_kwargs)
        results.append(out)
    return results


# ── High-level: synthesise from manifest ─────────────────────────────────────

def synthesise_from_manifest(
    manifest_path: Path,
    output_dir: Path,
    ems_root: Path,
    engine: str = "edge_tts",
    text_key: str = "text",
    **engine_kwargs,
) -> list[Path]:
    """
    Read a JSONL manifest file and synthesise audio for each entry.

    Each manifest line must contain at least the *text_key* field.
    Writes WAV files and returns their paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = [e[text_key] for e in entries if text_key in e]
    logger.info("Synthesising %d utterances from %s …", len(texts), manifest_path.name)
    return synthesise_with_orchestrator(texts, output_dir, ems_root, engine=engine, **engine_kwargs)

