"""
pipeline/transcription.py
Adapter layer that wraps Repository 2 (C3Imaging/speech-augmentation) ASR
utilities for use in the Speak AI Pipeline.

Supported backends (auto-selected based on availability):
  1. openai-whisper  — local inference, best quality, no API key needed
  2. HuggingFace Transformers whisper pipeline  — alternative local inference
  3. Wav2Vec2        — Repository 2 wav2vec2_infer_custom.py

Word-level timestamps are exposed when the backend supports them.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WordTiming:
    word: str
    start: float  # seconds
    end: float    # seconds


@dataclass
class TranscriptionResult:
    audio_path: Path
    text: str
    language: Optional[str] = None
    word_timings: list[WordTiming] = field(default_factory=list)
    backend: str = "unknown"
    confidence: Optional[float] = None


def _add_speech_aug_to_path(speech_aug_root: Path) -> None:
    root_str = str(speech_aug_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


# ── Backend: openai-whisper ───────────────────────────────────────────────────

def _transcribe_openai_whisper(
    audio_path: Path,
    model_name: str = "small",
    language: Optional[str] = None,
    word_timestamps: bool = True,
) -> TranscriptionResult:
    import whisper  # type: ignore

    import soundfile as sf
    import numpy as np

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    result = model.transcribe(audio, **kwargs)
    text = result["text"].strip()
    lang = result.get("language")

    timings: list[WordTiming] = []
    if word_timestamps and "segments" in result:
        for seg in result["segments"]:
            for w in seg.get("words", []):
                timings.append(
                    WordTiming(word=w["word"].strip(), start=w["start"], end=w["end"])
                )

    return TranscriptionResult(
        audio_path=audio_path,
        text=text,
        language=lang,
        word_timings=timings,
        backend=f"openai-whisper/{model_name}",
    )


# ── Backend: HuggingFace Transformers Whisper pipeline ───────────────────────

def _transcribe_hf_whisper(
    audio_path: Path,
    model_name: str = "openai/whisper-small",
) -> TranscriptionResult:
    from transformers import pipeline as hf_pipeline  # type: ignore

    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model_name,
        return_timestamps="word",
        chunk_length_s=30,
    )
    result = pipe(str(audio_path))
    text = result["text"].strip()

    timings: list[WordTiming] = []
    for chunk in result.get("chunks", []):
        ts = chunk.get("timestamp", (None, None))
        if ts[0] is not None:
            timings.append(WordTiming(word=chunk["text"].strip(), start=ts[0], end=ts[1] or ts[0]))

    return TranscriptionResult(
        audio_path=audio_path,
        text=text,
        word_timings=timings,
        backend=f"hf-transformers/{model_name}",
    )


# ── Backend: Wav2Vec2 via Repository 2 ───────────────────────────────────────

def _transcribe_wav2vec2(
    audio_path: Path,
    speech_aug_root: Path,
    model_name: str = "facebook/wav2vec2-base-960h",
) -> TranscriptionResult:
    _add_speech_aug_to_path(speech_aug_root)
    try:
        from asr.wav2vec2_infer_custom import infer_single  # type: ignore

        text = infer_single(str(audio_path), model_name=model_name)
        return TranscriptionResult(
            audio_path=audio_path,
            text=text,
            backend=f"wav2vec2/{model_name}",
        )
    except ImportError as exc:
        raise RuntimeError(
            f"Wav2Vec2 backend requires Repository 2 to be available at {speech_aug_root}: {exc}"
        ) from exc


# ── Public API ────────────────────────────────────────────────────────────────

BACKENDS = ("openai-whisper", "hf-whisper", "wav2vec2")


def transcribe(
    audio_path: Path,
    backend: str = "openai-whisper",
    model_name: Optional[str] = None,
    speech_aug_root: Optional[Path] = None,
    language: Optional[str] = None,
    word_timestamps: bool = True,
) -> TranscriptionResult:
    """
    Transcribe a single audio file using the specified backend.

    Parameters
    ----------
    audio_path:       Path to a 16 kHz mono WAV file.
    backend:          One of 'openai-whisper', 'hf-whisper', 'wav2vec2'.
    model_name:       Model identifier (backend-specific default used if None).
    speech_aug_root:  Root of Repository 2 — required for wav2vec2 backend.
    language:         ISO 639-1 language code hint (whisper backends only).
    word_timestamps:  Request per-word timestamps where supported.
    """
    if backend == "openai-whisper":
        return _transcribe_openai_whisper(
            audio_path,
            model_name=model_name or "small",
            language=language,
            word_timestamps=word_timestamps,
        )
    elif backend == "hf-whisper":
        return _transcribe_hf_whisper(
            audio_path,
            model_name=model_name or "openai/whisper-small",
        )
    elif backend == "wav2vec2":
        if speech_aug_root is None:
            raise ValueError("speech_aug_root must be provided for the wav2vec2 backend.")
        return _transcribe_wav2vec2(
            audio_path,
            speech_aug_root=speech_aug_root,
            model_name=model_name or "facebook/wav2vec2-base-960h",
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {BACKENDS}")


def transcribe_directory(
    input_dir: Path,
    output_dir: Path,
    extensions: tuple[str, ...] = (".wav",),
    **transcribe_kwargs,
) -> list[TranscriptionResult]:
    """
    Transcribe all audio files in *input_dir*, saving JSON sidecar files
    to *output_dir*.

    Returns a list of TranscriptionResult objects.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[TranscriptionResult] = []
    files = [f for f in sorted(input_dir.iterdir()) if f.suffix.lower() in extensions]

    if not files:
        logger.warning("No audio files found in %s", input_dir)
        return results

    for audio_file in files:
        try:
            result = transcribe(audio_file, **transcribe_kwargs)
        except Exception as exc:
            logger.error("Transcription failed for %s: %s", audio_file, exc)
            continue

        # Save JSON sidecar
        sidecar_path = output_dir / (audio_file.stem + ".json")
        sidecar_data = {
            "audio_path": str(result.audio_path),
            "text": result.text,
            "language": result.language,
            "backend": result.backend,
            "word_timings": [
                {"word": w.word, "start": w.start, "end": w.end}
                for w in result.word_timings
            ],
        }
        sidecar_path.write_text(json.dumps(sidecar_data, indent=2, ensure_ascii=False))
        results.append(result)
        logger.info("Transcribed %s → %s", audio_file.name, sidecar_path.name)

    logger.info("Transcribed %d / %d files.", len(results), len(files))
    return results

