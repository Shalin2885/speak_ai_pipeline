"""
pipeline/audio_preprocessing.py
Pre-processing utilities applied to raw input recordings before any
augmentation or transcription step.

Steps performed:
  1. Resample to target sample rate (default 16 kHz — required by Whisper).
  2. Convert to mono.
  3. Normalise loudness (peak normalisation to -3 dBFS).
  4. Trim leading / trailing silence.
  5. Write the cleaned file to data/processed/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

TARGET_SR = 16_000  # Hz — Whisper and Wav2Vec2 both expect 16 kHz
SILENCE_THRESHOLD_DB = -40.0  # dB below peak to consider silence
PEAK_TARGET_DBFS = -3.0  # dBFS target after normalisation


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Average channels if stereo/multi-channel."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio array using scipy (avoids heavy librosa import)."""
    if orig_sr == target_sr:
        return audio
    try:
        import librosa  # preferred: high-quality resampling
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(orig_sr, target_sr)
        return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def _peak_normalise(audio: np.ndarray, target_dbfs: float = PEAK_TARGET_DBFS) -> np.ndarray:
    peak = np.abs(audio).max()
    if peak == 0:
        return audio
    target_linear = 10 ** (target_dbfs / 20.0)
    return (audio / peak) * target_linear


def _trim_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = SILENCE_THRESHOLD_DB,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    try:
        import librosa
        trimmed, _ = librosa.effects.trim(
            audio,
            top_db=abs(threshold_db),
            frame_length=frame_length,
            hop_length=hop_length,
        )
        return trimmed
    except ImportError:
        return audio  # gracefully skip if librosa unavailable


def preprocess_audio(
    input_path: Path,
    output_dir: Path,
    target_sr: int = TARGET_SR,
    normalise: bool = True,
    trim: bool = True,
    overwrite: bool = False,
) -> Optional[Path]:
    """
    Pre-process a single audio file and write the result to *output_dir*.

    Returns the output path, or None if the file was skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name

    if output_path.exists() and not overwrite:
        logger.debug("Skipping (already processed): %s", output_path)
        return output_path

    try:
        audio, orig_sr = sf.read(str(input_path), dtype="float32", always_2d=False)
    except Exception as exc:
        logger.error("Failed to read %s: %s", input_path, exc)
        return None

    audio = _to_mono(audio)
    audio = _resample(audio, orig_sr, target_sr)

    if normalise:
        audio = _peak_normalise(audio)

    if trim:
        audio = _trim_silence(audio, target_sr)

    sf.write(str(output_path), audio, target_sr, subtype="PCM_16")
    logger.info("Preprocessed → %s", output_path)
    return output_path


def preprocess_directory(
    input_dir: Path,
    output_dir: Path,
    extensions: tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg"),
    **kwargs,
) -> list[Path]:
    """
    Pre-process all audio files in *input_dir* and write results to *output_dir*.

    Returns a list of successfully written output paths.
    """
    results: list[Path] = []
    files = [f for f in sorted(input_dir.iterdir()) if f.suffix.lower() in extensions]
    if not files:
        logger.warning("No audio files found in %s", input_dir)
        return results

    for f in files:
        out = preprocess_audio(f, output_dir, **kwargs)
        if out:
            results.append(out)

    logger.info("Preprocessed %d / %d files.", len(results), len(files))
    return results

