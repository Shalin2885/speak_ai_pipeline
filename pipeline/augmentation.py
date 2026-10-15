"""
pipeline/augmentation.py
Adapter layer that wraps Repository 2 (C3Imaging/speech-augmentation)
audio augmentation utilities for use in the Speak AI Pipeline.

It bridges:
  - CLEESE pitch/speed/reverb augmentation   → audio_augmentation/cleese_audio_augmentation.py
  - audiomentations-based augmentation        → simple augmentation chains
  - noise injection from input_audio/noise_samples/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def _add_speech_aug_to_path(speech_aug_root: Path) -> None:
    """Ensure the speech-augmentation repo root is importable."""
    root_str = str(speech_aug_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


# ── Simple noise injection (no external dep needed) ──────────────────────────

def inject_noise(
    audio: np.ndarray,
    noise: np.ndarray,
    snr_db: float = 15.0,
) -> np.ndarray:
    """
    Mix *noise* into *audio* at the requested signal-to-noise ratio.

    Both arrays must be float32 mono at the same sample rate.
    """
    audio_rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-9
    target_noise_rms = audio_rms / (10 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / noise_rms)

    # Trim or tile noise to match audio length
    if len(scaled_noise) < len(audio):
        repeats = int(np.ceil(len(audio) / len(scaled_noise)))
        scaled_noise = np.tile(scaled_noise, repeats)
    scaled_noise = scaled_noise[: len(audio)]

    return np.clip(audio + scaled_noise, -1.0, 1.0)


# ── audiomentations chain ─────────────────────────────────────────────────────

def build_audiomentations_augmenter(p: float = 0.5):
    """
    Build a standard augmentation chain using the *audiomentations* library.

    Returns None if the library is not installed (graceful degradation).
    """
    try:
        from audiomentations import (
            Compose,
            AddGaussianNoise,
            TimeStretch,
            PitchShift,
            Shift,
        )

        return Compose(
            [
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=p),
                TimeStretch(min_rate=0.9, max_rate=1.1, p=p),
                PitchShift(min_semitones=-2, max_semitones=2, p=p),
                Shift(min_shift=-0.1, max_shift=0.1, p=p),
            ]
        )
    except ImportError:
        logger.warning(
            "audiomentations not installed. Skipping audiomentations augmentation chain."
        )
        return None


# ── CLEESE-based augmentation (Repository 2) ─────────────────────────────────

def augment_with_cleese(
    input_path: Path,
    output_dir: Path,
    speech_aug_root: Path,
    num_variants: int = 3,
) -> list[Path]:
    """
    Apply CLEESE pitch/time augmentation (from Repository 2) to *input_path*.

    Writes *num_variants* augmented WAVs to *output_dir* and returns their paths.
    Falls back gracefully if CLEESE is not importable.
    """
    _add_speech_aug_to_path(speech_aug_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []

    try:
        from audio_augmentation.cleese_audio_augmentation import augment_file  # type: ignore

        for i in range(num_variants):
            out_path = output_dir / f"{input_path.stem}_cleese_{i:02d}.wav"
            augment_file(str(input_path), str(out_path))
            if out_path.exists():
                results.append(out_path)
                logger.info("CLEESE variant %d → %s", i, out_path)
    except ImportError as exc:
        logger.warning("CLEESE not available (%s). Skipping CLEESE augmentation.", exc)

    return results


# ── High-level directory augmentation ────────────────────────────────────────

def augment_directory(
    input_dir: Path,
    output_dir: Path,
    noise_dir: Optional[Path] = None,
    snr_db: float = 15.0,
    speech_aug_root: Optional[Path] = None,
    use_audiomentations: bool = True,
    use_cleese: bool = False,
    extensions: tuple[str, ...] = (".wav",),
) -> list[Path]:
    """
    Augment all audio files in *input_dir* and write to *output_dir*.

    Applies (in order, as configured):
      1. Audiomentations chain (pitch / time / noise)
      2. Noise injection from *noise_dir*
      3. CLEESE augmentation (requires speech_aug_root)

    Returns list of output file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    augmenter = build_audiomentations_augmenter() if use_audiomentations else None

    noise_files: list[Path] = []
    if noise_dir and noise_dir.exists():
        noise_files = [f for f in noise_dir.iterdir() if f.suffix.lower() in extensions]

    results: list[Path] = []
    audio_files = [f for f in sorted(input_dir.iterdir()) if f.suffix.lower() in extensions]

    for audio_file in audio_files:
        try:
            audio, sr = sf.read(str(audio_file), dtype="float32", always_2d=False)
        except Exception as exc:
            logger.error("Cannot read %s: %s", audio_file, exc)
            continue

        processed = audio.copy()

        # 1. audiomentations
        if augmenter is not None:
            processed = augmenter(samples=processed, sample_rate=sr)

        # 2. Noise injection
        if noise_files:
            import random
            noise_path = random.choice(noise_files)
            try:
                noise, noise_sr = sf.read(str(noise_path), dtype="float32", always_2d=False)
                if noise.ndim > 1:
                    noise = noise.mean(axis=1)
                processed = inject_noise(processed, noise, snr_db=snr_db)
            except Exception as exc:
                logger.warning("Noise injection failed for %s: %s", noise_path, exc)

        out_path = output_dir / audio_file.name
        sf.write(str(out_path), processed, sr, subtype="PCM_16")
        results.append(out_path)
        logger.info("Augmented → %s", out_path)

        # 3. CLEESE variants
        if use_cleese and speech_aug_root:
            cleese_results = augment_with_cleese(audio_file, output_dir, speech_aug_root)
            results.extend(cleese_results)

    logger.info("Augmented %d files → %s", len(results), output_dir)
    return results

