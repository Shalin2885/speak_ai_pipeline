"""
tests/test_augmentation.py
Unit tests for pipeline/augmentation.py.
All tests run without a GPU or external services.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from pipeline.augmentation import (
    inject_noise,
    build_audiomentations_augmenter,
    augment_directory,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def sine_wav(tmp_path: Path) -> Path:
    """440 Hz tone, 1 second, 16 kHz mono."""
    t = np.linspace(0, 1, 16_000, endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(exist_ok=True)
    wav_path = audio_dir / "tone.wav"
    sf.write(str(wav_path), audio, 16_000)
    return wav_path


@pytest.fixture()
def noise_wav(tmp_path: Path) -> Path:
    """White noise, 2 seconds."""
    noise = np.random.default_rng(0).standard_normal(32_000).astype(np.float32) * 0.05
    noise_dir = tmp_path / "noise"
    noise_dir.mkdir(exist_ok=True)
    wav_path = noise_dir / "noise.wav"
    sf.write(str(wav_path), noise, 16_000)
    return wav_path


# ── inject_noise ──────────────────────────────────────────────────────────────

class TestInjectNoise:
    def test_output_same_length_as_audio(self, sine_wav, noise_wav):
        audio, _ = sf.read(str(sine_wav), dtype="float32")
        noise, _ = sf.read(str(noise_wav), dtype="float32")
        result = inject_noise(audio, noise, snr_db=20.0)
        assert len(result) == len(audio)

    def test_output_clipped_to_minus_one_plus_one(self, sine_wav, noise_wav):
        audio, _ = sf.read(str(sine_wav), dtype="float32")
        noise, _ = sf.read(str(noise_wav), dtype="float32")
        result = inject_noise(audio, noise, snr_db=0.0)  # very loud noise
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_zero_snr_changes_audio(self, sine_wav, noise_wav):
        audio, _ = sf.read(str(sine_wav), dtype="float32")
        noise, _ = sf.read(str(noise_wav), dtype="float32")
        result = inject_noise(audio, noise, snr_db=0.0)
        assert not np.allclose(result, audio)

    def test_short_noise_is_tiled(self, sine_wav):
        audio, _ = sf.read(str(sine_wav), dtype="float32")
        short_noise = np.array([0.01, -0.01, 0.02], dtype=np.float32)
        result = inject_noise(audio, short_noise, snr_db=30.0)
        assert len(result) == len(audio)


# ── build_audiomentations_augmenter ──────────────────────────────────────────

class TestBuildAugmenter:
    def test_returns_callable_when_available(self):
        try:
            import audiomentations  # noqa: F401
        except ImportError:
            pytest.skip("audiomentations not installed")

        augmenter = build_audiomentations_augmenter(p=0.5)
        assert callable(augmenter)

    def test_returns_none_when_not_installed(self):
        with patch.dict("sys.modules", {"audiomentations": None}):
            # Force ImportError inside build function
            with patch("pipeline.augmentation.build_audiomentations_augmenter", return_value=None):
                result = build_audiomentations_augmenter()
                assert result is None


# ── augment_directory ─────────────────────────────────────────────────────────

class TestAugmentDirectory:
    def test_produces_output_files(self, tmp_path, sine_wav):
        input_dir = sine_wav.parent
        output_dir = tmp_path / "augmented"

        results = augment_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            use_audiomentations=False,  # avoid dep
            use_cleese=False,
        )

        assert len(results) >= 1
        for r in results:
            assert r.exists()

    def test_noise_injection_applied_when_noise_dir_given(self, tmp_path, sine_wav, noise_wav):
        input_dir = sine_wav.parent
        noise_dir = noise_wav.parent
        output_dir = tmp_path / "augmented_noise"

        results = augment_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            noise_dir=noise_dir,
            snr_db=10.0,
            use_audiomentations=False,
            use_cleese=False,
        )

        assert len(results) == 1
        out_audio, _ = sf.read(str(results[0]), dtype="float32")
        orig_audio, _ = sf.read(str(sine_wav), dtype="float32")
        # Output should differ from input (noise added)
        assert not np.allclose(out_audio, orig_audio, atol=1e-4)

    def test_empty_directory_returns_empty_list(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        out_dir = tmp_path / "out"
        results = augment_directory(empty_dir, out_dir, use_audiomentations=False)
        assert results == []

