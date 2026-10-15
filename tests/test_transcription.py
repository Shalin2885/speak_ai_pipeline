"""
tests/test_transcription.py
Unit tests for pipeline/transcription.py.

Tests run without a GPU or real audio files by mocking the ASR backends.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from pipeline.transcription import (
    TranscriptionResult,
    WordTiming,
    transcribe,
    transcribe_directory,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_audio(tmp_path: Path) -> Path:
    """Write a 1-second silent WAV file and return its path."""
    audio = np.zeros(16_000, dtype=np.float32)
    wav_path = tmp_path / "test_audio.wav"
    sf.write(str(wav_path), audio, 16_000)
    return wav_path


@pytest.fixture()
def dummy_result(tmp_audio: Path) -> TranscriptionResult:
    return TranscriptionResult(
        audio_path=tmp_audio,
        text="Hello world",
        language="en",
        word_timings=[
            WordTiming(word="Hello", start=0.0, end=0.5),
            WordTiming(word="world", start=0.5, end=1.0),
        ],
        backend="openai-whisper/small",
    )


# ── TranscriptionResult dataclass ─────────────────────────────────────────────

class TestTranscriptionResult:
    def test_defaults(self, tmp_audio: Path):
        result = TranscriptionResult(audio_path=tmp_audio, text="test")
        assert result.language is None
        assert result.word_timings == []
        assert result.backend == "unknown"
        assert result.confidence is None

    def test_word_timings_populated(self, dummy_result: TranscriptionResult):
        assert len(dummy_result.word_timings) == 2
        assert dummy_result.word_timings[0].word == "Hello"
        assert dummy_result.word_timings[1].end == pytest.approx(1.0)


# ── transcribe() — mocked backends ───────────────────────────────────────────

class TestTranscribe:
    @patch("pipeline.transcription._transcribe_openai_whisper")
    def test_openai_whisper_backend_called(self, mock_fn, tmp_audio):
        mock_fn.return_value = TranscriptionResult(audio_path=tmp_audio, text="ok")
        result = transcribe(tmp_audio, backend="openai-whisper", model_name="tiny")
        mock_fn.assert_called_once()
        assert result.text == "ok"

    @patch("pipeline.transcription._transcribe_hf_whisper")
    def test_hf_whisper_backend_called(self, mock_fn, tmp_audio):
        mock_fn.return_value = TranscriptionResult(audio_path=tmp_audio, text="hf ok")
        result = transcribe(tmp_audio, backend="hf-whisper")
        mock_fn.assert_called_once()
        assert result.text == "hf ok"

    def test_unknown_backend_raises(self, tmp_audio):
        with pytest.raises(ValueError, match="Unknown backend"):
            transcribe(tmp_audio, backend="invalid-backend")

    def test_wav2vec2_without_root_raises(self, tmp_audio):
        with pytest.raises(ValueError, match="speech_aug_root"):
            transcribe(tmp_audio, backend="wav2vec2")


# ── transcribe_directory() ─────────────────────────────────────────────────────

class TestTranscribeDirectory:
    @patch("pipeline.transcription.transcribe")
    def test_transcribes_all_wav_files(self, mock_transcribe, tmp_path):
        # Create two dummy wav files
        for name in ("a.wav", "b.wav"):
            wav = tmp_path / name
            sf.write(str(wav), np.zeros(16_000, dtype=np.float32), 16_000)

        mock_transcribe.side_effect = lambda p, **kw: TranscriptionResult(
            audio_path=p, text=f"text for {p.name}"
        )

        out_dir = tmp_path / "transcripts"
        results = transcribe_directory(tmp_path, out_dir, backend="openai-whisper")

        assert len(results) == 2
        assert (out_dir / "a.json").exists()
        assert (out_dir / "b.json").exists()

    @patch("pipeline.transcription.transcribe")
    def test_json_sidecar_content(self, mock_transcribe, tmp_path):
        wav = tmp_path / "sample.wav"
        sf.write(str(wav), np.zeros(16_000, dtype=np.float32), 16_000)
        mock_transcribe.return_value = TranscriptionResult(
            audio_path=wav, text="the quick brown fox", backend="openai-whisper/small"
        )

        out_dir = tmp_path / "out"
        transcribe_directory(tmp_path, out_dir)

        sidecar = json.loads((out_dir / "sample.json").read_text())
        assert sidecar["text"] == "the quick brown fox"
        assert sidecar["backend"] == "openai-whisper/small"

    def test_empty_directory_returns_empty_list(self, tmp_path):
        out_dir = tmp_path / "out"
        results = transcribe_directory(tmp_path, out_dir)
        assert results == []
