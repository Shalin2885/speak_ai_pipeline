"""
tests/test_dataset_builder.py
Unit tests for pipeline/dataset_builder.py.
Tests use temporary directories and synthetic JSON sidecars.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline.dataset_builder import build_manifest, split_manifest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_wav(path: Path, duration_s: float = 1.0, sr: int = 16_000) -> Path:
    audio = np.zeros(int(sr * duration_s), dtype=np.float32)
    sf.write(str(path), audio, sr)
    return path


def _make_sidecar(
    transcript_dir: Path,
    audio_path: Path,
    text: str,
    stem: str | None = None,
) -> Path:
    stem = stem or audio_path.stem
    sidecar = transcript_dir / f"{stem}.json"
    sidecar.write_text(
        json.dumps(
            {
                "audio_filepath": str(audio_path),
                "audio_path": str(audio_path),
                "text": text,
                "backend": "openai-whisper/small",
                "language": "en",
                "word_timings": [],
            },
            ensure_ascii=False,
        )
    )
    return sidecar


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def dataset_dirs(tmp_path: Path):
    audio_dir = tmp_path / "processed"
    audio_dir.mkdir()
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()

    # Create 3 audio + sidecar pairs
    for i in range(3):
        wav = _make_wav(audio_dir / f"utt_{i:02d}.wav")
        _make_sidecar(transcripts_dir, wav, text=f"utterance number {i}")

    return {
        "audio_dir": audio_dir,
        "transcripts_dir": transcripts_dir,
        "manifests_dir": manifests_dir,
    }


# ── build_manifest ────────────────────────────────────────────────────────────

class TestBuildManifest:
    def test_creates_jsonl_file(self, dataset_dirs):
        out = build_manifest(
            transcripts_dir=dataset_dirs["transcripts_dir"],
            manifests_dir=dataset_dirs["manifests_dir"],
        )
        assert out.exists()
        assert out.suffix == ".jsonl"

    def test_correct_entry_count(self, dataset_dirs):
        out = build_manifest(
            transcripts_dir=dataset_dirs["transcripts_dir"],
            manifests_dir=dataset_dirs["manifests_dir"],
        )
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == 3

    def test_entries_have_required_fields(self, dataset_dirs):
        out = build_manifest(
            transcripts_dir=dataset_dirs["transcripts_dir"],
            manifests_dir=dataset_dirs["manifests_dir"],
        )
        for line in out.read_text().splitlines():
            entry = json.loads(line)
            assert "audio_filepath" in entry
            assert "text" in entry
            assert "duration" in entry
            assert entry["duration"] > 0

    def test_raises_when_no_entries(self, tmp_path):
        empty = tmp_path / "empty_transcripts"
        empty.mkdir()
        manifests = tmp_path / "manifests"
        manifests.mkdir()
        with pytest.raises(ValueError, match="zero entries"):
            build_manifest(transcripts_dir=empty, manifests_dir=manifests)

    def test_quarantine_files_excluded(self, dataset_dirs, tmp_path):
        """Sidecars inside the quarantine subdir must not be included."""
        quarantine = dataset_dirs["transcripts_dir"] / "quarantine"
        quarantine.mkdir()
        # Put a sidecar in quarantine (should be excluded)
        wav = _make_wav(tmp_path / "quarantined.wav")
        q_sidecar = quarantine / "quarantined.json"
        q_sidecar.write_text(
            json.dumps({"audio_filepath": str(wav), "audio_path": str(wav), "text": "bad", "word_timings": []})
        )

        out = build_manifest(
            transcripts_dir=dataset_dirs["transcripts_dir"],
            manifests_dir=dataset_dirs["manifests_dir"],
        )
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        # Only the 3 valid entries, not the quarantined one
        assert len(lines) == 3


# ── split_manifest ────────────────────────────────────────────────────────────

class TestSplitManifest:
    @pytest.fixture()
    def sample_manifest(self, tmp_path: Path) -> Path:
        """Create a 10-entry dummy manifest."""
        manifest = tmp_path / "dataset.jsonl"
        lines = [
            json.dumps({"audio_filepath": f"/audio/{i}.wav", "text": f"sentence {i}", "duration": 1.0})
            for i in range(10)
        ]
        manifest.write_text("\n".join(lines) + "\n")
        return manifest

    def test_creates_three_split_files(self, sample_manifest, tmp_path):
        splits = split_manifest(sample_manifest, tmp_path, train_ratio=0.7, val_ratio=0.2)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        for p in splits.values():
            assert p.exists()

    def test_total_entries_conserved(self, sample_manifest, tmp_path):
        splits = split_manifest(sample_manifest, tmp_path, train_ratio=0.7, val_ratio=0.2)
        total = sum(
            len([l for l in p.read_text().splitlines() if l.strip()])
            for p in splits.values()
        )
        assert total == 10

    def test_train_is_largest_split(self, sample_manifest, tmp_path):
        splits = split_manifest(sample_manifest, tmp_path, train_ratio=0.8, val_ratio=0.1)
        n_train = len([l for l in splits["train"].read_text().splitlines() if l.strip()])
        n_val = len([l for l in splits["val"].read_text().splitlines() if l.strip()])
        n_test = len([l for l in splits["test"].read_text().splitlines() if l.strip()])
        assert n_train >= n_val
        assert n_train >= n_test

    def test_deterministic_with_same_seed(self, sample_manifest, tmp_path):
        splits_a = split_manifest(sample_manifest, tmp_path / "a", seed=0)
        splits_b = split_manifest(sample_manifest, tmp_path / "b", seed=0)
        assert splits_a["train"].read_text() == splits_b["train"].read_text()

    def test_different_seeds_produce_different_splits(self, sample_manifest, tmp_path):
        splits_a = split_manifest(sample_manifest, tmp_path / "a", seed=0)
        splits_b = split_manifest(sample_manifest, tmp_path / "b", seed=99)
        # Very unlikely to be identical for 10 entries with different seeds
        assert splits_a["train"].read_text() != splits_b["train"].read_text()

