"""
pipeline/config.py
Centralised configuration dataclass for the Speak AI Pipeline.
All paths and hyper-parameters are resolved from .env (via python-dotenv)
with sane defaults so the pipeline can run without any manual edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env if present; never raises


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


@dataclass
class PipelineConfig:
    # ── Repository roots ────────────────────────────────────────────────────
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    @property
    def ems_pipeline_root(self) -> Path:
        return self.repo_root / "third_party" / "ems_audio_syn_pipeline"

    @property
    def speech_aug_root(self) -> Path:
        return self.repo_root / "third_party" / "speech-augmentation"

    # ── Input paths ─────────────────────────────────────────────────────────
    input_audio_dir: Path = field(
        default_factory=lambda: Path(_env("INPUT_AUDIO_DIR", "./input_audio/original"))
    )
    noise_samples_dir: Path = field(
        default_factory=lambda: Path(_env("NOISE_SAMPLES_DIR", "./input_audio/noise_samples"))
    )

    # ── Output / intermediate paths ──────────────────────────────────────────
    processed_dir: Path = field(
        default_factory=lambda: Path(_env("PROCESSED_DIR", "./data/processed"))
    )
    transcripts_dir: Path = field(
        default_factory=lambda: Path(_env("TRANSCRIPTS_DIR", "./data/transcripts"))
    )
    augmented_dir: Path = field(
        default_factory=lambda: Path(_env("AUGMENTED_DIR", "./data/augmented"))
    )
    synthetic_dir: Path = field(
        default_factory=lambda: Path(_env("SYNTHETIC_DIR", "./data/synthetic"))
    )
    manifests_dir: Path = field(
        default_factory=lambda: Path(_env("MANIFESTS_DIR", "./data/manifests"))
    )
    whisper_output_dir: Path = field(
        default_factory=lambda: Path(
            _env("WHISPER_OUTPUT_DIR", "./outputs/final_dataset/whisper_finetuned")
        )
    )

    # ── Model settings ───────────────────────────────────────────────────────
    whisper_base_model: str = field(
        default_factory=lambda: _env("WHISPER_BASE_MODEL", "openai/whisper-tiny")
    )
    whisper_max_steps: int = field(
        default_factory=lambda: _env_int("WHISPER_MAX_STEPS", 2000)
    )
    whisper_batch_size: int = field(
        default_factory=lambda: _env_int("WHISPER_BATCH_SIZE", 4)
    )
    whisper_learning_rate: float = field(
        default_factory=lambda: _env_float("WHISPER_LEARNING_RATE", 1e-4)
    )
    whisper_num_epochs: int = field(
        default_factory=lambda: _env_int("WHISPER_NUM_EPOCHS", 3)
    )
    whisper_cpu_threads: int = field(
        default_factory=lambda: _env_int("WHISPER_CPU_THREADS", 2)
    )
    xtts_model_path: str = field(
        default_factory=lambda: _env("XTTS_MODEL_PATH", "./hf_cache/xtts_v2")
    )
    edge_tts_voice: str = field(
        default_factory=lambda: _env("EDGE_TTS_VOICE", "en-US-AriaNeural")
    )

    # ── Pipeline behaviour ───────────────────────────────────────────────────
    max_workers: int = field(default_factory=lambda: _env_int("MAX_WORKERS", 4))
    seed: int = field(default_factory=lambda: _env_int("SEED", 42))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # ── Ollama (FREE local LLM — no API key needed) ──────────────────────────
    llm_provider: str = field(
        default_factory=lambda: _env("LLM_PROVIDER", "ollama")
    )
    ollama_base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: _env("OLLAMA_MODEL", "gemma3:1b")
    )

    @property
    def hf_token(self) -> str:
        return os.environ.get("HF_TOKEN", "")

    def ensure_dirs(self) -> None:
        """Create all output directories if they do not exist."""
        for attr in (
            "processed_dir",
            "transcripts_dir",
            "augmented_dir",
            "synthetic_dir",
            "manifests_dir",
            "whisper_output_dir",
        ):
            path: Path = getattr(self, attr)
            path.mkdir(parents=True, exist_ok=True)

