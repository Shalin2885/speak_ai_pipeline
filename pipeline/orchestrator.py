"""
pipeline/orchestrator.py
End-to-end Speak AI Pipeline orchestrator.

Stage execution order:
  1. preprocess   — normalise & resample raw input audio
  2. transcribe   — ASR → JSON sidecar files
  3. validate     — quality-gate transcripts (quarantine bad ones)
  4. augment      — apply audio augmentation to preprocessed audio
  5. synthesise   — TTS from corpus / text manifests (optional)
  6. build        — assemble Whisper-ready JSONL manifest
  7. split        — train / val / test split

Each stage is idempotent: re-running skips already-produced outputs.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from .config import PipelineConfig
from .audio_preprocessing import preprocess_directory
from .transcription import transcribe_directory
from .transcript_validation import validate_batch
from .augmentation import augment_directory
from .speech_synthesis import synthesise_from_manifest
from .dataset_builder import build_manifest, split_manifest

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrates the full Speak AI data-preparation pipeline.

    Usage
    -----
    .. code-block:: python

        from pipeline import Orchestrator, PipelineConfig

        cfg = PipelineConfig()
        orc = Orchestrator(cfg)
        orc.run()

    Or run individual stages:

    .. code-block:: python

        orc.preprocess()
        orc.transcribe()
        orc.validate()
        ...
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self.cfg.ensure_dirs()
        self._configure_logging()

    # ── Logging ───────────────────────────────────────────────────────────────

    def _configure_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.cfg.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    # ── Stage 1: Preprocess ───────────────────────────────────────────────────

    def preprocess(self) -> list[Path]:
        """Resample, mono, normalise and trim all files in input_audio/original/."""
        logger.info("=== Stage 1: Preprocessing ===")
        t0 = time.time()
        results = preprocess_directory(
            input_dir=self.cfg.input_audio_dir,
            output_dir=self.cfg.processed_dir,
        )
        logger.info("Preprocessing done in %.1fs — %d files.", time.time() - t0, len(results))
        return results

    # ── Stage 2: Transcribe ───────────────────────────────────────────────────

    def transcribe(
        self,
        backend: str = "openai-whisper",
        model_name: Optional[str] = None,
        language: Optional[str] = None,
    ) -> list:
        """ASR transcription of preprocessed audio → JSON sidecars."""
        logger.info("=== Stage 2: Transcription (backend=%s) ===", backend)
        t0 = time.time()
        results = transcribe_directory(
            input_dir=self.cfg.processed_dir,
            output_dir=self.cfg.transcripts_dir,
            backend=backend,
            model_name=model_name,
            speech_aug_root=self.cfg.speech_aug_root,
            language=language,
        )
        logger.info("Transcription done in %.1fs — %d files.", time.time() - t0, len(results))
        return results

    # ── Stage 3: Validate ─────────────────────────────────────────────────────

    def validate(self, max_cer: float = 0.30) -> dict:
        """Quality-gate transcripts; quarantine failures."""
        logger.info("=== Stage 3: Transcript Validation ===")
        summary = validate_batch(
            transcripts_dir=self.cfg.transcripts_dir,
            max_cer=max_cer,
        )
        logger.info(
            "Validation: %d passed, %d failed.",
            summary["passed"],
            summary["failed"],
        )
        return summary

    # ── Stage 4: Augment ──────────────────────────────────────────────────────

    def augment(
        self,
        snr_db: float = 15.0,
        use_audiomentations: bool = True,
        use_cleese: bool = False,
    ) -> list[Path]:
        """Apply audio augmentation to preprocessed files."""
        logger.info("=== Stage 4: Augmentation ===")
        t0 = time.time()
        results = augment_directory(
            input_dir=self.cfg.processed_dir,
            output_dir=self.cfg.augmented_dir,
            noise_dir=self.cfg.noise_samples_dir,
            snr_db=snr_db,
            speech_aug_root=self.cfg.speech_aug_root,
            use_audiomentations=use_audiomentations,
            use_cleese=use_cleese,
        )
        logger.info("Augmentation done in %.1fs — %d files.", time.time() - t0, len(results))
        return results

    # ── Stage 5: Synthesise (optional) ───────────────────────────────────────

    def synthesise(
        self,
        manifest_path: Optional[Path] = None,
        engine: str = "edge_tts",
        **engine_kwargs,
    ) -> list[Path]:
        """
        TTS synthesis from a text manifest (optional stage).

        If *manifest_path* is None, looks for data/manifests/corpus.jsonl.
        """
        corpus = manifest_path or (self.cfg.manifests_dir / "corpus.jsonl")
        if not corpus.exists():
            logger.info("No corpus manifest found at %s — skipping synthesis.", corpus)
            return []

        logger.info("=== Stage 5: Speech Synthesis (engine=%s) ===", engine)
        t0 = time.time()
        results = synthesise_from_manifest(
            manifest_path=corpus,
            output_dir=self.cfg.synthetic_dir,
            ems_root=self.cfg.ems_pipeline_root,
            engine=engine,
            **engine_kwargs,
        )
        logger.info("Synthesis done in %.1fs — %d files.", time.time() - t0, len(results))
        return results

    # ── Stage 6: Build manifest ───────────────────────────────────────────────

    def build(
        self,
        output_name: str = "dataset",
        include_augmented: bool = True,
        include_synthetic: bool = True,
    ) -> Path:
        """Assemble the Whisper-ready JSONL manifest."""
        logger.info("=== Stage 6: Dataset Build ===")

        synth_manifest = self.cfg.manifests_dir / "corpus.jsonl"

        manifest_path = build_manifest(
            transcripts_dir=self.cfg.transcripts_dir,
            manifests_dir=self.cfg.manifests_dir,
            output_name=output_name,
            include_augmented=self.cfg.augmented_dir if include_augmented else None,
            include_synthetic=self.cfg.synthetic_dir if include_synthetic else None,
            synthetic_manifest=synth_manifest if synth_manifest.exists() else None,
        )
        return manifest_path

    # ── Stage 7: Split ────────────────────────────────────────────────────────

    def split(
        self,
        manifest_path: Optional[Path] = None,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
    ) -> dict[str, Path]:
        """Split the assembled manifest into train / val / test sets."""
        logger.info("=== Stage 7: Manifest Split ===")
        mp = manifest_path or (self.cfg.manifests_dir / "dataset.jsonl")
        return split_manifest(
            manifest_path=mp,
            manifests_dir=self.cfg.manifests_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=self.cfg.seed,
        )

    # ── Full run ──────────────────────────────────────────────────────────────

    def run(
        self,
        asr_backend: str = "openai-whisper",
        tts_engine: str = "edge_tts",
        run_synthesis: bool = False,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
    ) -> dict[str, Path]:
        """
        Execute all pipeline stages end-to-end.

        Returns the split manifest paths dict:
        {'train': Path, 'val': Path, 'test': Path}.
        """
        logger.info("╔══════════════════════════════════════════╗")
        logger.info("║   Speak AI Pipeline — Starting Full Run  ║")
        logger.info("╚══════════════════════════════════════════╝")
        t_start = time.time()

        self.preprocess()
        self.transcribe(backend=asr_backend)
        self.validate()
        self.augment()

        if run_synthesis:
            self.synthesise(engine=tts_engine)

        manifest_path = self.build()
        splits = self.split(manifest_path=manifest_path, train_ratio=train_ratio, val_ratio=val_ratio)

        elapsed = time.time() - t_start
        logger.info("Pipeline complete in %.1fs.", elapsed)
        logger.info("Train: %s", splits["train"])
        logger.info("Val:   %s", splits["val"])
        logger.info("Test:  %s", splits["test"])

        return splits
