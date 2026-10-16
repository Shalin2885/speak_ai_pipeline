"""
training/train_whisper.py
Thin wrapper around Repository 1's finetune_whisper.py.

Reads Whisper fine-tuning hyperparameters from PipelineConfig / .env
and delegates training to the upstream script.

Usage (CLI):
  python training/train_whisper.py

Usage (Python):
  from training.train_whisper import train_whisper
  train_whisper()
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def train_whisper(config: PipelineConfig | None = None) -> int:
    """
    Invoke Repository 1's finetune_whisper.py with config-driven arguments.

    Returns the subprocess return code.
    """
    cfg = config or PipelineConfig()
    script = cfg.ems_pipeline_root / "finetune_whisper.py"

    if not script.exists():
        raise FileNotFoundError(f"finetune_whisper.py not found at {script}")

    train_manifest = cfg.manifests_dir / "dataset_train.jsonl"
    val_manifest = cfg.manifests_dir / "dataset_val.jsonl"

    dataset_path = Path("data/whisper_ready")
    if not dataset_path.exists():
        logger.info("Prepared dataset not found at %s. Running prepare_dataset first...", dataset_path)
        from training.prepare_dataset import prepare_for_whisper
        rc = prepare_for_whisper(train_manifest, dataset_path, config=cfg)
        if rc != 0:
            logger.error("Dataset preparation failed.")
            return rc

    cmd = [
        sys.executable,
        str(script),
        "--model_name", cfg.whisper_base_model,
        "--dataset_path", str(dataset_path),
        "--output_dir", str(cfg.whisper_output_dir),
        "--per_device_train_batch_size", str(cfg.whisper_batch_size),
        "--per_device_eval_batch_size", str(cfg.whisper_batch_size),
        "--learning_rate", str(cfg.whisper_learning_rate),
        "--num_train_epochs", str(getattr(cfg, "whisper_num_epochs", 3)),
    ]

    # Pass HuggingFace token and enforce CPU thread limits to prevent overheating
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    threads = str(getattr(cfg, "whisper_cpu_threads", 2))
    env["OMP_NUM_THREADS"] = threads
    env["MKL_NUM_THREADS"] = threads
    env["TORCH_NUM_THREADS"] = threads
    if cfg.hf_token:
        env["HF_TOKEN"] = cfg.hf_token

    logger.info("Starting Whisper fine-tuning …")
    logger.info("Base model: %s | CPU threads capped to: %s | Epochs: %s | Batch size: %s",
                cfg.whisper_base_model, threads, getattr(cfg, "whisper_num_epochs", 3), cfg.whisper_batch_size)
    logger.info("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, env=env, check=False)

    if result.returncode != 0:
        logger.error("finetune_whisper.py exited with code %d", result.returncode)
    else:
        logger.info("Fine-tuning complete → %s", cfg.whisper_output_dir)

    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on the Speak AI dataset.")
    parser.add_argument("--model", default=None, help="Override WHISPER_BASE_MODEL env var.")
    parser.add_argument("--epochs", type=int, default=None, help="Override WHISPER_NUM_EPOCHS.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override WHISPER_BATCH_SIZE.")
    parser.add_argument("--threads", type=int, default=None, help="Override CPU threads (default 2).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = PipelineConfig()
    if args.model:
        cfg.whisper_base_model = args.model
    if args.epochs:
        cfg.whisper_num_epochs = args.epochs
    if args.batch_size:
        cfg.whisper_batch_size = args.batch_size
    if args.threads:
        cfg.whisper_cpu_threads = args.threads

    rc = train_whisper(cfg)
    sys.exit(rc)


if __name__ == "__main__":
    main()

