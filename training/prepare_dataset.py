"""
training/prepare_dataset.py
Thin wrapper around Repository 1's prepare_for_whisper.py.

Resolves paths from PipelineConfig and delegates to the upstream script,
so the Speak AI project doesn't duplicate preparation logic.

Usage (CLI):
  python training/prepare_dataset.py --manifest data/manifests/dataset_train.jsonl

Usage (Python):
  from training.prepare_dataset import prepare_for_whisper
  prepare_for_whisper(manifest_path, output_dir)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def prepare_for_whisper(
    manifest_path: Path,
    output_dir: Path,
    config: PipelineConfig | None = None,
) -> int:
    """
    Invoke Repository 1's prepare_for_whisper.py on the given manifest.

    Returns the subprocess return code.
    """
    cfg = config or PipelineConfig()
    script = cfg.ems_pipeline_root / "prepare_for_whisper.py"

    if not script.exists():
        raise FileNotFoundError(f"prepare_for_whisper.py not found at {script}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script),
        "--manifest", str(manifest_path),
        "--output_dir", str(output_dir),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        logger.error("prepare_for_whisper.py exited with code %d", result.returncode)
    else:
        logger.info("Dataset preparation complete → %s", output_dir)

    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset for Whisper fine-tuning.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/dataset_train.jsonl"),
        help="Path to the training JSONL manifest.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/whisper_ready"),
        help="Directory to write the prepared HuggingFace dataset.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rc = prepare_for_whisper(args.manifest, args.output_dir)
    sys.exit(rc)


if __name__ == "__main__":
    main()
