"""
training/evaluate.py
Thin wrapper around Repository 1's evaluate_finetuned_whisper.py.

Runs WER/CER evaluation on the test split and writes a JSON report
to outputs/reports/.

Usage (CLI):
  python training/evaluate.py

Usage (Python):
  from training.evaluate import evaluate_whisper
  results = evaluate_whisper()
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def evaluate_whisper(
    model_path: Path | None = None,
    test_manifest: Path | None = None,
    config: PipelineConfig | None = None,
) -> int:
    """
    Invoke Repository 1's evaluate_finetuned_whisper.py.

    Returns the subprocess return code.
    """
    cfg = config or PipelineConfig()
    script = cfg.ems_pipeline_root / "evaluate_finetuned_whisper.py"

    if not script.exists():
        raise FileNotFoundError(f"evaluate_finetuned_whisper.py not found at {script}")

    if model_path:
        model_dir = str(model_path)
    elif cfg.whisper_output_dir.exists() and any(cfg.whisper_output_dir.iterdir()):
        model_dir = str(cfg.whisper_output_dir)
    else:
        model_dir = cfg.whisper_base_model

    test_dataset = test_manifest or Path("data/whisper_ready/test")
    if not test_dataset.exists():
        logger.error("Test dataset not found at %s. Please run prepare_dataset.py first.", test_dataset)
        return 1

    report_dir = Path("outputs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    output_csv = report_dir / "eval_results.csv"

    cmd = [
        sys.executable,
        str(script),
        "--model_path", model_dir,
        "--test_dataset_path", str(test_dataset),
        "--output_csv", str(output_csv),
    ]

    logger.info("Running evaluation …")
    logger.info("Command: %s", " ".join(cmd))

    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    result = subprocess.run(cmd, env=env, check=False)

    if result.returncode == 0:
        logger.info("Evaluation complete. Reports written to %s", report_dir)
    else:
        logger.error("Evaluation script exited with code %d", result.returncode)

    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned Whisper model.")
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--test_manifest", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rc = evaluate_whisper(model_path=args.model_path, test_manifest=args.test_manifest)
    sys.exit(rc)


if __name__ == "__main__":
    main()

