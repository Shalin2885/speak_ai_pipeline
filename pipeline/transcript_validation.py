"""
pipeline/transcript_validation.py
Quality-gate utilities for ASR transcripts produced by transcription.py.

Checks performed:
  - Minimum and maximum transcript length (characters / words)
  - Confidence threshold (when the backend provides a confidence score)
  - Character-error-rate vs. a reference (optional)
  - Allowed character set / language detection sanity check
  - Duplicate detection within a batch

Failed transcripts are quarantined (moved to data/transcripts/quarantine/)
so the downstream dataset builder only sees clean data.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Optional

from .transcription import TranscriptionResult

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
MIN_WORDS = 2
MAX_WORDS = 500
MIN_CHARS = 5
MAX_CHARS = 4000
ALLOWED_LANG_RE = re.compile(r"^[a-zA-Z0-9\s,\.!\?'\"\-–—;:\(\)\[\]]+$")


def _word_count(text: str) -> int:
    return len(text.split())


def _char_count(text: str) -> int:
    return len(text.strip())


def _is_language_plausible(text: str) -> bool:
    """Rough check: flag transcripts that are mostly non-ASCII (possible gibberish)."""
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    return ascii_ratio >= 0.85


def _compute_cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate (simple edit-distance / reference-length)."""
    try:
        from jiwer import cer  # type: ignore
        return cer(reference, hypothesis)
    except ImportError:
        # Fallback: naive character-level Levenshtein approximation
        h, r = list(hypothesis), list(reference)
        d = [[0] * (len(r) + 1) for _ in range(len(h) + 1)]
        for i in range(len(h) + 1):
            d[i][0] = i
        for j in range(len(r) + 1):
            d[0][j] = j
        for i in range(1, len(h) + 1):
            for j in range(1, len(r) + 1):
                cost = 0 if h[i - 1] == r[j - 1] else 1
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
        return d[-1][-1] / max(len(r), 1)


# ── Per-result validation ─────────────────────────────────────────────────────

def validate_result(
    result: TranscriptionResult,
    reference: Optional[str] = None,
    max_cer: float = 0.30,
    min_confidence: float = 0.0,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> tuple[bool, list[str]]:
    """
    Validate a single TranscriptionResult.

    Returns (is_valid, list_of_failure_reasons).
    """
    reasons: list[str] = []
    text = result.text.strip()

    if _word_count(text) < min_words:
        reasons.append(f"Too few words: {_word_count(text)} < {min_words}")

    if _word_count(text) > max_words:
        reasons.append(f"Too many words: {_word_count(text)} > {max_words}")

    if _char_count(text) < MIN_CHARS:
        reasons.append(f"Too few characters: {_char_count(text)} < {MIN_CHARS}")

    if _char_count(text) > MAX_CHARS:
        reasons.append(f"Too many characters: {_char_count(text)} > {MAX_CHARS}")

    if not _is_language_plausible(text):
        reasons.append("Low ASCII ratio — possible wrong language or garbled output")

    if result.confidence is not None and result.confidence < min_confidence:
        reasons.append(f"Low confidence: {result.confidence:.2f} < {min_confidence:.2f}")

    if reference:
        cer = _compute_cer(text, reference)
        if cer > max_cer:
            reasons.append(f"CER too high: {cer:.2f} > {max_cer:.2f}")

    return len(reasons) == 0, reasons


# ── Batch validation ──────────────────────────────────────────────────────────

def validate_batch(
    transcripts_dir: Path,
    quarantine_dir: Optional[Path] = None,
    **validate_kwargs,
) -> dict[str, object]:
    """
    Validate all *_transcription.json* sidecar files in *transcripts_dir*.

    Quarantines invalid sidecars (and their audio counterparts) into
    *quarantine_dir* (default: transcripts_dir / 'quarantine').

    Returns a summary dict with counts and failure details.
    """
    if quarantine_dir is None:
        quarantine_dir = transcripts_dir / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(transcripts_dir.glob("*.json"))
    # Exclude files already in quarantine subdir
    json_files = [f for f in json_files if f.parent == transcripts_dir]

    passed = 0
    failed = 0
    failures: list[dict] = []
    seen_texts: set[str] = set()

    for jf in sorted(json_files):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Cannot parse %s: %s", jf, exc)
            failed += 1
            continue

        text = data.get("text", "").strip()

        # Duplicate check
        if text in seen_texts:
            reasons = ["Duplicate transcript"]
            _quarantine(jf, data, quarantine_dir, reasons)
            failed += 1
            failures.append({"file": jf.name, "reasons": reasons})
            continue

        seen_texts.add(text)

        from .transcription import TranscriptionResult, WordTiming  # local import avoids circular

        result = TranscriptionResult(
            audio_path=Path(data.get("audio_path", jf.stem)),
            text=text,
            language=data.get("language"),
            backend=data.get("backend", "unknown"),
            confidence=data.get("confidence"),
            word_timings=[
                WordTiming(**w) for w in data.get("word_timings", [])
            ],
        )

        is_valid, reasons = validate_result(result, **validate_kwargs)

        if is_valid:
            passed += 1
            logger.debug("PASS: %s", jf.name)
        else:
            failed += 1
            _quarantine(jf, data, quarantine_dir, reasons)
            failures.append({"file": jf.name, "reasons": reasons})
            logger.warning("FAIL: %s — %s", jf.name, "; ".join(reasons))

    summary = {
        "total": passed + failed,
        "passed": passed,
        "failed": failed,
        "failures": failures,
    }
    logger.info(
        "Validation complete. Passed: %d / %d. Quarantined: %d.",
        passed,
        passed + failed,
        failed,
    )
    return summary


def _quarantine(
    json_path: Path,
    data: dict,
    quarantine_dir: Path,
    reasons: list[str],
) -> None:
    """Move the sidecar JSON (and matching audio if found) to quarantine_dir."""
    # Annotate with failure reasons before moving
    data["_validation_failures"] = reasons
    dest_json = quarantine_dir / json_path.name
    dest_json.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    json_path.unlink(missing_ok=True)

    # Try to move the matching audio file
    audio_path = Path(data.get("audio_path", ""))
    if audio_path.exists():
        try:
            shutil.move(str(audio_path), str(quarantine_dir / audio_path.name))
        except Exception as exc:
            logger.warning("Could not quarantine audio %s: %s", audio_path, exc)

