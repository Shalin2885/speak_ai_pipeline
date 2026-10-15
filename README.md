# Speak AI Pipeline

An end-to-end speech data preparation and Whisper fine-tuning pipeline that integrates:

- **Repository 1 — [ems_audio_syn_pipeline](https://github.com/ChiShengChen/ems_audio_syn_pipeline)**: EMS-domain text generation, multi-engine TTS, and radio/noise augmentation.
- **Repository 2 — [speech-augmentation](https://github.com/C3Imaging/speech-augmentation)**: Wav2Vec2 / Whisper ASR inference, forced alignment, CLEESE audio augmentation, and speaker diarization.
- **Speak AI integration layer** (`pipeline/`, `training/`, `tests/`): adapters, orchestrator, transcript validation, dataset builder, and training wrappers.

---

## Project Structure

```
speak_ai_pipeline/
├── third_party/
│   ├── ems_audio_syn_pipeline/   # Repository 1 (read-only)
│   └── speech-augmentation/      # Repository 2 (read-only)
├── pipeline/                     # Speak AI integration code
│   ├── config.py                 # Central config (reads .env)
│   ├── audio_preprocessing.py    # Resample / normalise / trim
│   ├── augmentation.py           # Adapter → Repository 2 augmentation
│   ├── transcription.py          # ASR adapter (Whisper / Wav2Vec2)
│   ├── transcript_validation.py  # Quality gate + quarantine
│   ├── speech_synthesis.py       # TTS adapter → Repository 1
│   ├── dataset_builder.py        # JSONL manifest assembly + split
│   └── orchestrator.py           # End-to-end stage runner
├── training/
│   ├── prepare_dataset.py        # Wrapper → prepare_for_whisper.py
│   ├── train_whisper.py          # Wrapper → finetune_whisper.py
│   └── evaluate.py               # Wrapper → evaluate_finetuned_whisper.py
├── tests/
│   ├── test_transcription.py
│   ├── test_augmentation.py
│   └── test_dataset_builder.py
├── input_audio/
│   ├── original/                 # Drop your source WAV/MP3/FLAC here
│   └── noise_samples/            # Optional background noise files
├── data/                         # Generated — not tracked by git
├── outputs/                      # Logs, reports, final model
├── requirements.txt
├── .env.example                  # Copy to .env and fill in keys
└── .gitignore
```

---

## Quick Start

### 1. Clone & set up environment

```bash
git clone <your-fork> speak_ai_pipeline
cd speak_ai_pipeline
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set HF_TOKEN and OPENAI_API_KEY if using GPT for corpus generation
```

### 3. Add your recordings

Drop 16-kHz mono WAV files (or MP3/FLAC — they'll be converted automatically) into:

```
input_audio/original/
```

Optionally add background noise WAVs to `input_audio/noise_samples/`.

### 4. Run the full pipeline

```python
from pipeline import Orchestrator, PipelineConfig

cfg = PipelineConfig()
orc = Orchestrator(cfg)
splits = orc.run(asr_backend="openai-whisper", tts_engine="edge_tts")
# → data/manifests/dataset_train.jsonl
# → data/manifests/dataset_val.jsonl
# → data/manifests/dataset_test.jsonl
```

Or stage by stage:

```python
orc.preprocess()   # → data/processed/
orc.transcribe()   # → data/transcripts/
orc.validate()     # quarantine bad transcripts
orc.augment()      # → data/augmented/
orc.synthesise()   # → data/synthetic/  (optional — needs corpus.jsonl)
orc.build()        # → data/manifests/dataset.jsonl
orc.split()        # → train / val / test splits
```

### 5. Fine-tune Whisper

```bash
python training/prepare_dataset.py --manifest data/manifests/dataset_train.jsonl
python training/train_whisper.py
python training/evaluate.py
```

---

## Pipeline Stages

| # | Stage | Input | Output | Module |
|---|-------|-------|--------|--------|
| 1 | Preprocess | `input_audio/original/` | `data/processed/` | `audio_preprocessing.py` |
| 2 | Transcribe | `data/processed/` | `data/transcripts/*.json` | `transcription.py` |
| 3 | Validate | `data/transcripts/` | quarantine bad files | `transcript_validation.py` |
| 4 | Augment | `data/processed/` | `data/augmented/` | `augmentation.py` |
| 5 | Synthesise | `data/manifests/corpus.jsonl` | `data/synthetic/` | `speech_synthesis.py` |
| 6 | Build | transcripts + augmented + synthetic | `data/manifests/dataset.jsonl` | `dataset_builder.py` |
| 7 | Split | `dataset.jsonl` | train / val / test JSONL | `dataset_builder.py` |

---

## ASR Backends

| Backend | Quality | Speed | GPU | Notes |
|---------|---------|-------|-----|-------|
| `openai-whisper` | ★★★★★ | Medium | Optional | Default; word timestamps |
| `hf-whisper` | ★★★★☆ | Medium | Optional | HuggingFace Transformers |
| `wav2vec2` | ★★★☆☆ | Fast | Optional | Requires Repository 2; Linux/Mac |

Set via `transcribe(backend="openai-whisper")` or `orc.transcribe(backend=...)`.

---

## TTS Engines (Synthesis Stage)

| Engine | Quality | Speed | GPU | Notes |
|--------|---------|-------|-----|-------|
| `edge_tts` | ★★★★☆ | Fast | No | Default; no key needed |
| `xtts` | ★★★★★ | Slow | Recommended | Voice cloning |
| `bark` | ★★★★☆ | Very slow | Required | Expressive; Linux/Mac |

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

Tests are fully mocked — no GPU, no API keys, no large models needed.

---

## Repository Licences

- **ems_audio_syn_pipeline**: see [`third_party/ems_audio_syn_pipeline/LICENSE`](third_party/ems_audio_syn_pipeline/LICENSE)
- **speech-augmentation**: see [`third_party/speech-augmentation/README.md`](third_party/speech-augmentation/README.md)

The Speak AI integration layer (`pipeline/`, `training/`, `tests/`) is © Speak AI.
