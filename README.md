<img width="1910" height="826" alt="image" src="https://github.com/user-attachments/assets/732b4c6f-903a-4dab-a462-c4b5358a999e" /># Speak AI Pipeline

An end-to-end speech data preparation and Whisper fine-tuning pipeline that integrates:

- **Repository 1 — [ems_audio_syn_pipeline](https://github.com/ChiShengChen/ems_audio_syn_pipeline)**: EMS-domain text generation, multi-engine TTS, and radio/noise augmentation.
- **Repository 2 — [speech-augmentation](https://github.com/C3Imaging/speech-augmentation)**: Wav2Vec2 / Whisper ASR inference, forced alignment, CLEESE audio augmentation, and speaker diarization.
- **Speak AI integration layer** (`pipeline/`, `training/`, `tests/`): adapters, orchestrator, transcript validation, dataset builder, and training wrappers.

---

## Project Structure
<img width="1910" height="826" alt="image" src="https://github.com/user-attachments/assets/8a6b1a13-1417-4c7a-9c3c-c7fac3b421ae" />

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

# Core deps (always needed — no conflict risk):
pip install -r requirements.txt

# + Coqui TTS / XTTS v2 (pulls in gruut, needs jsonlines unpinned):
pip install -r requirements-tts.txt

# + Wav2Vec2 forced alignment (Linux / Mac only):
pip install -r requirements-asr.txt

# + Speaker diarization (GPU recommended):
pip install -r requirements-diarize.txt
```

> **Why split?** `TTS>=0.22.0` pulls in `gruut==2.2.3` which requires
> `jsonlines~=1.2.0`. Keeping TTS in its own file prevents that constraint
> from blocking the core install on Windows.

> [!IMPORTANT]
> **Windows — Coqui TTS requires Microsoft C++ Build Tools.**
> Coqui TTS compiles a Cython C extension (`monotonic_align.core`) at install
> time. Before running `pip install -r requirements-tts.txt` on Windows you
> must first install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> (choose "Desktop development with C++", ~6 GB download).
>
> **You do NOT need Coqui TTS for the default pipeline.** Edge TTS
> (already in `requirements.txt`) handles synthesis with no compilation, no
> GPU, and no API key. Only install `requirements-tts.txt` if you specifically
> need **XTTS v2 voice cloning**.

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

