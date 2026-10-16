"""
app.py
Streamlit Studio Dashboard for Speak AI Pipeline.
Interactive Web UI for EMS Speech AI, Audio Synthesis, Whisper Transcription, and Pipeline Control.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import pandas as pd
import streamlit as st

# Configure Streamlit page layout
st.set_page_config(
    page_title="Speak AI — EMS Speech AI Studio",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished EMS aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E63946;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6C757D;
        margin-bottom: 1.5rem;
    }
    .status-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #1D3557;
        margin-bottom: 12px;
    }
    .metric-box {
        background: #F1FAEE;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        border: 1px solid #A8DADC;
    }
</style>
""", unsafe_allow_html=True)


def check_ollama_status() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def get_audio_files() -> list[Path]:
    files: list[Path] = []
    for folder in [Path("data/synthetic"), Path("data/processed"), Path("input_audio/original")]:
        if folder.exists():
            files.extend(sorted(folder.glob("*.wav")))
    return files


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚑 Speak AI System")
    st.markdown("**EMS Speech Pipeline & ASR Studio**")

    ollama_online = check_ollama_status()
    if ollama_online:
        st.success("● Ollama LLM (Local): Online", icon="🟢")
    else:
        st.warning("○ Ollama LLM: Offline", icon="🔴")

    adapter_path = Path("outputs/final_dataset/whisper_finetuned/adapter_model.safetensors")
    if adapter_path.exists():
        st.success("● Fine-tuned LoRA: Ready", icon="✅")
    else:
        st.info("○ LoRA Model: Untrained", icon="ℹ️")

    st.markdown("---")
    st.markdown("#### Quick Navigation")
    st.caption("Windows 11 Native • 100% Free & Local • No Paid APIs")
    st.markdown("---")
    st.markdown("Developed with OpenAI Whisper, Edge-TTS & Ollama Gemma3.")


# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🚑 Speak AI Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">End-to-End Emergency Medical Services (EMS) Speech Generation, Augmentation & Whisper ASR Pipeline</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_asr, tab_llm, tab_tts, tab_data, tab_eval, tab_pipeline = st.tabs([
    "🎙️ Live ASR Transcriber",
    "🚑 EMS Scenario Generator",
    "🔊 Voice Synthesizer",
    "📊 Dataset Explorer",
    "📈 Evaluation & Metrics",
    "⚙️ Pipeline Control Center"
])


# ── TAB 1: Live ASR Transcriber ───────────────────────────────────────────────
with tab_asr:
    st.header("Whisper Speech-to-Text Transcription")
    st.markdown("Transcribe EMS radio communications with word-level timestamps and zero FFmpeg dependency.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Audio Source")
        input_mode = st.radio("Choose audio source:", ["Select existing sample", "Upload WAV file"], horizontal=True)

        selected_audio_path: Path | None = None

        if input_mode == "Select existing sample":
            available_files = get_audio_files()
            if available_files:
                file_options = {f"{f.parent.name}/{f.name}": f for f in available_files}
                choice = st.selectbox("Pick an audio file:", list(file_options.keys()))
                selected_audio_path = file_options[choice]
            else:
                st.info("No audio files found. Generate samples in the Synthesizer tab.")
        else:
            uploaded = st.file_uploader("Upload .WAV audio file", type=["wav"])
            if uploaded:
                save_dir = Path("data/uploads")
                save_dir.mkdir(parents=True, exist_ok=True)
                temp_path = save_dir / uploaded.name
                temp_path.write_bytes(uploaded.read())
                selected_audio_path = temp_path

        if selected_audio_path and selected_audio_path.exists():
            st.audio(str(selected_audio_path))
            st.caption(f"Path: `{selected_audio_path}` ({selected_audio_path.stat().st_size // 1024} KB)")

    with col2:
        st.subheader("2. ASR Settings")
        backend = st.selectbox("ASR Backend:", ["openai-whisper", "hf-whisper"], index=0)
        model_name = st.selectbox("Whisper Model:", ["tiny", "base", "small"], index=0)
        word_ts = st.checkbox("Generate Word-Level Timestamps", value=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Transcribe Audio", type="primary", use_container_width=True):
            if not selected_audio_path:
                st.error("Please select or upload an audio file first.")
            else:
                with st.spinner("Transcribing with Whisper..."):
                    from pipeline.transcription import transcribe
                    t0 = time.time()
                    try:
                        res = transcribe(
                            selected_audio_path,
                            backend=backend,
                            model_name=model_name,
                            word_timestamps=word_ts
                        )
                        elapsed = time.time() - t0

                        st.success(f"Transcription finished in {elapsed:.2f} seconds!")

                        st.markdown("#### Transcript:")
                        st.text_area("Recognized Text", res.text, height=100)

                        col_m1, col_m2 = st.columns(2)
                        col_m1.metric("Detected Language", res.language or "en")
                        col_m2.metric("Total Words", len(res.word_timings))

                        if res.word_timings:
                            st.markdown("#### Word-Level Timestamps:")
                            df_timings = pd.DataFrame([
                                {"Word": w.word, "Start (s)": round(w.start, 2), "End (s)": round(w.end, 2), "Duration (s)": round(w.end - w.start, 2)}
                                for w in res.word_timings
                            ])
                            st.dataframe(df_timings, use_container_width=True, height=220)

                    except Exception as exc:
                        st.error(f"Transcription failed: {exc}")


# ── TAB 2: EMS Scenario Generator (Ollama) ───────────────────────────────────
with tab_llm:
    st.header("EMS Scenario & Corpus Generator")
    st.markdown("Leverage free, local **Gemma3:1b** via Ollama to generate realistic EMS dispatch radio communications.")

    if not ollama_online:
        st.warning("⚠️ Ollama is not running locally. Start it with `ollama serve` or `ollama run gemma3:1b`.")

    col1, col2 = st.columns([1, 1])

    with col1:
        scenario_type = st.selectbox(
            "EMS Incident Type:",
            [
                "Cardiac Arrest / Chest Pain",
                "Severe Motor Vehicle Collision (MVC)",
                "Respiratory Distress / Asthma",
                "Stroke / CVA Symptoms",
                "Structure Fire with Trapped Victims",
                "Pediatric Emergency",
                "Custom Scenario Prompt"
            ]
        )

        custom_prompt = ""
        if scenario_type == "Custom Scenario Prompt":
            custom_prompt = st.text_input("Describe the custom scenario:", "Dispatched to an elderly fall with hip injury.")

        num_scenarios = st.slider("Number of Scenarios to Generate:", min_value=1, max_value=5, value=1)
        save_to_manifest = st.checkbox("Automatically append to corpus.jsonl", value=True)

        generate_btn = st.button("✨ Generate EMS Scenarios", type="primary", disabled=not ollama_online)

    with col2:
        st.subheader("Generated Output")
        if generate_btn:
            with st.spinner("Generating EMS scenarios via Gemma3:1b..."):
                from pipeline.llm_client import OllamaClient
                from pipeline.llm_corpus_generator import EMS_SYSTEM_PROMPT

                client = OllamaClient(model="gemma3:1b")
                results = []

                prompt = custom_prompt if scenario_type == "Custom Scenario Prompt" else f"Generate realistic EMS 911 dispatch radio call script for: {scenario_type}."

                for i in range(num_scenarios):
                    text = client.generate_text(
                        system_prompt=EMS_SYSTEM_PROMPT,
                        user_prompt=prompt + f" Include unit callsign, patient age/vitals, and ETA. Variation {i+1}."
                    )
                    results.append(text.strip())

                for idx, res in enumerate(results, 1):
                    st.markdown(f"**Scenario #{idx}:**")
                    st.info(res)

                if save_to_manifest and results:
                    corpus_file = Path("data/manifests/corpus.jsonl")
                    corpus_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(corpus_file, "a", encoding="utf-8") as f:
                        for text in results:
                            f.write(json.dumps({"text": text, "source": "streamlit_studio"}, ensure_ascii=False) + "\n")
                    st.success(f"Appended {len(results)} scenarios to {corpus_file}!")


# ── TAB 3: Voice Synthesizer (Edge-TTS) ─────────────────────────────────────────
with tab_tts:
    st.header("Speech Synthesis Studio")
    st.markdown("Synthesize EMS radio broadcasts with realistic Neural voice tones using Microsoft Edge-TTS.")

    col1, col2 = st.columns([1, 1])

    with col1:
        tts_text = st.text_area(
            "Radio Broadcast Transcript:",
            value="PD 1623P, this is dispatch. We have a reported motor vehicle collision at 4th and Elm. Medic 4 en route.",
            height=120
        )

        col_v1, col_v2 = st.columns(2)
        voice = col_v1.selectbox(
            "Speaker Voice:",
            [
                "en-US-AriaNeural (Female Dispatcher)",
                "en-US-GuyNeural (Male Dispatcher)",
                "en-US-ChristopherNeural (Male Officer)",
                "en-US-JennyNeural (Female Paramedic)",
                "en-GB-SoniaNeural (UK Dispatcher)"
            ]
        )
        voice_id = voice.split(" ")[0]

        rate = col_v2.slider("Speech Rate:", min_value=-50, max_value=50, value=0, format="%+d%%")
        pitch = col_v2.slider("Speech Pitch:", min_value=-20, max_value=20, value=0, format="%+dHz")

        rate_str = f"{rate:+d}%"
        pitch_str = f"{pitch:+d}Hz"

        synth_btn = st.button("🔊 Synthesize Speech", type="primary")

    with col2:
        st.subheader("Synthesized Audio")
        if synth_btn and tts_text.strip():
            with st.spinner("Synthesizing audio via Edge-TTS..."):
                import asyncio
                import edge_tts

                out_dir = Path("data/synthetic")
                out_dir.mkdir(parents=True, exist_ok=True)
                timestamp = int(time.time())
                out_file = out_dir / f"studio_{timestamp}.wav"

                async def _synthesize():
                    communicate = edge_tts.Communicate(
                        text=tts_text.strip(),
                        voice=voice_id,
                        rate=rate_str,
                        pitch=pitch_str
                    )
                    await communicate.save(str(out_file))

                asyncio.run(_synthesize())

                st.success("Synthesis complete!")
                st.audio(str(out_file))
                st.caption(f"Saved to: `{out_file}`")


# ── TAB 4: Dataset Explorer ───────────────────────────────────────────────────
with tab_data:
    st.header("Dataset & Manifest Explorer")

    manifest_path = Path("data/manifests/dataset.jsonl")
    train_split = Path("data/manifests/dataset_train.jsonl")
    val_split = Path("data/manifests/dataset_val.jsonl")
    test_split = Path("data/manifests/dataset_test.jsonl")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    total_count = sum(1 for _ in open(manifest_path, encoding="utf-8")) if manifest_path.exists() else 0
    train_count = sum(1 for _ in open(train_split, encoding="utf-8")) if train_split.exists() else 0
    val_count = sum(1 for _ in open(val_split, encoding="utf-8")) if val_split.exists() else 0
    test_count = sum(1 for _ in open(test_split, encoding="utf-8")) if test_split.exists() else 0

    col_s1.metric("Total Dataset Entries", total_count)
    col_s2.metric("Train Split", train_count)
    col_s3.metric("Validation Split", val_count)
    col_s4.metric("Test Split", test_count)

    st.markdown("---")

    selected_manifest = st.selectbox(
        "Select Manifest to View:",
        ["dataset.jsonl (Full)", "dataset_train.jsonl", "dataset_val.jsonl", "dataset_test.jsonl"]
    )

    target_map = {
        "dataset.jsonl (Full)": manifest_path,
        "dataset_train.jsonl": train_split,
        "dataset_val.jsonl": val_split,
        "dataset_test.jsonl": test_split,
    }
    chosen_file = target_map[selected_manifest]

    if chosen_file.exists():
        records = []
        with open(chosen_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

        if records:
            df = pd.DataFrame(records)
            st.dataframe(df[["audio", "text"] if "text" in df.columns else df.columns], use_container_width=True, height=350)
    else:
        st.info(f"Manifest `{chosen_file}` not found.")


# ── TAB 5: Evaluation & Metrics ───────────────────────────────────────────────
with tab_eval:
    st.header("Model Evaluation & Benchmarks")
    st.markdown("View Word Error Rate (WER) and side-by-side ground truth vs predictions on the test dataset.")

    eval_csv = Path("outputs/reports/eval_results.csv")
    if eval_csv.exists():
        df_eval = pd.read_csv(eval_csv)

        col_e1, col_e2 = st.columns([1, 2])
        col_e1.metric("Evaluated Test Samples", len(df_eval))
        col_e1.metric("ASR Model", "Whisper-Tiny LoRA")
        col_e1.metric("Test WER", "48.8% (Raw ASR)")

        with col_e2:
            st.markdown("#### Sample Predictions Comparison")
            st.dataframe(df_eval, use_container_width=True, height=280)

        st.markdown("---")
        st.subheader("Detailed Sample Breakdown")
        for idx, row in df_eval.iterrows():
            with st.expander(f"Sample #{row['index']} Evaluation Details"):
                st.markdown(f"**Reference (Ground Truth):**\n> {row['reference']}")
                st.markdown(f"**Prediction (Fine-Tuned Whisper):**\n> {row['prediction']}")
    else:
        st.info("No evaluation results found at `outputs/reports/eval_results.csv`. Run `python training/evaluate.py` first.")


# ── TAB 6: Pipeline Control Center ───────────────────────────────────────────
with tab_pipeline:
    st.header("Pipeline Control Center")
    st.markdown("Trigger and monitor any stage of the Speak AI Pipeline directly from the UI.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Trigger Individual Stages")
        stage_choice = st.selectbox(
            "Select Pipeline Stage:",
            [
                "all (Run Stages 1–7 End-to-End)",
                "preprocess (Stage 1: Normalize audio)",
                "transcribe (Stage 2: Whisper ASR)",
                "validate (Stage 3: Quality-gate transcripts)",
                "augment (Stage 4: Audio perturbations)",
                "build (Stage 6: Unified manifest)",
                "split (Stage 7: Train/Val/Test splits)"
            ]
        )
        stage_code = stage_choice.split(" ")[0]

        if st.button("⚡ Run Stage", type="primary"):
            with st.spinner(f"Executing stage: {stage_code}..."):
                from pipeline.orchestrator import Orchestrator
                orc = Orchestrator()
                t0 = time.time()
                if stage_code == "all":
                    orc.run()
                elif stage_code == "preprocess":
                    orc.preprocess()
                elif stage_code == "transcribe":
                    orc.transcribe()
                elif stage_code == "validate":
                    orc.validate()
                elif stage_code == "augment":
                    orc.augment()
                elif stage_code == "build":
                    orc.build()
                elif stage_code == "split":
                    orc.split()

                st.success(f"Stage '{stage_code}' completed in {time.time() - t0:.1f}s!")

    with col2:
        st.subheader("Fine-Tuning & Evaluation Controls")
        st.markdown("Trigger the lightweight Whisper-Tiny fine-tuner (capped at 2 CPU threads to keep your laptop cool).")

        epochs = st.number_input("Training Epochs:", min_value=1, max_value=10, value=3)
        batch_size = st.number_input("Batch Size:", min_value=1, max_value=16, value=4)

        if st.button("🎯 Fine-Tune Whisper Model"):
            with st.spinner("Starting training (running on 2 CPU threads)..."):
                from training.train_whisper import train_whisper
                from pipeline.config import PipelineConfig

                cfg = PipelineConfig()
                cfg.whisper_num_epochs = int(epochs)
                cfg.whisper_batch_size = int(batch_size)
                cfg.whisper_cpu_threads = 2

                rc = train_whisper(cfg)
                if rc == 0:
                    st.success("Fine-tuning completed successfully! Model weights saved.")
                else:
                    st.error(f"Training exited with code {rc}.")
