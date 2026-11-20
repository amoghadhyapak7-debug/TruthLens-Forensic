"""Video Forensics View for TruthLens Monitor."""
from __future__ import annotations

import requests
import streamlit as st
from pathlib import Path

try:
    from app.analyzers.normalization import normalize_video_verdict
except ImportError:
    def normalize_video_verdict(v_data):
        if not isinstance(v_data, dict):
            return {"verdict": str(v_data or "INCONCLUSIVE"), "fake_probability": 0.0, "confidence_score": 0.0, "models_used": ["GenD CLIP ViT-L/14"]}
        asmt = v_data.get("assessment") or {}
        v_raw = v_data.get("verdict")
        v_str = v_raw if isinstance(v_raw, str) else (v_raw.get("verdict") if isinstance(v_raw, dict) else asmt.get("verdict", "INCONCLUSIVE"))
        prob = asmt.get("manipulation_score_pct")
        fake_prob = float(prob) / 100.0 if prob is not None else 0.0
        return {"verdict": str(v_str), "fake_probability": fake_prob, "confidence_score": fake_prob, "models_used": v_data.get("models_used", ["GenD CLIP ViT-L/14", "MesoNet-4", "YuNet"])}



def render_video_view(api_url: str, all_records: list[dict]):
    st.header("🎬 Video Forensics Analyzer")
    st.caption("Deepfake detection powered by GenD CLIP ViT-L/14, MesoNet-4, YuNet & Integrated Wav2Vec2 Audio Analysis")

    video_records = [r for r in all_records if r.get("file_type") == "video"]
    if not video_records:
        st.info("No video evidence registered yet. Upload video files (.mp4, .mov, .avi) in the **Evidence Ingest** tab first.")
        return

    # Evidence selector
    ev_options = {f"{r.get('evidence_id')} - {r.get('original_filename')}": r for r in video_records}
    selected_label = st.selectbox("Select Video Evidence", options=list(ev_options.keys()), key="video_evidence_select")
    record = ev_options[selected_label]
    evidence_id = record.get("evidence_id")

    v_col1, v_col2 = st.columns([1, 1])
    with v_col1:
        st.write(f"**Evidence ID:** `{evidence_id}`")
        st.write(f"**Filename:** `{record.get('original_filename')}`")
        st.write(f"**SHA-256:** `{record.get('sha256', '')[:20]}...`")
        st.write(f"**File Size:** {record.get('file_size', 0):,} bytes")
    with v_col2:
        try:
            st.video(f"{api_url}/api/v1/evidence/{evidence_id}/content")
        except Exception:
            st.info("Video preview unavailable.")

    # Analysis Action Button
    st.markdown("---")
    res_key = f"video_analysis_result_{evidence_id}"

    # Try to load existing cached result from API
    if res_key not in st.session_state:
        try:
            r = requests.get(f"{api_url}/api/v1/analyzers/video/{evidence_id}", timeout=5)
            if r.status_code == 200:
                st.session_state[res_key] = r.json()
        except Exception:
            pass

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("🚀 Run Video & Audio Analysis", type="primary", key=f"run_vid_{evidence_id}")

    if run_btn:
        with st.spinner("Executing full forensic analysis: GenD ViT-L/14 + Audio extraction + Wav2Vec2 ONNX..."):
            try:
                resp = requests.post(f"{api_url}/api/v1/analyzers/video/{evidence_id}", timeout=600)
                if resp.status_code == 200:
                    st.session_state[res_key] = resp.json()
                    st.success("Analysis complete!")
                else:
                    st.error(f"Analysis failed: {resp.text}")
            except Exception as exc:
                st.error(f"Error during video analysis: {exc}")

    result = st.session_state.get(res_key)
    if not result:
        st.info("Click **Run Video & Audio Analysis** to perform deepfake inference.")
        return

    # Display Analysis Results
    v_data = result.get("video") if (isinstance(result, dict) and "video" in result) else result
    a_data = result.get("audio") if isinstance(result, dict) else None
    a_status = result.get("audio_extraction_status", "unknown") if isinstance(result, dict) else "unknown"

    norm_verdict = normalize_video_verdict(v_data)
    verdict_str = norm_verdict["verdict"]
    fake_prob = norm_verdict["fake_probability"]
    conf_score = norm_verdict["confidence_score"]

    st.subheader("1. Visual Deepfake Verdict (GenD CLIP ViT-L/14)")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        if "DEEPFAKE DETECTED" in verdict_str and "NO" not in verdict_str:
            st.error(f"🚨 **{verdict_str}**")
        elif "NO DEEPFAKE" in verdict_str:
            st.success(f"✅ **{verdict_str}**")
        else:
            st.warning(f"⚠️ **{verdict_str}**")
    with m_col2:
        st.metric("Synthetic Probability", f"{fake_prob:.1%}")
        st.progress(min(max(float(fake_prob), 0.0), 1.0))
    with m_col3:
        st.metric("Primary Model", "GenD ViT-L/14")
        st.caption("WACV 2026 Checkpoint (openai/clip-vit-large-patch14)")

    # Secondary detectors info
    sec_models = v_data.get("models_used", []) or ["GenD CLIP ViT-L/14", "MesoNet-4", "YuNet Face Detector"]
    st.caption(f"**Forensic Pipeline Models:** {', '.join(sec_models)}")

    # Display Critical Audio Deepfake Result
    st.markdown("---")
    st.subheader("2. Extracted Audio Stream Forensics (Wav2Vec2 ONNX)")
    if a_status == "no_audio_stream":
        st.info("ℹ️ **Silent Video:** Media file contains no audio track. Audio deepfake analysis is Not Applicable.")
    elif a_status == "extraction_failed":
        st.warning("⚠️ **Audio Extraction Failed:** Could not extract audio track for analysis.")
    elif a_data and a_status == "extracted":
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            a_res = a_data.get("result", "UNKNOWN")
            if "SYNTHETIC" in a_res and "NO" not in a_res:
                st.error(f"🚨 **{a_res}**")
            elif "NO STRONG" in a_res:
                st.success(f"✅ **{a_res}**")
            else:
                st.warning(f"⚠️ **{a_res}**")
            
            synth_score = a_data.get("synthetic_score")
            if synth_score is not None:
                st.metric("Audio Synthetic Score", f"{synth_score:.1%}")
                st.progress(min(max(float(synth_score), 0.0), 1.0))
        
        with a_col2:
            st.write(f"**Audio Source:** `{a_data.get('audio_source')}`")
            st.write(f"**Status:** `{a_data.get('status')}`")
            wav_p = a_data.get("wav_path")
            if wav_p and Path(wav_p).exists():
                try:
                    st.audio(wav_p)
                except Exception:
                    pass
    else:
        st.info("Audio analysis not available for this record.")

    # Forensic Findings & Anomaly Details
    st.markdown("---")
    st.subheader("3. Forensic Findings & Evidence Signals")
    findings = v_data.get("findings", [])
    if findings:
        for f in findings:
            sev = f.get("severity", "info").lower()
            icon = "🚨" if sev in ("critical", "high") else "⚠️" if sev == "medium" else "ℹ️"
            with st.expander(f"{icon} {f.get('title', 'Forensic Finding')}", expanded=True):
                st.write(f.get("description", ""))
                st.caption(f"Category: {f.get('category')} | Confidence: {f.get('confidence', 'N/A')}")
    else:
        st.write("No anomalous visual artifacts exceeded forensic severity thresholds.")

    # Technical Details
    with st.expander("Technical Metadata & Diagnostics", expanded=False):
        st.json(result, expanded=False)
