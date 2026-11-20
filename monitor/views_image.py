"""Image Forensics View for TruthLens Monitor."""
from __future__ import annotations

import requests
import streamlit as st
from pathlib import Path


def render_image_view(api_url: str, all_records: list[dict]):
    st.header("🖼️ Image Forensics Analyzer")
    st.caption("AI-generation & manipulation analysis powered by EfficientNet-B4, ResNet-50, ELA & Wavelet decomposition")

    image_records = [r for r in all_records if r.get("file_type") == "image"]
    if not image_records:
        st.info("No image evidence registered yet. Upload image files (.png, .jpg, .jpeg, .webp) in the **Evidence Ingest** tab first.")
        return

    ev_options = {f"{r.get('evidence_id')} - {r.get('original_filename')}": r for r in image_records}
    selected_label = st.selectbox("Select Image Evidence", options=list(ev_options.keys()), key="image_evidence_select")
    record = ev_options[selected_label]
    evidence_id = record.get("evidence_id")

    i_col1, i_col2 = st.columns([1, 1])
    with i_col1:
        st.write(f"**Evidence ID:** `{evidence_id}`")
        st.write(f"**Filename:** `{record.get('original_filename')}`")
        st.write(f"**SHA-256:** `{record.get('sha256', '')[:20]}...`")
        st.write(f"**File Size:** {record.get('file_size', 0):,} bytes")
    with i_col2:
        try:
            st.image(f"{api_url}/api/v1/evidence/{evidence_id}/content", use_container_width=True)
        except Exception:
            st.info("Image preview unavailable.")

    st.markdown("---")
    res_key = f"image_analysis_result_{evidence_id}"

    if res_key not in st.session_state:
        try:
            r = requests.get(f"{api_url}/api/v1/analyzers/image/{evidence_id}", timeout=5)
            if r.status_code == 200:
                st.session_state[res_key] = r.json()
        except Exception:
            pass

    if st.button("🚀 Run Image Forensic Analysis", type="primary", key=f"run_img_{evidence_id}"):
        with st.spinner("Analyzing image (EfficientNet-B4 + ResNet-50 + ELA + Wavelet)..."):
            try:
                resp = requests.post(f"{api_url}/api/v1/analyzers/image/{evidence_id}", timeout=300)
                if resp.status_code == 200:
                    st.session_state[res_key] = resp.json()
                    st.success("Analysis complete!")
                else:
                    st.error(f"Analysis failed: {resp.text}")
            except Exception as exc:
                st.error(f"Error during image analysis: {exc}")

    result = st.session_state.get(res_key)
    if not result:
        st.info("Click **Run Image Forensic Analysis** to perform neural and algorithmic inspection.")
        return

    # Verdict Card
    st.subheader("1. Forensic Verdict")
    verdict_str = result.get("verdict", "INCONCLUSIVE")
    confidence = result.get("confidence", 0.0)
    ai_prob = result.get("ai_generated_probability", 0.0)

    m1, m2, m3 = st.columns(3)
    with m1:
        if "AI-GENERATED" in verdict_str or "MANIPULATED" in verdict_str:
            st.error(f"🚨 **{verdict_str}**")
        elif "AUTHENTIC" in verdict_str or "CLEAR" in verdict_str:
            st.success(f"✅ **{verdict_str}**")
        else:
            st.warning(f"⚠️ **{verdict_str}**")
    with m2:
        st.metric("Confidence Score", f"{confidence:.1%}" if isinstance(confidence, (int, float)) else str(confidence))
    with m3:
        st.metric("AI Generation Probability", f"{ai_prob:.1%}" if isinstance(ai_prob, (int, float)) else str(ai_prob))

    # Analytical Details
    st.markdown("---")
    st.subheader("2. Detailed Forensic Checks")
    
    tab_ai, tab_ela, tab_noise, tab_meta = st.tabs(["AI Detection (EfficientNet-B4)", "Error Level Analysis (ELA)", "Noise & Wavelet", "EXIF & Metadata"])
    
    with tab_ai:
        ai_det = result.get("ai_detection", {})
        st.write(f"**Detector Model:** `timm/EfficientNet-B4`")
        st.write(f"**AI Score:** `{ai_det.get('score', ai_prob)}`")
        st.write(f"**Assessment:** {ai_det.get('assessment', 'N/A')}")
        
    with tab_ela:
        ela_data = result.get("ela", {})
        st.write(f"**ELA Mean Difference:** `{ela_data.get('mean_diff', 'N/A')}`")
        st.write(f"**Quality Factor:** `{ela_data.get('quality', 95)}`")
        st.caption("Error Level Analysis highlights compression discrepancies between spliced regions and authentic background.")

    with tab_noise:
        noise_data = result.get("noise", {})
        st.write(f"**Block Variance Ratio (BVR):** `{noise_data.get('bvr', 'N/A')}`")
        st.write(f"**Noise Inconsistency Detected:** `{noise_data.get('inconsistency_detected', False)}`")

    with tab_meta:
        meta = result.get("metadata", {})
        st.json(meta if isinstance(meta, dict) else {}, expanded=False)

    with st.expander("Technical Findings & Payload", expanded=False):
        st.json(result, expanded=False)
