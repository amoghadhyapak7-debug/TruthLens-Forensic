"""Multi-Modal Correlation View for TruthLens Monitor.

Improved clarity for the Video & Audio Signal Synthesis section.
This view COMBINES and PRESENTS existing forensic findings — it does NOT
generate, manipulate, or alter any media.
"""
from __future__ import annotations

import requests
import streamlit as st


def _anomaly_human_label(anomaly_type: str) -> str:
    """Translate technical anomaly codes to investigator-readable labels."""
    labels = {
        "FULL_MULTIMODAL_SYNTHETIC": "Both Visual + Audio Synthetic",
        "VISUAL_ONLY_MANIPULATION": "Visual Manipulation Only",
        "AUDIO_ONLY_MANIPULATION": "Audio Manipulation Only",
        "NO_SYNTHETIC_INDICATORS": "No Synthetic Indicators Found",
        "SILENT_VIDEO": "Silent Video (No Audio Track)",
        "PARTIAL_OR_INCONCLUSIVE": "Partial / Inconclusive",
        "NONE": "No Anomaly Classification",
    }
    return labels.get(anomaly_type, anomaly_type)


def _audio_verdict_explanation(audio_verdict: str) -> str:
    """Explain WHY an audio verdict might be NOT_APPLICABLE or INCONCLUSIVE."""
    explanations = {
        "NOT_APPLICABLE": "The video container has no audio track. Audio forensic analysis cannot be performed on silent video files.",
        "SILENT_VIDEO": "The video container has no audio track. Audio forensic analysis cannot be performed on silent video files.",
        "EXTRACTION_FAILED": "Audio extraction from the video container failed. The audio track may be corrupted, encoded in an unsupported codec, or inaccessible.",
        "INCONCLUSIVE": "The audio analysis could not reach a confident classification. The audio may be too short, heavily compressed, noisy, or clipped.",
        "SYNTHETIC_DETECTED": "Acoustic features exhibit patterns consistent with synthetic speech generation (voice cloning, text-to-speech, or AI audio synthesis).",
        "NOT_DETECTED": "Acoustic feature patterns are consistent with natural/organic speech. No strong synthetic indicators were identified.",
    }
    return explanations.get(audio_verdict, "Audio analysis status is unknown or was not performed.")


def render_correlate_view(api_url: str, all_records: list[dict]):
    st.header("🔗 Multi-Modal Forensic Correlator")
    st.caption("Cross-modality signal correlation, conflicting evidence analysis, and case timeline")

    if not all_records:
        st.info("No evidence registered in the system. Ingest evidence items first.")
        return

    # ─── Section 1: Video & Audio Combined Forensic Summary ─────────────
    st.subheader("1. Combined Video + Audio Forensic Summary")
    st.caption(
        "This section **combines existing forensic findings** from the visual analyzer "
        "(GenD CLIP ViT-L/14) and the audio analyzer (Wav2Vec2 ONNX) into a unified summary. "
        "It does **not** generate, synthesize, or manipulate any media."
    )

    vid_records = [r for r in all_records if r.get("file_type") == "video"]
    if vid_records:
        ev_opts = {f"{r.get('evidence_id')} - {r.get('original_filename')}": r.get("evidence_id") for r in vid_records}
        chosen_label = st.selectbox("Select Video Evidence", list(ev_opts.keys()), key="corr_vid_select")
        chosen_id = ev_opts[chosen_label]

        if st.button("📊 Generate Combined Video + Audio Forensic Summary", key=f"btn_corr_{chosen_id}"):
            try:
                r = requests.get(f"{api_url}/api/v1/correlation/{chosen_id}", timeout=10)
                if r.status_code == 200:
                    st.session_state[f"corr_res_{chosen_id}"] = r.json()
                else:
                    st.error(f"Correlation failed: {r.text}")
            except Exception as exc:
                st.error(f"Correlation error: {exc}")

        c_data = st.session_state.get(f"corr_res_{chosen_id}")
        if c_data:
            # ── Combined Conclusion ──
            st.markdown("---")
            st.markdown("### 🏷️ Combined Multi-Modal Conclusion")
            u_verdict = c_data.get("unified_verdict", "INCONCLUSIVE")
            if "DEEPFAKE" in u_verdict or "MANIPULATION" in u_verdict or "SYNTHETIC AUDIO" in u_verdict:
                st.error(f"🚨 **{u_verdict}**")
            elif "NO SYNTHETIC" in u_verdict:
                st.success(f"✅ **{u_verdict}**")
            else:
                st.warning(f"⚠️ **{u_verdict}**")

            # ── What this means ──
            forensic_synthesis = c_data.get("forensic_synthesis", "")
            if forensic_synthesis:
                st.markdown("**What this means:**")
                st.info(forensic_synthesis)

            # ── Anomaly Classification (with human-readable label) ──
            anomaly_type = c_data.get("anomaly_type", "")
            human_label = _anomaly_human_label(anomaly_type)
            st.write(f"**Anomaly Classification:** {human_label} (`{anomaly_type}`)")

            # ── Visual vs Audio Side-by-Side ──
            st.markdown("---")
            st.markdown("### Individual Modality Findings")
            col_v, col_a = st.columns(2)

            with col_v:
                st.markdown("#### 👁️ Visual Analysis (GenD ViT-L/14)")
                v_verdict = c_data.get("visual_verdict", "N/A")
                v_score = c_data.get("visual_score")

                # Verdict display
                if "DEEPFAKE DETECTED" in str(v_verdict) and "NO" not in str(v_verdict):
                    st.error(f"🚨 **{v_verdict}**")
                    st.caption("Visual frames exhibit patterns consistent with deepfake generation or face manipulation.")
                elif "NO DEEPFAKE" in str(v_verdict):
                    st.success(f"✅ **{v_verdict}**")
                    st.caption("Visual frames do not exhibit deepfake generation patterns within the detector's training distribution.")
                else:
                    st.warning(f"⚠️ **{v_verdict}**")
                    st.caption("Visual analysis was inconclusive or produced a borderline result.")

                if v_score is not None:
                    try:
                        st.metric("Synthetic Probability", f"{float(v_score):.1%}")
                    except (ValueError, TypeError):
                        st.write(f"**Score:** {v_score}")
                st.caption("**Model:** GenD CLIP ViT-L/14 (WACV 2026)")

            with col_a:
                st.markdown("#### 🎙️ Audio Analysis (Wav2Vec2 ONNX)")
                a_verdict = c_data.get("audio_verdict", "N/A")
                a_score = c_data.get("audio_score")

                # Verdict display with clear status
                if a_verdict == "SYNTHETIC_DETECTED":
                    st.error(f"🚨 **SYNTHETIC AUDIO INDICATORS DETECTED**")
                elif a_verdict == "NOT_DETECTED":
                    st.success(f"✅ **NO STRONG SYNTHETIC INDICATORS**")
                elif a_verdict in ("NOT_APPLICABLE", "SILENT_VIDEO"):
                    st.info(f"ℹ️ **AUDIO: NOT APPLICABLE**")
                elif a_verdict == "EXTRACTION_FAILED":
                    st.warning(f"⚠️ **AUDIO EXTRACTION FAILED**")
                elif a_verdict == "INCONCLUSIVE":
                    st.warning(f"⚠️ **AUDIO: INCONCLUSIVE**")
                else:
                    st.warning(f"⚠️ **{a_verdict}**")

                # Clear explanation of WHY this verdict
                explanation = _audio_verdict_explanation(a_verdict)
                st.caption(explanation)

                if a_score is not None:
                    try:
                        st.metric("Synthetic Score", f"{float(a_score):.1%}")
                    except (ValueError, TypeError):
                        st.write(f"**Score:** {a_score}")
                elif a_verdict not in ("NOT_APPLICABLE", "SILENT_VIDEO", "EXTRACTION_FAILED"):
                    st.caption("Score unavailable.")

                audio_analyzed = a_verdict not in ("NOT_APPLICABLE", "SILENT_VIDEO", "EXTRACTION_FAILED")
                st.caption(f"**Audio analyzed:** {'Yes' if audio_analyzed else 'No'} · **Model:** Wav2Vec2 ONNX Int4")

            # ── Conflicting Signals Warning ──
            conflicts = c_data.get("conflicting_signals", [])
            if conflicts:
                st.markdown("---")
                st.warning("⚠️ **Conflicting Multi-Modal Signals Detected**")
                for conf in conflicts:
                    st.write(f"• {conf}")
                st.caption(
                    "When visual and audio verdicts disagree, it may indicate selective "
                    "manipulation (e.g., face replacement with original audio, or audio "
                    "re-dubbing over authentic video). This warrants closer human review."
                )

            # ── Supporting Findings / Limitations ──
            limitations = c_data.get("limitations", [])
            if limitations:
                with st.expander("📋 Methodology & Limitations", expanded=False):
                    st.caption(
                        "This combined summary is derived from existing analysis results. "
                        "It does not perform any additional inference or media processing."
                    )
                    for lim in limitations:
                        st.write(f"• {lim}")
    else:
        st.info("Upload and analyze a video file to view combined video + audio forensic correlation.")

    # ─── Section 2: Case-Wide Correlation Overview ──────────────────────
    st.markdown("---")
    st.subheader("2. Case-Wide Evidence Timeline & Modality Coverage")
    try:
        case_corr_resp = requests.get(f"{api_url}/api/v1/correlation/case", timeout=10)
        if case_corr_resp.status_code == 200:
            case_corr = case_corr_resp.json()
            cov = case_corr.get("cross_modality_coverage", {})

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Video Evidence", "Present" if cov.get("has_video") else "Absent")
            c2.metric("Image Evidence", "Present" if cov.get("has_image") else "Absent")
            c3.metric("Audio Evidence", "Present" if cov.get("has_audio") else "Absent")
            c4.metric("Text Evidence", "Present" if cov.get("has_text") else "Absent")

            st.markdown("#### Ingestion Timeline")
            t_list = case_corr.get("timeline", [])
            if t_list:
                st.dataframe(t_list, hide_index=True, width="stretch")
    except Exception:
        st.info("Case correlation summary unavailable.")
