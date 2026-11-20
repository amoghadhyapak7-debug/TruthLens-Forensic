"""Unified Multi-Modal Analysis View for TruthLens Monitor.

Presents all analysis findings in one central place.
The user never selects an analyzer manually; routing is automatic.
"""
from __future__ import annotations

from pathlib import Path
import json
import logging
import requests
import streamlit as st

try:
    from app.analyzers.normalization import normalize_video_verdict
except ImportError:
    # Inline fallback if app is not directly in python path
    def normalize_video_verdict(v_data):
        if isinstance(v_data, str):
            try:
                parsed = json.loads(v_data)
                if isinstance(parsed, dict):
                    return normalize_video_verdict(parsed)
            except Exception:
                pass
            p = 0.95 if ("DEEPFAKE" in v_data and "NO" not in v_data) else 0.05
            return {"verdict": v_data, "fake_probability": p, "confidence_score": p, "basis": [], "models_used": ["GenD CLIP ViT-L/14"]}
        if not isinstance(v_data, dict):
            return {"verdict": "INCONCLUSIVE", "fake_probability": 0.0, "confidence_score": 0.0, "basis": [], "models_used": ["GenD CLIP ViT-L/14"]}
        if "video" in v_data and isinstance(v_data["video"], (dict, str)):
            return normalize_video_verdict(v_data["video"])
        raw_v = v_data.get("verdict")
        asmt = v_data.get("assessment") if isinstance(v_data.get("assessment"), dict) else {}
        v_str = raw_v if isinstance(raw_v, str) else (raw_v.get("verdict") if isinstance(raw_v, dict) else None)
        if not v_str:
            v_str = asmt.get("verdict") or asmt.get("classification") or "INCONCLUSIVE"
        prob = None
        if isinstance(raw_v, dict):
            prob = raw_v.get("fake_probability")
        if prob is None and asmt.get("manipulation_score_pct") is not None:
            prob = float(asmt["manipulation_score_pct"]) / 100.0
        elif prob is None and asmt.get("evidence_score") is not None:
            prob = float(asmt["evidence_score"])
        return {
            "verdict": str(v_str),
            "fake_probability": float(prob or 0.0),
            "confidence_score": float(asmt.get("confidence") or prob or 0.0),
            "basis": asmt.get("basis", []),
            "models_used": v_data.get("models_used", ["GenD CLIP ViT-L/14", "MesoNet-4", "YuNet"]),
        }



def render_analyze_view(api_url: str, all_records: list[dict]):
    st.header("🔍 Multi-Modal Forensic Analysis")
    st.caption("Unified forensic findings — analyzers operate automatically based on evidence type")

    if not all_records:
        st.info("No evidence registered yet. Upload evidence files in the **INGEST** tab first.")
        return

    # Filter controls
    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
    with f_col1:
        mod_filter = st.selectbox("Filter by Type", ["All", "video", "image", "audio", "text/document"], key="ana_mod_filter")
    with f_col2:
        cases = sorted(list({r.get("case_id") for r in all_records if r.get("case_id")}))
        case_filter = st.selectbox("Filter by Case", ["All"] + cases, key="ana_case_filter")
    with f_col3:
        st.write("")
        st.write("")
        if st.button("⚡ Run Analysis on All Evidence", type="secondary", key="run_all_batch"):
            with st.spinner("Executing automated analysis across all evidence items..."):
                try:
                    resp = requests.post(f"{api_url}/api/v1/analyze/batch", timeout=600)
                    if resp.status_code == 200:
                        st.success("Batch analysis complete!")
                        st.rerun()
                    else:
                        st.error(f"Batch analysis error: {resp.text}")
                except Exception as exc:
                    st.error(f"Batch request failed: {exc}")

    # Apply filters
    filtered = all_records
    if mod_filter != "All":
        if mod_filter == "text/document":
            filtered = [r for r in filtered if r.get("file_type") in ("text", "document", "structured_data")]
        else:
            filtered = [r for r in filtered if r.get("file_type") == mod_filter]
    if case_filter != "All":
        filtered = [r for r in filtered if r.get("case_id") == case_filter]

    st.write(f"Showing **{len(filtered)}** evidence item(s):")

    for record in filtered:
        ev_id = record.get("evidence_id")
        orig_name = record.get("original_filename", "unknown")
        f_type = record.get("file_type", "unknown")
        sha = record.get("sha256", "")

        # Check existing result
        res_data = None
        try:
            r = requests.get(f"{api_url}/api/v1/analyze/results/{ev_id}", timeout=2)
            if r.status_code == 200:
                res_data = r.json()
        except Exception:
            pass

        has_result = res_data is not None and "result" in res_data
        status_icon = "✅" if has_result else "⏳"
        status_text = "ANALYZED" if has_result else "PENDING ANALYSIS"

        with st.expander(f"{status_icon} **{ev_id}** · {orig_name} · *{f_type.upper()}* [{status_text}]", expanded=not has_result):
            st.write(f"**SHA-256:** `{sha}` &bull; **Size:** {record.get('file_size', 0):,} bytes &bull; **Case:** `{record.get('case_id') or 'N/A'}`")

            # Run button if not analyzed or to re-run
            btn_col, _ = st.columns([1, 3])
            with btn_col:
                btn_label = "🔄 Re-run Analysis" if has_result else "🚀 Run Automated Analysis"
                if st.button(btn_label, key=f"run_btn_{ev_id}"):
                    with st.spinner(f"Automatically analyzing {f_type} evidence via backend pipeline..."):
                        try:
                            resp = requests.post(f"{api_url}/api/v1/analyze/auto/{ev_id}", timeout=600)
                            if resp.status_code == 200:
                                st.success("Analysis complete!")
                                st.rerun()
                            else:
                                st.error(f"Analysis failed: {resp.text}")
                        except Exception as exc:
                            st.error(f"Analysis error: {exc}")

            if not has_result:
                st.info("Analysis has not been executed yet. Click above to run automated forensic routing.")
                continue

            raw_res = res_data.get("result", {})
            routed_name = res_data.get("routed_analyzer", f_type.capitalize() + " Analyzer")
            st.caption(f"**Automated Backend Route:** `{routed_name}`")

            # Render by file type
            if f_type == "video":
                _render_video_card(api_url, ev_id, record, raw_res)
            elif f_type == "image":
                _render_image_card(api_url, ev_id, record, raw_res)
            elif f_type == "audio":
                _render_audio_card(api_url, ev_id, record, raw_res)
            elif f_type in ("text", "document", "structured_data"):
                _render_text_card(api_url, ev_id, record, raw_res)


def _render_video_card(api_url: str, ev_id: str, record: dict, data: dict):
    # Support both wrapped and flat result
    v_data = data.get("video") if (isinstance(data, dict) and "video" in data) else data
    a_data = data.get("audio") if isinstance(data, dict) else None
    a_status = data.get("audio_extraction_status", "unknown") if isinstance(data, dict) else "unknown"

    # Normalize verdict data defensively
    norm_verdict = normalize_video_verdict(v_data)
    verdict_str = norm_verdict["verdict"]
    fake_prob = norm_verdict["fake_probability"]
    conf_score = norm_verdict["confidence_score"]
    models_list = norm_verdict.get("models_used", ["GenD CLIP ViT-L/14", "MesoNet-4", "YuNet"])

    # 1. Visual Verdict
    st.markdown("#### 1. Visual Forensic Verdict (GenD CLIP ViT-L/14)")
    c1, c2, c3 = st.columns(3)
    with c1:
        if "DEEPFAKE DETECTED" in verdict_str and "NO" not in verdict_str:
            st.error(f"🚨 **{verdict_str}**")
        elif "NO DEEPFAKE" in verdict_str:
            st.success(f"✅ **{verdict_str}**")
        else:
            st.warning(f"⚠️ **{verdict_str}**")
    with c2:
        st.metric("Synthetic Probability", f"{fake_prob:.1%}")
        st.progress(min(max(float(fake_prob), 0.0), 1.0))
    with c3:
        st.write(f"**Models:** {', '.join(models_list)}")
        st.caption(f"Confidence: {conf_score:.1%}")

    # 2. Extracted Audio Stream
    st.markdown("#### 2. Extracted Audio Track Analysis (Wav2Vec2 ONNX)")
    if a_status == "no_audio_stream":
        st.info("ℹ️ **Silent Video:** Video container has no audio stream. Audio analysis is Not Applicable.")
    elif a_status == "extraction_failed":
        st.warning("⚠️ **Audio Extraction Failed:** Could not extract audio track.")
    elif a_data and a_status == "extracted" and isinstance(a_data, dict):
        ac1, ac2 = st.columns(2)
        with ac1:
            a_res = str(a_data.get("result", "UNKNOWN"))
            if "SYNTHETIC" in a_res and "NO" not in a_res:
                st.error(f"🚨 **{a_res}**")
            elif "NO STRONG" in a_res:
                st.success(f"✅ **{a_res}**")
            else:
                st.warning(f"⚠️ **{a_res}**")
            a_score = a_data.get("synthetic_score")
            if a_score is not None:
                try:
                    st.metric("Audio Synthetic Score", f"{float(a_score):.1%}")
                except (ValueError, TypeError):
                    pass
        with ac2:
            wav_p = a_data.get("wav_path")
            if wav_p and Path(wav_p).exists():
                try:
                    st.audio(wav_p)
                except Exception:
                    pass

    # 3. Forensic Basis / Findings if present
    basis = norm_verdict.get("basis", [])
    if basis:
        with st.expander("Model Classification Basis & Insights", expanded=False):
            for b in basis:
                st.write(f"• {b}")

    # Video preview
    try:
        st.video(f"{api_url}/api/v1/evidence/{ev_id}/content")
    except Exception:
        pass


def _render_image_card(api_url: str, ev_id: str, record: dict, data: dict):
    if not isinstance(data, dict):
        st.warning(f"Unexpected image analysis format: {type(data).__name__}")
        return

    verdict_str = str(data.get("verdict", "INCONCLUSIVE"))
    confidence = data.get("confidence", 0.0)
    ai_prob = data.get("ai_generated_probability", 0.0)

    c1, c2, c3 = st.columns(3)
    with c1:
        if "AI-GENERATED" in verdict_str or "MANIPULATED" in verdict_str:
            st.error(f"🚨 **{verdict_str}**")
        elif "AUTHENTIC" in verdict_str or "CLEAR" in verdict_str:
            st.success(f"✅ **{verdict_str}**")
        else:
            st.warning(f"⚠️ **{verdict_str}**")
    with c2:
        st.metric("Confidence", f"{confidence:.1%}" if isinstance(confidence, (int, float)) else str(confidence))
    with c3:
        st.metric("AI Probability", f"{ai_prob:.1%}" if isinstance(ai_prob, (int, float)) else str(ai_prob))

    # Analytical details
    ela_data = data.get("ela") if isinstance(data.get("ela"), dict) else {}
    noise_data = data.get("noise") if isinstance(data.get("noise"), dict) else {}
    st.write(f"**ELA Mean Difference:** `{ela_data.get('mean_diff', 'N/A')}` &bull; **Noise BVR:** `{noise_data.get('bvr', 'N/A')}`")

    try:
        st.image(f"{api_url}/api/v1/evidence/{ev_id}/content", width=350)
    except Exception:
        pass


def _render_audio_card(api_url: str, ev_id: str, record: dict, data: dict):
    if not isinstance(data, dict):
        st.warning(f"Unexpected audio analysis format: {type(data).__name__}")
        return

    verdict = data.get("verdict")
    if isinstance(verdict, dict):
        conc = verdict.get("conclusion") or data.get("result", "UNKNOWN")
        score = verdict.get("score") or data.get("synthetic_score", 0.0)
    elif isinstance(verdict, str):
        conc = verdict
        score = data.get("synthetic_score", 0.0)
    else:
        conc = data.get("result", "UNKNOWN")
        score = data.get("synthetic_score", 0.0)

    c1, c2 = st.columns(2)
    with c1:
        if "SYNTHETIC" in str(conc).upper() and "NO" not in str(conc).upper():
            st.error(f"🚨 **{conc}**")
        elif "NO STRONG" in str(conc).upper():
            st.success(f"✅ **{conc}**")
        else:
            st.warning(f"⚠️ **{conc}**")
    with c2:
        try:
            score_val = float(score)
            st.metric("Synthetic Score", f"{score_val:.1%}")
            st.progress(min(max(score_val, 0.0), 1.0))
        except (ValueError, TypeError):
            st.write(f"**Score:** {score}")

    try:
        st.audio(f"{api_url}/api/v1/evidence/{ev_id}/content")
    except Exception:
        pass


def _render_text_card(api_url: str, ev_id: str, record: dict, data: dict):
    if not isinstance(data, dict):
        st.warning(f"Unexpected text analysis format: {type(data).__name__}")
        return

    ai_data = data.get("ai_detection") if isinstance(data.get("ai_detection"), dict) else {}
    ai_prob = ai_data.get("ai_probability", 0.0)

    c1, c2 = st.columns(2)
    with c1:
        if isinstance(ai_prob, (int, float)) and ai_prob >= 0.70:
            st.error(f"🚨 **PROBABLE AI-GENERATED TEXT ({ai_prob:.1%})**")
        elif isinstance(ai_prob, (int, float)) and ai_prob <= 0.30:
            st.success(f"✅ **PROBABLE HUMAN-AUTHORED TEXT ({ai_prob:.1%})**")
        else:
            st.warning(f"⚠️ **INCONCLUSIVE TEXT ASSESSMENT**")
    with c2:
        if isinstance(ai_prob, (int, float)):
            st.progress(min(max(float(ai_prob), 0.0), 1.0))

    claims = data.get("claims") if isinstance(data.get("claims"), list) else []
    if claims:
        st.write(f"**Extracted Claims ({len(claims)}):**")
        for c in claims[:5]:
            c_text = c.get("claim_text", str(c)) if isinstance(c, dict) else str(c)
            st.write(f"• {c_text}")

