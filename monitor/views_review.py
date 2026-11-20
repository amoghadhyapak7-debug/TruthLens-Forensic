"""Forensic Examiner Review View for TruthLens Monitor."""
from __future__ import annotations

import requests
import streamlit as st


def _build_auto_findings(ev_id: str, record: dict, analysis: dict, correlation: dict, review: dict) -> str:
    """Build a readable, factual auto-generated forensic summary from actual data.
    Never invents findings — only surfaces what exists in the provided data.
    """
    lines = []

    # --- INGEST ---
    lines.append("=== INGEST ===")
    lines.append(f"Evidence ID     : {record.get('evidence_id', 'N/A')}")
    lines.append(f"Original File   : {record.get('original_filename', 'N/A')}")
    lines.append(f"File Type       : {record.get('file_type', 'N/A')}")
    size = record.get("file_size") or 0
    lines.append(f"File Size       : {size:,} bytes ({size/1024:.1f} KB)")
    lines.append(f"SHA-256         : {record.get('sha256', 'N/A')}")
    lines.append(f"Upload Time     : {record.get('ingestion_timestamp', 'N/A')}")
    lines.append(f"Case ID         : {record.get('case_id') or 'No case assigned'}")
    lines.append(f"Validation      : {record.get('validation_status', 'N/A')}")
    lines.append(f"Analysis Eligible: {record.get('analysis_eligibility', 'N/A')}")

    # --- ANALYZE ---
    lines.append("")
    lines.append("=== ANALYSIS ===")
    f_type = record.get("file_type", "")

    if not analysis:
        lines.append("No analysis result available. Analyzer has not been run yet.")
    elif f_type == "video":
        try:
            from app.analyzers.normalization import normalize_video_verdict
            v_obj = analysis.get("video") if isinstance(analysis, dict) and "video" in analysis else analysis
            if v_obj:
                v_norm = normalize_video_verdict(v_obj)
                lines.append(f"Visual Verdict          : {v_norm.get('verdict', 'N/A')}")
                lines.append(f"Synthetic Probability   : {v_norm.get('fake_probability', 0):.4f}")
                lines.append(f"Confidence              : {v_norm.get('confidence', 0):.4f}")
                lines.append(f"Model                   : {v_norm.get('model', 'N/A')}")
                frames = v_obj.get("frames_analyzed") if isinstance(v_obj, dict) else None
                faces = v_obj.get("faces_detected") if isinstance(v_obj, dict) else None
                if frames is not None:
                    lines.append(f"Frames Analyzed         : {frames}")
                if faces is not None:
                    lines.append(f"Faces Detected          : {faces}")
                indicators = (v_obj.get("indicators") or v_obj.get("suspicious_indicators") or []) if isinstance(v_obj, dict) else []
                if indicators:
                    lines.append(f"Suspicious Indicators   : {', '.join(str(i) for i in indicators[:5])}")
        except Exception:
            lines.append("Video analysis result available but could not be parsed for summary.")

        a_obj = analysis.get("audio") if isinstance(analysis, dict) else {}
        a_status = analysis.get("audio_extraction_status", "N/A") if isinstance(analysis, dict) else "N/A"
        lines.append(f"Audio Extraction Status : {a_status}")
        if a_obj and isinstance(a_obj, dict):
            a_verdict = a_obj.get("result") or a_obj.get("verdict") or "N/A"
            lines.append(f"Audio Verdict           : {a_verdict}")
            transcript = a_obj.get("transcript") or a_obj.get("transcription")
            if transcript:
                t_str = str(transcript)[:200] + ("..." if len(str(transcript)) > 200 else "")
                lines.append(f"Audio Transcript (excerpt): {t_str}")
        else:
            lines.append("Audio Analysis          : Not available (no audio extracted or analyzed)")

    elif f_type == "image":
        verdict = analysis.get("verdict", "N/A")
        manip = analysis.get("manipulation") or {}
        lines.append(f"Image Verdict           : {verdict}")
        if isinstance(manip, dict):
            lines.append(f"Manipulation Score      : {manip.get('manipulation_score', 'N/A')}")
            lines.append(f"Manipulation Type       : {manip.get('manipulation_type', 'N/A')}")
        model = analysis.get("model_name") or analysis.get("model") or "N/A"
        lines.append(f"Model                   : {model}")
        indicators = analysis.get("indicators") or analysis.get("suspicious_indicators") or []
        if indicators:
            lines.append(f"Indicators              : {', '.join(str(i) for i in indicators[:5])}")

    elif f_type == "audio":
        verdict = analysis.get("verdict") or {}
        conclusion = verdict.get("conclusion") if isinstance(verdict, dict) else analysis.get("result", "N/A")
        lines.append(f"Audio Deepfake Verdict  : {conclusion}")
        lines.append(f"Score (Synthetic)       : {analysis.get('fake_score', 'N/A')}")
        lines.append(f"Score (Genuine)         : {analysis.get('genuine_score', 'N/A')}")
        lines.append(f"Score Band              : {analysis.get('score_band', 'N/A')}")
        model_info = analysis.get("model_info") or {}
        if isinstance(model_info, dict):
            lines.append(f"Model                   : {model_info.get('model_name', 'N/A')}")
        lines.append(f"Duration (s)            : {analysis.get('duration_seconds', 'N/A')}")
        transcript = analysis.get("transcript") or analysis.get("transcription")
        if transcript:
            t_str = str(transcript)[:200] + ("..." if len(str(transcript)) > 200 else "")
            lines.append(f"Transcript (excerpt)    : {t_str}")

    elif f_type in ("text", "document", "structured_data"):
        analyses = analysis.get("analyses", {}) if isinstance(analysis, dict) else {}
        ai_det = analysis.get("ai_detection") or analyses.get("ai_detector") or {}
        entities = analysis.get("entities") or analyses.get("entity_extraction", {}).get("entities") or []
        claims = analysis.get("claims") or analyses.get("claim_extraction", {}).get("claims") or []
        lines.append(f"AI-Text Probability     : {ai_det.get('ai_probability', 'N/A') if isinstance(ai_det, dict) else 'N/A'}")
        lines.append(f"AI Verdict / Status     : {ai_det.get('verdict') or ai_det.get('status') or 'N/A' if isinstance(ai_det, dict) else 'N/A'}")
        lines.append(f"Entities Found          : {len(entities) if isinstance(entities, (list, dict)) else 'N/A'}")
        lines.append(f"Claims Identified       : {len(claims) if isinstance(claims, (list, dict)) else 'N/A'}")

    else:
        lines.append(f"Analysis result present for type '{f_type}'. Consult raw analysis output.")

    # --- CORRELATE ---
    lines.append("")
    lines.append("=== CORRELATION ===")
    if f_type != "video":
        lines.append("Correlation: N/A (applies to video evidence only)")
    elif not correlation:
        lines.append("No correlation result found. Run video-audio correlation in the CORRELATE stage.")
    else:
        lines.append(f"Unified Verdict         : {correlation.get('unified_verdict') or correlation.get('verdict') or 'N/A'}")
        lines.append(f"Visual Verdict          : {correlation.get('visual_verdict', 'N/A')}")
        lines.append(f"Audio Verdict           : {correlation.get('audio_verdict', 'N/A')}")
        lines.append(f"Anomaly Classification  : {correlation.get('anomaly_type') or correlation.get('anomaly_classification') or 'N/A'}")
        synthesis = correlation.get("forensic_synthesis") or correlation.get("synthesis") or ""
        if synthesis:
            lines.append(f"Forensic Synthesis      : {str(synthesis)[:300]}")
        conflicts = correlation.get("conflicts") or []
        if conflicts:
            lines.append(f"Conflicts Detected      : {'; '.join(str(c) for c in conflicts[:3])}")
        limitations = correlation.get("limitations") or []
        if limitations:
            lines.append(f"Limitations             : {'; '.join(str(l) for l in limitations[:3])}")

    # --- PRIOR REVIEW ---
    lines.append("")
    lines.append("=== PRIOR REVIEW ===")
    prior_status = review.get("status", "UNREVIEWED") if review else "UNREVIEWED"
    prior_examiner = review.get("examiner") or "Not recorded" if review else "Not recorded"
    prior_updated = review.get("updated_at") or "No prior review" if review else "No prior review"
    lines.append(f"Prior Status            : {prior_status}")
    lines.append(f"Prior Examiner          : {prior_examiner}")
    lines.append(f"Prior Review Time       : {prior_updated}")

    return "\n".join(lines)


def render_review_view(api_url: str, all_records: list[dict]):
    st.header("⚖️ Forensic Examiner Review")
    st.caption("Human-in-the-loop audit, verification status sign-off, and examiner annotations")

    if not all_records:
        st.info("No evidence registered to review.")
        return

    ev_options = {f"{r.get('evidence_id')} - {r.get('original_filename')}": r.get("evidence_id") for r in all_records}
    selected_label = st.selectbox("Select Evidence Item to Audit", list(ev_options.keys()), key="review_ev_select")
    evidence_id = ev_options[selected_label]

    # Fetch all data for auto-population
    curr_review: dict = {}
    ev_record: dict = {}
    analysis_result: dict = {}
    correlation_result: dict = {}

    try:
        r = requests.get(f"{api_url}/api/v1/review/{evidence_id}", timeout=5)
        if r.status_code == 200:
            curr_review = r.json()
    except Exception:
        pass

    try:
        r = requests.get(f"{api_url}/api/v1/evidence/{evidence_id}", timeout=5)
        if r.status_code == 200:
            ev_record = r.json()
    except Exception:
        pass

    try:
        r = requests.get(f"{api_url}/api/v1/analyze/results/{evidence_id}", timeout=10)
        if r.status_code == 200:
            analysis_result = r.json().get("result", {})
    except Exception:
        pass

    f_type_check = ev_record.get("file_type", "")
    if f_type_check == "video":
        try:
            r = requests.get(f"{api_url}/api/v1/correlation/{evidence_id}", timeout=5)
            if r.status_code == 200:
                correlation_result = r.json()
        except Exception:
            pass

    st.write(
        f"**Target Evidence:** `{evidence_id}` &bull; "
        f"**Type:** `{ev_record.get('file_type', 'unknown').upper()}` &bull; "
        f"**Current Status:** `{curr_review.get('status', 'UNREVIEWED')}`"
    )

    # Build auto-findings text
    auto_findings_text = ""
    if ev_record:
        auto_findings_text = _build_auto_findings(
            evidence_id, ev_record, analysis_result, correlation_result, curr_review
        )

    # ── Auto-Generated Summary (read-only) ──────────────────────────────
    if auto_findings_text:
        st.markdown("---")
        st.markdown(
            "#### 🤖 Auto-Generated Forensic Summary *(System-Generated — Not Human-Verified)*",
        )
        st.info(
            "This summary is automatically compiled from INGEST, ANALYZE, CORRELATE, and REVIEW stage data. "
            "It is a machine-generated digest of existing findings and must not be treated as a human expert opinion."
        )
        st.text_area(
            label="Auto-Generated Forensic Summary (Read-Only)",
            value=auto_findings_text,
            height=320,
            disabled=True,
            key=f"review_auto_{evidence_id}",
        )

    # ── Review Form ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📝 Verification Decision & Examiner Input")

    # Pre-fill investigator opinion from saved record (if any)
    saved_opinion = curr_review.get("investigator_opinion") or ""

    with st.form(key=f"review_form_{evidence_id}"):
        col1, col2 = st.columns(2)
        with col1:
            status_choice = st.selectbox(
                "Verification Decision",
                options=["UNREVIEWED", "VERIFIED", "DISPUTED", "FLAGGED_FOR_ESCALATION"],
                index=["UNREVIEWED", "VERIFIED", "DISPUTED", "FLAGGED_FOR_ESCALATION"].index(
                    curr_review.get("status", "UNREVIEWED")
                ) if curr_review.get("status") in ["UNREVIEWED", "VERIFIED", "DISPUTED", "FLAGGED_FOR_ESCALATION"] else 0,
                key=f"review_status_select_{evidence_id}",
            )
        with col2:
            examiner_name = st.text_input(
                "Examiner Name / Badge",
                value=curr_review.get("examiner") or "Forensic Examiner",
                key=f"review_examiner_input_{evidence_id}",
            )

        st.markdown("**Add Your Opinion** *(Optional — manually entered, clearly distinguished from automated summary above)*")
        investigator_opinion = st.text_area(
            "Add Your Opinion (Optional)",
            value=saved_opinion,
            height=120,
            placeholder="Enter your expert opinion, additional observations, or case notes here. Leave blank if you have no additional remarks.",
            key=f"review_opinion_{evidence_id}",
        )

        submitted = st.form_submit_button("💾 Save Official Review Determination", type="primary")

    if submitted:
        # Combined notes = auto summary + opinion for backward compatibility
        combined_notes = ""
        if auto_findings_text:
            combined_notes += "[AUTO-GENERATED SUMMARY]\n" + auto_findings_text
        if investigator_opinion.strip():
            combined_notes += "\n\n[INVESTIGATOR OPINION]\n" + investigator_opinion.strip()

        try:
            resp = requests.post(
                f"{api_url}/api/v1/review/{evidence_id}",
                json={
                    "status": status_choice,
                    "notes": combined_notes,
                    "examiner": examiner_name,
                    "auto_findings": auto_findings_text,
                    "investigator_opinion": investigator_opinion.strip(),
                },
                timeout=5,
            )
            if resp.status_code == 200:
                st.success(f"✅ Review saved for {evidence_id} as **{status_choice}**!")
                st.rerun()
            else:
                st.error(f"Failed to save review: {resp.text}")
        except Exception as exc:
            st.error(f"Error submitting review: {exc}")

    # ── Audit Log ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Audit Log & Case Determinations")
    try:
        all_revs_resp = requests.get(f"{api_url}/api/v1/reviews", timeout=5)
        if all_revs_resp.status_code == 200:
            rev_map = all_revs_resp.json().get("reviews", {})
            if rev_map:
                rev_list = [
                    {
                        "Evidence ID": k,
                        "Status": v.get("status"),
                        "Examiner": v.get("examiner"),
                        "Updated (UTC)": v.get("updated_at"),
                        "Has Auto-Summary": "Yes" if v.get("auto_findings") else "No",
                        "Has Opinion": "Yes" if v.get("investigator_opinion") else "No",
                    }
                    for k, v in rev_map.items()
                ]
                st.dataframe(rev_list, hide_index=True, width="stretch")
            else:
                st.caption("No official reviews entered yet.")
    except Exception:
        pass
