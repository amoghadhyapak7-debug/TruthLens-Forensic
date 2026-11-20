"""Forensic Report Generation & Export View for TruthLens Monitor."""
from __future__ import annotations

import requests
import streamlit as st


def render_report_view(api_url: str, all_records: list[dict]):
    st.header("📑 Forensic Examination Report Generator")
    st.caption("Export tamper-evident, court-ready multi-modal investigation reports in PDF and JSON formats")

    if not all_records:
        st.info("No evidence registered in the system. Ingest evidence items first.")
        return

    # ── Case-Level Report Section ──────────────────────────────────────────
    st.subheader("📦 Case-Level Report Export")

    cases = sorted(list({r.get("case_id") for r in all_records if r.get("case_id")}))
    case_options = ["all"] + cases
    selected_case = st.selectbox(
        "Select Case Scope",
        options=case_options,
        format_func=lambda c: "All Registered Evidence" if c == "all" else f"Case: {c}",
        key="report_case_scope_select",
    )

    filtered_records = all_records if selected_case == "all" else [
        r for r in all_records if r.get("case_id") == selected_case
    ]
    st.write(f"**Included Evidence Artifacts:** {len(filtered_records)} items")
    st.dataframe(
        [
            {
                "Evidence ID": r.get("evidence_id"),
                "Filename": r.get("original_filename"),
                "Modality": r.get("file_type"),
                "SHA-256": (r.get("sha256") or "")[:16] + "...",
                "Size": f"{r.get('file_size', 0):,} B",
            }
            for r in filtered_records
        ],
        hide_index=True,
        width="stretch",
    )

    st.markdown("---")
    st.subheader("Download Comprehensive Case-Level Report")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📄 PDF Examination Report")
        st.caption("Official printable forensic documentation with executive summary, integrity tables, model cards, and caveats.")
        try:
            pdf_resp = requests.get(f"{api_url}/api/v1/report/{selected_case}?format=pdf", timeout=30)
            if pdf_resp.status_code == 200 and pdf_resp.content:
                st.download_button(
                    label="⬇️ Download PDF Forensic Report",
                    data=pdf_resp.content,
                    file_name=f"truthlens_report_{selected_case}.pdf",
                    mime="application/pdf",
                    type="primary",
                    key=f"report_dl_pdf_{selected_case}",
                )
            else:
                st.warning("Could not assemble PDF report.")
        except Exception as exc:
            st.error(f"Error fetching PDF: {exc}")

    with col2:
        st.markdown("#### 🗄️ Structured JSON Data Package")
        st.caption("Machine-readable JSON schema with complete findings for SIEM, case management, or secondary pipelines.")
        try:
            json_resp = requests.get(f"{api_url}/api/v1/report/{selected_case}?format=json", timeout=15)
            if json_resp.status_code == 200 and json_resp.content:
                st.download_button(
                    label="⬇️ Download JSON Data Package",
                    data=json_resp.content,
                    file_name=f"truthlens_report_{selected_case}.json",
                    mime="application/json",
                    key=f"report_dl_json_{selected_case}",
                )
            else:
                st.warning("Could not assemble JSON report.")
        except Exception as exc:
            st.error(f"Error fetching JSON: {exc}")

    # ── Per-Evidence Detail Inspector ─────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Per-Evidence Forensic Detail Inspector")
    st.caption(
        "Select an Evidence ID to view the complete forensic record (INGEST → ANALYZE → CORRELATE → REVIEW) "
        "and download an individual evidence-specific PDF or JSON report."
    )

    ev_map = {
        f"{r.get('evidence_id')} — {r.get('original_filename')}": r.get("evidence_id")
        for r in all_records
    }
    if not ev_map:
        st.info("No evidence items available.")
        return

    selected_ev_label = st.selectbox(
        "Select Evidence Item",
        list(ev_map.keys()),
        key="rpt_ev_selector",
    )
    sel_ev_id = ev_map[selected_ev_label]

    # Fetch all data for the selected evidence
    ev_record: dict = {}
    analysis_result: dict = {}
    correlation_result: dict = {}
    review_record: dict = {}

    try:
        r = requests.get(f"{api_url}/api/v1/evidence/{sel_ev_id}", timeout=5)
        if r.status_code == 200:
            ev_record = r.json()
    except Exception:
        pass

    try:
        r = requests.get(f"{api_url}/api/v1/analyze/results/{sel_ev_id}", timeout=10)
        if r.status_code == 200:
            analysis_result = r.json().get("result", {})
    except Exception:
        pass

    try:
        r = requests.get(f"{api_url}/api/v1/correlation/{sel_ev_id}", timeout=5)
        if r.status_code == 200:
            correlation_result = r.json()
    except Exception:
        pass

    try:
        r = requests.get(f"{api_url}/api/v1/review/{sel_ev_id}", timeout=5)
        if r.status_code == 200:
            review_record = r.json()
    except Exception:
        pass

    # ── INGEST detail ──
    with st.expander("📥 INGEST — Evidence Registration & Metadata", expanded=True):
        if ev_record:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Evidence ID:** `{ev_record.get('evidence_id', 'N/A')}`")
                st.markdown(f"**Original Filename:** {ev_record.get('original_filename', 'N/A')}")
                st.markdown(f"**File Type / Modality:** {ev_record.get('file_type', 'N/A')}")
                size_b = ev_record.get("file_size") or 0
                st.markdown(f"**File Size:** {size_b:,} bytes ({size_b/1024:.1f} KB)")
                st.markdown(f"**Upload Timestamp:** {ev_record.get('ingestion_timestamp', 'N/A')}")
            with col_b:
                st.markdown(f"**Case ID:** {ev_record.get('case_id') or 'No case assigned'}")
                sha = ev_record.get("sha256") or "N/A"
                st.markdown(f"**SHA-256:** `{sha}`")
                st.markdown(f"**Validation Status:** {ev_record.get('validation_status', 'N/A')}")
                st.markdown(f"**Analysis Eligibility:** {ev_record.get('analysis_eligibility', 'N/A')}")
                st.markdown(f"**Testing Evidence:** {ev_record.get('is_testing', False)}")

            meta = ev_record.get("metadata") or {}
            if meta:
                st.markdown("**Extracted File Metadata:**")
                st.json(meta)

            stages = ev_record.get("stages") or {}
            if stages:
                st.markdown("**Workflow Stages:**")
                st.json(stages)
        else:
            st.info(f"Could not load ingest record for {sel_ev_id}.")

    # ── ANALYZE detail ──
    with st.expander("🔬 ANALYZE — Full Forensic Analysis Findings", expanded=False):
        if analysis_result:
            st.json(analysis_result)
        elif ev_record:
            st.info("No analysis result found. Run the analyzer for this evidence first.")
        else:
            st.info("Evidence record unavailable; cannot check analysis status.")

    # ── CORRELATE detail ──
    f_type_sel = ev_record.get("file_type", "")
    with st.expander("🔗 CORRELATE — Cross-Modal Forensic Correlation", expanded=False):
        if f_type_sel == "video":
            if correlation_result:
                _render_correlation_ui(correlation_result)
            else:
                st.info("No correlation result found. Run video-audio correlation in the CORRELATE stage first.")
        else:
            st.info(f"Cross-modal correlation applies to video evidence only. This item is of type: **{f_type_sel or 'unknown'}**.")

    # ── REVIEW detail ──
    with st.expander("⚖️ REVIEW — Examiner Determination", expanded=False):
        if review_record:
            status = review_record.get("status", "UNREVIEWED")
            examiner = review_record.get("examiner") or "Not recorded"
            updated = review_record.get("updated_at") or "Not recorded"
            auto_f = review_record.get("auto_findings") or ""
            opinion = review_record.get("investigator_opinion") or ""
            notes = review_record.get("notes") or ""

            col_x, col_y = st.columns(2)
            with col_x:
                st.markdown(f"**Verification Status:** `{status}`")
                st.markdown(f"**Examiner:** {examiner}")
                st.markdown(f"**Last Updated:** {updated}")

            if auto_f:
                st.markdown("---")
                st.markdown("**🤖 Auto-Generated Forensic Summary** *(system-generated, not human-verified)*")
                st.info(auto_f)
            if opinion:
                st.markdown("---")
                st.markdown("**🧑‍💼 Investigator Opinion** *(manually entered)*")
                st.warning(opinion)
            if notes and not auto_f and not opinion:
                st.markdown("**Examiner Notes:**")
                st.write(notes)
        else:
            st.info("No review record found.")

    # ── Per-Evidence Download Buttons ──
    st.markdown("---")
    st.markdown("#### 📥 Download Evidence-Specific Forensic Report")
    col3, col4 = st.columns(2)
    with col3:
        try:
            pdf_r = requests.get(f"{api_url}/api/v1/report/evidence/{sel_ev_id}?format=pdf", timeout=30)
            if pdf_r.status_code == 200 and pdf_r.content:
                st.download_button(
                    label=f"⬇️ Download {sel_ev_id} PDF Report",
                    data=pdf_r.content,
                    file_name=f"{sel_ev_id}_Forensic_Report.pdf",
                    mime="application/pdf",
                    type="primary",
                    key=f"rpt_ev_pdf_{sel_ev_id}",
                )
            else:
                st.warning(f"Could not generate PDF for {sel_ev_id}.")
        except Exception as exc:
            st.error(f"Error fetching per-evidence PDF: {exc}")

    with col4:
        try:
            json_r = requests.get(f"{api_url}/api/v1/report/evidence/{sel_ev_id}?format=json", timeout=15)
            if json_r.status_code == 200 and json_r.content:
                st.download_button(
                    label=f"⬇️ Download {sel_ev_id} JSON Package",
                    data=json_r.content,
                    file_name=f"{sel_ev_id}_Forensic_Report.json",
                    mime="application/json",
                    key=f"rpt_ev_json_{sel_ev_id}",
                )
            else:
                st.warning(f"Could not generate JSON for {sel_ev_id}.")
        except Exception as exc:
            st.error(f"Error fetching per-evidence JSON: {exc}")


def _render_correlation_ui(corr: dict) -> None:
    """Render correlation result details in the UI."""
    unified = corr.get("unified_verdict") or corr.get("verdict") or "N/A"
    visual_v = corr.get("visual_verdict") or "N/A"
    audio_v = corr.get("audio_verdict") or "N/A"
    anomaly = corr.get("anomaly_type") or corr.get("anomaly_classification") or "N/A"
    synthesis = corr.get("forensic_synthesis") or corr.get("synthesis") or ""
    limitations = corr.get("limitations") or []
    conflicts = corr.get("conflicts") or []

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Unified Verdict:** `{unified}`")
        st.markdown(f"**Visual Forensic Verdict:** `{visual_v}`")
    with col_b:
        st.markdown(f"**Audio Forensic Verdict:** `{audio_v}`")
        st.markdown(f"**Anomaly Classification:** `{anomaly}`")

    if synthesis:
        st.markdown("**Forensic Synthesis:**")
        st.info(synthesis)

    if conflicts:
        st.markdown("**Detected Conflicts:**")
        for c in conflicts:
            st.warning(f"⚠️ {c}")

    if limitations:
        st.markdown("**Correlation Limitations:**")
        for lim in limitations:
            st.caption(f"• {lim}")
