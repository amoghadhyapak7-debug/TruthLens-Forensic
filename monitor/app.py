"""Standalone Streamlit development and evidence ingestion monitor."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Fix: Ensure project root is always at index 0 and monitor is removed from sys.path
# so that monitor/app.py does not shadow the backend 'app' package
ROOT = Path(__file__).resolve().parents[1]
MONITOR_DIR = str(Path(__file__).resolve().parent)
while MONITOR_DIR in sys.path:
    sys.path.remove(MONITOR_DIR)
while str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

import requests
import streamlit as st

try:
    from monitor.results import clear_registry_session_state, replace_upload_session
    from monitor.views_analyze import render_analyze_view
    from monitor.views_correlate import render_correlate_view
    from monitor.views_review import render_review_view
    from monitor.views_report import render_report_view
except ImportError:
    from results import clear_registry_session_state, replace_upload_session
    from views_analyze import render_analyze_view
    from views_correlate import render_correlate_view
    from views_review import render_review_view
    from views_report import render_report_view

PROGRESS_PATH = ROOT / "development" / "progress.json"
ACTIVITY_PATH = ROOT / "development" / "activity.json"
API_URL = os.getenv("TRUTHLENS_API_URL", "http://127.0.0.1:8000").rstrip("/")


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def get_api(path: str, default):
    try:
        response = requests.get(f"{API_URL}{path}", timeout=2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return default


def get_api_checked(path: str):
    try:
        response = requests.get(f"{API_URL}{path}", timeout=2)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def get_api_bytes(path: str):
    try:
        response = requests.get(f"{API_URL}{path}", timeout=600)
        response.raise_for_status()
        return response.content, None
    except requests.RequestException as exc:
        return None, str(exc)


def evidence_sort_key(record: dict) -> tuple[int, str]:
    evidence_id = str(record.get("evidence_id", ""))
    suffix = evidence_id.removeprefix("EVD-")
    return (int(suffix) if suffix.isdigit() else -1, str(record.get("ingestion_timestamp", "")))


def render_hash(label: str, sha256: str, key: str) -> None:
    if sha256 and sha256 != "—":
        st.code(sha256, language=None)
        st.caption(f"{label}: full SHA-256 shown above; use the code block copy control.")


def render_metadata(metadata: dict) -> None:
    """Render useful extracted values without flattening technical diagnostics."""
    if metadata.get("modality") != "video":
        st.json(metadata, expanded=False)
        return
    def shown(value):
        if value is None or value == "":
            return "Unavailable"
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value
    fields = [
        ("Container format", "container_format"), ("Duration (seconds)", "duration_seconds"),
        ("Width", "width"), ("Height", "height"), ("Resolution", "resolution"),
        ("Video codec", "video_codec"), ("FPS", "fps"), ("Frame count", "frame_count"),
        ("Bitrate (bps)", "bitrate"), ("Video stream count", "video_stream_count"),
        ("Audio stream count", "audio_stream_count"), ("Audio codec", "audio_codec"),
        ("Sample rate (Hz)", "sample_rate"), ("Metadata extraction", "metadata_extraction_status"),
        ("Stream readability", "stream_readable"),
    ]
    # Streamlit serializes dataframe columns through Arrow. Keep the display
    # column consistently textual when numbers and "Unavailable" coexist.
    st.dataframe([{"Property": label, "Value": shown(metadata.get(key))} for label, key in fields], hide_index=True, width="stretch")
    with st.expander("Technical stream details"):
        details = metadata.get("stream_metadata") or metadata.get("technical_details") or []
        if details:
            st.code("\n".join(details), language=None)
        else:
            st.caption("No parser stream details were available.")


def main():
    st.set_page_config(page_title="TruthLens — Multimodal Digital Forensics Platform", page_icon="🔎", layout="wide")

    if st.session_state.pop("registry_reset_cleanup_pending", False):
        clear_registry_session_state(st.session_state)

    # -------------------------------------------------------------
    # SIDEBAR: PLATFORM OVERVIEW & MISSION
    # -------------------------------------------------------------
    with st.sidebar:
        st.title("🔎 TRUTHLENS")
        st.markdown("**Multimodal Digital Forensics Platform**")
        st.caption("AI Manipulation Detection & Signal Correlation")

        with st.expander("01 | THE CHALLENGE", expanded=False):
            st.write(
                "AI-generated and manipulated media are becoming increasingly realistic across images, videos, audio, and text. "
                "Detecting deception is difficult when subtle manipulation spans multiple formats. There is a growing need for "
                "reliable, explainable forensic analysis that helps uncover suspicious indicators without treating automated "
                "predictions as definitive proof."
            )

        with st.expander("02 | OUR MISSION", expanded=False):
            st.write(
                "TruthLens is a multimodal digital forensics platform that analyzes media, detects potential manipulation, "
                "correlates findings across modalities, and delivers transparent, evidence-backed reports to support human "
                "verification and informed decision-making."
            )

        with st.expander("03 | CORE CAPABILITIES", expanded=True):
            st.markdown(
                "- **Image Forensics** — Detect visual manipulation and authenticity indicators.\n"
                "- **Video Forensics** — Examine frames, temporal patterns, and suspicious irregularities.\n"
                "- **Audio Forensics** — Analyze synthetic speech, voice manipulation, and audio anomalies.\n"
                "- **Text Analysis** — Examine content, context, and inconsistencies.\n"
                "- **Cross-Modal Correlation** — Connect evidence and identify potential mismatches across media.\n"
                "- **Explainable Findings** — Present indicators, confidence, limitations, and evidence references."
            )

        with st.expander("04 | ANALYSIS WORKFLOW", expanded=True):
            st.markdown(
                "**INGEST → ANALYZE → CORRELATE → REVIEW → REPORT**\n\n"
                "Upload evidence, analyze each modality, correlate findings, review indicators, and generate a traceable forensic report."
            )

        with st.expander("05 | FORENSIC REPORT", expanded=False):
            st.markdown(
                "Every report provides:\n"
                "- Flagged evidence and suspicious regions\n"
                "- Detected manipulation indicators\n"
                "- Cross-modal findings and inconsistencies\n"
                "- Confidence and uncertainty\n"
                "- Evidence references and analysis context\n"
                "- A summary for human review"
            )

        st.markdown("---")
        st.caption(f"Connected Backend: `{API_URL}`")
        reg_count = len(get_api("/api/v1/evidence", {}).get("evidence", []))
        st.caption(f"Active Registry Records: **{reg_count}**")



    st.title("🔎 TruthLens")
    st.caption("Multimodal Digital Forensics Platform · **INGEST → ANALYZE → CORRELATE → REVIEW → REPORT**")

    with st.expander("ℹ️ About TruthLens — Mission & Capabilities Overview", expanded=False):
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown("**01 | The Challenge**")
            st.caption("AI-generated and manipulated media are becoming increasingly realistic across images, videos, audio, and text. Detecting deception is difficult when subtle manipulation spans multiple formats. TruthLens delivers reliable, explainable forensic analysis without treating automated predictions as definitive proof.")
            st.markdown("**02 | Our Mission**")
            st.caption("A multimodal digital forensics platform that analyzes media, detects potential manipulation, correlates findings across modalities, and delivers transparent, evidence-backed reports to support human verification and informed decision-making.")
        with m_col2:
            st.markdown("**03 | Core Capabilities & 04 | Workflow**")
            st.markdown(
                "• **Image Forensics** (ELA, Noise, EXIF manipulation detection)\n"
                "• **Video Forensics** (GenD CLIP ViT-L/14, MesoNet-4, YuNet, Gemini)\n"
                "• **Audio Forensics** (Wav2Vec2 ONNX Int4 16kHz mono)\n"
                "• **Text Analysis** (AI generation probability, factual claim extraction)\n"
                "• **Cross-Modal Correlation** (Audio-Visual Synthesis & Timeline)\n"
                "• **Traceable Workflow:** `INGEST → ANALYZE → CORRELATE → REVIEW → REPORT`"
            )


    tab_ingest, tab_analyze, tab_correlate, tab_review, tab_report = st.tabs([
        "📥 INGEST",
        "🔍 ANALYZE",
        "🔗 CORRELATE",
        "⚖️ REVIEW",
        "📑 REPORT",
    ])

    with tab_ingest:
        st.subheader("Submit evidence")
        st.caption(f"Connected API: {API_URL}")
        if "upload_widget_version" not in st.session_state:
            st.session_state.upload_widget_version = 0
        upload_key = f"evidence_upload_{st.session_state.upload_widget_version}"
        uploads = st.file_uploader("Choose evidence files", accept_multiple_files=True, key=upload_key)
        case_id = st.text_input("Case ID (optional)", key="ingest_case_id_input")
        if uploads:
            st.write("Files selected for review:")
            st.dataframe([{"Filename": f.name, "Size": f"{f.size:,} bytes", "Extension": Path(f.name).suffix.lower()} for f in uploads], hide_index=True, width="stretch")
        if st.button("Ingest & Auto-Analyze Selected Files", type="primary", disabled=not uploads, key="ingest_submit_button"):
            try:
                multipart = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in uploads]
                fields = {"case_id": case_id} if case_id else {}
                with st.spinner("Registering evidence & computing SHA-256 hashes..."):
                    response = requests.post(f"{API_URL}/api/v1/ingest", files=multipart, data=fields, timeout=600)
                response.raise_for_status()
                timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
                ingest_results = response.json().get("results", [])
                replace_upload_session(st.session_state, ingest_results, timestamp)

                # Auto-route newly uploaded files to their appropriate backend analyzers
                with st.spinner("Automatically identifying file types and routing to backend analyzers..."):
                    for item in ingest_results:
                        ev_id = item.get("evidence_id")
                        if ev_id and item.get("status") == "completed":
                            try:
                                requests.post(f"{API_URL}/api/v1/analyze/auto/{ev_id}", timeout=600)
                            except Exception:
                                pass

                # Replace the uploader widget so the same selection cannot be submitted
                # again accidentally; latest results live in a separate session key.
                st.session_state.upload_widget_version += 1
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Ingestion API unavailable or request failed: {exc}")

        st.subheader("Current Upload Session Results")
        current_session = st.session_state.get("current_upload_session")
        if current_session and current_session.get("results"):
            st.caption(f"Only the latest upload operation · {current_session.get('timestamp')}")
            for index, result in enumerate(current_session["results"]):
                label = result["Upload result"]
                icon = {"NEW": "✅", "DUPLICATE": "🟡", "FAILED": "❌"}.get(label, "ℹ️")
                with st.expander(f"{icon} {result['Original filename']} · {label}", expanded=True):
                    cols = st.columns(3)
                    cols[0].write(f"**Evidence ID**  \n{result['Evidence ID']}")
                    cols[1].write(f"**File type**  \n{result['File type']}")
                    cols[2].write(f"**MIME type**  \n{result['MIME type']}")
                    cols = st.columns(3)
                    cols[0].write(f"**File size**  \n{result['File size']}")
                    cols[1].write(f"**Upload timestamp**  \n{result['Upload timestamp']}")
                    cols[2].write(f"**Duplicate reference**  \n{result['Duplicate reference']}")
                    if result.get("Metadata source") == "This upload (metadata extracted from the uploaded bytes)":
                        st.caption(f"Metadata was extracted from this upload. It is a duplicate of registered evidence {result['Duplicate reference']}.")
                    render_hash("SHA-256", result["SHA-256"], f"session-{index}")
                    eligibility = result.get("Analysis eligibility") or {}
                    if eligibility:
                        evidence_record = (result.get("evidence") or {})
                        validation = result.get("Validation status", evidence_record.get("validation_status", "Unavailable"))
                        metadata_status = (result.get("Metadata") or {}).get("metadata_extraction_status", "Unavailable")
                        stream = (result.get("Metadata") or {}).get("stream_readable")
                        stream_status = "Readable" if stream is True else ("Not readable" if stream is False else "Not detected / unavailable")
                        validation_source = result.get("Validation source", "This upload")
                        st.write(f"**File validation:** {validation}" + (f" ({validation_source})" if validation_source else " (this upload)"))
                        st.write(f"**Metadata extraction:** {metadata_status} · **Video stream:** {stream_status}")
                        for validation_error in result.get("Validation errors", []):
                            st.error(validation_error)
                        st.write(f"**Analysis eligibility:** {eligibility.get('status', 'UNKNOWN')}")
                        for reason in eligibility.get("reasons", []):
                            if eligibility.get("status") == "DUPLICATE_REFERENCE":
                                st.info(reason)
                            else:
                                st.error(reason)
                        for warning in eligibility.get("warnings", []):
                            st.warning(warning)
                    if result["Error"] != "—":
                        st.error(result["Error"])
                    for warning in result["Warnings"]:
                        st.warning(warning)
                    if result["Metadata"]:
                        with st.expander("Extracted metadata"):
                            render_metadata(result["Metadata"])
                    if result["Stages"]:
                        with st.expander("Processing stages"):
                            st.dataframe(result["Stages"], hide_index=True, width="stretch")
        else:
            st.info("Submit files to see results for that upload operation here.")

        status = get_api("/api/v1/ingest/status", {})
        st.subheader("Ingest Operations")
        st.caption(
            "Upload attempts vs. unique evidence records. "
            "Duplicates count as attempts but do not create new registry records."
        )
        cols = st.columns(5)
        op_labels = [
            ("Upload Attempts", "total_uploaded", "Total files submitted across all upload sessions"),
            ("Unique Evidence", "successfully_ingested", "Distinct evidence files registered (SHA-256 unique)"),
            ("Failed Uploads", "failed", "Uploads that failed validation or processing"),
            ("Duplicate Attempts", "duplicates", "Re-uploads of already-registered evidence (not stored again)"),
            ("Processing", "currently_processing", "Currently being ingested"),
        ]
        for col, (label, key, help_text) in zip(cols, op_labels):
            col.metric(label, status.get(key, 0), help=help_text)

        ingest_activity = get_api("/api/v1/ingest/activity?limit=100", {}).get("activity", [])
        st.subheader("Ingestion History")
        st.caption("Persistent activity across current and previous upload operations.")
        for entry in ingest_activity[:20]:
            st.write(f"**{entry.get('timestamp', '')}** · {entry.get('status', '')} — {entry.get('message', '')}")
        if st.button("Refresh Evidence Registry", key="refresh_evidence_registry"):
            st.rerun()
        registry_response, registry_error = get_api_checked("/api/v1/evidence")
        if registry_error:
            st.error(f"Evidence Registry could not be loaded from the API: {registry_error}")
            registry = None
        else:
            registry = registry_response.get("evidence", [])
            registry.sort(key=evidence_sort_key)
        readiness_response, readiness_error = get_api_checked("/api/v1/evidence/readiness")
        readiness_map = readiness_response.get("readiness", {}) if not readiness_error else {}
        if readiness_error and registry:
            st.warning(f"Analyzer readiness could not be refreshed: {readiness_error}")
        st.subheader("Evidence Registry")
        if registry is not None:
            st.caption(f"Total unique evidence records: {len(registry)}")

        # ─── Testing File Manager ───────────────────────────────────────────
        with st.expander("🧪 Testing File Manager", expanded=False):
            st.caption(
                "Manage temporary testing uploads. Real case evidence is excluded from this view. "
                "Classification is based on case ID and filename patterns — files are never auto-deleted."
            )
            testing_resp = get_api("/api/v1/testing/list", {})
            testing_items = testing_resp.get("testing", [])
            real_items = testing_resp.get("real_evidence", [])

            if testing_items:
                st.write(f"**{len(testing_items)} testing file(s) detected** · {len(real_items)} real evidence file(s) protected")
                st.dataframe([
                    {
                        "Evidence ID": t.get("evidence_id"),
                        "Filename": t.get("original_filename"),
                        "Type": t.get("file_type"),
                        "Size": t.get("file_size_human"),
                        "Case ID": t.get("case_id") or "(none)",
                        "Ingested": t.get("ingestion_timestamp"),
                    }
                    for t in testing_items
                ], hide_index=True, width="stretch")

                # Individual delete buttons
                for t_item in testing_items:
                    t_ev_id = t_item.get("evidence_id", "")
                    t_fname = t_item.get("original_filename", "")
                    del_key = f"del_test_{t_ev_id}"
                    confirm_key = f"confirm_del_{t_ev_id}"

                    col_name, col_btn = st.columns([3, 1])
                    with col_name:
                        st.write(f"`{t_ev_id}` — {t_fname}")
                    with col_btn:
                        if st.button("🗑️ Delete", key=del_key):
                            st.session_state[confirm_key] = True

                    if st.session_state.get(confirm_key):
                        st.warning(
                            f"**Confirm deletion of {t_ev_id}** ({t_fname})? "
                            f"This will remove the evidence file, all analysis results, artifacts, and the registry entry."
                        )
                        c_yes, c_no = st.columns(2)
                        if c_yes.button(f"✅ Confirm Delete {t_ev_id}", key=f"yes_{t_ev_id}"):
                            try:
                                resp = requests.delete(f"{API_URL}/api/v1/evidence/{t_ev_id}", timeout=30)
                                if resp.status_code == 200:
                                    st.success(f"✅ {t_ev_id} ({t_fname}) deleted successfully.")
                                    st.session_state.pop(confirm_key, None)
                                    st.rerun()
                                else:
                                    st.error(f"Deletion failed: {resp.text}")
                            except requests.RequestException as exc:
                                st.error(f"Deletion request error: {exc}")
                        if c_no.button(f"Cancel", key=f"no_{t_ev_id}"):
                            st.session_state.pop(confirm_key, None)
                            st.rerun()

                # Clear All Testing Files
                st.markdown("---")
                st.button("🧹 Clear All Testing Files", key="request_clear_testing",
                          on_click=lambda: st.session_state.update(clear_testing_pending=True))

                if st.session_state.get("clear_testing_pending"):
                    st.error(
                        f"**This will delete {len(testing_items)} testing file(s) and all their associated data.**\n\n"
                        "Files that will be deleted:"
                    )
                    for t_item in testing_items:
                        st.write(f"• `{t_item.get('evidence_id')}` — {t_item.get('original_filename')} ({t_item.get('file_size_human')})")

                    if real_items:
                        st.success(f"✅ {len(real_items)} real evidence file(s) will NOT be deleted.")

                    confirm_check = st.checkbox(
                        "I confirm deletion of all listed testing files",
                        key="confirm_clear_testing_check"
                    )
                    ct_yes, ct_no = st.columns(2)
                    if ct_yes.button("Confirm Clear All Testing", type="primary", key="yes_clear_testing", disabled=not confirm_check):
                        try:
                            resp = requests.post(
                                f"{API_URL}/api/v1/testing/clear-all",
                                json={"confirmation": "CLEAR TESTING FILES"},
                                timeout=120,
                            )
                            if resp.status_code == 200:
                                result = resp.json()
                                st.success(f"✅ Cleared {result.get('deleted_count', 0)} testing file(s).")
                                if result.get("failed_count", 0) > 0:
                                    st.warning(f"{result['failed_count']} file(s) could not be deleted.")
                                st.session_state.clear_testing_pending = False
                                st.rerun()
                            else:
                                st.error(f"Clear failed: {resp.text}")
                        except requests.RequestException as exc:
                            st.error(f"Request error: {exc}")
                    if ct_no.button("Cancel", key="no_clear_testing"):
                        st.session_state.clear_testing_pending = False
                        st.rerun()
            else:
                st.info("No testing files detected. All registered evidence appears to be real case evidence.")

        # ─── Registry Administration (existing) ─────────────────────────────
        with st.expander("Registry administration", expanded=False):
            st.button("🗑️ Clear All Evidence", type="primary", key="request_registry_clear",
                      on_click=lambda: st.session_state.update(registry_clear_pending=True))
            if st.session_state.get("registry_clear_pending"):
                st.error(
                    "This is destructive. It removes registered evidence records, each registered original file, "
                    "known evidence-scoped audio artifacts and analysis results, and incomplete .incoming-* upload files. "
                    "The persistent ingestion activity log, unregistered files, project files, and configuration are preserved."
                )
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button("Confirm Clear All", type="primary", key="confirm_registry_clear"):
                    try:
                        response = requests.post(f"{API_URL}/api/v1/evidence/reset", json={"confirmation": "CLEAR REGISTRY"}, timeout=600)
                        response.raise_for_status()
                        st.session_state.last_registry_reset_summary = response.json()
                        st.session_state.registry_clear_pending = False
                        st.session_state.registry_reset_cleanup_pending = True
                        st.rerun()
                    except requests.RequestException as exc:
                        st.error(f"Registry reset request failed: {exc}")
                if cancel_col.button("Cancel", key="cancel_registry_clear"):
                    st.session_state.registry_clear_pending = False
                    st.rerun()
        reset_summary = st.session_state.get("last_registry_reset_summary")
        if reset_summary:
            st.markdown("#### Latest registry reset")
            if reset_summary.get("complete"):
                st.success("Evidence registry reset completed.")
            else:
                st.error("Evidence registry reset was incomplete. Review the remaining records and errors below.")
            summary_cols = st.columns(4)
            summary_cols[0].metric("Evidence records removed", reset_summary.get("evidence_records_removed", 0))
            summary_cols[1].metric("Original files removed", reset_summary.get("original_files_removed", 0))
            summary_cols[2].metric("Artifact files removed", reset_summary.get("artifact_files_removed", 0))
            summary_cols[3].metric("Analysis records removed", reset_summary.get("analysis_records_removed", 0))
            if reset_summary.get("original_files_missing"):
                st.info(f"{reset_summary['original_files_missing']} registered original file(s) were already missing.")
            if reset_summary.get("registry_records_remaining"):
                st.warning(f"Registry records remaining: {reset_summary['registry_records_remaining']}")
            for error in reset_summary.get("errors", []):
                st.error(f"{error.get('evidence_id') or ''} {error.get('path') or ''}: {error.get('error', 'Unknown reset error')}")

        # ─── Evidence Registry Table with Classification Column ─────────────
        if registry:
            # Classify testing vs real for display
            testing_resp_for_table = get_api("/api/v1/testing/list", {})
            testing_ids = {t.get("evidence_id") for t in testing_resp_for_table.get("testing", [])}

            st.dataframe([
                {
                    "Evidence ID": r.get("evidence_id"),
                    "Classification": "🧪 Testing" if r.get("evidence_id") in testing_ids else "📁 Case Evidence",
                    "Filename": r.get("original_filename"),
                    "Evidence type": r.get("file_type"),
                    "Case ID": r.get("case_id") or "—",
                    "File size": r.get("file_size_human"),
                    "SHA-256": (r.get("sha256", "")[:12] + "…" if r.get("sha256") else "—"),
                    "Ingested": r.get("ingestion_timestamp"),
                    "Status": r.get("ingestion_status"),
                    "Readiness": readiness_map.get(r.get("evidence_id"), {}).get("status", "Unavailable"),
                    "Duplicate ref": r.get("duplicate_reference") or "—",
                }
                for r in registry
            ], width="stretch", hide_index=True)
            for record in registry:
                duplicate = record.get("duplicate_reference")
                ev_id = record.get("evidence_id", "")
                is_testing = ev_id in testing_ids
                class_icon = "🧪" if is_testing else "📁"
                title = f"{class_icon} {ev_id} · {record.get('original_filename')}" + (f" · duplicate of {duplicate}" if duplicate else "")
                with st.expander(title):
                    cols = st.columns(3)
                    cols[0].write(f"**Evidence type**  \n{record.get('file_type', '—')}")
                    cols[1].write(f"**MIME type**  \n{record.get('mime_type', '—')}")
                    cols[2].write(f"**File size**  \n{record.get('file_size_human', '—')}")
                    st.write(f"**Case ID:** {record.get('case_id') or '(none)'} · **Classification:** {'Testing' if is_testing else 'Case Evidence'}")
                    st.write(f"**Ingestion timestamp:** {record.get('ingestion_timestamp', '—')}")
                    st.write(f"**Status:** {record.get('ingestion_status', '—')}" + (f" · **Duplicate reference:** {duplicate}" if duplicate else ""))
                    render_hash("SHA-256", record.get("sha256", ""), f"registry-{ev_id}")

                    # Individual delete button for any evidence
                    if st.button(f"🗑️ Delete {ev_id}", key=f"reg_del_{ev_id}"):
                        st.session_state[f"reg_confirm_del_{ev_id}"] = True

                    if st.session_state.get(f"reg_confirm_del_{ev_id}"):
                        st.warning(f"Confirm deletion of {ev_id} ({record.get('original_filename')})? This removes the file, all analysis results, and the registry entry.")
                        rd_yes, rd_no = st.columns(2)
                        if rd_yes.button(f"Confirm Delete", key=f"reg_yes_{ev_id}"):
                            try:
                                resp = requests.delete(f"{API_URL}/api/v1/evidence/{ev_id}", timeout=30)
                                if resp.status_code == 200:
                                    st.success(f"✅ {ev_id} deleted.")
                                    st.session_state.pop(f"reg_confirm_del_{ev_id}", None)
                                    st.rerun()
                                else:
                                    st.error(f"Deletion failed: {resp.text}")
                            except requests.RequestException as exc:
                                st.error(f"Error: {exc}")
                        if rd_no.button(f"Cancel", key=f"reg_no_{ev_id}"):
                            st.session_state.pop(f"reg_confirm_del_{ev_id}", None)
                            st.rerun()

                    st.markdown("**Metadata**")
                    st.json(record.get("metadata") or {}, expanded=False)
        elif registry == []:
            st.info("The registry is reachable and currently contains no evidence records.")

    # -------------------------------------------------------------
    # WORKFLOW STAGES: ANALYZE → CORRELATE → REVIEW → REPORT
    # -------------------------------------------------------------
    all_evidence_records = get_api("/api/v1/evidence", {}).get("evidence", [])

    with tab_analyze:
        render_analyze_view(API_URL, all_evidence_records)

    with tab_correlate:
        render_correlate_view(API_URL, all_evidence_records)

    with tab_review:
        render_review_view(API_URL, all_evidence_records)

    with tab_report:
        render_report_view(API_URL, all_evidence_records)




if __name__ == "__main__":
    main()
