"""TruthLens Report Generator – Produces structured JSON and PDF forensic reports."""
from __future__ import annotations

import io
import json
import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.analyzers.normalization import normalize_video_verdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared PDF helpers
# ---------------------------------------------------------------------------

def _get_pdf_styles():
    """Return (styles, custom_style_dict) for ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    styles = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle(
            "TLTitle",
            parent=styles["Heading1"],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#1A202C"),
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "TLSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#4A5568"),
            spaceAfter=15,
        ),
        "heading": ParagraphStyle(
            "TLHeading",
            parent=styles["Heading2"],
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#2B6CB0"),
            spaceBefore=12,
            spaceAfter=6,
        ),
        "subheading": ParagraphStyle(
            "TLSubheading",
            parent=styles["Heading3"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#2C5282"),
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "TLBody",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#2D3748"),
        ),
        "label": ParagraphStyle(
            "TLLabel",
            parent=styles["Normal"],
            fontSize=8,
            leading=12,
            textColor=colors.HexColor("#718096"),
            fontName="Helvetica-Oblique",
        ),
        "code": ParagraphStyle(
            "TLCode",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            fontName="Courier",
            textColor=colors.HexColor("#2D3748"),
            backColor=colors.HexColor("#F7FAFC"),
        ),
        "warn": ParagraphStyle(
            "TLWarn",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#744210"),
            backColor=colors.HexColor("#FEFCBF"),
        ),
        "auto_label": ParagraphStyle(
            "TLAutoLabel",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#276749"),
            fontName="Helvetica-BoldOblique",
        ),
        "opinion_label": ParagraphStyle(
            "TLOpinionLabel",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#744210"),
            fontName="Helvetica-BoldOblique",
        ),
    }
    return styles, custom


def _kv_table(rows: list[tuple[str, str]], col_widths=(150, 370)):
    """Return a ReportLab Table of key-value pairs."""
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    data = [[k, v] for k, v in rows]
    t = Table(data, colWidths=list(col_widths))
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#4A5568")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#2D3748")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EDF2F7")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def _fmt(val: Any) -> str:
    """Safe format any value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, (dict, list)):
        return json.dumps(val, indent=2)
    return str(val)


def _truncate(s: str, n: int = 400) -> str:
    return s if len(s) <= n else s[:n] + "…"


# ---------------------------------------------------------------------------
# Per-Evidence JSON Report
# ---------------------------------------------------------------------------

def generate_evidence_json_report(
    evidence_id: str,
    record: Dict[str, Any],
    analysis_result: Optional[Dict[str, Any]] = None,
    review: Optional[Dict[str, Any]] = None,
    correlation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Complete per-evidence-ID JSON forensic package."""
    now = datetime.utcnow().isoformat() + "Z"
    return {
        "report_type": "per_evidence",
        "report_id": f"TL-EV-{evidence_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "generated_at": now,
        "system": "TruthLens Multi-Modal Forensic Investigation Platform",
        "evidence_id": evidence_id,
        "ingest": {
            "evidence_id": record.get("evidence_id"),
            "original_filename": record.get("original_filename"),
            "file_type": record.get("file_type"),
            "file_size_bytes": record.get("file_size"),
            "sha256": record.get("sha256"),
            "ingestion_timestamp": record.get("ingestion_timestamp"),
            "case_id": record.get("case_id"),
            "storage_path": record.get("storage_path"),
            "is_testing": record.get("is_testing", False),
            "validation_status": record.get("validation_status"),
            "analysis_eligibility": record.get("analysis_eligibility"),
            "metadata": record.get("metadata", {}),
            "stages": record.get("stages", {}),
        },
        "analysis": analysis_result if analysis_result is not None else {},
        "correlation": correlation if correlation is not None else {},
        "review": review if review is not None else {
            "evidence_id": evidence_id,
            "status": "UNREVIEWED",
            "notes": "",
            "examiner": "",
            "auto_findings": "",
            "investigator_opinion": "",
            "updated_at": None,
        },
        "forensic_standards": {
            "hashing": "SHA-256 integrity verification",
            "image_models": ["timm/EfficientNet-B4", "timm/ResNet-50", "ELA-95", "BVR-Wavelet"],
            "video_models": ["GenD CLIP ViT-L/14 (WACV 2026)", "MesoNet-4", "YuNet Face Detector"],
            "audio_models": ["ai8shiro/deepfake-audio-wav2vec2-ONNX"],
            "text_models": ["DocumentExtractor", "EntityExtractor", "ClaimExtractor", "LinguisticAnalyzer"],
        },
    }


# ---------------------------------------------------------------------------
# Per-Evidence PDF Report
# ---------------------------------------------------------------------------

def generate_evidence_pdf_report(
    evidence_id: str,
    record: Dict[str, Any],
    analysis_result: Optional[Dict[str, Any]] = None,
    review: Optional[Dict[str, Any]] = None,
    correlation: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Complete per-evidence PDF forensic examination report."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    except ImportError:
        logger.error("ReportLab is not installed; cannot generate PDF.")
        return b""

    _, S = _get_pdf_styles()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []

    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    f_type = record.get("file_type", "unknown")

    # ── Title ─────────────────────────────────────────────────────────────
    story.append(Paragraph("TruthLens Forensic Examination Report", S["title"]))
    story.append(Paragraph(
        f"Evidence ID: <b>{evidence_id}</b> &bull; Modality: {f_type.upper()} &bull; "
        f"Generated: {now_str} &bull; Classification: OFFICIAL FORENSIC ARTIFACT",
        S["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=14))

    # ── Section 1: Evidence Identification ────────────────────────────────
    story.append(Paragraph("1. Evidence Identification", S["heading"]))
    story.append(_kv_table([
        ("Evidence ID", record.get("evidence_id", "N/A")),
        ("Original Filename", record.get("original_filename", "N/A")),
        ("File Type / Modality", record.get("file_type", "N/A")),
        ("File Size", f"{record.get('file_size', 0):,} bytes" if record.get("file_size") else "N/A"),
        ("SHA-256 Hash", record.get("sha256", "N/A")),
        ("Upload Timestamp", record.get("ingestion_timestamp", "N/A")),
        ("Case ID", record.get("case_id") or "No case assigned"),
        ("Is Testing Evidence", str(record.get("is_testing", False))),
        ("Storage Path", str(record.get("storage_path", "N/A"))),
    ]))
    story.append(Spacer(1, 10))

    # ── Section 2: Ingest Details ──────────────────────────────────────────
    story.append(Paragraph("2. Ingest Details & Metadata", S["heading"]))
    metadata = record.get("metadata") or {}
    validation_status = record.get("validation_status") or "N/A"
    analysis_eligibility = record.get("analysis_eligibility") or "N/A"
    stages = record.get("stages") or {}

    story.append(_kv_table([
        ("Validation Status", str(validation_status)),
        ("Analysis Eligibility", str(analysis_eligibility)),
    ]))
    story.append(Spacer(1, 6))

    if metadata and isinstance(metadata, dict):
        story.append(Paragraph("Extracted File Metadata:", S["subheading"]))
        meta_rows = [(str(k), _truncate(str(v))) for k, v in metadata.items()]
        if meta_rows:
            story.append(_kv_table(meta_rows))
        story.append(Spacer(1, 6))

    if stages:
        story.append(Paragraph("Workflow Stages:", S["subheading"]))
        if isinstance(stages, dict):
            stage_rows = [(str(k), _truncate(str(v))) for k, v in stages.items()]
        elif isinstance(stages, list):
            stage_rows = [
                (
                    s.get("stage", f"Stage {idx+1}") if isinstance(s, dict) else f"Stage {idx+1}",
                    f"{s.get('status', 'N/A')} ({s.get('timestamp', '')})".strip() if isinstance(s, dict) else _truncate(str(s))
                )
                for idx, s in enumerate(stages)
            ]
        else:
            stage_rows = [("Stages", _truncate(str(stages)))]
        if stage_rows:
            story.append(_kv_table(stage_rows))
    story.append(Spacer(1, 10))

    # ── Section 3: Forensic Analysis Findings ─────────────────────────────
    story.append(Paragraph("3. Forensic Analysis Findings", S["heading"]))

    if not analysis_result:
        story.append(Paragraph("<i>No analysis result available for this evidence item. Analysis may not have been executed yet.</i>", S["body"]))
    else:
        _render_analysis_pdf(story, S, f_type, analysis_result, colors)

    story.append(Spacer(1, 10))

    # ── Section 4: Cross-Modal Correlation ────────────────────────────────
    story.append(Paragraph("4. Cross-Modal Correlation", S["heading"]))
    if not correlation:
        if f_type == "video":
            story.append(Paragraph("<i>No correlation result available. Run video-audio correlation in the CORRELATE stage first.</i>", S["body"]))
        else:
            story.append(Paragraph(f"<i>Cross-modal correlation is applicable to video evidence only. This evidence is of type: {f_type}.</i>", S["body"]))
    else:
        _render_correlation_pdf(story, S, correlation, colors)

    story.append(Spacer(1, 10))

    # ── Section 5: Examiner Review ────────────────────────────────────────
    story.append(Paragraph("5. Examiner Review & Determination", S["heading"]))
    if not review:
        story.append(Paragraph("<i>No review record found for this evidence item.</i>", S["body"]))
    else:
        rev_status = review.get("status", "UNREVIEWED")
        rev_examiner = review.get("examiner") or "Not recorded"
        rev_updated = review.get("updated_at") or "Not recorded"
        rev_auto = review.get("auto_findings") or ""
        rev_opinion = review.get("investigator_opinion") or ""
        rev_notes = review.get("notes") or ""

        story.append(_kv_table([
            ("Verification Status", rev_status),
            ("Examiner", rev_examiner),
            ("Last Updated", rev_updated),
        ]))
        story.append(Spacer(1, 8))

        if rev_auto:
            story.append(Paragraph("Auto-Generated Forensic Summary (System-Generated, Not Human-Verified):", S["auto_label"]))
            for line in rev_auto.splitlines():
                if line.strip():
                    story.append(Paragraph(line.strip(), S["body"]))
            story.append(Spacer(1, 6))

        if rev_opinion:
            story.append(Paragraph("Investigator Opinion (Manually Entered):", S["opinion_label"]))
            for line in rev_opinion.splitlines():
                if line.strip():
                    story.append(Paragraph(line.strip(), S["body"]))
            story.append(Spacer(1, 6))

        if rev_notes and not rev_auto and not rev_opinion:
            story.append(Paragraph("Examiner Notes:", S["subheading"]))
            for line in rev_notes.splitlines():
                if line.strip():
                    story.append(Paragraph(line.strip(), S["body"]))

    story.append(Spacer(1, 10))

    # ── Section 6: Methodology & Limitations ─────────────────────────────
    story.append(Paragraph("6. Forensic Scope and Methodological Limitations", S["heading"]))
    story.append(Paragraph(
        "1. Automated models (GenD CLIP ViT-L/14, Wav2Vec2 ONNX, EfficientNet-B4) evaluate statistical representations "
        "and acoustic/visual features against designated training distributions. Scores are probabilistic and do not "
        "constitute incontrovertible legal proof.<br/>"
        "2. Multi-generation compression, platform re-encoding (e.g. WhatsApp, Twitter), and intentional adversarial "
        "perturbations can influence confidence thresholds.<br/>"
        "3. Final evidentiary determinations must be conducted by certified forensic examiners incorporating independent corroboration.<br/>"
        "4. Auto-generated findings are summaries of machine-analysis outputs and must not be treated as human expert opinions.",
        S["body"],
    ))

    doc.build(story)
    return buf.getvalue()


def _render_analysis_pdf(story, S, f_type: str, res: dict, colors) -> None:
    """Render analysis findings section into the PDF story."""
    from reportlab.platypus import Paragraph, Spacer

    if f_type == "video":
        v_obj = res.get("video") if isinstance(res, dict) and "video" in res else res
        v_norm = normalize_video_verdict(v_obj or {})
        story.append(Paragraph("3.1 Visual (Video) Analysis", S["subheading"]))
        story.append(_kv_table([
            ("Verdict", v_norm.get("verdict", "N/A")),
            ("Synthetic Probability", f"{v_norm.get('fake_probability', 0):.4f}"),
            ("Confidence Level", f"{v_norm.get('confidence', 0):.4f}"),
            ("Model", v_norm.get("model", "N/A")),
            ("Frames Analyzed", str(v_obj.get("frames_analyzed", "N/A") if isinstance(v_obj, dict) else "N/A")),
            ("Faces Detected", str(v_obj.get("faces_detected", "N/A") if isinstance(v_obj, dict) else "N/A")),
        ]))
        story.append(Spacer(1, 6))

        # Indicators
        if isinstance(v_obj, dict):
            indicators = v_obj.get("indicators") or v_obj.get("suspicious_indicators") or []
            if indicators:
                story.append(Paragraph("Detected Suspicious Indicators:", S["subheading"]))
                for ind in indicators:
                    story.append(Paragraph(f"&bull; {_truncate(str(ind))}", S["body"]))
                story.append(Spacer(1, 4))

            frame_data = v_obj.get("frame_level_analysis") or v_obj.get("per_frame") or []
            if frame_data and isinstance(frame_data, list):
                story.append(Paragraph(f"Frame-Level Analysis ({min(len(frame_data), 10)} of {len(frame_data)} frames shown):", S["subheading"]))
                for fr in frame_data[:10]:
                    if isinstance(fr, dict):
                        line = "  ".join(f"{k}={_fmt(v)}" for k, v in fr.items())
                        story.append(Paragraph(f"&bull; {_truncate(line)}", S["body"]))
                story.append(Spacer(1, 4))

        # Audio embedded in video result
        story.append(Paragraph("3.2 Audio Stream (Extracted from Video)", S["subheading"]))
        a_obj = res.get("audio") if isinstance(res, dict) else {}
        a_status = res.get("audio_extraction_status", "N/A") if isinstance(res, dict) else "N/A"
        if not a_obj:
            story.append(Paragraph(f"<i>Audio extraction status: {a_status}. No audio analysis data available.</i>", S["body"]))
        else:
            a_res = a_obj.get("result") or a_obj.get("verdict") or "N/A"
            story.append(_kv_table([
                ("Extraction Status", str(a_status)),
                ("Audio Verdict", str(a_res)),
                ("Score", _fmt(a_obj.get("score") or a_obj.get("fake_score"))),
            ]))
            transcript = a_obj.get("transcript") or a_obj.get("transcription")
            if transcript:
                story.append(Spacer(1, 4))
                story.append(Paragraph("Transcript:", S["subheading"]))
                story.append(Paragraph(_truncate(str(transcript), 600), S["body"]))
        story.append(Spacer(1, 6))

    elif f_type == "image":
        story.append(Paragraph("3.1 Image Forensic Analysis", S["subheading"]))
        verdict = res.get("verdict", "N/A")
        manip = res.get("manipulation") or {}
        ela = res.get("ela") or {}
        story.append(_kv_table([
            ("Verdict", str(verdict)),
            ("Manipulation Score", _fmt(manip.get("manipulation_score") if isinstance(manip, dict) else None)),
            ("Manipulation Type", str(manip.get("manipulation_type", "N/A") if isinstance(manip, dict) else "N/A")),
            ("ELA Score", _fmt(ela.get("ela_score") if isinstance(ela, dict) else None)),
            ("Model", res.get("model_name") or res.get("model") or "N/A"),
        ]))
        story.append(Spacer(1, 6))

        # Models breakdown
        models = res.get("model_results") or res.get("models") or {}
        if models and isinstance(models, dict):
            story.append(Paragraph("Per-Model Results:", S["subheading"]))
            for m_name, m_res in models.items():
                if isinstance(m_res, dict):
                    row = "  ".join(f"{k}={_fmt(v)}" for k, v in m_res.items())
                    story.append(Paragraph(f"<b>{m_name}:</b> {_truncate(row)}", S["body"]))
            story.append(Spacer(1, 4))

        indicators = res.get("indicators") or res.get("suspicious_indicators") or []
        if indicators:
            story.append(Paragraph("Detected Indicators:", S["subheading"]))
            for ind in indicators:
                story.append(Paragraph(f"&bull; {_truncate(str(ind))}", S["body"]))

    elif f_type == "audio":
        story.append(Paragraph("3.1 Audio Deepfake Analysis", S["subheading"]))
        verdict = res.get("verdict") or {}
        conclusion = verdict.get("conclusion") if isinstance(verdict, dict) else res.get("result", "N/A")
        model_info = res.get("model_info") or {}
        story.append(_kv_table([
            ("Conclusion / Verdict", str(conclusion)),
            ("Score (Synthetic)", _fmt(res.get("fake_score") or res.get("score"))),
            ("Score (Genuine)", _fmt(res.get("genuine_score"))),
            ("Score Band", res.get("score_band", "N/A")),
            ("Model", model_info.get("model_name") if isinstance(model_info, dict) else "N/A"),
            ("Model Architecture", model_info.get("architecture") if isinstance(model_info, dict) else "N/A"),
            ("Sample Rate", _fmt(res.get("sample_rate"))),
            ("Duration (s)", _fmt(res.get("duration_seconds"))),
            ("Channels", _fmt(res.get("num_channels"))),
        ]))
        story.append(Spacer(1, 6))

        transcript = res.get("transcript") or res.get("transcription")
        if transcript:
            story.append(Paragraph("Transcript:", S["subheading"]))
            story.append(Paragraph(_truncate(str(transcript), 600), S["body"]))
            story.append(Spacer(1, 4))

        segment_data = res.get("segments") or res.get("segment_results") or []
        if segment_data and isinstance(segment_data, list):
            story.append(Paragraph(f"Segment-Level Analysis ({min(len(segment_data), 10)} of {len(segment_data)}):", S["subheading"]))
            for seg in segment_data[:10]:
                if isinstance(seg, dict):
                    story.append(Paragraph(f"&bull; {_truncate(str(seg))}", S["body"]))

    elif f_type in ("text", "document", "structured_data"):
        story.append(Paragraph("3.1 Text / Document Forensic Analysis", S["subheading"]))
        analyses = res.get("analyses", {}) if isinstance(res, dict) else {}
        ai_det = res.get("ai_detection") or analyses.get("ai_detector") or {}
        entities = res.get("entities") or analyses.get("entity_extraction", {}).get("entities") or []
        claims = res.get("claims") or analyses.get("claim_extraction", {}).get("claims") or []
        linguistic = res.get("linguistic_analysis") or analyses.get("linguistic_analysis", {})
        story.append(_kv_table([
            ("AI-Text Probability", _fmt(ai_det.get("ai_probability") if isinstance(ai_det, dict) else None)),
            ("AI Verdict / Status", str(ai_det.get("verdict") or ai_det.get("status") or "N/A" if isinstance(ai_det, dict) else "N/A")),
            ("Entities Found", str(len(entities) if isinstance(entities, (list, dict)) else "N/A")),
            ("Claims Identified", str(len(claims) if isinstance(claims, (list, dict)) else "N/A")),
        ]))
        story.append(Spacer(1, 6))

        if linguistic and isinstance(linguistic, dict):
            story.append(Paragraph("Linguistic Findings:", S["subheading"]))
            for k, v in linguistic.items():
                if isinstance(v, dict):
                    v_str = "  ".join(f"{sk}={_fmt(sv)}" for sk, sv in v.items())
                else:
                    v_str = _fmt(v)
                story.append(Paragraph(f"<b>{k}:</b> {_truncate(str(v_str))}", S["body"]))

    else:
        # Unknown type — render raw
        story.append(Paragraph(f"<i>Analysis result for type '{f_type}':</i>", S["body"]))
        raw = _truncate(json.dumps(res, indent=2), 1000)
        from reportlab.platypus import Paragraph as P
        story.append(P(raw.replace("\n", "<br/>"), S["code"]))


def _render_correlation_pdf(story, S, correlation: dict, colors) -> None:
    """Render correlation section into the PDF story."""
    from reportlab.platypus import Paragraph, Spacer

    unified = correlation.get("unified_verdict") or correlation.get("verdict") or "N/A"
    visual_v = correlation.get("visual_verdict") or "N/A"
    audio_v = correlation.get("audio_verdict") or "N/A"
    anomaly = correlation.get("anomaly_type") or correlation.get("anomaly_classification") or "N/A"
    synthesis = correlation.get("forensic_synthesis") or correlation.get("synthesis") or ""
    limitations = correlation.get("limitations") or []
    conflicts = correlation.get("conflicts") or []

    story.append(_kv_table([
        ("Unified Verdict", str(unified)),
        ("Visual Forensic Verdict", str(visual_v)),
        ("Audio Forensic Verdict", str(audio_v)),
        ("Anomaly Classification", str(anomaly)),
    ]))
    story.append(Spacer(1, 6))

    if synthesis:
        story.append(Paragraph("Forensic Synthesis:", S["subheading"]))
        for line in str(synthesis).splitlines():
            if line.strip():
                story.append(Paragraph(line.strip(), S["body"]))
        story.append(Spacer(1, 4))

    if conflicts:
        story.append(Paragraph("Detected Conflicts:", S["subheading"]))
        for c in conflicts:
            story.append(Paragraph(f"&bull; {_truncate(str(c))}", S["body"]))
        story.append(Spacer(1, 4))

    if limitations:
        story.append(Paragraph("Correlation Limitations:", S["subheading"]))
        for lim in limitations:
            story.append(Paragraph(f"&bull; {_truncate(str(lim))}", S["body"]))


# ---------------------------------------------------------------------------
# Case-Level JSON Report (Enhanced)
# ---------------------------------------------------------------------------

def generate_json_report(
    case_id: str,
    evidence_records: List[Dict[str, Any]],
    analysis_results: Dict[str, Dict[str, Any]],
    review_notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate comprehensive forensic JSON report (case-level)."""
    now = datetime.utcnow().isoformat() + "Z"

    evidence_packages = []
    for r in evidence_records:
        ev_id = r.get("evidence_id")
        analysis = analysis_results.get(ev_id) if ev_id else None
        review = review_notes.get(ev_id) if review_notes and ev_id else None
        evidence_packages.append({
            "ingest": {
                "evidence_id": ev_id,
                "original_filename": r.get("original_filename"),
                "file_type": r.get("file_type"),
                "file_size_bytes": r.get("file_size"),
                "sha256": r.get("sha256"),
                "ingestion_timestamp": r.get("ingestion_timestamp"),
                "case_id": r.get("case_id"),
                "storage_path": r.get("storage_path"),
                "is_testing": r.get("is_testing", False),
                "validation_status": r.get("validation_status"),
                "analysis_eligibility": r.get("analysis_eligibility"),
                "metadata": r.get("metadata", {}),
                "stages": r.get("stages", {}),
            },
            "analysis": analysis or {},
            "review": review or {
                "evidence_id": ev_id,
                "status": "UNREVIEWED",
                "notes": "",
                "examiner": "",
                "auto_findings": "",
                "investigator_opinion": "",
                "updated_at": None,
            },
        })

    report = {
        "report_type": "case_level",
        "report_id": f"TL-REP-{case_id or 'GENERAL'}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "generated_at": now,
        "case_id": case_id or "GENERAL_INVESTIGATION",
        "system": "TruthLens Multi-Modal Forensic Investigation Platform",
        "evidence_count": len(evidence_records),
        "chain_of_custody": [
            {
                "evidence_id": r.get("evidence_id"),
                "original_filename": r.get("original_filename"),
                "file_type": r.get("file_type"),
                "sha256": r.get("sha256"),
                "file_size_bytes": r.get("file_size"),
                "ingestion_timestamp": r.get("ingestion_timestamp"),
            }
            for r in evidence_records
        ],
        "evidence": evidence_packages,
        # Legacy flat fields kept for compatibility:
        "findings": analysis_results,
        "analyst_reviews": review_notes or {},
        "forensic_standards": {
            "hashing": "SHA-256 integrity verification",
            "image_models": ["timm/EfficientNet-B4", "timm/ResNet-50", "ELA-95", "BVR-Wavelet"],
            "video_models": ["GenD CLIP ViT-L/14 (WACV 2026)", "MesoNet-4", "YuNet Face Detector"],
            "audio_models": ["ai8shiro/deepfake-audio-wav2vec2-ONNX"],
            "text_models": ["DocumentExtractor", "EntityExtractor", "ClaimExtractor", "LinguisticAnalyzer"],
        },
    }
    return report


# ---------------------------------------------------------------------------
# Case-Level PDF Report (Enhanced)
# ---------------------------------------------------------------------------

def generate_pdf_report(
    case_id: str,
    evidence_records: List[Dict[str, Any]],
    analysis_results: Dict[str, Dict[str, Any]],
    review_notes: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Generate high-quality forensic PDF report using ReportLab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    except ImportError:
        logger.error("ReportLab is not installed; cannot generate PDF.")
        return b""

    _, S = _get_pdf_styles()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []

    # Title & Header
    story.append(Paragraph("TruthLens Forensic Examination Report", S["title"]))
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(
        f"Case ID: <b>{case_id or 'GENERAL_INVESTIGATION'}</b> &bull; Generated: {now_str} "
        f"&bull; Evidence Items: {len(evidence_records)} &bull; Classification: OFFICIAL FORENSIC ARTIFACT",
        S["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E0"), spaceAfter=12))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary", S["heading"]))
    story.append(Paragraph(
        f"This forensic report consolidates automated multi-modal analysis performed by the TruthLens platform "
        f"across {len(evidence_records)} registered evidence artifact(s). Evidence items have been verified with "
        f"cryptographic SHA-256 hashes to guarantee immutability and chain-of-custody integrity.",
        S["body"],
    ))
    story.append(Spacer(1, 10))

    # Chain of Custody Table
    story.append(Paragraph("2. Evidence Chain-of-Custody & Integrity", S["heading"]))
    table_data = [["Evidence ID", "Original Filename", "Type", "Size", "SHA-256 (first 16)"]]
    for r in evidence_records:
        h = str(r.get("sha256", ""))
        h_short = h[:16] + "..." if len(h) > 16 else h
        table_data.append([
            r.get("evidence_id", ""),
            r.get("original_filename", "")[:28],
            r.get("file_type", ""),
            f"{r.get('file_size', 0):,} B",
            h_short,
        ])
    t = Table(table_data, colWidths=[65, 155, 60, 65, 175])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2D3748")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Per-Evidence Findings
    story.append(Paragraph("3. Multi-Modal Analytical Findings — Per Evidence", S["heading"]))
    for r in evidence_records:
        ev_id = r.get("evidence_id")
        f_type = r.get("file_type", "unknown")
        res = analysis_results.get(ev_id, {})

        story.append(Paragraph(
            f"<b>Evidence {ev_id} — {r.get('original_filename', 'N/A')}</b> &mdash; Modality: {f_type.upper()}",
            S["subheading"],
        ))

        if not res:
            story.append(Paragraph("<i>Analysis pending or not executed yet.</i>", S["body"]))
        else:
            _render_analysis_pdf(story, S, f_type, res, colors)

        # Review for this evidence
        rev = (review_notes or {}).get(ev_id, {})
        if rev and isinstance(rev, dict) and rev.get("status") not in (None, "UNREVIEWED"):
            rev_status = rev.get("status", "UNREVIEWED")
            rev_examiner = rev.get("examiner") or "N/A"
            story.append(Paragraph(
                f"&nbsp;&nbsp;<b>Review Status:</b> {rev_status} | "
                f"<b>Examiner:</b> {rev_examiner} | "
                f"<b>Updated:</b> {rev.get('updated_at') or 'N/A'}",
                S["body"],
            ))
            auto_f = rev.get("auto_findings") or ""
            if auto_f:
                story.append(Paragraph("&nbsp;&nbsp;<i>(Auto-Generated Summary recorded in review record)</i>", S["label"]))
            op = rev.get("investigator_opinion") or ""
            if op:
                story.append(Paragraph(f"&nbsp;&nbsp;<b>Investigator Opinion:</b> {_truncate(op)}", S["body"]))

        story.append(Spacer(1, 10))

    story.append(Spacer(1, 6))

    # Examiner Review Determinations
    if review_notes:
        story.append(Paragraph("4. Forensic Examiner Determinations & Review Notes", S["heading"]))
        for ev_id, rev in review_notes.items():
            if isinstance(rev, dict):
                st_val = rev.get("status", "UNREVIEWED")
                ex_val = rev.get("examiner", "N/A")
                nt_val = rev.get("notes", "No notes recorded.")
                up_val = rev.get("updated_at", "N/A")
                auto_val = rev.get("auto_findings", "")
                opinion_val = rev.get("investigator_opinion", "")
                story.append(Paragraph(
                    f"&bull; <b>{ev_id}:</b> Determination: <b>{st_val}</b> &bull; Examiner: <u>{ex_val}</u> ({up_val})",
                    S["body"],
                ))
                if auto_val:
                    story.append(Paragraph(
                        f"&nbsp;&nbsp;&nbsp;<i>Auto-Generated Findings: {_truncate(auto_val, 300)}</i>",
                        S["label"],
                    ))
                if opinion_val:
                    story.append(Paragraph(
                        f"&nbsp;&nbsp;&nbsp;<b>Investigator Opinion:</b> {_truncate(opinion_val, 300)}",
                        S["body"],
                    ))
                if nt_val and not auto_val and not opinion_val:
                    story.append(Paragraph(
                        f"&nbsp;&nbsp;&nbsp;<i>Notes: {_truncate(nt_val, 300)}</i>",
                        S["body"],
                    ))
                story.append(Spacer(1, 4))
        story.append(Spacer(1, 10))

    # Limitations & Legal Notice
    section_num = "5" if review_notes else "4"
    story.append(Paragraph(f"{section_num}. Forensic Scope and Methodological Limitations", S["heading"]))
    story.append(Paragraph(
        "1. Automated models (GenD CLIP ViT-L/14, Wav2Vec2 ONNX, EfficientNet-B4) evaluate statistical representations "
        "and acoustic/visual features against designated training distributions. Scores are probabilistic and do not "
        "constitute incontrovertible legal proof.<br/>"
        "2. Multi-generation compression, platform re-encoding (e.g. WhatsApp, Twitter), and intentional adversarial "
        "perturbations can influence confidence thresholds.<br/>"
        "3. Final evidentiary determinations must be conducted by certified forensic examiners incorporating independent corroboration.<br/>"
        "4. Auto-generated examiner summaries are system-generated summaries of machine analysis and must not be presented "
        "as human expert opinions.",
        S["body"],
    ))

    doc.build(story)
    return buf.getvalue()
