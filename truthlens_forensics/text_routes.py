"""
TruthLens — Text Forensic Analysis Routes (Flask Blueprint)
Converted from FastAPI to Flask, integrated into Sapthagiri's architecture.
"""
import os
import uuid
import hashlib
import shutil
from pathlib import Path
from flask import Blueprint, request, jsonify, render_template, send_from_directory

from truthlens_forensics.analyzers.text import analyze_text
from truthlens_forensics.analyzers.gemini_assistant import get_gemini_synthesizer
from truthlens_forensics.config import BASE_DIR, STORAGE_DIR, ARTIFACT_DIR, MAX_FILE_SIZE_BYTES

text_bp = Blueprint("text", __name__)

# Text-specific paths
TEXT_UPLOAD_DIR = os.path.join(STORAGE_DIR, "text_uploads")
TEXT_ARTIFACT_DIR = os.path.join(ARTIFACT_DIR, "text")
os.makedirs(TEXT_UPLOAD_DIR, exist_ok=True)
os.makedirs(TEXT_ARTIFACT_DIR, exist_ok=True)

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".pdf", ".docx"}

# In-memory report store
_text_reports: dict = {}


def _compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@text_bp.route("/text")
def text_analyzer_page():
    """Renders the text forensic analyzer page."""
    return render_template("text_analyzer.html")


@text_bp.route("/api/text/health")
def text_health():
    return jsonify({
        "status": "ok",
        "service": "TruthLens Text Forensic Analyzer",
        "version": "2.0.0",
        "module": "text",
        "supported_formats": sorted(list(SUPPORTED_TEXT_EXTENSIONS)),
    })


@text_bp.route("/api/text/analyze", methods=["POST"])
def analyze_text_upload():
    """
    Upload and analyze a text evidence document.
    Supports multipart file upload or JSON EvidenceInput.
    """
    # JSON body mode
    if request.is_json:
        data = request.get_json() or {}
        evidence_id = data.get("evidence_id", f"EVD-{uuid.uuid4().hex[:8].upper()}")
        file_path = data.get("file_path", "")
        if not file_path:
            return jsonify({"error": "file_path is required in JSON payload"}), 400
        abs_path = file_path if os.path.isabs(file_path) else os.path.join(BASE_DIR, file_path)
        if not os.path.exists(abs_path):
            return jsonify({"error": f"File not found: {abs_path}"}), 404
        payload = {
            "evidence_id": evidence_id,
            "file_path": abs_path,
            "original_filename": data.get("original_filename", os.path.basename(abs_path)),
            "file_type": "text",
            "mime_type": data.get("mime_type", "text/plain"),
            "file_size": os.path.getsize(abs_path),
            "sha256": data.get("sha256") or _compute_sha256(abs_path),
            "ingestion_metadata": data.get("ingestion_metadata", {"source": "api_json"})
        }
        result = analyze_text(payload)
        _text_reports[evidence_id] = result
        return jsonify(result)

    # Multipart file upload mode
    if "file" not in request.files:
        return jsonify({"error": "No file or JSON body in request"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_TEXT_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file format '{ext}'. Supported: {sorted(list(SUPPORTED_TEXT_EXTENSIONS))}"
        }), 400

    # Read entirely to enforce size limit
    data_bytes = file.read()
    if len(data_bytes) > MAX_FILE_SIZE_BYTES:
        mb = len(data_bytes) / (1024 * 1024)
        return jsonify({
            "error": f"File size {mb:.2f} MB exceeds the 10 MB limit.",
            "max_allowed_mb": 10,
            "uploaded_mb": round(mb, 2),
        }), 413

    evidence_id = request.form.get("evidence_id", f"EVD-{uuid.uuid4().hex[:8].upper()}")
    tmp_id = uuid.uuid4().hex[:12]
    saved_path = os.path.join(TEXT_UPLOAD_DIR, f"{tmp_id}_{filename}")

    with open(saved_path, "wb") as f:
        f.write(data_bytes)

    sha256 = _compute_sha256(saved_path)
    artifact_dir = os.path.join(TEXT_ARTIFACT_DIR, evidence_id)

    payload = {
        "evidence_id": evidence_id,
        "file_path": saved_path,
        "original_filename": filename,
        "file_type": "text",
        "mime_type": file.mimetype or "application/octet-stream",
        "file_size": len(data_bytes),
        "sha256": sha256,
        "ingestion_metadata": {
            "artifact_dir": artifact_dir,
            "source": "web_upload"
        }
    }

    try:
        result = analyze_text(payload)
        _text_reports[evidence_id] = result
        return jsonify(result)
    except Exception as exc:
        return jsonify({
            "module": "text",
            "evidence_id": evidence_id,
            "status": "failed",
            "errors": [{"engine": "text_pipeline", "error_type": type(exc).__name__, "message": str(exc)}],
            "metadata": {}, "analyses": {}, "signals": [], "findings": [], "artifacts": [], "limitations": []
        }), 500


@text_bp.route("/api/text/report/<evidence_id>", methods=["GET"])
def get_text_report(evidence_id: str):
    """Retrieve a cached text forensic report by evidence ID."""
    if evidence_id not in _text_reports:
        return jsonify({"error": f"Text report for {evidence_id} not found"}), 404
    return jsonify(_text_reports[evidence_id])


@text_bp.route("/api/text/evidence", methods=["GET"])
def list_text_evidence():
    """List all analyzed text evidence IDs."""
    return jsonify({
        "status": "ok",
        "total": len(_text_reports),
        "evidence_ids": list(_text_reports.keys())
    })


# ─────────────────────────── Gemini Text Intelligence ──────────────────────

@text_bp.route("/api/text/gemini-summarize", methods=["POST"])
def text_gemini_summarize():
    """
    Synthesizes text forensic output into courtroom-ready intelligence using Google Gemini.
    """
    data = request.get_json(silent=True) or {}
    report = data.get("report")
    if not report and "evidence_id" in data:
        report = _text_reports.get(data["evidence_id"])

    if not report:
        return jsonify({"error": "Valid forensic report or evidence_id required"}), 400

    try:
        synthesizer = get_gemini_synthesizer()
        synthesis = synthesizer.synthesize(report)
        return jsonify(synthesis)
    except Exception as exc:
        return jsonify({
            "error": "Gemini text synthesis encountered an error",
            "details": str(exc)
        }), 500


# ─────────────────────────── Text Demo Samples ─────────────────────────────

TEXT_DEMO_SAMPLES = {
    "sample-disinformation": {
        "title": "Disinformation & Contradictory Statement",
        "filename": "whistleblower_contradiction_sample.txt",
        "text": (
            "PRESS RELEASE: Investigation on Project Orion.\n\n"
            "On March 14, 2024, lead scientist Dr. Raymond Vance verified that the reactor in Berlin was completely offline "
            "and undergoing standard decommissioning protocols.\n\n"
            "However, telemetry logs recovered from the Frankfurt facility show that Dr. Raymond Vance authorized a full-power "
            "thermal cycle on March 14, 2024 at 14:32 UTC in the Berlin facility. Furthermore, in October 2023, Dr. Vance had "
            "already submitted a sworn affidavit stating he had never visited Berlin or operated reactor components outside Zurich.\n\n"
            "Financial statements show an offshore wire transfer of $4,200,000 on March 12, 2024, prior to any public disclosures."
        )
    },
    "sample-synthetic": {
        "title": "AI-Generated Synthetic Technical Article",
        "filename": "synthetic_ai_tech_paper.txt",
        "text": (
            "In recent years, artificial intelligence has revolutionized computational paradigms across diverse global domains. "
            "The overarching integration of neural generative paradigms facilitates unparalleled synergy within multidimensional "
            "data ecosystems.\n\n"
            "It is crucial to recognize that the burgeoning landscape of algorithmic automation necessitates robust frameworks. "
            "Furthermore, stakeholders must systematically leverage synergistic methodologies to foster continuous optimization.\n\n"
            "In conclusion, the seamless convergence of transformative algorithms promises to unlock unprecedented capabilities "
            "while ensuring foundational benchmarks of excellence are meticulously maintained."
        )
    },
    "sample-authentic": {
        "title": "Authentic Official Press Communique",
        "filename": "official_consortium_statement.txt",
        "text": (
            "EUROPEAN ASTRONOMICAL CONSORTIUM - MEDIA COMMUNIQUE\n"
            "Release Date: November 12, 2024 | Location: Paris, France\n\n"
            "The European Astronomical Consortium confirmed the successful calibration of the Atacama Optical Array. "
            "Director Helene Mercier presented the baseline observations today at the Paris Observatory.\n\n"
            "The observations were recorded between October 10 and November 5, 2024. All primary mirror sensors "
            "performed within expected tolerance of 0.02 arcseconds. No anomalous interference was observed during "
            "the three-week calibration window.\n\n"
            "The final dataset will be released to member institutions on December 1, 2024 via the open science portal."
        )
    }
}


@text_bp.route("/api/text/demo/samples", methods=["GET"])
def get_text_demo_samples():
    """Lists available seed text samples for 1-click testing."""
    return jsonify({
        "status": "ok",
        "samples": [
            {
                "id": k,
                "title": v["title"],
                "filename": v["filename"],
                "preview": v["text"][:120] + "..."
            }
            for k, v in TEXT_DEMO_SAMPLES.items()
        ]
    })


@text_bp.route("/api/text/demo/load/<sample_id>", methods=["POST"])
def load_text_demo_sample(sample_id):
    """Executes immediate text analysis on a seed sample."""
    sample = TEXT_DEMO_SAMPLES.get(sample_id)
    if not sample:
        return jsonify({"error": f"Unknown text sample: {sample_id}"}), 404

    evidence_id = f"EVD-{uuid.uuid4().hex[:8].upper()}"
    saved_path = os.path.join(TEXT_UPLOAD_DIR, f"demo_{sample['filename']}")
    with open(saved_path, "w", encoding="utf-8") as f:
        f.write(sample["text"])

    payload = {
        "evidence_id": evidence_id,
        "file_path": saved_path,
        "original_filename": sample["filename"],
        "file_type": "text",
        "mime_type": "text/plain",
        "file_size": len(sample["text"].encode("utf-8")),
        "sha256": _compute_sha256(saved_path),
        "ingestion_metadata": {
            "artifact_dir": os.path.join(TEXT_ARTIFACT_DIR, evidence_id),
            "source": "demo_seed"
        }
    }

    report = analyze_text(payload)
    _text_reports[evidence_id] = report

    # Also run Gemini synthesis
    synthesizer = get_gemini_synthesizer()
    gemini_result = synthesizer.synthesize(report)

    return jsonify({
        "status": "success",
        "sample_id": sample_id,
        "raw_text": sample["text"],
        "report": report,
        "gemini_synthesis": gemini_result
    })


