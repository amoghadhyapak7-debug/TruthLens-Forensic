"""
Flask routes for TruthLens Image Forensic Analyzer
Supports:
  - Single-image and multi-image batch analysis via REST API and Web Dashboard
  - Temporary upload portal at /portal with 10 MB file-size enforcement
  - Auto-cleanup of temp uploads older than 30 minutes
"""
import os
import uuid
import time
import json
import hashlib
import datetime
import threading
import traceback
from flask import Blueprint, request, jsonify, send_from_directory, render_template

from truthlens_forensics.pipeline import ImageAnalysisPipeline
from truthlens_forensics.analyzers.gemini_assistant import get_gemini_synthesizer
from truthlens_forensics.config import (
    BASE_DIR, STORAGE_DIR, ARTIFACT_DIR, EVIDENCE_DIR, TEMP_DIR,
    MAX_FILE_SIZE_BYTES, TEMP_TTL_SECONDS, TEMP_CLEANUP_INTERVAL,
)

main_bp = Blueprint("main", __name__)

MAX_FILE_SIZE = MAX_FILE_SIZE_BYTES
TEMP_TTL_SEC  = TEMP_TTL_SECONDS
REGISTRY_PATH = os.path.join(EVIDENCE_DIR, "registry.json")


def _get_registry() -> list:
    """Returns the list of all stored and analyzed evidence records."""
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _update_registry(record: dict):
    """Appends or updates an evidence item in the registry."""
    try:
        registry = _get_registry()
        ev_id = record.get("evidence_id")
        # Remove existing if updating
        registry = [r for r in registry if r.get("evidence_id") != ev_id]
        registry.insert(0, record)
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
    except Exception as exc:
        print(f"Warning: Failed to update evidence registry: {exc}")


# Lazy singleton pipeline
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = ImageAnalysisPipeline(artifact_dir=ARTIFACT_DIR)
    return _pipeline


# ─────────────────────────── Temp-upload cleanup ───────────────────────────

def _cleanup_temp_uploads():
    """Background thread: delete temp upload folders older than TEMP_TTL_SEC."""
    while True:
        try:
            now = time.time()
            for entry in os.scandir(TEMP_DIR):
                if entry.is_dir():
                    age = now - entry.stat().st_mtime
                    if age > TEMP_TTL_SEC:
                        import shutil
                        shutil.rmtree(entry.path, ignore_errors=True)
        except Exception:
            pass
        time.sleep(TEMP_CLEANUP_INTERVAL)


_cleanup_thread = threading.Thread(target=_cleanup_temp_uploads, daemon=True)
_cleanup_thread.start()


# ─────────────────────────── Helper ────────────────────────────────────────

def _file_size_error(size_bytes):
    mb = size_bytes / (1024 * 1024)
    return jsonify({
        "error": f"File size {mb:.2f} MB exceeds the 10 MB limit. Please upload an image smaller than 10 MB.",
        "max_allowed_mb": 10,
        "uploaded_mb": round(mb, 2),
    }), 413


def _compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────── Pages ─────────────────────────────────────────

@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/portal")
def portal():
    """Temporary upload portal page."""
    return render_template("portal.html")


# ─────────────────────────── Health ────────────────────────────────────────

@main_bp.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "TruthLens Image Forensic Analyzer",
        "version": "2.0.0",
        "max_upload_mb": 10,
        "capabilities": [
            "single_image_forensics",
            "multi_image_batch_forensics",
            "prnu_sensor_fingerprinting",
            "wavelet_noise_residual",
            "pretrained_ai_generation_detection",
            "pretrained_deepfake_spatial_localization",
            "error_level_analysis",
            "copy_move_pca",
            "exif_metadata_integrity",
        ]
    })


# ─────────────────────────── Temp-upload Portal API ────────────────────────

@main_bp.route("/api/portal/upload", methods=["POST"])
def portal_upload():
    """
    Accepts a file upload into storage/temp_uploads/<session_id>/.
    Returns a session token the caller can pass to /api/portal/analyze.
    Does NOT run analysis — just validates size + format and stores temporarily.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file attached"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Read into memory first to check size before writing
    data = file.read()
    if len(data) > MAX_FILE_SIZE:
        return _file_size_error(len(data))

    session_id = uuid.uuid4().hex
    session_dir = os.path.join(TEMP_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    filename = file.filename or "upload_file"
    raw_ext = os.path.splitext(filename)[1].lower()
    if not raw_ext:
        if data.startswith(b"\xFF\xD8"): raw_ext = ".jpg"
        elif data.startswith(b"\x89PNG"): raw_ext = ".png"
        elif data.startswith(b"RIFF") and b"WEBP" in data[:16]: raw_ext = ".webp"
        elif data.startswith(b"BM"): raw_ext = ".bmp"
        elif data.startswith(b"ftyp") or (data.startswith(b"RIFF") and b"AVI " in data[:16]): raw_ext = ".mp4"
        else: raw_ext = ".jpg"

    dest = os.path.join(session_dir, "upload_file")
    with open(dest, "wb") as f:
        f.write(data)

    dest_ext = os.path.join(session_dir, f"upload_file{raw_ext}")
    if dest_ext != dest:
        with open(dest_ext, "wb") as f:
            f.write(data)

    sha256 = _compute_sha256(dest)
    size   = len(data)
    mb     = round(size / (1024 * 1024), 3)

    return jsonify({
        "status": "uploaded",
        "session_id": session_id,
        "original_filename": file.filename,
        "file_size_bytes": size,
        "file_size_mb": mb,
        "sha256": sha256,
        "mime_type": file.mimetype or "image/jpeg",
        "temp_path": os.path.join("storage", "temp_uploads", session_id, f"upload_file{raw_ext}"),
        "expires_in_minutes": TEMP_TTL_SEC // 60,
        "analyze_url": f"/api/portal/analyze/{session_id}",
        "message": f"File stored temporarily. It will be auto-deleted after {TEMP_TTL_SEC // 60} minutes.",
    })


@main_bp.route("/api/portal/analyze/<session_id>", methods=["POST"])
def portal_analyze(session_id):
    """Runs forensic analysis on a previously uploaded temp file."""
    session_dir = os.path.join(TEMP_DIR, session_id)
    dest = os.path.join(session_dir, "upload_file")

    # Prefer file with extension if available
    target_file = dest
    if os.path.exists(session_dir):
        for entry in os.listdir(session_dir):
            if entry.startswith("upload_file") and entry != "upload_file":
                target_file = os.path.join(session_dir, entry)
                break

    if not os.path.exists(target_file) and not os.path.exists(dest):
        return jsonify({"error": "Session expired or not found. Please re-upload the file."}), 404

    active_path = target_file if os.path.exists(target_file) else dest

    # Read metadata saved alongside
    sha256 = _compute_sha256(active_path)
    size   = os.path.getsize(active_path)
    mime   = request.json.get("mime_type", "image/jpeg") if request.is_json else "image/jpeg"
    fname  = request.json.get("original_filename", "upload_file") if request.is_json else "upload_file"
    ev_id  = request.json.get("evidence_id", f"EVD-{session_id[:6].upper()}") if request.is_json else f"EVD-{session_id[:6].upper()}"

    payload = {
        "evidence_id": ev_id,
        "file_path": active_path,
        "original_filename": fname,
        "file_type": "image",
        "mime_type": mime,
        "file_size": size,
        "sha256": sha256,
        "ingestion_metadata": {
            "uploaded_via": "temp_portal",
            "session_id": session_id,
        },
    }

    try:
        pipeline = get_pipeline()
        report = pipeline.analyze(payload)
        return jsonify(report)
    except Exception as exc:
        return jsonify({
            "module": "image",
            "evidence_id": ev_id,
            "status": "failed",
            "errors": [{"engine": "pipeline", "error_type": type(exc).__name__, "message": str(exc)}],
            "metadata": {}, "analyses": {}, "signals": [], "findings": [], "artifacts": [], "limitations": []
        }), 500


@main_bp.route("/api/portal/status/<session_id>", methods=["GET"])
def portal_status(session_id):
    """Returns status and TTL of a temp upload session."""
    session_dir = os.path.join(TEMP_DIR, session_id)
    dest = os.path.join(session_dir, "upload_file")
    if not os.path.exists(dest):
        return jsonify({"status": "expired_or_not_found"}), 404

    mtime   = os.path.getmtime(dest)
    age_sec = time.time() - mtime
    remaining = max(0, TEMP_TTL_SEC - age_sec)
    return jsonify({
        "status": "active",
        "session_id": session_id,
        "age_seconds": round(age_sec),
        "expires_in_seconds": round(remaining),
        "expires_in_minutes": round(remaining / 60, 1),
    })


@main_bp.route("/api/portal/delete/<session_id>", methods=["DELETE"])
def portal_delete(session_id):
    """Manually deletes a temp upload session."""
    import shutil
    session_dir = os.path.join(TEMP_DIR, session_id)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir, ignore_errors=True)
        return jsonify({"status": "deleted", "session_id": session_id})
    return jsonify({"status": "not_found"}), 404


# ─────────────────────────── Main Analyze API ──────────────────────────────

@main_bp.route("/api/analyze", methods=["POST"])
def analyze():
    pipeline = get_pipeline()

    # Case 1: JSON body (Single item or Batch)
    if request.is_json:
        data = request.get_json() or {}

        if isinstance(data, list):
            return jsonify(pipeline.analyze_batch(data))
        if "evidence_items" in data and isinstance(data["evidence_items"], list):
            return jsonify(pipeline.analyze_batch(data["evidence_items"]))

        file_path = data.get("file_path", "")
        if not file_path:
            return jsonify({"error": "file_path is required in JSON payload"}), 400

        # Enforce 10 MB limit on the referenced file
        abs_path = os.path.join(BASE_DIR, file_path) if not os.path.isabs(file_path) else file_path
        if os.path.exists(abs_path) and os.path.getsize(abs_path) > MAX_FILE_SIZE:
            return _file_size_error(os.path.getsize(abs_path))

        try:
            report = pipeline.analyze(data)
            return jsonify(report)
        except Exception as exc:
            tb = traceback.format_exc()
            ev_id = data.get("evidence_id", "EVD-UNKNOWN")
            return jsonify({
                "module": "image", "evidence_id": ev_id, "status": "failed",
                "errors": [{"engine": "pipeline", "error_type": type(exc).__name__, "message": str(exc), "traceback": tb}],
                "metadata": {}, "analyses": {}, "signals": [], "findings": [], "artifacts": [], "limitations": []
            }), 500

    # Case 2: Multipart form-data file upload
    if "file" not in request.files:
        return jsonify({"error": "No file or JSON body in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Read entirely to enforce 10 MB before writing anywhere
    data_bytes = file.read()
    if len(data_bytes) > MAX_FILE_SIZE:
        return _file_size_error(len(data_bytes))

    evidence_id = request.form.get("evidence_id", f"EVD-{uuid.uuid4().hex[:6].upper()}")
    ev_dir = os.path.join(STORAGE_DIR, "evidence", evidence_id)
    os.makedirs(ev_dir, exist_ok=True)

    filename = file.filename or "uploaded_evidence"
    ext = os.path.splitext(filename)[1].lower()

    # Intelligent Auto-Routing: If user uploaded a document to the image portal, analyze with text engine
    from truthlens_forensics.config import SUPPORTED_TEXT_EXTENSIONS
    if ext in SUPPORTED_TEXT_EXTENSIONS or data_bytes.startswith(b"%PDF"):
        from truthlens_forensics.analyzers.text import analyze_text
        from truthlens_forensics.text_routes import _text_reports, TEXT_ARTIFACT_DIR
        text_dest = os.path.join(ev_dir, filename)
        with open(text_dest, "wb") as f:
            f.write(data_bytes)
        sha256 = _compute_sha256(text_dest)
        text_payload = {
            "evidence_id": evidence_id,
            "file_path": text_dest,
            "original_filename": filename,
            "file_type": "text",
            "mime_type": file.mimetype or "text/plain",
            "file_size": len(data_bytes),
            "sha256": sha256,
            "ingestion_metadata": {
                "artifact_dir": os.path.join(TEXT_ARTIFACT_DIR, evidence_id),
                "source": "web_ui_auto_routed"
            }
        }
        text_report = analyze_text(text_payload)
        _text_reports[evidence_id] = text_report
        text_report["auto_routed_text"] = True
        return jsonify(text_report)

    # Determine extension to guarantee image decoders never fail
    raw_ext = ext
    if not raw_ext:
        if data_bytes.startswith(b"\xFF\xD8"): raw_ext = ".jpg"
        elif data_bytes.startswith(b"\x89PNG"): raw_ext = ".png"
        elif data_bytes.startswith(b"RIFF") and b"WEBP" in data_bytes[:16]: raw_ext = ".webp"
        elif data_bytes.startswith(b"BM"): raw_ext = ".bmp"
        elif data_bytes.startswith(b"ftyp") or (data_bytes.startswith(b"RIFF") and b"AVI " in data_bytes[:16]): raw_ext = ".mp4"
        else: raw_ext = ".jpg"

    dest_file_ext = os.path.join(ev_dir, f"original_file{raw_ext}")
    with open(dest_file_ext, "wb") as f:
        f.write(data_bytes)

    # Maintain extensionless alias for backward compatibility
    dest_file = os.path.join(ev_dir, "original_file")
    if dest_file != dest_file_ext:
        with open(dest_file, "wb") as f:
            f.write(data_bytes)

    sha256 = _compute_sha256(dest_file_ext)

    payload = {
        "evidence_id": evidence_id,
        "file_path": dest_file_ext,
        "original_filename": file.filename,
        "file_type": "video" if raw_ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"} else "image",
        "mime_type": file.mimetype or ("video/mp4" if raw_ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"} else "image/jpeg"),
        "file_size": len(data_bytes),
        "sha256": sha256,
        "ingestion_metadata": {"uploaded_via": "web_ui"},
    }

    try:
        report = pipeline.analyze(payload)

        # ── Permanently archive report & update registry ─────────
        try:
            report_file = os.path.join(ev_dir, "forensic_report.json")
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            meta_file = os.path.join(ev_dir, "metadata.json")
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump({
                    "evidence_id": evidence_id,
                    "original_filename": file.filename,
                    "file_size": len(data_bytes),
                    "sha256": sha256,
                    "mime_type": file.mimetype or "image/jpeg",
                    "uploaded_at": datetime.datetime.utcnow().isoformat() + "Z",
                }, f, indent=2)

            cat = report.get("analyses", {}).get("overall_forensic_assessment", {}).get("category", "inconclusive")
            _update_registry({
                "evidence_id": evidence_id,
                "original_filename": file.filename,
                "file_size_bytes": len(data_bytes),
                "sha256": sha256,
                "analyzed_at": datetime.datetime.utcnow().isoformat() + "Z",
                "category": cat,
                "signals_count": len(report.get("signals", [])),
                "findings_count": len(report.get("findings", [])),
                "artifacts_count": len(report.get("artifacts", [])),
                "file_path": os.path.relpath(dest_file, BASE_DIR).replace("\\", "/"),
            })
        except Exception as arc_err:
            print(f"Warning: Failed to archive evidence report: {arc_err}")

        return jsonify(report)
    except Exception as exc:
        tb = traceback.format_exc()
        return jsonify({
            "module": "image", "evidence_id": evidence_id, "status": "failed",
            "errors": [{"engine": "pipeline", "error_type": type(exc).__name__, "message": str(exc), "traceback": tb}],
            "metadata": {}, "analyses": {}, "signals": [], "findings": [], "artifacts": [], "limitations": []
        }), 500


# ─────────────────────────── Stored Evidence API ────────────────────────────

@main_bp.route("/api/evidence", methods=["GET"])
def list_stored_evidence():
    """Returns the list of all stored and analyzed evidence items."""
    return jsonify({
        "status": "ok",
        "total": len(_get_registry()),
        "evidence": _get_registry()
    })


@main_bp.route("/api/evidence/<evidence_id>", methods=["GET"])
def get_stored_evidence(evidence_id):
    """Retrieves the forensic report of a previously analyzed evidence item."""
    ev_dir = os.path.join(EVIDENCE_DIR, evidence_id)
    report_file = os.path.join(ev_dir, "forensic_report.json")
    if os.path.exists(report_file):
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({"error": f"Failed to read report: {e}"}), 500
    return jsonify({"error": f"Evidence {evidence_id} not found"}), 404


@main_bp.route("/api/evidence/<evidence_id>/file", methods=["GET"])
def get_stored_evidence_file(evidence_id):
    """Serves the original stored evidence file."""
    ev_dir = os.path.join(EVIDENCE_DIR, evidence_id)
    file_path = os.path.join(ev_dir, "original_file")
    if os.path.exists(file_path):
        return send_from_directory(ev_dir, "original_file")
    return jsonify({"error": "File not found"}), 404


# ─────────────────────────── Batch ─────────────────────────────────────────

@main_bp.route("/api/analyze-batch", methods=["POST"])
def analyze_batch():
    pipeline = get_pipeline()
    if not request.is_json:
        return jsonify({"error": "Request must be JSON with a list of evidence items or {'evidence_items': [...]}"}), 400

    data = request.get_json() or {}
    if isinstance(data, list):
        items = data
    elif "evidence_items" in data and isinstance(data["evidence_items"], list):
        items = data["evidence_items"]
    else:
        return jsonify({"error": "Expected a list of evidence items or {'evidence_items': [...]}"}), 400

    try:
        batch_report = pipeline.analyze_batch(items)
        return jsonify(batch_report)
    except Exception as exc:
        tb = traceback.format_exc()
        return jsonify({
            "module": "image_batch", "total_submitted": len(items), "status": "failed",
            "errors": [{"engine": "pipeline", "error_type": type(exc).__name__, "message": str(exc), "traceback": tb}],
            "reports": [], "comparative_verification": {}
        }), 500


# ─────────────────────────── Artifact serving ──────────────────────────────

@main_bp.route("/artifacts/<path:filename>")
def serve_artifact(filename):
    primary_path = os.path.join(ARTIFACT_DIR, filename)
    if os.path.exists(primary_path):
        return send_from_directory(os.path.dirname(primary_path), os.path.basename(primary_path))
    legacy_dir = os.path.join(BASE_DIR, "artifacts")
    return send_from_directory(legacy_dir, filename)


@main_bp.route("/storage/artifacts/<path:filename>")
def serve_storage_artifact(filename):
    file_path = os.path.join(ARTIFACT_DIR, filename)
    return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path))


# ─────────────────────────── Gemini Forensic Intelligence ─────────────────

@main_bp.route("/api/gemini/summarize", methods=["POST"])
def gemini_summarize():
    """
    Synthesizes complex forensic telemetry into an executive legal verdict,
    calibrated confidence assessment, and courtroom cross-examination questions
    powered by Google Gemini.
    """
    data = request.get_json(silent=True) or {}
    report = data.get("report")

    if not report and "evidence_id" in data:
        ev_id = data["evidence_id"]
        ev_dir = os.path.join(EVIDENCE_DIR, ev_id)
        report_file = os.path.join(ev_dir, "forensic_report.json")
        if os.path.exists(report_file):
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    report = json.load(f)
            except Exception:
                pass

    if not report:
        return jsonify({"error": "Valid forensic report or evidence_id required"}), 400

    try:
        synthesizer = get_gemini_synthesizer()
        synthesis = synthesizer.synthesize(report)
        return jsonify(synthesis)
    except Exception as exc:
        return jsonify({
            "error": "Gemini synthesis encountered an unexpected error",
            "details": str(exc),
            "source": "error_handler"
        }), 500


# ─────────────────────────── Instant Demo Samples API ──────────────────────

@main_bp.route("/api/demo/samples", methods=["GET"])
def get_demo_samples():
    """Provides seed sample metadata for judges to test with 1-click."""
    return jsonify({
        "status": "ok",
        "samples": [
            {
                "id": "sample-tampered",
                "title": "Tampered / Spliced Evidence",
                "type": "image",
                "description": "Composite image with localized patch splicing, ELA variance, and sensor mismatch.",
                "badge": "Tampered",
                "severity": "high"
            },
            {
                "id": "sample-ai",
                "title": "AI-Generated / Deepfake Image",
                "type": "image",
                "description": "Synthetic generation showing stripped camera EXIF, frequency anomalies, and high AI probability.",
                "badge": "AI Generated",
                "severity": "high"
            },
            {
                "id": "sample-authentic",
                "title": "Authentic Optical Photograph",
                "type": "image",
                "description": "Pristine digital camera capture with natural CMOS sensor PRNU noise and uniform JPEG compression.",
                "badge": "Authentic",
                "severity": "low"
            }
        ]
    })


@main_bp.route("/api/demo/load/<sample_id>", methods=["POST"])
def load_demo_sample(sample_id):
    """Generates or loads a real seed sample and analyzes it immediately."""
    from truthlens_forensics.generate_test_evidence import (
        create_authentic_evidence,
        create_manipulated_evidence,
        create_ai_generated_evidence,
    )
    pipeline = get_pipeline()

    try:
        if sample_id == "sample-authentic":
            ev_input = create_authentic_evidence()
        elif sample_id == "sample-tampered":
            ev_input = create_manipulated_evidence()
        elif sample_id == "sample-ai":
            ev_input = create_ai_generated_evidence()
        else:
            return jsonify({"error": f"Unknown sample ID: {sample_id}"}), 404

        report = pipeline.analyze(ev_input)

        # Permanent archival & registry update
        ev_dir = os.path.join(STORAGE_DIR, "evidence", ev_input["evidence_id"])
        os.makedirs(ev_dir, exist_ok=True)
        report_file = os.path.join(ev_dir, "forensic_report.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        cat = report.get("analyses", {}).get("overall_forensic_assessment", {}).get("category", "inconclusive")
        _update_registry({
            "evidence_id": ev_input["evidence_id"],
            "original_filename": ev_input.get("original_filename", "demo_sample.jpg"),
            "file_size_bytes": ev_input.get("file_size", 0),
            "sha256": ev_input.get("sha256", ""),
            "analyzed_at": datetime.datetime.utcnow().isoformat() + "Z",
            "category": cat,
            "signals_count": len(report.get("signals", [])),
            "findings_count": len(report.get("findings", [])),
            "file_path": ev_input.get("file_path", "").replace("\\", "/"),
        })

        # Pre-synthesize with Gemini
        synthesizer = get_gemini_synthesizer()
        gemini_result = synthesizer.synthesize(report)

        return jsonify({
            "status": "success",
            "report": report,
            "gemini_synthesis": gemini_result
        })
    except Exception as exc:
        tb = traceback.format_exc()
        return jsonify({
            "status": "error",
            "message": str(exc),
            "traceback": tb
        }), 500


