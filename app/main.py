"""TruthLens ingestion API."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel
from app.analyzers.audio import AudioAnalyzer
from app.analyzers.audio import _read_compressed, _read_wav
from app.analyzers.config import SETTINGS as AUDIO_SETTINGS
from app.analyzers.audio_models import detect_synthetic_audio

from app.ingest.config import LIMITS
from app.ingest.service import IngestService
from app.storage import (
    STORAGE_DIR,
    init_storage,
    get_analysis_file_path,
    get_report_save_path,
    get_extracted_audio_path,
)

init_storage()
app = FastAPI(title="TruthLens INGEST API", version="1.0.0")

# Security: CORS restricted to local frontend hosts (configurable via TRUTHLENS_CORS_ORIGINS)
cors_origins_env = os.getenv("TRUTHLENS_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501,http://localhost:8000,http://127.0.0.1:8000")
allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """OWASP Recommended Secure HTTP Headers."""
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; media-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';"
    return response


ingest_service = IngestService()


class RegistryResetRequest(BaseModel):
    confirmation: str


@app.get("/", response_class=HTMLResponse)
def root_index():
    return """<!DOCTYPE html>
<html>
<head>
    <title>TruthLens Multi-Modal Forensics API</title>
    <style>
        body { background: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 40px; max-width: 600px; text-align: center; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
        h1 { color: #58a6ff; margin-top: 0; font-size: 2rem; }
        p { color: #8b949e; line-height: 1.6; }
        .badge { display: inline-block; background: #238636; color: #fff; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; margin-bottom: 20px; }
        .btn { display: inline-block; background: #238636; color: #fff; text-decoration: none; padding: 12px 28px; border-radius: 8px; font-weight: 600; margin-top: 20px; transition: 0.2s; }
        .btn:hover { background: #2ea043; }
        .endpoints { text-align: left; background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-top: 24px; font-family: monospace; font-size: 0.85rem; }
        .endpoints div { margin: 6px 0; color: #79c0ff; }
    </style>
</head>
<body>
    <div class="card">
        <div class="badge">TruthLens v2.0 Live</div>
        <h1>TruthLens Forensic API</h1>
        <p>Court-ready digital forensics platform for tamper-evident multi-modal evidence intake, deepfake detection, cross-modality correlation, examiner review, and reporting.</p>
        <a href="/docs" class="btn">Explore Interactive API Docs (Swagger UI)</a>
        <div class="endpoints">
            <div>GET /api/v1/health &bull; Service Health Check</div>
            <div>GET /api/v1/evidence &bull; Ingested Evidence Registry</div>
            <div>POST /api/v1/ingest &bull; Multi-Modal Evidence Upload</div>
            <div>GET /api/v1/correlation/case &bull; Cross-Modal Synthesis</div>
            <div>GET /api/v1/report/all?format=pdf &bull; Forensic Report</div>
        </div>
    </div>
</body>
</html>"""


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "truthlens-ingest"}


@app.post("/api/v1/ingest")
def ingest(files: Annotated[list[UploadFile], File()], case_id: Annotated[str | None, Form()] = None) -> dict:
    if len(files) > LIMITS.max_files_per_batch:
        raise HTTPException(status_code=413, detail=f"A batch can contain at most {LIMITS.max_files_per_batch} files.")
    
    # Input validation & sanitization for optional case_id
    clean_case_id = None
    if case_id is not None:
        trimmed = case_id.strip()
        if len(trimmed) > 64 or not re.fullmatch(r"[A-Za-z0-9_\-\. ]*", trimmed):
            raise HTTPException(status_code=422, detail="Invalid case_id format. Must be alphanumeric and max 64 characters.")
        clean_case_id = trimmed or None

    sizes: list[int] = []
    for upload in files:
        size = upload.size
        if size is None:
            upload.file.seek(0, 2)
            size = upload.file.tell()
            upload.file.seek(0)
        sizes.append(int(size))
    if any(size > LIMITS.max_upload_file_bytes for size in sizes):
        raise HTTPException(status_code=413, detail=f"A file exceeds the {LIMITS.max_upload_file_bytes}-byte upload cap.")
    if sum(sizes) > LIMITS.max_batch_bytes:
        raise HTTPException(status_code=413, detail=f"Batch exceeds the {LIMITS.max_batch_bytes}-byte total upload cap.")
    results = [ingest_service.ingest(file.filename or "unnamed", file.file, clean_case_id) for file in files]
    return {"results": results, "count": len(results)}


@app.get("/api/v1/evidence")
def list_evidence() -> dict:
    return {"evidence": ingest_service.list_evidence()}


@app.post("/api/v1/evidence/reset")
def reset_evidence_registry(payload: RegistryResetRequest) -> dict:
    try:
        return ingest_service.clear_evidence_registry(payload.confirmation)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ── Individual Evidence Deletion & Testing Management ────────────────────────

@app.delete("/api/v1/evidence/{evidence_id}")
def delete_single_evidence(evidence_id: str) -> dict:
    """Delete a single evidence record and all its associated files."""
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    result = ingest_service.delete_evidence(evidence_id)
    if not result["success"]:
        raise HTTPException(status_code=500, detail={"message": "Deletion incomplete", "errors": result["errors"]})
    # Also clear in-memory deepfake cache
    _deepfake_memory_cache.pop(evidence_id, None)
    return result


@app.get("/api/v1/testing/list")
def list_testing_evidence() -> dict:
    """List evidence records classified as testing uploads."""
    records = ingest_service.list_evidence()
    testing = []
    real = []
    for r in records:
        entry = {
            "evidence_id": r.get("evidence_id"),
            "original_filename": r.get("original_filename"),
            "file_type": r.get("file_type"),
            "file_size_human": r.get("file_size_human"),
            "case_id": r.get("case_id"),
            "ingestion_timestamp": r.get("ingestion_timestamp"),
            "is_testing": ingest_service.classify_as_testing(r),
        }
        if entry["is_testing"]:
            testing.append(entry)
        else:
            real.append(entry)
    return {"testing": testing, "real_evidence": real, "testing_count": len(testing), "real_count": len(real)}


@app.get("/api/v1/database/stats")
def get_database_stats() -> dict:
    """Return local SQLite database and storage architecture stats."""
    return ingest_service.db.get_database_stats()


class ClearTestingRequest(BaseModel):
    confirmation: str


@app.post("/api/v1/testing/clear-all")
def clear_all_testing_evidence(payload: ClearTestingRequest) -> dict:
    """Delete all evidence classified as testing uploads."""
    if payload.confirmation != "CLEAR TESTING FILES":
        raise HTTPException(status_code=422, detail="Type 'CLEAR TESTING FILES' exactly to confirm.")
    records = ingest_service.list_evidence()
    testing_records = [r for r in records if ingest_service.classify_as_testing(r)]
    results = []
    for r in testing_records:
        ev_id = r.get("evidence_id")
        del_result = ingest_service.delete_evidence(ev_id)
        _deepfake_memory_cache.pop(ev_id, None)
        results.append(del_result)
    succeeded = sum(1 for r in results if r.get("success"))
    failed = sum(1 for r in results if not r.get("success"))
    return {
        "deleted_count": succeeded,
        "failed_count": failed,
        "results": results,
        "complete": failed == 0,
    }


@app.get("/api/v1/diagnostics/registry", include_in_schema=False)
def registry_diagnostics(request: Request) -> dict:
    """Developer-facing registry source/count details for monitor diagnostics."""
    if request.query_params.get("debug") != "true":
        raise HTTPException(status_code=404, detail="Not found")
    records = ingest_service.list_evidence()
    ids = [
        int(match.group(1))
        for record in records
        if (match := re.fullmatch(r"EVD-(\d+)", str(record.get("evidence_id", ""))))
    ]
    return {
        "registry_path": str(ingest_service.registry_path),
        "evidence_storage_path": str(ingest_service.evidence_dir),
        "persisted_record_count": len(records),
        "latest_evidence_id": f"EVD-{max(ids):03d}" if ids else None,
    }


@app.get("/api/v1/evidence/readiness")
def evidence_readiness_all() -> dict:
    records = ingest_service.list_evidence()
    return {
        "readiness": {
            record["evidence_id"]: ingest_service.get_analyzer_readiness(record["evidence_id"])
            for record in records if record.get("evidence_id")
        }
    }


@app.get("/api/v1/evidence/{evidence_id}")
def get_evidence(evidence_id: str) -> dict:
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    record = ingest_service.get_evidence(evidence_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    return record


@app.get("/api/v1/evidence/{evidence_id}/readiness")
def evidence_readiness(evidence_id: str) -> dict:
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    if ingest_service.get_evidence(evidence_id) is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    return ingest_service.get_analyzer_readiness(evidence_id)


@app.get("/api/v1/evidence/{evidence_id}/analyzer-input")
def analyzer_input(evidence_id: str) -> dict:
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    try:
        return ingest_service.build_analyzer_input(evidence_id)
    except ValueError as exc:
        status_code = 404 if ingest_service.get_evidence(evidence_id) is None else 409
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@app.post("/api/v1/analyzers/audio/{evidence_id}")
def analyze_audio(evidence_id: str, transcribe_speech: bool = Query(False),
                  detect_synthetic_audio: bool = Query(False), language: str | None = Query(None)) -> dict:
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    readiness = ingest_service.get_analyzer_readiness(evidence_id)
    if readiness["status"] not in {"READY", "READY_WITH_WARNINGS"}:
        code = 404 if ingest_service.get_evidence(evidence_id) is None else 409
        raise HTTPException(status_code=code, detail={"message": "Audio evidence is not ready.", "readiness": readiness})
    try:
        contract = ingest_service.build_analyzer_input(evidence_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if contract["file_type"] != "audio":
        raise HTTPException(status_code=422, detail="Audio Analyzer accepts audio evidence only.")
    return AudioAnalyzer(ingest_service.data_dir / "artifacts").analyze(
        contract, run_transcription=transcribe_speech,
        run_synthetic_detection=detect_synthetic_audio, language=language,
    )


_deepfake_memory_cache: dict[str, dict] = {}


def _save_deepfake_result(evidence_id: str, result: dict) -> None:
    _deepfake_memory_cache[evidence_id] = result
    # Save to service storage_dir/analysis/audio/
    try:
        central_path = Path(ingest_service.storage_dir) / "analysis" / "audio" / f"{evidence_id}.json"
        central_path.parent.mkdir(parents=True, exist_ok=True)
        central_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass
    # Legacy fallback to data/deepfake/
    try:
        out = Path(ingest_service.data_dir) / "deepfake"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{evidence_id}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def _load_deepfake_result(evidence_id: str) -> dict | None:
    if evidence_id in _deepfake_memory_cache:
        return _deepfake_memory_cache[evidence_id]
    # Check service storage_dir first
    try:
        central_path = Path(ingest_service.storage_dir) / "analysis" / "audio" / f"{evidence_id}.json"
        if central_path.is_file():
            data = json.loads(central_path.read_text(encoding="utf-8"))
            _deepfake_memory_cache[evidence_id] = data
            return data
    except Exception:
        pass
    # Fallback to legacy path
    path = Path(ingest_service.data_dir) / "deepfake" / f"{evidence_id}.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _deepfake_memory_cache[evidence_id] = data
            return data
        except Exception:
            return None
    return None


def _default_model_info(detected: dict | None = None) -> dict:
    det = detected or {}
    return {
        "model_name": det.get("detector_name") or "Wav2Vec2 audio spoof classifier (ONNX int4)",
        "model_id": AUDIO_SETTINGS.deepfake_model,
        "architecture": "Wav2Vec2 (Transformer acoustic feature encoder + classification head)",
        "source": f"https://huggingface.co/{AUDIO_SETTINGS.deepfake_model}",
        "revision": AUDIO_SETTINGS.deepfake_model_revision,
        "runtime": "ONNX Runtime (CPUExecutionProvider)",
        "quantization": "int4 ONNX",
        "supported_domain": "Speech anti-spoofing and synthetic speech detection",
        "license": det.get("license") or "MIT (model card)",
        "training_domain": det.get("training_domain") or "ASVspoof 2021 PA speech anti-spoofing",
        "limitations": det.get("limitations") or [
            "Trained primarily on ASVspoof 2021 PA benchmark audio.",
            "Performance may vary with compression codecs, acoustic environments, noise, accents, and languages.",
            "Model scores are uncalibrated softmax indicators, not calibrated probabilities.",
            "Does not verify identity, lip-sync, or recording device origins.",
        ],
    }


def _classify_score_band(score: float | None) -> dict:
    """10-level forensic model score band system in 10% increments.
    
    NOTE: These are explicitly MODEL SCORE BANDS describing relative acoustic feature activations,
    NOT probabilities of manipulation, Bayesian likelihoods, or classification accuracy percentages.
    The score band is a descriptive feature-space metric and does NOT override the final verdict.
    """
    if score is None:
        return {
            "band_index": None,
            "band_range": "N/A",
            "band_label": "Score Unavailable",
            "indicator_strength": "Unavailable",
            "color": "#6B7280",
            "badge": "UNAVAILABLE",
            "band_type": "Model Score Band (uncalibrated acoustic indicator; not probability)",
            "explanation": "Inference did not yield a valid acoustic indicator score.",
            "forensic_caution": "Audio quality or execution failure prevented evaluation.",
            "recommended_action": "Inspect audio encoding or resubmit alternative recording.",
        }
    val = max(0.0, min(1.0, float(score)))
    pct = val * 100.0

    bands = [
        (0.0, 10.0, 1, "0-10%", "Negligible Synthetic Indicators", "Negligible", "#10B981", "MINIMAL INDICATORS",
         "Model score band: Acoustic feature representations closely align with natural speech patterns with minimal neural spoof activation.",
         "Does not rule out physical recording tampering, splicing, or unseen generation techniques.",
         "Document baseline acoustic features; verify chain of custody."),
        (10.0, 20.0, 2, "10-20%", "Very Low Synthetic Indicators", "Very Low", "#10B981", "VERY LOW INDICATORS",
         "Model score band: Acoustic representations align predominantly with organic vocal characteristics; spoof activation is minimal.",
         "Compression artifacts or telephone codecs can occasionally distort natural variance.",
         "Proceed with standard intake verification; check container integrity."),
        (20.0, 30.0, 3, "20-30%", "Low Synthetic Indicators", "Low", "#34D399", "LOW INDICATORS",
         "Model score band: Neural acoustic patterns indicate natural speech dynamics; synthetic indicators remain low.",
         "Low scores do not constitute definitive proof of authenticity.",
         "Corroborate with background noise consistency and metadata timestamps."),
        (30.0, 40.0, 4, "30-40%", "Moderate-Low Synthetic Indicators", "Moderate-Low", "#6EE7B7", "MODERATE-LOW INDICATORS",
         "Model score band: Minor acoustic deviations noted, but classification remains skewed toward bonafide natural voice.",
         "Acoustic reverberation or low-pass filtering can trigger minor spectral anomalies.",
         "Check for room acoustic continuity across speech pauses."),
        (40.0, 50.0, 5, "40-50%", "Intermediate Baseline Indicators", "Intermediate Baseline", "#FBBF24", "INTERMEDIATE BASELINE",
         "Model score band: Acoustic feature activations fall within normal baseline voice variance below detection thresholds.",
         "Environmental noise, microphone coloration, or lossy audio encoding commonly elevate baseline acoustic features into this band.",
         "Corroborate with container metadata and recording environment context."),
        (50.0, 60.0, 6, "50-60%", "Upper Baseline Indicators", "Upper Baseline", "#F59E0B", "UPPER BASELINE",
         "Model score band: Moderate acoustic feature activation within baseline limits (<= 0.60). Natural speech recorded via webcams or compressed codecs often falls here.",
         "This score band reflects uncalibrated acoustic feature distance; it does not indicate AI generation and is below the detection threshold.",
         "Classify as NO STRONG SYNTHETIC INDICATORS unless contradictory segment evidence exists."),
        (60.0, 70.0, 7, "60-70%", "Borderline Indeterminate Band", "Borderline Inconclusive", "#F97316", "BORDERLINE INDETERMINATE",
         "Model score band: Acoustic feature representations fall in the indeterminate boundary region (0.60 – 0.70) between bonafide baseline and synthetic patterns.",
         "Neither bonafide baseline nor synthetic indicators are sufficiently decisive for classification.",
         "Classify as INCONCLUSIVE; request pristine uncompressed audio or secondary verification."),
        (70.0, 80.0, 8, "70-80%", "Moderate-High Synthetic Indicators", "Moderate-High", "#EF4444", "ELEVATED SUSPICION",
         "Model score band: Substantial acoustic spoof indicators detected; model feature activations significantly favor synthetic speech.",
         "Certain studio-compressed professional voices or autotune can trigger false elevations.",
         "Flag for forensic review; compare spectrogram against known speaker reference."),
        (80.0, 90.0, 9, "80-90%", "High Synthetic Indicators", "High", "#DC2626", "HIGH SUSPICION",
         "Model score band: Strong neural synthetic indicators identified across speech frames; high alignment with trained deepfake signatures.",
         "Not a calibrated probability of forgery; score reflects feature-space distance from natural benchmarks.",
         "Treat as high-risk synthetic candidate; conduct linguistic and voiceprint verification."),
        (90.0, 100.0, 10, "90-100%", "Very High Synthetic Indicators", "Very High", "#991B1B", "STRONG SUSPICION",
         "Model score band: Decisive synthetic acoustic patterns detected across speech segments; pronounced acoustic spoof artifacts.",
         "Even extreme model scores are heuristic indicators and do not identify the author or synthesis model.",
         "Prioritize forensic investigation; isolate suspected cloned phonemes and inspect container origin."),
    ]
    for min_p, max_p, idx, rng, lbl, ind, col, bdg, exp, cau, act in bands:
        if (pct >= min_p and pct <= max_p) or (pct >= min_p and max_p == 100.0):
            return {
                "band_index": idx,
                "band_range": rng,
                "band_label": lbl,
                "indicator_strength": ind,
                "color": col,
                "badge": bdg,
                "band_type": "Model Score Band (uncalibrated acoustic indicator; not probability)",
                "explanation": exp,
                "forensic_caution": cau,
                "recommended_action": act,
            }
    return bands[-1]


def _segment_waveform(
    mono: np.ndarray,
    sample_rate: int,
    seg_duration: float = 15.0,
    overlap: float = 1.0,
) -> list[dict]:
    """Programmatically partition a 1D audio waveform into overlapping slices.
    
    Returns a list of dicts with segment metadata:
    index, start_time, end_time, duration, start_sample, end_sample.
    """
    total_samples = len(mono)
    total_duration = total_samples / max(sample_rate, 1)
    if total_duration <= seg_duration:
        return [{
            "segment_index": 1,
            "start_time": 0.0,
            "end_time": round(float(total_duration), 2),
            "duration": round(float(total_duration), 2),
            "start_sample": 0,
            "end_sample": total_samples,
        }]

    step = seg_duration - overlap
    segments = []
    start_t = 0.0
    seg_idx = 1
    while start_t < total_duration:
        end_t = min(start_t + seg_duration, total_duration)
        start_samp = int(round(start_t * sample_rate))
        end_samp = min(int(round(end_t * sample_rate)), total_samples)
        if end_samp > start_samp:
            segments.append({
                "segment_index": seg_idx,
                "start_time": round(float(start_t), 2),
                "end_time": round(float(end_t), 2),
                "duration": round(float(end_t - start_t), 2),
                "start_sample": start_samp,
                "end_sample": end_samp,
            })
            seg_idx += 1
        if end_t >= total_duration:
            break
        start_t += step
    return segments


@app.post("/api/v1/analyzers/audio/{evidence_id}/deepfake")
def detect_audio_deepfake(evidence_id: str) -> dict:
    start_time = time.perf_counter()
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    record = ingest_service.get_evidence(evidence_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    file_path = record.get("file_path")
    if not file_path:
        raise HTTPException(status_code=404, detail="Evidence file path missing in registry")
    path = Path(file_path).resolve()
    try:
        path.relative_to(ingest_service.storage_dir.resolve())
    except ValueError:
        try:
            path.relative_to(ingest_service.evidence_dir.resolve())
        except ValueError:
            try:
                path.relative_to(ingest_service.data_dir.resolve())
            except ValueError:
                raise HTTPException(status_code=404, detail="Evidence path is outside evidence storage")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Evidence original file not found in storage")

    modality = record.get("file_type")
    meta = record.get("metadata") or {}
    mime = (record.get("mime_type") or "").lower()
    ext = path.suffix.lower()

    has_audio = (
        modality == "audio"
        or meta.get("audio_stream_count", 0) > 0
        or meta.get("audio_streams_present") is True
        or mime.startswith("audio/")
        or ext in {".wav", ".mp3", ".m4a"}
        or (modality == "video" and ext in {".mp4", ".mov"})
    )
    if not has_audio:
        raise HTTPException(status_code=422, detail="Evidence does not contain an extractable audio stream.")

    original_filename = record.get("original_filename") or path.name
    is_extracted_from_video = bool(modality == "video" or ext in {".mp4", ".mov"})
    file_format = (meta.get("container_format") or ext.lstrip(".").upper() or "Unknown").upper()
    audio_codec = meta.get("audio_codec") or ("PCM" if ext == ".wav" else None)

    samples: np.ndarray | None = None
    rate: int = AUDIO_SETTINGS.analysis_sample_rate
    original_sample_rate = meta.get("sample_rate")
    original_channels = meta.get("audio_stream_count")
    decoding_method = "Unknown"
    audio_warnings: list[str] = []

    try:
        if ext == ".wav":
            try:
                samples, rate, channels_out, bit_depth = _read_wav(path, AUDIO_SETTINGS)
                decoding_method = f"Native PCM WAV ({bit_depth}-bit)"
                if original_sample_rate is None:
                    original_sample_rate = rate
                if original_channels is None:
                    original_channels = channels_out
            except Exception:
                samples, rate, channels_out, _, _ = _read_compressed(path, AUDIO_SETTINGS)
                decoding_method = "FFmpeg PCM decode (WAV fallback)"
                if original_channels is None:
                    original_channels = channels_out
        else:
            samples, rate, channels_out, _, _ = _read_compressed(path, AUDIO_SETTINGS)
            decoding_method = f"FFmpeg ({meta.get('audio_codec') or 'audio stream'})"
            if original_channels is None:
                original_channels = channels_out
    except Exception as exc:
        duration_proc = round(time.perf_counter() - start_time, 3)
        result = {
            "evidence_id": evidence_id,
            "status": "Failed",
            "result": "INCONCLUSIVE",
            "score": None,
            "model_name": AUDIO_SETTINGS.deepfake_model,
            "explanation": f"Failed to extract audio stream from evidence: {exc}",
            "verdict": {
                "title": "INCONCLUSIVE — INSUFFICIENT EVIDENCE",
                "subtitle": "The available audio or model output does not support a reliable classification.",
                "classification": "INCONCLUSIVE",
                "conclusion": "TruthLens could not reach a confident classification using the available evidence. Additional or higher-quality audio may be required.",
                "summary": f"Audio decoding failed: {exc}",
            },
            "scores": {
                "synthetic_score": None,
                "genuine_score": None,
                "threshold_detected": 0.55,
                "threshold_not_detected": 0.45,
                "score_margin": None,
                "margin_description": "Decoding failed; score unavailable",
                "classification_rule": "Flagged if score >= 0.5500; Not flagged if score <= 0.4500; Inconclusive if 0.4500 < score < 0.5500",
                "interpretation": "This score represents the model's output and is not a calibrated probability of audio manipulation.",
            },
            "why_verdict": {
                "evaluated": "Audio stream extraction attempted via FFmpeg/PCM decoder.",
                "score": None,
                "threshold": "0.4500 - 0.5500",
                "decision": "INCONCLUSIVE",
                "reason": f"Audio stream decoding encountered an error: {exc}",
                "what_it_means": "The audio data could not be extracted into a valid waveform for model analysis.",
                "what_it_does_not_prove": "- It does not prove the audio is genuine or fake.\n- Additional or higher-quality audio evidence is required.",
            },
            "audio_properties": {
                "original_filename": original_filename,
                "format": file_format,
                "audio_codec": audio_codec,
                "duration_seconds": meta.get("duration_seconds"),
                "original_sample_rate": original_sample_rate,
                "processing_sample_rate": rate,
                "original_channels": original_channels,
                "processed_channels": 1,
                "samples_processed": 0,
                "audio_extracted_from_video": is_extracted_from_video,
                "decoding_status": f"Failed: {exc}",
                "errors_or_warnings": [str(exc)],
            },
            "signal_quality": {
                "rms_energy": None,
                "peak_amplitude": None,
                "clipping_count": 0,
                "clipping_percentage": 0.0,
                "silence_detected": False,
                "low_energy_detected": False,
                "disclaimer": "Signal-quality measurements describe the audio recording characteristics. They are not direct indicators of whether audio is authentic or AI-generated.",
            },
            "model_info": _default_model_info(),
            "timing": {"processing_duration_seconds": duration_proc},
            "details": {"error": str(exc)},
        }
        _save_deepfake_result(evidence_id, result)
        return result

    if samples is None or samples.size == 0 or not np.all(np.isfinite(samples)):
        duration_proc = round(time.perf_counter() - start_time, 3)
        result = {
            "evidence_id": evidence_id,
            "status": "Failed",
            "result": "INCONCLUSIVE",
            "score": None,
            "model_name": AUDIO_SETTINGS.deepfake_model,
            "explanation": "No valid audio samples could be decoded from the evidence file.",
            "verdict": {
                "title": "INCONCLUSIVE — INSUFFICIENT EVIDENCE",
                "subtitle": "The available audio or model output does not support a reliable classification.",
                "classification": "INCONCLUSIVE",
                "conclusion": "TruthLens could not reach a confident classification using the available evidence. Additional or higher-quality audio may be required.",
                "summary": "Audio stream was empty or contained non-finite values.",
            },
            "scores": {
                "synthetic_score": None,
                "genuine_score": None,
                "threshold_detected": 0.70,
                "threshold_not_detected": 0.60,
                "score_margin": None,
                "margin_description": "Empty audio; score unavailable",
                "classification_rule": "Flagged if score >= 0.7000; Not flagged if score <= 0.6000; Inconclusive if 0.6000 < score < 0.7000",
                "interpretation": "This score represents the model's output and is not a calibrated probability of audio manipulation.",
            },
            "why_verdict": {
                "evaluated": "Decoded waveform validation check.",
                "score": None,
                "threshold": "0.6000 - 0.7000",
                "decision": "INCONCLUSIVE",
                "reason": "Decoded audio contained zero samples or invalid numerical values.",
                "what_it_means": "No valid acoustic data was available for neural feature extraction.",
                "what_it_does_not_prove": "- It does not prove the audio is genuine or fake.\n- Additional or higher-quality audio evidence is required.",
            },
            "audio_properties": {
                "original_filename": original_filename,
                "format": file_format,
                "audio_codec": audio_codec,
                "duration_seconds": 0.0,
                "original_sample_rate": original_sample_rate or rate,
                "processing_sample_rate": rate,
                "original_channels": original_channels or 1,
                "processed_channels": 1,
                "samples_processed": 0,
                "audio_extracted_from_video": is_extracted_from_video,
                "decoding_status": "Empty audio stream",
                "errors_or_warnings": ["Decoded audio contains 0 samples"],
            },
            "signal_quality": {
                "rms_energy": 0.0,
                "peak_amplitude": 0.0,
                "clipping_count": 0,
                "clipping_percentage": 0.0,
                "silence_detected": True,
                "low_energy_detected": True,
                "disclaimer": "Signal-quality measurements describe the audio recording characteristics. They are not direct indicators of whether audio is authentic or AI-generated.",
            },
            "model_info": _default_model_info(),
            "timing": {"processing_duration_seconds": duration_proc},
            "details": {},
        }
        _save_deepfake_result(evidence_id, result)
        return result

    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    mono = mono.reshape(-1).astype(np.float32)
    samples_count = int(mono.size)
    duration_s = round(float(samples_count / max(rate, 1)), 2)
    peak = round(float(np.max(np.abs(mono))), 4) if samples_count > 0 else 0.0
    rms = round(float(np.sqrt(np.mean(mono**2))), 4) if samples_count > 0 else 0.0
    std = round(float(np.std(mono)), 5) if samples_count > 0 else 0.0
    clipping_count = int(np.sum(np.abs(mono) >= 0.999))
    clipping_pct = round(float(clipping_count / max(samples_count, 1) * 100), 2)
    is_silent = bool(peak < 1e-4 or std < 1e-5)
    is_low_energy = bool(peak < 0.02 or rms < 0.005)

    if is_silent:
        duration_proc = round(time.perf_counter() - start_time, 3)
        result = {
            "evidence_id": evidence_id,
            "status": "Completed",
            "result": "INCONCLUSIVE",
            "score": None,
            "model_name": AUDIO_SETTINGS.deepfake_model,
            "explanation": "Audio contains near-zero signal energy or is primarily silence, preventing reliable spoof assessment.",
            "verdict": {
                "title": "INCONCLUSIVE — INSUFFICIENT EVIDENCE",
                "subtitle": "The available audio or model output does not support a reliable classification.",
                "classification": "INCONCLUSIVE",
                "conclusion": "TruthLens could not reach a confident classification using the available evidence. Additional or higher-quality audio may be required.",
                "summary": "The audio waveform is silent or contains near-zero signal energy (peak amplitude < 0.0001).",
            },
            "scores": {
                "synthetic_score": None,
                "genuine_score": None,
                "threshold_detected": 0.70,
                "threshold_not_detected": 0.60,
                "score_margin": None,
                "margin_description": "Silent audio; score omitted to avoid false classification",
                "classification_rule": "Flagged if score >= 0.7000; Not flagged if score <= 0.6000; Inconclusive if 0.6000 < score < 0.7000",
                "interpretation": "This score represents the model's output and is not a calibrated probability of audio manipulation.",
            },
            "why_verdict": {
                "evaluated": "Physical signal energy and peak amplitude checks.",
                "score": None,
                "threshold": "0.6000 - 0.7000",
                "decision": "INCONCLUSIVE",
                "reason": f"The audio amplitude is near zero (peak: {peak:.5f}, RMS: {rms:.5f}). Intelligible speech is required for Wav2Vec2 feature extraction.",
                "what_it_means": "No classification is possible on silence without generating random or misleading scores.",
                "what_it_does_not_prove": "- It does not prove the audio is genuine or fake.\n- Additional or higher-quality audio evidence is required.",
            },
            "audio_properties": {
                "original_filename": original_filename,
                "format": file_format,
                "audio_codec": audio_codec,
                "duration_seconds": duration_s,
                "original_sample_rate": original_sample_rate or rate,
                "processing_sample_rate": 16000,
                "original_channels": original_channels or 1,
                "processed_channels": 1,
                "samples_processed": samples_count,
                "audio_extracted_from_video": is_extracted_from_video,
                "decoding_status": f"Successful ({decoding_method})",
                "errors_or_warnings": ["Audio is silent or near-zero energy"],
            },
            "signal_quality": {
                "rms_energy": rms,
                "peak_amplitude": peak,
                "clipping_count": clipping_count,
                "clipping_percentage": clipping_pct,
                "silence_detected": True,
                "low_energy_detected": True,
                "disclaimer": "Signal-quality measurements describe the audio recording characteristics. They are not direct indicators of whether audio is authentic or AI-generated.",
            },
            "model_info": _default_model_info(),
            "timing": {"processing_duration_seconds": duration_proc},
            "details": {"peak_amplitude": peak, "standard_deviation": std, "rms_energy": rms},
        }
        _save_deepfake_result(evidence_id, result)
        return result

    # Segment waveform if audio exceeds 15.0 seconds
    use_segmentation = duration_s > 15.0
    seg_duration = 15.0
    overlap_duration = 1.0

    raw_segments = _segment_waveform(mono, rate, seg_duration=seg_duration, overlap=overlap_duration)
    segment_records: list[dict] = []
    
    last_detected = None
    all_raw_logits = []

    for seg_meta in raw_segments:
        s_idx = seg_meta["start_sample"]
        e_idx = seg_meta["end_sample"]
        seg_samples = mono[s_idx:e_idx]
        seg_t0 = time.perf_counter()

        # Skip silence-dominated segments to avoid spurious Wav2Vec2 artifact scores
        seg_rms = float(np.sqrt(np.mean(seg_samples ** 2))) if seg_samples.size > 0 else 0.0
        if seg_rms < 0.01:
            segment_records.append({
                "segment_index": seg_meta["segment_index"],
                "start_time": seg_meta["start_time"],
                "end_time": seg_meta["end_time"],
                "duration": seg_meta["duration"],
                "synthetic_score": None,
                "logit_margin": None,
                "raw_logits": None,
                "classification": "SKIPPED",
                "processing_duration": 0.0,
                "status": "Skipped",
                "error": "Silence: RMS below minimum threshold (0.01)",
            })
            continue

        seg_record = {
            "segment_index": seg_meta["segment_index"],
            "start_time": seg_meta["start_time"],
            "end_time": seg_meta["end_time"],
            "duration": seg_meta["duration"],
            "synthetic_score": None,
            "logit_margin": None,
            "raw_logits": None,
            "classification": "INCONCLUSIVE",
            "processing_duration": 0.0,
            "status": "Failed",
            "error": None,
        }

        try:
            seg_det = detect_synthetic_audio(seg_samples, rate)
            last_detected = seg_det
            seg_proc = round(time.perf_counter() - seg_t0, 3)
            seg_record["processing_duration"] = seg_proc

            if seg_det.get("status") == "completed":
                c_scores = seg_det.get("class_scores") or {}
                f_score = next((float(v) for k, v in c_scores.items() if any(t in str(k).lower() for t in ("spoof", "fake", "synthetic"))), None)
                r_score = next((float(v) for k, v in c_scores.items() if any(t in str(k).lower() for t in ("bonafide", "real", "authentic"))), None)
                if f_score is None and r_score is not None:
                    f_score = 1.0 - r_score

                # Extract raw logits and compute logit margin (logit[1] - logit[0] = fake - real)
                seg_logits = seg_det.get("raw_logits")
                if seg_logits and len(seg_logits) >= 2:
                    seg_logit_margin = round(float(seg_logits[1]) - float(seg_logits[0]), 4)
                    seg_record["logit_margin"] = seg_logit_margin
                    seg_record["raw_logits"] = seg_logits
                    all_raw_logits.append(seg_logits)

                if f_score is not None:
                    seg_record["synthetic_score"] = round(f_score, 4)
                    seg_record["status"] = "Completed"
                    # Segment-level classification uses the new provisional policy
                    if f_score >= 0.70:
                        seg_record["classification"] = "SYNTHETIC AUDIO INDICATORS DETECTED"
                    elif f_score <= 0.60:
                        seg_record["classification"] = "NO STRONG SYNTHETIC AUDIO INDICATORS DETECTED"
                    else:
                        seg_record["classification"] = "INCONCLUSIVE"
                else:
                    seg_record["status"] = "Failed"
                    seg_record["error"] = "Class scores unavailable"
            else:
                seg_record["status"] = "Failed"
                seg_record["error"] = seg_det.get("error") or "Segment inference non-completed"
        except Exception as exc:
            seg_record["processing_duration"] = round(time.perf_counter() - seg_t0, 3)
            seg_record["status"] = "Failed"
            seg_record["error"] = str(exc)

        segment_records.append(seg_record)

    # Evaluate Segmentation & Aggregation
    total_segments = len(segment_records)
    successful_segments = [s for s in segment_records if s["status"] == "Completed" and s["synthetic_score"] is not None]
    skipped_segments = [s for s in segment_records if s["status"] == "Skipped"]
    failed_segments = [s for s in segment_records if s["status"] == "Failed"]

    total_analyzed_duration = round(sum(s["duration"] for s in successful_segments), 2)
    audio_coverage_pct = round((total_analyzed_duration / max(duration_s, 0.01)) * 100, 2)
    # Clamp coverage to max 100.0% (due to overlap, sum of segment durations can exceed total duration)
    coverage_reported = min(100.0, audio_coverage_pct)

    if successful_segments:
        weights = [s["duration"] for s in successful_segments]
        scores_arr = [s["synthetic_score"] for s in successful_segments]
        aggregate_synth_score = float(np.average(scores_arr, weights=weights))
        min_seg_score = float(np.min(scores_arr))
        max_seg_score = float(np.max(scores_arr))
        mean_seg_score = float(np.mean(scores_arr))
        duration_weighted_score = aggregate_synth_score
        score_range = round(max_seg_score - min_seg_score, 4)
        high_disagreement = score_range > 0.35
        pct_strong_indicators = round(
            len([s for s in successful_segments if (s["synthetic_score"] or 0) >= 0.70]) /
            max(len(successful_segments), 1) * 100, 1
        )
        num_strong_segments = len([s for s in successful_segments if (s["synthetic_score"] or 0) >= 0.70])
        # Compute duration-weighted average logit margin across segments that have it
        seg_logit_margins = [s["logit_margin"] for s in successful_segments if s.get("logit_margin") is not None]
        if seg_logit_margins:
            margin_weights = [s["duration"] for s in successful_segments if s.get("logit_margin") is not None]
            avg_logit_margin = round(float(np.average(seg_logit_margins, weights=margin_weights)), 4)
        else:
            avg_logit_margin = None
    else:
        aggregate_synth_score = None
        min_seg_score = None
        max_seg_score = None
        mean_seg_score = None
        duration_weighted_score = None
        score_range = 0.0
        high_disagreement = False
        pct_strong_indicators = 0.0
        num_strong_segments = 0
        avg_logit_margin = None

    seg_verdicts = [s["classification"] for s in segment_records]
    failed_errors = [f"Segment {s['segment_index']}: {s['error']}" for s in failed_segments if s.get("error")]

    # Check if analysis coverage is sufficient (at least 50% coverage and at least 1 successful segment)
    is_coverage_sufficient = (len(successful_segments) > 0) and (coverage_reported >= 50.0)

    # -------------------------------------------------------------------------
    # PROVISIONAL DECISION ENGINE (v2)
    # Policy:
    #   - Synthetic score <= 0.60 -> NO STRONG SYNTHETIC INDICATORS DETECTED
    #     (unless genuine data-integrity issue, insufficient coverage, severe quality degradation,
    #      or contradictory segment evidence justifies INCONCLUSIVE)
    #   - Synthetic score >= 0.70 AND supporting model evidence -> SYNTHETIC AUDIO INDICATORS DETECTED
    #   - Synthetic score > 0.60 and < 0.70 -> INCONCLUSIVE (indeterminate boundary zone)
    #
    # Logit margin is used as supporting diagnostic context and conflict check,
    # not as a second blocking hurdle that prevents non-synthetic verdicts.
    # -------------------------------------------------------------------------
    DETECT_SCORE_MIN = 0.70
    CLEAR_SCORE_MAX = 0.60

    if not is_coverage_sufficient or aggregate_synth_score is None:
        outcome = "INCONCLUSIVE"
        verdict_title = "INCONCLUSIVE — INSUFFICIENT EVIDENCE"
        verdict_subtitle = "The available audio or model output does not support a reliable classification."
        verdict_conclusion = "TruthLens could not reach a confident classification using the available evidence. Additional or higher-quality audio may be required."
        margin = None
        margin_desc = "Insufficient coverage or inference failure"
        err_detail = f": {failed_errors[0]}" if failed_errors else ""
        verdict_summary = f"Analysis produced insufficient reliable audio coverage ({coverage_reported}% analyzed, {len(failed_segments)} failed segments){err_detail}."
        why_reason = f"Only {len(successful_segments)} of {total_segments} audio segments could be analyzed ({coverage_reported}% coverage){err_detail}. Decision rules require sufficient coverage across the recording."
        what_it_means = "The model could not extract sufficient acoustic representations across the full sample."
        what_it_does_not_prove = "- It does not prove the audio is genuine or fake.\n- Additional or higher-quality audio evidence is required."
        explanation = verdict_summary
        final_score = None
        real_score = None
        logit_margin_display = None
        segment_consistency_desc = "Insufficient data"
    else:
        final_score = round(aggregate_synth_score, 4)
        real_score = round(1.0 - final_score, 4)
        logit_margin_display = avg_logit_margin

        # Segment consistency descriptor
        if score_range <= 0.15:
            segment_consistency_desc = f"Consistent (range: {score_range:.3f})"
        elif score_range <= 0.35:
            segment_consistency_desc = f"Moderate variation (range: {score_range:.3f})"
        else:
            segment_consistency_desc = f"High disagreement across segments (range: {score_range:.3f})"

        # Check for severe quality degradation (e.g. clipping >= 25%)
        is_severe_quality_degradation = bool(clipping_pct >= 25.0)

        # ---------------------------------------------------------------------
        # Evaluation by Score Boundaries
        # ---------------------------------------------------------------------
        if final_score <= CLEAR_SCORE_MAX:
            # Score <= 0.60 -> Negative verdict unless safety exception
            if is_severe_quality_degradation:
                outcome = "INCONCLUSIVE"
                verdict_title = "INCONCLUSIVE — SEVERE QUALITY DEGRADATION"
                verdict_subtitle = "Audio signal degradation prevents a reliable authenticity assessment."
                verdict_conclusion = "TruthLens could not reach a confident classification due to severe signal degradation in the recording."
                margin = round(0.60 - final_score, 4)
                margin_desc = f"score {final_score:.4f} <= 0.6000 but severe clipping ({clipping_pct:.1f}%)"
                verdict_summary = f"While the model score is {final_score:.4f}, excessive audio clipping ({clipping_pct:.1f}%) impairs reliable classification."
                why_reason = f"The recording exhibits severe clipping ({clipping_pct:.1f}% of samples clipped at maximum amplitude). Audio signal saturation distorts spectral characteristics and prevents dependable spoof detection."
                what_it_means = "Recording artifacts and non-linear saturation prevent accurate neural acoustic evaluation."
                what_it_does_not_prove = "- It does not prove the audio is genuine or fake.\n- A clean unclipped recording is required for verification."
                explanation = f"INCONCLUSIVE — score {final_score:.4f} but severe clipping degradation ({clipping_pct:.1f}%)."
            elif use_segmentation and (num_strong_segments >= 2 or (pct_strong_indicators >= 30.0 and len(successful_segments) >= 3)):
                # Contradictory evidence: substantial segments show strong synthetic indicators >= 0.70
                outcome = "INCONCLUSIVE"
                verdict_title = "INCONCLUSIVE — CONTRADICTORY SEGMENT EVIDENCE"
                verdict_subtitle = "Segment-level analysis produced contradictory acoustic evidence."
                verdict_conclusion = "TruthLens cannot reach a confident classification because different segments of the recording produced contradictory results."
                margin = round(0.60 - final_score, 4)
                margin_desc = f"contradictory segment evidence ({num_strong_segments} segments >= 0.7000)"
                verdict_summary = (
                    f"While the duration-weighted aggregate score is {final_score:.4f} (<= 0.6000), "
                    f"{num_strong_segments} of {len(successful_segments)} analyzed segments ({pct_strong_indicators:.1f}%) "
                    f"exhibited strong synthetic indicators (>= 0.7000)."
                )
                why_reason = (
                    f"The recording was divided into {total_segments} overlapping segments. "
                    f"Although the duration-weighted average is {final_score:.4f}, {num_strong_segments} segments "
                    f"exhibited synthetic scores >= 0.7000 (score range: {score_range:.3f}). "
                    f"This substantial internal contradiction between segments justifies an INCONCLUSIVE verdict."
                )
                what_it_means = "Different portions of the recording exhibit divergent acoustic properties. This may indicate partial splicing, localized voice replacement, or varying acoustic environments."
                what_it_does_not_prove = "- It does not confirm or rule out manipulation.\n- Segment-by-segment forensic review is advised."
                explanation = f"INCONCLUSIVE — contradictory segment evidence ({num_strong_segments} segments >= 0.7000, aggregate: {final_score:.4f})."
            elif use_segmentation and score_range >= 0.50 and max_seg_score >= 0.70 and min_seg_score <= 0.25:
                # Extreme segment conflict
                outcome = "INCONCLUSIVE"
                verdict_title = "INCONCLUSIVE — CONTRADICTORY SEGMENT EVIDENCE"
                verdict_subtitle = "Extreme score divergence across segments prevents a definitive verdict."
                verdict_conclusion = "TruthLens cannot reach a confident classification due to extreme acoustic divergence between analyzed segments."
                margin = round(0.60 - final_score, 4)
                margin_desc = f"extreme segment divergence (range: {score_range:.4f})"
                verdict_summary = f"Segment scores range from {min_seg_score:.4f} to {max_seg_score:.4f} (range: {score_range:.4f}). Contradictory evidence across segments prevents classification."
                why_reason = (
                    f"Segment scores exhibit extreme divergence (min: {min_seg_score:.4f}, max: {max_seg_score:.4f}, range: {score_range:.4f}). "
                    f"Because individual segments strongly disagree, the aggregate score cannot be reliably interpreted as a uniform classification."
                )
                what_it_means = "The recording contains highly disparate acoustic segments that cannot be reconciled into a single confident verdict."
                what_it_does_not_prove = "- It does not prove the audio is genuine or fake.\n- Localized segment analysis is recommended."
                explanation = f"INCONCLUSIVE — extreme segment disagreement (range: {score_range:.4f})."
            else:
                # Confirmed negative verdict (Score <= 0.60)
                outcome = "DEEPFAKE NOT DETECTED"
                verdict_title = "NO STRONG SYNTHETIC AUDIO INDICATORS DETECTED"
                verdict_subtitle = "The model did not identify sufficient acoustic patterns associated with synthetic or spoofed speech."
                verdict_conclusion = "No strong synthetic-audio indicators were detected by this model during this analysis. This does not establish that the audio is authentic or free from manipulation."
                margin = round(0.60 - final_score, 4)
                margin_desc = f"-{margin:.4f} below threshold (0.6000)"
                verdict_summary = (
                    f"The model produced a synthetic-audio score of {final_score:.4f}, which is below the "
                    f"detection threshold of 0.6000 by {margin:.4f}."
                )
                if use_segmentation:
                    if num_strong_segments == 1 and len(successful_segments) > 1:
                        isolated_seg = next(s for s in successful_segments if (s["synthetic_score"] or 0) >= 0.70)
                        why_reason = (
                            f"The recording was divided into {total_segments} overlapping segments. "
                            f"{len(successful_segments)} segments were successfully analyzed with a duration-weighted aggregate score of {final_score:.4f}. "
                            f"The score is below the negative decision boundary of 0.6000 by {margin:.4f}. "
                            f"Note: One isolated segment (Segment {isolated_seg['segment_index']}: score {isolated_seg['synthetic_score']:.4f}) "
                            f"showed elevated indicators, but the majority of segments ({len(successful_segments) - 1}/{len(successful_segments)}) "
                            f"and the overall weighted aggregate remain below the detection threshold."
                        )
                    else:
                        why_reason = (
                            f"The recording was divided into {total_segments} overlapping segments. "
                            f"{len(successful_segments)} segments were successfully analyzed. "
                            f"Duration-weighted aggregation produced an overall synthetic score of {final_score:.4f}, "
                            f"which is below the negative decision boundary of 0.6000 by {margin:.4f}."
                        )
                else:
                    why_reason = (
                        f"The model returned a synthetic-audio score of {final_score:.4f}. "
                        f"This score is below the configured negative detection threshold of 0.6000 by {margin:.4f}. "
                        f"Under current decision rules, no strong synthetic audio indicators were detected."
                    )
                what_it_means = "The classifier did not detect strong synthetic or spoofed speech acoustic patterns in this recording."
                what_it_does_not_prove = (
                    "- It does not establish that the audio is authentic or free from manipulation.\n"
                    "- It does not verify speaker identity or recording authenticity.\n"
                    "- Advanced, unseen, or hybrid synthetic speech methods may not be detected by this model."
                )
                explanation = f"No strong synthetic audio indicators detected (score: {final_score:.4f}, threshold: 0.6000)."

        elif final_score >= DETECT_SCORE_MIN:
            # Score >= 0.70: check supporting model evidence
            if is_severe_quality_degradation:
                outcome = "INCONCLUSIVE"
                verdict_title = "INCONCLUSIVE — SEVERE QUALITY DEGRADATION"
                verdict_subtitle = "Severe signal clipping prevents reliable confirmation of synthetic indicators."
                verdict_conclusion = "TruthLens cannot confirm synthetic indicators due to severe clipping distortion in the audio recording."
                margin = round(final_score - 0.70, 4)
                margin_desc = f"score {final_score:.4f} >= 0.7000 but severe clipping ({clipping_pct:.1f}%)"
                verdict_summary = f"While the model score is {final_score:.4f}, excessive audio clipping ({clipping_pct:.1f}%) impairs reliable classification."
                why_reason = f"The model returned an elevated score ({final_score:.4f}), but severe clipping ({clipping_pct:.1f}% samples clipped) introduces harmonic distortion that can mimic synthetic artifacts. A clean unclipped sample is required."
                what_it_means = "Clipping saturation artifacts can cause false neural feature activations."
                what_it_does_not_prove = "- It does not confirm that the audio is synthetic.\n- A clean recording without digital clipping is required."
                explanation = f"INCONCLUSIVE — elevated score ({final_score:.4f}) but severe clipping degradation ({clipping_pct:.1f}%)."
            elif use_segmentation and len(successful_segments) >= 3 and pct_strong_indicators < 25.0:
                # Contradictory: overall pulled up but less than 25% of segments support synthetic
                outcome = "INCONCLUSIVE"
                verdict_title = "INCONCLUSIVE — CONTRADICTORY SEGMENT EVIDENCE"
                verdict_subtitle = "Segment distribution does not adequately support an overall synthetic classification."
                verdict_conclusion = "TruthLens cannot confirm synthetic indicators because fewer than 25% of segments exhibit strong synthetic scores."
                margin = round(final_score - 0.70, 4)
                margin_desc = f"contradictory segment distribution ({pct_strong_indicators:.1f}% strong segments)"
                verdict_summary = f"Aggregate score is {final_score:.4f}, but only {pct_strong_indicators:.1f}% of segments exhibit strong synthetic indicators."
                why_reason = (
                    f"Although the aggregate score reached {final_score:.4f}, only {pct_strong_indicators:.1f}% of segments "
                    f"showed strong synthetic indicators (>= 0.7000). The majority of segments do not corroborate synthetic origin."
                )
                what_it_means = "The elevated aggregate score is driven by a localized outlier segment rather than consistent synthetic speech patterns."
                what_it_does_not_prove = "- It does not prove the recording is synthetic.\n- Localized segment review is required."
                explanation = f"INCONCLUSIVE — aggregate {final_score:.4f} but insufficient segment corroboration ({pct_strong_indicators:.1f}% strong segments)."
            else:
                # Confirmed detection (Score >= 0.70 with supporting evidence)
                outcome = "DEEPFAKE DETECTED"
                verdict_title = "SYNTHETIC AUDIO INDICATORS DETECTED"
                verdict_subtitle = "The model identified audio patterns associated with synthetic or spoofed speech."
                verdict_conclusion = "The analyzed audio produced a synthetic-audio score above the configured detection threshold. The sample has been flagged for further verification."
                margin = round(final_score - 0.70, 4)
                margin_desc = f"+{margin:.4f} above threshold (0.7000)"
                verdict_summary = (
                    f"The model produced a synthetic-audio score of {final_score:.4f}. This exceeds the "
                    f"detection threshold of 0.7000 by {margin:.4f}, so the sample was flagged for further verification."
                )
                if use_segmentation:
                    if pct_strong_indicators >= 80.0:
                        consistency_note = "Strong synthetic indicators were consistently observed across all analyzed segments."
                    else:
                        consistency_note = f"{pct_strong_indicators:.1f}% of analyzed segments exhibited strong synthetic indicators."
                    why_reason = (
                        f"The recording was divided into {total_segments} overlapping segments. "
                        f"{len(successful_segments)} segments were successfully analyzed. "
                        f"Their synthetic-audio scores were combined using duration-weighted aggregation to produce an overall score of {final_score:.4f}. "
                        f"The aggregate score exceeds the detection threshold of 0.7000 by {margin:.4f}. {consistency_note}"
                    )
                else:
                    why_reason = (
                        f"The model returned a synthetic-audio score of {final_score:.4f}. "
                        f"The configured detection threshold is 0.7000. The score exceeds the threshold by {margin:.4f}. "
                        f"Under the current decision rules, this sample was flagged for synthetic-audio indicators."
                    )
                what_it_means = "The classifier identified patterns that its trained model associates with synthetic or spoofed audio."
                what_it_does_not_prove = (
                    "- It does not independently prove that the audio was AI-generated.\n"
                    "- It does not identify the tool used to generate the audio.\n"
                    "- It does not establish the identity of the speaker.\n"
                    "- It does not prove malicious intent.\n"
                    "- It does not establish when or how the audio was manipulated."
                )
                explanation = f"Synthetic audio indicators detected with a model score of {final_score:.4f} (threshold: 0.7000)."

        else:
            # Score is between 0.60 and 0.70: INCONCLUSIVE (Borderline zone)
            outcome = "INCONCLUSIVE"
            verdict_title = "INCONCLUSIVE — BORDERLINE ACOUSTIC INDICATORS"
            verdict_subtitle = "The acoustic evidence falls in the borderline indeterminate zone (0.6000 – 0.7000) and does not support a confident classification."
            verdict_conclusion = "TruthLens cannot reach a confident classification based on the current acoustic evidence. The model output falls within the ambiguous boundary region between authentic and synthetic speech."
            margin = round(abs(final_score - 0.65), 4)
            margin_desc = f"within indeterminate zone [0.6000, 0.7000] (offset from midpoint: {margin:.4f})"
            if logit_margin_display is not None:
                margin_desc += f" · logit margin: {logit_margin_display:+.4f}"
            verdict_summary = (
                f"The model produced a synthetic-audio score of {final_score:.4f}, which falls within the "
                f"indeterminate boundary zone (0.6000 – 0.7000)."
            )
            if use_segmentation:
                why_reason = (
                    f"The recording was divided into {total_segments} overlapping segments. "
                    f"{len(successful_segments)} segments were successfully analyzed with duration-weighted aggregate score {final_score:.4f}. "
                    f"This score falls within the indeterminate zone (0.6000 to 0.7000). Neither authentic baseline nor synthetic indicators are sufficiently decisive."
                )
            else:
                why_reason = (
                    f"The model returned a synthetic-audio score of {final_score:.4f}, which falls within the "
                    f"indeterminate zone (0.6000 to 0.7000). Neither authentic baseline nor synthetic indicators are sufficiently decisive for classification."
                )
            what_it_means = "The model output does not provide statistical separation sufficient for classification."
            what_it_does_not_prove = "- It does not prove the audio is genuine or fake.\n- Additional or higher-quality audio evidence is required."
            explanation = f"Model score ({final_score:.4f}) falls within the indeterminate zone (0.6000 – 0.7000); insufficient evidence for classification."

    duration_proc = round(time.perf_counter() - start_time, 3)

    segmentation_payload = {
        "segmentation_used": use_segmentation,
        "segment_duration_seconds": seg_duration,
        "overlap_seconds": overlap_duration,
        "total_segments": total_segments,
        "successful_segments": len(successful_segments),
        "skipped_segments": len(skipped_segments),
        "failed_segments": len(failed_segments),
        "total_audio_duration": duration_s,
        "total_analyzed_duration": total_analyzed_duration,
        "audio_coverage_percentage": coverage_reported,
        "aggregation_method": "duration_weighted",
        "aggregate_score": round(aggregate_synth_score, 4) if aggregate_synth_score is not None else None,
        "min_segment_score": round(min_seg_score, 4) if min_seg_score is not None else None,
        "max_segment_score": round(max_seg_score, 4) if max_seg_score is not None else None,
        "mean_segment_score": round(mean_seg_score, 4) if mean_seg_score is not None else None,
        "duration_weighted_score": round(duration_weighted_score, 4) if duration_weighted_score is not None else None,
        "segment_score_distribution": [s["synthetic_score"] for s in successful_segments],
        "segment_verdicts": seg_verdicts,
        "failed_segment_errors": failed_errors,
        "pct_segments_strong_indicators": pct_strong_indicators if successful_segments else 0.0,
        "high_segment_disagreement": high_disagreement,
        "segment_score_range": round(score_range, 4) if successful_segments else None,
        "avg_logit_margin": avg_logit_margin,
        "segment_results": segment_records,
    }

    # Compute score band for the final score
    score_band = _classify_score_band(final_score)

    result = {
        "evidence_id": evidence_id,
        "status": "Completed" if is_coverage_sufficient else "Failed",
        "result": outcome,
        "score": final_score,
        "model_name": (last_detected.get("detector_name") if last_detected else None) or AUDIO_SETTINGS.deepfake_model,
        "explanation": explanation,
        "score_band": score_band,
        "verdict": {
            "title": verdict_title,
            "subtitle": verdict_subtitle,
            "classification": outcome,
            "conclusion": verdict_conclusion,
            "summary": verdict_summary,
        },
        "scores": {
            "synthetic_score": final_score,
            "genuine_score": real_score,
            "threshold_detected": 0.70,
            "threshold_not_detected": 0.60,
            "score_margin": margin,
            "margin_description": margin_desc,
            "logit_margin": logit_margin_display,
            "classification_rule": (
                "NO STRONG SYNTHETIC INDICATORS DETECTED if score <= 0.6000; "
                "SYNTHETIC AUDIO INDICATORS DETECTED if score >= 0.7000 with supporting model evidence; "
                "INCONCLUSIVE if 0.6000 < score < 0.7000 or contradictory evidence"
            ),
            "interpretation": "This score represents the model's output (or duration-weighted aggregate across segments) and is not a calibrated probability of audio manipulation.",
        },
        "why_verdict": {
            "evaluated": f"Wav2Vec2 evaluated 16 kHz acoustic feature representations from {total_segments} audio segment(s)." if use_segmentation else "Wav2Vec2 evaluated 16 kHz acoustic feature representations from the audio waveform.",
            "score": final_score,
            "logit_margin": logit_margin_display,
            "threshold": 0.70 if outcome == "DEEPFAKE DETECTED" else (0.60 if outcome == "DEEPFAKE NOT DETECTED" else "0.6000 – 0.7000 (indeterminate zone)"),
            "decision": outcome,
            "reason": why_reason,
            "segment_consistency": segment_consistency_desc if successful_segments else "No segments analyzed",
            "calibration_status": "not_calibrated — ASVspoof 2021 PA domain; scores are acoustic feature activations, not calibrated probabilities",
            "what_it_means": what_it_means,
            "what_it_does_not_prove": what_it_does_not_prove,
        },
        "audio_properties": {
            "original_filename": original_filename,
            "format": file_format,
            "audio_codec": audio_codec,
            "duration_seconds": duration_s,
            "original_sample_rate": original_sample_rate or rate,
            "processing_sample_rate": 16000,
            "original_channels": original_channels or 1,
            "processed_channels": 1,
            "samples_processed": samples_count,
            "audio_extracted_from_video": is_extracted_from_video,
            "decoding_status": f"Successful ({decoding_method})",
            "errors_or_warnings": audio_warnings + failed_errors,
        },
        "signal_quality": {
            "rms_energy": rms,
            "peak_amplitude": peak,
            "clipping_count": clipping_count,
            "clipping_percentage": clipping_pct,
            "silence_detected": is_silent,
            "low_energy_detected": is_low_energy,
            "disclaimer": "Signal-quality measurements describe the audio recording characteristics. They are not direct indicators of whether audio is authentic or AI-generated.",
        },
        "segmentation": segmentation_payload,
        "model_info": _default_model_info(last_detected),
        "timing": {
            "processing_duration_seconds": duration_proc,
        },
        "details": {
            "class_scores": {"real": real_score, "fake": final_score} if final_score is not None else {},
            "prediction": outcome,
            "raw_logits": all_raw_logits[0] if (all_raw_logits and len(all_raw_logits) == 1) else all_raw_logits,
            "class_mapping": {"0": "real", "1": "fake"},
            "decision_thresholds": {
                "deepfake_detected_min": 0.70,
                "deepfake_not_detected_max": 0.60,
                "inconclusive_zone": [0.60, 0.70],
            },
            "limitations": (last_detected.get("limitations") if last_detected else []) or [
                "Operational duration-weighted aggregation across segments is a heuristic evaluation, not a joint probability.",
            ],
            "aggregation_strategy": "duration-weighted aggregation of individual segment classification outputs",
            "calibration_status": "not_calibrated — ASVspoof 2021 PA domain",
        },
    }
    _save_deepfake_result(evidence_id, result)
    return result


@app.get("/api/v1/analyzers/audio/{evidence_id}/deepfake")
def get_audio_deepfake_analysis(evidence_id: str) -> dict:
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    record = ingest_service.get_evidence(evidence_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    saved = _load_deepfake_result(evidence_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="No saved deepfake analysis for this evidence")
    return saved


@app.get("/api/v1/evidence/{evidence_id}/content")
def evidence_content(evidence_id: str):
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    record = ingest_service.get_evidence(evidence_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    has_audio = (
        record.get("file_type") == "audio"
        or (record.get("metadata") or {}).get("audio_stream_count", 0) > 0
        or (record.get("metadata") or {}).get("audio_streams_present") is True
        or (record.get("mime_type") or "").startswith("audio/")
        or Path(str(record.get("file_path", ""))).suffix.lower() in {".wav", ".mp3", ".m4a"}
    )
    if not has_audio and record.get("file_type") != "video":
        raise HTTPException(status_code=422, detail="Audio playback is available for audio evidence only.")
    path = Path(str(record.get("file_path", ""))).resolve()
    try:
        path.relative_to(ingest_service.storage_dir.resolve())
    except ValueError:
        try:
            path.relative_to(ingest_service.evidence_dir.resolve())
        except ValueError:
            try:
                path.relative_to(ingest_service.data_dir.resolve())
            except ValueError as exc:
                raise HTTPException(status_code=404, detail="Evidence content not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Evidence content not found")
    return FileResponse(path, media_type=record.get("mime_type") or "application/octet-stream",
                        filename=record.get("original_filename") or path.name)


@app.get("/api/v1/analyzers/audio/{evidence_id}/artifacts/{run_id}/{filename}")
def audio_artifact(evidence_id: str, run_id: str, filename: str):
    if not re.fullmatch(r"EVD-\d+", evidence_id) or not re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{8}", run_id):
        raise HTTPException(status_code=404, detail="Artifact not found")
    safe_name = Path(filename).name
    for base in (Path(ingest_service.storage_dir), Path(ingest_service.data_dir)):
        root = (base / "artifacts" / evidence_id / "audio" / run_id).resolve()
        candidate = (root / safe_name).resolve()
        try:
            candidate.relative_to(root)
            if candidate.is_file():
                return FileResponse(candidate)
        except ValueError:
            continue
    raise HTTPException(status_code=404, detail="Artifact not found")


def _latest_audio_result(evidence_id: str) -> dict:
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Analysis not found")
    for base in (Path(ingest_service.storage_dir), Path(ingest_service.data_dir)):
        latest = (base / "artifacts" / evidence_id / "audio" / "latest.json").resolve()
        root = (base / "artifacts" / evidence_id / "audio").resolve()
        if latest.is_file():
            try:
                latest.relative_to(root)
                pointer = json.loads(latest.read_text(encoding="utf-8"))
                result_path = Path(pointer["result_path"]).resolve()
                if result_path.is_file():
                    return json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                continue
    # Fallback to _load_deepfake_result
    cached = _load_deepfake_result(evidence_id)
    if cached:
        return cached
    raise HTTPException(status_code=404, detail="No saved audio analysis is available")


@app.get("/api/v1/analyzers/audio/{evidence_id}")
def get_audio_analysis(evidence_id: str) -> dict:
    return _latest_audio_result(evidence_id)


@app.get("/api/v1/analyzers/audio/{evidence_id}/transcription")
def get_audio_transcription(evidence_id: str) -> dict:
    return _latest_audio_result(evidence_id).get("transcription", {})


@app.get("/api/v1/analyzers/audio/{evidence_id}/timeline")
def get_audio_timeline(evidence_id: str) -> dict:
    result = _latest_audio_result(evidence_id)
    return {"evidence_id": evidence_id, "timeline_segments": result.get("timeline_segments", [])}


@app.get("/api/v1/analyzers/audio/{evidence_id}/modules")
def get_audio_modules(evidence_id: str) -> dict:
    result = _latest_audio_result(evidence_id)
    return {"evidence_id": evidence_id, "modules": result.get("modules", {})}


@app.get("/api/v1/ingest/status")
def ingest_status() -> dict:
    evidence = ingest_service.list_evidence()
    activity = ingest_service.list_activity(500)
    return {
        "total_uploaded": len(evidence) + sum(item.get("status") == "DUPLICATE" for item in activity),
        "successfully_ingested": len(evidence),
        "failed": sum(item.get("status") == "FAILED" for item in activity),
        "duplicates": sum(item.get("status") == "DUPLICATE" for item in activity),
        "currently_processing": 0,
    }


@app.get("/api/v1/ingest/activity")
def ingest_activity(limit: int = 100) -> dict:
    return {"activity": ingest_service.list_activity(limit)}


# ── Image Analyzer ────────────────────────────────────────────────────────────

def _save_analysis_result(evidence_id: str, modality: str, result: dict) -> None:
    # Save to service storage_dir/analysis/{modality}/
    mod = "audio" if modality in ("audio", "deepfake") else modality
    try:
        central_path = Path(ingest_service.storage_dir) / "analysis" / mod / f"{evidence_id}.json"
        central_path.parent.mkdir(parents=True, exist_ok=True)
        central_path.write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
    except Exception:
        pass
    # Legacy data_dir fallback
    try:
        out = Path(ingest_service.data_dir) / modality
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{evidence_id}.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
    except Exception:
        pass


def _load_analysis_result(evidence_id: str, modality: str) -> dict | None:
    # Check service storage_dir first
    mod = "audio" if modality in ("audio", "deepfake") else modality
    try:
        central_path = Path(ingest_service.storage_dir) / "analysis" / mod / f"{evidence_id}.json"
        if central_path.is_file():
            return json.loads(central_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    # Legacy fallback
    path = Path(ingest_service.data_dir) / modality / f"{evidence_id}.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


@app.post("/api/v1/analyzers/image/{evidence_id}")
def analyze_image(evidence_id: str) -> dict:
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    record = ingest_service.get_evidence(evidence_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    if record.get("file_type") != "image":
        raise HTTPException(status_code=422, detail="Image Analyzer accepts image evidence only.")
    try:
        contract = ingest_service.build_analyzer_input(evidence_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    from app.analyzers.image_analyzer import analyze_image_evidence
    result = analyze_image_evidence(contract)
    _save_analysis_result(evidence_id, "image", result)
    return result


@app.get("/api/v1/analyzers/image/{evidence_id}")
def get_image_result(evidence_id: str) -> dict:
    result = _load_analysis_result(evidence_id, "image")
    if result is None:
        raise HTTPException(status_code=404, detail="No image analysis result found.")
    return result


# ── Text Analyzer ─────────────────────────────────────────────────────────────

@app.post("/api/v1/analyzers/text/{evidence_id}")
def analyze_text(evidence_id: str) -> dict:
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    record = ingest_service.get_evidence(evidence_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    if record.get("file_type") not in {"text", "document", "structured_data"}:
        raise HTTPException(status_code=422, detail="Text Analyzer accepts text, document, or structured_data evidence only.")
    try:
        contract = ingest_service.build_analyzer_input(evidence_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    from app.analyzers.text_analyzer import analyze_text_evidence
    result = analyze_text_evidence(contract)
    _save_analysis_result(evidence_id, "text", result)
    return result


@app.get("/api/v1/analyzers/text/{evidence_id}")
def get_text_result(evidence_id: str) -> dict:
    result = _load_analysis_result(evidence_id, "text")
    if result is None:
        raise HTTPException(status_code=404, detail="No text analysis result found.")
    return result


# ── Video Analyzer ────────────────────────────────────────────────────────────

@app.post("/api/v1/analyzers/video/{evidence_id}")
def analyze_video_evidence_endpoint(evidence_id: str) -> dict:
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    record = ingest_service.get_evidence(evidence_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    if record.get("file_type") != "video":
        raise HTTPException(status_code=422, detail="Video Analyzer accepts video evidence only.")
    try:
        contract = ingest_service.build_analyzer_input(evidence_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    from app.analyzers.video_analyzer import analyze_video_evidence
    result = analyze_video_evidence(
        contract=contract,
        data_dir=ingest_service.data_dir,
        run_audio_deepfake_fn=None,  # audio analysis handled internally
    )
    _save_analysis_result(evidence_id, "video", result)
    return result


@app.get("/api/v1/analyzers/video/{evidence_id}")
def get_video_result(evidence_id: str) -> dict:
    result = _load_analysis_result(evidence_id, "video")
    if result is None:
        raise HTTPException(status_code=404, detail="No video analysis result found.")
    return result


# ── Evidence Content Server ───────────────────────────────────────────────────

@app.get("/api/v1/evidence/{evidence_id}/content")
def serve_evidence_content(evidence_id: str) -> Response:
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    record = ingest_service.get_evidence(evidence_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    file_path = Path(record.get("file_path", ""))
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Evidence file not found")
    mime = record.get("mime_type", "application/octet-stream")
    return FileResponse(str(file_path), media_type=mime)


# ── Unified Auto-Routing Endpoints ──────────────────────────────────────────

@app.post("/api/v1/analyze/auto/{evidence_id}")
def auto_route_analysis(evidence_id: str) -> dict:
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    record = ingest_service.get_evidence(evidence_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    file_type = record.get("file_type")
    if file_type == "video":
        res = analyze_video_evidence_endpoint(evidence_id)
        routed = "Amogh Video Analyzer (GenD CLIP ViT-L/14 + Audio Wav2Vec2)"
    elif file_type == "image":
        res = analyze_image(evidence_id)
        routed = "Ramesh Image Analyzer (EfficientNet-B4 + ResNet-50 + ELA)"
    elif file_type == "audio":
        res = detect_audio_deepfake(evidence_id)
        routed = "Rohan Audio Analyzer (Wav2Vec2 ONNX)"
    elif file_type in ("text", "document", "structured_data"):
        res = analyze_text(evidence_id)
        routed = "Rakshitha Text & Context Analyzer"
    else:
        raise HTTPException(status_code=422, detail=f"Unsupported file type for analysis: {file_type}")
    
    return {
        "evidence_id": evidence_id,
        "original_filename": record.get("original_filename"),
        "file_type": file_type,
        "routed_analyzer": routed,
        "result": res,
    }


@app.get("/api/v1/analyze/results/{evidence_id}")
def get_unified_analysis_result(evidence_id: str) -> dict:
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    record = ingest_service.get_evidence(evidence_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    file_type = record.get("file_type")
    mod = "video" if file_type == "video" else "image" if file_type == "image" else "deepfake" if file_type == "audio" else "text"
    res = _load_analysis_result(evidence_id, mod)
    if res is None:
        raise HTTPException(status_code=404, detail=f"No analysis result found for {evidence_id}")
    return {
        "evidence_id": evidence_id,
        "original_filename": record.get("original_filename"),
        "file_type": file_type,
        "modality": mod,
        "result": res,
    }


@app.post("/api/v1/analyze/batch")
def auto_route_batch(evidence_ids: list[str] | None = None) -> dict:
    if evidence_ids is None:
        records = ingest_service.list_evidence()
        evidence_ids = [r["evidence_id"] for r in records if "evidence_id" in r]
    
    batch_results = []
    for ev_id in evidence_ids:
        try:
            batch_results.append(auto_route_analysis(ev_id))
        except Exception as exc:
            batch_results.append({
                "evidence_id": ev_id,
                "status": "failed",
                "error": str(exc),
            })
    return {"results": batch_results}


# ── Correlation & Review Endpoints ──────────────────────────────────────────

class ReviewPayload(BaseModel):
    status: str = "VERIFIED"
    notes: str = ""
    examiner: str = "Forensic Examiner"
    auto_findings: str = ""
    investigator_opinion: str = ""


@app.get("/api/v1/correlation/case")
def get_case_correlation() -> dict:
    from app.analyzers.correlator import correlate_case
    records = ingest_service.list_evidence()
    results = {}
    for r in records:
        ev_id = r.get("evidence_id")
        f_type = r.get("file_type")
        mod = "video" if f_type == "video" else "image" if f_type == "image" else "deepfake" if f_type == "audio" else "text"
        res = _load_analysis_result(ev_id, mod)
        if res:
            results[ev_id] = res
    return correlate_case(records, results)


@app.get("/api/v1/correlation/{evidence_id}")
def get_evidence_correlation(evidence_id: str) -> dict:
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    record = ingest_service.get_evidence(evidence_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    if record.get("file_type") != "video":
        raise HTTPException(status_code=400, detail="Correlation currently supported for multi-modal video evidence.")
    
    from app.analyzers.correlator import correlate_video_audio
    vid_res = _load_analysis_result(evidence_id, "video") or {}
    return correlate_video_audio(vid_res.get("video", {}), vid_res.get("audio"))


@app.post("/api/v1/review/{evidence_id}")
def submit_analyst_review(evidence_id: str, payload: ReviewPayload) -> dict:
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    from app.review import save_review
    return save_review(
        data_dir=ingest_service.data_dir,
        evidence_id=evidence_id,
        status=payload.status,
        notes=payload.notes,
        examiner=payload.examiner,
        auto_findings=payload.auto_findings,
        investigator_opinion=payload.investigator_opinion,
    )


@app.get("/api/v1/review/{evidence_id}")
def get_analyst_review(evidence_id: str) -> dict:
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")
    from app.review import get_review
    return get_review(data_dir=ingest_service.data_dir, evidence_id=evidence_id)


@app.get("/api/v1/reviews")
def list_analyst_reviews() -> dict:
    from app.review import list_reviews
    return {"reviews": list_reviews(data_dir=ingest_service.data_dir)}


# ── Report Export Endpoints ──────────────────────────────────────────────────

@app.get("/api/v1/report/{case_id}")
def export_forensic_report(case_id: str, format: str = Query("json")) -> Response:
    from app.report import generate_json_report, generate_pdf_report
    from app.review import list_reviews

    records = ingest_service.list_evidence()
    if case_id and case_id != "all":
        records = [r for r in records if r.get("case_id") == case_id or not r.get("case_id")]
    
    results = {}
    for r in records:
        ev_id = r.get("evidence_id")
        f_type = r.get("file_type")
        mod = "video" if f_type == "video" else "image" if f_type == "image" else "deepfake" if f_type == "audio" else "text"
        res = _load_analysis_result(ev_id, mod)
        if res:
            results[ev_id] = res
    
    reviews = list_reviews(data_dir=ingest_service.data_dir)

    if format.lower() == "pdf":
        pdf_bytes = generate_pdf_report(case_id, records, results, reviews)
        try:
            save_path = get_report_save_path(case_id, "pdf")
            save_path.write_bytes(pdf_bytes)
        except Exception:
            pass
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=truthlens_report_{case_id}.pdf"},
        )
    else:
        json_rep = generate_json_report(case_id, records, results, reviews)
        json_str = json.dumps(json_rep, indent=2)
        try:
            save_path = get_report_save_path(case_id, "json")
            save_path.write_text(json_str, encoding="utf-8")
        except Exception:
            pass
        return Response(
            content=json_str,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=truthlens_report_{case_id}.json"},
        )


@app.get("/api/v1/report/evidence/{evidence_id}")
def export_evidence_report(evidence_id: str, format: str = Query("json")) -> Response:
    """Generate and return a complete per-evidence-ID detailed forensic report."""
    if not re.fullmatch(r"EVD-\d+", evidence_id):
        raise HTTPException(status_code=404, detail="Evidence not found")

    from app.report import generate_evidence_pdf_report, generate_evidence_json_report
    from app.review import get_review

    record = ingest_service.get_evidence(evidence_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Evidence {evidence_id} not found")

    f_type = record.get("file_type", "")
    mod = "video" if f_type == "video" else "image" if f_type == "image" else "deepfake" if f_type == "audio" else "text"
    analysis = _load_analysis_result(evidence_id, mod)
    if analysis is None and f_type == "audio":
        analysis = _load_deepfake_result(evidence_id)

    review = get_review(ingest_service.data_dir, evidence_id)

    # For video evidence, also try to build correlation
    correlation: dict | None = None
    if f_type == "video" and analysis is not None:
        try:
            from app.analyzers.correlator import correlate_video_audio
            vid_obj = analysis.get("video") if isinstance(analysis, dict) else analysis
            aud_obj = analysis.get("audio") if isinstance(analysis, dict) else None
            if vid_obj:
                correlation = correlate_video_audio(vid_obj, aud_obj)
        except Exception as _exc:
            logger.warning("Correlation failed for %s during report generation: %s", evidence_id, _exc)

    if format.lower() == "pdf":
        pdf_bytes = generate_evidence_pdf_report(evidence_id, record, analysis, review, correlation)
        try:
            save_path = get_report_save_path(f"{evidence_id}_Forensic_Report", "pdf")
            save_path.write_bytes(pdf_bytes)
        except Exception:
            pass
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={evidence_id}_Forensic_Report.pdf"},
        )
    else:
        json_rep = generate_evidence_json_report(evidence_id, record, analysis, review, correlation)
        json_str = json.dumps(json_rep, indent=2)
        try:
            save_path = get_report_save_path(f"{evidence_id}_Forensic_Report", "json")
            save_path.write_text(json_str, encoding="utf-8")
        except Exception:
            pass
        return Response(
            content=json_str,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={evidence_id}_Forensic_Report.json"},
        )

