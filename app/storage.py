"""Centralized storage management for TruthLens.

Provides standard directories and accessors for:
storage/
    database/
        truthlens.db
        evidence_registry.json
    evidence/
        images/
        videos/
        audio/
        documents/
    extracted/
        audio/
        video_frames/
    analysis/
        image/
        video/
        audio/
        text/
        reviews/
    reports/
        pdf/
        json/
    testing/
        uploads/
        temporary/
    logs/
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Project root
ROOT = Path(__file__).resolve().parents[1]

# Base storage directory (can be overridden by TRUTHLENS_STORAGE_DIR)
STORAGE_DIR = Path(os.getenv("TRUTHLENS_STORAGE_DIR", ROOT / "storage")).resolve()

# Database
DATABASE_DIR = STORAGE_DIR / "database"
DB_PATH = DATABASE_DIR / "truthlens.db"
REGISTRY_PATH = DATABASE_DIR / "evidence_registry.json"

# Evidence vaults organized by type
EVIDENCE_DIR = STORAGE_DIR / "evidence"
EVIDENCE_IMAGES_DIR = EVIDENCE_DIR / "images"
EVIDENCE_VIDEOS_DIR = EVIDENCE_DIR / "videos"
EVIDENCE_AUDIO_DIR = EVIDENCE_DIR / "audio"
EVIDENCE_DOCUMENTS_DIR = EVIDENCE_DIR / "documents"

# Extracted artifacts
EXTRACTED_DIR = STORAGE_DIR / "extracted"
EXTRACTED_AUDIO_DIR = EXTRACTED_DIR / "audio"
EXTRACTED_FRAMES_DIR = EXTRACTED_DIR / "video_frames"

# Analysis outputs
ANALYSIS_DIR = STORAGE_DIR / "analysis"
ANALYSIS_IMAGE_DIR = ANALYSIS_DIR / "image"
ANALYSIS_VIDEO_DIR = ANALYSIS_DIR / "video"
ANALYSIS_AUDIO_DIR = ANALYSIS_DIR / "audio"
ANALYSIS_TEXT_DIR = ANALYSIS_DIR / "text"
ANALYSIS_REVIEWS_DIR = ANALYSIS_DIR / "reviews"

# Reports
REPORTS_DIR = STORAGE_DIR / "reports"
REPORTS_PDF_DIR = REPORTS_DIR / "pdf"
REPORTS_JSON_DIR = REPORTS_DIR / "json"

# Testing storage (isolated from actual case evidence)
TESTING_DIR = STORAGE_DIR / "testing"
TESTING_UPLOADS_DIR = TESTING_DIR / "uploads"
TESTING_TEMP_DIR = TESTING_DIR / "temporary"

# Logs
LOGS_DIR = STORAGE_DIR / "logs"
INGEST_ACTIVITY_PATH = LOGS_DIR / "ingest_activity.json"


def init_storage() -> None:
    """Ensure all storage directory trees exist."""
    directories = [
        DATABASE_DIR,
        EVIDENCE_IMAGES_DIR,
        EVIDENCE_VIDEOS_DIR,
        EVIDENCE_AUDIO_DIR,
        EVIDENCE_DOCUMENTS_DIR,
        EXTRACTED_AUDIO_DIR,
        EXTRACTED_FRAMES_DIR,
        ANALYSIS_IMAGE_DIR,
        ANALYSIS_VIDEO_DIR,
        ANALYSIS_AUDIO_DIR,
        ANALYSIS_TEXT_DIR,
        ANALYSIS_REVIEWS_DIR,
        REPORTS_PDF_DIR,
        REPORTS_JSON_DIR,
        TESTING_UPLOADS_DIR,
        TESTING_TEMP_DIR,
        LOGS_DIR,
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)


def get_evidence_destination_dir(file_type: str, is_testing: bool = False) -> Path:
    """Determine the correct storage folder for an evidence upload."""
    init_storage()
    if is_testing:
        return TESTING_UPLOADS_DIR

    file_type_clean = (file_type or "").lower().strip()
    if file_type_clean == "image":
        return EVIDENCE_IMAGES_DIR
    elif file_type_clean == "video":
        return EVIDENCE_VIDEOS_DIR
    elif file_type_clean == "audio":
        return EVIDENCE_AUDIO_DIR
    elif file_type_clean in {"text", "document", "structured_data"}:
        return EVIDENCE_DOCUMENTS_DIR
    else:
        # Fallback to documents for unknown text formats
        return EVIDENCE_DOCUMENTS_DIR


def get_extracted_audio_path(evidence_id: str) -> Path:
    """Path to store extracted WAV audio from video evidence."""
    EXTRACTED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    return EXTRACTED_AUDIO_DIR / f"{evidence_id}_extracted.wav"


def get_extracted_frames_dir(evidence_id: str) -> Path:
    """Directory to store extracted video frames for an evidence item."""
    frames_dir = EXTRACTED_FRAMES_DIR / evidence_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    return frames_dir


def get_analysis_file_path(evidence_id: str, modality: str) -> Path:
    """Path to store or load structured analyzer findings."""
    mod = (modality or "").lower()
    if mod in ("deepfake", "audio"):
        target_dir = ANALYSIS_AUDIO_DIR
    elif mod == "video":
        target_dir = ANALYSIS_VIDEO_DIR
    elif mod == "image":
        target_dir = ANALYSIS_IMAGE_DIR
    elif mod in ("text", "document"):
        target_dir = ANALYSIS_TEXT_DIR
    elif mod == "reviews":
        target_dir = ANALYSIS_REVIEWS_DIR
    else:
        target_dir = ANALYSIS_DIR / mod

    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{evidence_id}.json"


def get_report_save_path(case_id: str, format_type: str, timestamp_str: Optional[str] = None) -> Path:
    """Path to store generated forensic report (PDF or JSON)."""
    fmt = format_type.lower()
    ts = timestamp_str or ""
    suffix = f"_{ts}" if ts else ""
    filename = f"truthlens_report_{case_id}{suffix}.{fmt}"

    if fmt == "pdf":
        REPORTS_PDF_DIR.mkdir(parents=True, exist_ok=True)
        return REPORTS_PDF_DIR / filename
    else:
        REPORTS_JSON_DIR.mkdir(parents=True, exist_ok=True)
        return REPORTS_JSON_DIR / filename
