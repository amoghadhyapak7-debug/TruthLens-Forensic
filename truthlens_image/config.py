"""TruthLens Image Forensics - Configuration Module

Defines supported image formats, configurable upload limits, forensic thresholds,
and model settings for the image forensic detector.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "evidence_cache"

# Supported image formats
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Configurable upload file size limit (in Megabytes)
MAX_IMAGE_UPLOAD_SIZE_MB = float(os.getenv("MAX_IMAGE_UPLOAD_SIZE_MB", "25.0"))
MAX_IMAGE_UPLOAD_SIZE_BYTES = int(MAX_IMAGE_UPLOAD_SIZE_MB * 1024 * 1024)

# Google Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

# Error Level Analysis (ELA) settings
ELA_JPEG_QUALITY = int(os.getenv("ELA_JPEG_QUALITY", "92"))
ELA_SCALE_MULTIPLIER = float(os.getenv("ELA_SCALE_MULTIPLIER", "15.0"))
ELA_ANOMALY_THRESHOLD = float(os.getenv("ELA_ANOMALY_THRESHOLD", "45.0"))

# Forensic signal thresholds
HIGH_FREQUENCY_NOISE_THRESHOLD = float(os.getenv("IMAGE_NOISE_THRESHOLD", "1.8"))
LAPLACIAN_BLUR_THRESHOLD = float(os.getenv("IMAGE_BLUR_THRESHOLD", "50.0"))
COLOR_DISCREPANCY_THRESHOLD = float(os.getenv("IMAGE_COLOR_DISCREPANCY_THRESHOLD", "0.32"))

# Perceptual hash settings
PHASH_SIZE = int(os.getenv("IMAGE_PHASH_SIZE", "8"))

# Classification labels
CLASS_NO_MANIPULATION = "no_manipulation_detected"
CLASS_POTENTIALLY_MANIPULATED = "potentially_manipulated"
CLASS_POTENTIALLY_AI_GENERATED = "potentially_ai_generated"
CLASS_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
CLASS_ANALYSIS_FAILED = "analysis_failed"

VALID_CLASSIFICATIONS = [
    CLASS_NO_MANIPULATION,
    CLASS_POTENTIALLY_MANIPULATED,
    CLASS_POTENTIALLY_AI_GENERATED,
    CLASS_INSUFFICIENT_EVIDENCE,
    CLASS_ANALYSIS_FAILED,
]
