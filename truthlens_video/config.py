"""TruthLens Video Forensics - Configuration Module

Handles environment variable loading, default forensic thresholds,
model specifications, and path definitions.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = (PROJECT_ROOT / "storage" / "extracted" / "video_frames") if (PROJECT_ROOT / "storage").exists() else (PROJECT_ROOT / "evidence_cache")

# Supported video formats
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# Google Gemini API configuration
# Keeps model fully configurable via environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))

# Frame extraction settings
DEFAULT_FRAME_SAMPLE_RATE_SEC = float(os.getenv("FRAME_SAMPLE_RATE_SEC", "1.0"))  # Sample 1 frame per second
MAX_FRAMES_PER_VIDEO = int(os.getenv("MAX_FRAMES_PER_VIDEO", "40"))             # Prevent OOM on long videos
MIN_FRAMES_PER_VIDEO = int(os.getenv("MIN_FRAMES_PER_VIDEO", "5"))
SCENE_CHANGE_THRESHOLD = float(os.getenv("SCENE_CHANGE_THRESHOLD", "0.35"))      # Normalized difference for cut detection
DUPLICATE_FRAME_THRESHOLD = float(os.getenv("DUPLICATE_FRAME_THRESHOLD", "0.02"))  # Threshold for frozen/duplicate frames

# Forensic analysis thresholds
LAPLACIAN_BLUR_THRESHOLD = float(os.getenv("LAPLACIAN_BLUR_THRESHOLD", "80.0"))
HIGH_FREQUENCY_NOISE_THRESHOLD = float(os.getenv("HIGH_FREQUENCY_NOISE_THRESHOLD", "1.5"))
TEMPORAL_JITTER_THRESHOLD = float(os.getenv("TEMPORAL_JITTER_THRESHOLD", "2.8"))
LIGHTING_INCONSISTENCY_THRESHOLD = float(os.getenv("LIGHTING_INCONSISTENCY_THRESHOLD", "0.28"))

# Multi-video comparison settings
PHASH_SIZE = int(os.getenv("PHASH_SIZE", "8"))                                   # 8x8 = 64-bit perceptual hash
PHASH_HAMMING_MATCH_THRESHOLD = int(os.getenv("PHASH_MATCH_THRESHOLD", "6"))     # Distance <= 6 considered matching frame
VIDEO_SIMILARITY_MATCH_THRESHOLD = float(os.getenv("VIDEO_SIMILARITY_MATCH_THRESHOLD", "0.65"))
SHARED_SOURCE_MIN_MATCHING_FRAMES = int(os.getenv("SHARED_SOURCE_MIN_MATCHING_FRAMES", "3"))

# Assessment classification labels
CLASS_NO_MANIPULATION = "no_manipulation_detected"
CLASS_POTENTIALLY_MANIPULATED = "potentially_manipulated"
CLASS_POTENTIALLY_AI_GENERATED = "potentially_ai_generated"
CLASS_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
CLASS_ANALYSIS_FAILED = "analysis_failed"

# Human-readable verdict strings surfaced in the report UI (Strict binary options)
VERDICT_DEEPFAKE = "DEEPFAKE DETECTED"
VERDICT_NOT_DEEPFAKE = "NO DEEPFAKE DETECTED"

# Backwards compatibility aliases
VERDICT_ORIGINAL = VERDICT_NOT_DEEPFAKE
VERDICT_DEEPFAKE_DETECTED = VERDICT_DEEPFAKE
VERDICT_NO_DEEPFAKE_DETECTED = VERDICT_NOT_DEEPFAKE
VERDICT_NO_MANIPULATION = VERDICT_NOT_DEEPFAKE
VERDICT_LIKELY_DEEPFAKE = VERDICT_DEEPFAKE
VERDICT_POSSIBLE_MANIPULATION = VERDICT_DEEPFAKE
VERDICT_LIKELY_AI_GENERATED = VERDICT_DEEPFAKE
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
VERDICT_INSUFFICIENT = "INCONCLUSIVE"

VALID_CLASSIFICATIONS = [
    CLASS_NO_MANIPULATION,
    CLASS_POTENTIALLY_MANIPULATED,
    CLASS_POTENTIALLY_AI_GENERATED,
    CLASS_INSUFFICIENT_EVIDENCE,
    CLASS_ANALYSIS_FAILED,
]

# -------------------------------------------------------
# Weighted scoring thresholds for the verdict engine
# Each forensic check contributes points to a total score.
# Score >= VERDICT_MANIPULATED_THRESHOLD  → DEEPFAKE / MANIPULATED
# Score >= VERDICT_SUSPICIOUS_THRESHOLD   → DEEPFAKE / MANIPULATED (lower bar when multiple weak signals)
# Score == 0                              → ORIGINAL
# else                                    → INSUFFICIENT EVIDENCE (single weak signal)
# -------------------------------------------------------
VERDICT_MANIPULATED_THRESHOLD = float(os.getenv("VERDICT_MANIPULATED_THRESHOLD", "3.0"))
VERDICT_SUSPICIOUS_THRESHOLD  = float(os.getenv("VERDICT_SUSPICIOUS_THRESHOLD",  "1.5"))

__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_CACHE_DIR",
    "SUPPORTED_EXTENSIONS",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_MAX_RETRIES",
    "GEMINI_TIMEOUT_SECONDS",
    "DEFAULT_FRAME_SAMPLE_RATE_SEC",
    "MAX_FRAMES_PER_VIDEO",
    "MIN_FRAMES_PER_VIDEO",
    "SCENE_CHANGE_THRESHOLD",
    "DUPLICATE_FRAME_THRESHOLD",
    "LAPLACIAN_BLUR_THRESHOLD",
    "HIGH_FREQUENCY_NOISE_THRESHOLD",
    "TEMPORAL_JITTER_THRESHOLD",
    "LIGHTING_INCONSISTENCY_THRESHOLD",
    "PHASH_SIZE",
    "PHASH_HAMMING_MATCH_THRESHOLD",
    "VIDEO_SIMILARITY_MATCH_THRESHOLD",
    "SHARED_SOURCE_MIN_MATCHING_FRAMES",
    "CLASS_NO_MANIPULATION",
    "CLASS_POTENTIALLY_MANIPULATED",
    "CLASS_POTENTIALLY_AI_GENERATED",
    "CLASS_INSUFFICIENT_EVIDENCE",
    "CLASS_ANALYSIS_FAILED",
    "VERDICT_DEEPFAKE",
    "VERDICT_NOT_DEEPFAKE",
    "VERDICT_DEEPFAKE_DETECTED",
    "VERDICT_NO_DEEPFAKE_DETECTED",
    "VERDICT_ORIGINAL",
    "VERDICT_INCONCLUSIVE",
    "VERDICT_INSUFFICIENT",
    "VERDICT_LIKELY_AI_GENERATED",
    "VERDICT_LIKELY_DEEPFAKE",
    "VERDICT_POSSIBLE_MANIPULATION",
    "VERDICT_NO_MANIPULATION",
    "VALID_CLASSIFICATIONS",
    "VERDICT_MANIPULATED_THRESHOLD",
    "VERDICT_SUSPICIOUS_THRESHOLD",
]

