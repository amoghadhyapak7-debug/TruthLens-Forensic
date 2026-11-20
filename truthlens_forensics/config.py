"""
TruthLens — Centralized Configuration
All tunable constants in one place for clean code quality.
Unified: Image forensics (Sapthagiri) + Text forensics (Rakshita).
"""
import os

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STORAGE_DIR   = os.path.join(BASE_DIR, "storage")
ARTIFACT_DIR  = os.path.join(STORAGE_DIR, "artifacts")
EVIDENCE_DIR  = os.path.join(STORAGE_DIR, "evidence")
TEMP_DIR      = os.path.join(STORAGE_DIR, "temp_uploads")

# ── Upload Limits ────────────────────────────────────────────────
MAX_FILE_SIZE_MB    = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# ── Temp Portal ──────────────────────────────────────────────────
TEMP_TTL_SECONDS  = 30 * 60   # 30 minutes
TEMP_CLEANUP_INTERVAL = 300   # check every 5 minutes

# ── Analyzer ─────────────────────────────────────────────────────
ANALYZER_VERSION = "2.0.0"
MAX_IMAGE_PIXELS = 50_000_000   # 50 MP upper bound
SUPPORTED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP", "GIF"}

# ── AI Detector (EfficientNet-B4 — Google Brain architecture) ────
AI_MODEL_ID     = "efficientnet_b4"
AI_INPUT_SIZE   = (380, 380)
AI_THRESHOLD_HIGH   = 0.65
AI_THRESHOLD_MEDIUM = 0.45

# ── Manipulation Detector (ResNet-50) ────────────────────────────
MANIP_MODEL_ID         = "resnet50"
MANIP_INPUT_SIZE       = (224, 224)
MANIP_PEAK_THRESHOLD   = 3.2   # z-score outlier threshold (prevents natural foreground false alarms)
MANIP_HIGH_SCORE       = 0.80  # calibrated anomaly score for High signal
MANIP_MEDIUM_SCORE     = 0.60

# ── ELA ──────────────────────────────────────────────────────────
ELA_QUALITY          = 95
ELA_AMPLIFICATION    = 20
ELA_MIN_ERROR_FLOOR  = 12.0    # ignores baseline JPEG quantization noise < 12
ELA_HIGH_FRAC_HIGH   = 0.08
ELA_HIGH_FRAC_MEDIUM = 0.03

# ── Noise / Wavelet ──────────────────────────────────────────────
NOISE_BVR_HIGH   = 2.2         # genuine sensor mismatch / splicing
NOISE_BVR_MEDIUM = 1.6         # moderate dispersion
NOISE_BVR_NORMAL = 1.2         # uniform sensor baseline

# ── Signal / Assessment ──────────────────────────────────────────
AI_PROB_DEFINITIVE     = 0.45   # stripped EXIF + no camera → ai_generated
AI_PROB_HIGH           = 0.70
MANIP_SCORE_CONFIRM    = 0.80   # score requiring physical corroboration
EDITING_SW_MANIP_SCORE = 0.65

# ── Flask ─────────────────────────────────────────────────────────
FLASK_HOST  = "0.0.0.0"
FLASK_PORT  = 5000
FLASK_DEBUG = False

# ── Text Module ───────────────────────────────────────────────────
TEXT_UPLOAD_DIR = os.path.join(STORAGE_DIR, "text_uploads")
TEXT_ARTIFACT_DIR = os.path.join(ARTIFACT_DIR, "text")
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".pdf", ".docx"}

