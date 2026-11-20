"""TruthLens Video Forensics Adapter — wraps Amogh's truthlens_video pipeline.

Responsibilities:
1. Run truthlens_video.analyze_video() on the registered evidence.
2. Extract audio from the video using FFmpeg (via imageio_ffmpeg or subprocess).
3. Save extracted audio as a child artifact linked to the parent evidence ID.
4. Call Rohan's existing audio deepfake detection on the extracted audio.
5. Return both video result and audio result linked to the same evidence ID.
"""
from __future__ import annotations
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _extract_audio_ffmpeg(video_path: Path, output_wav: Path) -> tuple[bool, str]:
    """Extract audio stream from a video file to WAV using FFmpeg."""
    try:
        # Try imageio_ffmpeg first (already a dependency)
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = "ffmpeg"
        
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(video_path),
            "-vn",                    # no video
            "-acodec", "pcm_s16le",   # PCM WAV output
            "-ar", "16000",           # 16kHz for Wav2Vec2
            "-ac", "1",               # mono
            str(output_wav),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            stderr_lower = stderr.lower()
            silent_indicators = [
                "does not contain any stream",
                "matches no streams",
                "no audio stream",
                "output file does not contain any stream",
                "no audio",
                "no streams",
            ]
            if any(ind in stderr_lower for ind in silent_indicators):
                return False, "NO_AUDIO_STREAM"
            return False, f"FFmpeg error: {stderr[:300]}"
        
        if not output_wav.exists() or output_wav.stat().st_size < 100:
            return False, "NO_AUDIO_STREAM"
        return True, "ok"
    except subprocess.TimeoutExpired:
        return False, "Audio extraction timed out"
    except Exception as exc:
        return False, f"Audio extraction error: {exc}"


def analyze_video_evidence(
    contract: Dict[str, Any],
    data_dir: Path,
    run_audio_deepfake_fn,  # callable: (evidence_id: str) -> dict
) -> Dict[str, Any]:
    """
    Full video forensic analysis with audio deepfake detection on extracted audio.
    
    Returns:
        {
            "video": <truthlens_video result>,
            "audio": <audio deepfake result or None>,
            "audio_extraction_status": "extracted" | "no_audio_stream" | "failed" | "not_applicable",
            "audio_evidence_id": <str if extracted else None>,
        }
    """
    evidence_id = contract.get("evidence_id", "EVD-UNK")
    video_path = Path(contract["file_path"])
    
    # --- Step 1: Run Amogh's video analyzer ---
    video_result: Dict[str, Any] = {}
    try:
        from truthlens_video import analyze_video
        video_result = analyze_video(
            file_path=video_path,
            evidence_id=evidence_id,
            options={
                "enable_gemini": bool(os.getenv("GEMINI_API_KEY")),
                "gemini_api_key": os.getenv("GEMINI_API_KEY"),
                "cache_dir": str(data_dir / "artifacts" / evidence_id / "video"),
            },
        )
    except Exception as exc:
        logger.error(f"Video analysis failed for {evidence_id}: {exc}", exc_info=True)
        video_result = {
            "evidence_id": evidence_id,
            "status": "failed",
            "errors": [str(exc)],
            "findings": [],
            "signals": [],
        }

    # --- Step 2: Extract audio and run audio deepfake detection ---
    audio_result: Optional[Dict[str, Any]] = None
    audio_extraction_status = "not_attempted"
    child_audio_evidence_id: Optional[str] = None

    try:
        from app.storage import get_extracted_audio_path
        wav_path = get_extracted_audio_path(evidence_id)
    except Exception:
        audio_dir = data_dir / "artifacts" / evidence_id / "extracted_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        wav_path = audio_dir / f"{evidence_id}_extracted.wav"

    success, msg = _extract_audio_ffmpeg(video_path, wav_path)

    if success:
        audio_extraction_status = "extracted"
        # Mirror to legacy path for backward compatibility
        try:
            legacy_dir = data_dir / "artifacts" / evidence_id / "extracted_audio"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            legacy_wav = legacy_dir / f"{evidence_id}_extracted.wav"
            if legacy_wav.resolve() != wav_path.resolve():
                import shutil
                shutil.copy2(wav_path, legacy_wav)
        except Exception:
            pass
        # Register the extracted audio as a child artifact (not a full user upload)
        # We call the audio deepfake function directly with the WAV path
        try:
            # Build a minimal contract for the audio deepfake detector
            audio_result = _run_audio_on_extracted(
                wav_path=wav_path,
                parent_evidence_id=evidence_id,
                run_audio_deepfake_fn=run_audio_deepfake_fn,
                data_dir=data_dir,
            )
        except Exception as exc:
            logger.error(f"Audio deepfake analysis failed for extracted audio of {evidence_id}: {exc}", exc_info=True)
            audio_result = {
                "status": "failed",
                "parent_video_evidence_id": evidence_id,
                "audio_source": "extracted_from_video",
                "errors": [str(exc)],
            }
    elif msg == "NO_AUDIO_STREAM":
        audio_extraction_status = "no_audio_stream"
        logger.info(f"Video {evidence_id} has no audio stream — skipping audio analysis.")
    else:
        audio_extraction_status = "extraction_failed"
        logger.warning(f"Audio extraction failed for {evidence_id}: {msg}")
        audio_result = {
            "status": "failed",
            "parent_video_evidence_id": evidence_id,
            "audio_source": "extracted_from_video",
            "errors": [msg],
        }

    # Save combined result to disk
    combined = {
        "evidence_id": evidence_id,
        "video": video_result,
        "audio": audio_result,
        "audio_extraction_status": audio_extraction_status,
    }
    try:
        out_path = data_dir / "artifacts" / evidence_id / "video_combined.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass

    return combined


def _run_audio_on_extracted(
    wav_path: Path,
    parent_evidence_id: str,
    run_audio_deepfake_fn,
    data_dir: Path,
) -> Dict[str, Any]:
    """
    Run the Wav2Vec2 audio deepfake model on an extracted WAV file.
    Uses the existing detect_synthetic_audio function from app.analyzers.audio_models directly.
    Does NOT ingest the extracted audio as a new user evidence record.
    """
    from app.analyzers.audio import _read_wav
    from app.analyzers.audio_models import detect_synthetic_audio as _detect
    from app.analyzers.config import SETTINGS
    import numpy as np

    try:
        samples, rate, channels, bit_depth = _read_wav(wav_path, SETTINGS)
    except Exception:
        # Fallback: use _read_compressed
        from app.analyzers.audio import _read_compressed
        samples, rate, channels, _, _ = _read_compressed(wav_path, SETTINGS)

    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    mono = mono.reshape(-1).astype(np.float32)
    
    if mono.size == 0 or not np.all(np.isfinite(mono)):
        return {
            "status": "inconclusive",
            "result": "INCONCLUSIVE",
            "parent_video_evidence_id": parent_evidence_id,
            "audio_source": "extracted_from_video",
            "explanation": "Extracted audio waveform was empty or invalid.",
        }
    
    # Segment and run inference (reuse main.py segmentation logic)
    # For simplicity, run on first 15s max
    max_samples = int(15.0 * rate)
    segment = mono[:max_samples]
    
    detected = _detect(segment, rate)
    
    result = {
        "status": "completed",
        "parent_video_evidence_id": parent_evidence_id,
        "audio_source": "extracted_from_video",
        "wav_path": str(wav_path),
        "inference": detected,
    }
    
    # Extract verdict
    if detected.get("status") == "completed":
        c_scores = detected.get("class_scores") or {}
        f_score = next(
            (float(v) for k, v in c_scores.items()
             if any(t in str(k).lower() for t in ("spoof", "fake", "synthetic"))),
            None
        )
        if f_score is not None:
            if f_score >= 0.70:
                result["result"] = "SYNTHETIC AUDIO INDICATORS DETECTED"
            elif f_score <= 0.60:
                result["result"] = "NO STRONG SYNTHETIC INDICATORS DETECTED"
            else:
                result["result"] = "INCONCLUSIVE"
            result["synthetic_score"] = round(f_score, 4)
    
    return result
