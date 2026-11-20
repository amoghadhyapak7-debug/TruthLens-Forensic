"""TruthLens Normalization Utilities.

Provides robust normalization of analyzer outputs across modalities
(Video, Image, Audio, Text) to defend against schema variations, string-vs-dictionary
mismatches, and legacy contract formats.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def normalize_video_verdict(v_data: Any) -> Dict[str, Any]:
    """
    Safely normalizes video forensic result / verdict into a standard contract:
    {
        "verdict": str,            # e.g. "DEEPFAKE DETECTED", "NO DEEPFAKE DETECTED", "INCONCLUSIVE"
        "fake_probability": float,  # 0.0 to 1.0
        "confidence_score": float,  # 0.0 to 1.0
        "classification": str,     # normalized classification category
        "basis": list[str],        # explanation points
        "models_used": list[str],  # list of models used
        "status": str,             # "completed" | "failed" | "inconclusive"
    }

    Handles:
    - Entire video record dict containing 'assessment', 'verdict', 'deepfake_metrics'
    - Nested dict: v_data['verdict'] is a dict with {'verdict', 'fake_probability'}
    - String verdict: v_data['verdict'] is a str like "DEEPFAKE DETECTED"
    - JSON string: '{"verdict": "...", "fake_probability": 0.9}'
    - Standalone string passed directly
    - None or unexpected types
    """
    default_res = {
        "verdict": "INCONCLUSIVE",
        "fake_probability": 0.0,
        "confidence_score": 0.0,
        "classification": "INCONCLUSIVE",
        "basis": [],
        "models_used": ["GenD CLIP ViT-L/14", "MesoNet-4", "YuNet"],
        "status": "inconclusive",
    }

    if v_data is None:
        return default_res

    # If passed a standalone string
    if isinstance(v_data, str):
        v_str = v_data.strip()
        if v_str.startswith("{") and v_str.endswith("}"):
            try:
                parsed = json.loads(v_str)
                if isinstance(parsed, dict):
                    return normalize_video_verdict(parsed)
            except Exception:
                pass
        
        # Plain text verdict string
        prob = 0.0
        if "DEEPFAKE DETECTED" in v_str and "NO" not in v_str:
            prob = 0.95
        elif "NO DEEPFAKE" in v_str or "AUTHENTIC" in v_str:
            prob = 0.05
        return {
            **default_res,
            "verdict": v_str or "INCONCLUSIVE",
            "classification": v_str or "INCONCLUSIVE",
            "fake_probability": prob,
            "confidence_score": prob if prob > 0 else 0.5,
            "status": "completed",
        }

    if not isinstance(v_data, dict):
        logger.warning(f"Unexpected video data type: {type(v_data)} ({v_data})")
        return {
            **default_res,
            "verdict": f"UNKNOWN_FORMAT ({type(v_data).__name__})",
        }

    # Unwrap if wrapped inside a parent 'video' key
    if "video" in v_data and isinstance(v_data["video"], (dict, str)):
        return normalize_video_verdict(v_data["video"])

    verdict_raw = v_data.get("verdict")
    assessment = v_data.get("assessment") if isinstance(v_data.get("assessment"), dict) else {}
    deepfake_metrics = v_data.get("deepfake_metrics") if isinstance(v_data.get("deepfake_metrics"), dict) else {}
    status = v_data.get("status", "completed")
    models_used = v_data.get("models_used") or default_res["models_used"]

    verdict_str: Optional[str] = None
    fake_prob: Optional[float] = None
    confidence: Optional[float] = None
    basis = assessment.get("basis", []) or v_data.get("basis", [])

    # Case 1: verdict_raw is a dict
    if isinstance(verdict_raw, dict):
        verdict_str = verdict_raw.get("verdict") or verdict_raw.get("classification")
        fake_prob = verdict_raw.get("fake_probability", verdict_raw.get("confidence_score"))
        confidence = verdict_raw.get("confidence", verdict_raw.get("confidence_score"))

    # Case 2: verdict_raw is a string
    elif isinstance(verdict_raw, str):
        v_str = verdict_raw.strip()
        if v_str.startswith("{") and v_str.endswith("}"):
            try:
                parsed = json.loads(v_str)
                if isinstance(parsed, dict):
                    verdict_str = parsed.get("verdict") or parsed.get("classification")
                    fake_prob = parsed.get("fake_probability", parsed.get("confidence_score"))
                    confidence = parsed.get("confidence", parsed.get("confidence_score"))
            except Exception:
                verdict_str = v_str
        else:
            verdict_str = v_str

    # Check top-level keys in v_data
    if fake_prob is None or fake_prob == 0.0:
        for k in ("fake_probability", "confidence_score", "score", "evidence_score"):
            if v_data.get(k) is not None:
                try:
                    fake_prob = float(v_data[k])
                    break
                except (ValueError, TypeError):
                    pass

    if confidence is None or confidence == 0.0:
        for k in ("confidence", "confidence_score"):
            if v_data.get(k) is not None:
                try:
                    confidence = float(v_data[k])
                    break
                except (ValueError, TypeError):
                    pass

    # Case 3: Extract from assessment dict if not yet found
    if not verdict_str or verdict_str in ("INCONCLUSIVE", "UNKNOWN"):
        if assessment.get("verdict"):
            verdict_str = assessment.get("verdict")
        elif assessment.get("classification"):
            verdict_str = assessment.get("classification")

    if fake_prob is None or fake_prob == 0.0:
        manip_pct = assessment.get("manipulation_score_pct")
        if manip_pct is not None:
            try:
                fake_prob = float(manip_pct) / 100.0
            except (ValueError, TypeError):
                pass
        elif assessment.get("evidence_score") is not None:
            try:
                fake_prob = float(assessment["evidence_score"])
            except (ValueError, TypeError):
                pass

    if confidence is None or confidence == 0.0:
        if assessment.get("confidence") is not None:
            try:
                confidence = float(assessment["confidence"])
            except (ValueError, TypeError):
                pass

    # Case 4: Extract from deepfake_metrics signal
    if fake_prob is None or fake_prob == 0.0:
        sig_prob = deepfake_metrics.get("fake_probability")
        if sig_prob is None and isinstance(deepfake_metrics.get("signal_data"), dict):
            sig_prob = deepfake_metrics["signal_data"].get("fake_probability")
        if sig_prob is not None:
            try:
                fake_prob = float(sig_prob)
            except (ValueError, TypeError):
                pass

    # Handle status failed
    if status == "failed" and (not verdict_str or verdict_str == "INCONCLUSIVE"):
        verdict_str = "ANALYSIS FAILED"

    # Fallbacks and sanitization
    final_verdict = str(verdict_str or "INCONCLUSIVE").strip()
    try:
        final_prob = float(fake_prob if fake_prob is not None else 0.0)
    except (ValueError, TypeError):
        final_prob = 0.0
    final_prob = min(max(final_prob, 0.0), 1.0)

    try:
        final_conf = float(confidence if confidence is not None else final_prob)
    except (ValueError, TypeError):
        final_conf = final_prob
    final_conf = min(max(final_conf, 0.0), 1.0)

    return {
        "verdict": final_verdict,
        "fake_probability": final_prob,
        "confidence_score": final_conf,
        "classification": assessment.get("classification") or final_verdict,
        "basis": list(basis) if isinstance(basis, (list, tuple)) else [str(basis)],
        "models_used": list(models_used) if isinstance(models_used, (list, tuple)) else [str(models_used)],
        "status": status,
    }
