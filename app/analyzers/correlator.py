"""TruthLens Multi-Modal Correlator.

Analyzes relationships across multiple evidence items and modalities:
1. Video + Audio cross-modality correlation for the same video.
2. Cross-evidence timeline, metadata, and thematic consistency.
3. Conflicting signal detection with forensic reasoning.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.analyzers.normalization import normalize_video_verdict

logger = logging.getLogger(__name__)


def correlate_video_audio(video_result: Dict[str, Any], audio_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Correlate visual findings (from GenD/MesoNet) with audio findings (from Wav2Vec2)
    for a single video evidence item.
    """
    v_norm = normalize_video_verdict(video_result)
    v_verdict = v_norm["verdict"]
    v_score = v_norm["fake_probability"]

    a_verdict = "NOT_APPLICABLE"
    a_score = None
    if audio_result and isinstance(audio_result, dict):
        a_status = audio_result.get("status")
        if a_status == "completed":
            a_res = audio_result.get("result", "")
            if "SYNTHETIC AUDIO INDICATORS DETECTED" in a_res:
                a_verdict = "SYNTHETIC_DETECTED"
            elif "NO STRONG SYNTHETIC" in a_res:
                a_verdict = "NOT_DETECTED"
            else:
                a_verdict = "INCONCLUSIVE"
            a_score = audio_result.get("synthetic_score")
        elif audio_result.get("audio_source") == "extracted_from_video" and audio_result.get("status") == "no_audio_stream":
            a_verdict = "SILENT_VIDEO"
        elif audio_result.get("status") == "failed":
            a_verdict = "EXTRACTION_FAILED"

    # Multi-modal synthesis logic (grounded in actual signals)
    conflicts = []
    anomaly_type = "NONE"
    explanation = ""

    v_is_fake = "DEEPFAKE" in str(v_verdict).upper() and "NO" not in str(v_verdict).upper()
    v_is_real = "NO DEEPFAKE" in str(v_verdict).upper()
    a_is_fake = a_verdict == "SYNTHETIC_DETECTED"
    a_is_real = a_verdict == "NOT_DETECTED"

    if v_is_fake and a_is_fake:
        anomaly_type = "FULL_MULTIMODAL_SYNTHETIC"
        unified_verdict = "HIGH CONFIDENCE MULTI-MODAL DEEPFAKE"
        explanation = (
            "Both visual (GenD CLIP ViT-L/14) and auditory (Wav2Vec2 ONNX) streams exhibit strong synthetic indicators. "
            "High probability of end-to-end generative AI video production (e.g. generative avatar, deepfake video with voice clone)."
        )
    elif v_is_fake and a_is_real:
        anomaly_type = "VISUAL_ONLY_MANIPULATION"
        unified_verdict = "VISUAL DEEPFAKE WITH AUTHENTIC AUDIO"
        conflicts.append("Visual stream flagged as synthetic while audio stream shows no synthetic indicators.")
        explanation = (
            "Visual frames exhibit face manipulation or synthetic generation indicators, but the audio track lacks synthetic indicators. "
            "Typical pattern for face-swap deepfakes grafted onto authentic interview or broadcast recordings."
        )
    elif v_is_real and a_is_fake:
        anomaly_type = "AUDIO_ONLY_MANIPULATION"
        unified_verdict = "AUTHENTIC VIDEO WITH SYNTHETIC AUDIO DUB"
        conflicts.append("Audio stream flagged as synthetic while visual frames show no synthetic indicators.")
        explanation = (
            "Audio track contains acoustic artifacts of synthetic speech generation (voice cloning / speech synthesis), "
            "while camera original visual characteristics appear intact. Indicates audio replacement, deceptive redubbing, or out-of-context voiceover."
        )
    elif v_is_real and a_is_real:
        anomaly_type = "NO_SYNTHETIC_INDICATORS"
        unified_verdict = "NO SYNTHETIC INDICATORS DETECTED"
        explanation = (
            "Neither visual frames nor extracted audio stream exhibited patterns characteristic of current generative models. "
            "Note: Does not rule out high-effort manual editing or novel generation methods outside detector training distributions."
        )
    elif a_verdict in ("SILENT_VIDEO", "NOT_APPLICABLE"):
        anomaly_type = "SILENT_VIDEO"
        unified_verdict = f"VISUAL: {v_verdict} (AUDIO: N/A - NO AUDIO TRACK)"
        explanation = (
            "Video stream was analyzed independently; audio analysis was not performed because "
            "the media container has no audio track, or audio could not be extracted."
        )
    else:
        anomaly_type = "PARTIAL_OR_INCONCLUSIVE"
        unified_verdict = f"VISUAL: {v_verdict} | AUDIO: {a_verdict}"
        explanation = "One or both modalities yielded inconclusive or unanalyzable signals. Review individual forensic findings for detail."

    return {
        "visual_verdict": v_verdict,
        "visual_score": v_score,
        "audio_verdict": a_verdict,
        "audio_score": a_score,
        "unified_verdict": unified_verdict,
        "anomaly_type": anomaly_type,
        "conflicting_signals": conflicts,
        "forensic_synthesis": explanation,
        "limitations": [
            "GenD ViT-L/14 specializes in facial and whole-frame visual artifacts; compression may introduce borderline noise.",
            "Wav2Vec2 ONNX analyzes acoustic feature representations in 15s windows; background noise or re-encoding can reduce confidence.",
            "Cross-modality correlation does not replace human forensic examiner review.",
        ],
    }


def correlate_case(evidence_records: List[Dict[str, Any]], results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Produce a cross-evidence forensic correlation across all evidence items in the case.
    """
    findings_summary = []
    timestamps = []
    file_types = {}

    for rec in evidence_records:
        ev_id = rec.get("evidence_id")
        f_type = rec.get("file_type", "unknown")
        file_types[f_type] = file_types.get(f_type, 0) + 1
        
        ingest_time = rec.get("ingestion_timestamp")
        if ingest_time:
            timestamps.append({"evidence_id": ev_id, "timestamp": ingest_time, "filename": rec.get("original_filename")})
        
        ev_res = results.get(ev_id, {})
        findings_summary.append({
            "evidence_id": ev_id,
            "filename": rec.get("original_filename"),
            "file_type": f_type,
            "sha256": rec.get("sha256"),
            "analysis_available": bool(ev_res),
        })

    return {
        "total_evidence_count": len(evidence_records),
        "modality_breakdown": file_types,
        "timeline": sorted(timestamps, key=lambda x: str(x.get("timestamp"))),
        "evidence_summary": findings_summary,
        "cross_modality_coverage": {
            "has_video": "video" in file_types,
            "has_image": "image" in file_types,
            "has_audio": "audio" in file_types,
            "has_text": any(k in file_types for k in ["text", "document", "structured_data"]),
        },
    }
