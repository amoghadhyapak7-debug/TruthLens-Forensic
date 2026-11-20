"""TruthLens Video Forensics - Correlation Contract & Exporter

Constructs, validates, and serializes the structured JSON output conforming to the
TruthLens inter-module contract for Stage 3 (CORRELATE).

Transparent Decision Engine
----------------------------
Three verdict categories:
1. DEEPFAKE DETECTED:
   - MesoNet-4 mean manipulation probability >= 60% across face frames, OR
   - Peak manipulation score >= 70% with >= 2 suspicious frames, OR
   - Strong multi-modal forensic signal (weighted score >= 3.0) with facial forgery cue.
2. NO DEEPFAKE DETECTED:
   - Completed examination with low manipulation scores and no meaningful forensic anomalies.
3. INCONCLUSIVE:
   - Borderline model scores, insufficient face frames, or isolated weak signals
     without multi-frame corroboration.

No AI-generated video classification — this is a deepfake detector, not an AI-media detector.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from truthlens_video.config import (
    CLASS_ANALYSIS_FAILED,
    CLASS_INSUFFICIENT_EVIDENCE,
    CLASS_NO_MANIPULATION,
    CLASS_POTENTIALLY_MANIPULATED,
    VALID_CLASSIFICATIONS,
    VERDICT_DEEPFAKE,
    VERDICT_NOT_DEEPFAKE,
    VERDICT_DEEPFAKE_DETECTED,
    VERDICT_NO_DEEPFAKE_DETECTED,
    VERDICT_INCONCLUSIVE,
    VERDICT_MANIPULATED_THRESHOLD,
    VERDICT_SUSPICIOUS_THRESHOLD,
)

# Keep aliases so any other module that imports the old names still works
VERDICT_LIKELY_DEEPFAKE = VERDICT_DEEPFAKE
VERDICT_NO_MANIPULATION = VERDICT_NOT_DEEPFAKE


# ---------------------------------------------------------------------------
# Indicator weight table (deepfake / manipulation signals only)
# ---------------------------------------------------------------------------
_INDICATOR_WEIGHTS: Dict[str, float] = {
    # Neural deepfake detection (highest weights)
    "facial_deepfake_forgery_detected": 3.0,
    "facial_boundary_blending_seam":    2.0,
    # Manipulation / compositing signals
    "facial_texture_over_smoothness":   2.0,
    "temporal_motion_anomaly":          1.5,
    "compression_block_inconsistency":  1.5,
    "channel_correlation_anomaly":      1.5,
    "illumination_inconsistency":       1.0,
    "facial_geometry_instability":      1.5,
    "abnormal_sharpness_discrepancy":   1.0,
    # Gemini visual anomaly (AI-observed qualitative observations)
    "gemini_observed_lighting":         0.8,
    "gemini_observed_anatomy":          1.0,
    "gemini_observed_edges":            0.8,
    "gemini_observed_physics":          0.8,
    "gemini_observed_temporal_consistency": 1.2,
    "gemini_observed_visual_anomaly":   0.8,
    "gemini_synthesis_indicator":       1.5,
}
_DEFAULT_WEIGHT = 0.5


def _indicator_weight(indicator_name: str) -> float:
    name = (indicator_name or "").lower().strip()
    # Skip AI-gen-only indicators — they are no longer part of deepfake scoring
    if "ai_generated_synthetic_frame" in name or "suppressed_sensor_noise" in name:
        return 0.0
    for key, w in _INDICATOR_WEIGHTS.items():
        if key in name:
            return w
    return _DEFAULT_WEIGHT


def _score_findings(
    findings: List[Dict[str, Any]],
    gemini_data: Dict[str, Any],
) -> Tuple[float, int, List[str]]:
    """Accumulates a weighted evidence score from deepfake/manipulation signals only."""
    total_score = 0.0
    manip_count = 0
    basis: List[str] = []

    for f in findings:
        ind = (f.get("indicator_name") or "").lower()
        w = _indicator_weight(ind)
        if w == 0.0:
            continue  # skip disabled indicators
        total_score += w
        manip_count += 1
        ts = f.get("timestamp")
        ts_str = f" at {ts:.2f}s" if ts is not None else ""
        basis.append(
            f"[Signal ×{w:.1f}] {f.get('description', '')}{ts_str}"
        )

    # Gemini visual anomaly observations
    gemini_anomaly_count = 0
    for obs in gemini_data.get("visual_observations", []):
        if obs.get("potential_anomaly_detected"):
            gemini_anomaly_count += 1
            ind_name = f"gemini_observed_{obs.get('aspect', 'visual_anomaly')}"
            w = _indicator_weight(ind_name)
            total_score += w
            basis.append(
                f"[Gemini observation ×{w:.1f}] {obs.get('observation', '')} "
                f"(aspect: {obs.get('aspect', 'unknown')})"
            )

    # Gemini synthesis indicators
    for gen_ind in gemini_data.get("ai_synthesis_indicators", []):
        if gen_ind and len(gen_ind.strip()) > 5:
            w = _indicator_weight("gemini_synthesis_indicator")
            total_score += w
            basis.append(f"[Gemini synthesis ×{w:.1f}] {gen_ind.strip()}")

    return total_score, manip_count, basis


def determine_assessment(
    findings: List[Dict[str, Any]],
    signals: List[Dict[str, Any]],
    gemini_data: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes an honest deepfake assessment with a clean 3-verdict output.

    Verdicts:
        DEEPFAKE DETECTED          — meaningful manipulation evidence from neural models
        NO DEEPFAKE DETECTED       — completed examination, no meaningful indicators
        INCONCLUSIVE               — borderline scores or insufficient face data
    """
    gemini = gemini_data or {}

    total_score, manip_count, basis = _score_findings(findings, gemini)
    total_indicators = len([f for f in findings if _indicator_weight((f.get("indicator_name") or "").lower()) > 0])
    high_severity_count = sum(1 for f in findings if f.get("severity") == "high")

    # Extract deepfake neural inference signal
    deepfake_sig = next(
        (s for s in signals if s.get("signal_name") == "deepfake_neural_inference"),
        {},
    )
    frames_with_faces = deepfake_sig.get("frames_with_faces", 0)
    total_faces_detected = deepfake_sig.get("total_faces_detected", 0)
    manip_score_pct = deepfake_sig.get("mean_manipulation_pct")
    original_score_pct = deepfake_sig.get("mean_original_pct")
    max_manip_score = deepfake_sig.get("max_manipulation_score", 0.0) or 0.0
    suspicious_face_frames = deepfake_sig.get("suspicious_face_frames_count", 0)
    model_name = deepfake_sig.get("model_name", "GenD CLIP ViT-L/14")
    face_detector = deepfake_sig.get("face_detector", "YuNet CNN Face Detector")

    has_face_forgery_cue = any(
        f.get("indicator_name") in ("facial_deepfake_forgery_detected", "facial_boundary_blending_seam")
        for f in findings
    )

    reliability = "High"
    reliability_notes: List[str] = []

    # -------------------------------------------------------------------------
    # DECISION ENGINE: STRICT BINARY VERDICTS (DEEPFAKE vs NOT DEEPFAKE)
    # Never contradicts the displayed model scores.
    # If evidence is sparse or borderline, reliability is clearly marked as Low.
    # -------------------------------------------------------------------------
    if frames_with_faces > 0 and manip_score_pct is not None:
        # Deepfake classification threshold: primary neural model manipulation probability >= 55.0%
        # Or borderline (>= 50.0%) with corroborated facial boundary seam / high-confidence forgery cues
        if manip_score_pct >= 55.0 or (manip_score_pct >= 50.0 and has_face_forgery_cue and total_score >= 3.0):
            classification = CLASS_POTENTIALLY_MANIPULATED
            verdict = VERDICT_DEEPFAKE

            if manip_score_pct >= 60.0 or (has_face_forgery_cue and total_score >= 3.0):
                reliability = "High" if frames_with_faces >= 5 else "Moderate"
                reliability_notes.append(
                    f"Strong manipulation evidence supported by neural models ({manip_score_pct:.1f}%) "
                    f"across {frames_with_faces} tracked face frames."
                )
            else:
                reliability = "Low"
                reliability_notes.append(
                    f"Borderline manipulation probability ({manip_score_pct:.1f}% vs {original_score_pct:.1f}% authentic). "
                    "Classified as DEEPFAKE based on best-supported probability."
                )

            if frames_with_faces < 4:
                reliability = "Low"
                reliability_notes.append(
                    f"Sparse face tracking: only {frames_with_faces} frame(s) with detected faces."
                )

            basis.insert(
                0,
                f"Neural face forgery detector ({model_name}) identified manipulation probability of {manip_score_pct:.1f}% "
                f"(authenticity: {original_score_pct:.1f}%) across {frames_with_faces} face frame(s).",
            )

        else:
            # manip_score_pct <= 50.0% -> NOT DEEPFAKE
            classification = CLASS_NO_MANIPULATION
            verdict = VERDICT_NOT_DEEPFAKE

            if manip_score_pct <= 45.0 and total_score < VERDICT_SUSPICIOUS_THRESHOLD:
                reliability = "High" if frames_with_faces >= 5 else "Moderate"
                reliability_notes.append(
                    f"Consistent authenticity probability ({original_score_pct:.1f}%) across {frames_with_faces} face frame(s) "
                    "with zero anomalous boundary seams."
                )
            else:
                reliability = "Low"
                reliability_notes.append(
                    f"Borderline score ({original_score_pct:.1f}% authentic vs {manip_score_pct:.1f}% manipulation). "
                    "Classified as NOT DEEPFAKE based on best-supported probability."
                )

            if frames_with_faces < 4:
                reliability = "Low"
                reliability_notes.append(
                    f"Sparse face tracking: only {frames_with_faces} frame(s) with detected faces."
                )

            basis.insert(
                0,
                f"Neural face forgery detector confirmed authenticity probability of {original_score_pct:.1f}% "
                f"(manipulation: {manip_score_pct:.1f}%) across {frames_with_faces} face frame(s).",
            )

    else:
        # No human faces detected across video frames
        if total_score >= 2.0:
            classification = CLASS_POTENTIALLY_MANIPULATED
            verdict = VERDICT_DEEPFAKE
            reliability = "Low"
            reliability_notes.append(
                f"No human faces detected in video. Verdict derived from non-facial forensic signals (evidence score {total_score:.1f}). "
                "Low reliability for facial deepfake assessment."
            )
            basis.insert(0, f"Classified as DEEPFAKE based on secondary forensic signal anomalies (score: {total_score:.1f}).")
        else:
            classification = CLASS_NO_MANIPULATION
            verdict = VERDICT_NOT_DEEPFAKE
            reliability = "Low"
            reliability_notes.append(
                "No human faces detected in video frames. Facial deepfake models could not be executed. "
                "Classified as NOT DEEPFAKE due to absence of anomalous signals, but assessment has low reliability."
            )
            basis.insert(0, "No human faces detected in examined frames. No anomalous forensic signals observed.")

    return {
        "classification": classification,
        "verdict": verdict,
        "reliability": reliability,
        "reliability_note": " ".join(reliability_notes),
        "evidence_score": round(total_score, 2),
        "manipulation_score_pct": manip_score_pct,
        "original_score_pct": original_score_pct,
        # AI-gen fields kept as None — engine is disabled
        "ai_generation_score_pct": None,
        "ai_peak_score_pct": None,
        "ai_authentic_score_pct": None,
        "ai_flagged_frames_count": 0,
        "ai_model_name": "disabled",
        "ai_model_status": "disabled",
        "ai_frame_predictions": [],
        "model_name": model_name,
        "face_detector": face_detector,
        "total_faces_detected": total_faces_detected,
        "frames_with_faces": frames_with_faces,
        "total_frames_examined": len(metadata.get("keyframes", [])) or deepfake_sig.get("total_frames_examined", 0),
        "suspicious_frames_count": suspicious_face_frames or sum(1 for f in findings if f.get("frame_reference")),
        "confidence": None,
        "basis": basis,
        "what_would_change_verdict": [],
        "gend_status": deepfake_sig.get("gend_status"),
        "gend_model_name": deepfake_sig.get("gend_model_name"),
        "gend_mean_fake_pct": deepfake_sig.get("gend_mean_fake_pct"),
        "gend_max_fake_pct": deepfake_sig.get("gend_max_fake_pct"),
        "gend_mean_real_pct": deepfake_sig.get("gend_mean_real_pct"),
        "mesonet_mean_fake_pct": deepfake_sig.get("mesonet_mean_fake_pct"),
    }


def _count_unique_methods(findings: List[Dict[str, Any]]) -> int:
    """Counts how many distinct forensic methods contributed findings."""
    methods = {f.get("method_used", "") for f in findings if f.get("method_used")}
    return len(methods)


def _build_insufficient_guidance(
    score: float,
    findings: List[Dict[str, Any]],
    gemini_data: Dict[str, Any],
    is_face_video: bool = False,
) -> List[str]:
    """Explains what additional evidence would change the verdict."""
    gaps: List[str] = []
    detected_methods = {f.get("method_used", "") for f in findings}

    if is_face_video:
        gaps.append(
            "Facial forgery detector produced an intermediate score without multi-frame corroboration. "
            "A second independent signal (such as facial boundary seams, texture smoothing, or temporal jitter) is needed."
        )
    else:
        gaps.append(
            "No human faces were detected across examined frames. Face forgery detection (MesoNet-4) could not be executed."
        )

    if "high_pass_noise_residual" not in detected_methods:
        gaps.append(
            "High-frequency sensor pattern noise (PRNU) did not show significant suppression."
        )
    if "dct_block_boundary_energy" not in detected_methods:
        gaps.append(
            "DCT compression artifact analysis showed consistent compression across frames."
        )
    if not gemini_data.get("available"):
        gaps.append(
            "Google Gemini AI video understanding was not configured. Connecting Gemini may surface additional qualitative visual observations."
        )
    if score > 0:
        remaining = VERDICT_SUSPICIOUS_THRESHOLD - score
        gaps.append(
            f"Current evidence score ({score:.1f}) is {remaining:.1f} points below the minimum "
            f"corroborated threshold ({VERDICT_SUSPICIOUS_THRESHOLD:.1f})."
        )
    return gaps


def build_evidence_record(
    evidence_id: str,
    metadata: Dict[str, Any],
    findings: List[Dict[str, Any]],
    signals: List[Dict[str, Any]],
    source_matching_features: Dict[str, Any],
    gemini_data: Optional[Dict[str, Any]] = None,
    video_relationships: Optional[List[Dict[str, Any]]] = None,
    errors: Optional[List[str]] = None,
    status: str = "completed",
) -> Dict[str, Any]:
    """Builds the canonical TruthLens Stage 2 Video Forensic output JSON contract."""
    gemini = gemini_data or {}
    errs = errors or []
    relationships = video_relationships or []

    # Determine assessment
    if status == "failed":
        assessment = {
            "classification": CLASS_ANALYSIS_FAILED,
            "verdict": "ANALYSIS FAILED",
            "evidence_score": 0.0,
            "manipulation_score_pct": None,
            "original_score_pct": None,
            "confidence": None,
            "basis": [f"Analysis could not be completed: {', '.join(errs)}"],
            "what_would_change_verdict": [],
        }
    else:
        assessment = determine_assessment(findings, signals, gemini, metadata)

    # Standard limitations list
    limitations = [
        "Forensic classification represents automated signal measurements and neural inference scores; not judicial proof.",
        "Keyframe sampling inspects periodic intervals across the video rather than 100% of video frames.",
    ]
    if gemini.get("limitations"):
        limitations.extend(gemini["limitations"])
    elif not gemini.get("available"):
        limitations.append("Google Gemini API understanding was not active (API key missing or request bypassed).")

    # Extract deepfake metrics signal for easy top-level consumption
    deepfake_sig = next(
        (s for s in signals if s.get("signal_name") == "deepfake_neural_inference"),
        {},
    )

    record: Dict[str, Any] = {
        "evidence_id": evidence_id,
        "media_type": "video",
        "status": status,
        "metadata": metadata,
        "assessment": assessment,
        "verdict": assessment.get("verdict"),
        "deepfake_metrics": deepfake_sig,
        "findings": findings,
        "signals": signals,
        "source_matching_features": source_matching_features,
        "video_relationships": relationships,
        "errors": errs,
        "limitations": limitations,
    }

    # Add Gemini contextual summary if available
    if gemini.get("summary"):
        record["ai_context_summary"] = gemini["summary"]
    if gemini.get("events_described"):
        record["ai_described_events"] = gemini["events_described"]

    return record


def export_record_to_json(record: Dict[str, Any], file_path: Path | str) -> Path:
    """Exports structured record to JSON file for Stage 3 Correlation integration."""
    target_path = Path(file_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return target_path
