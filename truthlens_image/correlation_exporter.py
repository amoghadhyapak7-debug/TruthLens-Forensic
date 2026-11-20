"""TruthLens Image Forensics - Structured Contract Exporter

Constructs and validates the structured JSON output conforming to the
TruthLens inter-module contract for Stage 3 (CORRELATE).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from truthlens_image.config import (
    CLASS_ANALYSIS_FAILED,
    CLASS_INSUFFICIENT_EVIDENCE,
    CLASS_NO_MANIPULATION,
    CLASS_POTENTIALLY_AI_GENERATED,
    CLASS_POTENTIALLY_MANIPULATED,
)


def determine_image_assessment(
    findings: List[Dict[str, Any]],
    signals: List[Dict[str, Any]],
    gemini_data: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes an honest assessment classification and documented basis for image evidence."""
    basis: List[str] = []

    ai_gen_count = 0
    tampering_count = 0
    high_severity_count = 0

    for f in findings:
        ind = f.get("indicator_name", "").lower()
        sev = f.get("severity", "low")

        if sev == "high":
            high_severity_count += 1

        if "suppressed_sensor_noise" in ind or "synthesis" in ind:
            ai_gen_count += 1
            basis.append(f"Generative signal: {f.get('description')} (method: {f.get('method_used')})")
        elif "compression" in ind or "noise_variance_mismatch" in ind:
            tampering_count += 1
            basis.append(f"Tampering indicator: {f.get('description')} (method: {f.get('method_used')})")
        else:
            basis.append(f"Visual anomaly: {f.get('description')}")

    if metadata.get("editing_software_flagged"):
        basis.append(f"EXIF metadata indicates editing software: {metadata['editing_software_flagged']}")
        tampering_count += 1

    for obs in gemini_data.get("visual_observations", []):
        if obs.get("potential_anomaly_detected"):
            basis.append(f"Gemini AI observation: {obs.get('observation')} (Aspect: {obs.get('aspect')})")

    total_anomalies = len(findings)

    if total_anomalies == 0 and not metadata.get("editing_software_flagged"):
        classification = CLASS_NO_MANIPULATION
        confidence = None
        basis.append("No compression discrepancies, abnormal noise floors, or tampering artifacts detected.")
    elif ai_gen_count >= 2 or (ai_gen_count >= 1 and total_anomalies >= 2):
        classification = CLASS_POTENTIALLY_AI_GENERATED
        confidence = None
        basis.append("Corroborating indicators of synthetic image generation detected.")
    elif tampering_count >= 1 or total_anomalies >= 2:
        classification = CLASS_POTENTIALLY_MANIPULATED
        confidence = None
        basis.append("Multiple forensic irregularities or compression discrepancies detected.")
    elif total_anomalies == 1:
        classification = CLASS_INSUFFICIENT_EVIDENCE
        confidence = None
        basis.append("Single isolated anomaly detected. Insufficient evidence for definitive classification.")
    else:
        classification = CLASS_INSUFFICIENT_EVIDENCE
        confidence = None
        basis.append("Signals were inconclusive.")

    return {
        "classification": classification,
        "confidence": confidence,
        "basis": basis,
    }


def build_image_evidence_record(
    evidence_id: str,
    metadata: Dict[str, Any],
    findings: List[Dict[str, Any]],
    signals: List[Dict[str, Any]],
    source_matching_features: Dict[str, Any],
    gemini_data: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None,
    status: str = "completed",
) -> Dict[str, Any]:
    """Builds the canonical TruthLens Stage 2 Image Forensic output JSON contract."""
    gemini = gemini_data or {}
    errs = errors or []

    if status == "failed":
        assessment = {
            "classification": CLASS_ANALYSIS_FAILED,
            "confidence": None,
            "basis": [f"Analysis failed: {', '.join(errs)}"],
        }
    else:
        assessment = determine_image_assessment(findings, signals, gemini, metadata)

    limitations = [
        "Forensic classification represents automated signal measurements and heuristic indicators; not judicial proof.",
        "Error Level Analysis is sensitive to prior JPEG re-compression generations.",
    ]
    if gemini.get("limitations"):
        limitations.extend(gemini["limitations"])
    elif not gemini.get("available"):
        limitations.append("Google Gemini API understanding was not active (API key missing or request bypassed).")

    record: Dict[str, Any] = {
        "evidence_id": evidence_id,
        "media_type": "image",
        "status": status,
        "metadata": metadata,
        "assessment": assessment,
        "findings": findings,
        "signals": signals,
        "source_matching_features": source_matching_features,
        "errors": errs,
        "limitations": limitations,
    }

    if gemini.get("summary"):
        record["ai_context_summary"] = gemini["summary"]

    return record
