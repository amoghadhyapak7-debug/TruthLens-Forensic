"""TruthLens Image Forensics Module

Core analysis module for static image evidence in TruthLens.
Provides independent, batch-capable APIs for single and multi-image forensic processing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from truthlens_image.config import (
    DEFAULT_CACHE_DIR,
    MAX_IMAGE_UPLOAD_SIZE_BYTES,
)
from truthlens_image.metadata_extractor import extract_image_metadata
from truthlens_image.forensic_engine import ImageForensicEngine
from truthlens_image.gemini_image_analyzer import GeminiImageAnalyzer
from truthlens_image.correlation_exporter import build_image_evidence_record

logger = logging.getLogger(__name__)


def analyze_image(
    file_path: str | Path,
    evidence_id: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyzes a single image file for digital tampering and AI synthesis indicators.

    Args:
        file_path: Path to image (.jpg, .jpeg, .png, .webp).
        evidence_id: Unique evidence identifier (e.g. 'IMG-001').
        options: Optional configuration overrides.

    Returns:
        Structured forensic result dictionary.
    """
    opts = options or {}
    path = Path(file_path).resolve()
    ev_id = evidence_id or f"IMG-{path.stem[:8].upper()}"

    # Step 1: Metadata extraction and upload limit validation
    try:
        max_bytes = int(opts.get("max_upload_size_bytes", MAX_IMAGE_UPLOAD_SIZE_BYTES))
        metadata = extract_image_metadata(path, max_size_bytes=max_bytes)
    except Exception as e:
        logger.error(f"Image metadata/validation error for {path}: {e}")
        return build_image_evidence_record(
            evidence_id=ev_id,
            metadata={"file_name": path.name, "file_path": str(path)},
            findings=[],
            signals=[],
            source_matching_features={},
            gemini_data=None,
            errors=[f"Validation error: {str(e)}"],
            status="failed",
        )

    # Step 2: Signal Forensics (ELA, noise residuals, edge gradients)
    try:
        forensic_engine = ImageForensicEngine()
        findings, signals, fingerprints, ela_artifact = forensic_engine.analyze_image(
            path, evidence_id=ev_id
        )

        source_matching_features = {
            "evidence_id": ev_id,
            "sha256": metadata["sha256"],
            "dhash": fingerprints.get("dhash"),
            "ahash": fingerprints.get("ahash"),
            "color_vector": fingerprints.get("color_vector"),
            "ela_artifact_path": ela_artifact,
        }

        # Step 3: Google Gemini AI semantic analysis
        enable_gemini = opts.get("enable_gemini", True)
        gemini_data = None
        if enable_gemini:
            gemini_analyzer = GeminiImageAnalyzer(
                api_key=opts.get("gemini_api_key"),
                model_name=opts.get("gemini_model"),
            )
            gemini_data = gemini_analyzer.analyze_image_context(path, metadata=metadata)
            if gemini_data.get("ai_findings"):
                findings.extend(gemini_data["ai_findings"])

        # Step 4: Construct canonical evidence record
        record = build_image_evidence_record(
            evidence_id=ev_id,
            metadata=metadata,
            findings=findings,
            signals=signals,
            source_matching_features=source_matching_features,
            gemini_data=gemini_data,
            errors=[],
            status="completed",
        )
        record["ela_artifact"] = ela_artifact
        return record

    except Exception as e:
        logger.error(f"Forensic engine failure on {path}: {e}", exc_info=True)
        return build_image_evidence_record(
            evidence_id=ev_id,
            metadata=metadata,
            findings=[],
            signals=[],
            source_matching_features={},
            gemini_data=None,
            errors=[f"Forensic engine error: {str(e)}"],
            status="failed",
        )


def analyze_multiple_images(
    file_paths: List[str | Path],
    evidence_ids: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Batch-processes multiple uploaded images individually without aborting on failures.

    Args:
        file_paths: List of file paths to images.
        evidence_ids: Optional list of unique evidence IDs.
        options: Configuration dictionary.
        progress_callback: Optional callback func(current_idx, total_count, current_filename).

    Returns:
        Combined results overview with individual evidence records.
    """
    total = len(file_paths)
    individual_results: List[Dict[str, Any]] = []

    for i, p_item in enumerate(file_paths):
        p = Path(p_item)
        ev_id = evidence_ids[i] if (evidence_ids and i < len(evidence_ids)) else f"IMG-{i+1:03d}"

        if progress_callback:
            progress_callback(i + 1, total, p.name)

        try:
            res = analyze_image(p, evidence_id=ev_id, options=options)
        except Exception as e:
            res = build_image_evidence_record(
                evidence_id=ev_id,
                metadata={"file_name": p.name, "file_path": str(p)},
                findings=[],
                signals=[],
                source_matching_features={},
                errors=[f"Unexpected batch worker exception: {str(e)}"],
                status="failed",
            )
        individual_results.append(res)

    # Compile combined summary overview
    successful = sum(1 for r in individual_results if r.get("status") == "completed")
    failed = sum(1 for r in individual_results if r.get("status") == "failed")
    flagged = sum(
        1
        for r in individual_results
        if r.get("assessment", {}).get("classification")
        in ("potentially_manipulated", "potentially_ai_generated")
    )
    clean = sum(
        1
        for r in individual_results
        if r.get("assessment", {}).get("classification") == "no_manipulation_detected"
    )

    combined_overview = {
        "total_images": total,
        "successful_analyses": successful,
        "failed_analyses": failed,
        "flagged_tampered_or_ai": flagged,
        "clean_verified": clean,
        "total_findings_count": sum(len(r.get("findings", [])) for r in individual_results),
    }

    return {
        "status": "completed" if failed == 0 else ("partial" if successful > 0 else "failed"),
        "total_images_processed": total,
        "summary": combined_overview,
        "evidence_results": individual_results,
    }


__all__ = [
    "analyze_image",
    "analyze_multiple_images",
    "extract_image_metadata",
    "ImageForensicEngine",
    "GeminiImageAnalyzer",
]
