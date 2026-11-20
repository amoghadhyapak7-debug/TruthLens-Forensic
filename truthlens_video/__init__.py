"""TruthLens Video Forensics Module

Core analysis module for Stage 2 (ANALYZE) of the TruthLens Digital Forensics Platform.
Exports independent, reusable Python APIs for single and multi-video forensic processing.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from truthlens_video.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_FRAME_SAMPLE_RATE_SEC,
    MAX_FRAMES_PER_VIDEO,
    VERDICT_DEEPFAKE,
    VERDICT_NOT_DEEPFAKE,
    VERDICT_DEEPFAKE_DETECTED,
    VERDICT_NO_DEEPFAKE_DETECTED,
    VERDICT_INCONCLUSIVE,
    VERDICT_INSUFFICIENT,
    VERDICT_LIKELY_AI_GENERATED,
    VERDICT_LIKELY_DEEPFAKE,
    VERDICT_NO_MANIPULATION,
    VERDICT_ORIGINAL,
    VERDICT_POSSIBLE_MANIPULATION,
)
from truthlens_video.deepfake_detector import DeepfakeDetectionEngine
from truthlens_video.gend_detector import GenDInferenceEngine, GenDModel
from truthlens_video.ai_generator_detector import AIGeneratorDetectorEngine
from truthlens_video.metadata_extractor import extract_video_metadata
from truthlens_video.frame_extractor import extract_frames
from truthlens_video.forensic_engine import VisualForensicEngine
from truthlens_video.gemini_analyzer import GeminiVideoAnalyzer
from truthlens_video.comparator import (
    extract_video_fingerprints,
    compare_two_videos,
    analyze_video_corpus,
)
from truthlens_video.correlation_exporter import (
    build_evidence_record,
    export_record_to_json,
)

logger = logging.getLogger(__name__)


def analyze_video(
    file_path: str | Path,
    evidence_id: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyzes a single video file for forensic manipulation and AI generation indicators.

    Args:
        file_path: Path to the video file (.mp4, .mov, .avi, .mkv).
        evidence_id: Optional identifier (e.g. 'EVD-001'). If omitted, generated automatically.
        options: Optional configuration overrides (e.g. enable_gemini, sample_rate_sec).

    Returns:
        Structured forensic result dictionary compatible with the TruthLens Correlation Engine.
    """
    opts = options or {}
    path = Path(file_path).resolve()
    ev_id = evidence_id or f"EVD-{path.stem[:8].upper()}"

    # Handle file non-existence or open errors gracefully
    try:
        # Step 1: Metadata Extraction
        metadata = extract_video_metadata(path)
        metadata["file_path"] = str(path)
    except Exception as e:
        logger.error(f"Failed to extract metadata for {path}: {e}")
        failed_record = build_evidence_record(
            evidence_id=ev_id,
            metadata={"file_name": path.name, "file_path": str(path)},
            findings=[],
            signals=[],
            source_matching_features={},
            gemini_data=None,
            video_relationships=[],
            errors=[f"Metadata extraction error: {str(e)}"],
            status="failed",
        )
        return failed_record

    try:
        # Step 2: Frame Extraction
        sample_rate = float(opts.get("sample_rate_sec", DEFAULT_FRAME_SAMPLE_RATE_SEC))
        max_frames = int(opts.get("max_frames", MAX_FRAMES_PER_VIDEO))
        cache_dir = opts.get("cache_dir", DEFAULT_CACHE_DIR)

        frame_result = extract_frames(
            video_path=path,
            evidence_id=ev_id,
            sample_rate_sec=sample_rate,
            max_frames=max_frames,
            output_dir=cache_dir,
        )

        # Step 3: Visual Forensic Analysis (Traditional CV & Signal Processing)
        forensic_engine = VisualForensicEngine.get_instance()
        findings, signals, metrics = forensic_engine.analyze_frames(
            frames=frame_result.frames,
            fps=frame_result.fps,
        )

        # Step 4: Perceptual Fingerprints for Multi-Video Matching
        source_features = extract_video_fingerprints(
            evidence_id=ev_id,
            frames=frame_result.frames,
            metadata=metadata,
            frame_images=forensic_engine.last_frame_images,
        )

        # Step 5: AI-Assisted Video Understanding (Google Gemini API)
        enable_gemini = opts.get("enable_gemini", True)
        gemini_data = None
        if enable_gemini:
            gemini_analyzer = GeminiVideoAnalyzer(
                api_key=opts.get("gemini_api_key"),
                model_name=opts.get("gemini_model"),
            )
            gemini_data = gemini_analyzer.analyze_video_context(
                frames=frame_result.frames,
                video_metadata=metadata,
            )
            # Merge Gemini findings into overall findings
            if gemini_data.get("ai_findings"):
                findings.extend(gemini_data["ai_findings"])

        # Step 6: Construct Canonical Evidence Record for Correlation Engine
        evidence_record = build_evidence_record(
            evidence_id=ev_id,
            metadata=metadata,
            findings=findings,
            signals=signals,
            source_matching_features=source_features,
            gemini_data=gemini_data,
            video_relationships=[],
            errors=[],
            status="completed",
        )

        # Add frame artifacts metadata to evidence record for frontend/review
        evidence_record["extracted_frames_summary"] = {
            "total_extracted": len(frame_result.frames),
            "output_directory": frame_result.output_dir,
            "scene_changes_timestamps": frame_result.scene_changes,
            "duplicate_frames_timestamps": frame_result.duplicate_frames,
            "keyframe_paths": [f.relative_path for f in frame_result.frames],
            "keyframes": [
                {
                    "frame_index": f.frame_index,
                    "timestamp": f.timestamp_seconds,
                    "relative_path": f.relative_path,
                    "absolute_path": f.absolute_path,
                    "file_name": Path(f.absolute_path).name,
                    "is_scene_change": f.is_scene_change,
                    "is_duplicate": f.is_duplicate,
                    "laplacian_variance": f.laplacian_variance,
                    "is_suspicious": f.is_suspicious,
                    "suspicion_reasons": f.suspicion_reasons,
                }
                for f in frame_result.frames
            ],
        }

        # Persist structured JSON artifact
        json_out_path = Path(frame_result.output_dir).parent / "forensic_report.json"
        export_record_to_json(evidence_record, json_out_path)

        return evidence_record

    except Exception as e:
        logger.error(f"Forensic pipeline error processing {path}: {e}", exc_info=True)
        return build_evidence_record(
            evidence_id=ev_id,
            metadata=metadata,
            findings=[],
            signals=[],
            source_matching_features={},
            gemini_data=None,
            video_relationships=[],
            errors=[f"Forensic pipeline error: {str(e)}"],
            status="failed",
        )


def analyze_multiple_videos(
    file_paths: List[str | Path],
    evidence_ids: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None,
    compare: bool = True,
) -> Dict[str, Any]:
    """Batch-processes multiple uploaded videos and computes cross-video comparisons.

    Args:
        file_paths: List of paths to video files.
        evidence_ids: Optional list of corresponding evidence IDs.
        options: Optional configuration overrides.
        compare: If True, performs cross-video similarity and cluster comparison.

    Returns:
        Dictionary containing individual results and cross-video comparison insights.
    """
    individual_results: List[Dict[str, Any]] = [None] * len(file_paths)  # type: ignore[list-item]

    def _process_one(i: int, path_item: "str | Path") -> tuple[int, Dict[str, Any]]:
        p = Path(path_item)
        ev_id = evidence_ids[i] if (evidence_ids and i < len(evidence_ids)) else f"EVD-{i+1:03d}"
        try:
            result = analyze_video(p, evidence_id=ev_id, options=options)
        except Exception as exc:
            logger.error(f"Unhandled error processing video {p}: {exc}", exc_info=True)
            result = {
                "status": "failed",
                "metadata": {"file_name": p.name, "file_path": str(p)},
                "assessment": {},
                "findings": [],
                "signals": [],
                "source_matching_features": {},
                "errors": [f"Unhandled pipeline error: {exc}"],
            }
        return i, result

    # max_workers=2: GenD ViT-L/14 is memory-intensive (~1.2 GB); keeping concurrency low
    # prevents OOM while still overlapping I/O-bound frame extraction with model inference.
    max_workers = min(2, len(file_paths))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_one, i, p): i for i, p in enumerate(file_paths)}
        for future in as_completed(futures):
            try:
                idx, res = future.result()
                individual_results[idx] = res
            except Exception as exc:
                i = futures[future]
                logger.error(f"Future error for video index {i}: {exc}", exc_info=True)
                individual_results[i] = {
                    "status": "failed",
                    "metadata": {},
                    "assessment": {},
                    "findings": [],
                    "signals": [],
                    "source_matching_features": {},
                    "errors": [f"Future error: {exc}"],
                }

    corpus_comparison = {}
    if compare and len(individual_results) > 1:
        # Only compare successfully completed videos
        completed_videos = [v for v in individual_results if v and v.get("status") == "completed"]
        if len(completed_videos) > 1:
            corpus_comparison = analyze_video_corpus(completed_videos)

    return {
        "status": "completed",
        "total_videos_processed": len(individual_results),
        "successful_analyses": sum(1 for v in individual_results if v and v.get("status") == "completed"),
        "failed_analyses": sum(1 for v in individual_results if v and v.get("status") == "failed"),
        "videos": individual_results,
        "corpus_comparison": corpus_comparison,
    }


def compare_videos(video_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compares pre-analyzed video evidence records for cross-video correlation."""
    return analyze_video_corpus(video_records)


__all__ = [
    "analyze_video",
    "analyze_multiple_videos",
    "compare_videos",
    "extract_video_metadata",
    "extract_frames",
    "VisualForensicEngine",
    "GeminiVideoAnalyzer",
    "DeepfakeDetectionEngine",
    "GenDInferenceEngine",
    "GenDModel",
    "AIGeneratorDetectorEngine",
    "VERDICT_LIKELY_AI_GENERATED",
    "VERDICT_LIKELY_DEEPFAKE",
    "VERDICT_POSSIBLE_MANIPULATION",
    "VERDICT_NO_MANIPULATION",
    "VERDICT_INCONCLUSIVE",
    "VERDICT_DEEPFAKE",
    "VERDICT_NOT_DEEPFAKE",
    "VERDICT_DEEPFAKE_DETECTED",
    "VERDICT_NO_DEEPFAKE_DETECTED",
    "VERDICT_ORIGINAL",
    "VERDICT_INCONCLUSIVE",
    "VERDICT_INSUFFICIENT",
]
