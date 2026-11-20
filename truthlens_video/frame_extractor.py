"""TruthLens Video Forensics - Frame Extractor

Extracts frames at intelligent intervals using OpenCV with FFmpeg backend,
tracks timestamps, detects scene cuts, identifies duplicated/frozen frames,
and persists keyframe artifacts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from truthlens_video.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_FRAME_SAMPLE_RATE_SEC,
    DUPLICATE_FRAME_THRESHOLD,
    MAX_FRAMES_PER_VIDEO,
    MIN_FRAMES_PER_VIDEO,
    SCENE_CHANGE_THRESHOLD,
)


@dataclass
class ExtractedFrame:
    """Represents an extracted video frame with forensic metadata."""
    frame_index: int
    timestamp_seconds: float
    relative_path: str
    absolute_path: str
    diff_score_from_previous: float = 0.0
    is_scene_change: bool = False
    is_duplicate: bool = False
    laplacian_variance: float = 0.0
    is_suspicious: bool = False
    suspicion_reasons: List[str] = field(default_factory=list)


@dataclass
class FrameExtractionResult:
    """Summary of the frame extraction process."""
    frames: List[ExtractedFrame]
    scene_changes: List[float]
    duplicate_frames: List[float]
    total_frames_in_video: int
    fps: float
    sampling_step: int
    output_dir: str


def compute_frame_difference(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes normalized mean absolute difference between two BGR frames."""
    if img1 is None or img2 is None or img1.shape != img2.shape:
        return 0.0
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    return float(np.mean(diff)) / 255.0


def compute_laplacian_variance(img_bgr: np.ndarray) -> float:
    """Computes Laplacian variance (measure of edge energy / blurriness)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(
    video_path: Path | str,
    evidence_id: str,
    sample_rate_sec: float = DEFAULT_FRAME_SAMPLE_RATE_SEC,
    max_frames: int = MAX_FRAMES_PER_VIDEO,
    output_dir: Optional[Path | str] = None,
) -> FrameExtractionResult:
    """Extracts keyframes at suitable intervals, detecting scene cuts and frozen frames.
    
    Args:
        video_path: Path to the video file.
        evidence_id: Identifier for the evidence item (e.g. 'EVD-001').
        sample_rate_sec: Desired interval in seconds between sampled frames.
        max_frames: Upper bound on extracted frames to preserve memory/disk.
        output_dir: Directory where frame images will be written.

    Returns:
        FrameExtractionResult containing the extracted frames and summary metrics.
    """
    path = Path(video_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    # Setup output directory
    if output_dir is None:
        target_dir = DEFAULT_CACHE_DIR / evidence_id / "frames"
    else:
        target_dir = Path(output_dir) / evidence_id / "frames"
    target_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path.name}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 100  # Fallback

    # Calculate duration and target frame count spanning the entire video
    duration_sec = total_frames / fps if total_frames > 0 else 10.0

    # Intelligent adaptive sampling spanning the entire video:
    # Short videos (<= 10s): sample densely (min 18 frames or ~4 fps) so face models have sufficient samples
    # Medium videos (10s - 30s): sample 20-30 frames
    # Longer videos (> 30s): sample uniformly up to max_frames
    if duration_sec <= 10.0:
        target_count = min(max_frames, max(18, int(duration_sec * 4)))
    elif duration_sec <= 30.0:
        target_count = min(max_frames, max(22, int(duration_sec * 2)))
    else:
        target_count = min(max_frames, max(28, int(max_frames * 0.85)))

    # Compute step to span evenly across the entire duration
    if total_frames > target_count:
        step = max(1, total_frames // target_count)
    else:
        step = 1

    extracted_frames: List[ExtractedFrame] = []
    scene_changes: List[float] = []
    duplicate_frames: List[float] = []

    prev_frame_bgr: Optional[np.ndarray] = None
    curr_frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if curr_frame_idx % step == 0:
            timestamp = round(curr_frame_idx / fps, 3)
            diff_score = 0.0
            is_cut = False
            is_dup = False

            if prev_frame_bgr is not None:
                diff_score = compute_frame_difference(prev_frame_bgr, frame_bgr)
                if diff_score >= SCENE_CHANGE_THRESHOLD:
                    is_cut = True
                    scene_changes.append(timestamp)
                elif diff_score <= DUPLICATE_FRAME_THRESHOLD:
                    is_dup = True
                    duplicate_frames.append(timestamp)

            lap_var = compute_laplacian_variance(frame_bgr)

            frame_filename = f"frame_{curr_frame_idx:06d}_{timestamp:.2f}s.jpg"
            frame_file_path = target_dir / frame_filename

            # Save JPEG artifact
            cv2.imwrite(str(frame_file_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            # Relative path from project root for portability
            try:
                rel_path = str(frame_file_path.relative_to(Path.cwd()))
            except Exception:
                rel_path = str(frame_file_path)

            extracted_frame = ExtractedFrame(
                frame_index=curr_frame_idx,
                timestamp_seconds=timestamp,
                relative_path=rel_path,
                absolute_path=str(frame_file_path),
                diff_score_from_previous=round(diff_score, 4),
                is_scene_change=is_cut,
                is_duplicate=is_dup,
                laplacian_variance=round(lap_var, 2),
                is_suspicious=False,
            )
            extracted_frames.append(extracted_frame)
            prev_frame_bgr = frame_bgr.copy()

            if len(extracted_frames) >= max_frames:
                break

        curr_frame_idx += 1

    cap.release()

    return FrameExtractionResult(
        frames=extracted_frames,
        scene_changes=scene_changes,
        duplicate_frames=duplicate_frames,
        total_frames_in_video=total_frames,
        fps=round(fps, 2),
        sampling_step=step,
        output_dir=str(target_dir),
    )
