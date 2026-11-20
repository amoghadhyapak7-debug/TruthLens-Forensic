"""TruthLens Video Forensics - Metadata Extractor

Extracts container, stream, and file-level metadata including SHA-256 checksums,
codecs, resolutions, frame rates, and structural integrity checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

import cv2
import imageio_ffmpeg

from truthlens_video.config import SUPPORTED_EXTENSIONS


def compute_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Computes SHA-256 cryptographic hash of a file in chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def format_duration(seconds: float) -> str:
    """Formats seconds into mm:ss or hh:mm:ss string."""
    if seconds < 0:
        return "00:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_ffmpeg_metadata(file_path: Path | str) -> Dict[str, Any]:
    """Extracts container and stream metadata using bundled FFmpeg executable if available."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not ffmpeg_exe or not os.path.exists(ffmpeg_exe):
        return {}

    try:
        # Run ffmpeg -i <file> to get container metadata from stderr
        cmd = [ffmpeg_exe, "-hide_banner", "-i", str(file_path)]
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        stderr_output = process.stderr or ""
        
        metadata: Dict[str, Any] = {
            "ffmpeg_raw_header": [],
            "streams_detected": [],
            "encoder": None,
            "creation_time": None,
        }

        for line in stderr_output.splitlines():
            line_str = line.strip()
            if "Stream #" in line_str:
                metadata["streams_detected"].append(line_str)
            elif "encoder" in line_str.lower() and not metadata["encoder"]:
                metadata["encoder"] = line_str.split(":", 1)[-1].strip()
            elif "creation_time" in line_str.lower() and not metadata["creation_time"]:
                metadata["creation_time"] = line_str.split(":", 1)[-1].strip()

        return metadata
    except Exception as e:
        return {"error": f"FFmpeg probe failed: {str(e)}"}


def extract_video_metadata(file_path: Path | str) -> Dict[str, Any]:
    """Extracts comprehensive metadata for a video file.
    
    Returns a dictionary containing:
        - file_name, file_extension, file_size_bytes, file_size_mb
        - sha256_hash
        - resolution (width, height, aspect_ratio)
        - fps, total_frames, duration_seconds, duration_formatted
        - codec (FourCC and FFmpeg stream codec)
        - stream_details and container_tags
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    file_size_bytes = path.stat().st_size
    if file_size_bytes == 0:
        raise ValueError(f"Video file is empty (0 bytes): {path.name}")

    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    sha256_hash = compute_sha256(path)

    # OpenCV video capture
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(
            f"Could not open video file '{path.name}'. The file may be corrupt or encoded in an unsupported codec."
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))

    # FourCC decode
    try:
        fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    except Exception:
        fourcc = "unknown"

    cap.release()

    # Integrity verification
    if fps <= 0 or total_frames <= 0:
        # Fallback calculation if OpenCV cannot determine frame rate
        fps = fps if fps > 0 else 24.0
        duration_seconds = 0.0
    else:
        duration_seconds = round(total_frames / fps, 2)

    aspect_ratio = f"{width}:{height}" if height > 0 else "unknown"

    # FFmpeg deeper inspection
    ffmpeg_meta = get_ffmpeg_metadata(path)

    return {
        "file_name": path.name,
        "file_extension": ext,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "sha256": sha256_hash,
        "resolution": {
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
        },
        "frame_rate": round(fps, 2),
        "total_frames": total_frames,
        "duration_seconds": duration_seconds,
        "duration_formatted": format_duration(duration_seconds),
        "codec_fourcc": fourcc.strip(),
        "streams": ffmpeg_meta.get("streams_detected", []),
        "encoder": ffmpeg_meta.get("encoder"),
        "creation_time": ffmpeg_meta.get("creation_time"),
    }
