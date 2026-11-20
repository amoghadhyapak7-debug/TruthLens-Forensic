"""TruthLens Image Forensics - Metadata & Integrity Extractor

Extracts image metadata, EXIF tags, dimensions, color profiles,
computes chunked SHA-256 hashes, and enforces configurable file size limits.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, ExifTags

from truthlens_image.config import (
    MAX_IMAGE_UPLOAD_SIZE_BYTES,
    MAX_IMAGE_UPLOAD_SIZE_MB,
    SUPPORTED_IMAGE_EXTENSIONS,
)


def compute_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Computes SHA-256 cryptographic hash of an image file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_exif_data(pil_img: Image.Image) -> Dict[str, Any]:
    """Extracts human-readable EXIF metadata tags from an image if present."""
    exif_data: Dict[str, Any] = {}
    try:
        raw_exif = pil_img.getexif()
        if not raw_exif:
            return exif_data

        for tag_id, value in raw_exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            # Filter out non-serializable raw bytes
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="ignore").strip()
                except Exception:
                    value = f"<bytes: {len(value)}>"
            exif_data[tag_name] = str(value)
    except Exception:
        pass
    return exif_data


def extract_image_metadata(
    file_path: Path | str,
    max_size_bytes: int = MAX_IMAGE_UPLOAD_SIZE_BYTES,
) -> Dict[str, Any]:
    """Validates and extracts metadata for an uploaded image.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If format is unsupported, file is empty, or size exceeds limit.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )

    file_size_bytes = path.stat().st_size
    if file_size_bytes == 0:
        raise ValueError(f"Image file is empty (0 bytes): {path.name}")

    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    max_mb = round(max_size_bytes / (1024 * 1024), 1)

    if file_size_bytes > max_size_bytes:
        raise ValueError(
            f"File '{path.name}' ({file_size_mb} MB) exceeds the maximum upload limit of {max_mb} MB."
        )

    sha256_hash = compute_sha256(path)

    try:
        with Image.open(path) as img:
            width, height = img.size
            format_name = img.format or ext.lstrip(".").upper()
            mode = img.mode
            exif = extract_exif_data(img)
    except Exception as e:
        raise ValueError(f"Could not read image structure for '{path.name}': {e}") from e

    aspect_ratio = f"{width}:{height}" if height > 0 else "unknown"

    # Inspect EXIF for known image editing software
    software_tag = exif.get("Software", "")
    editing_software_detected = None
    known_editors = ["photoshop", "gimp", "canva", "lightroom", "picsart", "snapseed"]
    for editor in known_editors:
        if editor in software_tag.lower():
            editing_software_detected = software_tag
            break

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
        "format": format_name,
        "color_mode": mode,
        "has_exif": bool(exif),
        "exif_summary": {
            "camera_make": exif.get("Make"),
            "camera_model": exif.get("Model"),
            "date_time": exif.get("DateTime") or exif.get("DateTimeOriginal"),
            "software": software_tag or None,
        },
        "editing_software_flagged": editing_software_detected,
    }
