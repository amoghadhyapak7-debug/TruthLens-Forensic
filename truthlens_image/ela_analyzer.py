"""TruthLens Image Forensics - Error Level Analysis (ELA)

Implements Error Level Analysis (ELA) to detect compression artifact discrepancies,
cloning, splicing, and localized digital manipulation in image evidence.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from truthlens_image.config import (
    DEFAULT_CACHE_DIR,
    ELA_ANOMALY_THRESHOLD,
    ELA_JPEG_QUALITY,
    ELA_SCALE_MULTIPLIER,
)


def perform_ela(
    image_path: Path | str,
    evidence_id: str,
    quality: int = ELA_JPEG_QUALITY,
    multiplier: float = ELA_SCALE_MULTIPLIER,
    output_dir: Optional[Path | str] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Executes Error Level Analysis on an image file.

    Returns:
        metrics: Dictionary with ELA statistical metrics and anomaly detection status.
        artifact_path: Relative path to the generated ELA heatmap image.
    """
    path = Path(image_path).resolve()
    target_dir = Path(output_dir or DEFAULT_CACHE_DIR / "image_frames")
    target_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(path) as orig_img:
        # Convert to RGB to ensure uniform color space
        rgb_orig = orig_img.convert("RGB")

        # Save to in-memory buffer at specified JPEG quality
        buffer = io.BytesIO()
        rgb_orig.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)

        # Reload compressed version
        resaved = Image.open(buffer)

        # Compute difference
        diff = ImageChops.difference(rgb_orig, resaved)

        # Calculate raw statistics
        diff_arr = np.array(diff, dtype=np.float32)
        mean_diff = float(np.mean(diff_arr))
        max_diff = float(np.max(diff_arr))
        std_diff = float(np.std(diff_arr))

        # Check for localized ELA hotspots (splicing or modification)
        p95_diff = float(np.percentile(diff_arr, 95))
        ratio_max_to_mean = (max_diff / (mean_diff + 0.01)) if mean_diff >= 0 else 0.0
        anomaly_detected = (
            (p95_diff > ELA_ANOMALY_THRESHOLD)
            or (std_diff > 10.0)
            or (max_diff > 25.0 and ratio_max_to_mean > 15.0)
        )

        # Scale difference for investigator visualization
        extrema = diff.getextrema()
        max_val = max([ex[1] for ex in extrema]) if extrema else 1
        scale = 255.0 / max_val if max_val > 0 else 1.0
        diff_scaled = ImageEnhance.Brightness(diff).enhance(scale * 1.5)

        # Save artifact
        ela_filename = f"ela_{evidence_id}.jpg"
        ela_file_path = target_dir / ela_filename
        diff_scaled.save(ela_file_path, "JPEG")

    try:
        rel_path = str(ela_file_path.relative_to(Path.cwd()))
    except Exception:
        rel_path = str(ela_file_path)

    metrics = {
        "mean_error_level": round(mean_diff, 2),
        "max_error_level": round(max_diff, 2),
        "std_dev_error_level": round(std_diff, 2),
        "p95_error_level": round(p95_diff, 2),
        "ela_anomaly_detected": anomaly_detected,
        "compression_quality_used": quality,
    }

    return metrics, rel_path
