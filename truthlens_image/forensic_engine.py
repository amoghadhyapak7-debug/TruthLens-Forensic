"""TruthLens Image Forensics - Visual Forensic Engine

Implements objective, measurement-grounded signal analyses for static images:
- Error Level Analysis (compression inconsistencies)
- Spatial noise residual analysis (PRNU / generative smoothness)
- Edge energy and gradient boundary blur detection
- Perceptual image fingerprinting (dHash, aHash)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from truthlens_image.config import (
    COLOR_DISCREPANCY_THRESHOLD,
    HIGH_FREQUENCY_NOISE_THRESHOLD,
    LAPLACIAN_BLUR_THRESHOLD,
    PHASH_SIZE,
)
from truthlens_image.ela_analyzer import perform_ela


def compute_image_dhash(image_path: Path | str, hash_size: int = PHASH_SIZE) -> str:
    """Computes difference hash (dHash) for static image."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            pixels = np.array(img, dtype=np.int32)
            diff = pixels[:, 1:] > pixels[:, :-1]
            return "".join("1" if b else "0" for b in diff.flatten())
    except Exception:
        return "0" * (hash_size * hash_size)


def compute_image_ahash(image_path: Path | str, hash_size: int = PHASH_SIZE) -> str:
    """Computes average hash (aHash) for static image."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
            pixels = np.array(img, dtype=np.float32)
            avg = np.mean(pixels)
            diff = pixels >= avg
            return "".join("1" if b else "0" for b in diff.flatten())
    except Exception:
        return "0" * (hash_size * hash_size)


class ImageForensicEngine:
    """Executes forensic signal tests on an individual image evidence item."""

    def analyze_image(
        self,
        image_path: Path | str,
        evidence_id: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
        """Executes full suite of signal tests on an image file.

        Returns:
            findings: List of flagged forensic anomaly dicts.
            signals: Summary measurement dicts.
            fingerprints: Perceptual hash and color vector dict.
            ela_artifact_path: Relative path to saved ELA heatmap image.
        """
        path = Path(image_path).resolve()
        findings: List[Dict[str, Any]] = []
        signals: List[Dict[str, Any]] = []

        # 1. Error Level Analysis (ELA)
        ela_metrics, ela_artifact = perform_ela(path, evidence_id=evidence_id)
        signals.append({
            "signal_name": "error_level_analysis",
            "mean_error_level": ela_metrics["mean_error_level"],
            "max_error_level": ela_metrics["max_error_level"],
            "p95_error_level": ela_metrics["p95_error_level"],
            "anomaly_detected": ela_metrics["ela_anomaly_detected"],
        })

        if ela_metrics["ela_anomaly_detected"]:
            findings.append({
                "indicator_name": "compression_level_discrepancy",
                "description": (
                    f"Inconsistent JPEG error levels detected (P95: {ela_metrics['p95_error_level']}, "
                    f"StdDev: {ela_metrics['std_dev_error_level']}). Indicates potential localized splicing, "
                    "re-saving, or digital tampering."
                ),
                "timestamp": None,
                "frame_reference": ela_artifact,
                "severity": "medium",
                "method_used": "error_level_analysis_ela",
                "supporting_observations": f"Max error: {ela_metrics['max_error_level']}, baseline: {ela_metrics['mean_error_level']}",
            })

        # Load with OpenCV for spatial & frequency analysis
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            return findings, signals, {}, ela_artifact

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 2. High-Frequency Noise Residual Analysis
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        residual = cv2.absdiff(gray, blurred)
        overall_noise_var = float(np.var(residual))

        # Block-level noise consistency (detecting composite pasting)
        h, w = gray.shape
        grid_rows, grid_cols = 4, 4
        rh, cw = h // grid_rows, w // grid_cols
        block_variances: List[float] = []

        if rh > 10 and cw > 10:
            for r in range(grid_rows):
                for c in range(grid_cols):
                    block = residual[r * rh : (r + 1) * rh, c * cw : (c + 1) * cw]
                    block_variances.append(float(np.var(block)))

        noise_spread = float(np.std(block_variances)) if block_variances else 0.0

        signals.append({
            "signal_name": "sensor_noise_profile",
            "overall_noise_variance": round(overall_noise_var, 3),
            "block_noise_std_dev": round(noise_spread, 3),
            "noise_consistency": "natural" if overall_noise_var >= HIGH_FREQUENCY_NOISE_THRESHOLD else "unusually_smooth",
        })

        if overall_noise_var < HIGH_FREQUENCY_NOISE_THRESHOLD:
            findings.append({
                "indicator_name": "suppressed_sensor_noise",
                "description": (
                    f"Unusually low high-frequency sensor noise floor (variance {overall_noise_var:.2f}). "
                    "Natural optical sensors exhibit sensor pattern grain; flat smooth textures are characteristic of AI generation."
                ),
                "timestamp": None,
                "frame_reference": path.name,
                "severity": "low",
                "method_used": "high_pass_noise_residual",
                "supporting_observations": f"Residual variance: {overall_noise_var:.3f} (threshold: {HIGH_FREQUENCY_NOISE_THRESHOLD})",
            })
        elif noise_spread > 8.0:
            findings.append({
                "indicator_name": "localized_noise_variance_mismatch",
                "description": (
                    f"Significant divergence in sensor noise variance across image regions (std dev {noise_spread:.2f}). "
                    "May indicate composite elements sourced from different cameras or generators."
                ),
                "timestamp": None,
                "frame_reference": path.name,
                "severity": "medium",
                "method_used": "block_noise_residual_variance",
                "supporting_observations": f"Block std dev: {noise_spread:.2f}",
            })

        # 3. Edge Gradient & Blur Discrepancy
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        signals.append({
            "signal_name": "gradient_sharpness",
            "laplacian_variance": round(lap_var, 2),
        })

        if lap_var < LAPLACIAN_BLUR_THRESHOLD:
            findings.append({
                "indicator_name": "gradient_energy_attenuation",
                "description": (
                    f"Low edge gradient energy detected (Laplacian variance {lap_var:.1f}). "
                    "Image exhibits heavy smoothing, defocus blur, or synthetic softening."
                ),
                "timestamp": None,
                "frame_reference": path.name,
                "severity": "low",
                "method_used": "laplacian_edge_energy",
                "supporting_observations": f"Laplacian score: {lap_var:.1f}",
            })

        # 4. Color & Perceptual Fingerprints
        dhash_str = compute_image_dhash(path)
        ahash_str = compute_image_ahash(path)

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        color_vector = hist.flatten().tolist()

        fingerprints = {
            "dhash": dhash_str,
            "ahash": ahash_str,
            "color_vector": color_vector,
            "ela_artifact_path": ela_artifact,
        }

        return findings, signals, fingerprints, ela_artifact
