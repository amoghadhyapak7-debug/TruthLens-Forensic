"""
Error Level Analysis (ELA) Analyzer
Detects differing compression levels resulting from digital splicing, localized editing,
or multi-source compositing. Exports amplified ELA heatmap visualization artifact.
"""
import io
import os
import traceback
import numpy as np
from typing import Dict, Any


class ELAAnalyzer:
    RESAVE_QUALITY = 95
    AMPLIFICATION = 20  # Contrast amplification multiplier

    def __init__(self, artifact_dir: str):
        self.artifact_dir = artifact_dir

    def analyze(self, image_path: str, evidence_id: str) -> Dict[str, Any]:
        result = {
            "engine": "ela_analysis",
            "resave_quality": self.RESAVE_QUALITY,
            "amplification_factor": self.AMPLIFICATION,
            "mean_error": 0.0,
            "max_error": 0.0,
            "std_error": 0.0,
            "high_ela_fraction": 0.0,
            "anomaly_regions": [],
            "artifact": None,
            "artifacts": [],
            "interpretation": "",
            "limitations": [
                "ELA is calibrated for JPEG compression and is most informative on images with natural compression cycles.",
                "Multiple uniform re-saves across the entire image reduce ELA sensitivity.",
                "High-frequency natural edges (textures, foliage) naturally produce moderate ELA signals without tampering."
            ],
        }

        try:
            from PIL import Image

            img = Image.open(image_path).convert("RGB")
            orig_arr = np.array(img, dtype=np.float32)

            # Re-save to in-memory JPEG at quality 95
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.RESAVE_QUALITY)
            buf.seek(0)
            resaved_img = Image.open(buf).convert("RGB")
            resaved_arr = np.array(resaved_img, dtype=np.float32)

            # Absolute difference
            diff_map = np.abs(orig_arr - resaved_arr)
            diff_gray = diff_map.mean(axis=2)

            # Statistical metrics
            mean_err = float(diff_gray.mean())
            max_err = float(diff_gray.max())
            std_err = float(diff_gray.std())

            result["mean_error"] = round(mean_err, 4)
            result["max_error"] = round(max_err, 4)
            result["std_error"] = round(std_err, 4)

            # High deviation threshold: requires absolute error to exceed baseline JPEG noise floor (>= 12.0)
            # and exceed 2.2 standard deviations above the image mean.
            thresh = max(12.0, mean_err + 2.2 * std_err)
            high_mask = diff_gray >= thresh
            fraction = float(np.mean(high_mask))
            result["high_ela_fraction"] = round(fraction, 4)

            # Local block analysis for anomaly bounding boxes
            h, w = diff_gray.shape
            block_size = max(16, min(h, w) // 32)
            anomalies = []
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = diff_gray[y : y + block_size, x : x + block_size]
                    blk_mean = float(block.mean())
                    if blk_mean >= thresh and blk_mean >= 15.0:
                        anomalies.append({
                            "x": x,
                            "y": y,
                            "width": block_size,
                            "height": block_size,
                            "intensity": round(blk_mean, 2),
                        })

            anomalies.sort(key=lambda b: b["intensity"], reverse=True)
            result["anomaly_regions"] = anomalies[:12]

            # Generate amplified heatmap visualization artifact
            amplified = np.clip(diff_map * self.AMPLIFICATION, 0, 255).astype(np.uint8)
            artifact_path = self._save_artifact(amplified, evidence_id)
            if artifact_path:
                art_obj = {
                    "artifact_id": f"ART-{evidence_id}-ELA",
                    "type": "ela_heatmap",
                    "file_path": artifact_path,
                    "description": "Error Level Analysis compression variance heatmap (brighter pixels indicate distinct compression history).",
                }
                result["artifact"] = art_obj
                result["artifacts"].append(art_obj)

            # Forensic interpretation
            if fraction > 0.08 or (max_err > 50 and len(anomalies) >= 3):
                result["interpretation"] = (
                    f"Pronounced compression inconsistencies detected (High-ELA Fraction: {fraction:.2%}, Max Error: {max_err:.1f}). "
                    f"Isolated regions exhibit distinct quantization histories indicative of splicing or localized re-encoding."
                )
            elif fraction > 0.03:
                result["interpretation"] = (
                    f"Moderate compression variance observed (High-ELA Fraction: {fraction:.2%}). "
                    "Select regions demonstrate elevated residual errors."
                )
            else:
                result["interpretation"] = (
                    f"Homogeneous compression characteristics across image (High-ELA Fraction: {fraction:.2%}). "
                    "No localized recompression anomalies detected."
                )

        except Exception as exc:
            result["error"] = str(exc)
            result["traceback"] = traceback.format_exc()

        return result

    def _save_artifact(self, amplified_arr: np.ndarray, evidence_id: str) -> str:
        try:
            from PIL import Image

            target_dir = os.path.join(self.artifact_dir, evidence_id)
            os.makedirs(target_dir, exist_ok=True)
            out_path = os.path.join(target_dir, "ela_heatmap.png")
            Image.fromarray(amplified_arr).save(out_path)
            return out_path
        except Exception:
            return None

