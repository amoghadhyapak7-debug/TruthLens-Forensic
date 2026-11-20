"""
Image Consistency Analyzer
Checks for inconsistencies in blur gradients, lighting direction,
and chromatic aberration patterns.
"""
import os
import numpy as np
import traceback
from typing import Dict, Any


class ConsistencyAnalyzer:

    def analyze(self, image_path: str) -> Dict[str, Any]:
        result = {
            "engine": "consistency_analysis",
            "blur_consistency": {},
            "lighting_consistency": {},
            "chromatic_aberration": {},
            "jpeg_ghost_score": None,
            "interpretation": "",
            "artifacts": [],
            "limitations": [
                "Blur and lighting consistency checks are heuristic and may flag legitimate creative photography.",
                "JPEG ghost detection is only meaningful for JPEG source images.",
                "Chromatic aberration analysis is limited without lens-specific profiles.",
            ],
        }
        try:
            from PIL import Image
            import numpy as np
            from scipy.ndimage import gaussian_filter, uniform_filter

            img = Image.open(image_path).convert("RGB")
            arr = np.array(img, dtype=np.float32)
            gray = arr.mean(axis=2)

            # ── Blur Consistency ────────────────────────────────────
            # Compute local blur estimate via Laplacian variance across quadrants
            lap = self._laplacian(gray)
            h, w = gray.shape
            quadrant_sharpness = {
                "top_left":     float(np.var(lap[:h//2, :w//2])),
                "top_right":    float(np.var(lap[:h//2, w//2:])),
                "bottom_left":  float(np.var(lap[h//2:, :w//2])),
                "bottom_right": float(np.var(lap[h//2:, w//2:])),
            }
            vals = list(quadrant_sharpness.values())
            blur_cv = np.std(vals) / (np.mean(vals) + 1e-6)
            result["blur_consistency"] = {
                "quadrant_laplacian_variance": {k: round(v, 2) for k, v in quadrant_sharpness.items()},
                "coefficient_of_variation": round(float(blur_cv), 4),
                "inconsistency_detected": bool(blur_cv > 1.0),
            }

            # ── Lighting Consistency ────────────────────────────────
            # Approximate lighting direction from gradient orientation histogram
            gy = np.gradient(gray, axis=0)
            gx = np.gradient(gray, axis=1)
            angle = np.arctan2(gy, gx) * 180 / np.pi
            magnitude = np.sqrt(gx**2 + gy**2)

            # Weighted histogram of gradient angles
            hist, edges = np.histogram(angle, bins=36, range=(-180, 180),
                                       weights=magnitude)
            dominant_angle = float(edges[np.argmax(hist)])
            angle_spread = float(np.std(angle[magnitude > magnitude.mean()]))

            result["lighting_consistency"] = {
                "dominant_gradient_angle_deg": round(dominant_angle, 1),
                "gradient_angle_spread_deg": round(angle_spread, 1),
                "high_spread_flag": bool(angle_spread > 90),
            }

            # ── Chromatic Aberration ────────────────────────────────
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            rg_diff = float(np.abs(r - g).mean())
            rb_diff = float(np.abs(r - b).mean())
            gb_diff = float(np.abs(g - b).mean())
            result["chromatic_aberration"] = {
                "mean_rg_channel_diff": round(rg_diff, 3),
                "mean_rb_channel_diff": round(rb_diff, 3),
                "mean_gb_channel_diff": round(gb_diff, 3),
                "high_aberration_flag": bool(max(rg_diff, rb_diff, gb_diff) > 25),
            }

            # ── JPEG Ghost ──────────────────────────────────────────
            # Minimum error across a range of re-save qualities
            try:
                import io
                orig = Image.open(image_path).convert("RGB")
                orig_arr = np.array(orig, dtype=np.float32)
                min_ghost_score = float("inf")
                for q in [70, 80, 90, 95]:
                    buf = io.BytesIO()
                    orig.save(buf, format="JPEG", quality=q)
                    buf.seek(0)
                    recomp = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
                    score = float(np.abs(orig_arr - recomp).mean())
                    min_ghost_score = min(min_ghost_score, score)
                result["jpeg_ghost_score"] = round(min_ghost_score, 4)
            except Exception:
                result["jpeg_ghost_score"] = None

            # ── Interpretation ──────────────────────────────────────
            flags = []
            if result["blur_consistency"]["inconsistency_detected"]:
                flags.append("inconsistent blur across quadrants")
            if result["lighting_consistency"]["high_spread_flag"]:
                flags.append("high gradient angle spread")
            if result["chromatic_aberration"]["high_aberration_flag"]:
                flags.append("high chromatic aberration detected")
            result["interpretation"] = (
                "Consistency anomalies: " + "; ".join(flags) if flags
                else "No strong consistency anomalies detected."
            )

        except Exception as exc:
            result["error"] = str(exc)
            result["traceback"] = traceback.format_exc()

        return result

    @staticmethod
    def _laplacian(gray: np.ndarray) -> np.ndarray:
        """Fast discrete Laplacian."""
        from scipy.ndimage import laplace
        return laplace(gray)

