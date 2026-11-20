"""
Statistical & Copy-Move Consistency Analyzer
Analyzes RGB & YCbCr histograms, edge/blur gradient consistency,
and block-level feature matching via SVD/PCA to detect cloned/copy-moved regions.
"""
import os
import traceback
import numpy as np
from typing import Dict, Any, List


class StatisticalAnalyzer:
    def analyze(self, image_path: str, evidence_id: str = "EVD-001") -> Dict[str, Any]:
        result = {
            "engine": "statistical_analysis",
            "histogram_anomalies": {},
            "copy_move_matches": [],
            "copy_move_detected": False,
            "blur_inconsistency_score": 0.0,
            "edge_density": 0.0,
            "pca_block_variance": None,
            "interpretation": "",
            "artifacts": [],
            "limitations": [
                "Copy-move block matching uses quantized feature hashing; heavily rotated or scaled duplicates may evade detection.",
                "Histogram quantization spikes flag potential multiple compression/editing passes, but can occur naturally in high-contrast graphics.",
                "Edge and blur metrics are influenced by natural optical depth of field (bokeh)."
            ],
        }

        try:
            from PIL import Image

            img = Image.open(image_path).convert("RGB")
            arr = np.array(img)
            gray = np.array(img.convert("L"), dtype=np.float32)
            h, w = gray.shape

            # 1. Histogram & Double Quantization Analysis
            hist_info = {}
            for idx, ch in enumerate(["red", "green", "blue"]):
                c_data = arr[:, :, idx].flatten()
                counts, _ = np.histogram(c_data, bins=256, range=(0, 256))
                c_mean = float(counts.mean())
                c_std = float(counts.std())
                spikes = int((counts > (c_mean + 3.0 * c_std)).sum())
                hist_info[ch] = {
                    "mean": round(float(c_data.mean()), 2),
                    "std": round(float(c_data.std()), 2),
                    "spike_bins": spikes,
                    "quantization_artifact": bool(spikes > 10),
                }

            # YCbCr luminance (Y channel)
            ycbcr = img.convert("YCbCr")
            y_data = np.array(ycbcr)[:, :, 0].flatten()
            y_counts, _ = np.histogram(y_data, bins=256, range=(0, 256))
            y_spikes = int((y_counts > (y_counts.mean() + 3.0 * y_counts.std())).sum())
            hist_info["luminance_y"] = {
                "mean": round(float(y_data.mean()), 2),
                "std": round(float(y_data.std()), 2),
                "spike_bins": y_spikes,
                "quantization_artifact": bool(y_spikes > 10),
            }
            result["histogram_anomalies"] = hist_info

            # 2. Edge density & Blur gradient consistency
            gy = np.abs(np.diff(gray, axis=0))
            gx = np.abs(np.diff(gray, axis=1))
            edge_mag = np.sqrt(gy[:, : w - 1] ** 2 + gx[: h - 1, :] ** 2)
            edge_dens = float(np.mean(edge_mag > 12.0))
            result["edge_density"] = round(edge_dens, 4)

            # Local blur gradient variance across 4 quadrants
            qh, qw = h // 2, w // 2
            q_vars = [
                float(np.var(edge_mag[:qh, :qw])),
                float(np.var(edge_mag[:qh, qw:])),
                float(np.var(edge_mag[qh:, :qw])),
                float(np.var(edge_mag[qh:, qw:])),
            ]
            q_mean = np.mean(q_vars) + 1e-6
            blur_cv = float(np.std(q_vars) / q_mean)
            result["blur_inconsistency_score"] = round(blur_cv, 4)

            # 3. Block-based Copy-Move Detection (via quantized DCT/SVD features)
            block_size = 32
            hashes = {}
            duplicates = []

            for by in range(0, h - block_size, block_size):
                for bx in range(0, w - block_size, block_size):
                    blk = gray[by : by + block_size, bx : bx + block_size]
                    
                    # Disregard flat/untextured blocks (e.g. flat sky or uniform regions)
                    if float(np.std(blk)) < 8.0:
                        continue

                    # Quantize block mean and spatial gradient to form robust hash key
                    small = blk[::8, ::8].astype(np.int32)
                    key = (small // 12).tobytes()

                    if key in hashes:
                        prev_x, prev_y = hashes[key]
                        # Disregard immediately adjacent identical blocks (e.g. solid background)
                        dist = np.sqrt((prev_x - bx) ** 2 + (prev_y - by) ** 2)
                        if dist >= 64:
                            duplicates.append({
                                "source_region": {"x": prev_x, "y": prev_y, "width": block_size, "height": block_size},
                                "target_region": {"x": bx, "y": by, "width": block_size, "height": block_size},
                                "separation_px": round(float(dist), 1),
                            })
                    else:
                        hashes[key] = (bx, by)

            result["copy_move_matches"] = duplicates[:15]
            result["copy_move_detected"] = len(duplicates) >= 3

            # 4. PCA Block Variance (pure NumPy SVD)
            blocks = []
            rng = np.random.default_rng(42)
            num_samples = min(200, (h // block_size) * (w // block_size))
            for _ in range(num_samples):
                ry = int(rng.integers(0, max(1, h - block_size)))
                rx = int(rng.integers(0, max(1, w - block_size)))
                blocks.append(gray[ry : ry + block_size, rx : rx + block_size].flatten())

            if blocks:
                b_mat = np.vstack(blocks).astype(np.float32)
                b_mat -= b_mat.mean(axis=0)
                _, s, _ = np.linalg.svd(b_mat, full_matrices=False)
                eigs = s ** 2
                top_ratio = float(eigs[0] / (eigs.sum() + 1e-6))
                result["pca_block_variance"] = round(top_ratio, 4)

            # Interpretation
            flags = []
            if result["copy_move_detected"]:
                flags.append(f"{len(duplicates)} duplicate/cloned block pairs detected across non-adjacent regions")
            if any(v["quantization_artifact"] for v in hist_info.values()):
                flags.append("quantization peaks in color channels indicative of multiple processing passes")
            if blur_cv > 1.2:
                flags.append(f"high blur gradient divergence across quadrants (score: {blur_cv:.2f})")

            if flags:
                result["interpretation"] = (
                    "Statistical forensic anomalies identified: " + "; ".join(flags) + "."
                )
            else:
                result["interpretation"] = (
                    f"Statistical and spatial gradient profiles consistent with natural imagery (Edge density: {edge_dens:.3f})."
                )

        except Exception as exc:
            result["error"] = str(exc)
            result["traceback"] = traceback.format_exc()

        return result

