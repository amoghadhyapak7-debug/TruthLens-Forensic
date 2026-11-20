"""TruthLens Video Forensics - Visual Forensic Engine

Implements computer vision and signal processing forensic algorithms:
- Temporal continuity and motion jitter analysis
- High-frequency sensor noise residual analysis (PRNU-inspired)
- Lighting, luminance, and chrominance consistency
- Gradient boundary seam and blur discrepancy detection
- Facial boundary and aspect-ratio stability tracking
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from truthlens_video.config import (
    HIGH_FREQUENCY_NOISE_THRESHOLD,
    LAPLACIAN_BLUR_THRESHOLD,
    LIGHTING_INCONSISTENCY_THRESHOLD,
    TEMPORAL_JITTER_THRESHOLD,
)
from truthlens_video.deepfake_detector import (
    DeepfakeDetectionEngine,
    measure_facial_boundary_seam,
)
from truthlens_video.gend_detector import GenDInferenceEngine
from truthlens_video.ai_generator_detector import AIGeneratorDetectorEngine
from truthlens_video.frame_extractor import ExtractedFrame


class VisualForensicEngine:
    """Executes objective, measurement-grounded forensic analyses on extracted video frames."""

    _instance: Optional["VisualForensicEngine"] = None

    def __init__(self):
        # Dedicated neural detection engines:
        # 1. AI-Generated Synthetic Media Detector (CIFAKE-EfficientNet-B0)
        # 2. Deepfake & Face Forgery Detector (GenD CLIP ViT-L/14 + MesoNet-4 + YuNet CNN)
        self.ai_gen_engine = AIGeneratorDetectorEngine.get_instance()
        self.deepfake_engine = DeepfakeDetectionEngine.get_instance()
        self.gend_engine = GenDInferenceEngine.get_instance()
        self.last_frame_images: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Gracefully load OpenCV face detector if available
        self.face_cascade = None
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    self.face_cascade = cascade
            except Exception:
                self.face_cascade = None

    @classmethod
    def get_instance(cls) -> "VisualForensicEngine":
        """Reuses singleton VisualForensicEngine across all analyzed videos."""
        if cls._instance is None:
            cls._instance = VisualForensicEngine()
        return cls._instance

    def analyze_frames(
        self,
        frames: List[ExtractedFrame],
        fps: float,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Runs the suite of visual forensic checks over the sequence of extracted frames.

        Returns:
            findings: List of structured finding dictionaries.
            signals: Summary measurements and indicators.
            metrics: Raw metrics collected across frames.
        """
        findings: List[Dict[str, Any]] = []
        signals: List[Dict[str, Any]] = []

        if not frames:
            return findings, signals, {}

        # Preload extracted frames once into RAM (BGR and Grayscale) to eliminate redundant disk reads
        frame_images: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for f in frames:
            bgr = cv2.imread(f.absolute_path)
            if bgr is not None:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                frame_images[f.absolute_path] = (bgr, gray)
        self.last_frame_images = frame_images

        face_data: Dict[str, Any] = {}

        # 0A. Dedicated AI-Generated Video Detection (CIFAKE-EfficientNet-B0)
        ai_gen_findings, ai_gen_signal = self._analyze_ai_generated_inference(frames)
        findings.extend(ai_gen_findings)
        signals.append(ai_gen_signal)

        # 0B. Deep learning face forgery detection (GenD CLIP ViT-L/14 + MesoNet-4 + YuNet CNN)
        deepfake_findings, deepfake_signal = self._analyze_deepfake_inference(frames, frame_images, face_data)
        findings.extend(deepfake_findings)
        signals.append(deepfake_signal)

        # 1. Temporal motion and continuity analysis
        temporal_findings, temporal_signal = self._analyze_temporal_continuity(frames, fps, frame_images)
        findings.extend(temporal_findings)
        signals.append(temporal_signal)

        # 2. Sensor noise residual analysis
        noise_findings, noise_signal = self._analyze_sensor_noise_residuals(frames, frame_images)
        findings.extend(noise_findings)
        signals.append(noise_signal)

        # 3. Lighting & chromatic consistency
        lighting_findings, lighting_signal = self._analyze_lighting_chrominance(frames, frame_images)
        findings.extend(lighting_findings)
        signals.append(lighting_signal)

        # 4. Boundary and gradient inconsistency
        boundary_findings, boundary_signal = self._analyze_boundary_gradients(frames)
        findings.extend(boundary_findings)
        signals.append(boundary_signal)

        # 5. Facial stability analysis (if faces are detected)
        face_findings, face_signal = self._analyze_facial_stability(frames, frame_images, face_data)
        findings.extend(face_findings)
        if face_signal:
            signals.append(face_signal)

        # 6. Compression artifact (DCT blockiness) analysis
        compression_findings, compression_signal = self._analyze_compression_artifacts(frames, frame_images)
        findings.extend(compression_findings)
        signals.append(compression_signal)

        # 7. Color channel correlation (splicing / composite indicator)
        channel_findings, channel_signal = self._analyze_channel_correlation(frames, frame_images)
        findings.extend(channel_findings)
        signals.append(channel_signal)

        # 8. Facial texture over-smoothness (deepfake skin indicator)
        skin_findings, skin_signal = self._analyze_facial_texture_smoothness(frames, frame_images, face_data)
        findings.extend(skin_findings)
        if skin_signal:
            signals.append(skin_signal)

        # Mark suspicious frames based on findings
        for finding in findings:
            frame_ref = finding.get("frame_reference")
            if frame_ref:
                for f in frames:
                    if Path(f.absolute_path).name == frame_ref or f.relative_path == frame_ref:
                        f.is_suspicious = True
                        f.suspicion_reasons.append(finding["indicator_name"])

        metrics = {
            "total_frames_analyzed": len(frames),
            "findings_count": len(findings),
            "suspicious_frames_count": sum(1 for f in frames if f.is_suspicious),
        }

        return findings, signals, metrics

    def _analyze_temporal_continuity(
        self,
        frames: List[ExtractedFrame],
        fps: float,
        frame_images: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Examines frame-to-frame difference continuity to flag unnatural temporal jumps."""
        findings: List[Dict[str, Any]] = []
        diffs = [f.diff_score_from_previous for f in frames[1:]]

        if not diffs:
            return findings, {"signal_name": "temporal_continuity", "status": "insufficient_frames"}

        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs)) if len(diffs) > 1 else 0.0

        for i, frame in enumerate(frames[1:], start=1):
            # Skip expected scene cuts from anomalous jitter warnings
            if frame.is_scene_change:
                continue

            diff = frame.diff_score_from_previous
            # Check for sudden massive jump not explained by scene cut
            if std_diff > 0.001 and (diff - mean_diff) > (TEMPORAL_JITTER_THRESHOLD * std_diff):
                findings.append({
                    "indicator_name": "temporal_motion_anomaly",
                    "description": (
                        f"Abrupt temporal discontinuity detected (frame difference {diff:.3f} vs baseline {mean_diff:.3f}). "
                        "May indicate spliced content, dropped frames, or synthetic rendering jitter."
                    ),
                    "timestamp": frame.timestamp_seconds,
                    "frame_reference": Path(frame.absolute_path).name,
                    "severity": "medium",
                    "method_used": "interframe_difference_variance",
                    "supporting_observations": f"Diff z-score: {(diff - mean_diff) / std_diff:.2f}",
                })

        # Periodic temporal oscillation check (alternating frame displacement common in deepfake synthesis)
        if len(frames) >= 4:
            try:
                if frame_images:
                    gray_frames = [frame_images[f.absolute_path][1] for f in frames if f.absolute_path in frame_images]
                else:
                    gray_frames = [cv2.imread(f.absolute_path, cv2.IMREAD_GRAYSCALE) for f in frames]
                    gray_frames = [g for g in gray_frames if g is not None]
                if len(gray_frames) >= 4:
                    d1 = [float(cv2.absdiff(gray_frames[i], gray_frames[i-1]).mean()) for i in range(1, len(gray_frames))]
                    d2 = [float(cv2.absdiff(gray_frames[i], gray_frames[i-2]).mean()) for i in range(2, len(gray_frames))]
                    m1 = float(np.mean(d1))
                    m2 = float(np.mean(d2))
                    osc_ratio = m1 / max(m2, 0.05)
                    if osc_ratio >= 1.8 and m1 > 0.6:
                        findings.append({
                            "indicator_name": "temporal_motion_anomaly",
                            "description": (
                                f"Periodic frame-to-frame motion oscillation detected (step-1/step-2 diff ratio {osc_ratio:.2f}). "
                                "Frequent artifact of deepfake frame synthesis where inserted faces wobble around the head trajectory."
                            ),
                            "timestamp": frames[1].timestamp_seconds,
                            "frame_reference": Path(frames[1].absolute_path).name,
                            "severity": "medium",
                            "method_used": "interframe_oscillation_analysis",
                            "supporting_observations": f"Step-1 mean diff: {m1:.3f}, Step-2 mean diff: {m2:.3f}, ratio: {osc_ratio:.2f}",
                        })
            except Exception as e:
                logger.debug(f"Oscillation check exception: {e}")

        signal = {
            "signal_name": "temporal_continuity",
            "mean_interframe_difference": round(mean_diff, 4),
            "interframe_std_dev": round(std_diff, 4),
            "anomalies_detected": len(findings),
        }
        return findings, signal

    def _analyze_sensor_noise_residuals(
        self,
        frames: List[ExtractedFrame],
        frame_images: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Analyzes high-frequency residual noise patterns across frames.
        
        Natural optical sensors exhibit sensor pattern noise (PRNU). AI-generated
        imagery or heavily compressed/smoothed composites often present absent or
        abnormally flat high-frequency residuals.
        """
        findings: List[Dict[str, Any]] = []
        noise_variances: List[float] = []

        for frame in frames:
            if frame_images and frame.absolute_path in frame_images:
                img = frame_images[frame.absolute_path][1]
            else:
                img = cv2.imread(frame.absolute_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # High-pass filter: subtract Gaussian blur from original
            blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
            residual = cv2.absdiff(img, blurred)
            variance = float(np.var(residual))
            noise_variances.append(variance)

        mean_noise = float(np.mean(noise_variances)) if noise_variances else 0.0
        low_noise_frames = [
            (f, v) for f, v in zip(frames, noise_variances) if v < HIGH_FREQUENCY_NOISE_THRESHOLD
        ]

        # Only flag if overall sequence mean is suppressed OR >= 40% of frames exhibit unnatural smoothness
        if mean_noise < HIGH_FREQUENCY_NOISE_THRESHOLD or (
            len(frames) > 0 and len(low_noise_frames) / len(frames) >= 0.4
        ):
            for frame, v in low_noise_frames:
                findings.append({
                    "indicator_name": "suppressed_sensor_noise",
                    "description": (
                        f"Unusually low high-frequency sensor noise floor (frame variance {v:.2f}, sequence mean {mean_noise:.2f}). "
                        "Natural camera captures typically exhibit physical sensor grain; smooth plastic textures are common in AI synthesis."
                    ),
                    "timestamp": frame.timestamp_seconds,
                    "frame_reference": Path(frame.absolute_path).name,
                    "severity": "medium" if mean_noise < (HIGH_FREQUENCY_NOISE_THRESHOLD * 0.5) else "low",
                    "method_used": "high_pass_noise_residual",
                    "supporting_observations": f"Noise variance: {v:.3f} (threshold: {HIGH_FREQUENCY_NOISE_THRESHOLD})",
                })

        signal = {
            "signal_name": "sensor_noise_profile",
            "mean_residual_variance": round(mean_noise, 3),
            "noise_consistency": "natural" if mean_noise >= HIGH_FREQUENCY_NOISE_THRESHOLD else "unusually_smooth",
        }
        return findings, signal

    def _analyze_lighting_chrominance(
        self,
        frames: List[ExtractedFrame],
        frame_images: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Inspects luminance and chrominance stability across consecutive frames."""
        findings: List[Dict[str, Any]] = []
        lum_means: List[float] = []
        sat_means: List[float] = []

        for frame in frames:
            if frame_images and frame.absolute_path in frame_images:
                img_bgr = frame_images[frame.absolute_path][0]
            else:
                img_bgr = cv2.imread(frame.absolute_path)
            if img_bgr is None:
                continue

            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            lum_means.append(float(np.mean(v)))
            sat_means.append(float(np.mean(s)))

        # Compare adjacent frames within the same scene
        for i in range(1, len(frames)):
            if frames[i].is_scene_change:
                continue

            lum_shift = abs(lum_means[i] - lum_means[i - 1]) / 255.0
            if lum_shift > LIGHTING_INCONSISTENCY_THRESHOLD:
                findings.append({
                    "indicator_name": "illumination_inconsistency",
                    "description": (
                        f"Rapid localized illumination shift observed ({lum_shift * 100:.1f}% luminance change). "
                        "May point to compositing artifacts, face insertion, or mismatched environmental lighting."
                    ),
                    "timestamp": frames[i].timestamp_seconds,
                    "frame_reference": Path(frames[i].absolute_path).name,
                    "severity": "medium",
                    "method_used": "luminance_chrominance_gradient",
                    "supporting_observations": f"Luminance delta: {lum_shift:.3f}",
                })

        signal = {
            "signal_name": "lighting_chrominance_stability",
            "mean_luminance": round(float(np.mean(lum_means)), 2) if lum_means else 0.0,
            "mean_saturation": round(float(np.mean(sat_means)), 2) if sat_means else 0.0,
            "lighting_instabilities_detected": len(findings),
        }
        return findings, signal

    def _analyze_boundary_gradients(
        self, frames: List[ExtractedFrame]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Checks for gradient boundaries, sharp cut lines, or localized blur mismatches."""
        findings: List[Dict[str, Any]] = []
        laplacian_scores = [f.laplacian_variance for f in frames]

        if not laplacian_scores:
            return findings, {"signal_name": "gradient_sharpness", "status": "no_data"}

        mean_lap = float(np.mean(laplacian_scores))

        for frame in frames:
            # Extreme blur or localized sharpness divergence compared to sequence mean
            if (frame.laplacian_variance < 5.0 and mean_lap > (LAPLACIAN_BLUR_THRESHOLD * 2.0)) or (
                frame.laplacian_variance < mean_lap * 0.45 and mean_lap > 100.0
            ):
                findings.append({
                    "indicator_name": "abnormal_sharpness_discrepancy",
                    "description": (
                        f"Significant loss of edge gradient energy (Laplacian variance {frame.laplacian_variance:.1f} vs sequence average {mean_lap:.1f}). "
                        "Often caused by smoothing filters, deepfake warping, or compression masking."
                    ),
                    "timestamp": frame.timestamp_seconds,
                    "frame_reference": Path(frame.absolute_path).name,
                    "severity": "low",
                    "method_used": "laplacian_edge_energy",
                    "supporting_observations": f"Laplacian score: {frame.laplacian_variance:.1f}",
                })

        signal = {
            "signal_name": "gradient_sharpness",
            "sequence_mean_laplacian": round(mean_lap, 2),
            "sharpness_anomalies": len(findings),
        }
        return findings, signal

    def _analyze_facial_stability(
        self,
        frames: List[ExtractedFrame],
        frame_images: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        face_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Detects faces and tracks bounding box stability and aspect ratio consistency."""
        findings: List[Dict[str, Any]] = []
        face_records: List[Dict[str, Any]] = []

        if self.face_cascade is None:
            return findings, None

        for frame in frames:
            # Reuse faces already detected in the deepfake inference step
            if face_data and frame.absolute_path in face_data:
                detected_faces = face_data[frame.absolute_path]
                for face in detected_faces:
                    x, y, w, h = face.bbox
                    aspect_ratio = float(w) / float(h)
                    face_records.append({
                        "timestamp": frame.timestamp_seconds,
                        "frame_ref": Path(frame.absolute_path).name,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "aspect_ratio": round(aspect_ratio, 3),
                    })
            else:
                # Fallback: use Haar cascade if face_data not available
                if frame_images and frame.absolute_path in frame_images:
                    img = frame_images[frame.absolute_path][0]
                    gray = frame_images[frame.absolute_path][1]
                else:
                    img = cv2.imread(frame.absolute_path)
                    if img is None:
                        continue
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
                )

                for (x, y, w, h) in faces:
                    aspect_ratio = float(w) / float(h)
                    face_records.append({
                        "timestamp": frame.timestamp_seconds,
                        "frame_ref": Path(frame.absolute_path).name,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "aspect_ratio": round(aspect_ratio, 3),
                    })

        if not face_records:
            return findings, None

        aspect_ratios = [r["aspect_ratio"] for r in face_records]
        ar_std = float(np.std(aspect_ratios)) if len(aspect_ratios) > 1 else 0.0

        # Unnatural facial geometry deformation
        if ar_std > 0.18:
            for rec in face_records:
                if abs(rec["aspect_ratio"] - np.mean(aspect_ratios)) > (2.0 * ar_std):
                    findings.append({
                        "indicator_name": "facial_geometry_instability",
                        "description": (
                            f"Facial aspect ratio warped abnormally ({rec['aspect_ratio']:.2f} vs average {np.mean(aspect_ratios):.2f}). "
                            "May indicate facial replacement, warp artifacts, or AI face re-enactment instability."
                        ),
                        "timestamp": rec["timestamp"],
                        "frame_reference": rec["frame_ref"],
                        "severity": "medium",
                        "method_used": "facial_morphology_tracking",
                        "supporting_observations": f"BBox: {rec['bbox']}, Aspect Ratio: {rec['aspect_ratio']}",
                    })

        signal = {
            "signal_name": "facial_presence_and_morphology",
            "faces_detected_count": len(face_records),
            "aspect_ratio_stability_std": round(ar_std, 3),
            "facial_anomalies_detected": len(findings),
        }
        return findings, signal

    def _analyze_compression_artifacts(
        self,
        frames: List[ExtractedFrame],
        frame_images: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Detects abnormal JPEG blockiness or inconsistent DCT compression patterns.

        AI-generated frames and composited inserts often show mismatched DCT block
        artifacts compared to the surrounding video compression envelope.
        """
        findings: List[Dict[str, Any]] = []
        blockiness_scores: List[float] = []

        for frame in frames:
            if frame_images and frame.absolute_path in frame_images:
                img = frame_images[frame.absolute_path][1]  # grayscale already cached
            else:
                img = cv2.imread(frame.absolute_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Measure horizontal/vertical DCT block boundary energy (8-pixel grid)
            h, w = img.shape
            h_diffs, v_diffs = [], []
            for y in range(8, h - 8, 8):
                row_diff = float(np.mean(np.abs(img[y, :].astype(float) - img[y - 1, :].astype(float))))
                h_diffs.append(row_diff)
            for x in range(8, w - 8, 8):
                col_diff = float(np.mean(np.abs(img[:, x].astype(float) - img[:, x - 1].astype(float))))
                v_diffs.append(col_diff)

            block_boundary_energy = float(np.mean(h_diffs + v_diffs)) if (h_diffs or v_diffs) else 0.0

            # Compare to non-block interior pixel diffs
            interior_diffs = []
            for y in range(1, h - 1, 2):
                for x in range(1, w - 1, 2):
                    if (y % 8) != 0 and (x % 8) != 0:
                        interior_diffs.append(abs(int(img[y, x]) - int(img[y - 1, x])))
            interior_energy = float(np.mean(interior_diffs)) if interior_diffs else 1.0

            # Blockiness ratio: high ratio = strong grid-aligned compression energy = potential insertion
            ratio = block_boundary_energy / max(interior_energy, 0.5)
            blockiness_scores.append(ratio)

        mean_blockiness = float(np.mean(blockiness_scores)) if blockiness_scores else 0.0
        std_blockiness = float(np.std(blockiness_scores)) if len(blockiness_scores) > 1 else 0.0

        # Flag frames with abnormally high blockiness vs sequence baseline
        threshold_high = mean_blockiness + (2.5 * std_blockiness)
        for frame, score in zip(frames, blockiness_scores):
            if std_blockiness > 0.1 and score > threshold_high and score > 2.0:
                findings.append({
                    "indicator_name": "compression_block_inconsistency",
                    "description": (
                        f"DCT block boundary energy significantly elevated at this frame (ratio {score:.2f} vs sequence mean {mean_blockiness:.2f}). "
                        "Localized JPEG blockiness spikes can indicate that content was inserted from a differently-compressed source."
                    ),
                    "timestamp": frame.timestamp_seconds,
                    "frame_reference": Path(frame.absolute_path).name,
                    "severity": "medium",
                    "method_used": "dct_block_boundary_energy",
                    "supporting_observations": f"Block energy ratio: {score:.3f}, sequence std: {std_blockiness:.3f}",
                })

        signal = {
            "signal_name": "compression_artifact_profile",
            "mean_dct_blockiness_ratio": round(mean_blockiness, 3),
            "blockiness_std": round(std_blockiness, 3),
            "compression_anomalies": len(findings),
        }
        return findings, signal

    def _analyze_channel_correlation(
        self,
        frames: List[ExtractedFrame],
        frame_images: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Examines per-frame RGB channel inter-correlation for splicing signatures.

        Genuine camera footage exhibits a stable, physically-consistent relationship
        between colour channels driven by illuminant and sensor response. Composites
        often show anomalous de-correlation between channels at splice boundaries.
        """
        findings: List[Dict[str, Any]] = []
        correlation_scores: List[float] = []

        for frame in frames:
            if frame_images and frame.absolute_path in frame_images:
                img = frame_images[frame.absolute_path][0]
            else:
                img = cv2.imread(frame.absolute_path)
            if img is None:
                continue

            b, g, r = cv2.split(img.astype(np.float32))
            # Pearson correlation between R-G and B-G channel pairs
            rg_corr = float(np.corrcoef(r.flatten(), g.flatten())[0, 1])
            bg_corr = float(np.corrcoef(b.flatten(), g.flatten())[0, 1])
            # Composite score: low values indicate de-correlated channels
            combined = (rg_corr + bg_corr) / 2.0
            correlation_scores.append(combined)

        if not correlation_scores:
            return findings, {"signal_name": "channel_correlation", "status": "no_data"}

        mean_corr = float(np.mean(correlation_scores))
        std_corr = float(np.std(correlation_scores)) if len(correlation_scores) > 1 else 0.0

        # Flag frames where channel correlation drops sharply below sequence baseline
        low_threshold = mean_corr - (2.2 * std_corr)
        for frame, score in zip(frames, correlation_scores):
            if std_corr > 0.03 and score < low_threshold and score < 0.75:
                findings.append({
                    "indicator_name": "channel_correlation_anomaly",
                    "description": (
                        f"Unusual RGB channel de-correlation detected (score {score:.3f} vs sequence mean {mean_corr:.3f}). "
                        "Colour channel de-correlation can indicate composited or colour-graded insertion inconsistent with surrounding frames."
                    ),
                    "timestamp": frame.timestamp_seconds,
                    "frame_reference": Path(frame.absolute_path).name,
                    "severity": "medium",
                    "method_used": "rgb_channel_pearson_correlation",
                    "supporting_observations": f"R-G / B-G channel correlation: {score:.4f}",
                })

        signal = {
            "signal_name": "channel_correlation",
            "mean_rgb_correlation": round(mean_corr, 4),
            "correlation_std": round(std_corr, 4),
            "channel_anomalies": len(findings),
        }
        return findings, signal

    def _analyze_facial_texture_smoothness(
        self,
        frames: List[ExtractedFrame],
        frame_images: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        face_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Measures skin-region texture complexity on detected faces.

        Deepfake face generators typically produce unnaturally smooth skin textures
        lacking the micro-pore and fine-detail complexity of real facial skin.
        Over-smoothness in a face region is a known generative AI indicator.
        """
        findings: List[Dict[str, Any]] = []

        if self.face_cascade is None:
            return findings, None

        texture_scores: List[float] = []
        per_frame: List[Tuple[Any, float]] = []

        for frame in frames:
            # Reuse faces already detected in the deepfake inference step
            if face_data and frame.absolute_path in face_data:
                detected_faces = face_data[frame.absolute_path]
                if not detected_faces:
                    continue
                # Reuse cached grayscale image
                if frame_images and frame.absolute_path in frame_images:
                    gray = frame_images[frame.absolute_path][1]
                else:
                    img_bgr = cv2.imread(frame.absolute_path)
                    if img_bgr is None:
                        continue
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                # Pick largest face by area
                best = max(detected_faces, key=lambda f: f.bbox[2] * f.bbox[3])
                x, y, w, h = best.bbox
                face_roi = gray[max(0, y):min(gray.shape[0], y + h), max(0, x):min(gray.shape[1], x + w)]
            else:
                # Fallback: detect faces fresh
                if frame_images and frame.absolute_path in frame_images:
                    img = frame_images[frame.absolute_path][0]
                    gray = frame_images[frame.absolute_path][1]
                else:
                    img = cv2.imread(frame.absolute_path)
                    if img is None:
                        continue
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
                )
                if len(faces) == 0:
                    continue

                # Take the largest detected face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_roi = gray[y:y + h, x:x + w]

            if face_roi.size == 0:
                continue

            # Local Binary Pattern variance as texture complexity measure
            # We approximate using Laplacian of the face ROI
            lap_var = float(cv2.Laplacian(face_roi, cv2.CV_64F).var())
            texture_scores.append(lap_var)
            per_frame.append((frame, lap_var))

        if not texture_scores:
            return findings, None

        mean_texture = float(np.mean(texture_scores))
        std_texture = float(np.std(texture_scores)) if len(texture_scores) > 1 else 0.0

        # Absolute threshold: face Laplacian < 15 indicates near-absence of skin texture
        LOW_TEXTURE_ABS = 15.0

        for frame, score in per_frame:
            is_sequence_outlier = std_texture > 2.0 and score < (mean_texture - 2.0 * std_texture)
            is_absolutely_smooth = score < LOW_TEXTURE_ABS
            if is_absolutely_smooth or is_sequence_outlier:
                findings.append({
                    "indicator_name": "facial_texture_over_smoothness",
                    "description": (
                        f"Detected face region exhibits unusually low texture complexity (Laplacian variance {score:.1f}, "
                        f"sequence average {mean_texture:.1f}). "
                        "Generative deepfake models typically produce over-smoothed skin lacking natural micro-texture and pore detail."
                    ),
                    "timestamp": frame.timestamp_seconds,
                    "frame_reference": Path(frame.absolute_path).name,
                    "severity": "medium",
                    "method_used": "face_roi_laplacian_texture",
                    "supporting_observations": f"Face texture score: {score:.2f} (threshold: {LOW_TEXTURE_ABS})",
                })

        signal = {
            "signal_name": "facial_texture_complexity",
            "mean_face_texture_score": round(mean_texture, 2),
            "texture_std": round(std_texture, 2),
            "smooth_face_frames": len(findings),
        }
        return findings, signal

    def _analyze_ai_generated_inference(
        self, frames: List[ExtractedFrame]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """AI-generated video detection is disabled.

        This detector was removed to prevent false positives on genuine videos.
        The deepfake detection pipeline (GenD CLIP ViT-L/14 + MesoNet-4) remains active.
        """
        # Return an empty/null signal so the verdict engine ignores AI-gen scoring entirely.
        signal = {
            "signal_name": "ai_generated_video_detection",
            "model_name": "CIFAKE-EfficientNet-B0 (disabled)",
            "model_status": "disabled",
            "frames_analyzed": 0,
            "mean_ai_generated_pct": None,
            "peak_ai_generated_pct": None,
            "mean_authentic_pct": None,
            "flagged_synthetic_frames_count": 0,
            "frame_predictions": [],
            "findings": [],
        }
        return [], signal

    def _analyze_deepfake_inference(
        self,
        frames: List[ExtractedFrame],
        frame_images: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        face_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Executes deep learning inference (GenD ViT-L/14 & MesoNet-4) on detected facial regions using batching.
        
        Performs genuine neural inference on face crops extracted across video frames:
        1. YuNet CNN / Haar face detection
        2. Vectorized batch execution for GenD CLIP ViT-L/14 and MesoNet-4
        3. Facial boundary seam gradient analysis
        4. Calculation of genuine manipulation & real probabilities
        """
        findings: List[Dict[str, Any]] = []

        if not self.deepfake_engine.is_loaded:
            signal = {
                "signal_name": "deepfake_neural_inference",
                "model_name": "MesoNet-4 (FaceForensics++ Deepfake & Face2Face)",
                "model_status": "model_not_loaded",
                "frames_with_faces": 0,
                "total_faces_detected": 0,
                "mean_manipulation_score": None,
                "mean_manipulation_pct": None,
                "max_manipulation_score": None,
                "mean_original_score": None,
                "mean_original_pct": None,
                "suspicious_face_frames_count": 0,
            }
            return findings, signal

        frame_manip_scores: List[float] = []
        gend_fake_scores: List[float] = []
        frames_with_faces = 0
        total_faces_detected = 0

        # Step 1: Detect faces and collect face crops across all frames
        frame_face_items = []
        all_meso_crops = []
        all_gend_crops = []

        for frame in frames:
            if frame_images and frame.absolute_path in frame_images:
                img = frame_images[frame.absolute_path][0]
            else:
                img = cv2.imread(frame.absolute_path)
            if img is None:
                continue

            detected_faces = self.deepfake_engine.face_detector.detect_faces(img)
            if face_data is not None:
                face_data[frame.absolute_path] = detected_faces

            if detected_faces:
                frames_with_faces += 1
                total_faces_detected += len(detected_faces)
                for face in detected_faces:
                    bbox = face.bbox
                    x, y, w, h = bbox
                    face_crop = img[max(0, y):min(img.shape[0], y+h), max(0, x):min(img.shape[1], x+w)]
                    if face_crop.size == 0:
                        face_crop = img

                    meso_crop = face.face_crop_256 if face.face_crop_256 is not None else face_crop
                    idx = len(all_meso_crops)
                    all_meso_crops.append(meso_crop)
                    all_gend_crops.append(face_crop)
                    frame_face_items.append({
                        "frame": frame,
                        "img": img,
                        "bbox": bbox,
                        "crop_idx": idx,
                    })

        # Step 2: Vectorized batch neural model evaluation
        meso_scores = []
        if all_meso_crops:
            meso_scores = self.deepfake_engine.infer_face_crops_batch(all_meso_crops, batch_size=16)

        gend_scores = []
        if all_gend_crops and self.gend_engine and self.gend_engine.is_loaded:
            gend_scores = self.gend_engine.infer_faces_batch(all_gend_crops, batch_size=8)

        # Step 3: Map neural predictions and boundary seam measurements to findings
        per_frame_items: Dict[str, List[Dict[str, Any]]] = {}
        for item in frame_face_items:
            f_ref = item["frame"].absolute_path
            per_frame_items.setdefault(f_ref, []).append(item)

        for f_ref, items in per_frame_items.items():
            frame_meso_vals = [
                meso_scores[it["crop_idx"]]["manipulation_score"]
                for it in items
                if it["crop_idx"] < len(meso_scores)
            ]
            if frame_meso_vals:
                frame_manip_scores.append(max(frame_meso_vals))

            for it in items:
                idx = it["crop_idx"]
                frame = it["frame"]
                img = it["img"]
                bbox = it["bbox"]
                score = meso_scores[idx]["manipulation_score"] if idx < len(meso_scores) else 0.0

                # 1. GenD CLIP ViT-L/14 Deepfake Evaluation (Primary Detector)
                if idx < len(gend_scores):
                    g_res = gend_scores[idx]
                    if g_res.get("gend_inferred"):
                        gend_fake_scores.append(g_res["gend_fake_prob"])
                        if g_res["gend_fake_prob"] >= 0.65:
                            g_sev = "high" if g_res["gend_fake_prob"] >= 0.80 else "medium"
                            findings.append({
                                "indicator_name": "facial_deepfake_forgery_detected",
                                "description": (
                                    f"GenD CLIP ViT-L/14 detected face forgery artifacts (manipulation probability: {g_res['gend_fake_prob']*100:.1f}%). "
                                    "Generalizable deepfake detector identified latent manipulation artifacts."
                                ),
                                "timestamp": frame.timestamp_seconds,
                                "frame_reference": Path(frame.absolute_path).name,
                                "severity": g_sev,
                                "method_used": "gend_clip_vit_l14",
                                "supporting_observations": f"GenD Fake score: {g_res['gend_fake_prob']:.4f}, Real score: {g_res['gend_real_prob']:.4f}, BBox: {bbox}",
                            })

                # 2. MesoNet-4 Deepfake Neural Model Flag (Secondary Detector)
                if score >= 0.65:
                    severity = "high" if score >= 0.80 else "medium"
                    findings.append({
                        "indicator_name": "facial_deepfake_forgery_detected",
                        "description": (
                            f"Deep learning detector identified face forgery artifacts (manipulation probability: {score*100:.1f}%). "
                            "MesoNet-4 model trained on FaceForensics++ detected mesoscopic facial blending inconsistencies."
                        ),
                        "timestamp": frame.timestamp_seconds,
                        "frame_reference": Path(frame.absolute_path).name,
                        "severity": severity,
                        "method_used": "mesonet_4_deepfake_classifier",
                        "supporting_observations": f"Meso4 score: {score:.4f}, Authenticity score: {1.0 - score:.4f}, BBox: {bbox}",
                    })

                # 3. Facial Boundary Seam Gradient Discontinuity
                seam_ratio = measure_facial_boundary_seam(img, bbox)
                if seam_ratio >= 1.50:
                    findings.append({
                        "indicator_name": "facial_boundary_blending_seam",
                        "description": (
                            f"Anomalous gradient energy discrepancy along facial boundary seam (outer/inner ratio {seam_ratio:.2f}). "
                            "Frequent artifact of pasted or morphed face masks where composite edges meet the background."
                        ),
                        "timestamp": frame.timestamp_seconds,
                        "frame_reference": Path(frame.absolute_path).name,
                        "severity": "medium",
                        "method_used": "facial_boundary_sobel_gradient",
                        "supporting_observations": f"Border/interior gradient ratio: {seam_ratio:.3f}",
                    })

        if frames_with_faces > 0:
            meso_mean = float(np.mean(frame_manip_scores))
            meso_max = float(np.max(frame_manip_scores))
            suspicious_count = sum(1 for s in frame_manip_scores if s >= 0.65)
            g_mean = float(np.mean(gend_fake_scores)) if gend_fake_scores else None
            g_max = float(np.max(gend_fake_scores)) if gend_fake_scores else None

            # GenD is the primary detector; MesoNet is kept separate without arbitrary averaging
            if g_mean is not None:
                primary_score = g_mean
                primary_model_name = "GenD CLIP ViT-L/14"
            else:
                primary_score = meso_mean
                primary_model_name = "MesoNet-4 (FaceForensics++)"

            mean_real = float(1.0 - primary_score)

            gend_status = (
                "loaded_and_active" if (self.gend_engine and self.gend_engine.is_loaded)
                else ("downloading" if (self.gend_engine and "downloading" in self.gend_engine.load_message.lower())
                      else "model_not_loaded")
            )

            signal = {
                "signal_name": "deepfake_neural_inference",
                "model_name": primary_model_name,
                "model_version": "GenD CLIP ViT-L/14 (WACV 2026) / MesoNet-4 (Afchar 2018)",
                "face_detector": self.deepfake_engine.face_detector.backend_name,
                "model_status": "loaded_and_active",
                "frames_with_faces": frames_with_faces,
                "total_faces_detected": total_faces_detected,
                "mean_manipulation_score": round(primary_score, 4),
                "mean_manipulation_pct": round(primary_score * 100, 1),
                "max_manipulation_score": round(g_max if g_max is not None else meso_max, 4),
                "mean_original_score": round(mean_real, 4),
                "mean_original_pct": round(mean_real * 100, 1),
                "mesonet_mean_fake_pct": round(meso_mean * 100, 1),
                "suspicious_face_frames_count": suspicious_count,
                "gend_status": gend_status,
                "gend_model_name": "GenD CLIP ViT-L/14 (WACV 2026)",
                "gend_mean_fake_pct": round(g_mean * 100, 1) if g_mean is not None else None,
                "gend_max_fake_pct": round(g_max * 100, 1) if g_max is not None else None,
                "gend_mean_real_pct": round((1.0 - g_mean) * 100, 1) if g_mean is not None else None,
            }
        else:
            signal = {
                "signal_name": "deepfake_neural_inference",
                "model_name": "MesoNet-4 (FaceForensics++ Deepfake & Face2Face)",
                "model_version": "Afchar et al. WIFS 2018 (PyTorch native)",
                "face_detector": self.deepfake_engine.face_detector.backend_name,
                "model_status": "no_faces_detected",
                "frames_with_faces": 0,
                "total_faces_detected": 0,
                "mean_manipulation_score": None,
                "mean_manipulation_pct": None,
                "max_manipulation_score": None,
                "mean_original_score": None,
                "mean_original_pct": None,
                "suspicious_face_frames_count": 0,
                "limitation": "No human faces detected across examined frames. Dedicated facial deepfake model could not be evaluated; traditional signal analysis was applied.",
            }

        return findings, signal
