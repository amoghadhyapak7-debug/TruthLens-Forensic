"""TruthLens Video Forensics - Deepfake & Face Forgery Detection Engine

Implements dedicated deep learning face forgery detection using:
1. MesoNet-4 (Meso4) CNN architecture (Afchar et al., IEEE WIFS 2018), trained
   on FaceForensics++ / Deepfake datasets (deepfake face swaps and Face2Face reenactments).
2. YuNet CNN Face Detector (OpenCV Zoo / WIDER FACE benchmark) with 5-point facial landmarks.
3. Fallback to Haar Cascade Classifier if ONNX runtime is unavailable.
4. Facial boundary blending analysis (seam gradient discrepancy between face and background).
5. Temporal landmark consistency tracking across chronological frames.

Outputs genuine, uncalibrated model probabilities and structured forensic observations.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
MESO4_DF_PATH = MODELS_DIR / "meso4_df.pt"
MESO4_F2F_PATH = MODELS_DIR / "meso4_f2f.pt"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"


# ===========================================================================
# 1. MesoNet-4 Architecture (PyTorch implementation)
# ===========================================================================
class Meso4(nn.Module):
    """MesoNet-4 architecture for mesoscopic facial video forgery detection.
    
    Reference:
        Darius Afchar, Vincent Nozick, Junichi Yamagishi, Isao Echizen.
        "MesoNet: a Compact Facial Video Forgery Detection Network",
        IEEE International Workshop on Information Forensics and Security (WIFS 2018).
    """

    def __init__(self):
        super().__init__()
        # Conv block 1
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 2
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)

        # Conv block 3
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)

        # Conv block 4
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        # Classification head
        self.fc1 = nn.Linear(1024, 16)
        self.leaky = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Tensor of shape (B, 3, 256, 256) normalized to [0, 1].
            
        Returns:
            Tensor of shape (B, 1) containing P(Real).
        """
        x = self.pool2(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool2(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.leaky(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


# ===========================================================================
# 2. Face Detection Data Structures & Wrappers
# ===========================================================================
@dataclass
class DetectedFace:
    """Represents a detected face in a video frame."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None  # 5 landmarks if available
    face_crop_256: Optional[np.ndarray] = None        # (256, 256, 3) normalized BGR crop


class FaceDetector:
    """Robust multi-scale face detector using OpenCV YuNet with Haar cascade fallback."""

    def __init__(self):
        self.yunet_detector = None
        self.haar_cascade = None
        self._backend = "none"

        # Attempt to initialize YuNet CNN detector
        if YUNET_PATH.exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                # Default input size (will be updated dynamically per frame)
                # score_threshold set to 0.60 (standard YuNet threshold) to avoid false positives on background textures
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    str(YUNET_PATH),
                    "",
                    (320, 320),
                    score_threshold=0.60,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                self._backend = "yunet_cnn"
                logger.info("Initialized YuNet CNN Face Detector successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize YuNet: {e}")
                self.yunet_detector = None

        # Fallback Haar Cascade
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    self.haar_cascade = cascade
                    if self._backend == "none":
                        self._backend = "haar_cascade"
            except Exception as e:
                logger.warning(f"Failed to initialize Haar Cascade: {e}")

    @property
    def backend_name(self) -> str:
        return self._backend

    def detect_faces(self, img_bgr: np.ndarray) -> List[DetectedFace]:
        """Detects faces in a BGR frame and returns cropped 256x256 face images."""
        if img_bgr is None or img_bgr.size == 0:
            return []

        h, w = img_bgr.shape[:2]
        detected: List[DetectedFace] = []

        # 1. Try YuNet CNN detector
        if self.yunet_detector is not None:
            try:
                self.yunet_detector.setInputSize((w, h))
                ret, faces = self.yunet_detector.detect(img_bgr)
                if ret and faces is not None:
                    for f in faces:
                        # YuNet output format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
                        fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                        conf = float(f[14])
                        if conf < 0.60 or fw < 15 or fh < 15:
                            continue

                        # Extract 5 landmarks
                        landmarks = [
                            (int(f[4]), int(f[5])),   # right eye
                            (int(f[6]), int(f[7])),   # left eye
                            (int(f[8]), int(f[9])),   # nose tip
                            (int(f[10]), int(f[11])), # right mouth
                            (int(f[12]), int(f[13])), # left mouth
                        ]

                        crop = self._crop_and_pad_face(img_bgr, fx, fy, fw, fh)
                        detected.append(
                            DetectedFace(
                                bbox=(fx, fy, fw, fh),
                                confidence=round(conf, 4),
                                landmarks=landmarks,
                                face_crop_256=crop,
                            )
                        )
                    if detected:
                        return detected
            except Exception as e:
                logger.debug(f"YuNet detection exception: {e}")

        # 2. Fallback to Haar Cascade (only when YuNet CNN detector is unavailable)
        if self.yunet_detector is None and self.haar_cascade is not None:
            try:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.haar_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                for (x, y, w_box, h_box) in faces:
                    crop = self._crop_and_pad_face(img_bgr, x, y, w_box, h_box)
                    detected.append(
                        DetectedFace(
                            bbox=(int(x), int(y), int(w_box), int(h_box)),
                            confidence=0.75,
                            landmarks=None,
                            face_crop_256=crop,
                        )
                    )
                if detected:
                    return detected

                # 3. Small-frame upscaling fallback (for low-res videos e.g. 240p/360p)
                if max(h, w) < 600:
                    scaled = cv2.resize(gray, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
                    faces_scaled = self.haar_cascade.detectMultiScale(
                        scaled, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )
                    for (sx, sy, sw, sh) in faces_scaled:
                        orig_x = int(sx / 1.5)
                        orig_y = int(sy / 1.5)
                        orig_w = int(sw / 1.5)
                        orig_h = int(sh / 1.5)
                        crop = self._crop_and_pad_face(img_bgr, orig_x, orig_y, orig_w, orig_h)
                        detected.append(
                            DetectedFace(
                                bbox=(orig_x, orig_y, orig_w, orig_h),
                                confidence=0.70,
                                landmarks=None,
                                face_crop_256=crop,
                            )
                        )
            except Exception as e:
                logger.debug(f"Haar cascade detection exception: {e}")

        return detected

    @staticmethod
    def crop_face_for_xception(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """FaceForensics++ standard face crop: 1.3x margin resized to 299x299 RGB tensor format."""
        H, W = img.shape[:2]
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(W, x + w + margin_x)
        y2 = min(H, y + h + margin_y)

        face_roi = img[y1:y2, x1:x2]
        if face_roi.size == 0:
            return cv2.resize(img, (299, 299), interpolation=cv2.INTER_CUBIC)
        return cv2.resize(face_roi, (299, 299), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _crop_and_pad_face(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Expands the face bbox slightly and resizes to 256x256 as required by MesoNet."""
        H, W = img.shape[:2]
        # Expand bounding box by 15% margin to capture hairline and chin boundary seams
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(W, x + w + margin_x)
        y2 = min(H, y + h + margin_y)

        face_roi = img[y1:y2, x1:x2]
        if face_roi.size == 0:
            return cv2.resize(img, (256, 256))
        return cv2.resize(face_roi, (256, 256), interpolation=cv2.INTER_AREA)


# ===========================================================================
# 3. Deepfake Inference Engine (Singleton Model Loader)
# ===========================================================================
class DeepfakeDetectionEngine:
    """Manages deep learning model loading, face detection, and forgery inference."""

    _instance: Optional[DeepfakeDetectionEngine] = None

    def __init__(self):
        self.device = torch.device("cpu")
        self.face_detector = FaceDetector()
        self.meso_df: Optional[Meso4] = None
        self.meso_f2f: Optional[Meso4] = None
        self.is_loaded = False
        self._load_models()

    @classmethod
    def get_instance(cls) -> DeepfakeDetectionEngine:
        if cls._instance is None:
            cls._instance = DeepfakeDetectionEngine()
        return cls._instance

    def _load_models(self) -> None:
        """Loads pretrained Meso4 models from disk."""
        try:
            if MESO4_DF_PATH.exists():
                model_df = Meso4()
                state_df = torch.load(MESO4_DF_PATH, map_location=self.device, weights_only=True)
                model_df.load_state_dict(state_df)
                model_df.eval()
                self.meso_df = model_df

            if MESO4_F2F_PATH.exists():
                model_f2f = Meso4()
                state_f2f = torch.load(MESO4_F2F_PATH, map_location=self.device, weights_only=True)
                model_f2f.load_state_dict(state_f2f)
                model_f2f.eval()
                self.meso_f2f = model_f2f

            self.is_loaded = bool(self.meso_df is not None)
            logger.info(f"DeepfakeDetectionEngine loaded. Model status: is_loaded={self.is_loaded}")
        except Exception as e:
            logger.error(f"Failed to load deepfake detection models: {e}", exc_info=True)
            self.is_loaded = False

    def infer_face_crops_batch(self, face_crops_256: List[np.ndarray], batch_size: int = 16) -> List[Dict[str, float]]:
        """Runs vectorized Meso4 inference on a list of 256x256 face crops."""
        if not face_crops_256:
            return []

        if not self.is_loaded or self.meso_df is None:
            return [{"manipulation_score": 0.0, "real_score": 1.0, "model_inferred": False} for _ in face_crops_256]

        results: List[Dict[str, float]] = []
        n_total = len(face_crops_256)

        for i in range(0, n_total, batch_size):
            chunk = face_crops_256[i : i + batch_size]
            tensors = []
            valid_indices = []

            for idx_in_chunk, crop in enumerate(chunk):
                if crop is not None and getattr(crop, "size", 0) > 0:
                    try:
                        face_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        t = torch.from_numpy(face_rgb).float() / 255.0
                        t = t.permute(2, 0, 1).unsqueeze(0)
                        tensors.append(t)
                        valid_indices.append(idx_in_chunk)
                    except Exception as e:
                        logger.debug(f"[MesoNet] Crop preprocessing error: {e}")

            if not tensors:
                for _ in chunk:
                    results.append({"manipulation_score": 0.0, "real_score": 1.0, "model_inferred": False})
                continue

            batch_tensor = torch.cat(tensors, dim=0).to(self.device)

            with torch.no_grad():
                p_real_df_batch = self.meso_df(batch_tensor).squeeze(-1)
                if self.meso_f2f is not None:
                    p_real_f2f_batch = self.meso_f2f(batch_tensor).squeeze(-1)
                else:
                    p_real_f2f_batch = p_real_df_batch

            t_ptr = 0
            for idx_in_chunk in range(len(chunk)):
                if idx_in_chunk in valid_indices:
                    real_df = float(p_real_df_batch[t_ptr].item()) if p_real_df_batch.ndim > 0 else float(p_real_df_batch.item())
                    real_f2f = float(p_real_f2f_batch[t_ptr].item()) if p_real_f2f_batch.ndim > 0 else float(p_real_f2f_batch.item())

                    p_manip_df = float(np.clip(1.0 - real_df, 0.0, 1.0))
                    p_manip_f2f = float(np.clip(1.0 - real_f2f, 0.0, 1.0))
                    manipulation_score = max(p_manip_df, p_manip_f2f)
                    real_score = 1.0 - manipulation_score

                    results.append({
                        "manipulation_score": round(manipulation_score, 4),
                        "real_score": round(real_score, 4),
                        "deepfake_score": round(p_manip_df, 4),
                        "face2face_score": round(p_manip_f2f, 4),
                        "model_inferred": True,
                    })
                    t_ptr += 1
                else:
                    results.append({"manipulation_score": 0.0, "real_score": 1.0, "model_inferred": False})

        return results

    def infer_face_crop(self, face_bgr_256: np.ndarray) -> Dict[str, float]:
        """Runs inference on a single 256x256 face crop."""
        res = self.infer_face_crops_batch([face_bgr_256], batch_size=1)
        return res[0] if res else {"manipulation_score": 0.0, "real_score": 1.0, "model_inferred": False}

    def analyze_frame(self, frame_bgr: np.ndarray, timestamp: float, frame_name: str) -> Dict[str, Any]:
        """Detects faces in frame and runs deepfake inference on each face."""
        faces = self.face_detector.detect_faces(frame_bgr)
        face_results = []

        for idx, face in enumerate(faces):
            inf = self.infer_face_crop(face.face_crop_256)
            face_results.append({
                "face_index": idx,
                "bbox": face.bbox,
                "confidence": face.confidence,
                "has_landmarks": face.landmarks is not None,
                "manipulation_score": inf["manipulation_score"],
                "real_score": inf["real_score"],
            })

        max_manip = max([r["manipulation_score"] for r in face_results], default=0.0)

        return {
            "timestamp": timestamp,
            "frame_name": frame_name,
            "face_count": len(faces),
            "faces": face_results,
            "max_manipulation_score": max_manip,
        }


# ===========================================================================
# 4. Facial Seam & Landmark Consistency Trackers
# ===========================================================================
def measure_facial_boundary_seam(img_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    """Measures edge gradient discontinuity along the facial boundary seam.
    
    Face swaps frequently leave blurred or mismatched gradient artifacts along
    the convex hull / perimeter of the inserted face mask.
    Evaluates multi-scale perimeter bands (10%, 20%, 25%) to capture variable blending radii.
    """
    x, y, w, h = bbox
    H, W = img_bgr.shape[:2]
    if w < 10 or h < 10 or x < 0 or y < 0:
        return 0.0

    max_seam = 0.0
    for margin in [0.10, 0.20, 0.25]:
        pad_x = int(w * margin)
        pad_y = int(h * margin)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(W, x + w + pad_x), min(H, y + h + pad_y)

        patch = img_bgr[y1:y2, x1:x2]
        if patch.size == 0:
            continue

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)

        ph, pw = patch.shape[:2]
        pad = max(4, int(min(w, h) * 0.08))
        if ph <= 2 * pad or pw <= 2 * pad:
            continue

        inner_mask = np.zeros((ph, pw), dtype=bool)
        inner_mask[pad:ph-pad, pad:pw-pad] = True
        outer_mask = ~inner_mask

        inner_energy = float(np.mean(grad_mag[inner_mask])) if np.any(inner_mask) else 1.0
        outer_energy = float(np.mean(grad_mag[outer_mask])) if np.any(outer_mask) else 1.0

        seam_ratio = outer_energy / max(inner_energy, 1.0)
        if seam_ratio > max_seam:
            max_seam = seam_ratio

    return round(float(max_seam), 3)
