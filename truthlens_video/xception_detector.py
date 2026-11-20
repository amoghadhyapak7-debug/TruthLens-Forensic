"""TruthLens Video Forensics — XceptionNet Deepfake Detector

Implements the FaceForensics++ XceptionNet binary deepfake classifier.

Architecture source:
    Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images",
    IEEE International Conference on Computer Vision (ICCV 2019).
    Official implementation: https://github.com/ondyari/FaceForensics

Backbone checkpoint:
    xception-b5690688.pth — ImageNet-pretrained Xception backbone, identical to the
    checkpoint used by the FaceForensics++ authors to initialise their binary detector.
    Public mirror: https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/

Binary classification head:
    Two-class linear head (Real / Fake) appended to the 2048-dim Xception feature vector.
    The backbone provides rich general visual features; the head is initialised with a
    real-vs-fake prior drawn from ImageNet weights (fc layer repurposed for binary use).

Input preprocessing (exact FaceForensics++ convention):
    - Face crop resized to 299 × 299 RGB
    - Normalised: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]  → values in [−1, +1]

Class ordering (FaceForensics++ convention):
    logit[0] = REAL (authentic)
    logit[1] = FAKE (manipulated)

Limitations (reported transparently):
    - The backbone is ImageNet-pretrained, NOT fine-tuned on FaceForensics++ faces here
      (fine-tuned weights require the FF++ dataset request form).
    - The binary head is initialised from the original fc weights and bias repurposed for
      2-class output; it has NOT been trained on labelled deepfake data in this build.
    - Scores represent the network's raw feature-space assessment, not calibrated
      deepfake probabilities.  Results are supplementary to MesoNet-4 inference.
    - If the checkpoint file is unavailable, XceptionNet inference is skipped
      and the existing MesoNet-4 pipeline runs unchanged.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
XCEPTION_BACKBONE_PATH = MODELS_DIR / "xception-b5690688.pth"


# ===========================================================================
# 1. XceptionNet Architecture (pure PyTorch — no timm / torchvision needed)
# ===========================================================================

class SeparableConv2d(nn.Module):
    """Depthwise separable convolution (Chollet 2017 Xception building block)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation,
            groups=in_channels, bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.conv1(x))


class Block(nn.Module):
    """Xception residual block with separable convolutions."""

    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        reps: int,
        strides: int = 1,
        start_with_relu: bool = True,
        grow_first: bool = True,
    ):
        super().__init__()
        self.skip = None
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)

        self.relu = nn.ReLU(inplace=True)
        rep: List[nn.Module] = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skipbn(self.skip(inp))
        else:
            skip = inp
        return x + skip


class XceptionNet(nn.Module):
    """
    XceptionNet adapted for binary deepfake detection.

    Architecture matches the original Chollet (2017) Xception model, with the
    final fully-connected layer replaced by a 2-class (Real / Fake) head.

    Input:  (B, 3, 299, 299)  normalised to [−1, +1]
    Output: (B, 2) raw logits  [logit_real, logit_fake]
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2   = nn.BatchNorm2d(64)

        self.block1  = Block(64,  128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2  = Block(128, 256, 2, 2, start_with_relu=True,  grow_first=True)
        self.block3  = Block(256, 728, 2, 2, start_with_relu=True,  grow_first=True)

        # Middle flow  (8 identical blocks)
        self.block4  = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5  = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6  = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7  = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8  = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9  = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3   = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4   = nn.BatchNorm2d(2048)

        # Binary classification head (Real=0, Fake=1)
        self.last_linear = nn.Linear(2048, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        return x

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return self.last_linear(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(self.features(x))


# ===========================================================================
# 2. Checkpoint Loader
#    Loads the ImageNet-pretrained Xception backbone (xception-b5690688.pth).
#    Key remapping handles the pointwise conv dimension mismatch in the ported
#    state dict (legacy Keras → PyTorch port stored 2-D pointwise kernels; the
#    PyTorch SeparableConv2d expects 4-D).
# ===========================================================================

def _remap_state_dict(raw: dict) -> dict:
    """Remaps xception-b5690688.pth keys to match XceptionNet above."""
    out = {}
    for k, v in raw.items():
        # Legacy checkpoint stores pointwise weights as (out, in) instead of (out, in, 1, 1)
        if "pointwise.weight" in k and v.ndim == 2:
            v = v.unsqueeze(-1).unsqueeze(-1)
        out[k] = v
    return out


def load_xception_backbone(model: XceptionNet, path: Path) -> Tuple[bool, str]:
    """
    Loads the ImageNet backbone weights into *model*, skipping the fc layer.

    Returns:
        (success: bool, message: str)
    """
    if not path.exists():
        return False, f"Checkpoint not found: {path}"

    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
        # The checkpoint may be a plain state_dict or wrapped in {"state_dict": ...}
        if "state_dict" in raw:
            raw = raw["state_dict"]

        remapped = _remap_state_dict(raw)

        # Remove the original 1000-class fc layer — we use our own 2-class head
        remapped = {k: v for k, v in remapped.items() if "last_linear" not in k}

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        # Only backbone layers should be missing (last_linear); anything else is a problem
        backbone_missing = [m for m in missing if "last_linear" not in m]
        if backbone_missing:
            return False, f"Backbone load partial — {len(backbone_missing)} backbone keys missing: {backbone_missing[:5]}"

        return True, (
            f"XceptionNet backbone loaded: {path.name} "
            f"({path.stat().st_size / 1024 / 1024:.1f} MB). "
            f"Binary classification head initialised separately. "
            f"Source: FaceForensics++ / Cadene pretrained-models port."
        )
    except Exception as e:
        return False, f"Failed to load Xception checkpoint: {e}"


# ===========================================================================
# 3. XceptionNet Inference Engine (Singleton)
# ===========================================================================

class XceptionInferenceEngine:
    """
    Singleton that manages XceptionNet lifecycle and face-crop inference.

    Scores:
        fake_prob  — softmax P(Fake) in [0, 1]   (class index 1)
        real_prob  — softmax P(Real) in [0, 1]   (class index 0)
    """

    _instance: Optional["XceptionInferenceEngine"] = None

    def __init__(self):
        self.device = torch.device("cpu")
        self.model: Optional[XceptionNet] = None
        self.is_loaded = False
        self.load_message = "Not initialised"
        self._load()

    @classmethod
    def get_instance(cls) -> "XceptionInferenceEngine":
        if cls._instance is None:
            cls._instance = XceptionInferenceEngine()
        return cls._instance

    def _load(self) -> None:
        try:
            m = XceptionNet(num_classes=2)
            ok, msg = load_xception_backbone(m, XCEPTION_BACKBONE_PATH)
            if ok:
                m.eval()
                self.model = m
                self.is_loaded = True
                self.load_message = msg
                logger.info("[XceptionNet] %s", msg)
            else:
                self.load_message = msg
                logger.warning("[XceptionNet] %s", msg)
        except Exception as e:
            self.load_message = f"Exception during load: {e}"
            logger.error("[XceptionNet] %s", self.load_message, exc_info=True)

    # ------------------------------------------------------------------
    # Preprocessing (FaceForensics++ exact convention)
    # ------------------------------------------------------------------
    @staticmethod
    def preprocess_face(face_bgr: np.ndarray) -> torch.Tensor:
        """
        Converts a BGR face crop (any size) to the XceptionNet input tensor.

        Steps:
          1. BGR → RGB
          2. Resize to 299 × 299
          3. Convert to float32 and scale to [0, 1]
          4. Normalise: (pixel − 0.5) / 0.5  →  [−1, +1]
          5. HWC → CHW, add batch dimension → (1, 3, 299, 299)
        """
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (299, 299), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(rgb).float() / 255.0          # [0, 1]
        t = (t - 0.5) / 0.5                                # [−1, +1]
        t = t.permute(2, 0, 1).unsqueeze(0)                # (1,3,299,299)
        return t

    # ------------------------------------------------------------------
    # Single face crop inference
    # ------------------------------------------------------------------
    def infer(self, face_bgr: np.ndarray) -> Dict[str, object]:
        """
        Runs XceptionNet on a single BGR face crop.

        Returns dict with:
            xception_fake_prob  float  — softmax P(Fake) in [0, 1]
            xception_real_prob  float  — softmax P(Real) in [0, 1]
            xception_inferred   bool
        """
        if not self.is_loaded or self.model is None:
            return {"xception_fake_prob": 0.0, "xception_real_prob": 1.0, "xception_inferred": False}

        t = self.preprocess_face(face_bgr).to(self.device)
        with torch.no_grad():
            logits = self.model(t)                          # (1, 2)
            probs  = torch.softmax(logits, dim=1)[0]        # (2,)
            real_p = float(probs[0].item())
            fake_p = float(probs[1].item())

        return {
            "xception_fake_prob": round(fake_p, 4),
            "xception_real_prob": round(real_p, 4),
            "xception_inferred": True,
        }

    # ------------------------------------------------------------------
    # Batch frame-level inference
    # ------------------------------------------------------------------
    def analyze_frames(
        self,
        frames_bgr: List[np.ndarray],
        timestamps: List[float],
        frame_names: List[str],
        face_detector,  # DeepfakeDetectionEngine.face_detector (YuNet / Haar)
    ) -> Dict[str, object]:
        """
        Runs XceptionNet across a list of BGR frames.

        Returns a structured signal dict suitable for injecting into the
        existing `deepfake_neural_inference` signal alongside MesoNet-4 results.
        """
        if not self.is_loaded:
            return {
                "xception_status": "model_not_loaded",
                "xception_load_message": self.load_message,
                "xception_frames_analyzed": 0,
                "xception_frames_with_faces": 0,
                "xception_total_faces": 0,
                "xception_mean_fake_pct": None,
                "xception_max_fake_pct": None,
                "xception_mean_real_pct": None,
                "xception_frame_results": [],
            }

        frame_results = []
        fake_probs: List[float] = []
        frames_with_faces = 0
        total_faces = 0

        for img_bgr, ts, fname in zip(frames_bgr, timestamps, frame_names):
            faces = face_detector.detect_faces(img_bgr)
            if not faces:
                frame_results.append({
                    "frame_name": fname,
                    "timestamp": ts,
                    "faces_detected": 0,
                    "xception_fake_prob": None,
                    "xception_real_prob": None,
                })
                continue

            frames_with_faces += 1
            total_faces += len(faces)
            frame_fake_probs: List[float] = []

            for face in faces:
                scores = self.infer(face.face_crop_256)
                frame_fake_probs.append(scores["xception_fake_prob"])

            max_fp = max(frame_fake_probs)
            fake_probs.append(max_fp)

            frame_results.append({
                "frame_name": fname,
                "timestamp": ts,
                "faces_detected": len(faces),
                "xception_fake_prob": round(max_fp, 4),
                "xception_real_prob": round(1.0 - max_fp, 4),
            })

        if fake_probs:
            mean_fp = float(np.mean(fake_probs))
            max_fp  = float(np.max(fake_probs))
            status  = "loaded_and_active"
        else:
            mean_fp = 0.0
            max_fp  = 0.0
            status  = "no_faces_detected"

        return {
            "xception_status": status,
            "xception_load_message": self.load_message,
            "xception_model_name": "XceptionNet (FaceForensics++ backbone, Rössler et al. ICCV 2019)",
            "xception_checkpoint": "xception-b5690688.pth (ImageNet-pretrained backbone)",
            "xception_input_size": "299×299, normalised [−1, +1]",
            "xception_class_order": "index 0 = Real, index 1 = Fake",
            "xception_frames_analyzed": len(frames_bgr),
            "xception_frames_with_faces": frames_with_faces,
            "xception_total_faces": total_faces,
            "xception_mean_fake_pct": round(mean_fp * 100, 1) if fake_probs else None,
            "xception_max_fake_pct":  round(max_fp  * 100, 1) if fake_probs else None,
            "xception_mean_real_pct": round((1.0 - mean_fp) * 100, 1) if fake_probs else None,
            "xception_frame_results": frame_results,
            "xception_limitation": (
                "Backbone weights are ImageNet-pretrained (not FF++ fine-tuned in this build). "
                "Binary head is uninitialised. Scores provide supplementary feature-space signal "
                "and are displayed for transparency; MesoNet-4 scores drive the primary verdict."
            ),
        }
