"""TruthLens Forensic Model Checkpoint Downloader.

Downloads the large pretrained model weights (GenD CLIP ViT-L/14, XceptionNet)
that exceed GitHub's 100 MB repository file size limit.
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "truthlens_video" / "models"
CIFAKE_DIR = MODELS_DIR / "cifake"

MODELS = {
    "GenD CLIP ViT-L/14": {
        "url": "https://huggingface.co/yermandy/GenD_CLIP_L_14/resolve/main/gend_clip_vit_l14.safetensors",
        "path": MODELS_DIR / "gend_clip_vit_l14.safetensors",
        "size_desc": "1.16 GB",
    },
    "XceptionNet Pretrained Weights": {
        "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/xception-b5690688.pth",
        "path": MODELS_DIR / "xception-b5690688.pth",
        "size_desc": "87.4 MB",
    },
}


def download_models() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CIFAKE_DIR.mkdir(parents=True, exist_ok=True)

    for name, info in MODELS.items():
        p = info["path"]
        if p.exists() and p.stat().st_size > 1000:
            logger.info("[OK] %s already present at %s", name, p)
        else:
            logger.info("Downloading %s (%s) to %s...", name, info["size_desc"], p)
            try:
                urllib.request.urlretrieve(info["url"], str(p))
                logger.info("[DONE] %s downloaded successfully.", name)
            except Exception as exc:
                logger.warning(
                    "[NOTE] Automatic download failed (%s). Please manually download from: %s",
                    exc,
                    info["url"],
                )


if __name__ == "__main__":
    download_models()
