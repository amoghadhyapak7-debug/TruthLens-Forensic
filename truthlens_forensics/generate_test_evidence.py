"""
Test Evidence Generator
Generates three realistic test images for single & multi-image batch forensic verification:
1. EVD-001: Authentic camera photograph (real photographic capture, Sony camera EXIF, uniform natural sensor noise)
2. EVD-002: Spliced / manipulated image (spliced foreign patch, Photoshop software tag, timestamp mismatch, copy-move clone)
3. EVD-003: AI-generated synthetic image (stripped EXIF, high-frequency spectral artifacts, diffusion-like features)
"""
import io
import os
import hashlib
import urllib.request
import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STORAGE_DIR = os.path.join(BASE_DIR, "storage", "evidence")


def get_base_photo() -> Image.Image:
    """Retrieves standard high-quality natural photograph."""
    cache_path = os.path.join(BASE_DIR, "storage", "base_sample.jpg")
    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            pass

    urls = [
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/butterfly.jpg",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TruthLens/2.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            img.save(cache_path, quality=95)
            return img
        except Exception:
            continue

    # Fallback to rich non-periodic natural gradient if network is unavailable
    w, h = 640, 480
    rng = np.random.default_rng(42)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        arr[y, :, 0] = int(120 + 80 * (y / h))
        arr[y, :, 1] = int(150 + 60 * (1 - y / h))
        arr[y, :, 2] = int(210 - 70 * (y / h))
    arr = np.clip(arr.astype(np.float32) + rng.normal(0, 4.0, (h, w, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def create_authentic_evidence() -> dict:
    ev_id = "EVD-001"
    ev_dir = os.path.join(STORAGE_DIR, ev_id)
    os.makedirs(ev_dir, exist_ok=True)
    file_path = os.path.join(ev_dir, "original_file")

    img = get_base_photo().copy()

    # Add authentic Sony camera EXIF tags
    exif = img.getexif()
    exif[271] = "Sony"
    exif[272] = "ILCE-7RM4"
    exif[306] = "2026:08:12 11:20:15"       # DateTime
    exif[36867] = "2026:08:12 11:20:15"     # DateTimeOriginal

    img.save(file_path, format="JPEG", quality=95, exif=exif)

    with open(file_path, "rb") as f:
        content = f.read()
    sha256 = hashlib.sha256(content).hexdigest()

    return {
        "evidence_id": ev_id,
        "file_path": os.path.relpath(file_path, BASE_DIR).replace("\\", "/"),
        "original_filename": "sony_dsc08912.jpg",
        "file_type": "image",
        "mime_type": "image/jpeg",
        "file_size": len(content),
        "sha256": sha256,
        "ingestion_metadata": {"case_id": "CASE-2026-X8", "source": "field_camera_sd_card"},
    }


def create_manipulated_evidence() -> dict:
    ev_id = "EVD-002"
    ev_dir = os.path.join(STORAGE_DIR, ev_id)
    os.makedirs(ev_dir, exist_ok=True)
    file_path = os.path.join(ev_dir, "original_file")

    img = get_base_photo().copy()
    w, h = img.size
    arr = np.array(img).copy()

    # Insert a high-contrast foreign spliced patch in the upper center quadrant
    pw, ph = min(160, w // 3), min(120, h // 3)
    px, py = w // 3, h // 4
    rng = np.random.default_rng(202)
    spliced_patch = rng.integers(180, 255, (ph, pw, 3), dtype=np.uint8)
    arr[py : py + ph, px : px + pw] = spliced_patch

    # Duplicate / copy-move block to simulate content cloning
    bs = 32
    if h > bs * 2 and w > bs * 2:
        src_block = arr[10 : 10 + bs, 10 : 10 + bs].copy()
        arr[h - bs - 10 : h - 10, w - bs - 10 : w - 10] = src_block

    tampered_img = Image.fromarray(arr)

    # Add EXIF with Photoshop software signature and timestamp discrepancy
    exif = tampered_img.getexif()
    exif[271] = "Sony"
    exif[272] = "ILCE-7RM4"
    exif[305] = "Adobe Photoshop 2024 (Windows)"
    exif[306] = "2026:08:14 19:45:00"       # Modified later
    exif[36867] = "2026:08:12 11:20:15"     # Original capture

    tampered_img.save(file_path, format="JPEG", quality=88, exif=exif)

    with open(file_path, "rb") as f:
        content = f.read()
    sha256 = hashlib.sha256(content).hexdigest()

    return {
        "evidence_id": ev_id,
        "file_path": os.path.relpath(file_path, BASE_DIR).replace("\\", "/"),
        "original_filename": "tampered_composite.jpg",
        "file_type": "image",
        "mime_type": "image/jpeg",
        "file_size": len(content),
        "sha256": sha256,
        "ingestion_metadata": {"case_id": "CASE-2026-X8", "source": "suspect_laptop_download"},
    }


def create_ai_generated_evidence() -> dict:
    ev_id = "EVD-003"
    ev_dir = os.path.join(STORAGE_DIR, ev_id)
    os.makedirs(ev_dir, exist_ok=True)
    file_path = os.path.join(ev_dir, "original_file")

    w, h = 512, 512
    # Create synthetic pattern characteristic of AI diffusion / GAN checkerboard
    rng = np.random.default_rng(303)
    y, x = np.ogrid[:h, :w]

    # Strong high-frequency interference pattern (checkerboard frequency artifact)
    freq_artifact = 24.0 * np.sin(x / 3.0) * np.cos(y / 3.0)
    base = 128.0 + 50.0 * np.sin(x / 40.0) + 40.0 * np.cos(y / 40.0)

    r = np.clip(base + freq_artifact + rng.uniform(-15, 15, (h, w)), 0, 255)
    g = np.clip(base * 0.95 + freq_artifact * 1.3 + rng.uniform(-12, 12, (h, w)), 0, 255)
    b = np.clip(base * 1.05 - freq_artifact + rng.uniform(-18, 18, (h, w)), 0, 255)

    arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
    img = Image.fromarray(arr)

    # Save without ANY EXIF metadata (stripped headers typical of diffusion outputs)
    img.save(file_path, format="JPEG", quality=92)

    with open(file_path, "rb") as f:
        content = f.read()
    sha256 = hashlib.sha256(content).hexdigest()

    return {
        "evidence_id": ev_id,
        "file_path": os.path.relpath(file_path, BASE_DIR).replace("\\", "/"),
        "original_filename": "ai_midjourney_output.jpg",
        "file_type": "image",
        "mime_type": "image/jpeg",
        "file_size": len(content),
        "sha256": sha256,
        "ingestion_metadata": {"case_id": "CASE-2026-X8", "source": "social_media_scrape"},
    }


def generate_all_test_evidence() -> list:
    e1 = create_authentic_evidence()
    e2 = create_manipulated_evidence()
    e3 = create_ai_generated_evidence()
    return [e1, e2, e3]


if __name__ == "__main__":
    import json
    items = generate_all_test_evidence()
    print(json.dumps(items, indent=2))

