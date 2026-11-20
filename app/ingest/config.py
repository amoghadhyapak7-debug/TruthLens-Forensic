"""Centralized upload and analyzer eligibility limits.

Each value can be overridden with a matching TRUTHLENS_LIMIT_* environment
variable. Values expressed in bytes use MiB-based defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

MIB = 1024 * 1024


@dataclass(frozen=True)
class IngestLimits:
    max_files_per_batch: int = 10
    max_batch_bytes: int = 200 * MIB
    max_upload_file_bytes: int = 200 * MIB
    image_max_bytes: int = 20 * MIB
    image_max_pixels: int = 100_000_000
    video_max_bytes: int = 100 * MIB
    video_max_duration_seconds: float = 10.0
    audio_max_bytes: int = 25 * MIB
    audio_max_duration_seconds: float = 20.0
    document_max_bytes: int = 10 * MIB
    text_max_characters: int = 100_000
    structured_max_bytes: int = 10 * MIB
    structured_max_characters: int = 100_000

    @classmethod
    def from_env(cls) -> "IngestLimits":
        values = {}
        for field in fields(cls):
            env_name = f"TRUTHLENS_LIMIT_{field.name.upper()}"
            raw = os.getenv(env_name)
            if raw is not None:
                values[field.name] = float(raw) if field.type in (float, "float") else int(raw)
        return cls(**values)


LIMITS = IngestLimits.from_env()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}
TEXT_EXTENSIONS = {".txt", ".md"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx"}
STRUCTURED_EXTENSIONS = {".csv", ".json"}
SUPPORTED_EXTENSIONS = (
    IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS | TEXT_EXTENSIONS
    | DOCUMENT_EXTENSIONS | STRUCTURED_EXTENSIONS
)


def modality_for_extension(extension: str) -> str | None:
    extension = extension.lower()
    for modality, supported in (
        ("image", IMAGE_EXTENSIONS), ("video", VIDEO_EXTENSIONS),
        ("audio", AUDIO_EXTENSIONS), ("text", TEXT_EXTENSIONS),
        ("document", DOCUMENT_EXTENSIONS), ("structured_data", STRUCTURED_EXTENSIONS),
    ):
        if extension in supported:
            return modality
    return None
