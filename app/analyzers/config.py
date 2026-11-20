"""Audio signal-analysis configuration, centralized and environment overrideable."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioAnalyzerSettings:
    analysis_sample_rate: int = 24_000
    max_decoded_samples: int = 8_000_000
    frame_seconds: float = 0.025
    hop_seconds: float = 0.010
    silence_threshold_dbfs: float = -45.0
    silence_min_seconds: float = 0.20
    high_silence_ratio: float = 0.50
    clipping_threshold: float = 0.999
    clipping_min_ratio: float = 0.001
    discontinuity_mad_multiplier: float = 7.0
    high_frequency_floor_ratio: float = 0.002
    voiced_f0_min_hz: float = 70.0
    voiced_f0_max_hz: float = 400.0
    voiced_periodicity_min: float = 0.35
    transcription_model: str = "tiny"
    transcription_model_revision: str = "d90ca5fe260221311c53c58e660288d3deb8d356"
    transcription_device: str = "cpu"
    transcription_compute_type: str = "int8"
    deepfake_model: str = "ai8shiro/deepfake-audio-wav2vec2-ONNX"
    deepfake_model_revision: str = "78a7991b7f413bd6ce15efd7cb27000736acc78a"
    deepfake_model_file: str = "onnx/model_q4.onnx"
    deepfake_device: str = "cpu"
    deepfake_max_seconds: float = 20.0

    @classmethod
    def from_env(cls) -> "AudioAnalyzerSettings":
        return cls(
            analysis_sample_rate=int(os.getenv("TRUTHLENS_AUDIO_SAMPLE_RATE", cls.analysis_sample_rate)),
            max_decoded_samples=int(os.getenv("TRUTHLENS_AUDIO_MAX_DECODED_SAMPLES", cls.max_decoded_samples)),
            frame_seconds=float(os.getenv("TRUTHLENS_AUDIO_FRAME_SECONDS", cls.frame_seconds)),
            hop_seconds=float(os.getenv("TRUTHLENS_AUDIO_HOP_SECONDS", cls.hop_seconds)),
            silence_threshold_dbfs=float(os.getenv("TRUTHLENS_AUDIO_SILENCE_DBFS", cls.silence_threshold_dbfs)),
            silence_min_seconds=float(os.getenv("TRUTHLENS_AUDIO_SILENCE_MIN_SECONDS", cls.silence_min_seconds)),
            high_silence_ratio=float(os.getenv("TRUTHLENS_AUDIO_HIGH_SILENCE_RATIO", cls.high_silence_ratio)),
            clipping_threshold=float(os.getenv("TRUTHLENS_AUDIO_CLIPPING_THRESHOLD", cls.clipping_threshold)),
            clipping_min_ratio=float(os.getenv("TRUTHLENS_AUDIO_CLIPPING_MIN_RATIO", cls.clipping_min_ratio)),
            discontinuity_mad_multiplier=float(os.getenv("TRUTHLENS_AUDIO_DISCONTINUITY_MAD", cls.discontinuity_mad_multiplier)),
            high_frequency_floor_ratio=float(os.getenv("TRUTHLENS_AUDIO_HIGH_FREQ_FLOOR_RATIO", cls.high_frequency_floor_ratio)),
            voiced_f0_min_hz=float(os.getenv("TRUTHLENS_AUDIO_F0_MIN_HZ", cls.voiced_f0_min_hz)),
            voiced_f0_max_hz=float(os.getenv("TRUTHLENS_AUDIO_F0_MAX_HZ", cls.voiced_f0_max_hz)),
            voiced_periodicity_min=float(os.getenv("TRUTHLENS_AUDIO_VOICED_PERIODICITY", cls.voiced_periodicity_min)),
            transcription_model=os.getenv("TRUTHLENS_TRANSCRIPTION_MODEL", cls.transcription_model),
            transcription_model_revision=os.getenv("TRUTHLENS_TRANSCRIPTION_REVISION", cls.transcription_model_revision),
            transcription_device=os.getenv("TRUTHLENS_TRANSCRIPTION_DEVICE", cls.transcription_device),
            transcription_compute_type=os.getenv("TRUTHLENS_TRANSCRIPTION_COMPUTE", cls.transcription_compute_type),
            deepfake_model=os.getenv("TRUTHLENS_AUDIO_DEEPFAKE_MODEL", cls.deepfake_model),
            deepfake_model_revision=os.getenv("TRUTHLENS_AUDIO_DEEPFAKE_REVISION", cls.deepfake_model_revision),
            deepfake_model_file=os.getenv("TRUTHLENS_AUDIO_DEEPFAKE_FILE", cls.deepfake_model_file),
            deepfake_device=os.getenv("TRUTHLENS_AUDIO_DEEPFAKE_DEVICE", cls.deepfake_device),
            deepfake_max_seconds=float(os.getenv("TRUTHLENS_AUDIO_DEEPFAKE_MAX_SECONDS", cls.deepfake_max_seconds)),
        )


SETTINGS = AudioAnalyzerSettings.from_env()
