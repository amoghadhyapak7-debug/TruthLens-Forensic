"""Optional local speech transcription and synthetic-audio model adapters.

Models load lazily on explicit request and remain cached for subsequent runs.
Neither adapter assigns project-wide evidence/signal/finding identifiers.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from app.analyzers.config import SETTINGS
import json
import os

WHISPER_MODEL = SETTINGS.transcription_model
WHISPER_REVISION = SETTINGS.transcription_model_revision
WHISPER_DEVICE = SETTINGS.transcription_device
WHISPER_COMPUTE_TYPE = SETTINGS.transcription_compute_type
DEEPFAKE_MODEL = SETTINGS.deepfake_model
DEEPFAKE_REVISION = SETTINGS.deepfake_model_revision
DEEPFAKE_MODEL_FILE = SETTINGS.deepfake_model_file
DEEPFAKE_DEVICE = SETTINGS.deepfake_device
DEEPFAKE_SAMPLE_RATE = 16_000
MAX_DEEPFAKE_SECONDS = SETTINGS.deepfake_max_seconds


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@lru_cache(maxsize=2)
def _whisper_model(model_name: str, revision: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(repo_id=f"Systran/faster-whisper-{model_name}", revision=revision)
    return WhisperModel(model_path, device=device, compute_type=compute_type)


def transcribe(samples: np.ndarray, sample_rate: int, language: str | None = None) -> dict[str, Any]:
    """Run Whisper locally and return only model-provided segment times."""
    base = {"status": "unavailable", "model": WHISPER_MODEL,
            "model_id": f"Systran/faster-whisper-{WHISPER_MODEL}",
            "model_revision": WHISPER_REVISION,
            "model_family": "OpenAI Whisper, CTranslate2 conversion",
            "license": "Apache-2.0 (upstream Whisper model card); faster-whisper library is MIT",
            "model_library": "faster-whisper", "library_version": _package_version("faster-whisper"),
            "language": None, "language_detection_probability": None, "text": "", "segments": [],
            "word_timestamps_available": False, "error": None}
    if _package_version("faster-whisper") is None:
        base["error"] = "Optional dependency faster-whisper is not installed. Install requirements-audio-ml.txt."
        return base
    try:
        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate <= 0 or mono.size == 0 or not np.all(np.isfinite(mono)):
            raise ValueError("Transcription input must be non-empty finite audio with a valid sample rate.")
        model = _whisper_model(WHISPER_MODEL, WHISPER_REVISION, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)
        segments_iter, info = model.transcribe(mono, language=language, word_timestamps=False,
                                                vad_filter=True, beam_size=5)
        segments = []
        for segment in segments_iter:
            text = str(segment.text).strip()
            if text:
                segments.append({"start_seconds": float(segment.start),
                                 "end_seconds": float(segment.end), "text": text,
                                 "segment_type": "speech", "source": "whisper"})
        base.update(status="completed", language=getattr(info, "language", None),
                    language_detection_probability=getattr(info, "language_probability", None),
                    text=" ".join(item["text"] for item in segments), segments=segments,
                    model_configuration={"device": WHISPER_DEVICE, "compute_type": WHISPER_COMPUTE_TYPE,
                                         "model_revision": WHISPER_REVISION,
                                         "word_timestamps": False})
    except Exception as exc:
        base.update(status="failed", error=f"Transcription failed: {exc}")
    return base


@lru_cache(maxsize=2)
def _deepfake_model(model_id: str, revision: str, model_file: str, device: str):
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    config_path = hf_hub_download(model_id, "config.json", revision=revision)
    model_path = hf_hub_download(model_id, model_file, revision=revision)
    with open(config_path, encoding="utf-8") as source:
        config = json.load(source)
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
    available = ort.get_available_providers()
    if device.lower() in {"cuda", "gpu"}:
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDA was requested but onnxruntime-gpu/CUDAExecutionProvider is unavailable.")
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
    return session, config


def _resample_mono(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    mono = mono.reshape(-1)
    if mono.size == 0 or not np.all(np.isfinite(mono)):
        raise ValueError("Detector input must be non-empty finite audio.")
    if source_rate == target_rate:
        return mono

    # Band-limited polyphase resampling via scipy when available to prevent aliasing artifacts
    try:
        import math
        import scipy.signal
        gcd = math.gcd(source_rate, target_rate)
        up = target_rate // gcd
        down = source_rate // gcd
        resampled = scipy.signal.resample_poly(mono, up, down).astype(np.float32)
        return resampled
    except Exception:
        # Fallback to linear interpolation
        target_length = max(1, round(mono.size * target_rate / source_rate))
        source_positions = np.arange(mono.size, dtype=np.float64) / source_rate
        target_positions = np.arange(target_length, dtype=np.float64) / target_rate
        return np.interp(target_positions, source_positions, mono).astype(np.float32)


def detect_synthetic_audio(samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Run pretrained Wav2Vec2 classifier; scores are uncalibrated outputs."""
    base = {"status": "unavailable", "detector_name": "Wav2Vec2 audio spoof classifier (ONNX int4)",
            "model_id": DEEPFAKE_MODEL, "model_version": None,
            "model_revision": DEEPFAKE_REVISION, "license": "MIT (model card)",
            "training_domain": "ASVspoof 2021 PA speech anti-spoofing (model-card statement)",
            "input_preprocessing": {"target_sample_rate_hz": DEEPFAKE_SAMPLE_RATE,
                                     "channels": "mono", "max_duration_seconds": MAX_DEEPFAKE_SECONDS},
            "raw_logits": None, "class_scores": None, "score_interpretation": None,
            "calibration_status": "not_calibrated", "prediction": None, "error": None,
            "limitations": ["Model card metrics are self-reported on a balanced ASVspoof 2021 PA subset.",
                            "Performance may change across codecs, languages, environments, compression, and audio domains.",
                            "This output is an indicator and not proof of authenticity or manipulation."]}
    if _package_version("onnxruntime") is None or _package_version("huggingface-hub") is None:
        base["error"] = "Optional dependencies onnxruntime and huggingface-hub are required. Install requirements-audio-ml.txt."
        return base
    try:
        mono = _resample_mono(samples, sample_rate, DEEPFAKE_SAMPLE_RATE)
        if len(mono) / DEEPFAKE_SAMPLE_RATE > MAX_DEEPFAKE_SECONDS:
            base.update(status="unsupported", error=f"Detector accepts at most {MAX_DEEPFAKE_SECONDS:g} seconds per input.")
            return base
        session, config = _deepfake_model(DEEPFAKE_MODEL, DEEPFAKE_REVISION, DEEPFAKE_MODEL_FILE, DEEPFAKE_DEVICE)
        # The repository's Wav2Vec2 feature extractor specifies per-clip
        # normalization and 16 kHz mono input. Scores remain uncalibrated.
        normalized = ((mono - float(np.mean(mono))) / max(float(np.std(mono)), 1e-7)).astype(np.float32)
        inputs = {}
        for input_meta in session.get_inputs():
            if input_meta.name == "input_values":
                inputs[input_meta.name] = normalized[None, :]
            elif input_meta.name == "attention_mask":
                inputs[input_meta.name] = np.ones((1, len(normalized)), dtype=np.int64)
            else:
                pass  # Ignore unused optional inputs if model session does not strictly require them
        raw_outputs = session.run(None, inputs)
        if not raw_outputs:
            raise ValueError("Detector returned no model output.")
        logits = np.asarray(raw_outputs[0], dtype=np.float64).reshape(-1)
        if logits.size < 2 or not np.all(np.isfinite(logits)):
            raise ValueError("Detector returned invalid classification logits.")
        exp_scores = np.exp(logits - np.max(logits))
        scores = exp_scores / np.sum(exp_scores)
        labels = config.get("id2label", {})
        class_scores = {str(labels.get(str(index), labels.get(index, index))): float(scores[index])
                        for index in range(int(scores.shape[-1]))}
        prediction_index = int(np.argmax(scores))
        label = str(labels.get(str(prediction_index), labels.get(prediction_index, prediction_index)))
        base.update(status="completed", model_version=DEEPFAKE_REVISION,
                    raw_logits=[float(value) for value in logits.tolist()],
                    class_scores=class_scores,
                    score_interpretation="softmax-normalized class scores from model logits; not calibrated probabilities",
                    prediction=label,
                    input_preprocessing={**base["input_preprocessing"],
                                         "resampling": f"linear interpolation {sample_rate} Hz to {DEEPFAKE_SAMPLE_RATE} Hz",
                                         "model_input_sample_rate_hz": DEEPFAKE_SAMPLE_RATE,
                                         "device": session.get_providers()[0], "quantization": "int4 ONNX",
                                         "normalization": "per-clip mean/standard-deviation per model feature extractor"})
    except Exception as exc:
        base.update(status="failed", error=f"Synthetic-audio model inference failed: {exc}")
    return base


def module_status(status: str, explanation: str | None = None) -> dict[str, Any]:
    return {"status": status, **({"explanation": explanation} if explanation else {})}
