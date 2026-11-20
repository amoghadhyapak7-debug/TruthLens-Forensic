from types import SimpleNamespace

import numpy as np
import pytest

from app.analyzers import audio_models


def test_transcription_returns_only_model_segment_timestamps(monkeypatch):
    class Model:
        def transcribe(self, samples, **kwargs):
            assert kwargs["word_timestamps"] is False
            return iter([SimpleNamespace(start=0.2, end=1.3, text=" Hello world.")]), SimpleNamespace(language="en", language_probability=0.97)
    monkeypatch.setattr(audio_models, "_package_version", lambda name: "test-version")
    monkeypatch.setattr(audio_models, "_whisper_model", lambda *args: Model())
    result = audio_models.transcribe(np.ones(16000, dtype=np.float32), 16000)
    assert result["status"] == "completed"
    assert result["text"] == "Hello world."
    assert result["segments"] == [{"start_seconds": 0.2, "end_seconds": 1.3, "text": "Hello world.", "segment_type": "speech", "source": "whisper"}]
    assert result["word_timestamps_available"] is False
    assert result["language_detection_probability"] == 0.97


def test_transcription_unavailable_and_load_failure(monkeypatch):
    monkeypatch.setattr(audio_models, "_package_version", lambda name: None)
    unavailable = audio_models.transcribe(np.ones(100), 16000)
    assert unavailable["status"] == "unavailable" and unavailable["segments"] == []
    monkeypatch.setattr(audio_models, "_package_version", lambda name: "test")
    monkeypatch.setattr(audio_models, "_whisper_model", lambda *args: (_ for _ in ()).throw(RuntimeError("weights missing")))
    failed = audio_models.transcribe(np.ones(100), 16000)
    assert failed["status"] == "failed" and "weights missing" in failed["error"]


def test_detector_validates_and_returns_uncalibrated_model_scores(monkeypatch):
    class Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]
        def get_inputs(self):
            return [SimpleNamespace(name="input_values")]
        def run(self, _, inputs):
            assert inputs["input_values"].shape == (1, 16000)
            return [np.array([[1.0, 2.0]], dtype=np.float32)]
    monkeypatch.setattr(audio_models, "_package_version", lambda name: "test")
    monkeypatch.setattr(audio_models, "_deepfake_model", lambda *args: (Session(), {"id2label": {"0":"real", "1":"fake"}}))
    result = audio_models.detect_synthetic_audio(np.sin(np.arange(16000) / 10).astype(np.float32), 16000)
    assert result["status"] == "completed"
    assert result["prediction"] == "fake"
    assert result["calibration_status"] == "not_calibrated"
    assert "not calibrated" in result["score_interpretation"]
    assert abs(sum(result["class_scores"].values()) - 1) < 1e-6


def test_detector_rejects_long_audio_and_isolates_inference_failure(monkeypatch):
    monkeypatch.setattr(audio_models, "_package_version", lambda name: "test")
    too_long = audio_models.detect_synthetic_audio(np.ones(16000 * 21), 16000)
    assert too_long["status"] == "unsupported"
    class BadSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input_values")]
        def run(self, *args):
            raise RuntimeError("runtime failure")
    monkeypatch.setattr(audio_models, "_deepfake_model", lambda *args: (BadSession(), {"id2label": {}}))
    failed = audio_models.detect_synthetic_audio(np.ones(16000), 16000)
    assert failed["status"] == "failed" and "runtime failure" in failed["error"]


def test_nonfinite_and_empty_audio_rejected_by_preprocessing():
    with pytest.raises(ValueError):
        audio_models._resample_mono(np.array([]), 16000, 16000)
    with pytest.raises(ValueError):
        audio_models._resample_mono(np.array([np.nan]), 16000, 16000)
