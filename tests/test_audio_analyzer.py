import io
import json
import math
import struct
import wave
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.analyzers.audio import AudioAnalyzer
from app.ingest.service import IngestService
from app.main import app


def wav_file(path: Path, samples: np.ndarray, rate: int = 24000, channels: int = 1) -> Path:
    values = np.asarray(samples)
    if values.ndim == 1:
        values = values[:, None]
    if channels > 1 and values.shape[1] == 1:
        values = np.repeat(values, channels, axis=1)
    values = np.clip(values, -1, 1)
    pcm = (values * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as out:
        out.setnchannels(values.shape[1])
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    return path


def ingest_wav(service: IngestService, path: Path, name: str = "sample.wav") -> dict:
    result = service.ingest(name, io.BytesIO(path.read_bytes()))
    assert result["status"] == "completed"
    return result


def test_wav_silence_signal_contract_artifacts_and_immutability(tmp_path):
    source = wav_file(tmp_path / "silence.wav", np.zeros(12000))
    original = source.read_bytes()
    service = IngestService(tmp_path / "data")
    uploaded = ingest_wav(service, source)
    contract = service.build_analyzer_input(uploaded["evidence_id"])
    output = AudioAnalyzer(tmp_path / "artifacts").analyze(contract)
    assert set(contract) == {"evidence_id", "file_path", "original_filename", "file_type", "mime_type", "file_size", "sha256", "ingestion_metadata"}
    assert output["analysis_status"] == "COMPLETED", output["errors"]
    assert output["audio_metadata"]["duration_seconds"] == 0.5
    assert "HIGH_SILENCE_RATIO" in {s["signal_type"] for s in output["signals"]}
    assert output["model_info"]["available"] is False
    assert output["findings"][0]["supporting_signal_indices"] == [1]
    assert source.read_bytes() == original
    assert all(Path(a["path"]).is_file() for a in output["artifacts"])
    assert all(uploaded["evidence_id"] in a["name"] for a in output["artifacts"] if a["name"] != "analysis.json")
    saved = json.loads(next(Path(output["execution_metadata"]["artifact_directory"]).glob("*.json")).read_text())
    assert saved["analysis_status"] == "COMPLETED"


def test_voiced_candidate_clipping_and_channels(tmp_path):
    rate = 24000
    t = np.arange(rate) / rate
    samples = np.sin(2 * np.pi * 180 * t) * 1.0
    path = wav_file(tmp_path / "voice.wav", samples, rate, channels=2)
    service = IngestService(tmp_path / "data")
    uploaded = ingest_wav(service, path)
    output = AudioAnalyzer(tmp_path / "artifacts").analyze(service.build_analyzer_input(uploaded["evidence_id"]))
    assert output["analysis_status"] == "COMPLETED", output["errors"]
    assert output["audio_metadata"]["channel_count"] == 2
    assert output["audio_metadata"]["bit_depth"] == 16
    assert output["audio_metadata"]["clipping_detected"]
    assert output["audio_metadata"]["estimated_f0_hz"]["median"] is not None
    assert "SPEECH_ACTIVITY_CANDIDATE" in {s["signal_type"] for s in output["signals"]}


def test_mp3_and_m4a_decode_if_ffmpeg_available(tmp_path):
    import imageio_ffmpeg
    import subprocess

    for ext, codec in (("mp3", "libmp3lame"), ("m4a", "aac")):
        path = tmp_path / f"tone.{ext}"
        completed = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5", "-c:a", codec, "-y", str(path)], capture_output=True, timeout=60)
        assert completed.returncode == 0, completed.stderr
        service = IngestService(tmp_path / f"data-{ext}")
        uploaded = ingest_wav(service, path, path.name)
        output = AudioAnalyzer(tmp_path / f"artifacts-{ext}").analyze(service.build_analyzer_input(uploaded["evidence_id"]))
        assert output["analysis_status"] == "COMPLETED", output["errors"]
        assert output["audio_metadata"]["sample_rate"] == 24000


def test_api_analysis_rejects_not_ready_and_serves_artifacts(tmp_path, monkeypatch):
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)
    path = wav_file(tmp_path / "api.wav", np.sin(2 * math.pi * 200 * np.arange(4000) / 8000), 8000)
    uploaded = ingest_wav(service, path, "api.wav")
    client = TestClient(app)
    response = client.post(f"/api/v1/analyzers/audio/{uploaded['evidence_id']}")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["analysis_status"] == "COMPLETED"
    assert client.get(f"/api/v1/analyzers/audio/{uploaded['evidence_id']}").json()["analysis_status"] == "COMPLETED"
    assert "modules" in client.get(f"/api/v1/analyzers/audio/{uploaded['evidence_id']}/modules").json()
    assert "timeline_segments" in client.get(f"/api/v1/analyzers/audio/{uploaded['evidence_id']}/timeline").json()
    assert "status" in client.get(f"/api/v1/analyzers/audio/{uploaded['evidence_id']}/transcription").json()
    assert client.get(f"/api/v1/evidence/{uploaded['evidence_id']}/content").status_code == 200
    artifact = next(item for item in data["artifacts"] if item["media_type"] == "image/png")
    asset = client.get(f"/api/v1/analyzers/audio/{uploaded['evidence_id']}/artifacts/{data['execution_metadata']['run_id']}/{artifact['name']}")
    assert asset.status_code == 200 and asset.headers["content-type"] == "image/png"
    assert client.get(f"/api/v1/analyzers/audio/{uploaded['evidence_id']}/artifacts/../x/{artifact['name']}").status_code == 404
    monkeypatch.setattr("app.analyzers.audio.transcribe", lambda *args, **kwargs: {"status":"completed", "model":"mock", "text":"test", "segments":[{"start_seconds":0.1,"end_seconds":0.3,"text":"test"}],"language":"en"})
    monkeypatch.setattr("app.analyzers.audio.detect_synthetic_audio", lambda *args, **kwargs: {"status":"unavailable", "error":"mock missing"})
    optional = client.post(f"/api/v1/analyzers/audio/{uploaded['evidence_id']}?transcribe_speech=true&detect_synthetic_audio=true")
    assert optional.status_code == 200
    assert optional.json()["analysis_status"] == "PARTIAL"
    assert optional.json()["transcription"]["segments"][0]["start_seconds"] == 0.1
    assert optional.json()["modules"]["synthetic_audio_detection"]["status"] == "unavailable"


def test_contract_integrity_and_unsupported_rejected(tmp_path):
    source = wav_file(tmp_path / "good.wav", np.zeros(8000))
    service = IngestService(tmp_path / "data")
    uploaded = ingest_wav(service, source)
    contract = service.build_analyzer_input(uploaded["evidence_id"])
    bad = dict(contract, sha256="0" * 64)
    assert AudioAnalyzer(tmp_path / "artifacts").analyze(bad)["analysis_status"] == "FAILED"
    assert AudioAnalyzer(tmp_path / "artifacts").analyze(dict(contract, extra="field"))["analysis_status"] == "FAILED"


def test_optional_model_failure_keeps_signal_analysis_available(tmp_path, monkeypatch):
    source = wav_file(tmp_path / "partial.wav", np.sin(np.arange(8000) * 0.1), 8000)
    service = IngestService(tmp_path / "data")
    uploaded = ingest_wav(service, source)
    monkeypatch.setattr("app.analyzers.audio.transcribe", lambda *args, **kwargs: {"status":"failed", "text":"", "segments":[], "error":"model load error"})
    monkeypatch.setattr("app.analyzers.audio.detect_synthetic_audio", lambda *args, **kwargs: {"status":"unavailable", "error":"model unavailable", "prediction":None})
    result = AudioAnalyzer(tmp_path / "artifacts").analyze(service.build_analyzer_input(uploaded["evidence_id"]), run_transcription=True, run_synthetic_detection=True)
    assert result["analysis_status"] == "PARTIAL"
    assert result["modules"]["signal_forensics"]["status"] == "completed"
    assert result["modules"]["transcription"]["status"] == "failed"
    assert result["modules"]["synthetic_audio_detection"]["status"] == "unavailable"
    assert result["audio_metadata"]["duration_seconds"] == 1.0
    assert any(error["stage"] == "TRANSCRIPTION" for error in result["errors"])
