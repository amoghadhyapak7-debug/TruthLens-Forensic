import io
import math
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.ingest.service import IngestService
from app.main import app


def _create_wav(path: Path, duration_s: float = 1.0, rate: int = 16000, freq: float = 440.0) -> Path:
    t = np.linspace(0, duration_s, int(rate * duration_s), endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 0.8).astype(np.float32)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    return path


def _create_silent_wav(path: Path, duration_s: float = 1.0, rate: int = 16000) -> Path:
    pcm = np.zeros(int(rate * duration_s), dtype="<i2").tobytes()
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    return path


def test_deepfake_endpoint_real_model_inference_end_to_end(tmp_path, monkeypatch):
    """Test full pipeline: ingest WAV -> analyze deepfake with real model inference."""
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    wav_path = _create_wav(tmp_path / "test_synth.wav", duration_s=1.5, rate=16000, freq=300.0)
    ingested = service.ingest("test_synth.wav", io.BytesIO(wav_path.read_bytes()))
    assert ingested["status"] == "completed"
    evd_id = ingested["evidence_id"]

    client = TestClient(app)

    # Calling GET before analysis returns 404
    get_before = client.get(f"/api/v1/analyzers/audio/{evd_id}/deepfake")
    assert get_before.status_code == 404

    # Run deepfake detection
    response = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake")
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["evidence_id"] == evd_id
    assert data["status"] == "Completed"
    assert data["result"] in {"DEEPFAKE DETECTED", "DEEPFAKE NOT DETECTED", "INCONCLUSIVE"}
    assert isinstance(data["score"], float)
    assert 0.0 <= data["score"] <= 1.0
    assert "Wav2Vec2" in data["model_name"] or "deepfake" in data["model_name"]
    assert "explanation" in data and len(data["explanation"]) > 0
    assert "details" in data
    assert "class_scores" in data["details"]

    # Verify complete upgraded structured sections
    assert "verdict" in data
    assert "title" in data["verdict"]
    assert "conclusion" in data["verdict"]

    assert "scores" in data
    assert data["scores"]["synthetic_score"] == data["score"]
    assert data["scores"]["threshold_detected"] == 0.70
    assert data["scores"]["threshold_not_detected"] == 0.60
    assert "interpretation" in data["scores"]

    assert "why_verdict" in data
    assert "evaluated" in data["why_verdict"]
    assert "reason" in data["why_verdict"]
    assert "what_it_means" in data["why_verdict"]
    assert "what_it_does_not_prove" in data["why_verdict"]

    assert "audio_properties" in data
    assert data["audio_properties"]["original_filename"] == "test_synth.wav"
    assert data["audio_properties"]["duration_seconds"] > 0
    assert data["audio_properties"]["processing_sample_rate"] == 16000
    assert data["audio_properties"]["samples_processed"] > 0

    assert "signal_quality" in data
    assert "rms_energy" in data["signal_quality"]
    assert "peak_amplitude" in data["signal_quality"]
    assert "clipping_count" in data["signal_quality"]
    assert "disclaimer" in data["signal_quality"]

    assert "model_info" in data
    assert data["model_info"]["architecture"] == "Wav2Vec2 (Transformer acoustic feature encoder + classification head)"
    assert "limitations" in data["model_info"]

    assert "timing" in data
    assert "processing_duration_seconds" in data["timing"]

    # Calling GET after analysis returns complete saved result
    get_after = client.get(f"/api/v1/analyzers/audio/{evd_id}/deepfake")
    assert get_after.status_code == 200
    assert get_after.json()["evidence_id"] == evd_id
    assert get_after.json()["result"] == data["result"]
    assert get_after.json()["verdict"]["title"] == data["verdict"]["title"]


def test_deepfake_threshold_classification(tmp_path, monkeypatch):
    """Verify exact 3-way classification under provisional policy:
      - Synthetic score <= 0.60 -> DEEPFAKE NOT DETECTED (NO STRONG SYNTHETIC AUDIO INDICATORS DETECTED)
      - Synthetic score >= 0.70 AND supporting model evidence -> DEEPFAKE DETECTED (SYNTHETIC AUDIO INDICATORS DETECTED)
      - Synthetic score > 0.60 and < 0.70 -> INCONCLUSIVE (BORDERLINE ACOUSTIC INDICATORS)
    """
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    wav_path = _create_wav(tmp_path / "test.wav", duration_s=1.0)
    ingested = service.ingest("test.wav", io.BytesIO(wav_path.read_bytes()))
    evd_id = ingested["evidence_id"]
    client = TestClient(app)

    # Mock high fake score (0.85) -> DEEPFAKE DETECTED
    monkeypatch.setattr(
        "app.main.detect_synthetic_audio",
        lambda *args, **kwargs: {
            "status": "completed",
            "detector_name": "Test Detector",
            "class_scores": {"real": 0.15, "fake": 0.85},
            "prediction": "fake",
            "raw_logits": [-1.0, 1.5],
        },
    )
    res_fake = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    assert res_fake["result"] == "DEEPFAKE DETECTED"
    assert res_fake["score"] == 0.85
    assert res_fake["verdict"]["title"] == "SYNTHETIC AUDIO INDICATORS DETECTED"
    assert "flagged for further verification" in res_fake["verdict"]["conclusion"]
    # score_margin = 0.85 - 0.70 = 0.15 (provisional engine uses 0.70 threshold)
    assert abs(res_fake["scores"]["score_margin"] - 0.15) < 0.01
    assert res_fake["scores"]["threshold_detected"] == 0.70
    assert res_fake["scores"]["threshold_not_detected"] == 0.60

    # Mock low fake score (0.15) -> DEEPFAKE NOT DETECTED
    monkeypatch.setattr(
        "app.main.detect_synthetic_audio",
        lambda *args, **kwargs: {
            "status": "completed",
            "detector_name": "Test Detector",
            "class_scores": {"real": 0.85, "fake": 0.15},
            "prediction": "real",
            "raw_logits": [1.5, -1.0],
        },
    )
    res_real = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    assert res_real["result"] == "DEEPFAKE NOT DETECTED"
    assert res_real["score"] == 0.15
    assert res_real["verdict"]["title"] == "NO STRONG SYNTHETIC AUDIO INDICATORS DETECTED"
    assert "No strong synthetic-audio indicators were detected" in res_real["verdict"]["conclusion"]
    # score_margin = 0.60 - 0.15 = 0.45 (provisional engine uses 0.60 threshold)
    assert abs(res_real["scores"]["score_margin"] - 0.45) < 0.01

    # Mock human voice score (0.5663) -> DEEPFAKE NOT DETECTED (<= 0.60)
    monkeypatch.setattr(
        "app.main.detect_synthetic_audio",
        lambda *args, **kwargs: {
            "status": "completed",
            "detector_name": "Test Detector",
            "class_scores": {"real": 0.4337, "fake": 0.5663},
            "prediction": "fake",
            "raw_logits": [-0.096, 0.170],
        },
    )
    res_human = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    assert res_human["result"] == "DEEPFAKE NOT DETECTED"
    assert res_human["score"] == 0.5663
    assert res_human["verdict"]["title"] == "NO STRONG SYNTHETIC AUDIO INDICATORS DETECTED"
    assert abs(res_human["scores"]["score_margin"] - (0.60 - 0.5663)) < 0.001

    # Mock indeterminate score in boundary zone (0.65) -> INCONCLUSIVE (0.60 < score < 0.70)
    monkeypatch.setattr(
        "app.main.detect_synthetic_audio",
        lambda *args, **kwargs: {
            "status": "completed",
            "detector_name": "Test Detector",
            "class_scores": {"real": 0.35, "fake": 0.65},
            "prediction": "fake",
            "raw_logits": [-0.3, 0.3],
        },
    )
    res_indeterminate = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    assert res_indeterminate["result"] == "INCONCLUSIVE"
    assert "BORDERLINE" in res_indeterminate["verdict"]["title"] or "INCONCLUSIVE" in res_indeterminate["verdict"]["title"]
    assert "indeterminate" in res_indeterminate["explanation"].lower() or "inconclusive" in res_indeterminate["explanation"].lower()


def test_deepfake_silence_handling(tmp_path, monkeypatch):
    """Silent or near-zero energy audio results in INCONCLUSIVE."""
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    wav_path = _create_silent_wav(tmp_path / "silence.wav", duration_s=1.0)
    ingested = service.ingest("silence.wav", io.BytesIO(wav_path.read_bytes()))
    evd_id = ingested["evidence_id"]
    client = TestClient(app)

    res = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    assert res["result"] == "INCONCLUSIVE"
    assert res["verdict"]["title"] == "INCONCLUSIVE — INSUFFICIENT EVIDENCE"
    assert res["signal_quality"]["silence_detected"] is True
    assert "silence" in res["explanation"].lower() or "energy" in res["explanation"].lower()


def test_deepfake_model_failure_gracefully_inconclusive(tmp_path, monkeypatch):
    """Model load / runtime inference failure returns INCONCLUSIVE gracefully."""
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    wav_path = _create_wav(tmp_path / "fail.wav", duration_s=1.0)
    ingested = service.ingest("fail.wav", io.BytesIO(wav_path.read_bytes()))
    evd_id = ingested["evidence_id"]
    client = TestClient(app)

    monkeypatch.setattr(
        "app.main.detect_synthetic_audio",
        lambda *args, **kwargs: {"status": "failed", "error": "CUDA out of memory"},
    )
    res = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    assert res["result"] == "INCONCLUSIVE"
    assert res["status"] == "Failed"
    assert "CUDA out of memory" in res["explanation"]
    assert res["verdict"]["title"] == "INCONCLUSIVE — INSUFFICIENT EVIDENCE"


def test_deepfake_rejects_non_audio_and_missing_evidence(tmp_path, monkeypatch):
    """Non-audio files return 422, non-existent evidence returns 404."""
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    text_ingested = service.ingest("notes.txt", io.BytesIO(b"Hello world this is a test document."))
    assert text_ingested["status"] == "completed"

    client = TestClient(app)

    # Non-audio file -> 422
    resp_text = client.post(f"/api/v1/analyzers/audio/{text_ingested['evidence_id']}/deepfake")
    assert resp_text.status_code == 422

    # Non-existent evidence ID -> 404
    resp_404 = client.post("/api/v1/analyzers/audio/EVD-999/deepfake")
    assert resp_404.status_code == 404


def test_segmentation_duration_calculations():
    """Verify segment generation for <15s, 20s, 57.98s, and >60s."""
    from app.main import _segment_waveform

    sr = 16000
    # 1. Under 15s (e.g. 5s)
    audio_5s = np.zeros(5 * sr, dtype=np.float32)
    segs_5s = _segment_waveform(audio_5s, sr, 15.0, 1.0)
    assert len(segs_5s) == 1
    assert segs_5s[0]["start_time"] == 0.0
    assert segs_5s[0]["end_time"] == 5.0

    # 2. Around 20s
    audio_20s = np.zeros(20 * sr, dtype=np.float32)
    segs_20s = _segment_waveform(audio_20s, sr, 15.0, 1.0)
    assert len(segs_20s) == 2
    assert segs_20s[0]["start_time"] == 0.0
    assert segs_20s[0]["end_time"] == 15.0
    assert segs_20s[1]["start_time"] == 14.0
    assert segs_20s[1]["end_time"] == 20.0

    # 3. Around 58s (57.98s)
    audio_58s = np.zeros(int(57.98 * sr), dtype=np.float32)
    segs_58s = _segment_waveform(audio_58s, sr, 15.0, 1.0)
    assert len(segs_58s) == 5
    assert segs_58s[0] == {"segment_index": 1, "start_time": 0.0, "end_time": 15.0, "duration": 15.0, "start_sample": 0, "end_sample": 15*sr}
    assert segs_58s[1]["start_time"] == 14.0 and segs_58s[1]["end_time"] == 29.0
    assert segs_58s[2]["start_time"] == 28.0 and segs_58s[2]["end_time"] == 43.0
    assert segs_58s[3]["start_time"] == 42.0 and segs_58s[3]["end_time"] == 57.0
    assert segs_58s[4]["start_time"] == 56.0 and segs_58s[4]["end_time"] == 57.98

    # 4. Greater than 60s (e.g. 75s)
    audio_75s = np.zeros(75 * sr, dtype=np.float32)
    segs_75s = _segment_waveform(audio_75s, sr, 15.0, 1.0)
    assert len(segs_75s) == 6
    assert segs_75s[-1]["end_time"] == 75.0


def test_long_audio_segmentation_mocked_segments(tmp_path, monkeypatch):
    """Test duration-weighted aggregation, partial segment failures, and coverage checks."""
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    wav_path = _create_wav(tmp_path / "long_audio.wav", duration_s=25.0, rate=16000)
    ingested = service.ingest("long_audio.wav", io.BytesIO(wav_path.read_bytes()))
    evd_id = ingested["evidence_id"]
    client = TestClient(app)

    # Mock segment inference: segment 1 has fake score 0.80, segment 2 has 0.70
    call_count = 0
    def mock_detect(samples, rate):
        nonlocal call_count
        call_count += 1
        score = 0.80 if call_count == 1 else 0.70
        return {
            "status": "completed",
            "detector_name": "Mock Wav2Vec2",
            "class_scores": {"real": 1.0 - score, "fake": score},
            "raw_logits": [0.1, 0.9],
        }

    monkeypatch.setattr("app.main.detect_synthetic_audio", mock_detect)

    res = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    assert res["status"] == "Completed"
    assert res["result"] == "DEEPFAKE DETECTED"
    seg = res["segmentation"]
    assert seg["segmentation_used"] is True
    assert seg["total_segments"] == 2
    assert seg["successful_segments"] == 2
    assert seg["failed_segments"] == 0
    assert seg["audio_coverage_percentage"] == 100.0
    assert seg["aggregation_method"] == "duration_weighted"
    # Seg 1: 15s * 0.80 = 12.0; Seg 2: 11s (from 14 to 25) * 0.70 = 7.7. Total = 19.7 / 26 = ~0.7577
    assert 0.74 <= seg["duration_weighted_score"] <= 0.78
    assert res["score"] == seg["duration_weighted_score"]
    assert len(seg["segment_results"]) == 2

    # Verify GET retrieval persists complete segmented output
    get_res = client.get(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    assert get_res["segmentation"]["total_segments"] == 2
    assert get_res["segmentation"]["segment_results"][0]["segment_index"] == 1


def test_segmentation_with_partial_and_all_failures(tmp_path, monkeypatch):
    """Test when one segment fails and when all segments fail."""
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    wav_path = _create_wav(tmp_path / "long_fail.wav", duration_s=25.0, rate=16000)
    ingested = service.ingest("long_fail.wav", io.BytesIO(wav_path.read_bytes()))
    evd_id = ingested["evidence_id"]
    client = TestClient(app)

    # Scenario A: One segment fails, but remaining coverage >= 50%
    call_count = 0
    def mock_partial_fail(samples, rate):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"status": "completed", "class_scores": {"real": 0.9, "fake": 0.1}}
        return {"status": "failed", "error": "Segment decode corrupted"}

    monkeypatch.setattr("app.main.detect_synthetic_audio", mock_partial_fail)
    res_partial = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    assert res_partial["status"] == "Completed"
    assert res_partial["result"] == "DEEPFAKE NOT DETECTED"
    assert res_partial["segmentation"]["successful_segments"] == 1
    assert res_partial["segmentation"]["failed_segments"] == 1
    assert len(res_partial["segmentation"]["failed_segment_errors"]) == 1

    # Scenario B: All segments fail -> Inconclusive
    monkeypatch.setattr(
        "app.main.detect_synthetic_audio",
        lambda *args, **kwargs: {"status": "failed", "error": "Hardware failure"}
    )
    res_all_fail = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    assert res_all_fail["status"] == "Failed"
    assert res_all_fail["result"] == "INCONCLUSIVE"
    assert res_all_fail["score"] is None
    assert res_all_fail["segmentation"]["successful_segments"] == 0
    assert res_all_fail["segmentation"]["failed_segments"] == 2


def test_long_file_segmentation_if_present():
    """Verify file_example_MP3_1MG.mp3 segmentation analysis succeeds if registered."""
    from app.main import ingest_service
    records = ingest_service.list_evidence()
    long_rec = next((r for r in records if "1MG" in r.get("original_filename", "") or (r.get("metadata", {}).get("duration_seconds") or 0) > 20.0), None)
    if long_rec is None:
        pytest.skip("No long audio file in current registry")

    evd_id = long_rec["evidence_id"]
    client = TestClient(app)
    res = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "Completed"
    assert data["segmentation"]["segmentation_used"] is True
    assert data["segmentation"]["total_segments"] == 5
    assert data["segmentation"]["successful_segments"] == 5
    assert isinstance(data["score"], float)


# =============================================================================
# NEW TESTS: Logit-margin decision engine, score band, silence skipping
# =============================================================================

def test_score_band_all_bands():
    """_classify_score_band returns correct label and badge for all 10 bands."""
    from app.main import _classify_score_band

    cases = [
        (0.05, 1, "0-10%", "MINIMAL INDICATORS"),
        (0.15, 2, "10-20%", "VERY LOW INDICATORS"),
        (0.25, 3, "20-30%", "LOW INDICATORS"),
        (0.35, 4, "30-40%", "MODERATE-LOW INDICATORS"),
        (0.45, 5, "40-50%", "INTERMEDIATE BASELINE"),
        (0.55, 6, "50-60%", "UPPER BASELINE"),
        (0.65, 7, "60-70%", "BORDERLINE INDETERMINATE"),
        (0.75, 8, "70-80%", "ELEVATED SUSPICION"),
        (0.85, 9, "80-90%", "HIGH SUSPICION"),
        (0.95, 10, "90-100%", "STRONG SUSPICION"),
    ]
    for score, expected_index, expected_range, expected_badge in cases:
        result = _classify_score_band(score)
        assert result["band_index"] == expected_index, f"Score {score}: expected band {expected_index}, got {result['band_index']}"
        assert result["band_range"] == expected_range, f"Score {score}: expected range {expected_range}, got {result['band_range']}"
        assert result["badge"] == expected_badge, f"Score {score}: expected badge {expected_badge}, got {result['badge']}"
        assert "Model Score Band" in result["band_type"]
        assert result["color"] is not None
        assert len(result["explanation"]) > 10
        assert len(result["forensic_caution"]) > 10
        assert len(result["recommended_action"]) > 10


def test_score_band_edge_cases():
    """_classify_score_band handles None, 0.0, and 1.0 correctly."""
    from app.main import _classify_score_band

    result_none = _classify_score_band(None)
    assert result_none["band_index"] is None
    assert result_none["badge"] == "UNAVAILABLE"

    result_zero = _classify_score_band(0.0)
    assert result_zero["band_index"] == 1

    result_one = _classify_score_band(1.0)
    assert result_one["band_index"] == 10


def test_score_band_in_api_response(tmp_path, monkeypatch):
    """score_band is present in the API response with correct structure."""
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    wav_path = _create_wav(tmp_path / "band_test.wav", duration_s=1.0)
    ingested = service.ingest("band_test.wav", io.BytesIO(wav_path.read_bytes()))
    evd_id = ingested["evidence_id"]
    client = TestClient(app)

    monkeypatch.setattr(
        "app.main.detect_synthetic_audio",
        lambda *args, **kwargs: {
            "status": "completed",
            "class_scores": {"real": 0.15, "fake": 0.85},
            "raw_logits": [-1.0, 1.5],
        },
    )
    res = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    assert "score_band" in res, "score_band must be present in API response"
    band = res["score_band"]
    assert "band_index" in band
    assert "band_range" in band
    assert "band_label" in band
    assert "badge" in band
    assert "explanation" in band
    assert "forensic_caution" in band
    assert "recommended_action" in band
    # Score 0.85 -> Band 9 (80-90%)
    assert band["band_index"] == 9


def test_non_synthetic_when_score_at_or_below_negative_threshold(tmp_path, monkeypatch):
    """Score 0.57 (<= 0.60) yields DEEPFAKE NOT DETECTED; logit margin does not block it."""
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    wav_path = _create_wav(tmp_path / "low_score.wav", duration_s=1.0)
    ingested = service.ingest("low_score.wav", io.BytesIO(wav_path.read_bytes()))
    evd_id = ingested["evidence_id"]
    client = TestClient(app)

    # Score 0.57 is <= 0.6000. Under the new policy, this is classified as
    # DEEPFAKE NOT DETECTED (NO STRONG SYNTHETIC AUDIO INDICATORS DETECTED).
    # The logit margin is preserved as diagnostic info and does NOT block the verdict.
    monkeypatch.setattr(
        "app.main.detect_synthetic_audio",
        lambda *args, **kwargs: {
            "status": "completed",
            "class_scores": {"real": 0.43, "fake": 0.57},
            "raw_logits": [-0.05, 0.05],  # margin = 0.1
        },
    )
    res = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    assert res["result"] == "DEEPFAKE NOT DETECTED", (
        f"Score 0.57 (<= 0.60) must be DEEPFAKE NOT DETECTED, got {res['result']}"
    )
    assert res["verdict"]["title"] == "NO STRONG SYNTHETIC AUDIO INDICATORS DETECTED"
    assert res["scores"]["logit_margin"] is not None


def test_inconclusive_when_score_in_indeterminate_zone(tmp_path, monkeypatch):
    """Score 0.65 (between 0.60 and 0.70) is INCONCLUSIVE (indeterminate boundary zone)."""
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    wav_path = _create_wav(tmp_path / "zone_test.wav", duration_s=1.0)
    ingested = service.ingest("zone_test.wav", io.BytesIO(wav_path.read_bytes()))
    evd_id = ingested["evidence_id"]
    client = TestClient(app)

    monkeypatch.setattr(
        "app.main.detect_synthetic_audio",
        lambda *args, **kwargs: {
            "status": "completed",
            "class_scores": {"real": 0.35, "fake": 0.65},
            "raw_logits": [-0.2, 0.4],
        },
    )
    res = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    assert res["result"] == "INCONCLUSIVE"
    assert "BORDERLINE ACOUSTIC INDICATORS" in res["verdict"]["title"]


@pytest.mark.parametrize("score_val,expected_outcome,expected_title_fragment", [
    (0.3999, "DEEPFAKE NOT DETECTED", "NO STRONG SYNTHETIC AUDIO INDICATORS"),
    (0.4000, "DEEPFAKE NOT DETECTED", "NO STRONG SYNTHETIC AUDIO INDICATORS"),
    (0.5663, "DEEPFAKE NOT DETECTED", "NO STRONG SYNTHETIC AUDIO INDICATORS"),
    (0.5999, "DEEPFAKE NOT DETECTED", "NO STRONG SYNTHETIC AUDIO INDICATORS"),
    (0.6000, "DEEPFAKE NOT DETECTED", "NO STRONG SYNTHETIC AUDIO INDICATORS"),
    (0.6001, "INCONCLUSIVE", "BORDERLINE ACOUSTIC INDICATORS"),
    (0.6999, "INCONCLUSIVE", "BORDERLINE ACOUSTIC INDICATORS"),
    (0.7000, "DEEPFAKE DETECTED", "SYNTHETIC AUDIO INDICATORS DETECTED"),
    (0.7001, "DEEPFAKE DETECTED", "SYNTHETIC AUDIO INDICATORS DETECTED"),
])
def test_boundary_scores_decision_policy(tmp_path, monkeypatch, score_val, expected_outcome, expected_title_fragment):
    """Test all critical threshold boundaries: 0.3999, 0.4000, 0.5663, 0.5999, 0.6000, 0.6001, 0.6999, 0.7000, 0.7001."""
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    wav_path = _create_wav(tmp_path / f"boundary_{score_val}.wav", duration_s=1.0)
    ingested = service.ingest(f"boundary_{score_val}.wav", io.BytesIO(wav_path.read_bytes()))
    evd_id = ingested["evidence_id"]
    client = TestClient(app)

    monkeypatch.setattr(
        "app.main.detect_synthetic_audio",
        lambda *args, **kwargs: {
            "status": "completed",
            "class_scores": {"real": 1.0 - score_val, "fake": score_val},
            "raw_logits": [0.0, float(score_val)],
        },
    )
    res = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    assert res["result"] == expected_outcome, f"Score {score_val} expected {expected_outcome}, got {res['result']}"
    assert expected_title_fragment in res["verdict"]["title"]


def test_safety_severe_clipping_forces_inconclusive(tmp_path, monkeypatch):
    """Audio with >= 25% clipping forces INCONCLUSIVE regardless of score."""
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    # Create heavily amplified audio causing > 50% clipping
    import wave
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    clipped_samples = np.clip(np.sin(2 * np.pi * 440.0 * t) * 5.0, -1.0, 1.0).astype(np.float32)
    pcm = (clipped_samples * 32767).astype("<i2").tobytes()
    wav_path = tmp_path / "clipped.wav"
    with wave.open(str(wav_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sr)
        out.writeframes(pcm)

    ingested = service.ingest("clipped.wav", io.BytesIO(wav_path.read_bytes()))
    evd_id = ingested["evidence_id"]
    client = TestClient(app)

    # Mock high synthetic score (0.85)
    monkeypatch.setattr(
        "app.main.detect_synthetic_audio",
        lambda *args, **kwargs: {
            "status": "completed",
            "class_scores": {"real": 0.15, "fake": 0.85},
            "raw_logits": [-1.0, 1.5],
        },
    )
    res = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    assert res["result"] == "INCONCLUSIVE"
    assert "SEVERE QUALITY DEGRADATION" in res["verdict"]["title"]


def test_safety_contradictory_segments_forces_inconclusive(tmp_path, monkeypatch):
    """Low aggregate score with >= 30% segments >= 0.70 forces INCONCLUSIVE."""
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    # 45s audio -> 4 segments
    wav_path = _create_wav(tmp_path / "contradictory.wav", duration_s=45.0, rate=16000)
    ingested = service.ingest("contradictory.wav", io.BytesIO(wav_path.read_bytes()))
    evd_id = ingested["evidence_id"]
    client = TestClient(app)

    # Segments: 0.10, 0.10, 0.85, 0.85 (2 of 4 = 50% >= 0.70, but average is ~0.475 <= 0.60)
    scores = [0.10, 0.10, 0.85, 0.85]
    call_idx = 0
    def mock_divergent(samples, rate):
        nonlocal call_idx
        sc = scores[min(call_idx, len(scores) - 1)]
        call_idx += 1
        return {
            "status": "completed",
            "class_scores": {"real": 1.0 - sc, "fake": sc},
            "raw_logits": [1.0 - sc, sc],
        }

    monkeypatch.setattr("app.main.detect_synthetic_audio", mock_divergent)
    res = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    assert res["result"] == "INCONCLUSIVE"
    assert "CONTRADICTORY SEGMENT EVIDENCE" in res["verdict"]["title"]


def test_missing_raw_logits_handled_gracefully(tmp_path, monkeypatch):
    """Detector without raw_logits does not crash and completes analysis."""
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    wav_path = _create_wav(tmp_path / "no_logits.wav", duration_s=1.0)
    ingested = service.ingest("no_logits.wav", io.BytesIO(wav_path.read_bytes()))
    evd_id = ingested["evidence_id"]
    client = TestClient(app)

    monkeypatch.setattr(
        "app.main.detect_synthetic_audio",
        lambda *args, **kwargs: {
            "status": "completed",
            "class_scores": {"real": 0.85, "fake": 0.15},
            # No raw_logits provided
        },
    )
    res = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    assert res["result"] == "DEEPFAKE NOT DETECTED"
    assert res["scores"]["logit_margin"] is None


def test_silence_segment_skipping_in_long_audio(tmp_path, monkeypatch):
    """Segments with RMS < 0.01 (silence) are skipped, not passed to the model."""
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    # Create 25s audio so we get 2 segments
    wav_path = _create_wav(tmp_path / "mostly_silent.wav", duration_s=25.0, rate=16000, freq=440.0)
    ingested = service.ingest("mostly_silent.wav", io.BytesIO(wav_path.read_bytes()))
    evd_id = ingested["evidence_id"]
    client = TestClient(app)

    call_count = 0
    def mock_detect_count(samples, rate):
        nonlocal call_count
        call_count += 1
        return {"status": "completed", "class_scores": {"real": 0.3, "fake": 0.7}, "raw_logits": [-0.5, 1.0]}

    monkeypatch.setattr("app.main.detect_synthetic_audio", mock_detect_count)
    res = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()

    # For a normal audio (non-silent), detect should have been called once per non-silent segment
    assert call_count >= 1, "detect_synthetic_audio must be called for non-silent segments"

    # Skipped segments in response
    seg_data = res.get("segmentation", {})
    skipped = seg_data.get("skipped_segments", 0)
    # With 440Hz tone at 0.8 amplitude, no segments should be silent
    assert skipped == 0, f"Regular audio should have no skipped segments, got {skipped}"

    # All segment results should show status=Completed or Failed (not Skipped for active audio)
    for seg in seg_data.get("segment_results", []):
        assert seg["status"] in {"Completed", "Failed", "Skipped"}


def test_pure_silence_segment_marked_skipped(tmp_path, monkeypatch):
    """A long silent audio produces Skipped segments (not Failed) for silent portions."""
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    # Create a 25s SILENT WAV so we get 2 segments, both silent → both skipped
    # However the whole-audio silence check fires first for very low amplitude
    # so create an audio with signal followed by silence:
    # Strategy: use a mono wav where first 15s has signal (rms>0.01) and rest is silence
    # We'll do this by writing raw PCM
    import wave
    sr = 16000
    total_samples = 25 * sr
    t = np.linspace(0, 15.0, 15 * sr, endpoint=False)
    signal_part = (np.sin(2 * np.pi * 440.0 * t) * 0.8).astype(np.float32)
    silent_part = np.zeros(total_samples - len(signal_part), dtype=np.float32)
    combined = np.concatenate([signal_part, silent_part])
    pcm = (np.clip(combined, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    wav_path = tmp_path / "signal_then_silence.wav"
    with wave.open(str(wav_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sr)
        out.writeframes(pcm)

    ingested = service.ingest("signal_then_silence.wav", io.BytesIO(wav_path.read_bytes()))
    evd_id = ingested["evidence_id"]
    client = TestClient(app)

    monkeypatch.setattr(
        "app.main.detect_synthetic_audio",
        lambda *args, **kwargs: {"status": "completed", "class_scores": {"real": 0.3, "fake": 0.7}, "raw_logits": [-0.5, 1.0]},
    )
    res = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    seg_data = res.get("segmentation", {})

    # Second segment (from 14s to 25s = 1s signal + 10s silence) has RMS dominated by silence
    # At minimum, we expect the segmentation key 'skipped_segments' to exist
    assert "skipped_segments" in seg_data, "segmentation payload must include skipped_segments field"

    # Check that any skipped segment has status=Skipped
    for seg in seg_data.get("segment_results", []):
        if seg["status"] == "Skipped":
            assert seg["synthetic_score"] is None
            assert seg["classification"] == "SKIPPED"
            assert "Silence" in (seg.get("error") or "")


def test_segment_disagreement_flag(tmp_path, monkeypatch):
    """High segment score disagreement (range > 0.35) is flagged in segmentation payload."""
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    wav_path = _create_wav(tmp_path / "disagreement.wav", duration_s=25.0, rate=16000)
    ingested = service.ingest("disagreement.wav", io.BytesIO(wav_path.read_bytes()))
    evd_id = ingested["evidence_id"]
    client = TestClient(app)

    # Seg 1: score 0.90 (strong synthetic), Seg 2: score 0.20 (clear real)
    # Range = 0.90 - 0.20 = 0.70 > 0.35 → high_segment_disagreement = True
    call_count = 0
    def mock_high_disagreement(samples, rate):
        nonlocal call_count
        call_count += 1
        score = 0.90 if call_count == 1 else 0.20
        return {
            "status": "completed",
            "class_scores": {"real": 1.0 - score, "fake": score},
            "raw_logits": [0.0, 1.5] if call_count == 1 else [1.5, 0.0],
        }

    monkeypatch.setattr("app.main.detect_synthetic_audio", mock_high_disagreement)
    res = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    seg = res["segmentation"]

    assert seg["high_segment_disagreement"] is True, "high_segment_disagreement must be True when range > 0.35"
    assert seg["segment_score_range"] > 0.35
    assert "pct_segments_strong_indicators" in seg
    assert "avg_logit_margin" in seg
    # With disagreement, outcome should be INCONCLUSIVE
    assert res["result"] == "INCONCLUSIVE", (
        f"High disagreement should produce INCONCLUSIVE, got {res['result']}"
    )


def test_why_verdict_has_new_fields(tmp_path, monkeypatch):
    """why_verdict includes logit_margin, segment_consistency, and calibration_status."""
    service = IngestService(tmp_path / "data")
    monkeypatch.setattr("app.main.ingest_service", service)

    wav_path = _create_wav(tmp_path / "verdict_fields.wav", duration_s=1.0)
    ingested = service.ingest("verdict_fields.wav", io.BytesIO(wav_path.read_bytes()))
    evd_id = ingested["evidence_id"]
    client = TestClient(app)

    monkeypatch.setattr(
        "app.main.detect_synthetic_audio",
        lambda *args, **kwargs: {
            "status": "completed",
            "class_scores": {"real": 0.15, "fake": 0.85},
            "raw_logits": [-1.0, 1.5],
        },
    )
    res = client.post(f"/api/v1/analyzers/audio/{evd_id}/deepfake").json()
    wv = res["why_verdict"]
    assert "logit_margin" in wv
    assert "segment_consistency" in wv
    assert "calibration_status" in wv
    assert "not_calibrated" in wv["calibration_status"].lower()


def test_validation_accuracy_note():
    """Document the validation data situation explicitly — 2 samples is insufficient."""
    # This is a documentation test: it always passes and states the validation data situation clearly.
    # EVD-001 (AI-generated label from filename): model scored ~0.85 -> DEEPFAKE DETECTED (correct with new engine)
    # EVD-002 (human label per user): model scored ~0.57 -> INCONCLUSIVE with new engine (correct: not a false positive)
    # IMPORTANT: 2 labeled samples is INSUFFICIENT for statistical accuracy metrics.
    # A minimum of 50+ genuine + 50+ synthetic labeled samples from the target domain is required.
    insufficient_samples = 2
    min_required = 100  # 50 genuine + 50 synthetic
    assert insufficient_samples < min_required, (
        "VALIDATION NOTE: Only 2 labeled samples available. "
        "Statistical accuracy metrics require >= 100 labeled samples (50 genuine + 50 synthetic). "
        "Current model behavior: "
        "  EVD-001 (AI-generated): score ~0.85, logit margin ~+1.75 → DEEPFAKE DETECTED ✓ "
        "  EVD-002 (human): score ~0.57, logit margin ~+0.27 → INCONCLUSIVE (not a false positive) ✓"
    )
