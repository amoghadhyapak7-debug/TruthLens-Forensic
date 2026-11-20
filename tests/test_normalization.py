from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analyzers.normalization import normalize_video_verdict


def test_normalize_video_verdict_none():
    res = normalize_video_verdict(None)
    assert res["verdict"] == "INCONCLUSIVE"
    assert res["fake_probability"] == 0.0
    assert res["confidence_score"] == 0.0


def test_normalize_video_verdict_plain_string():
    res = normalize_video_verdict("DEEPFAKE DETECTED")
    assert res["verdict"] == "DEEPFAKE DETECTED"
    assert res["fake_probability"] == 0.95

    res2 = normalize_video_verdict("NO DEEPFAKE DETECTED")
    assert res2["verdict"] == "NO DEEPFAKE DETECTED"
    assert res2["fake_probability"] == 0.05


def test_normalize_video_verdict_json_string():
    raw = '{"verdict": "DEEPFAKE DETECTED", "fake_probability": 0.88, "confidence": 0.91}'
    res = normalize_video_verdict(raw)
    assert res["verdict"] == "DEEPFAKE DETECTED"
    assert abs(res["fake_probability"] - 0.88) < 1e-4
    assert abs(res["confidence_score"] - 0.91) < 1e-4


def test_normalize_video_verdict_truthlens_video_actual_output():
    # Exactly what truthlens_video.analyze_video returns!
    data = {
        "evidence_id": "EVD-001",
        "media_type": "video",
        "status": "completed",
        "assessment": {
            "classification": "DEEPFAKE DETECTED",
            "verdict": "DEEPFAKE DETECTED",
            "evidence_score": 0.946,
            "manipulation_score_pct": 94.6,
            "confidence": 0.946,
            "basis": ["Facial inconsistency detected by MesoNet", "GenD score elevated"],
        },
        "verdict": "DEEPFAKE DETECTED",  # THIS IS A STRING IN TRUTHLENS_VIDEO
        "deepfake_metrics": {
            "signal_name": "deepfake_neural_inference",
            "fake_probability": 0.946,
        },
        "findings": [{"title": "Facial artifacts", "severity": "high"}],
    }

    res = normalize_video_verdict(data)
    assert res["verdict"] == "DEEPFAKE DETECTED"
    assert abs(res["fake_probability"] - 0.946) < 1e-4
    assert abs(res["confidence_score"] - 0.946) < 1e-4
    assert len(res["basis"]) == 2


def test_normalize_video_verdict_wrapped():
    inner = {
        "verdict": "NO DEEPFAKE DETECTED",
        "assessment": {"manipulation_score_pct": 5.0, "confidence": 0.90},
    }
    res = normalize_video_verdict({"video": inner})
    assert res["verdict"] == "NO DEEPFAKE DETECTED"
    assert abs(res["fake_probability"] - 0.05) < 1e-4


def test_normalize_video_verdict_dict_verdict():
    data = {
        "verdict": {
            "verdict": "DEEPFAKE DETECTED",
            "fake_probability": 0.77,
            "confidence_score": 0.85,
        }
    }
    res = normalize_video_verdict(data)
    assert res["verdict"] == "DEEPFAKE DETECTED"
    assert abs(res["fake_probability"] - 0.77) < 1e-4
    assert abs(res["confidence_score"] - 0.85) < 1e-4


def test_normalize_video_verdict_unexpected_type():
    res = normalize_video_verdict(12345)
    assert "UNKNOWN_FORMAT" in res["verdict"]
    assert res["fake_probability"] == 0.0


def test_render_video_card_does_not_crash_on_str_verdict():
    from unittest.mock import MagicMock, patch
    from monitor.views_analyze import _render_video_card

    # Input where verdict is a raw string from truthlens_video
    sample_data = {
        "video": {
            "evidence_id": "EVD-VIDEO-01",
            "verdict": "DEEPFAKE DETECTED",
            "assessment": {
                "classification": "DEEPFAKE DETECTED",
                "verdict": "DEEPFAKE DETECTED",
                "manipulation_score_pct": 94.6,
                "confidence": 0.95,
            },
            "findings": [{"title": "MesoNet Artifacts", "severity": "high"}],
        },
        "audio": {
            "result": "NO STRONG SYNTHETIC INDICATORS DETECTED",
            "synthetic_score": 0.08,
            "wav_path": "non_existent.wav",
        },
        "audio_extraction_status": "extracted",
    }

    mock_col = MagicMock()
    with patch("streamlit.markdown"), \
         patch("streamlit.columns", side_effect=lambda n: [mock_col] * (n if isinstance(n, int) else len(n))), \
         patch("streamlit.error"), \
         patch("streamlit.success"), \
         patch("streamlit.warning"), \
         patch("streamlit.info"), \
         patch("streamlit.metric"), \
         patch("streamlit.progress"), \
         patch("streamlit.write"), \
         patch("streamlit.caption"), \
         patch("streamlit.video"), \
         patch("streamlit.audio"), \
         patch("streamlit.expander"):
        # This must execute without any AttributeError: 'str' object has no attribute 'get'
        _render_video_card("http://127.0.0.1:8000", "EVD-VIDEO-01", {}, sample_data)


def test_correlate_video_audio_with_str_verdict():
    from app.analyzers.correlator import correlate_video_audio

    # Video result has str verdict
    vid_res = {
        "verdict": "DEEPFAKE DETECTED",
        "assessment": {
            "verdict": "DEEPFAKE DETECTED",
            "manipulation_score_pct": 94.6,
        },
    }
    aud_res = {
        "status": "completed",
        "result": "NO STRONG SYNTHETIC INDICATORS DETECTED",
        "synthetic_score": 0.05,
    }

    corr = correlate_video_audio(vid_res, aud_res)
    assert corr["visual_verdict"] == "DEEPFAKE DETECTED"
    assert corr["audio_verdict"] == "NOT_DETECTED"
    assert "VISUAL DEEPFAKE WITH AUTHENTIC AUDIO" in corr["unified_verdict"]

