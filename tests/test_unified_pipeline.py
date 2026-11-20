"""End-to-end integration test for the Unified TruthLens Application."""
from __future__ import annotations

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_text_analysis_flow(client):
    # 1. Ingest a sample text file
    sample_text = (
        "Artificial intelligence and deepfake technologies have advanced significantly. "
        "The suspect claimed he was in Chicago on the night of November 15, 2024. "
        "Financial transactions indicate $500,000 was transferred to an offshore account."
    )
    files = [("files", ("case_notes.txt", io.BytesIO(sample_text.encode("utf-8")), "text/plain"))]
    resp = client.post("/api/v1/ingest", files=files, data={"case_id": "TEST-CASE-01"})
    assert resp.status_code == 200
    res_list = resp.json().get("results", [])
    assert len(res_list) == 1
    ev_id = res_list[0].get("evidence_id") or res_list[0].get("duplicate_reference")
    assert ev_id and ev_id.startswith("EVD-")

    # 2. Analyze via text endpoint
    t_resp = client.post(f"/api/v1/analyzers/text/{ev_id}")
    assert t_resp.status_code == 200
    t_data = t_resp.json()
    assert "ai_detection" in t_data or "status" in t_data
    assert t_data.get("evidence_id") == ev_id

    # 3. Retrieve saved result
    get_resp = client.get(f"/api/v1/analyzers/text/{ev_id}")
    assert get_resp.status_code == 200
    assert get_resp.json().get("evidence_id") == ev_id


def test_image_analysis_flow(client):
    # Create a small valid test PNG image
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    img.save(buf, format="PNG")
    buf.seek(0)

    files = [("files", ("suspect_photo.png", buf, "image/png"))]
    resp = client.post("/api/v1/ingest", files=files, data={"case_id": "TEST-CASE-01"})
    assert resp.status_code == 200
    res_list = resp.json()["results"]
    ev_id = res_list[0].get("evidence_id") or res_list[0].get("duplicate_reference")
    assert ev_id and ev_id.startswith("EVD-")

    # Analyze via image endpoint
    i_resp = client.post(f"/api/v1/analyzers/image/{ev_id}")
    assert i_resp.status_code == 200
    i_data = i_resp.json()
    assert "verdict" in i_data or "module" in i_data
    assert i_data.get("evidence_id") == ev_id


def test_review_and_reporting_flow(client):
    # Test review endpoints
    ev_id = "EVD-001"
    rev_payload = {
        "status": "VERIFIED",
        "notes": "Forensic audit confirmed visual artifacts match known GAN synthesis patterns.",
        "examiner": "Lead Examiner #42",
    }
    r_resp = client.post(f"/api/v1/review/{ev_id}", json=rev_payload)
    assert r_resp.status_code == 200
    assert r_resp.json().get("status") == "VERIFIED"

    get_rev = client.get(f"/api/v1/review/{ev_id}")
    assert get_rev.status_code == 200
    assert get_rev.json().get("examiner") == "Lead Examiner #42"

    # Test report JSON export
    rep_json = client.get("/api/v1/report/all?format=json")
    assert rep_json.status_code == 200
    rep_data = rep_json.json()
    assert "chain_of_custody" in rep_data
    assert "system" in rep_data

    # Test report PDF export
    rep_pdf = client.get("/api/v1/report/all?format=pdf")
    assert rep_pdf.status_code == 200
    assert rep_pdf.headers.get("content-type") == "application/pdf"
    assert len(rep_pdf.content) > 100


def test_video_analysis_and_correlation_flow(client):
    video_path = Path(r"c:\Users\Amogh P Adhyapak\Downloads\sapthagiri hackathon amogs part\tests\accuracy_eval_data\df_fake_swap_std.mp4")
    if not video_path.is_file():
        pytest.skip("Test video file not found")

    with open(video_path, "rb") as vf:
        files = [("files", ("df_fake_swap_std.mp4", vf, "video/mp4"))]
        resp = client.post("/api/v1/ingest", files=files, data={"case_id": "TEST-CASE-VID"})
    
    assert resp.status_code == 200
    res_list = resp.json()["results"]
    ev_id = res_list[0].get("evidence_id") or res_list[0].get("duplicate_reference")
    assert ev_id and ev_id.startswith("EVD-")

    # Analyze via video endpoint
    v_resp = client.post(f"/api/v1/analyzers/video/{ev_id}")
    assert v_resp.status_code == 200
    v_data = v_resp.json()
    assert "video" in v_data
    assert "audio_extraction_status" in v_data
    assert v_data.get("evidence_id") == ev_id

    # Correlation check
    c_resp = client.get(f"/api/v1/correlation/{ev_id}")
    assert c_resp.status_code == 200
    c_data = c_resp.json()
    assert "unified_verdict" in c_data
    assert "anomaly_type" in c_data

