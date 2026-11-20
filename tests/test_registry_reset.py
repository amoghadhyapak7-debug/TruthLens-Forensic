import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.ingest.service import IngestService
from app.main import app
from monitor.results import clear_registry_session_state


def _upload(service, name="one.txt", data=b"one"):
    return service.ingest(name, io.BytesIO(data))


def test_clear_registry_removes_associated_files_and_artifacts_but_preserves_history(tmp_path):
    service = IngestService(tmp_path)
    result = _upload(service)
    evidence_id = result["evidence_id"]
    record = service.get_evidence(evidence_id)
    artifact_run = "20261119T130806-a1b2c3d4"
    artifact_dir = tmp_path / "artifacts" / evidence_id / "audio" / artifact_run
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "analysis.json").write_text("{}", encoding="utf-8")
    (artifact_dir / f"{evidence_id}_audio_{artifact_run}_waveform.png").write_bytes(b"png")
    latest = artifact_dir.parent / "latest.json"
    latest.write_text("{}", encoding="utf-8")
    unregistered = tmp_path / "evidence" / "keep.txt"
    unregistered.write_text("keep", encoding="utf-8")
    project_file = tmp_path / "project.txt"
    project_file.write_text("keep", encoding="utf-8")
    activity_before = service.activity_path.read_bytes()

    summary = service.clear_evidence_registry("CLEAR REGISTRY")

    assert summary["complete"] is True
    assert summary["evidence_records_removed"] == 1
    assert summary["original_files_removed"] == 1
    assert summary["artifact_files_removed"] == 3
    assert summary["analysis_records_removed"] == 1
    assert not Path(record["file_path"]).exists()
    assert not artifact_dir.exists()
    assert service.list_evidence() == []
    assert service.activity_path.read_bytes() == activity_before
    assert unregistered.read_text(encoding="utf-8") == "keep"
    assert project_file.read_text(encoding="utf-8") == "keep"


def test_reset_requires_exact_confirmation_and_api_returns_422(tmp_path, monkeypatch):
    service = IngestService(tmp_path)
    result = _upload(service)
    with pytest.raises(ValueError):
        service.clear_evidence_registry("clear registry")
    assert Path(service.get_evidence(result["evidence_id"])["file_path"]).exists()

    import app.main as main_module
    monkeypatch.setattr(main_module, "ingest_service", service)
    client = TestClient(app)
    response = client.post("/api/v1/evidence/reset", json={"confirmation": "no"})
    assert response.status_code == 422
    assert len(service.list_evidence()) == 1
    response = client.post("/api/v1/evidence/reset", json={"confirmation": "CLEAR REGISTRY"})
    assert response.status_code == 200
    assert response.json()["complete"] is True
    assert service.list_evidence() == []


def test_reset_preserves_unsafe_registry_paths_and_reports_partial_failure(tmp_path):
    service = IngestService(tmp_path)
    result = _upload(service)
    outside = tmp_path.parent / "outside-registry-test.txt"
    outside.write_text("preserve", encoding="utf-8")
    records = service.list_evidence()
    records[0]["file_path"] = str(outside)
    service._atomic_json(service.registry_path, records)

    summary = service.clear_evidence_registry("CLEAR REGISTRY")

    assert summary["complete"] is False
    assert summary["registry_records_remaining"] == 1
    assert outside.read_text(encoding="utf-8") == "preserve"
    assert service.get_evidence(result["evidence_id"]) is not None


def test_clear_registry_session_state_resets_results_and_widget_key():
    state = {
        "upload_widget_version": 3,
        "evidence_upload_3": ["stale"],
        "current_upload_session": {"results": ["stale"]},
        "audio_analysis_result": {"done": True},
        "audio_analysis_evidence_id": "EVD-001",
        "analyzer_input_evidence": "EVD-001",
        "unrelated": "keep",
    }

    clear_registry_session_state(state)

    assert state["upload_widget_version"] == 4
    assert "evidence_upload_3" not in state
    assert "current_upload_session" not in state
    assert "audio_analysis_result" not in state
    assert "analyzer_input_evidence" not in state
    assert state["unrelated"] == "keep"


def test_missing_original_is_reported_and_record_cleared(tmp_path):
    service = IngestService(tmp_path)
    result = _upload(service)
    Path(service.get_evidence(result["evidence_id"])["file_path"]).unlink()

    summary = service.clear_evidence_registry("CLEAR REGISTRY")

    assert summary["complete"] is True
    assert summary["original_files_missing"] == 1
    assert service.list_evidence() == []


def test_malformed_registry_is_not_silently_overwritten(tmp_path):
    service = IngestService(tmp_path)
    malformed = b"{broken registry"
    service.registry_path.write_bytes(malformed)

    summary = service.clear_evidence_registry("CLEAR REGISTRY")

    assert summary["complete"] is False
    assert summary["errors"]
    assert service.registry_path.read_bytes() == malformed
