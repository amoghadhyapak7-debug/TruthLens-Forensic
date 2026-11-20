import io
import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.ingest.service import IngestService
from app.main import app


def test_valid_upload_hash_id_and_metadata(tmp_path):
    service = IngestService(tmp_path)
    result = service.ingest('sample.txt', io.BytesIO(b'hello truthlens'))
    assert result['status'] == 'completed'
    assert result['evidence_id'] == 'EVD-001'
    assert result['sha256'] == '19dc11bd76552384898c08f166b65dd07b19869d18a9fb4f18d28f1cae785fa3'
    record = service.get_evidence('EVD-001')
    assert record['metadata']['encoding'] == 'utf-8'
    assert Path(record['file_path']).read_bytes() == b'hello truthlens'


def test_empty_unsupported_oversized_and_bad_image(tmp_path):
    service = IngestService(tmp_path, max_file_size=4)
    assert service.ingest('empty.txt', io.BytesIO(b''))['status'] == 'failed'
    assert service.ingest('bad.exe', io.BytesIO(b'abc'))['status'] == 'failed'
    assert service.ingest('large.txt', io.BytesIO(b'12345'))['status'] == 'failed'
    bad_image = IngestService(tmp_path / 'images').ingest('broken.png', io.BytesIO(b'not an image'))
    assert bad_image['status'] == 'completed'
    assert bad_image['errors']
    assert bad_image['analysis_eligibility']['status'] == 'NOT_READY'


def test_duplicate_and_persistence_after_restart(tmp_path):
    first_service = IngestService(tmp_path)
    first = first_service.ingest('first.txt', io.BytesIO(b'same bytes'))
    duplicate = first_service.ingest('other.txt', io.BytesIO(b'same bytes'))
    assert first['evidence_id'] == 'EVD-001'
    assert duplicate['status'] == 'duplicate'
    assert duplicate['duplicate_reference'] == 'EVD-001'
    restarted = IngestService(tmp_path)
    second = restarted.ingest('second.txt', io.BytesIO(b'different'))
    assert second['evidence_id'] == 'EVD-002'
    assert len(restarted.list_evidence()) == 2


def test_api_multiple_uploads_one_failing_and_registry(tmp_path, monkeypatch):
    import app.main as main_module
    monkeypatch.setattr(main_module, 'ingest_service', IngestService(tmp_path))
    client = TestClient(app)
    response = client.post('/api/v1/ingest', files=[('files', ('ok.txt', b'ok', 'text/plain')), ('files', ('bad.exe', b'no', 'application/octet-stream'))])
    assert response.status_code == 200
    results = response.json()['results']
    assert results[0]['status'] == 'completed'
    assert results[1]['status'] == 'failed'
    assert client.get('/api/v1/evidence').json()['evidence'][0]['evidence_id'] == 'EVD-001'
    assert client.get('/api/v1/evidence/EVD-001').status_code == 200
    assert client.get('/api/v1/evidence/EVD-999').status_code == 404
    assert client.get('/api/v1/ingest/status').json()['failed'] == 1
    assert client.get('/api/v1/ingest/activity').json()['activity']


def test_registry_diagnostics_tracks_persisted_api_records(tmp_path, monkeypatch):
    import app.main as main_module
    service = IngestService(tmp_path)
    monkeypatch.setattr(main_module, 'ingest_service', service)
    client = TestClient(app)
    first = client.post('/api/v1/ingest', files=[('files', ('one.txt', b'one', 'text/plain'))]).json()['results'][0]
    duplicate = client.post('/api/v1/ingest', files=[('files', ('one-copy.txt', b'one', 'text/plain'))]).json()['results'][0]
    second = client.post('/api/v1/ingest', files=[('files', ('two.txt', b'two', 'text/plain'))]).json()['results'][0]
    assert first['status'] == 'completed' and first['evidence_id'] == 'EVD-001'
    assert duplicate['status'] == 'duplicate' and duplicate['duplicate_reference'] == 'EVD-001'
    assert second['status'] == 'completed' and second['evidence_id'] == 'EVD-002'
    evidence = client.get('/api/v1/evidence').json()['evidence']
    assert client.get('/api/v1/diagnostics/registry').status_code == 404
    diagnostics = client.get('/api/v1/diagnostics/registry?debug=true').json()
    assert len(evidence) == diagnostics['persisted_record_count'] == 2
    assert diagnostics['latest_evidence_id'] == 'EVD-002'
    assert Path(diagnostics['registry_path']).exists()


def test_api_analyzer_contract_and_batch_caps(tmp_path, monkeypatch):
    from dataclasses import replace
    import app.main as main_module
    from app.ingest.config import LIMITS
    monkeypatch.setattr(main_module, 'ingest_service', IngestService(tmp_path))
    monkeypatch.setattr(main_module, 'LIMITS', replace(LIMITS, max_files_per_batch=10, max_batch_bytes=5, max_upload_file_bytes=5))
    client = TestClient(app)
    uploaded = client.post('/api/v1/ingest', files=[('files', ('contract.txt', b'valid', 'text/plain'))]).json()['results'][0]
    evidence_id = uploaded['evidence_id']
    readiness = client.get(f'/api/v1/evidence/{evidence_id}/readiness')
    contract = client.get(f'/api/v1/evidence/{evidence_id}/analyzer-input')
    all_readiness = client.get('/api/v1/evidence/readiness').json()['readiness']
    assert readiness.json()['status'] in {'READY', 'READY_WITH_WARNINGS'}
    assert all_readiness[evidence_id]['status'] == readiness.json()['status']
    assert contract.status_code == 200
    assert set(contract.json()) == {'evidence_id', 'file_path', 'original_filename', 'file_type', 'mime_type', 'file_size', 'sha256', 'ingestion_metadata'}
    assert client.get('/api/v1/evidence/EVD-404/analyzer-input').status_code == 404
    too_many_bytes = client.post('/api/v1/ingest', files=[('files', ('a.txt', b'abc', 'text/plain')), ('files', ('b.txt', b'def', 'text/plain'))])
    assert too_many_bytes.status_code == 413
    too_many_files = client.post('/api/v1/ingest', files=[('files', (f'{i}.txt', b'x', 'text/plain')) for i in range(11)])
    assert too_many_files.status_code == 413
