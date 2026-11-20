from monitor.results import present_upload_result, replace_upload_session


def test_current_session_maps_new_duplicate_and_failed_results():
    timestamp = "2026-11-19T14:00:00+05:30"
    items = [
        {"status": "completed", "original_filename": "new.png", "evidence_id": "EVD-006", "sha256": "a" * 64, "file_size": 1024, "evidence": {"file_type": "png", "mime_type": "image/png", "file_size_human": "1.00 KB"}},
        {"status": "duplicate", "original_filename": "same.png", "duplicate_reference": "EVD-005", "sha256": "b" * 64, "file_size": 10, "metadata": {"extension": ".png", "mime_type": "image/png"}},
        {"status": "failed", "original_filename": "bad.exe", "error": "Unsupported file type: .exe"},
    ]
    results = [present_upload_result(item, timestamp) for item in items]
    assert [item["Upload result"] for item in results] == ["NEW", "DUPLICATE", "FAILED"]
    assert results[0]["Evidence ID"] == "EVD-006"
    assert results[1]["Evidence ID"] == "—"
    assert results[1]["Duplicate reference"] == "EVD-005"
    assert results[1]["SHA-256"] == "b" * 64
    assert results[2]["Error"] == "Unsupported file type: .exe"
    assert all(item["Upload timestamp"] == timestamp for item in results)


def test_new_upload_session_replaces_previous_session():
    state = {"current_upload_session": {"timestamp": "old", "results": [{"Original filename": "stale.txt"}]}}
    item = {"status": "duplicate", "original_filename": "current.txt", "duplicate_reference": "EVD-003", "sha256": "c" * 64}
    replace_upload_session(state, [item], "new")
    assert state["current_upload_session"]["timestamp"] == "new"
    assert [r["Original filename"] for r in state["current_upload_session"]["results"]] == ["current.txt"]
    replace_upload_session(state, [{"status": "failed", "original_filename": "batch-bad.exe", "error": "unsupported"}], "newer")
    assert [r["Original filename"] for r in state["current_upload_session"]["results"]] == ["batch-bad.exe"]


def test_duplicate_upload_metadata_is_attributed_to_matched_registry_record():
    item = {"status":"duplicate", "original_filename":"test audio one.mp4",
        "duplicate_reference":"EVD-013", "metadata_source_evidence_id":"EVD-013",
        "validation_source_evidence_id":"EVD-013", "validation_status":"unreadable",
        "metadata":{"modality":"video", "container_format":"mov", "duration_seconds":10.41,
                    "metadata_extraction_status":"unavailable"}}
    result = present_upload_result(item, "now")
    assert result["Evidence ID"] == "—"
    assert result["Duplicate reference"] == "EVD-013"
    assert result["Metadata source"] == result["Validation source"] == "Registered evidence EVD-013"
    assert result["Validation status"] == "unreadable"
    assert result["Metadata"]["duration_seconds"] == 10.41


def test_duplicate_parser_error_is_not_repeated_in_generic_error_row():
    error = "Video contains no readable video stream."
    result = present_upload_result({"status":"duplicate", "original_filename":"bad.mp4",
        "duplicate_reference":"EVD-013", "errors":[error], "validation_errors":[error],
        "analysis_eligibility":{"status":"DUPLICATE_REFERENCE", "reasons":["Matches EVD-013."], "warnings":[]}}, "now")
    assert result["Error"] == "—"
    assert result["Validation errors"] == [error]
