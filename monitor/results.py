"""Presentation helpers for an isolated latest-upload session."""


def present_upload_result(item: dict, timestamp: str) -> dict:
    """Normalize one API result for the latest-session display."""
    evidence = item.get("evidence") or {}
    status = item.get("status")
    labels = {"completed": "NEW", "duplicate": "DUPLICATE", "failed": "FAILED"}
    eligibility = item.get("analysis_eligibility") or evidence.get("analysis_eligibility") or {}
    duplicate_reference = item.get("duplicate_reference") or "—"
    validation_reasons = set(eligibility.get("reasons", []))
    errors = item.get("errors", [])
    if isinstance(errors, str):
        errors = [errors]
    validation_errors = set(item.get("validation_errors", []))
    # Upload cards show readiness reasons in their own section. Avoid repeating
    # the same validation message in the generic error row while retaining any
    # distinct storage/registration errors.
    distinct_errors = list(dict.fromkeys(error for error in errors if error not in validation_reasons and error not in validation_errors))
    return {
        "Original filename": item.get("original_filename", "—"),
        "Upload result": labels.get(status, str(status or "UNKNOWN").upper()),
        "Evidence ID": item.get("evidence_id") or "—",
        "SHA-256": item.get("sha256") or "—",
        "File type": evidence.get("file_type") or item.get("file_type") or item.get("metadata", {}).get("modality") or item.get("metadata", {}).get("extension", "—"),
        "MIME type": evidence.get("mime_type") or item.get("mime_type") or item.get("metadata", {}).get("mime_type", "—"),
        "File size": evidence.get("file_size_human") or item.get("file_size_human") or (f"{item['file_size']:,} bytes" if item.get("file_size") is not None else "—"),
        "Upload timestamp": timestamp,
        "Duplicate reference": duplicate_reference,
        "Error": item.get("error") or ("; ".join(distinct_errors) if distinct_errors else "—"),
        "Analysis eligibility": eligibility,
        "Validation status": evidence.get("validation_status", item.get("validation_status", "Unavailable")),
        "Validation source": item.get("validation_source_label") or (f"Registered evidence {item['validation_source_evidence_id']}" if item.get("validation_source_evidence_id") else "This upload"),
        "Metadata source": item.get("metadata_source_label") or (f"Registered evidence {item['metadata_source_evidence_id']}" if item.get("metadata_source_evidence_id") else "This upload"),
        "Validation errors": list(dict.fromkeys(item.get("validation_errors", []))),
        "Warnings": item.get("warnings", []),
        "Stages": item.get("stages", []),
        "Metadata": item.get("metadata") or evidence.get("metadata") or {},
    }


def replace_upload_session(state: dict, items: list[dict], timestamp: str) -> None:
    """Replace the prior operation; never append across upload sessions."""
    state["current_upload_session"] = {
        "timestamp": timestamp,
        "results": [present_upload_result(item, timestamp) for item in items],
    }


def clear_registry_session_state(state: dict) -> None:
    """Drop UI results and widget selections tied to removed evidence."""
    old_version = int(state.get("upload_widget_version", 0))
    state.pop(f"evidence_upload_{old_version}", None)
    for key in list(state.keys()):
        if key.startswith("deepfake_") or key in (
            "current_upload_session", "audio_analysis_result", "audio_analysis_evidence_id",
            "analyzer_input_evidence", "audio_analyzer_evidence", "audio_run_transcription",
            "audio_transcription_language", "audio_run_synthetic", "audio_deepfake_evidence_select",
        ):
            state.pop(key, None)
    state["upload_widget_version"] = old_version + 1
