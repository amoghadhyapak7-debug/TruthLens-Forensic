"""TruthLens Investigation Storage Manager

Handles persistent storage, retrieval, and indexing of forensic investigation reports.
Each completed investigation is assigned a unique case ID and persisted as a structured
JSON document under evidence_cache/reports/, ensuring zero data loss during navigation.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

STORAGE_DIR = Path(__file__).resolve().parent / "evidence_cache" / "reports"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def generate_case_id(prefix: str = "CASE") -> str:
    """Generates a readable, timestamped unique case/report ID."""
    now_str = datetime.datetime.now().strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:6].upper()
    return f"{prefix}-{now_str}-{short_uuid}"


def save_case_report(report_data: Dict[str, Any]) -> str:
    """Saves a completed investigation report to persistent disk storage.

    Args:
        report_data: Dictionary containing investigation metadata, evidence results,
                     findings, and summary metrics.

    Returns:
        The unique case_id assigned to this report.
    """
    case_id = report_data.get("case_id") or generate_case_id()
    report_data["case_id"] = case_id

    if "created_at" not in report_data:
        report_data["created_at"] = datetime.datetime.now().isoformat()

    file_path = STORAGE_DIR / f"{case_id}.json"
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved investigation report: {file_path}")
        return case_id
    except Exception as e:
        logger.error(f"Failed to persist report {case_id}: {e}", exc_info=True)
        raise IOError(f"Could not save investigation report: {e}") from e


def list_case_reports() -> List[Dict[str, Any]]:
    """Returns an index of all saved investigation reports, ordered newest first."""
    reports: List[Dict[str, Any]] = []

    if not STORAGE_DIR.exists():
        return reports

    for file_path in STORAGE_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Build lightweight index entry
            reports.append({
                "case_id": data.get("case_id", file_path.stem),
                "created_at": data.get("created_at", ""),
                "investigation_title": data.get("investigation_title", "Investigation Report"),
                "evidence_type": data.get("evidence_type", "multimodal"),
                "total_evidence_count": data.get("total_evidence_count", len(data.get("evidence_results", []))),
                "evidence_ids": data.get("evidence_ids", []),
                "summary": data.get("summary", {}),
                "file_path": str(file_path),
            })
        except Exception as e:
            logger.warning(f"Error reading report file {file_path}: {e}")

    # Sort descending by created_at
    reports.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return reports


def load_case_report(case_id: str) -> Optional[Dict[str, Any]]:
    """Loads a full investigation report by its unique case_id."""
    clean_id = Path(case_id).stem
    file_path = STORAGE_DIR / f"{clean_id}.json"

    if not file_path.exists():
        logger.warning(f"Report not found: {file_path}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading report {file_path}: {e}", exc_info=True)
        return None


def delete_case_report(case_id: str) -> bool:
    """Deletes a saved investigation report file."""
    clean_id = Path(case_id).stem
    file_path = STORAGE_DIR / f"{clean_id}.json"
    if file_path.exists():
        try:
            file_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Error deleting report {file_path}: {e}")
            return False
    return False
