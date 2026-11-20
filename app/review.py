"""TruthLens Analyst Review Manager."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from app.storage import ANALYSIS_REVIEWS_DIR
except ImportError:
    ANALYSIS_REVIEWS_DIR = None


def save_review(
    data_dir: Path,
    evidence_id: str,
    status: str,
    notes: str,
    examiner: str,
    auto_findings: str = "",
    investigator_opinion: str = "",
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    rev_record = {
        "evidence_id": evidence_id,
        "status": status,  # UNREVIEWED, VERIFIED, DISPUTED, FLAGGED
        "notes": notes,
        "examiner": examiner or "Forensic Examiner",
        "updated_at": now,
        "auto_findings": auto_findings,
        "investigator_opinion": investigator_opinion,
    }
    payload = json.dumps(rev_record, indent=2)

    # Save to centralized storage/analysis/reviews/
    if ANALYSIS_REVIEWS_DIR is not None:
        ANALYSIS_REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
        (ANALYSIS_REVIEWS_DIR / f"{evidence_id}.json").write_text(payload, encoding="utf-8")

    # Dual-save to data_dir/reviews for backward compatibility
    rev_dir = data_dir / "reviews"
    rev_dir.mkdir(parents=True, exist_ok=True)
    (rev_dir / f"{evidence_id}.json").write_text(payload, encoding="utf-8")
    return rev_record


def get_review(data_dir: Path, evidence_id: str) -> Dict[str, Any]:
    # Check centralized storage first
    if ANALYSIS_REVIEWS_DIR is not None:
        p = ANALYSIS_REVIEWS_DIR / f"{evidence_id}.json"
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass

    # Check data_dir
    path = data_dir / "reviews" / f"{evidence_id}.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "evidence_id": evidence_id,
        "status": "UNREVIEWED",
        "notes": "",
        "examiner": "",
        "updated_at": None,
        "auto_findings": "",
        "investigator_opinion": "",
    }


def list_reviews(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    res = {}
    # Load from centralized storage
    if ANALYSIS_REVIEWS_DIR is not None and ANALYSIS_REVIEWS_DIR.is_dir():
        for p in ANALYSIS_REVIEWS_DIR.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                res[p.stem] = data
            except Exception:
                pass

    # Merge legacy
    rev_dir = data_dir / "reviews"
    if rev_dir.is_dir():
        for p in rev_dir.glob("*.json"):
            if p.stem not in res:
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    res[p.stem] = data
                except Exception:
                    pass
    return res
