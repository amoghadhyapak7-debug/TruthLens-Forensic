"""SQLite database layer for TruthLens structured records.

Maintains local structured records in SQLite alongside the filesystem evidence vault:
- Table: evidence_records (structured metadata, hashes, paths, readiness, testing flags)
- Table: ingest_activity (persistent ingestion and operations audit trail)
- Table: analysis_records (analysis verdicts, synthetic scores, models used)

Large binary media files are stored on disk in the filesystem vault, NOT inside SQLite.
Database records reference the exact filesystem paths and evidence IDs.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from app.storage import DB_PATH
except ImportError:
    DB_PATH = Path("storage/database/truthlens.db")


class DatabaseManager:
    """Manages the local SQLite database for TruthLens."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        """Create structured tables if they don't exist."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS evidence_records (
                    evidence_id TEXT PRIMARY KEY,
                    case_id TEXT,
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT,
                    file_path TEXT NOT NULL,
                    storage_location TEXT,
                    file_type TEXT NOT NULL,
                    file_extension TEXT,
                    mime_type TEXT,
                    file_size INTEGER NOT NULL,
                    file_size_human TEXT,
                    sha256 TEXT NOT NULL,
                    ingestion_timestamp TEXT NOT NULL,
                    validation_status TEXT,
                    ingestion_status TEXT,
                    duplicate_reference TEXT,
                    is_testing INTEGER DEFAULT 0,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_evidence_sha256 ON evidence_records(sha256);
                CREATE INDEX IF NOT EXISTS idx_evidence_case ON evidence_records(case_id);
                CREATE INDEX IF NOT EXISTS idx_evidence_type ON evidence_records(file_type);
                CREATE INDEX IF NOT EXISTS idx_evidence_testing ON evidence_records(is_testing);

                CREATE TABLE IF NOT EXISTS ingest_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    filename TEXT,
                    status TEXT NOT NULL,
                    message TEXT,
                    details_json TEXT
                );

                CREATE TABLE IF NOT EXISTS analysis_records (
                    evidence_id TEXT PRIMARY KEY,
                    modality TEXT NOT NULL,
                    routed_analyzer TEXT,
                    verdict TEXT,
                    confidence_score REAL,
                    synthetic_probability REAL,
                    models_used_json TEXT,
                    result_json TEXT,
                    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (evidence_id) REFERENCES evidence_records(evidence_id) ON DELETE CASCADE
                );
            """)

    def upsert_evidence(self, record: Dict[str, Any], is_testing: bool = False) -> None:
        """Insert or update a structured evidence record."""
        ev_id = record.get("evidence_id")
        if not ev_id:
            return

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO evidence_records (
                    evidence_id, case_id, original_filename, stored_filename,
                    file_path, storage_location, file_type, file_extension,
                    mime_type, file_size, file_size_human, sha256,
                    ingestion_timestamp, validation_status, ingestion_status,
                    duplicate_reference, is_testing, metadata_json
                ) VALUES (
                    :evidence_id, :case_id, :original_filename, :stored_filename,
                    :file_path, :storage_location, :file_type, :file_extension,
                    :mime_type, :file_size, :file_size_human, :sha256,
                    :ingestion_timestamp, :validation_status, :ingestion_status,
                    :duplicate_reference, :is_testing, :metadata_json
                ) ON CONFLICT(evidence_id) DO UPDATE SET
                    case_id = excluded.case_id,
                    original_filename = excluded.original_filename,
                    stored_filename = excluded.stored_filename,
                    file_path = excluded.file_path,
                    file_type = excluded.file_type,
                    file_size = excluded.file_size,
                    sha256 = excluded.sha256,
                    validation_status = excluded.validation_status,
                    ingestion_status = excluded.ingestion_status,
                    is_testing = excluded.is_testing,
                    metadata_json = excluded.metadata_json
            """, {
                "evidence_id": ev_id,
                "case_id": record.get("case_id"),
                "original_filename": record.get("original_filename", "unnamed"),
                "stored_filename": record.get("stored_filename"),
                "file_path": str(record.get("file_path", "")),
                "storage_location": str(record.get("storage_location", "")),
                "file_type": record.get("file_type", "unknown"),
                "file_extension": record.get("file_extension", ""),
                "mime_type": record.get("mime_type", ""),
                "file_size": int(record.get("file_size", 0)),
                "file_size_human": record.get("file_size_human", ""),
                "sha256": record.get("sha256", ""),
                "ingestion_timestamp": record.get("ingestion_timestamp", ""),
                "validation_status": record.get("validation_status", "unverified"),
                "ingestion_status": record.get("ingestion_status", "completed"),
                "duplicate_reference": record.get("duplicate_reference"),
                "is_testing": 1 if is_testing else 0,
                "metadata_json": json.dumps(record.get("metadata") or {}),
            })

    def delete_evidence(self, evidence_id: str) -> None:
        """Delete an evidence record and cascading references from the database."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM analysis_records WHERE evidence_id = ?", (evidence_id,))
            conn.execute("DELETE FROM evidence_records WHERE evidence_id = ?", (evidence_id,))

    def clear_all_evidence(self) -> None:
        """Remove all evidence records from the database."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM analysis_records")
            conn.execute("DELETE FROM evidence_records")

    def record_activity(self, timestamp: str, filename: str, status: str, message: str, details: Optional[Dict] = None) -> None:
        """Record an ingestion audit entry in the database."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO ingest_activity (timestamp, filename, status, message, details_json)
                VALUES (?, ?, ?, ?, ?)
            """, (timestamp, filename, status, message, json.dumps(details or {})))

    def sync_all_records(self, records: List[Dict[str, Any]], classify_func=None) -> None:
        """Synchronize a full list of registry records with the SQLite database."""
        with self._get_connection() as conn:
            current_ids = set()
            for rec in records:
                ev_id = rec.get("evidence_id")
                if ev_id:
                    current_ids.add(ev_id)
                    is_t = classify_func(rec) if classify_func else False
                    self.upsert_evidence(rec, is_testing=is_t)

            # Clean up rows no longer in registry
            if current_ids:
                placeholders = ",".join("?" for _ in current_ids)
                conn.execute(f"DELETE FROM evidence_records WHERE evidence_id NOT IN ({placeholders})", list(current_ids))
            else:
                conn.execute("DELETE FROM evidence_records")

    def get_database_stats(self) -> Dict[str, Any]:
        """Return statistics about structured records in the SQLite database."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM evidence_records").fetchone()[0]
            testing = conn.execute("SELECT COUNT(*) FROM evidence_records WHERE is_testing = 1").fetchone()[0]
            real = conn.execute("SELECT COUNT(*) FROM evidence_records WHERE is_testing = 0").fetchone()[0]
            activity_count = conn.execute("SELECT COUNT(*) FROM ingest_activity").fetchone()[0]
            return {
                "db_path": str(self.db_path),
                "total_records": total,
                "testing_records": testing,
                "case_evidence_records": real,
                "activity_log_entries": activity_count,
            }
