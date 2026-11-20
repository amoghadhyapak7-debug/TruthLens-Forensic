"""Migration script to migrate legacy data/ layout to centralized storage/ structure."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"

from app.storage import (
    init_storage,
    STORAGE_DIR,
    DATABASE_DIR,
    DB_PATH,
    REGISTRY_PATH,
    EVIDENCE_DIR,
    EVIDENCE_IMAGES_DIR,
    EVIDENCE_VIDEOS_DIR,
    EVIDENCE_AUDIO_DIR,
    EVIDENCE_DOCUMENTS_DIR,
    EXTRACTED_AUDIO_DIR,
    EXTRACTED_FRAMES_DIR,
    ANALYSIS_IMAGE_DIR,
    ANALYSIS_VIDEO_DIR,
    ANALYSIS_AUDIO_DIR,
    ANALYSIS_TEXT_DIR,
    ANALYSIS_REVIEWS_DIR,
    LOGS_DIR,
    INGEST_ACTIVITY_PATH,
    get_evidence_destination_dir,
)


def run_migration():
    print("=" * 60)
    print("TruthLens Centralized Storage Migration")
    print("=" * 60)
    
    # 1. Initialize directory structure
    init_storage()
    print("Initialized centralized storage directories under:", STORAGE_DIR)

    migration_report = {
        "evidence_files": 0,
        "extracted_audio": 0,
        "extracted_frames": 0,
        "analysis_results": 0,
        "reviews": 0,
        "database_migrated": False,
        "registry_updated": False,
        "errors": [],
    }

    # 2. Migrate Database
    old_db = DATA_DIR / "truthlens.db"
    if old_db.is_file():
        shutil.copy2(old_db, DB_PATH)
        migration_report["database_migrated"] = True
        print(f"Copied SQLite database: {old_db} -> {DB_PATH}")

    # 3. Migrate Ingest Activity Log
    old_activity = DATA_DIR / "ingest_activity.json"
    if old_activity.is_file():
        shutil.copy2(old_activity, INGEST_ACTIVITY_PATH)
        print(f"Copied activity log: {old_activity} -> {INGEST_ACTIVITY_PATH}")

    # 4. Migrate Analysis JSONs
    # Audio / deepfake
    for f in (DATA_DIR / "deepfake").glob("*.json"):
        shutil.copy2(f, ANALYSIS_AUDIO_DIR / f.name)
        migration_report["analysis_results"] += 1
    # Video
    for f in (DATA_DIR / "video").glob("*.json"):
        shutil.copy2(f, ANALYSIS_VIDEO_DIR / f.name)
        migration_report["analysis_results"] += 1
    # Image
    for f in (DATA_DIR / "image").glob("*.json"):
        shutil.copy2(f, ANALYSIS_IMAGE_DIR / f.name)
        migration_report["analysis_results"] += 1
    # Text
    for f in (DATA_DIR / "text").glob("*.json"):
        shutil.copy2(f, ANALYSIS_TEXT_DIR / f.name)
        migration_report["analysis_results"] += 1
    # Reviews
    for f in (DATA_DIR / "reviews").glob("*.json"):
        shutil.copy2(f, ANALYSIS_REVIEWS_DIR / f.name)
        migration_report["reviews"] += 1

    print(f"Migrated analysis files: {migration_report['analysis_results']} analysis, {migration_report['reviews']} reviews.")

    # 5. Migrate Extracted Artifacts (audio and frames from artifacts/)
    artifacts_dir = DATA_DIR / "artifacts"
    if artifacts_dir.is_dir():
        for ev_dir in artifacts_dir.iterdir():
            if not ev_dir.is_dir():
                continue
            ev_id = ev_dir.name
            
            # Extracted audio
            ext_audio_dir = ev_dir / "extracted_audio"
            if ext_audio_dir.is_dir():
                for wav_file in ext_audio_dir.glob("*.wav"):
                    dest = EXTRACTED_AUDIO_DIR / wav_file.name
                    shutil.copy2(wav_file, dest)
                    migration_report["extracted_audio"] += 1
                    print(f"Migrated extracted audio: {wav_file.name} -> {dest}")

            # Extracted video frames
            v_frames_dir = ev_dir / "video" / ev_id / "frames"
            if v_frames_dir.is_dir():
                dest_frames_dir = EXTRACTED_FRAMES_DIR / ev_id
                dest_frames_dir.mkdir(parents=True, exist_ok=True)
                for frame_file in v_frames_dir.glob("*.jpg"):
                    shutil.copy2(frame_file, dest_frames_dir / frame_file.name)
                    migration_report["extracted_frames"] += 1

    # 6. Read and Migrate Evidence Registry and Evidence Files
    old_registry_file = DATA_DIR / "evidence_registry.json"
    if old_registry_file.is_file():
        with open(old_registry_file, "r", encoding="utf-8") as f:
            registry_records = json.load(f)

        updated_records = []
        for record in registry_records:
            ev_id = record.get("evidence_id")
            f_type = record.get("file_type", "unknown")
            stored_name = record.get("stored_filename")
            old_file_path = Path(record.get("file_path", ""))

            # Target directory by type
            target_vault = get_evidence_destination_dir(f_type)
            
            # Check where source file is located
            source_file = None
            if old_file_path.is_file():
                source_file = old_file_path
            elif stored_name and (DATA_DIR / "evidence" / stored_name).is_file():
                source_file = DATA_DIR / "evidence" / stored_name

            if source_file and source_file.is_file():
                dest_file = target_vault / source_file.name
                shutil.copy2(source_file, dest_file)
                migration_report["evidence_files"] += 1
                
                # Update record with new canonical path and storage location
                record["file_path"] = str(dest_file)
                record["storage_location"] = str(target_vault)
                if isinstance(record.get("hash"), dict):
                    record["hash"]["file_reference"] = str(dest_file)
                print(f"Migrated evidence {ev_id} ({f_type}): {source_file.name} -> {dest_file}")
            else:
                print(f"Note: Evidence {ev_id} file not found on local disk ({old_file_path}). Keeping path reference.")

            updated_records.append(record)

        # Write updated registry to both storage/database/ and data/ (for backward compat)
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(updated_records, f, indent=2, ensure_ascii=False)
        with open(old_registry_file, "w", encoding="utf-8") as f:
            json.dump(updated_records, f, indent=2, ensure_ascii=False)
        migration_report["registry_updated"] = True

        # Also update SQLite database records with new file paths
        if DB_PATH.is_file():
            with sqlite3.connect(str(DB_PATH)) as conn:
                for r in updated_records:
                    conn.execute("""
                        UPDATE evidence_records
                        SET file_path = ?, storage_location = ?
                        WHERE evidence_id = ?
                    """, (r.get("file_path"), r.get("storage_location"), r.get("evidence_id")))
                conn.commit()
            print("Updated SQLite database file paths successfully.")

    # Also check any orphan evidence files in data/evidence/ that weren't in registry
    old_evidence_dir = DATA_DIR / "evidence"
    if old_evidence_dir.is_dir():
        for orphan in old_evidence_dir.iterdir():
            if orphan.is_file() and not orphan.name.startswith("."):
                ext = orphan.suffix.lower()
                if ext in (".png", ".jpg", ".jpeg", ".webp"):
                    dest = EVIDENCE_IMAGES_DIR / orphan.name
                elif ext in (".mp4", ".mov", ".avi", ".mkv"):
                    dest = EVIDENCE_VIDEOS_DIR / orphan.name
                elif ext in (".wav", ".mp3", ".m4a"):
                    dest = EVIDENCE_AUDIO_DIR / orphan.name
                else:
                    dest = EVIDENCE_DOCUMENTS_DIR / orphan.name
                
                if not dest.exists():
                    shutil.copy2(orphan, dest)
                    print(f"Migrated unindexed file: {orphan.name} -> {dest}")

    print("\nMigration Complete Summary:")
    for k, v in migration_report.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    run_migration()
