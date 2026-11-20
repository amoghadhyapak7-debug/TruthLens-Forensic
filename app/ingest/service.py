"""Persistent, filesystem-backed evidence ingestion.

The registry and evidence directory can be moved by setting TRUTHLENS_DATA_DIR.
Uploaded bytes are streamed to disk and hashed in the same pass, preserving the
original content without loading large files into memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

CHUNK_SIZE = 1024 * 1024
from app.ingest.config import LIMITS, SUPPORTED_EXTENSIONS, modality_for_extension
from app.ingest.database import DatabaseManager
from app.ingest.metadata import analyzer_limit_reasons, extract_metadata
from app.storage import (
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
    ANALYSIS_AUDIO_DIR,
    ANALYSIS_VIDEO_DIR,
    ANALYSIS_IMAGE_DIR,
    ANALYSIS_TEXT_DIR,
    ANALYSIS_REVIEWS_DIR,
    TESTING_UPLOADS_DIR,
    TESTING_TEMP_DIR,
    LOGS_DIR,
    INGEST_ACTIVITY_PATH,
    get_evidence_destination_dir,
    init_storage,
)

ALLOWED_EXTENSIONS = SUPPORTED_EXTENSIONS
STAGES = ("RECEIVED", "VALIDATING", "HASHING", "METADATA_EXTRACTION", "STORAGE", "REGISTRATION", "COMPLETED", "FAILED")
REGISTRY_CLEAR_PHRASE = "CLEAR REGISTRY"
_EVIDENCE_ID_RE = re.compile(r"EVD-\d{3,}")
_ANALYSIS_RUN_RE = re.compile(r"\d{8}T\d{6}-[0-9a-f]{8}")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def _safe_original_name(name: str) -> str:
    # Keep a display name only. Never use it to form a path.
    return Path(name.replace("\\", "/")).name[:255] or "unnamed"


class IngestService:
    """Ingest individual binary streams and persist evidence records."""

    def __init__(self, data_dir: str | Path | None = None, max_file_size: int | None = None, limits=None):
        init_storage()
        if data_dir is not None:
            self.storage_dir = Path(data_dir).resolve()
            self.data_dir = self.storage_dir
            self.evidence_dir = self.storage_dir / "evidence"
            self.registry_path = self.storage_dir / "database" / "evidence_registry.json" if (self.storage_dir / "database").exists() else self.storage_dir / "evidence_registry.json"
            self.activity_path = self.storage_dir / "logs" / "ingest_activity.json" if (self.storage_dir / "logs").exists() else self.storage_dir / "ingest_activity.json"
            self.db = DatabaseManager(self.storage_dir / "database" / "truthlens.db" if (self.storage_dir / "database").exists() else self.storage_dir / "truthlens.db")
        else:
            self.storage_dir = STORAGE_DIR
            self.data_dir = self.storage_dir
            self.evidence_dir = EVIDENCE_DIR
            self.registry_path = REGISTRY_PATH
            self.activity_path = INGEST_ACTIVITY_PATH
            self.db = DatabaseManager(DB_PATH)

        self.limits = limits or LIMITS
        self.max_file_size = max_file_size if max_file_size is not None else self.limits.max_upload_file_bytes
        self._lock = threading.RLock()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.registry_path, self.activity_path):
            if not path.exists():
                self._atomic_json(path, [])
        try:
            self.db.sync_all_records(self.list_evidence(), classify_func=self.classify_as_testing)
        except Exception:
            pass

    def get_dest_vault(self, file_type: str, is_testing: bool = False) -> Path:
        """Return the appropriate subfolder based on auto-detected file type."""
        if self.storage_dir == STORAGE_DIR:
            return get_evidence_destination_dir(file_type, is_testing=is_testing)
        else:
            if is_testing:
                d = self.storage_dir / "testing" / "uploads"
            else:
                clean_type = (file_type or "").lower().strip()
                sub = "images" if clean_type == "image" else "videos" if clean_type == "video" else "audio" if clean_type == "audio" else "documents"
                d = self.evidence_dir / sub
            d.mkdir(parents=True, exist_ok=True)
            return d

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                json.dump(value, out, indent=2, ensure_ascii=False)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _read(self, path: Path) -> list[dict[str, Any]]:
        try:
            with path.open(encoding="utf-8") as src:
                value = json.load(src)
            return value if isinstance(value, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _append_activity(self, entry: dict[str, Any]) -> None:
        with self._lock:
            entries = self._read(self.activity_path)
            entries.append(entry)
            self._atomic_json(self.activity_path, entries[-500:])

    def _activity(self, name: str, status: str, message: str, **details: Any) -> None:
        now_ts = _now()
        self._append_activity({"timestamp": now_ts, "filename": name, "status": status, "message": message, "details": details})
        try:
            self.db.record_activity(now_ts, name, status, message, details)
        except Exception:
            pass

    def list_evidence(self) -> list[dict[str, Any]]:
        return self._read(self.registry_path)

    def get_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        return next((item for item in self.list_evidence() if item.get("evidence_id") == evidence_id), None)

    def list_activity(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._read(self.activity_path)[-max(0, limit):][::-1]

    @staticmethod
    def _lexical_direct_child(path: str | Path, root: Path) -> Path:
        root_abs = Path(os.path.abspath(root))
        candidate = Path(os.path.abspath(path))
        if root_abs.is_symlink() or Path(os.path.realpath(root_abs)) != root_abs:
            raise ValueError(f"Configured storage root is a symlink or resolves outside its root: {root_abs}")
        if candidate == root_abs:
            raise ValueError(f"Path cannot be the root itself: {candidate}")
        try:
            candidate.relative_to(root_abs)
        except ValueError:
            raise ValueError(f"Path is not inside the configured storage root: {candidate}")
        return candidate

    @staticmethod
    def _remove_leaf(path: Path) -> tuple[bool, str | None]:
        """Unlink a file or symlink itself, never following symlinks."""
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            return False, None
        except OSError as exc:
            return False, str(exc)
        import stat
        if stat.S_ISLNK(mode) or stat.S_ISREG(mode):
            try:
                path.unlink()
                return True, None
            except OSError as exc:
                return False, str(exc)
        return False, f"Expected a regular file or symlink, found a non-file entry: {path}"

    def _remove_registered_artifacts(self, evidence_id: str) -> dict[str, Any]:
        """Remove only the known generated audio artifact layout for one EVD ID."""
        root = self.data_dir / "artifacts"
        summary = {"files_removed": 0, "analysis_records_removed": 0, "errors": []}
        if not root.exists() and not root.is_symlink():
            return summary
        try:
            target = self._lexical_direct_child(root / evidence_id, root)
        except (OSError, ValueError) as exc:
            summary["errors"].append({"path": str(root), "error": str(exc)})
            return summary
        if not target.exists() and not target.is_symlink():
            return summary
        if target.is_symlink():
            removed, error = self._remove_leaf(target)
            summary["files_removed"] += int(removed)
            if error:
                summary["errors"].append({"path": str(target), "error": error})
            return summary
        if not target.is_dir():
            summary["errors"].append({"path": str(target), "error": "Evidence artifact entry is not a directory."})
            return summary

        def remove_file(path: Path, allowed: set[str]) -> None:
            if path.name not in allowed:
                summary["errors"].append({"path": str(path), "error": "Unexpected artifact entry; preserved."})
                return
            removed, error = self._remove_leaf(path)
            summary["files_removed"] += int(removed)
            if path.name == "analysis.json" and removed:
                summary["analysis_records_removed"] += 1
            if error:
                summary["errors"].append({"path": str(path), "error": error})

        try:
            with os.scandir(target) as entries:
                evidence_entries = list(entries)
        except OSError as exc:
            summary["errors"].append({"path": str(target), "error": str(exc)})
            return summary
        for entry in evidence_entries:
            audio_dir = Path(entry.path)
            if entry.name != "audio":
                summary["errors"].append({"path": str(audio_dir), "error": "Unexpected evidence artifact entry; preserved."})
                continue
            if entry.is_symlink():
                removed, error = self._remove_leaf(audio_dir)
                summary["files_removed"] += int(removed)
                if error:
                    summary["errors"].append({"path": str(audio_dir), "error": error})
                continue
            if not entry.is_dir(follow_symlinks=False):
                summary["errors"].append({"path": str(audio_dir), "error": "Expected audio artifact directory; preserved."})
                continue
            try:
                with os.scandir(audio_dir) as entries:
                    audio_entries = list(entries)
            except OSError as exc:
                summary["errors"].append({"path": str(audio_dir), "error": str(exc)})
                continue
            for audio_entry in audio_entries:
                item = Path(audio_entry.path)
                if audio_entry.name == "latest.json":
                    remove_file(item, {"latest.json"})
                    continue
                if not _ANALYSIS_RUN_RE.fullmatch(audio_entry.name):
                    summary["errors"].append({"path": str(item), "error": "Unexpected audio artifact entry; preserved."})
                    continue
                if audio_entry.is_symlink():
                    removed, error = self._remove_leaf(item)
                    summary["files_removed"] += int(removed)
                    if error:
                        summary["errors"].append({"path": str(item), "error": error})
                    continue
                if not audio_entry.is_dir(follow_symlinks=False):
                    summary["errors"].append({"path": str(item), "error": "Expected analysis run directory; preserved."})
                    continue
                run_id = audio_entry.name
                expected_files = {
                    "analysis.json", f"{evidence_id}_audio_{run_id}_transcript.json",
                    f"{evidence_id}_audio_{run_id}_waveform.png",
                    f"{evidence_id}_audio_{run_id}_spectrogram.png",
                    f"{evidence_id}_audio_{run_id}_frequency_spectrum.png",
                    f"{evidence_id}_audio_{run_id}_timeline.png",
                }
                try:
                    with os.scandir(item) as entries:
                        run_entries = list(entries)
                except OSError as exc:
                    summary["errors"].append({"path": str(item), "error": str(exc)})
                    continue
                for run_entry in run_entries:
                    run_path = Path(run_entry.path)
                    if run_entry.is_dir(follow_symlinks=False) and not run_entry.is_symlink():
                        summary["errors"].append({"path": str(run_path), "error": "Nested directory is not part of the generated artifact layout; preserved."})
                    else:
                        remove_file(run_path, expected_files)
                try:
                    item.rmdir()
                except OSError:
                    pass
            try:
                audio_dir.rmdir()
            except OSError:
                pass
        try:
            target.rmdir()
        except OSError:
            pass
        return summary

    def clear_evidence_registry(self, confirmation: str) -> dict[str, Any]:
        """Clear registered evidence and only associated app-owned files."""
        if confirmation != REGISTRY_CLEAR_PHRASE:
            raise ValueError(f"Type {REGISTRY_CLEAR_PHRASE!r} exactly to clear the evidence registry.")
        with self._lock:
            summary: dict[str, Any] = {
                "complete": False, "evidence_records_removed": 0,
                "original_files_removed": 0, "original_files_missing": 0,
                "artifact_files_removed": 0, "analysis_records_removed": 0,
                "temporary_files_removed": 0, "registry_records_remaining": 0,
                "errors": [], "preserved": ["ingest_activity.json", "unregistered files", "project files and configuration"],
            }
            if self.registry_path.is_symlink() or self.registry_path.parent.resolve() != self.data_dir:
                summary["errors"].append({"path": str(self.registry_path), "error": "Registry path failed storage-root validation."})
                summary["registry_records_remaining"] = len(self.list_evidence())
                return summary
            try:
                with self.registry_path.open(encoding="utf-8") as registry_file:
                    raw_records = json.load(registry_file)
            except (OSError, json.JSONDecodeError) as exc:
                summary["errors"].append({"path": str(self.registry_path), "error": f"Registry could not be read safely; no evidence was removed: {exc}"})
                summary["registry_records_remaining"] = len(self.list_evidence())
                return summary
            if not isinstance(raw_records, list):
                summary["errors"].append({"path": str(self.registry_path), "error": "Registry root is not a list; no evidence was removed."})
                return summary
            records = self.list_evidence()
            evidence_root = self.storage_dir
            if evidence_root.is_symlink() or Path(os.path.realpath(evidence_root)) != evidence_root:
                summary["errors"].append({"path": str(evidence_root), "error": "Evidence storage root is a symlink or resolves outside the configured data directory."})
                summary["registry_records_remaining"] = len(records)
                return summary
            failed_records: list[dict[str, Any]] = []
            for record in records:
                if not isinstance(record, dict):
                    summary["errors"].append({"path": str(self.registry_path), "error": "Malformed registry entry; preserved."})
                    failed_records.append(record)
                    continue
                evidence_id = str(record.get("evidence_id", ""))
                if not _EVIDENCE_ID_RE.fullmatch(evidence_id):
                    summary["errors"].append({"evidence_id": evidence_id or None, "error": "Invalid evidence ID; associated paths were preserved."})
                    failed_records.append(record)
                    continue
                try:
                    original = self._lexical_direct_child(str(record.get("file_path", "")), evidence_root)
                except (OSError, ValueError) as exc:
                    summary["errors"].append({"evidence_id": evidence_id, "path": str(record.get("file_path", "")), "error": str(exc)})
                    failed_records.append(record)
                    continue
                try:
                    original.lstat()
                    original_exists = True
                except FileNotFoundError:
                    original_exists = False
                except OSError as exc:
                    summary["errors"].append({"evidence_id": evidence_id, "path": str(original), "error": str(exc)})
                    failed_records.append(record)
                    continue
                if original_exists and not (original.is_symlink() or original.is_file()):
                    summary["errors"].append({"evidence_id": evidence_id, "path": str(original), "error": "Registered original is not a regular file or symlink; preserved."})
                    failed_records.append(record)
                    continue
                artifact_result = self._remove_registered_artifacts(evidence_id)
                summary["artifact_files_removed"] += artifact_result["files_removed"]
                summary["analysis_records_removed"] += artifact_result["analysis_records_removed"]
                if artifact_result["errors"]:
                    summary["errors"].extend({"evidence_id": evidence_id, **error} for error in artifact_result["errors"])
                    failed_records.append(record)
                    continue
                if original_exists:
                    removed, error = self._remove_leaf(original)
                    if error:
                        summary["errors"].append({"evidence_id": evidence_id, "path": str(original), "error": error})
                        failed_records.append(record)
                        continue
                    summary["original_files_removed"] += int(removed)
                else:
                    summary["original_files_missing"] += 1
                summary["evidence_records_removed"] += 1

            # .incoming-* are created exclusively by this service under the
            # evidence root. Their direct-child location is revalidated first.
            try:
                for entry in os.scandir(evidence_root):
                    if not entry.name.startswith(".incoming-"):
                        continue
                    temporary = self._lexical_direct_child(entry.path, evidence_root)
                    if entry.is_dir(follow_symlinks=False) and not entry.is_symlink():
                        summary["errors"].append({"path": str(temporary), "error": "Temporary entry is a directory; preserved."})
                        continue
                    removed, error = self._remove_leaf(temporary)
                    summary["temporary_files_removed"] += int(removed)
                    if error:
                        summary["errors"].append({"path": str(temporary), "error": error})
            except OSError as exc:
                summary["errors"].append({"path": str(evidence_root), "error": str(exc)})

            try:
                self._atomic_json(self.registry_path, failed_records)
                summary["registry_records_remaining"] = len(failed_records)
                try:
                    if summary["registry_records_remaining"] == 0:
                        self.db.clear_all_evidence()
                    else:
                        self.db.sync_all_records(failed_records, classify_func=self.classify_as_testing)
                except Exception:
                    pass
            except OSError as exc:
                summary["errors"].append({"path": str(self.registry_path), "error": f"Could not persist reset registry state: {exc}"})
                summary["registry_records_remaining"] = len(self.list_evidence())
            summary["complete"] = summary["registry_records_remaining"] == 0 and not summary["errors"]
            return summary

    def delete_evidence(self, evidence_id: str) -> dict[str, Any]:
        """Delete a single evidence record and all its associated files.

        Deletes: original evidence file, registered artifacts (extracted audio,
        video frames, analysis JSONs), and analysis result files in data/deepfake,
        data/video, data/image, data/text, data/reviews.
        Removes the registry entry.

        Returns a summary dict with keys: success, files_removed, errors.
        """
        import shutil

        summary: dict[str, Any] = {
            "evidence_id": evidence_id,
            "success": False,
            "files_removed": [],
            "errors": [],
        }

        if not _EVIDENCE_ID_RE.fullmatch(evidence_id):
            summary["errors"].append("Invalid evidence ID format.")
            return summary

        with self._lock:
            records = self.list_evidence()
            record = next((r for r in records if r.get("evidence_id") == evidence_id), None)
            if record is None:
                summary["errors"].append(f"Evidence record {evidence_id} not found.")
                return summary

            evidence_root = self.storage_dir
            if evidence_root.is_symlink() or Path(os.path.realpath(evidence_root)) != evidence_root:
                summary["errors"].append("Evidence storage root failed safety validation.")
                return summary

            # 1. Remove original evidence file
            file_path_str = str(record.get("file_path", ""))
            if file_path_str:
                try:
                    original = self._lexical_direct_child(file_path_str, evidence_root)
                    if original.exists() or original.is_symlink():
                        removed, error = self._remove_leaf(original)
                        if removed:
                            summary["files_removed"].append(str(original))
                        if error:
                            summary["errors"].append(f"Original file: {error}")
                except (OSError, ValueError) as exc:
                    # Path may be on another machine (e.g. Rohan's machine) — not an error
                    if "not a direct child" in str(exc).lower() or "resolves outside" in str(exc).lower() or "not inside" in str(exc).lower():
                        summary["files_removed"].append(f"(original path on another machine: {file_path_str})")
                    else:
                        summary["errors"].append(f"Original file path error: {exc}")

            # 2. Remove extracted artifacts: audio and video frames
            extracted_audio = self.storage_dir / "extracted" / "audio" / f"{evidence_id}_extracted.wav"
            if extracted_audio.exists():
                try:
                    extracted_audio.unlink()
                    summary["files_removed"].append(str(extracted_audio))
                except OSError as exc:
                    summary["errors"].append(f"Extracted audio: {exc}")

            extracted_frames = self.storage_dir / "extracted" / "video_frames" / evidence_id
            if extracted_frames.exists():
                try:
                    shutil.rmtree(str(extracted_frames))
                    summary["files_removed"].append(str(extracted_frames))
                except OSError as exc:
                    summary["errors"].append(f"Extracted frames: {exc}")

            # Remove registered artifacts directory (data/artifacts/EVD-xxx/ or storage/artifacts/EVD-xxx/)
            for art_root in (self.storage_dir / "artifacts" / evidence_id, self.data_dir / "artifacts" / evidence_id):
                if art_root.exists():
                    try:
                        shutil.rmtree(str(art_root))
                        summary["files_removed"].append(str(art_root))
                    except OSError as exc:
                        summary["errors"].append(f"Artifacts dir: {exc}")

            # 3. Remove analysis result files for all modalities
            # In centralized storage/analysis/
            for subdir in ("audio", "video", "image", "text", "reviews"):
                result_path = self.storage_dir / "analysis" / subdir / f"{evidence_id}.json"
                if result_path.exists():
                    try:
                        result_path.unlink()
                        summary["files_removed"].append(str(result_path))
                    except OSError as exc:
                        summary["errors"].append(f"analysis/{subdir}/{evidence_id}.json: {exc}")

            # In legacy data/ subdirectories
            for subdir in ("deepfake", "video", "image", "text", "reviews"):
                result_path = self.data_dir / subdir / f"{evidence_id}.json"
                if result_path.exists():
                    try:
                        result_path.unlink()
                        summary["files_removed"].append(str(result_path))
                    except OSError as exc:
                        summary["errors"].append(f"{subdir}/{evidence_id}.json: {exc}")

            # 4. Remove from registry
            remaining = [r for r in records if r.get("evidence_id") != evidence_id]
            try:
                self._atomic_json(self.registry_path, remaining)
                try:
                    self.db.delete_evidence(evidence_id)
                except Exception:
                    pass
                summary["success"] = True
            except OSError as exc:
                summary["errors"].append(f"Registry update failed: {exc}")
                return summary

            self._activity(
                record.get("original_filename", evidence_id),
                "DELETED",
                f"Evidence {evidence_id} deleted by user request",
                evidence_id=evidence_id,
            )
            return summary

    def classify_as_testing(self, record: dict[str, Any]) -> bool:
        """Return True if a registry record is likely a testing upload.

        Rules (applied in order, ANY match = testing):
        1. case_id is None/empty OR case_id contains 'TEST' (case-insensitive)
        2. original_filename contains any of: test, demo, sample, eval, debug,
           sample1, sample2, sample3, sample4, df_fake, df_orig, whatsapp
           (case-insensitive)

        None of these rules auto-delete — they only flag for display.
        Files are NEVER automatically deleted; the user must confirm.
        """
        case_id = record.get("case_id") or ""
        fname = record.get("original_filename") or ""

        if not case_id or "test" in case_id.lower():
            return True

        testing_keywords = [
            "test", "demo", "sample", "eval", "debug",
            "df_fake", "df_orig", "sample1", "sample2", "sample3", "sample4",
        ]
        fname_lower = fname.lower()
        return any(kw in fname_lower for kw in testing_keywords)

    def _build_contract(self, record: dict[str, Any]) -> dict[str, Any]:
        path = Path(str(record.get("file_path", "")))
        if not path.is_file():
            raise ValueError(f"Original evidence file is missing: {path}")
        expected_size = record.get("file_size")
        actual_size = path.stat().st_size
        if expected_size is None or actual_size != int(expected_size):
            raise ValueError(f"Evidence size mismatch: registry={expected_size}, file={actual_size}")
        ext = path.suffix.lower()
        modality = modality_for_extension(ext)
        if modality is None:
            raise ValueError(f"Unsupported evidence file type: {ext or '(no extension)'}")
        stored_type = str(record.get("file_type", "")).lower()
        if stored_type not in {modality, ext.lstrip(".")}:
            raise ValueError(f"Evidence file_type '{stored_type}' does not match detected modality '{modality}'.")
        expected_hash = str(record.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
            raise ValueError("Evidence record has no valid SHA-256 hash.")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != expected_hash.lower():
            raise ValueError("Evidence integrity check failed: original bytes do not match the stored SHA-256.")
        mime_type = record.get("mime_type")
        if not mime_type:
            raise ValueError("Evidence record has no MIME type.")
        return {
            "evidence_id": record["evidence_id"],
            "file_path": str(path),
            "original_filename": record["original_filename"],
            "file_type": modality,
            "mime_type": mime_type,
            "file_size": actual_size,
            "sha256": digest.hexdigest(),
            "ingestion_metadata": record.get("metadata") or {},
        }

    def get_analyzer_readiness(self, evidence_id: str) -> dict[str, Any]:
        record = self.get_evidence(evidence_id)
        if record is None:
            return {"evidence_id": evidence_id, "status": "NOT_READY", "reasons": ["Evidence record does not exist."], "warnings": []}
        if record.get("duplicate_reference"):
            return {"evidence_id": evidence_id, "status": "DUPLICATE_REFERENCE", "reasons": [f"Evidence is a duplicate reference to {record['duplicate_reference']}."], "warnings": []}
        reasons: list[str] = []
        warnings_out = list(record.get("warnings", []))
        eligibility = record.get("analysis_eligibility")
        # Intake stores the extractor's validation and eligibility decisions.
        # Reuse those same decisions so registry readiness cannot drift from
        # the upload result or duplicate the same parser error on a second pass.
        if eligibility:
            try:
                self._build_contract(record)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return {"evidence_id": evidence_id, "status": "NOT_READY",
                        "reasons": [str(exc)], "warnings": list(dict.fromkeys(warnings_out))}
            return {
                "evidence_id": evidence_id,
                "status": eligibility.get("status", "NOT_READY"),
                "reasons": list(dict.fromkeys(eligibility.get("reasons", []))),
                "warnings": list(dict.fromkeys(eligibility.get("warnings", warnings_out))),
            }
        ext = Path(str(record.get("file_path", ""))).suffix.lower()
        modality = modality_for_extension(ext)
        if modality is None:
            reasons.append(f"Unsupported evidence file type: {ext or '(no extension)'}")
        else:
            try:
                contract = self._build_contract(record)
                extracted = extract_metadata(Path(record["file_path"]), ext, int(record["file_size"]), self.limits)
                if not extracted["content_valid"]:
                    reasons.extend(extracted["errors"] or ["File content could not be validated."])
                reasons.extend(analyzer_limit_reasons(modality, int(record["file_size"]), extracted["metadata"], self.limits))
                warnings_out.extend(extracted["warnings"])
                if not contract.get("ingestion_metadata"):
                    warnings_out.append("No modality metadata was recorded during ingestion.")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                reasons.append(str(exc))
        if reasons:
            status = "NOT_READY"
        elif warnings_out:
            status = "READY_WITH_WARNINGS"
        else:
            status = "READY"
        return {"evidence_id": evidence_id, "status": status, "reasons": list(dict.fromkeys(reasons)), "warnings": list(dict.fromkeys(warnings_out))}

    def build_analyzer_input(self, evidence_id: str) -> dict[str, Any]:
        record = self.get_evidence(evidence_id)
        if record is None:
            raise ValueError(f"Evidence record not found: {evidence_id}")
        readiness = self.get_analyzer_readiness(evidence_id)
        if readiness["status"] not in {"READY", "READY_WITH_WARNINGS"}:
            raise ValueError(f"Evidence {evidence_id} is not ready: {'; '.join(readiness['reasons'])}")
        return self._build_contract(record)

    def ingest(self, filename: str, source: BinaryIO, case_id: str | None = None) -> dict[str, Any]:
        # Reset holds this same lock for its full operation; serializing a full
        # upload prevents an in-flight .incoming file or registration from
        # appearing halfway through a registry clear.
        with self._lock:
            return self._ingest_locked(filename, source, case_id)

    def _ingest_locked(self, filename: str, source: BinaryIO, case_id: str | None = None) -> dict[str, Any]:
        original = _safe_original_name(filename)
        result: dict[str, Any] = {
            "original_filename": original, "status": "processing", "stage": "RECEIVED",
            "stages": [{"stage": "RECEIVED", "status": "completed", "timestamp": _now()}],
            "errors": [], "warnings": [],
        }
        temp_path: Path | None = None

        def stage(name: str) -> None:
            result["stage"] = name
            result["stages"].append({"stage": name, "status": "in_progress", "timestamp": _now()})
            self._activity(original, name, f"{name.replace('_', ' ').title()} started")

        def finish_stage(name: str) -> None:
            for item in reversed(result["stages"]):
                if item["stage"] == name and item["status"] == "in_progress":
                    item.update(status="completed", timestamp=_now())
                    break

        try:
            stage("VALIDATING")
            extension = Path(original).suffix.lower()
            if not extension or extension not in ALLOWED_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {extension or '(no extension)'}")
            finish_stage("VALIDATING")

            stage("HASHING")
            temp_incoming_dir = self.storage_dir / "testing" / "temporary" if (self.storage_dir / "testing").exists() else self.evidence_dir
            temp_incoming_dir.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(prefix=".incoming-", dir=temp_incoming_dir)
            os.close(fd)
            temp_path = Path(temp_name)
            digest = hashlib.sha256()
            total = 0
            with temp_path.open("wb") as dest:
                while True:
                    chunk = source.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > self.max_file_size:
                        raise ValueError(f"File exceeds the {self.max_file_size}-byte upload limit")
                    digest.update(chunk)
                    dest.write(chunk)
                dest.flush()
                os.fsync(dest.fileno())
            if total == 0:
                raise ValueError("Empty files are not accepted")
            sha256 = digest.hexdigest()
            finish_stage("HASHING")

            with self._lock:
                existing = next((item for item in self.list_evidence() if item.get("sha256") == sha256), None)
            if existing:
                # Probe the attempted upload itself. A duplicate is not a new
                # registry record, so inherited registry metadata must never be
                # presented as though it was extracted from this upload session.
                try:
                    duplicate_extracted = extract_metadata(temp_path, extension, total, self.limits)
                except Exception as exc:
                    duplicate_extracted = {
                        "modality": modality_for_extension(extension),
                        "mime_type": existing.get("mime_type"), "metadata": {},
                        "warnings": [f"Metadata extraction failed: {exc}"],
                        "errors": [], "content_valid": None,
                    }
                duplicate_meta = {"modality": duplicate_extracted["modality"],
                                  "extension": extension, **duplicate_extracted["metadata"]}
                duplicate_validation = (
                    "unverified" if duplicate_extracted["content_valid"] is None else
                    "invalid_content" if duplicate_extracted["content_valid"] is False else
                    "valid" if duplicate_extracted["modality"] == "video" else
                    "unreadable" if duplicate_extracted["errors"] else "valid"
                )
                result.update(
                    status="duplicate", stage="COMPLETED", duplicate_reference=existing["evidence_id"],
                    sha256=sha256, file_size=total, metadata=duplicate_meta,
                    file_type=duplicate_extracted["modality"], mime_type=duplicate_extracted["mime_type"],
                    validation_status=duplicate_validation,
                    metadata_source_label="This upload (metadata extracted from the uploaded bytes)",
                    validation_source_label="This upload",
                    validation_errors=duplicate_extracted["errors"],
                    warnings=list(dict.fromkeys(result["warnings"] + duplicate_extracted["warnings"])),
                    analysis_eligibility={"status": "DUPLICATE_REFERENCE", "reasons": [f"Matches {existing['evidence_id']}."], "warnings": []},
                )
                result["stages"].append({"stage": "COMPLETED", "status": "completed", "timestamp": _now()})
                self._activity(original, "DUPLICATE", f"Matches existing evidence {existing['evidence_id']}", sha256=sha256)
                return result

            stage("METADATA_EXTRACTION")
            try:
                extracted = extract_metadata(temp_path, extension, total, self.limits)
            except Exception as exc:
                fallback_mime = {
                    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
                    ".mp4": "video/mp4", ".mov": "video/quicktime", ".wav": "audio/wav", ".mp3": "audio/mpeg",
                    ".m4a": "audio/mp4", ".txt": "text/plain", ".md": "text/markdown", ".pdf": "application/pdf",
                    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ".csv": "text/csv", ".json": "application/json",
                }[extension]
                extracted = {"modality": modality_for_extension(extension), "mime_type": fallback_mime,
                             "metadata": {}, "warnings": [f"Metadata extraction failed: {exc}"],
                             "errors": [], "content_valid": None}
            modality = extracted["modality"]
            mime_type = extracted["mime_type"]
            metadata = {"modality": modality, "extension": extension, **extracted["metadata"]}
            result["warnings"].extend(extracted["warnings"])
            result["errors"].extend(extracted["errors"])
            limit_reasons = analyzer_limit_reasons(modality, total, extracted["metadata"], self.limits)
            readiness_reasons = list(dict.fromkeys(list(extracted["errors"]) + limit_reasons))
            if extracted["content_valid"] is False and not readiness_reasons:
                readiness_reasons.append("File content does not match its declared format.")
            elif extracted["content_valid"] is None and not readiness_reasons:
                readiness_reasons.append("File content could not be verified by an available metadata tool.")
            readiness_status = "NOT_READY" if readiness_reasons else ("READY_WITH_WARNINGS" if result["warnings"] else "READY")
            analysis_eligibility = {"status": readiness_status, "reasons": readiness_reasons, "warnings": list(result["warnings"])}
            if extracted["content_valid"] is None:
                validation_status = "unverified"
            elif extracted["content_valid"] is False:
                validation_status = "invalid_content"
            elif modality == "video":
                # Container/file signature validation is distinct from whether
                # a video stream is present and decodable (metadata.stream_readable).
                validation_status = "valid"
            elif extracted["errors"]:
                validation_status = "unreadable"
            else:
                validation_status = "valid"
            finish_stage("METADATA_EXTRACTION")

            stage("STORAGE")
            stored_name = f"{uuid.uuid4().hex}{extension}"
            is_test = self.classify_as_testing({"case_id": case_id, "original_filename": original})
            dest_vault = self.get_dest_vault(modality, is_testing=is_test)
            dest_vault.mkdir(parents=True, exist_ok=True)
            stored_path = dest_vault / stored_name
            try:
                os.replace(temp_path, stored_path)
            except OSError:
                import shutil
                shutil.move(temp_path, stored_path)
            temp_path = None
            finish_stage("STORAGE")

            stage("REGISTRATION")
            with self._lock:
                records = self.list_evidence()
                # Also protect against a concurrent ingest with the same content.
                duplicate = next((item for item in records if item.get("sha256") == sha256), None)
                if duplicate:
                    stored_path.unlink(missing_ok=True)
                    result.update(
                        status="duplicate", stage="COMPLETED", duplicate_reference=duplicate["evidence_id"],
                        sha256=sha256, file_size=total, file_size_human=_human_size(total),
                        metadata=metadata, file_type=modality, mime_type=mime_type,
                        validation_status=validation_status,
                        metadata_source_label="This upload (metadata extracted from the uploaded bytes)",
                        validation_source_label="This upload",
                        validation_errors=extracted["errors"],
                        analysis_eligibility={"status": "DUPLICATE_REFERENCE", "reasons": [f"Matches {duplicate['evidence_id']}."], "warnings": []},
                    )
                    result["stages"].append({"stage": "COMPLETED", "status": "completed", "timestamp": _now()})
                    self._activity(original, "DUPLICATE", f"Matches existing evidence {duplicate['evidence_id']}", sha256=sha256)
                    return result
                existing_ids = [
                    int(match.group(1))
                    for row in records
                    if (match := re.fullmatch(r"EVD-(\d+)", str(row.get("evidence_id", ""))))
                ]
                max_id = max(existing_ids, default=0)
                evidence_id = f"EVD-{max_id + 1:03d}"
                record = {
                    "evidence_id": evidence_id, "case_id": case_id, "original_filename": original,
                    "stored_filename": stored_name, "file_path": str(stored_path), "storage_location": str(dest_vault),
                    "file_type": modality, "file_extension": extension, "mime_type": mime_type, "file_size": total,
                    "file_size_human": _human_size(total), "sha256": sha256,
                    "hash": {"algorithm": "SHA-256", "value": sha256, "status": "completed", "file_reference": str(stored_path)},
                    "ingestion_timestamp": _now(), "metadata": metadata,
                    "validation_status": validation_status, "ingestion_status": "completed",
                    "analysis_eligibility": analysis_eligibility,
                    "duplicate_reference": None, "errors": result["errors"], "warnings": result["warnings"],
                    "stages": result["stages"] + [{"stage": "COMPLETED", "status": "completed", "timestamp": _now()}],
                }
                records.append(record)
                self._atomic_json(self.registry_path, records)
                try:
                    self.db.upsert_evidence(record, is_testing=self.classify_as_testing(record))
                except Exception:
                    pass
            result.update(status="completed", stage="COMPLETED", evidence_id=evidence_id, sha256=sha256, file_size=total, metadata=metadata, evidence=record, analysis_eligibility=analysis_eligibility)
            self._activity(original, "COMPLETED", f"Registered as {evidence_id}", sha256=sha256)
            return result
        except Exception as exc:
            result.update(status="failed", stage="FAILED", error=str(exc), errors=[str(exc)])
            result["stages"].append({"stage": "FAILED", "status": "failed", "timestamp": _now(), "error": str(exc)})
            self._activity(original, "FAILED", str(exc), stage=result.get("stage"))
            return result
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

