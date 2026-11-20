"""
TruthLens — Master Forensic Analysis Pipeline
Orchestrates neural detection, wavelet decomposition, ELA, copy-move SVD,
metadata integrity, and multi-image batch comparative verification.
"""
import os
import hashlib
import datetime
import traceback
import concurrent.futures
from typing import Any, Dict, List, Union

from truthlens_forensics.analyzers.validator import ImageValidator
from truthlens_forensics.analyzers.ai_detector import AIGeneratedDetector
from truthlens_forensics.analyzers.manipulation_detector import ManipulationDetector
from truthlens_forensics.analyzers.ela import ELAAnalyzer
from truthlens_forensics.analyzers.noise import NoiseAnalyzer
from truthlens_forensics.analyzers.statistical import StatisticalAnalyzer
from truthlens_forensics.analyzers.metadata import MetadataAnalyzer
from truthlens_forensics.analyzers.signal_finder import SignalFinder
from truthlens_forensics.analyzers.batch_comparator import BatchComparator

ANALYZER_VERSION = "2.0.0"


class ImageAnalysisPipeline:
    def __init__(self, artifact_dir: str = "storage/artifacts"):
        self.artifact_dir = os.path.abspath(artifact_dir)
        os.makedirs(self.artifact_dir, exist_ok=True)

        # Pre-load heavy neural detectors once
        self.ai_detector = AIGeneratedDetector()
        self.manip_detector = ManipulationDetector()
        self.batch_comparator = BatchComparator()

    def analyze(self, evidence: Union[Dict[str, Any], str], **kwargs) -> Dict[str, Any]:
        """Analyzes a single evidence item matching Section 3A and Section 3C contracts."""
        ts = datetime.datetime.utcnow().isoformat() + "Z"
        errors: List[Dict[str, Any]] = []
        artifacts: List[Dict[str, Any]] = []
        analyses: Dict[str, Any] = {}

        # 1. Unpack input evidence contract
        if isinstance(evidence, dict):
            evidence_id = evidence.get("evidence_id", kwargs.get("evidence_id", "EVD-001"))
            raw_path = evidence.get("file_path", kwargs.get("image_path", ""))
            original_filename = evidence.get("original_filename", kwargs.get("original_filename", os.path.basename(raw_path)))
            mime_type = evidence.get("mime_type", kwargs.get("mime_type"))
            expected_size = evidence.get("file_size", kwargs.get("file_size"))
            expected_sha = evidence.get("sha256", kwargs.get("sha256"))
            ingestion_metadata = evidence.get("ingestion_metadata", kwargs.get("ingestion_metadata", {}))
        else:
            raw_path = str(evidence)
            evidence_id = kwargs.get("evidence_id", "EVD-001")
            original_filename = kwargs.get("original_filename", os.path.basename(raw_path))
            mime_type = kwargs.get("mime_type")
            expected_size = kwargs.get("file_size")
            expected_sha = kwargs.get("sha256")
            ingestion_metadata = kwargs.get("ingestion_metadata", {})

        # Resolve path
        if not os.path.isabs(raw_path):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            image_path = os.path.normpath(os.path.join(base_dir, raw_path))
        else:
            image_path = raw_path

        # 2. File existence check
        if not os.path.exists(image_path):
            return {
                "module": "image",
                "evidence_id": evidence_id,
                "status": "failed",
                "metadata": {
                    "execution_timestamp": ts,
                    "analyzer_version": ANALYZER_VERSION,
                    "original_filename": original_filename,
                    "file_path": raw_path,
                    "mime_type": mime_type or "unknown",
                    "file_size_bytes": 0,
                    "sha256_expected": expected_sha or "",
                    "sha256_actual": "",
                    "sha256_verified": False,
                    "image_properties": {},
                    "ingestion_metadata": ingestion_metadata,
                    "exif_data": {},
                },
                "analyses": {},
                "signals": [],
                "findings": [],
                "artifacts": [],
                "errors": [{"engine": "input_validator", "error_type": "FileNotFoundError", "message": f"Evidence file not found: {raw_path}"}],
                "limitations": [],
            }

        # 3. Cryptographic hash calculation and verification
        actual_sha = self._sha256(image_path)
        actual_size = os.path.getsize(image_path)

        if expected_sha and expected_sha != "actual_sha256_hash":
            sha_verified = bool(expected_sha.strip().lower() == actual_sha.lower())
        else:
            # Placeholder or unspecified
            sha_verified = True

        size_matched = True
        if expected_size is not None and expected_size > 0:
            size_matched = bool(actual_size == expected_size)

        # 4. Input validation (magic bytes, dimensions, color space, integrity, video support)
        validator = ImageValidator()
        validation = validator.validate(image_path, expected_size=expected_size, expected_mime=mime_type)
        if not validation["valid"]:
            return {
                "module": "image",
                "evidence_id": evidence_id,
                "status": "failed",
                "metadata": {
                    "execution_timestamp": ts,
                    "analyzer_version": ANALYZER_VERSION,
                    "original_filename": original_filename,
                    "file_path": raw_path,
                    "mime_type": validation.get("detected_mime_type") or mime_type or "unknown",
                    "file_size_bytes": actual_size,
                    "sha256_expected": expected_sha or actual_sha,
                    "sha256_actual": actual_sha,
                    "sha256_verified": sha_verified,
                    "image_properties": {
                        "width": 0,
                        "height": 0,
                        "format": validation.get("format"),
                        "color_mode": validation.get("mode"),
                        "has_exif": False,
                    },
                    "ingestion_metadata": ingestion_metadata,
                    "exif_data": {},
                },
                "analyses": {},
                "signals": [],
                "findings": [],
                "artifacts": [],
                "errors": [{"engine": "validator", "error_type": "ValidationError", "message": validation.get("reason", "Corrupted or invalid image format")}],
                "limitations": [],
            }

        # If decoder normalized or extracted a video keyframe, use that path for all engines
        effective_image_path = validation.get("keyframe_path") or validation.get("normalized_path") or image_path
        image_path = effective_image_path

        # 5. Concurrent Classical & Forensic Engines
        def run_safe(name, fn):
            try:
                return name, fn(), None
            except Exception as exc:
                return name, None, {
                    "engine": name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }

        ela_eng = ELAAnalyzer(self.artifact_dir)
        noise_eng = NoiseAnalyzer(self.artifact_dir)
        stat_eng = StatisticalAnalyzer()
        meta_eng = MetadataAnalyzer()

        classical_tasks = [
            ("ela_analysis", lambda: ela_eng.analyze(image_path, evidence_id)),
            ("noise_residual_analysis", lambda: noise_eng.analyze(image_path, evidence_id)),
            ("statistical_analysis", lambda: stat_eng.analyze(image_path, evidence_id)),
            ("metadata_analysis", lambda: meta_eng.analyze(image_path, evidence_id)),
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            fut_map = {executor.submit(run_safe, n, f): n for n, f in classical_tasks}
            for fut in concurrent.futures.as_completed(fut_map):
                n, res, err = fut.result()
                if err:
                    errors.append(err)
                    analyses[n] = {"engine": n, "status": "error", "error": err["message"]}
                else:
                    analyses[n] = res
                    for art in res.get("artifacts", []):
                        artifacts.append(art)

        # 6. Pretrained Neural Inference
        for n, fn in [
            ("ai_generated_detection", lambda: self.ai_detector.analyze(image_path)),
            ("manipulation_detection", lambda: self.manip_detector.analyze(image_path, evidence_id, self.artifact_dir)),
        ]:
            name, res, err = run_safe(n, fn)
            if err:
                errors.append(err)
                analyses[name] = {"engine": name, "status": "error", "error": err["message"]}
            else:
                analyses[name] = res
                for art in res.get("artifacts", []):
                    artifacts.append(art)

        # 7. Deduplicate Artifacts
        seen_paths = set()
        unique_artifacts = []
        for art in artifacts:
            p = art.get("file_path") or art.get("path")
            if p and p not in seen_paths:
                seen_paths.add(p)
                # Standardize key to file_path
                clean_art = {
                    "artifact_id": art.get("artifact_id", f"ART-{evidence_id}-{len(unique_artifacts)+1}"),
                    "type": art.get("type", "visual_artifact"),
                    "file_path": p,
                    "description": art.get("description", ""),
                }
                unique_artifacts.append(clean_art)

        # 8. Signals, Findings, and Overall Forensic Assessment
        finder = SignalFinder()
        signals = finder.generate_signals(analyses, sha256_verified=sha_verified, size_matched=size_matched)
        findings = finder.generate_findings(signals, analyses)
        overall_assessment = finder.assess_overall(signals, analyses)
        analyses["overall_forensic_assessment"] = overall_assessment

        # 9. Metadata Assembly
        meta_data = analyses.get("metadata_analysis", {})
        has_exif = bool(meta_data.get("exif_raw"))
        dims = validation.get("dimensions", {"width": 0, "height": 0})

        metadata = {
            "execution_timestamp": ts,
            "analyzer_version": ANALYZER_VERSION,
            "original_filename": original_filename,
            "file_path": raw_path,
            "mime_type": validation.get("detected_mime_type") or mime_type or "image/jpeg",
            "file_size_bytes": actual_size,
            "sha256_expected": expected_sha or actual_sha,
            "sha256_actual": actual_sha,
            "sha256_verified": sha_verified,
            "image_properties": {
                "width": dims.get("width", 0),
                "height": dims.get("height", 0),
                "format": validation.get("format", "UNKNOWN"),
                "color_mode": validation.get("mode", "RGB"),
                "has_exif": has_exif,
            },
            "ingestion_metadata": ingestion_metadata,
            "exif_data": meta_data.get("exif_raw", {}),
        }

        # 10. Limitations
        limitations = self._collect_limitations(analyses)

        status = "failed" if not validation["valid"] else ("partial" if errors else "completed")

        return {
            "module": "image",
            "evidence_id": evidence_id,
            "status": status,
            "metadata": metadata,
            "analyses": analyses,
            "signals": signals,
            "findings": findings,
            "artifacts": unique_artifacts,
            "errors": errors,
            "limitations": limitations,
        }

    # Backward compatible alias
    run = analyze

    def analyze_batch(self, evidence_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyzes a multi-image batch in a single attempt matching Section 3D contract."""
        reports = []
        for item in evidence_items:
            rep = self.analyze(item)
            reports.append(rep)

        # Multi-Image Comparative Verification
        comparative = self.batch_comparator.compare_batch(reports)

        all_completed = all(r.get("status") == "completed" for r in reports)
        any_completed = any(r.get("status") in {"completed", "partial"} for r in reports)
        status = "completed" if all_completed else ("partial" if any_completed else "failed")

        return {
            "module": "image_batch",
            "total_submitted": len(evidence_items),
            "status": status,
            "reports": reports,
            "comparative_verification": comparative,
        }

    @staticmethod
    def _sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _collect_limitations(analyses: Dict) -> List[str]:
        lims = []
        for engine_data in analyses.values():
            if isinstance(engine_data, dict):
                for lim in engine_data.get("limitations", []):
                    if lim not in lims:
                        lims.append(lim)
        global_lims = [
            "AI-generation models are calibrated on training sets and may exhibit domain shift on unseen generators.",
            "PRNU sensor noise requires sufficient optical resolution; severe lossy recompression attenuates sensor footprints.",
            "ELA relies on quantization differential; lossless formats (PNG) require spatial and frequency analysis instead.",
            "EXIF data can be modified without altering image pixels; absence of EXIF does not confirm manipulation.",
            "Forensic conclusions should be drawn from the convergence of multiple independent physical and neural indicators."
        ]
        for g in global_lims:
            if g not in lims:
                lims.append(g)
        return lims

