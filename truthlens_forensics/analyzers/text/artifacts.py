from __future__ import annotations
import os
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional


class ArtifactManager:
    """
    Persists structured forensic artifacts to disk deterministically:
    - extracted_text.txt
    - text_extraction.json
    - entities.json
    - claims.json
    - linguistic_stats.json
    - consistency_analysis.json
    - ai_detection.json (when executed)
    - highlighted_evidence.json
    """

    @classmethod
    def get_artifact_directory(
        cls,
        evidence_id: str,
        ingestion_metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        meta = ingestion_metadata or {}
        override_dir = meta.get("artifact_dir") or meta.get("artifacts_dir")
        if override_dir:
            out_dir = Path(override_dir)
        else:
            # Default to storage/artifacts/{evidence_id}/text
            out_dir = Path("storage") / "artifacts" / evidence_id / "text"
        
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    @classmethod
    def save_artifacts(
        cls,
        evidence_id: str,
        raw_text: str,
        analyses: Dict[str, Any],
        signals: List[Dict[str, Any]],
        findings: List[Dict[str, Any]],
        ingestion_metadata: Optional[Dict[str, Any]] = None,
        final_output: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        target_dir = cls.get_artifact_directory(evidence_id, ingestion_metadata)
        artifacts: List[Dict[str, Any]] = []

        def _write_artifact(filename: str, content: str, art_type: str, desc: str) -> None:
            file_path = target_dir / filename
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            size = file_path.stat().st_size
            artifacts.append({
                "artifact_id": str(uuid.uuid4())[:8],
                "name": filename,
                "path": str(file_path.resolve()),
                "artifact_type": art_type,
                "description": desc,
                "file_size_bytes": size,
                "relative_url": f"/artifacts/{evidence_id}/text/{filename}"
            })

        # 1. extracted_text.txt
        _write_artifact("extracted_text.txt", raw_text, "text/plain", "Raw extracted text from evidence document.")

        # 2. text_extraction.json
        _write_artifact(
            "text_extraction.json",
            json.dumps(analyses.get("text_extraction", {}), indent=2, ensure_ascii=False),
            "application/json",
            "Document extraction metadata, structural counts, and paragraph/sentence maps."
        )

        # 3. entities.json
        _write_artifact(
            "entities.json",
            json.dumps(analyses.get("entity_extraction", {}), indent=2, ensure_ascii=False),
            "application/json",
            "Structured entities and mentions with character offsets."
        )

        # 4. claims.json
        _write_artifact(
            "claims.json",
            json.dumps(analyses.get("claim_extraction", {}), indent=2, ensure_ascii=False),
            "application/json",
            "Factual, temporal, causal, and opinion claims extracted from text."
        )

        # 5. linguistic_stats.json
        _write_artifact(
            "linguistic_stats.json",
            json.dumps(analyses.get("linguistic_analysis", {}), indent=2, ensure_ascii=False),
            "application/json",
            "Deterministic linguistic metrics, readability scores, and repetition patterns."
        )

        # 6. consistency_analysis.json
        _write_artifact(
            "consistency_analysis.json",
            json.dumps(analyses.get("consistency_analysis", {}), indent=2, ensure_ascii=False),
            "application/json",
            "Internal cross-referencing contradictions, mismatches, and duplicate blocks."
        )

        # 7. ai_detection.json (when executed)
        ai_det = analyses.get("ai_detector", {})
        if ai_det.get("inference_ran"):
            _write_artifact(
                "ai_detection.json",
                json.dumps(ai_det, indent=2, ensure_ascii=False),
                "application/json",
                "Model-backed AI text detection outputs, raw logits, and model limitations."
            )

        # 8. highlighted_evidence.json
        # Build structured annotations for UI/rendering
        annotations = []
        for s in signals:
            for p in s.get("evidence_spans", []):
                if isinstance(p, dict) and ("start_char" in p or "text" in p):
                    annotations.append({
                        "signal_id": s["signal_id"],
                        "signal_type": s["signal_type"],
                        "description": s["description"],
                        "passage": p
                    })

        highlight_payload = {
            "evidence_id": evidence_id,
            "annotations_count": len(annotations),
            "signals_summary": [{"id": s["signal_id"], "type": s["signal_type"], "desc": s["description"]} for s in signals],
            "findings_summary": [{"id": f["finding_id"], "category": f["category"], "severity": f["severity"], "desc": f["description"]} for f in findings],
            "highlighted_spans": annotations
        }
        _write_artifact(
            "highlighted_evidence.json",
            json.dumps(highlight_payload, indent=2, ensure_ascii=False),
            "application/json",
            "Traceable character span highlights and forensic annotations for report display."
        )

        # 9. standardized_output.json (Complete verifiable analytical result)
        if final_output:
            _write_artifact(
                "standardized_output.json",
                json.dumps(final_output, indent=2, ensure_ascii=False),
                "application/json",
                "Verifiable structured final analytical results delivered exclusively as standardized JSON."
            )

        return artifacts

