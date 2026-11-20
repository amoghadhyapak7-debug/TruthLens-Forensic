"""TruthLens Image Forensics Adapter — wraps Ramesh's ImageAnalysisPipeline."""
from __future__ import annotations
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from truthlens_forensics.pipeline import ImageAnalysisPipeline
        import os
        artifact_dir = os.getenv("TRUTHLENS_DATA_DIR", "data") + "/artifacts/image"
        os.makedirs(artifact_dir, exist_ok=True)
        _pipeline = ImageAnalysisPipeline(artifact_dir=artifact_dir)
    return _pipeline

def analyze_image_evidence(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Run Ramesh's image forensic pipeline on a registered evidence item."""
    try:
        pipeline = _get_pipeline()
        result = pipeline.analyze(contract)
        return result
    except Exception as exc:
        logger.error(f"Image analysis failed: {exc}", exc_info=True)
        return {
            "module": "image",
            "evidence_id": contract.get("evidence_id", "unknown"),
            "status": "failed",
            "errors": [str(exc)],
            "findings": [],
            "signals": [],
        }
