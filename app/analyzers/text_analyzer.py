"""TruthLens Text Forensics Adapter — wraps Rakshitha's TextAnalyzer."""
from __future__ import annotations
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_analyzer = None

def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from truthlens_forensics.analyzers.text.analyzer import TextAnalyzer
        _analyzer = TextAnalyzer()
    return _analyzer

def analyze_text_evidence(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Run Rakshitha's text forensic analyzer on a registered evidence item."""
    try:
        analyzer = _get_analyzer()
        result = analyzer.analyze_text(contract)
        return result
    except Exception as exc:
        logger.error(f"Text analysis failed: {exc}", exc_info=True)
        return {
            "module": "text",
            "evidence_id": contract.get("evidence_id", "unknown"),
            "status": "failed",
            "errors": [str(exc)],
            "findings": [],
            "signals": [],
        }
