from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from truthlens_forensics.analyzers.text.schemas import (
    TextEvidenceInput,
    TextAnalyzerOutput,
    TextExtractionAnalysis,
    ClaimExtractionAnalysis,
    EntityExtractionAnalysis,
    TemporalAnalysis,
    LocationAnalysis,
    LinguisticAnalysis,
    ConsistencyAnalysis,
    TextComparisonAnalysis,
    AIDetectorAnalysis,
)
from truthlens_forensics.analyzers.text.extractors import DocumentExtractor, ExtractionError
from truthlens_forensics.analyzers.text.entities import EntityExtractor
from truthlens_forensics.analyzers.text.claims import ClaimExtractor
from truthlens_forensics.analyzers.text.temporal_location import TemporalLocationAnalyzer
from truthlens_forensics.analyzers.text.linguistics import LinguisticAnalyzer
from truthlens_forensics.analyzers.text.consistency import ConsistencyAnalyzer
from truthlens_forensics.analyzers.text.comparison import TextComparisonAnalyzer
from truthlens_forensics.analyzers.text.ai_detector import AITextDetector
from truthlens_forensics.analyzers.text.signals_findings import SignalFindingSynthesizer
from truthlens_forensics.analyzers.text.artifacts import ArtifactManager

logger = logging.getLogger(__name__)


class TextAnalyzer:
    """
    TruthLens Text Forensics Analyzer.

    Specification:
    Integrate a standardized extractor service that validates and reads TXT, MD, CSV, JSON, PDF, and DOCX files.
    Modify the pipeline to execute all analyses—including entity, claim, temporal, location, linguistic, internal consistency, and AI detection—in isolation.
    Ensure all processed text and final analytical results are stored as verifiable, structured artifact files, and mandate that the final output is delivered exclusively as a standardized JSON structure.
    """

    def analyze_text(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Primary forensic analysis entry point for text evidence.
        Guarantees exact schema compliance and zero unhandled exceptions.
        """
        evidence_id = str(request.get("evidence_id", "EVD-UNKNOWN"))
        file_path_str = str(request.get("file_path", ""))
        original_filename = request.get("original_filename") or Path(file_path_str).name
        expected_sha256 = request.get("sha256")
        ingestion_metadata = request.get("ingestion_metadata") or {}

        errors: List[str] = []
        limitations: List[str] = [
            "Text forensics analyzes intra-document semantic and statistical structures.",
            "Module operates independently without cross-evidence correlation or final report generation."
        ]

        # Initialize analyses structure
        analyses: Dict[str, Any] = {
            "text_extraction": TextExtractionAnalysis(status="failed").model_dump(),
            "claim_extraction": ClaimExtractionAnalysis(status="skipped").model_dump(),
            "entity_extraction": EntityExtractionAnalysis(status="skipped").model_dump(),
            "temporal_analysis": TemporalAnalysis(status="skipped").model_dump(),
            "location_analysis": LocationAnalysis(status="skipped").model_dump(),
            "linguistic_analysis": LinguisticAnalysis(status="skipped").model_dump(),
            "consistency_analysis": ConsistencyAnalysis(status="skipped").model_dump(),
            "text_comparison": TextComparisonAnalysis(status="skipped").model_dump(),
            "ai_detector": AIDetectorAnalysis(status="not_implemented").model_dump(),
        }

        top_level_metadata: Dict[str, Any] = {
            "evidence_id": evidence_id,
            "original_filename": original_filename,
            "file_path": file_path_str,
            "file_type": request.get("file_type", "text"),
            "mime_type": request.get("mime_type"),
            "file_size": request.get("file_size"),
            "sha256": expected_sha256,
            "ingestion_metadata": ingestion_metadata
        }

        # Step 1 & 2: Validate file and extract document
        extraction_data = None
        try:
            extraction_data = DocumentExtractor.extract_document(file_path_str, expected_sha256)
        except FileNotFoundError as e:
            errors.append(f"File not found: {str(e)}")
            return self._build_output("failed", evidence_id, top_level_metadata, analyses, [], [], [], errors, limitations)
        except ValueError as e:
            errors.append(f"Validation error: {str(e)}")
            return self._build_output("failed", evidence_id, top_level_metadata, analyses, [], [], [], errors, limitations)
        except ExtractionError as e:
            errors.append(f"Extraction failure: {str(e)}")
            return self._build_output("failed", evidence_id, top_level_metadata, analyses, [], [], [], errors, limitations)
        except Exception as e:
            errors.append(f"Unexpected extraction error: {str(e)}")
            return self._build_output("failed", evidence_id, top_level_metadata, analyses, [], [], [], errors, limitations)

        # Update metadata and extraction analysis
        top_level_metadata["file_size"] = extraction_data["file_size"]
        top_level_metadata["sha256"] = extraction_data["sha256"]
        top_level_metadata["mime_type"] = extraction_data["mime_type"]
        top_level_metadata["character_encoding"] = extraction_data["character_encoding"]

        raw_text = extraction_data["raw_text"]
        paragraphs = extraction_data["paragraphs"]
        sentences = extraction_data["sentences"]

        analyses["text_extraction"] = {
            "status": "completed",
            "extraction_method": extraction_data["extraction_method"],
            "character_encoding": extraction_data["character_encoding"],
            "embedded_metadata": extraction_data["embedded_metadata"],
            "structural_counts": extraction_data["structural_counts"],
            "language_detection": extraction_data["language_detection"],
            "paragraphs": paragraphs,
            "sentences": sentences,
            "raw_text_length": len(raw_text),
            "errors": [],
            "limitations": []
        }

        has_partial_failure = False

        # Step 3: Entity Extraction
        try:
            analyses["entity_extraction"] = EntityExtractor.extract_entities(raw_text, paragraphs)
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            has_partial_failure = True
            errors.append(f"Entity extraction error: {str(e)}")
            analyses["entity_extraction"] = {"status": "failed", "errors": [str(e)], "entities": []}

        entities = analyses["entity_extraction"].get("entities", [])

        # Step 4: Claim Extraction
        try:
            analyses["claim_extraction"] = ClaimExtractor.extract_claims(sentences, entities)
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            has_partial_failure = True
            errors.append(f"Claim extraction error: {str(e)}")
            analyses["claim_extraction"] = {"status": "failed", "errors": [str(e)], "claims": []}

        claims = analyses["claim_extraction"].get("claims", [])

        # Step 5: Temporal Analysis
        try:
            analyses["temporal_analysis"] = TemporalLocationAnalyzer.analyze_temporal(raw_text, sentences, entities)
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            has_partial_failure = True
            errors.append(f"Temporal analysis error: {str(e)}")
            analyses["temporal_analysis"] = {"status": "failed", "errors": [str(e)], "temporal_expressions": []}

        # Step 6: Location Analysis
        try:
            analyses["location_analysis"] = TemporalLocationAnalyzer.analyze_location(raw_text, sentences, entities)
        except Exception as e:
            logger.error(f"Location analysis failed: {e}")
            has_partial_failure = True
            errors.append(f"Location analysis error: {str(e)}")
            analyses["location_analysis"] = {"status": "failed", "errors": [str(e)], "location_mentions": []}

        # Step 7: Linguistic Analysis
        try:
            analyses["linguistic_analysis"] = LinguisticAnalyzer.analyze_linguistics(raw_text, paragraphs, sentences)
        except Exception as e:
            logger.error(f"Linguistic analysis failed: {e}")
            has_partial_failure = True
            errors.append(f"Linguistic analysis error: {str(e)}")
            analyses["linguistic_analysis"] = {"status": "failed", "errors": [str(e)]}

        # Step 8: Internal Consistency Analysis
        try:
            analyses["consistency_analysis"] = ConsistencyAnalyzer.analyze_consistency(
                text=raw_text,
                paragraphs=paragraphs,
                sentences=sentences,
                entities=entities,
                claims=claims,
                temporal_analysis=analyses["temporal_analysis"],
                location_analysis=analyses["location_analysis"],
                embedded_metadata=extraction_data["embedded_metadata"]
            )
        except Exception as e:
            logger.error(f"Consistency analysis failed: {e}")
            has_partial_failure = True
            errors.append(f"Consistency analysis error: {str(e)}")
            analyses["consistency_analysis"] = {"status": "failed", "errors": [str(e)], "inconsistencies": []}

        # Step 9: Text Comparison
        try:
            analyses["text_comparison"] = TextComparisonAnalyzer.compare_blocks(paragraphs, ingestion_metadata)
        except Exception as e:
            logger.error(f"Text comparison failed: {e}")
            has_partial_failure = True
            errors.append(f"Text comparison error: {str(e)}")
            analyses["text_comparison"] = {"status": "failed", "errors": [str(e)]}

        # Step 10: AI-Generated Text Detector
        try:
            analyses["ai_detector"] = AITextDetector.detect_ai_text(raw_text)
        except Exception as e:
            logger.error(f"AI detector failed: {e}")
            # Non-fatal sub-component failure
            has_partial_failure = True
            errors.append(f"AI detector error: {str(e)}")
            analyses["ai_detector"] = {"status": "failed", "errors": [str(e)]}

        # Step 11: Signals & Findings Synthesis
        signals: List[Dict[str, Any]] = []
        findings: List[Dict[str, Any]] = []
        try:
            signals, findings = SignalFindingSynthesizer.synthesize(analyses)
        except Exception as e:
            logger.error(f"Signals/Findings synthesis error: {e}")
            has_partial_failure = True
            errors.append(f"Signals synthesis error: {str(e)}")

        # Step 12: Write Forensic Artifacts
        artifacts: List[Dict[str, Any]] = []
        overall_status = "partial" if has_partial_failure else "completed"
        try:
            draft_output = {
                "module": "text",
                "evidence_id": evidence_id,
                "status": overall_status,
                "metadata": top_level_metadata,
                "analyses": analyses,
                "signals": signals,
                "findings": findings,
                "errors": errors,
                "limitations": limitations
            }
            artifacts = ArtifactManager.save_artifacts(
                evidence_id=evidence_id,
                raw_text=raw_text,
                analyses=analyses,
                signals=signals,
                findings=findings,
                ingestion_metadata=ingestion_metadata,
                final_output=draft_output
            )
        except Exception as e:
            logger.error(f"Artifact persistence error: {e}")
            errors.append(f"Artifact creation error: {str(e)}")

        return self._build_output(
            overall_status,
            evidence_id,
            top_level_metadata,
            analyses,
            signals,
            findings,
            artifacts,
            errors,
            limitations
        )

    def _build_output(
        self,
        status: str,
        evidence_id: str,
        metadata: Dict[str, Any],
        analyses: Dict[str, Any],
        signals: List[Dict[str, Any]],
        findings: List[Dict[str, Any]],
        artifacts: List[Dict[str, Any]],
        errors: List[str],
        limitations: List[str]
    ) -> Dict[str, Any]:
        output = TextAnalyzerOutput(
            module="text",
            evidence_id=evidence_id,
            status=status,
            metadata=metadata,
            analyses=analyses,
            signals=signals,
            findings=findings,
            artifacts=artifacts,
            errors=errors,
            limitations=limitations
        )
        return output.model_dump()


def analyze_text(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main integration entry point matching TruthLens specification.
    """
    analyzer = TextAnalyzer()
    return analyzer.analyze_text(request)

