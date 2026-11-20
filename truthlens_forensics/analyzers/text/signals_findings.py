from __future__ import annotations
import uuid
from typing import Dict, Any, List, Optional, Tuple


class SignalFindingSynthesizer:
    """
    Synthesizes purely observable signals and evidence-backed forensic findings.
    Strictly forbids hallucinating or fabricating signals/findings: clean documents
    produce empty signals and findings lists. Does not assign global SIG-* or FND-* IDs.
    """

    @classmethod
    def synthesize(
        cls,
        analyses: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        signals: List[Dict[str, Any]] = []
        findings: List[Dict[str, Any]] = []

        # 1. Extraction warnings
        extraction = analyses.get("text_extraction", {})
        for w in extraction.get("embedded_metadata", {}).get("extraction_warnings", []):
            signals.append({
                "signal_id": str(uuid.uuid4())[:8],
                "signal_type": "extraction_warning",
                "source_analysis": "text_extraction",
                "description": f"Extraction warning encountered: {w}",
                "evidence_spans": [],
                "metrics": {},
                "confidence": 0.95
            })

        # 2. Consistency Analysis Signals
        consistency = analyses.get("consistency_analysis", {})
        for item in consistency.get("inconsistencies", []):
            c_type = item.get("classification")
            desc = item.get("description", "")
            passages = item.get("supporting_passages", [])

            sig_type = "internal_contradiction"
            if "numerical" in desc.lower() or "conflicting numerical" in desc.lower() or "amount" in desc.lower():
                sig_type = "conflicting_quantities"
            elif "dates" in desc.lower() or "chronology" in desc.lower() or "timestamp" in desc.lower():
                sig_type = "conflicting_dates"
            elif "location" in desc.lower():
                sig_type = "conflicting_locations"
            elif "duplicate" in desc.lower():
                sig_type = "duplicate_paragraph"
            elif "metadata" in desc.lower() or "author" in desc.lower():
                sig_type = "metadata_mismatch"
            elif "negation" in desc.lower() or "contradiction" in desc.lower():
                sig_type = "contradictory_claims"

            signals.append({
                "signal_id": str(uuid.uuid4())[:8],
                "signal_type": sig_type,
                "source_analysis": "consistency_analysis",
                "description": desc,
                "evidence_spans": passages,
                "metrics": {"classification": c_type},
                "confidence": 0.90 if c_type == "confirmed_textual_mismatch" else 0.75
            })

        # 3. Linguistic & Statistical Signals
        linguistics = analyses.get("linguistic_analysis", {})
        # A. High repetitions
        rep = linguistics.get("repetition_metrics", {})
        repeated_4g = rep.get("repeated_phrases", {}).get("fourgrams", [])
        if any(fg.get("count", 0) >= 3 for fg in repeated_4g):
            sample = [f"'{fg['phrase']}' (x{fg['count']})" for fg in repeated_4g if fg.get("count", 0) >= 3]
            signals.append({
                "signal_id": str(uuid.uuid4())[:8],
                "signal_type": "repeated_phrases",
                "source_analysis": "linguistic_analysis",
                "description": f"High multi-word repetition detected: {', '.join(sample)}",
                "evidence_spans": [],
                "metrics": {"repeated_fourgrams": repeated_4g},
                "confidence": 0.85
            })

        # B. Unusual punctuation
        punct = linguistics.get("punctuation_patterns", {})
        if punct.get("unusual_punctuation_flag"):
            signals.append({
                "signal_id": str(uuid.uuid4())[:8],
                "signal_type": "unusual_punctuation_or_casing",
                "source_analysis": "linguistic_analysis",
                "description": f"Unusual punctuation patterns: {punct.get('multiple_exclamations', 0)} excessive exclamation runs, {punct.get('multiple_ellipses', 0)} ellipsis runs.",
                "evidence_spans": [],
                "metrics": punct,
                "confidence": 0.70
            })

        # C. Unusual capitalization
        caps = linguistics.get("capitalization_patterns", {})
        if caps.get("excessive_caps_flag"):
            signals.append({
                "signal_id": str(uuid.uuid4())[:8],
                "signal_type": "unusual_punctuation_or_casing",
                "source_analysis": "linguistic_analysis",
                "description": f"Excessive uppercase styling: {round(caps.get('uppercase_char_ratio', 0)*100, 1)}% uppercase characters, {caps.get('all_caps_word_count', 0)} ALL-CAPS words.",
                "evidence_spans": [],
                "metrics": caps,
                "confidence": 0.80
            })

        # D. Section shifts
        shifts = linguistics.get("section_shifts", [])
        if shifts:
            signals.append({
                "signal_id": str(uuid.uuid4())[:8],
                "signal_type": "linguistic_shift",
                "source_analysis": "linguistic_analysis",
                "description": f"Abrupt structural or stylistic shift detected across {len(shifts)} paragraph transitions.",
                "evidence_spans": [],
                "metrics": {"shift_events": shifts},
                "confidence": 0.65
            })

        # 4. Text comparison (intra-document duplicate blocks)
        comparison = analyses.get("text_comparison", {})
        dup_blocks = comparison.get("duplicate_blocks", [])
        # Only add if not already captured in consistency
        already_has_dup_sig = any(s["signal_type"] == "duplicate_paragraph" for s in signals)
        if dup_blocks and not already_has_dup_sig:
            signals.append({
                "signal_id": str(uuid.uuid4())[:8],
                "signal_type": "duplicate_paragraph",
                "source_analysis": "text_comparison",
                "description": f"Found {len(dup_blocks)} duplicate or near-duplicate paragraph blocks.",
                "evidence_spans": [b.get("first_paragraph", {}) for b in dup_blocks],
                "metrics": {"duplicate_count": len(dup_blocks)},
                "confidence": 0.90
            })

        # 5. AI detector score signal
        ai_det = analyses.get("ai_detector", {})
        if ai_det.get("inference_ran") and ai_det.get("status") == "completed":
            raw_scores = ai_det.get("raw_scores", {})
            fake_score = raw_scores.get("Fake", 0.0)
            if fake_score > 0.80:
                signals.append({
                    "signal_id": str(uuid.uuid4())[:8],
                    "signal_type": "ai_detector_score",
                    "source_analysis": "ai_detector",
                    "description": f"AI detector model '{ai_det.get('model_name')}' flagged elevated synthetic likelihood (score: {fake_score}).",
                    "evidence_spans": [],
                    "metrics": raw_scores,
                    "confidence": fake_score
                })

        # -------------------------------------------------------------
        # SYNTHESIZE FINDINGS (Evidence-backed only, zero hallucinations)
        # -------------------------------------------------------------

        # Category A: Internal Inconsistency
        inconsistency_sigs = [s for s in signals if s["signal_type"] in {"conflicting_quantities", "contradictory_claims", "conflicting_locations"}]
        if inconsistency_sigs:
            passages = []
            for s in inconsistency_sigs:
                passages.extend(s.get("evidence_spans", []))
            
            severities = ["high" if any(s.get("metrics", {}).get("classification") == "confirmed_textual_mismatch" for s in inconsistency_sigs) else "medium"]
            findings.append({
                "finding_id": str(uuid.uuid4())[:8],
                "category": "internal_inconsistency",
                "description": f"Document contains {len(inconsistency_sigs)} mutually conflicting factual or numerical assertions.",
                "severity": severities[0],
                "supporting_signals": [s["signal_id"] for s in inconsistency_sigs],
                "relevant_passages": passages[:6],
                "explanation": "Multiple mutually contradictory factual statements, numbers, or spatial claims were detected within the evidence text.",
                "limitations": ["Evaluated strictly from internal document assertions without external ground-truth correlation."],
                "confidence": round(sum(s["confidence"] for s in inconsistency_sigs) / len(inconsistency_sigs), 2)
            })

        # Category B: Temporal Anomaly
        temporal_sigs = [s for s in signals if s["signal_type"] == "conflicting_dates"]
        if temporal_sigs:
            passages = []
            for s in temporal_sigs:
                passages.extend(s.get("evidence_spans", []))

            findings.append({
                "finding_id": str(uuid.uuid4())[:8],
                "category": "temporal_anomaly",
                "description": f"Detected {len(temporal_sigs)} chronological contradictions or timeline reversals.",
                "severity": "medium",
                "supporting_signals": [s["signal_id"] for s in temporal_sigs],
                "relevant_passages": passages[:4],
                "explanation": "Narrative sequence markers or explicit dates conflict with chronological ordering within the narrative.",
                "limitations": ["Relative date references may lack explicit anchor dates."],
                "confidence": round(sum(s["confidence"] for s in temporal_sigs) / len(temporal_sigs), 2)
            })

        # Category C: Duplication Anomaly
        dup_sigs = [s for s in signals if s["signal_type"] in {"duplicate_paragraph", "repeated_phrases"}]
        if dup_sigs:
            findings.append({
                "finding_id": str(uuid.uuid4())[:8],
                "category": "duplication_anomaly",
                "description": "Evidence contains repeated verbatim paragraph blocks or highly redundant structural phrasing.",
                "severity": "low",
                "supporting_signals": [s["signal_id"] for s in dup_sigs],
                "relevant_passages": [p for s in dup_sigs for p in s.get("evidence_spans", [])][:4],
                "explanation": "Substantial textual duplication indicates copy-pasting, boilerplate re-use, or unpruned draft compilation.",
                "limitations": ["Boilerplate or legal disclaimers can legitimately repeat across sections."],
                "confidence": round(sum(s["confidence"] for s in dup_sigs) / len(dup_sigs), 2)
            })

        # Category D: Metadata Anomaly
        meta_sigs = [s for s in signals if s["signal_type"] == "metadata_mismatch"]
        if meta_sigs:
            findings.append({
                "finding_id": str(uuid.uuid4())[:8],
                "category": "metadata_anomaly",
                "description": "Container metadata diverges from textual declarations in document body.",
                "severity": "medium",
                "supporting_signals": [s["signal_id"] for s in meta_sigs],
                "relevant_passages": [p for s in meta_sigs for p in s.get("evidence_spans", [])],
                "explanation": "Embedded file container properties (such as author or creation tool) contradict body statements.",
                "limitations": ["Document conversion tools or templates may preserve legacy container metadata."],
                "confidence": round(sum(s["confidence"] for s in meta_sigs) / len(meta_sigs), 2)
            })

        # Category E: Linguistic Anomaly
        ling_sigs = [s for s in signals if s["signal_type"] in {"unusual_punctuation_or_casing", "linguistic_shift"}]
        if len(ling_sigs) >= 2:  # Only emit finding if corroborated by multiple signals
            findings.append({
                "finding_id": str(uuid.uuid4())[:8],
                "category": "linguistic_anomaly",
                "description": "Significant stylistic irregularity or abrupt section-level structural shifts observed.",
                "severity": "low",
                "supporting_signals": [s["signal_id"] for s in ling_sigs],
                "relevant_passages": [],
                "explanation": "Text exhibits non-standard typography, erratic capitalization, or sharp paragraph length variance.",
                "limitations": ["Stylistic variance is not evidence of deception and may reflect informal or multi-author writing."],
                "confidence": 0.65
            })

        return signals, findings

