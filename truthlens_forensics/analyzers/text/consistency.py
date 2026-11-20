from __future__ import annotations
import re
import uuid
import difflib
from typing import Dict, Any, List, Optional, Tuple


class ConsistencyAnalyzer:
    """
    Intra-document cross-checking service detecting:
    - Conflicting dates or times
    - Conflicting quantities or monetary figures
    - Conflicting names, titles, or identities
    - Conflicting locations
    - Contradictory factual statements
    - Exact or near-duplicate paragraphs/blocks
    - Information mismatches across sections / embedded metadata vs body
    """

    @classmethod
    def analyze_consistency(
        cls,
        text: str,
        paragraphs: List[Dict[str, Any]],
        sentences: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        claims: List[Dict[str, Any]],
        temporal_analysis: Dict[str, Any],
        location_analysis: Dict[str, Any],
        embedded_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        inconsistencies = []

        # 1. Forward conflicts from temporal analysis
        for tc in temporal_analysis.get("potential_chronology_conflicts", []):
            p1 = tc.get("first_reference", {})
            p2 = tc.get("second_reference", {})
            inconsistencies.append({
                "item_id": str(uuid.uuid4())[:8],
                "classification": "confirmed_textual_mismatch" if tc.get("conflict_type") == "conflicting_dates_for_same_event" else "potential_contradiction",
                "description": tc.get("description", "Temporal chronology conflict detected."),
                "supporting_passages": [
                    {"text": p1.get("context", p1.get("date", "")), "role": "earlier_reference"},
                    {"text": p2.get("context", p2.get("date", "")), "role": "conflicting_reference"}
                ],
                "explanation": tc.get("explanation", "Narrative chronology or explicit timestamps conflict."),
                "limitations": ["Evaluated strictly from intra-text timestamps without external timeline access."]
            })

        # 2. Forward conflicts from location analysis
        for lc in location_analysis.get("potential_location_conflicts", []):
            passages = []
            if "supporting_sentence" in lc:
                passages.append({"text": lc["supporting_sentence"], "role": "simultaneous_claim"})
            if "first_reference" in lc and "second_reference" in lc:
                passages.append({"text": lc["first_reference"], "role": "primary_location"})
                passages.append({"text": lc["second_reference"], "role": "conflicting_location"})
            
            inconsistencies.append({
                "item_id": str(uuid.uuid4())[:8],
                "classification": "potential_contradiction",
                "description": lc.get("description", "Location conflict detected."),
                "supporting_passages": passages,
                "explanation": lc.get("explanation", "Mutually exclusive locations asserted for the same entity or event."),
                "limitations": ["Evaluated based on textual location co-occurrence and context."]
            })

        # 3. Conflicting quantities or monetary figures for the same subject
        quantity_conflicts = cls._detect_conflicting_numbers(sentences, entities)
        inconsistencies.extend(quantity_conflicts)

        # 4. Conflicting titles / roles / identity assertions
        identity_conflicts = cls._detect_identity_conflicts(sentences, entities)
        inconsistencies.extend(identity_conflicts)

        # 5. Direct contradictory factual statements (negation / polarity conflict)
        factual_contradictions = cls._detect_factual_contradictions(sentences)
        inconsistencies.extend(factual_contradictions)

        # 6. Exact or near-duplicate paragraphs / blocks
        duplicate_blocks = cls._detect_duplicate_blocks(paragraphs)
        inconsistencies.extend(duplicate_blocks)

        # 7. Metadata vs Body text mismatches
        meta_mismatches = cls._detect_metadata_mismatches(text, embedded_metadata)
        inconsistencies.extend(meta_mismatches)

        return {
            "status": "completed",
            "items_count": len(inconsistencies),
            "inconsistencies": inconsistencies,
            "errors": [],
            "limitations": [
                "Consistency analysis performs intra-document cross-referencing and factual coherence checks.",
                "Only statements exhibiting clear semantic opposition or numerical incompatibility are classified as contradictory."
            ]
        }

    @classmethod
    def _detect_conflicting_numbers(
        cls,
        sentences: List[Dict[str, Any]],
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detects conflicting monetary amounts or quantities attached to the same subject phrase.
        e.g., 'The payment amount was $50,000' vs 'The payment amount was $75,000'
        """
        conflicts = []
        num_patterns = [
            r'(?:total\s+(?:loss|cost|budget|amount|revenue|price|damage)|payment\s+of|sum\s+of|fine\s+of|settlement\s+of|transferred|worth|valued\s+at)\s+([\$€£¥]?\s?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:million|billion|k|M|B|USD|EUR))?)'
        ]

        subject_map: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}

        for s in sentences:
            s_text = s["text"]
            for pat in num_patterns:
                for match in re.finditer(pat, s_text, re.IGNORECASE):
                    # extract subject (prefix) and amount
                    matched_phrase = match.group(0)
                    amount = match.group(1).strip()
                    # Normalized subject key
                    subj_key = re.sub(r'[\$€£¥]?\s?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:million|billion|k|M|B|USD|EUR))?', '', matched_phrase).strip().lower()
                    if subj_key:
                        if subj_key not in subject_map:
                            subject_map[subj_key] = []
                        subject_map[subj_key].append((amount, s_text, s))

        for subj, occurrences in subject_map.items():
            if len(occurrences) > 1:
                # check if amounts differ
                distinct_amounts = {occ[0] for occ in occurrences}
                if len(distinct_amounts) > 1:
                    conflicts.append({
                        "item_id": str(uuid.uuid4())[:8],
                        "classification": "confirmed_textual_mismatch",
                        "description": f"Conflicting numerical figures asserted for '{subj}': {sorted(list(distinct_amounts))}",
                        "supporting_passages": [
                            {"text": occurrences[0][1], "amount": occurrences[0][0], "span": occurrences[0][2]},
                            {"text": occurrences[1][1], "amount": occurrences[1][0], "span": occurrences[1][2]}
                        ],
                        "explanation": f"The document provides divergent numerical or financial figures for the same item '{subj}'.",
                        "limitations": ["Evaluated strictly from textual co-occurrence within the document."]
                    })

        return conflicts

    @classmethod
    def _detect_identity_conflicts(
        cls,
        sentences: List[Dict[str, Any]],
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detects conflicting titles, roles, or organizational affiliations for the same person.
        """
        conflicts = []
        role_pattern = re.compile(
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:serves\s+as|is\s+the|was\s+the|appointed\s+as|employed\s+as|acting\s+as)\s+([A-Z][A-Za-z\s]+?)(?:[.,;]|\s+at\b|\s+of\b|\s+and\b|$)',
            re.IGNORECASE
        )

        person_roles: Dict[str, List[Tuple[str, str]]] = {}
        for s in sentences:
            for match in role_pattern.finditer(s["text"]):
                p_name = match.group(1).strip().title()
                role = match.group(2).strip().lower()
                if len(role) > 3 and not any(w in role for w in ["the", "a", "an", "then"]):
                    if p_name not in person_roles:
                        person_roles[p_name] = []
                    person_roles[p_name].append((role, s["text"]))

        for p_name, roles in person_roles.items():
            distinct_roles = {r[0] for r in roles}
            if len(distinct_roles) > 1:
                conflicts.append({
                    "item_id": str(uuid.uuid4())[:8],
                    "classification": "potential_contradiction",
                    "description": f"Conflicting roles or titles asserted for '{p_name}': {list(distinct_roles)}",
                    "supporting_passages": [
                        {"text": roles[0][1], "role_claimed": roles[0][0]},
                        {"text": roles[1][1], "role_claimed": roles[1][0]}
                    ],
                    "explanation": f"Multiple distinct primary roles or titles are attributed to '{p_name}' across the document.",
                    "limitations": ["May represent role promotion or multi-role status if temporal progression exists."]
                })

        return conflicts

    @classmethod
    def _detect_factual_contradictions(cls, sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detects direct polarity contradictions (e.g. statement asserted then directly negated).
        e.g., 'The system was compromised' vs 'The system was not compromised'
        """
        conflicts = []
        
        # Check pairs of sentences for direct polarity reversal
        for i in range(len(sentences)):
            s1_text = sentences[i]["text"]
            for j in range(i + 1, min(len(sentences), i + 10)):
                s2_text = sentences[j]["text"]
                
                # Check negation contradiction
                # Match core words
                words1 = set(re.findall(r'\b[a-zA-Z]{3,}\b', s1_text.lower()))
                words2 = set(re.findall(r'\b[a-zA-Z]{3,}\b', s2_text.lower()))
                
                common = words1.intersection(words2)
                # If high word overlap
                if len(common) >= 4 and len(common) / max(len(words1), len(words2)) > 0.6:
                    neg1 = bool(re.search(r'\b(?:not|never|neither|no|denied|failed\s+to)\b', s1_text.lower()))
                    neg2 = bool(re.search(r'\b(?:not|never|neither|no|denied|failed\s+to)\b', s2_text.lower()))
                    if neg1 != neg2:
                        conflicts.append({
                            "item_id": str(uuid.uuid4())[:8],
                            "classification": "potential_contradiction",
                            "description": "Direct assertion and negation contradiction detected between statements.",
                            "supporting_passages": [
                                {"text": s1_text, "polarity": "negative" if neg1 else "affirmative"},
                                {"text": s2_text, "polarity": "negative" if neg2 else "affirmative"}
                            ],
                            "explanation": "One statement affirms an action or state while the second statement negates it with high lexical overlap.",
                            "limitations": ["Requires semantic contextual interpretation to establish if condition was updated over time."]
                        })

        return conflicts

    @classmethod
    def _detect_duplicate_blocks(cls, paragraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detects exact or near-duplicate paragraphs within the document.
        """
        duplicates = []
        n = len(paragraphs)
        if n < 2:
            return duplicates

        for i in range(n):
            p1 = paragraphs[i]
            t1 = p1.get("text", "").strip()
            if len(t1.split()) < 8:  # Skip short headers
                continue
            
            for j in range(i + 1, n):
                p2 = paragraphs[j]
                t2 = p2.get("text", "").strip()
                if len(t2.split()) < 8:
                    continue

                if t1.lower() == t2.lower():
                    duplicates.append({
                        "item_id": str(uuid.uuid4())[:8],
                        "classification": "duplicate_information",
                        "description": f"Exact duplicate paragraph block detected (paragraphs {p1['index']} and {p2['index']}).",
                        "supporting_passages": [
                            {"paragraph_index": p1["index"], "text": t1, "start_char": p1["start_char"]},
                            {"paragraph_index": p2["index"], "text": t2, "start_char": p2["start_char"]}
                        ],
                        "explanation": "Identical paragraph content repeated verbatim across separate sections.",
                        "limitations": ["May indicate copy-pasting, boilerplate text, or redundant document assembly."]
                    })
                else:
                    # Near-duplicate check via similarity ratio
                    sim = difflib.SequenceMatcher(None, t1.lower(), t2.lower()).ratio()
                    if sim > 0.88:
                        duplicates.append({
                            "item_id": str(uuid.uuid4())[:8],
                            "classification": "duplicate_information",
                            "description": f"Near-duplicate paragraph block detected (similarity: {round(sim, 2)}) between paragraphs {p1['index']} and {p2['index']}.",
                            "supporting_passages": [
                                {"paragraph_index": p1["index"], "text": t1, "start_char": p1["start_char"]},
                                {"paragraph_index": p2["index"], "text": t2, "start_char": p2["start_char"]}
                            ],
                            "explanation": "Near-identical paragraph content repeated with minimal variation.",
                            "limitations": ["Near-duplicates may reflect draft revisions or unpruned boilerplate."]
                        })

        return duplicates

    @classmethod
    def _detect_metadata_mismatches(
        cls,
        text: str,
        embedded_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detects discrepancies between embedded container metadata (PDF/DOCX) and document text.
        """
        mismatches = []
        if not embedded_metadata:
            return mismatches

        # 1. Author mismatch
        meta_author = None
        if "pdf_metadata" in embedded_metadata and embedded_metadata["pdf_metadata"].get("author"):
            meta_author = embedded_metadata["pdf_metadata"]["author"]
        elif "docx_metadata" in embedded_metadata and embedded_metadata["docx_metadata"].get("author"):
            meta_author = embedded_metadata["docx_metadata"]["author"]

        if meta_author and len(meta_author.strip()) > 3:
            # Check if body asserts a different author
            body_author_match = re.search(r'\b(?:authored\s+by|written\s+by|prepared\s+by)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b', text, re.IGNORECASE)
            if body_author_match:
                body_author = body_author_match.group(1).strip()
                if body_author.lower() != meta_author.lower():
                    mismatches.append({
                        "item_id": str(uuid.uuid4())[:8],
                        "classification": "confirmed_textual_mismatch",
                        "description": f"Discrepancy between embedded file author ('{meta_author}') and body text author statement ('{body_author}').",
                        "supporting_passages": [
                            {"text": f"Embedded File Metadata Author: {meta_author}", "source": "file_container_metadata"},
                            {"text": body_author_match.group(0), "source": "body_text"}
                        ],
                        "explanation": "The author specified in container properties contradicts the author asserted in body text.",
                        "limitations": ["Container metadata may reflect template creator or software operator."]
                    })

        return mismatches

