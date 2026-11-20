from __future__ import annotations
import re
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import dateutil.parser


class TemporalLocationAnalyzer:
    """
    Extracts, normalizes (ISO-8601), and tracks temporal and spatial markers.
    Identifies chronological reversals, contradictory timelines, and spatial conflicts
    based strictly on contextual evidence.
    """

    @classmethod
    def analyze_temporal(
        cls,
        text: str,
        sentences: List[Dict[str, Any]],
        entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not text:
            return {
                "status": "completed",
                "temporal_expressions": [],
                "chronological_order": [],
                "potential_chronology_conflicts": [],
                "errors": [],
                "limitations": ["Document text is empty; no temporal expressions found."]
            }

        temporal_expressions = []
        parsed_dates = []

        date_entities = [e for e in entities if e.get("entity_type") in {"DATE", "TIME"}]

        for de in date_entities:
            raw_val = de["value"]
            norm_iso = None
            try:
                # Try parsing with dateutil and normalize timezone
                parsed_dt = dateutil.parser.parse(raw_val, fuzzy=True, default=datetime(2000, 1, 1))
                if parsed_dt.tzinfo is not None:
                    parsed_dt = parsed_dt.astimezone(timezone.utc).replace(tzinfo=None)
                norm_iso = parsed_dt.isoformat()
            except Exception:
                norm_iso = None

            for m in de.get("mentions", []):
                temporal_expressions.append({
                    "raw_text": raw_val,
                    "normalized_iso": norm_iso,
                    "entity_type": de["entity_type"],
                    "start_char": m["start_char"],
                    "end_char": m["end_char"],
                    "paragraph_index": m["paragraph_index"],
                    "context": m["context"]
                })

                if norm_iso:
                    parsed_dates.append({
                        "raw_text": raw_val,
                        "iso": norm_iso,
                        "parsed_dt": parsed_dt,
                        "start_char": m["start_char"],
                        "paragraph_index": m["paragraph_index"],
                        "context": m["context"]
                    })

        # Sort temporal expressions by appearance
        temporal_expressions.sort(key=lambda x: x["start_char"])
        parsed_dates.sort(key=lambda x: x["start_char"])

        # Detect chronology conflicts and narrative reversals
        conflicts = []

        # 1. Narrative sequencing check (e.g. "later", "subsequently", "the following day" followed by an earlier date)
        seq_reversal_pattern = re.compile(
            r'\b(?:subsequently|later|afterwards|the\s+following\s+day|thereafter|following\s+this|next)\b',
            re.IGNORECASE
        )

        for i in range(len(parsed_dates) - 1):
            d1 = parsed_dates[i]
            d2 = parsed_dates[i + 1]

            # Text between d1 and d2
            span_text = text[d1["start_char"]:d2["start_char"]]
            if seq_reversal_pattern.search(span_text):
                if d2["parsed_dt"] < d1["parsed_dt"]:
                    conflicts.append({
                        "conflict_type": "chronological_reversal",
                        "description": f"Narrative indicates subsequent event ('later'/'subsequently') but timestamp '{d2['raw_text']}' predates earlier timestamp '{d1['raw_text']}'.",
                        "first_reference": {
                            "date": d1["raw_text"],
                            "iso": d1["iso"],
                            "context": d1["context"]
                        },
                        "second_reference": {
                            "date": d2["raw_text"],
                            "iso": d2["iso"],
                            "context": d2["context"]
                        },
                        "explanation": "Chronological progression in narrative conflicts with explicit timestamps."
                    })

        # 2. Same-event / incompatible date contradiction check
        # Look for identical action predicates with different dates
        event_date_map: Dict[str, List[Dict[str, Any]]] = {}
        action_match = re.compile(r'\b(?:occurred|happened|signed|initiated|conducted|established|born|graduated|concluded|arrested)\b', re.IGNORECASE)
        
        for s in sentences:
            s_text = s["text"]
            if action_match.search(s_text):
                # Check which parsed dates fall in this sentence
                s_dates = [d for d in parsed_dates if s["start_char"] <= d["start_char"] <= s["end_char"]]
                if s_dates:
                    # Clean the action statement of the specific date to compare
                    base_stmt = s_text
                    for d in s_dates:
                        base_stmt = base_stmt.replace(d["raw_text"], "DATE_REF")
                    key = re.sub(r'\s+', ' ', base_stmt.strip().lower())
                    if key not in event_date_map:
                        event_date_map[key] = []
                    event_date_map[key].append({
                        "sentence": s_text,
                        "dates": s_dates
                    })

        for key, occurrences in event_date_map.items():
            if len(occurrences) > 1:
                all_raw = set()
                all_iso = set()
                for occ in occurrences:
                    for d in occ["dates"]:
                        all_raw.add(d["raw_text"])
                        if d["iso"]:
                            all_iso.add(d["iso"][:10])  # compare YYYY-MM-DD
                if len(all_iso) > 1:
                    conflicts.append({
                        "conflict_type": "conflicting_dates_for_same_event",
                        "description": f"Same event description is asserted with contradictory dates: {sorted(list(all_raw))}",
                        "first_reference": {
                            "date": occurrences[0]["dates"][0]["raw_text"],
                            "context": occurrences[0]["sentence"]
                        },
                        "second_reference": {
                            "date": occurrences[1]["dates"][0]["raw_text"],
                            "context": occurrences[1]["sentence"]
                        },
                        "explanation": "Multiple contradictory dates asserted for the same described occurrence."
                    })

        chronological_order = [
            {"raw_text": d["raw_text"], "iso": d["iso"], "start_char": d["start_char"]}
            for d in sorted(parsed_dates, key=lambda x: x["parsed_dt"])
        ]

        return {
            "status": "completed",
            "temporal_expressions": temporal_expressions,
            "chronological_order": chronological_order,
            "potential_chronology_conflicts": conflicts,
            "errors": [],
            "limitations": [
                "Relative temporal expressions without reference epoch are extracted without absolute normalization.",
                "Chronological conflicts are flagged only where explicit textual contradiction or reversal markers are present."
            ]
        }

    @classmethod
    def analyze_location(
        cls,
        text: str,
        sentences: List[Dict[str, Any]],
        entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not text:
            return {
                "status": "completed",
                "location_mentions": [],
                "distinct_locations": [],
                "potential_location_conflicts": [],
                "errors": [],
                "limitations": ["Document text is empty; no locations found."]
            }

        location_mentions = []
        loc_entities = [e for e in entities if e.get("entity_type") == "LOCATION"]
        distinct_locations = list({e["value"] for e in loc_entities})

        for le in loc_entities:
            for m in le.get("mentions", []):
                location_mentions.append({
                    "location": le["value"],
                    "start_char": m["start_char"],
                    "end_char": m["end_char"],
                    "paragraph_index": m["paragraph_index"],
                    "context": m["context"]
                })

        location_mentions.sort(key=lambda x: x["start_char"])

        # Detect spatial conflicts:
        # e.g., Same subject or event asserted in multiple contradictory locations simultaneously
        conflicts = []
        simultaneous_pat = re.compile(
            r'\b(?:at\s+the\s+same\s+time|simultaneously|concurrently|meanwhile)\b',
            re.IGNORECASE
        )

        for s in sentences:
            s_text = s["text"]
            # Find locations in this single sentence
            s_locs = [m for m in location_mentions if s["start_char"] <= m["start_char"] <= s["end_char"]]
            distinct_s_locs = list({m["location"].lower() for m in s_locs})
            if len(distinct_s_locs) >= 2:
                # Check if sentence asserts a subject was simultaneously in both locations
                if simultaneous_pat.search(s_text) or re.search(r'\b(?:was\s+in|located\s+in|present\s+in)\s+[A-Z][a-z]+.*and\s+(?:in|at)\s+[A-Z][a-z]+', s_text):
                    conflicts.append({
                        "conflict_type": "simultaneous_location_contradiction",
                        "description": f"Sentence asserts presence in multiple contradictory locations simultaneously: {[m['location'] for m in s_locs]}",
                        "supporting_sentence": s_text,
                        "locations": [m["location"] for m in s_locs],
                        "explanation": "Subject or event is reported as occurring concurrently in mutually exclusive physical locations."
                    })

        # Also check for contradictory headquarters claims
        hq_pattern = re.compile(r'\b(?:headquarters|headquartered)\s+(?:are\s+|were\s+|is\s+|was\s+)?(?:located\s+in\s+|in\s+|at\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', re.IGNORECASE)
        hq_claims = []
        for s in sentences:
            m = hq_pattern.search(s["text"])
            if m:
                hq_claims.append({"loc": m.group(1).strip(), "sentence": s["text"]})
        
        if len(hq_claims) > 1:
            locs = {c["loc"].lower(): c for c in hq_claims}
            if len(locs) > 1:
                items = list(locs.values())
                conflicts.append({
                    "conflict_type": "conflicting_headquarters_location",
                    "description": f"Document asserts conflicting primary locations/headquarters: {list(locs.keys())}",
                    "first_reference": items[0]["sentence"],
                    "second_reference": items[1]["sentence"],
                    "explanation": "Multiple contradictory locations asserted for the same organizational attribute."
                })

        return {
            "status": "completed",
            "location_mentions": location_mentions,
            "distinct_locations": sorted(distinct_locations),
            "potential_location_conflicts": conflicts,
            "errors": [],
            "limitations": [
                "Location extraction tracks spatial named entities and geopolitical references.",
                "Distinct location references across separate events are expected in normal text and are not flagged as conflicts without direct contextual overlap."
            ]
        }

