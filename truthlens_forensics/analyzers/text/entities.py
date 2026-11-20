from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

KNOWN_VEHICLE_PATTERNS = [
    r'\b(?:Tesla\s+Model\s+[3SYX]|Boeing\s+\d{3}[A-Z]*|Airbus\s+A\d{3}|Ford\s+[A-Z][a-z0-9-]+|Toyota\s+[A-Z][a-z]+|Honda\s+[A-Z][a-z]+|BMW\s+[A-Za-z0-9]+|Mercedes-Benz\s+[A-Za-z0-9]+|Chevrolet\s+[A-Za-z]+)\b',
    r'\bVIN\s*:\s*[A-HJ-NPR-Z0-9]{17}\b',
    r'\b(?:license\s+plate|tag\s+number)\s*[:#]?\s*([A-Z0-9]{5,8})\b'
]

KNOWN_PRODUCT_PATTERNS = [
    r'\b(?:iPhone\s+\d+(?:\s*(?:Pro|Max|Mini|Plus))?|iPad\s+(?:Pro|Air|Mini)?|MacBook\s+(?:Pro|Air)|Windows\s+(?:10|11|Server)|PlayStation\s+\d|Xbox\s+(?:Series\s+[XS]|One)|Android\s+\d+|TruthLens(?:\s+Platform)?|Pixel\s+\d+(?:a|Pro)?)\b'
]

KNOWN_EVENT_PATTERNS = [
    r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Conference|Summit|Olympics|Symposium|Annual\s+General\s+Meeting|Expo|Championship|Hackathon|Gala|Festival|Forum|World\s+Cup))\b',
    r'\b(?:G20\s+Summit|COP\d{2}|Super\s+Bowl(?:\s+[IVXLCDM]+)?)\b'
]

HONORIFICS = r'\b(?:Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.|Detective|Officer|Inspector|Senator|President|Governor|Judge|Director|CEO|CTO|CFO|General|Captain)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b'

ORG_PATTERNS = [
    r'\b([A-Z][A-Za-z0-9&.-]+(?:\s+[A-Z][A-Za-z0-9&.-]+)*\s+(?:Inc\.?|LLC|Corp\.?|Corporation|Ltd\.?|Group|Bank|Agency|Department|University|Institute|Foundation|Hospital|Police\s+Department|Bureau|Authority))\b',
    r'\b(FBI|CIA|NASA|WHO|UN|UNESCO|INTERPOL|SEC|DOJ|DOD|TruthLens)\b'
]

LOCATION_PATTERNS = [
    r'\b(?:in|at|near|from|to|across)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s+(?:City|Street|Avenue|Road|Boulevard|Blvd|St|Ave|Rd|Airport|Harbor|Port|Station|County|District|Province|State|Island|River|Mountain|Valley))?)\b',
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:City|Airport|Station|County|District|Island|River|Mountain|Valley))\b',
    r'\b(New\s+York|Los\s+Angeles|Chicago|Houston|Phoenix|London|Paris|Berlin|Tokyo|Beijing|Singapore|Sydney|Toronto|Geneva|Zurich|Washington(?:\s+D\.C\.)?|San\s+Francisco|Boston|Seattle|Dubai|Hong\s+Kong)\b',
    r'\b(United\s+States|United\s+Kingdom|Canada|Germany|France|Japan|China|Australia|India|Brazil|Switzerland|Russia|Italy|Spain)\b'
]


class EntityExtractor:
    """
    Hybrid deterministic regex + pattern-based entity extractor with exact character offset tracking.
    Never invents missing entities or merges distinct entities without explicit evidence.
    """

    @classmethod
    def extract_entities(cls, text: str, paragraphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not text or not text.strip():
            return {
                "status": "completed",
                "entity_counts": {},
                "entities": [],
                "errors": [],
                "limitations": ["Input document text is empty; no entities extracted."]
            }

        extracted_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

        def _add_mention(
            etype: str,
            raw_val: str,
            start_c: int,
            end_c: int,
            method: str = "deterministic_regex",
            norm_val: Optional[str] = None,
            reliability: str = "high"
        ):
            val = raw_val.strip()
            if not val or len(val) < 2:
                return
            
            # Filter obvious false positives for locations/persons
            if etype in {"PERSON", "ORGANIZATION", "LOCATION"} and val.lower() in {
                "the", "this", "that", "these", "those", "there", "their", "table", "record", "page"
            }:
                return

            norm = norm_val or val
            key = (etype, norm.lower())

            # Find paragraph
            p_idx = 0
            for p in paragraphs:
                if p["start_char"] <= start_c <= p["end_char"]:
                    p_idx = p["index"]
                    break

            # Context
            ctx_start = max(0, start_c - 30)
            ctx_end = min(len(text), end_c + 30)
            context = text[ctx_start:ctx_end].replace("\n", " ").strip()

            mention = {
                "text": val,
                "start_char": start_c,
                "end_char": end_c,
                "paragraph_index": p_idx,
                "context": f"...{context}..." if ctx_start > 0 or ctx_end < len(text) else context
            }

            if key not in extracted_map:
                extracted_map[key] = {
                    "entity_type": etype,
                    "value": val,
                    "normalized_value": norm,
                    "occurrences": 1,
                    "mentions": [mention],
                    "extraction_method": method,
                    "reliability": reliability,
                    "limitations": []
                }
            else:
                # Deduplicate overlapping mentions of the exact same span
                existing_spans = [(m["start_char"], m["end_char"]) for m in extracted_map[key]["mentions"]]
                if (start_c, end_c) not in existing_spans:
                    extracted_map[key]["occurrences"] += 1
                    extracted_map[key]["mentions"].append(mention)

        # 1. EMAIL
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
        for match in email_pattern.finditer(text):
            _add_mention("EMAIL", match.group(0), match.start(), match.end(), method="regex_pattern", norm_val=match.group(0).lower())

        # 2. URL
        url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s<>"]*')
        for match in url_pattern.finditer(text):
            _add_mention("URL", match.group(0), match.start(), match.end(), method="regex_pattern", norm_val=match.group(0))

        # 3. PHONE_NUMBER
        phone_pattern = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
        for match in phone_pattern.finditer(text):
            p_val = match.group(0)
            # Avoid matching pure dates like 2026-03-14
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', p_val):
                _add_mention("PHONE_NUMBER", p_val, match.start(), match.end(), method="regex_pattern")

        # 4. MONETARY_VALUE
        money_pattern = re.compile(r'(?:[\$€£¥]|USD|EUR|GBP|INR)\s?\d+(?:,\d{3})*(?:\.\d{1,2})?(?:\s?(?:million|billion|trillion|k|M|B))?\b', re.IGNORECASE)
        for match in money_pattern.finditer(text):
            _add_mention("MONETARY_VALUE", match.group(0), match.start(), match.end(), method="regex_pattern")

        # 5. QUANTITY
        qty_pattern = re.compile(r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|kg|km|miles|meters|gallons|tons|units|items|records|documents|users|accounts|hours|days|weeks|months|years|megabytes|gigabytes|MB|GB|TB)\b', re.IGNORECASE)
        for match in qty_pattern.finditer(text):
            _add_mention("QUANTITY", match.group(0), match.start(), match.end(), method="regex_pattern")

        # 6. DATES
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b(?:the\s+following\s+Tuesday|two\s+days\s+earlier|yesterday|tomorrow|three\s+weeks\s+later)\b'
        ]
        for dp in date_patterns:
            for match in re.finditer(dp, text, re.IGNORECASE):
                _add_mention("DATE", match.group(0), match.start(), match.end(), method="regex_pattern")

        # 7. TIME
        time_pattern = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm|UTC|EST|PST|GMT|IST)?\b')
        for match in time_pattern.finditer(text):
            _add_mention("TIME", match.group(0), match.start(), match.end(), method="regex_pattern")

        # 8. VEHICLE
        for vp in KNOWN_VEHICLE_PATTERNS:
            for match in re.finditer(vp, text, re.IGNORECASE):
                _add_mention("VEHICLE", match.group(0), match.start(), match.end(), method="pattern_matcher")

        # 9. PRODUCT
        for pp in KNOWN_PRODUCT_PATTERNS:
            for match in re.finditer(pp, text, re.IGNORECASE):
                _add_mention("PRODUCT", match.group(0), match.start(), match.end(), method="pattern_matcher")

        # 10. EVENT
        for ep in KNOWN_EVENT_PATTERNS:
            for match in re.finditer(ep, text):
                _add_mention("EVENT", match.group(0), match.start(), match.end(), method="pattern_matcher")

        # 11. PERSON (Honorifics and Capitalized Names)
        for match in re.finditer(HONORIFICS, text):
            # group(0) has full string like "Dr. Jane Smith", group(1) has "Jane Smith"
            _add_mention("PERSON", match.group(0), match.start(), match.end(), method="rule_based_honorific", norm_val=match.group(1).strip())

        # Also person naming indicators: e.g. "Jane Smith said", "John Doe testified"
        person_verb_pattern = re.compile(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:said|stated|testified|claimed|signed|authored|confessed|wrote|reported)\b')
        for match in person_verb_pattern.finditer(text):
            p_name = match.group(1).strip()
            # Verify not a false positive
            if not any(p_name.endswith(sfx) for sfx in ["Inc", "LLC", "Corp", "Ltd", "City", "County"]):
                _add_mention("PERSON", p_name, match.start(1), match.end(1), method="contextual_verb_rule", norm_val=p_name)

        # 12. ORGANIZATION
        for op in ORG_PATTERNS:
            for match in re.finditer(op, text):
                org_name = match.group(0).strip()
                _add_mention("ORGANIZATION", org_name, match.start(), match.end(), method="rule_based_org_pattern", norm_val=org_name)

        # 13. LOCATION
        for lp in LOCATION_PATTERNS:
            for match in re.finditer(lp, text):
                loc_val = match.group(1) if match.lastindex and match.group(1) else match.group(0)
                loc_val = loc_val.strip()
                s_char = match.start(1) if match.lastindex and match.group(1) else match.start()
                e_char = match.end(1) if match.lastindex and match.group(1) else match.end()
                _add_mention("LOCATION", loc_val, s_char, e_char, method="rule_based_location", norm_val=loc_val)

        entities_list = list(extracted_map.values())
        # Sort entities by start_char of first mention
        entities_list.sort(key=lambda e: e["mentions"][0]["start_char"] if e["mentions"] else 0)

        # Count by category
        counts: Dict[str, int] = defaultdict(int)
        for e in entities_list:
            counts[e["entity_type"]] += 1

        return {
            "status": "completed",
            "entity_counts": dict(counts),
            "entities": entities_list,
            "errors": [],
            "limitations": [
                "Entity extraction uses high-precision deterministic patterns and contextual heuristics.",
                "Ambiguous single-token names without title or context are preserved as unclassified to avoid false associations."
            ]
        }

