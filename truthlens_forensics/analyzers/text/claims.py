from __future__ import annotations
import re
import uuid
from typing import Dict, Any, List, Optional
from collections import defaultdict


class ClaimExtractor:
    """
    Extracts and classifies factual, attributional, and epistemic statements
    from segmented sentences with exact character spans and linked entities.
    Strictly reports text claims without asserting external factual truth.
    """

    OPINION_MARKERS = [
        r'\b(?:i\s+think|we\s+believe|in\s+my\s+opinion|in\s+our\s+view|it\s+seems\s+that|personally|i\s+feel|arguably|remarkably|unbelievable)\b'
    ]
    SPECULATION_MARKERS = [
        r'\b(?:might\s+have|could\s+have|possibly|perhaps|hypothetically|it\s+is\s+possible\s+that|presumed|assuming\s+that|maybe)\b'
    ]
    UNCERTAIN_MARKERS = [
        r'\b(?:it\s+is\s+unclear|unconfirmed|allegedly|reportedly|purportedly|not\s+yet\s+verified|unknown\s+whether|remains\s+to\s+be\s+seen)\b'
    ]
    CAUSAL_MARKERS = [
        r'\b(?:because|therefore|as\s+a\s+result|consequently|led\s+to|caused\s+by|due\s+to|resulting\s+in|stemmed\s+from)\b'
    ]
    ATTRIBUTION_MARKERS = [
        r'\b(?:according\s+to|stated\s+that|testified\s+that|claimed\s+that|reported\s+that|alleged\s+that|noted\s+that|affirmed\s+that)\b'
    ]
    IDENTITY_MARKERS = [
        r'\b(?:is\s+the|was\s+the|served\s+as|identified\s+as|known\s+as|appointed\s+as|employed\s+as|role\s+of)\b'
    ]

    @classmethod
    def extract_claims(
        cls,
        sentences: List[Dict[str, Any]],
        entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not sentences:
            return {
                "status": "completed",
                "claims_count": 0,
                "claim_types_distribution": {},
                "claims": [],
                "errors": [],
                "limitations": ["No sentences available for claim extraction."]
            }

        claims_list = []
        counts: Dict[str, int] = defaultdict(int)

        for s in sentences:
            text = s["text"].strip()
            if len(text) < 5:
                continue

            # Identify linked entities in this sentence's span
            s_start = s["start_char"]
            s_end = s["end_char"]
            
            linked_entities: List[str] = []
            linked_dates: List[str] = []
            linked_numbers: List[str] = []
            linked_locations: List[str] = []

            for ent in entities:
                for mention in ent.get("mentions", []):
                    m_start = mention["start_char"]
                    m_end = mention["end_char"]
                    if (s_start <= m_start < s_end) or (s_start < m_end <= s_end):
                        val = ent["value"]
                        if val not in linked_entities:
                            linked_entities.append(val)
                        if ent["entity_type"] == "DATE" and val not in linked_dates:
                            linked_dates.append(val)
                        elif ent["entity_type"] in {"MONETARY_VALUE", "QUANTITY"} and val not in linked_numbers:
                            linked_numbers.append(val)
                        elif ent["entity_type"] == "LOCATION" and val not in linked_locations:
                            linked_locations.append(val)

            category, secondary_tags = cls._classify_sentence(text, linked_dates, linked_numbers, linked_locations)
            counts[category] += 1

            claim_obj = {
                "claim_id": str(uuid.uuid4())[:8],
                "statement": text,
                "category": category,
                "secondary_tags": secondary_tags,
                "entities": linked_entities,
                "dates": linked_dates,
                "numbers": linked_numbers,
                "locations": linked_locations,
                "source_span": {
                    "start_char": s_start,
                    "end_char": s_end,
                    "paragraph_index": s["paragraph_index"],
                    "sentence_index": s["index"],
                    "context": text
                },
                "extraction_status": "completed"
            }
            claims_list.append(claim_obj)

        return {
            "status": "completed",
            "claims_count": len(claims_list),
            "claim_types_distribution": dict(counts),
            "claims": claims_list,
            "errors": [],
            "limitations": [
                "Claims represent statements asserted inside the text itself and are extracted for forensic tracking.",
                "Categorization indicates the semantic nature of the claim, not an external factual verification."
            ]
        }

    @classmethod
    def _classify_sentence(
        cls,
        text: str,
        dates: List[str],
        numbers: List[str],
        locations: List[str]
    ) -> (str, List[str]):
        secondary: List[str] = []
        lower = text.lower()

        # 1. Question
        if text.endswith("?") or re.match(r'^(?:who|what|when|where|why|how|is|are|did|does|do|can|could|would|will)\b', lower):
            return "question", secondary

        # 2. Speculation
        if any(re.search(pat, lower) for pat in cls.SPECULATION_MARKERS):
            secondary.append("speculative")
            return "speculation", secondary

        # 3. Uncertain statement
        if any(re.search(pat, lower) for pat in cls.UNCERTAIN_MARKERS):
            secondary.append("unverified")
            return "uncertain_statement", secondary

        # 4. Opinion
        if any(re.search(pat, lower) for pat in cls.OPINION_MARKERS):
            return "opinion", secondary

        # 5. Causal claim
        if any(re.search(pat, lower) for pat in cls.CAUSAL_MARKERS):
            secondary.append("causal_assertion")
            if dates:
                secondary.append("temporal_context")
            return "causal_claim", secondary

        # 6. Attribution claim
        if any(re.search(pat, lower) for pat in cls.ATTRIBUTION_MARKERS):
            secondary.append("reported_speech")
            return "attribution_claim", secondary

        # 7. Identity claim
        if any(re.search(pat, lower) for pat in cls.IDENTITY_MARKERS):
            return "identity_claim", secondary

        # 8. Numerical claim
        if numbers or re.search(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text):
            secondary.append("quantitative")
            if dates:
                secondary.append("time_stamped")
            return "numerical_claim", secondary

        # 9. Temporal claim
        if dates or re.search(r'\b(?:yesterday|tomorrow|today|morning|afternoon|evening|night|on\s+[A-Z][a-z]+|in\s+\d{4})\b', lower):
            secondary.append("chronological")
            return "temporal_claim", secondary

        # 10. Location claim
        if locations or re.search(r'\b(?:at|in|near|inside|outside|located\s+at)\s+[A-Z][a-z]+\b', text):
            secondary.append("spatial")
            return "location_claim", secondary

        # 11. Action claim
        action_verbs = r'\b(?:transferred|deleted|modified|accessed|signed|arrested|delivered|invaded|launched|installed|executed|withdrew|deposited|authorized)\b'
        if re.search(action_verbs, lower):
            secondary.append("action_oriented")
            return "action_claim", secondary

        # Default: Event claim
        return "event_claim", secondary

