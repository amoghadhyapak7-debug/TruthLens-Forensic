from __future__ import annotations
import re
import difflib
from typing import Dict, Any, List, Optional
from pathlib import Path


class TextComparisonAnalyzer:
    """
    Computes intra-document paragraph/block similarity (detecting copy-paste and boilerplate)
    and optional cross-document comparison if reference text or comparison file path is supplied.
    """

    @classmethod
    def jaccard_similarity(cls, s1: str, s2: str) -> float:
        w1 = set(re.findall(r'\b[a-zA-Z0-9\'-]+\b', s1.lower()))
        w2 = set(re.findall(r'\b[a-zA-Z0-9\'-]+\b', s2.lower()))
        if not w1 or not w2:
            return 0.0
        intersection = w1.intersection(w2)
        union = w1.union(w2)
        return round(len(intersection) / len(union), 4)

    @classmethod
    def compare_blocks(
        cls,
        paragraphs: List[Dict[str, Any]],
        ingestion_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        meta = ingestion_metadata or {}
        n = len(paragraphs)
        
        pairwise_scores: List[float] = []
        duplicate_blocks: List[Dict[str, Any]] = []

        for i in range(n):
            p1 = paragraphs[i]
            t1 = p1.get("text", "").strip()
            if len(t1.split()) < 6:
                continue

            for j in range(i + 1, n):
                p2 = paragraphs[j]
                t2 = p2.get("text", "").strip()
                if len(t2.split()) < 6:
                    continue

                jaccard = cls.jaccard_similarity(t1, t2)
                seq_sim = round(difflib.SequenceMatcher(None, t1.lower(), t2.lower()).ratio(), 4)
                max_sim = max(jaccard, seq_sim)
                pairwise_scores.append(max_sim)

                if max_sim >= 0.75:
                    duplicate_blocks.append({
                        "first_paragraph": {
                            "index": p1["index"],
                            "text": t1[:160] + "..." if len(t1) > 160 else t1,
                            "start_char": p1["start_char"],
                            "end_char": p1["end_char"]
                        },
                        "second_paragraph": {
                            "index": p2["index"],
                            "text": t2[:160] + "..." if len(t2) > 160 else t2,
                            "start_char": p2["start_char"],
                            "end_char": p2["end_char"]
                        },
                        "similarity_score": max_sim,
                        "similarity_type": "exact_match" if max_sim >= 0.99 else "near_duplicate"
                    })

        mean_sim = round(sum(pairwise_scores) / len(pairwise_scores), 4) if pairwise_scores else 0.0
        max_sim = max(pairwise_scores) if pairwise_scores else 0.0

        intra_sim = {
            "comparisons_evaluated": len(pairwise_scores),
            "max_pairwise_similarity": max_sim,
            "mean_pairwise_similarity": mean_sim,
            "high_similarity_pairs_count": len(duplicate_blocks)
        }

        # Optional reference comparison
        ref_comparison = None
        ref_text = meta.get("reference_text")
        ref_path_str = meta.get("comparison_file_path")

        if not ref_text and ref_path_str:
            ref_p = Path(ref_path_str)
            if ref_p.exists() and ref_p.is_file():
                try:
                    with open(ref_p, "r", encoding="utf-8", errors="replace") as f:
                        ref_text = f.read()
                except Exception:
                    pass

        if ref_text:
            # Combine current text paragraphs
            full_text = " ".join(p["text"] for p in paragraphs)
            jacc = cls.jaccard_similarity(full_text, ref_text)
            diff_ratio = round(difflib.SequenceMatcher(None, full_text, ref_text).ratio(), 4)
            ref_comparison = {
                "reference_source": ref_path_str or "ingestion_metadata.reference_text",
                "jaccard_similarity": jacc,
                "sequence_matcher_ratio": diff_ratio,
                "status": "comparison_executed"
            }

        limitations = [
            "Intra-document block similarity uses normalized Jaccard word set and character sequence matching.",
        ]
        if not ref_comparison:
            limitations.append("No external reference text provided; single-document comparison performed.")

        return {
            "status": "completed",
            "intra_document_similarity": intra_sim,
            "duplicate_blocks": duplicate_blocks,
            "reference_comparison": ref_comparison,
            "errors": [],
            "limitations": limitations
        }

