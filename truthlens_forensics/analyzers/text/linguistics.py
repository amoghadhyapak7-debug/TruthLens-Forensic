from __future__ import annotations
import re
import math
import statistics
from typing import Dict, Any, List, Tuple
from collections import Counter


class LinguisticAnalyzer:
    """
    Computes deterministic, reproducible linguistic, statistical, readability,
    repetition, and cross-section shift metrics.
    Never treats statistical anomalies as standalone proof of deception.
    """

    @classmethod
    def count_syllables(cls, word: str) -> int:
        word = word.lower().strip()
        if not word:
            return 0
        if len(word) <= 3:
            return 1
        # Remove trailing e, es, ed
        word = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', word)
        vowel_runs = re.findall(r'[aeiouy]{1,2}', word)
        return max(1, len(vowel_runs))

    @classmethod
    def analyze_linguistics(
        cls,
        text: str,
        paragraphs: List[Dict[str, Any]],
        sentences: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not text or not text.strip():
            return {
                "status": "completed",
                "sentence_length_distribution": {},
                "word_length_distribution": {},
                "vocabulary_diversity": {},
                "repetition_metrics": {},
                "punctuation_patterns": {},
                "capitalization_patterns": {},
                "readability_scores": {},
                "paragraph_metrics": {},
                "section_shifts": [],
                "errors": [],
                "limitations": ["Document text is empty; linguistic metrics skipped."]
            }

        # Words
        words = re.findall(r'\b[a-zA-Z0-9\'-]+\b', text)
        words_alpha = [w.lower() for w in words if re.search(r'[a-zA-Z]', w)]
        total_words = len(words_alpha)

        # 1. Sentence length distribution (words per sentence)
        s_lengths = [len(re.findall(r'\b[a-zA-Z0-9\'-]+\b', s["text"])) for s in sentences]
        if not s_lengths:
            s_lengths = [total_words] if total_words else [0]

        sent_dist = {
            "min": float(min(s_lengths)),
            "max": float(max(s_lengths)),
            "mean": round(statistics.mean(s_lengths), 2),
            "median": round(statistics.median(s_lengths), 2),
            "std_dev": round(statistics.stdev(s_lengths), 2) if len(s_lengths) > 1 else 0.0
        }

        # 2. Word length distribution (characters per word)
        w_lengths = [len(w) for w in words_alpha]
        if w_lengths:
            word_dist = {
                "min": float(min(w_lengths)),
                "max": float(max(w_lengths)),
                "mean": round(statistics.mean(w_lengths), 2),
                "std_dev": round(statistics.stdev(w_lengths), 2) if len(w_lengths) > 1 else 0.0
            }
        else:
            word_dist = {"min": 0.0, "max": 0.0, "mean": 0.0, "std_dev": 0.0}

        # 3. Vocabulary diversity: TTR and Hapax Legomena Ratio
        if words_alpha:
            counts = Counter(words_alpha)
            unique_words = len(counts)
            hapax_count = sum(1 for count in counts.values() if count == 1)
            ttr = round(unique_words / total_words, 4)
            hapax_ratio = round(hapax_count / total_words, 4)
        else:
            ttr = 0.0
            hapax_ratio = 0.0

        vocab_diversity = {
            "type_token_ratio": ttr,
            "hapax_legomena_ratio": hapax_ratio,
            "total_tokens": total_words,
            "unique_types": len(counts) if words_alpha else 0
        }

        # 4. Repeated n-grams / phrases (2-gram, 3-gram, 4-gram)
        def _get_ngrams(n: int) -> List[Tuple[str, int]]:
            ngrams = [" ".join(words_alpha[i:i+n]) for i in range(len(words_alpha) - n + 1)]
            n_counts = Counter(ngrams)
            return [(k, v) for k, v in n_counts.most_common(10) if v > 1]

        repeated_ngrams = {
            "bigrams": [{"phrase": p, "count": c} for p, c in _get_ngrams(2)[:5]],
            "trigrams": [{"phrase": p, "count": c} for p, c in _get_ngrams(3)[:5]],
            "fourgrams": [{"phrase": p, "count": c} for p, c in _get_ngrams(4)[:5]]
        }

        # Repeated sentence starters
        starters = []
        for s in sentences:
            s_words = s["text"].strip().split()
            if len(s_words) >= 2:
                starters.append(" ".join(s_words[:2]).lower())
        starter_counts = Counter(starters)
        repeated_starters = [{"starter": k, "count": v} for k, v in starter_counts.most_common(5) if v > 2]

        repetition_metrics = {
            "repeated_phrases": repeated_ngrams,
            "repeated_sentence_starters": repeated_starters
        }

        # 5. Punctuation patterns
        punct_counts = Counter(re.findall(r'[.,!?;:\'\"–—()\[\]{}]', text))
        exclamation_runs = len(re.findall(r'!{2,}', text))
        ellipsis_runs = len(re.findall(r'\.{3,}|…', text))
        mixed_quotes = bool(re.search(r'["\']', text) and re.search(r'[“”‘’]', text))

        punctuation_patterns = {
            "counts": dict(punct_counts),
            "multiple_exclamations": exclamation_runs,
            "multiple_ellipses": ellipsis_runs,
            "mixed_quote_styles": mixed_quotes,
            "unusual_punctuation_flag": exclamation_runs > 0 or ellipsis_runs > 2
        }

        # 6. Capitalization patterns
        char_alpha = [c for c in text if c.isalpha()]
        total_alpha = len(char_alpha)
        upper_alpha = sum(1 for c in char_alpha if c.isupper())
        all_caps_words = [w for w in words if len(w) > 1 and w.isupper() and w.isalpha()]
        all_caps_ratio = round(upper_alpha / total_alpha, 4) if total_alpha else 0.0

        capitalization_patterns = {
            "uppercase_char_ratio": all_caps_ratio,
            "all_caps_word_count": len(all_caps_words),
            "sample_all_caps": all_caps_words[:10],
            "excessive_caps_flag": all_caps_ratio > 0.25 or len(all_caps_words) > 10
        }

        # 7. Readability scores: Flesch Reading Ease & Flesch-Kincaid Grade Level
        num_sentences = max(1, len(sentences))
        total_syllables = sum(cls.count_syllables(w) for w in words_alpha)
        
        if total_words > 0:
            asl = total_words / num_sentences  # Average Sentence Length
            asw = total_syllables / total_words  # Average Syllables per Word
            flesch_reading_ease = round(206.835 - (1.015 * asl) - (84.6 * asw), 2)
            flesch_kincaid_grade = round((0.39 * asl) + (11.8 * asw) - 15.59, 2)
        else:
            flesch_reading_ease = 100.0
            flesch_kincaid_grade = 0.0

        readability_scores = {
            "flesch_reading_ease": max(0.0, min(100.0, flesch_reading_ease)),
            "flesch_kincaid_grade": max(0.0, flesch_kincaid_grade),
            "total_syllables": total_syllables,
            "avg_words_per_sentence": round(total_words / num_sentences, 2)
        }

        # 8. Paragraph metrics & cross-section shifts
        p_word_counts = [p.get("word_count", 0) for p in paragraphs]
        p_mean = statistics.mean(p_word_counts) if p_word_counts else 0.0
        p_variance = statistics.variance(p_word_counts) if len(p_word_counts) > 1 else 0.0

        paragraph_metrics = {
            "paragraph_count": len(paragraphs),
            "mean_paragraph_word_count": round(p_mean, 2),
            "paragraph_length_variance": round(p_variance, 2)
        }

        # Cross-section shifts: compare consecutive paragraphs for abrupt shifts
        section_shifts = []
        for i in range(len(paragraphs) - 1):
            p1 = paragraphs[i]
            p2 = paragraphs[i + 1]
            w1 = p1.get("word_count", 0)
            w2 = p2.get("word_count", 0)
            
            # Substantial change in length or style
            if w1 > 10 and w2 > 10:
                length_ratio = max(w1, w2) / max(1, min(w1, w2))
                if length_ratio > 4.0:
                    section_shifts.append({
                        "from_paragraph": p1["index"],
                        "to_paragraph": p2["index"],
                        "shift_type": "abrupt_length_divergence",
                        "description": f"Abrupt structural length shift: paragraph {p1['index']} ({w1} words) vs paragraph {p2['index']} ({w2} words).",
                        "ratio": round(length_ratio, 2)
                    })

        return {
            "status": "completed",
            "sentence_length_distribution": sent_dist,
            "word_length_distribution": word_dist,
            "vocabulary_diversity": vocab_diversity,
            "repetition_metrics": repetition_metrics,
            "punctuation_patterns": punctuation_patterns,
            "capitalization_patterns": capitalization_patterns,
            "readability_scores": readability_scores,
            "paragraph_metrics": paragraph_metrics,
            "section_shifts": section_shifts,
            "errors": [],
            "limitations": [
                "Linguistic metrics are statistical descriptors calculated over the full text.",
                "Statistical variations, high repetition, or grade-level changes are contextual markers and never treated as standalone proof of deception."
            ]
        }

