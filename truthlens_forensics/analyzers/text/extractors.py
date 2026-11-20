from __future__ import annotations
import os
import re
import csv
import io
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import charset_normalizer
import pypdf
import docx

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".pdf", ".docx"}

MIME_TYPE_MAP = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
}


class ExtractionError(Exception):
    """Raised when extraction from a document fails."""
    pass


class DocumentExtractor:
    """
    Standardized extractor service that validates and reads TXT, MD, CSV, JSON, PDF, and DOCX files.
    Never alters original files, extracts embedded metadata, and tracks paragraph/sentence character spans.
    """

    @staticmethod
    def calculate_sha256(file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def detect_encoding(raw_bytes: bytes) -> str:
        try:
            results = charset_normalizer.from_bytes(raw_bytes)
            best = results.best()
            if best and best.encoding:
                return best.encoding
        except Exception:
            pass
        return "utf-8"

    @classmethod
    def validate_file(cls, file_path_str: str, expected_sha256: Optional[str] = None) -> Tuple[Path, str, int, str]:
        path = Path(file_path_str)
        if not path.exists():
            raise FileNotFoundError(f"Evidence file does not exist: {file_path_str}")
        if not path.is_file():
            raise ValueError(f"Evidence path is not a file: {file_path_str}")
        
        file_size = path.stat().st_size
        if file_size == 0:
            raise ValueError(f"Evidence file is empty (0 bytes): {file_path_str}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats: {sorted(list(SUPPORTED_EXTENSIONS))}")

        calculated_hash = cls.calculate_sha256(path)
        if expected_sha256 and expected_sha256.strip():
            if calculated_hash.lower() != expected_sha256.strip().lower():
                raise ValueError(
                    f"SHA-256 mismatch: expected {expected_sha256}, calculated {calculated_hash}"
                )

        return path, ext, file_size, calculated_hash

    @classmethod
    def extract_document(
        cls,
        file_path_str: str,
        expected_sha256: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validates the file, extracts raw text, embedded metadata, character encoding,
        and builds paragraph and sentence maps with exact character offsets.
        """
        path, ext, file_size, calculated_hash = cls.validate_file(file_path_str, expected_sha256)
        
        raw_text = ""
        encoding = "utf-8"
        metadata: Dict[str, Any] = {}
        extraction_method = "utf8_reader"

        try:
            if ext in {".txt", ".md"}:
                with open(path, "rb") as f:
                    content_bytes = f.read()
                encoding = cls.detect_encoding(content_bytes)
                try:
                    raw_text = content_bytes.decode(encoding)
                except UnicodeDecodeError:
                    raw_text = content_bytes.decode("utf-8", errors="replace")
                
                extraction_method = "markdown_parser" if ext == ".md" else "utf8_reader"
                metadata["format"] = "markdown" if ext == ".md" else "plain_text"
                if ext == ".md":
                    # Extract headers
                    headers = re.findall(r'^(#{1,6})\s+(.+)$', raw_text, re.MULTILINE)
                    metadata["markdown_headers"] = [{"level": len(h[0]), "title": h[1].strip()} for h in headers]

            elif ext == ".csv":
                with open(path, "rb") as f:
                    content_bytes = f.read()
                encoding = cls.detect_encoding(content_bytes)
                text_content = content_bytes.decode(encoding, errors="replace")
                
                extraction_method = "csv_parser"
                sample = text_content[:2048]
                try:
                    dialect = csv.Sniffer().sniff(sample)
                    delimiter = dialect.delimiter
                except Exception:
                    delimiter = ","

                reader = csv.reader(io.StringIO(text_content), delimiter=delimiter)
                rows = list(reader)
                row_count = len(rows)
                header = rows[0] if rows else []
                metadata["csv_headers"] = header
                metadata["row_count"] = row_count
                metadata["column_count"] = len(header)
                metadata["delimiter"] = delimiter

                # Generate structured narrative text from CSV rows
                lines = []
                if header:
                    lines.append("Table Headers: " + ", ".join(header))
                for idx, r in enumerate(rows[1:], start=1):
                    row_parts = [f"{header[i]}: {r[i]}" if i < len(header) else str(val) for i, val in enumerate(r)]
                    lines.append(f"Record {idx}: " + "; ".join(row_parts))
                raw_text = "\n\n".join(lines) if lines else text_content

            elif ext == ".json":
                with open(path, "rb") as f:
                    content_bytes = f.read()
                encoding = cls.detect_encoding(content_bytes)
                text_content = content_bytes.decode(encoding, errors="replace")
                
                extraction_method = "json_parser"
                try:
                    data = json.loads(text_content)
                except Exception as e:
                    raise ExtractionError(f"Corrupted or invalid JSON file: {e}")

                metadata["json_root_type"] = type(data).__name__
                if isinstance(data, dict):
                    metadata["top_level_keys"] = list(data.keys())
                    metadata["key_count"] = len(data)
                elif isinstance(data, list):
                    metadata["array_length"] = len(data)

                # Convert JSON to readable text blocks while keeping field mappings
                def _json_to_text(obj: Any, prefix: str = "") -> List[str]:
                    res = []
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            p = f"{prefix}.{k}" if prefix else k
                            if isinstance(v, (dict, list)):
                                res.extend(_json_to_text(v, p))
                            else:
                                res.append(f"{p}: {v}")
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            p = f"{prefix}[{i}]"
                            if isinstance(item, (dict, list)):
                                res.extend(_json_to_text(item, p))
                            else:
                                res.append(f"{p}: {item}")
                    else:
                        res.append(f"{prefix}: {obj}")
                    return res

                raw_text = "\n\n".join(_json_to_text(data))
                if not raw_text.strip():
                    raw_text = text_content

            elif ext == ".pdf":
                extraction_method = "pypdf"
                try:
                    reader = pypdf.PdfReader(str(path))
                except Exception as e:
                    raise ExtractionError(f"Corrupted or unreadable PDF: {e}")

                metadata["page_count"] = len(reader.pages)
                metadata["is_encrypted"] = reader.is_encrypted
                pdf_info = reader.metadata
                if pdf_info:
                    metadata["pdf_metadata"] = {
                        "title": pdf_info.title,
                        "author": pdf_info.author,
                        "subject": pdf_info.subject,
                        "creator": pdf_info.creator,
                        "producer": pdf_info.producer,
                        "creation_date": str(pdf_info.creation_date) if pdf_info.creation_date else None,
                        "modification_date": str(pdf_info.modification_date) if pdf_info.modification_date else None,
                    }

                page_texts = []
                for p_idx, page in enumerate(reader.pages):
                    try:
                        p_text = page.extract_text() or ""
                        if p_text.strip():
                            page_texts.append(p_text.strip())
                    except Exception as pe:
                        metadata.setdefault("extraction_warnings", []).append(f"Page {p_idx+1} extract warning: {pe}")
                raw_text = "\n\n".join(page_texts)

            elif ext == ".docx":
                extraction_method = "python_docx"
                try:
                    doc = docx.Document(str(path))
                except Exception as e:
                    raise ExtractionError(f"Corrupted or unreadable DOCX: {e}")

                props = doc.core_properties
                metadata["docx_metadata"] = {
                    "author": props.author,
                    "created": str(props.created) if props.created else None,
                    "modified": str(props.modified) if props.modified else None,
                    "title": props.title,
                    "subject": props.subject,
                    "revision": props.revision,
                    "category": props.category,
                    "comments": props.comments,
                }

                p_texts = [p.text for p in doc.paragraphs if p.text.strip()]
                # Also include table text
                for table in doc.tables:
                    for row in table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                        if row_text:
                            p_texts.append(row_text)

                raw_text = "\n\n".join(p_texts)

        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(f"Error during document extraction: {e}")

        # Structure analysis: paragraphs & sentences
        paragraphs, sentences, structural_counts = cls.segment_text_structure(raw_text)

        # Language detection (heuristic / stopword profile)
        lang_info = cls.detect_language(raw_text)

        return {
            "raw_text": raw_text,
            "character_encoding": encoding,
            "extraction_method": extraction_method,
            "file_size": file_size,
            "sha256": calculated_hash,
            "mime_type": MIME_TYPE_MAP.get(ext, "application/octet-stream"),
            "embedded_metadata": metadata,
            "structural_counts": structural_counts,
            "language_detection": lang_info,
            "paragraphs": paragraphs,
            "sentences": sentences,
        }

    @classmethod
    def segment_text_structure(cls, text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
        """
        Segments raw text into paragraphs and sentences with exact character offsets.
        """
        paragraphs: List[Dict[str, Any]] = []
        sentences: List[Dict[str, Any]] = []
        
        if not text:
            return [], [], {"character_count": 0, "word_count": 0, "paragraph_count": 0, "sentence_count": 0}

        # Find paragraphs preserving exact character offsets
        # Split on sequences of 2 or more newlines or single newlines if separated by blank spaces
        para_pattern = re.compile(r'(?:^[ \t]*\S.*(?:\r?\n[ \t]*\S.*)*)', re.MULTILINE)
        
        total_words = 0
        p_index = 0
        s_index = 0

        for match in para_pattern.finditer(text):
            p_text = match.group(0).strip()
            if not p_text:
                continue
            p_start = match.start()
            p_end = match.end()
            
            p_words = p_text.split()
            p_word_count = len(p_words)
            total_words += p_word_count

            # Segment sentences within this paragraph
            p_sentences = cls._split_into_sentences(p_text, p_start, p_index, s_index)
            s_index += len(p_sentences)
            sentences.extend(p_sentences)

            paragraphs.append({
                "index": p_index,
                "text": p_text,
                "start_char": p_start,
                "end_char": p_end,
                "word_count": p_word_count,
                "sentence_count": len(p_sentences)
            })
            p_index += 1

        structural_counts = {
            "character_count": len(text),
            "word_count": total_words,
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences)
        }

        return paragraphs, sentences, structural_counts

    @staticmethod
    def _split_into_sentences(para_text: str, para_start_char: int, p_idx: int, s_start_idx: int) -> List[Dict[str, Any]]:
        """Splits paragraph text into sentences with exact character offsets."""
        results = []
        s_idx = s_start_idx
        pattern = re.compile(r'[^.!?]*(?:[.!?]+(?:\s+|$)|$)')
        for match in pattern.finditer(para_text):
            sent_str = match.group(0).strip()
            if not sent_str:
                continue
            s_start = para_start_char + match.start()
            s_end = para_start_char + match.end()
            words = sent_str.split()
            results.append({
                "index": s_idx,
                "paragraph_index": p_idx,
                "text": sent_str,
                "start_char": s_start,
                "end_char": s_end,
                "word_count": len(words)
            })
            s_idx += 1
        if not results and para_text.strip():
            results.append({
                "index": s_idx,
                "paragraph_index": p_idx,
                "text": para_text.strip(),
                "start_char": para_start_char,
                "end_char": para_start_char + len(para_text),
                "word_count": len(para_text.split())
            })
        return results
    @staticmethod
    def detect_language(text: str) -> Dict[str, Any]:
        """
        Deterministic language identification using stopword distribution.
        """
        if not text or len(text.strip()) < 10:
            return {"language": "unknown", "confidence": 0.0, "method": "heuristic_empty"}
        
        words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{2,}\b', text)]
        if not words:
            return {"language": "unknown", "confidence": 0.0, "method": "heuristic_no_latin"}
        
        common_en_stopwords = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her",
            "she", "or", "an", "will", "my", "one", "all", "would", "there",
            "their", "what", "so", "up", "out", "if", "about", "who", "get",
            "which", "go", "me", "when", "make", "can", "like", "time", "no",
            "just", "him", "know", "take", "people", "into", "year", "your",
            "good", "some", "could", "them", "see", "other", "than", "then",
            "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first",
            "well", "way", "even", "new", "want", "because", "any", "these",
            "give", "day", "most", "us", "is", "was", "are", "were", "been"
        }
        
        matches = sum(1 for w in words if w in common_en_stopwords)
        ratio = matches / len(words)
        
        if ratio > 0.15:
            conf = min(0.99, round(0.5 + ratio, 2))
            return {"language": "en", "confidence": conf, "method": "stopword_frequency_profile"}
        
        return {"language": "en", "confidence": 0.5, "method": "default_latin_fallback"}


# Explicit standardized extractor service alias
class StandardizedExtractorService(DocumentExtractor):
    """
    Standardized extractor service that validates and reads TXT, MD, CSV, JSON, PDF, and DOCX files.
    """
    pass

