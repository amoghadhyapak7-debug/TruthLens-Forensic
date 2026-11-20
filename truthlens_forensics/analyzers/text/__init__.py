"""
TruthLens Text Forensics Module

Specification:
"Integrate a standardized extractor service that validates and reads TXT, MD, CSV, JSON, PDF, and DOCX files.
Modify the pipeline to execute all analyses—including entity, claim, temporal, location, linguistic, internal consistency, and AI detection—in isolation.
Ensure all processed text and final analytical results are stored as verifiable, structured artifact files, and mandate that the final output is delivered exclusively as a standardized JSON structure."
"""
from truthlens_forensics.analyzers.text.analyzer import analyze_text, TextAnalyzer
from truthlens_forensics.analyzers.text.extractors import DocumentExtractor, StandardizedExtractorService

__all__ = [
    "analyze_text",
    "TextAnalyzer",
    "DocumentExtractor",
    "StandardizedExtractorService",
]

