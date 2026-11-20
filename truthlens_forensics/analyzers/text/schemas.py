from __future__ import annotations
import uuid
from typing import Any, Optional, List, Dict, Union
from pydantic import BaseModel, ConfigDict, Field


class TextEvidenceInput(BaseModel):
    model_config = ConfigDict(frozen=False, extra="allow")
    evidence_id: str
    file_path: str
    original_filename: str
    file_type: str = "text"
    mime_type: Optional[str] = None
    file_size: Optional[int] = None
    sha256: Optional[str] = None
    ingestion_metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceSpan(BaseModel):
    model_config = ConfigDict(frozen=False)
    start_char: int
    end_char: int
    paragraph_index: int = 0
    sentence_index: Optional[int] = None
    context: Optional[str] = None


class EntityMention(BaseModel):
    model_config = ConfigDict(frozen=False)
    text: str
    start_char: int
    end_char: int
    paragraph_index: int = 0
    context: str = ""


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(frozen=False)
    entity_type: str
    value: str
    normalized_value: Optional[str] = None
    occurrences: int = 1
    mentions: List[EntityMention] = Field(default_factory=list)
    extraction_method: str = "deterministic_regex"
    reliability: str = "high"
    limitations: List[str] = Field(default_factory=list)


class ExtractedClaim(BaseModel):
    model_config = ConfigDict(frozen=False)
    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    statement: str
    category: str
    secondary_tags: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    numbers: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    source_span: SourceSpan
    extraction_status: str = "completed"


class ConsistencyItem(BaseModel):
    model_config = ConfigDict(frozen=False)
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    classification: str  # "confirmed_textual_mismatch" | "potential_contradiction" | "ambiguous_statement" | "duplicate_information"
    description: str
    supporting_passages: List[Dict[str, Any]] = Field(default_factory=list)
    explanation: str
    limitations: List[str] = Field(default_factory=list)


class AnalysisSignal(BaseModel):
    model_config = ConfigDict(frozen=False)
    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    signal_type: str
    source_analysis: str
    description: str
    evidence_spans: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0


class ForensicFinding(BaseModel):
    model_config = ConfigDict(frozen=False)
    finding_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    category: str
    description: str
    severity: str  # "low", "medium", "high"
    supporting_signals: List[str] = Field(default_factory=list)
    relevant_passages: List[Dict[str, Any]] = Field(default_factory=list)
    explanation: str
    limitations: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None


class ForensicArtifact(BaseModel):
    model_config = ConfigDict(frozen=False)
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    path: str
    artifact_type: str
    description: str
    file_size_bytes: int = 0
    relative_url: Optional[str] = None


class ParagraphSpan(BaseModel):
    model_config = ConfigDict(frozen=False)
    index: int
    text: str
    start_char: int
    end_char: int
    word_count: int
    sentence_count: int = 0


class SentenceSpan(BaseModel):
    model_config = ConfigDict(frozen=False)
    index: int
    paragraph_index: int
    text: str
    start_char: int
    end_char: int
    word_count: int


# Sub-analyses structures
class TextExtractionAnalysis(BaseModel):
    model_config = ConfigDict(frozen=False)
    status: str = "completed"
    extraction_method: str = "utf8_reader"
    character_encoding: str = "utf-8"
    embedded_metadata: Dict[str, Any] = Field(default_factory=dict)
    structural_counts: Dict[str, int] = Field(default_factory=dict)
    language_detection: Dict[str, Any] = Field(default_factory=dict)
    paragraphs: List[Dict[str, Any]] = Field(default_factory=list)
    sentences: List[Dict[str, Any]] = Field(default_factory=list)
    raw_text_length: int = 0
    errors: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class ClaimExtractionAnalysis(BaseModel):
    model_config = ConfigDict(frozen=False)
    status: str = "completed"
    claims_count: int = 0
    claim_types_distribution: Dict[str, int] = Field(default_factory=dict)
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class EntityExtractionAnalysis(BaseModel):
    model_config = ConfigDict(frozen=False)
    status: str = "completed"
    entity_counts: Dict[str, int] = Field(default_factory=dict)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class TemporalAnalysis(BaseModel):
    model_config = ConfigDict(frozen=False)
    status: str = "completed"
    temporal_expressions: List[Dict[str, Any]] = Field(default_factory=list)
    chronological_order: List[Dict[str, Any]] = Field(default_factory=list)
    potential_chronology_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class LocationAnalysis(BaseModel):
    model_config = ConfigDict(frozen=False)
    status: str = "completed"
    location_mentions: List[Dict[str, Any]] = Field(default_factory=list)
    distinct_locations: List[str] = Field(default_factory=list)
    potential_location_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class LinguisticAnalysis(BaseModel):
    model_config = ConfigDict(frozen=False)
    status: str = "completed"
    sentence_length_distribution: Dict[str, float] = Field(default_factory=dict)
    word_length_distribution: Dict[str, float] = Field(default_factory=dict)
    vocabulary_diversity: Dict[str, float] = Field(default_factory=dict)
    repetition_metrics: Dict[str, Any] = Field(default_factory=dict)
    punctuation_patterns: Dict[str, Any] = Field(default_factory=dict)
    capitalization_patterns: Dict[str, Any] = Field(default_factory=dict)
    readability_scores: Dict[str, float] = Field(default_factory=dict)
    paragraph_metrics: Dict[str, Any] = Field(default_factory=dict)
    section_shifts: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class ConsistencyAnalysis(BaseModel):
    model_config = ConfigDict(frozen=False)
    status: str = "completed"
    items_count: int = 0
    inconsistencies: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class TextComparisonAnalysis(BaseModel):
    model_config = ConfigDict(frozen=False)
    status: str = "completed"
    intra_document_similarity: Dict[str, Any] = Field(default_factory=dict)
    duplicate_blocks: List[Dict[str, Any]] = Field(default_factory=list)
    reference_comparison: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class AIDetectorAnalysis(BaseModel):
    model_config = ConfigDict(frozen=False)
    status: str = "not_implemented"
    model_name: Optional[str] = None
    model_version_or_source: Optional[str] = None
    input_requirements: Dict[str, Any] = Field(default_factory=dict)
    raw_output: Any = None
    raw_scores: Dict[str, float] = Field(default_factory=dict)
    inference_ran: bool = False
    errors: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class TextAnalyzerOutput(BaseModel):
    """
    Standardized Output Schema for TruthLens Text Analyzer Module.
    """
    model_config = ConfigDict(frozen=False)
    module: str = "text"
    evidence_id: str
    status: str = "completed"  # "completed" | "partial" | "failed"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    analyses: Dict[str, Any] = Field(default_factory=dict)
    signals: List[Dict[str, Any]] = Field(default_factory=list)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)

