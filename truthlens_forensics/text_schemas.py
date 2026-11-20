from __future__ import annotations
import datetime
import uuid
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field

# Re-export text analyzer schemas for convenience
from truthlens_forensics.analyzers.text.schemas import (
    TextEvidenceInput,
    TextAnalyzerOutput,
    ExtractedEntity,
    ExtractedClaim,
    ConsistencyItem,
    AnalysisSignal,
    ForensicFinding,
    ForensicArtifact,
)


class EvidenceInput(BaseModel):
    """Standard evidence input model for TruthLens forensics platform."""
    model_config = ConfigDict(frozen=False)
    evidence_id: str
    file_path: str
    original_filename: str
    file_type: str = "text"
    mime_type: Optional[str] = None
    file_size: Optional[int] = None
    sha256: Optional[str] = None
    ingestion_metadata: Dict[str, Any] = Field(default_factory=dict)


class ArtifactFile(BaseModel):
    model_config = ConfigDict(frozen=False)
    path: str
    artifact_type: str
    description: str
    relative_url: Optional[str] = None


class StandardEvidenceReport(BaseModel):
    """Standard output schema for TruthLens text forensics evidence processing."""
    model_config = ConfigDict(frozen=False)
    module: str = "text"
    evidence_id: str
    status: str = "completed"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    analyses: Dict[str, Any] = Field(default_factory=dict)
    signals: List[Dict[str, Any]] = Field(default_factory=list)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)

