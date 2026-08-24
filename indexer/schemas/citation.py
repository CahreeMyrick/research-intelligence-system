from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ResolvedCitation(BaseModel):
    """Resolved citation relationship (source_document -> target_document)."""
    id: str
    source_document_id: str
    target_document_id: str
    reference_id: int
    raw_text: str
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    resolution_method: str  # 'doi', 'arxiv', 'title'
    resolution_confidence: float = 1.0


class UnresolvedReference(BaseModel):
    """Reference from a document that did not resolve to a corpus paper."""
    id: str
    source_document_id: str
    reference_id: int
    raw_text: str
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
