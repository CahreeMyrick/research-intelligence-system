from __future__ import annotations

from enum import Enum
from typing import Optional, Set
from pydantic import BaseModel, Field, field_validator


class RetrievalMethod(str, Enum):
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    HYBRID = "hybrid"
    CITATION_EXPANSION = "citation_expansion"


class CitationDirection(str, Enum):
    OUTGOING = "outgoing"  # A -> papers cited by A
    INCOMING = "incoming"  # papers that cite A -> A
    BOTH = "both"          # both directions


class RetrievalQuery(BaseModel):
    """
    Structured retrieval query specification.
    Encapsulates search terms, weights, candidate pool limits, metadata filters,
    and optional citation expansion settings.
    """
    text: str = Field(..., min_length=1, description="Raw search query text.")
    top_k: int = Field(default=10, ge=1, le=100, description="Final number of results to return.")
    candidate_k: int = Field(default=50, ge=1, le=500, description="Candidate pool size for retrieval stages.")
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for semantic vector score.")
    lexical_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for lexical full-text score.")
    max_results_per_document: int = Field(default=3, ge=1, description="Maximum chunks to return from a single document.")
    
    # Metadata filters
    year_from: Optional[int] = Field(default=None, description="Filter for publication year >= year_from.")
    year_to: Optional[int] = Field(default=None, description="Filter for publication year <= year_to.")
    document_id: Optional[str] = Field(default=None, description="Filter for specific document ID.")
    doi: Optional[str] = Field(default=None, description="Filter for specific DOI.")
    arxiv_id: Optional[str] = Field(default=None, description="Filter for specific arXiv identifier.")

    # Optional citation expansion
    expand_citations: bool = Field(default=False, description="Whether to expand candidate pool using citation graph.")
    citation_direction: CitationDirection = Field(default=CitationDirection.BOTH, description="Direction for citation traversal.")
    citation_depth: int = Field(default=1, ge=1, le=2, description="Traversal depth for citation expansion.")
    max_citation_candidates: int = Field(default=25, ge=1, le=100, description="Maximum citation-expanded candidates.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query text cannot be empty or whitespace only.")
        return cleaned


class SearchResult(BaseModel):
    """
    Final grounded evidence result unit returned to callers and LLMs.
    Guaranteed to contain actual chunk source text and structural provenance.
    """
    document_id: str
    document_title: Optional[str] = None
    publication_year: Optional[int] = None
    
    section_id: str
    section_title: str
    section_path: list[str] = Field(default_factory=list)
    
    chunk_id: str
    source_text: str
    
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    
    score: float = Field(..., description="Normalized fused relevance score (0.0 to 1.0).")
    retrieval_method: RetrievalMethod = Field(..., description="Primary or hybrid method contributing to this result.")


class SearchResponse(BaseModel):
    """
    Complete response envelope for a retrieval operation.
    """
    original_query: str
    candidate_count: int
    returned_count: int
    duration_seconds: float = 0.0
    results: list[SearchResult] = Field(default_factory=list)


class CandidateChunk(BaseModel):
    """
    Internal retrieval candidate representation carrying raw multi-method scores
    prior to score normalization, weighted fusion, and diversity filtering.
    """
    chunk_id: str
    document_id: str
    document_title: Optional[str] = None
    publication_year: Optional[int] = None
    section_id: str
    section_title: str
    section_path: list[str] = Field(default_factory=list)
    source_text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    
    semantic_score: Optional[float] = None
    lexical_score: Optional[float] = None
    citation_score: Optional[float] = None
    final_score: float = 0.0
    matched_methods: Set[str] = Field(default_factory=set)
