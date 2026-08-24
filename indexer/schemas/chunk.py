from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ChunkArtifact(BaseModel):
    """
    Reproducible chunk artifact format saved to disk: corpus/chunks/<doc_id>.json.
    Does NOT store raw embedding vectors.
    """
    chunk_id: str
    document_id: str
    section_id: str
    section_title: str
    section_path: list[str] = Field(default_factory=list)
    source_text: str
    embedding_text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    paragraph_positions: list[int] = Field(default_factory=list)
    token_count: int
    chunk_index: int
    chunker_version: str
    chunker_config_hash: str


class DocumentChunksArtifact(BaseModel):
    """Container for all chunks produced for a document."""
    document_id: str
    chunk_count: int
    chunker_version: str
    chunker_config_hash: str
    chunks: list[ChunkArtifact] = Field(default_factory=list)
