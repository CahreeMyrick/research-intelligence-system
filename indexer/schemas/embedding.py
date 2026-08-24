from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel


class EmbeddingLevel(str, Enum):
    DOCUMENT = "document"
    SECTION = "section"
    CHUNK = "chunk"


class EmbeddingRecord(BaseModel):
    """Represents a generated embedding for storage in PostgreSQL."""
    id: str
    document_id: Optional[str] = None
    section_id: Optional[str] = None
    chunk_id: Optional[str] = None
    level: EmbeddingLevel
    model_name: str
    model_version: Optional[str] = None
    dimensions: int
    embedding: list[float]
