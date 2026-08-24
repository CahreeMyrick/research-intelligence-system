from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentProcessingRecord(BaseModel):
    """State record tracking a document's indexing version and status."""
    document_id: str
    parser_version: Optional[str] = None
    chunker_version: Optional[str] = None
    chunker_config_hash: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_model_version: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    last_indexed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None
