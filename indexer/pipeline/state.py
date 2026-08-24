from __future__ import annotations

from enum import Enum
from typing import Optional

from ingest.schema import Document as IngestDocument
from indexer.chunking.base import ChunkerConfig
from indexer.embeddings.base import EmbeddingProvider
from indexer.schemas.processing import DocumentProcessingRecord, ProcessingStatus


class ProcessingDecision(str, Enum):
    NEEDS_FULL_INDEXING = "needs_full_indexing"
    NEEDS_RECHUNK_AND_REEMBED = "needs_rechunk_and_reembed"
    NEEDS_REEMBED_ONLY = "needs_reembed_only"
    NEEDS_CITATION_RESOLUTION_ONLY = "needs_citation_resolution_only"
    UP_TO_DATE = "up_to_date"


def determine_processing_state(
    document: IngestDocument,
    current_record: Optional[DocumentProcessingRecord],
    chunker_config: ChunkerConfig,
    embedding_provider: EmbeddingProvider,
    parser_version: str = "1.0",
    force: bool = False,
) -> ProcessingDecision:
    """
    Determine whether a document requires re-indexing based on versioning
    and configuration changes.
    """
    if force or current_record is None:
        return ProcessingDecision.NEEDS_FULL_INDEXING

    if current_record.status != ProcessingStatus.COMPLETED:
        return ProcessingDecision.NEEDS_FULL_INDEXING

    if current_record.parser_version != parser_version:
        return ProcessingDecision.NEEDS_FULL_INDEXING

    # Check chunker version & configuration hash
    current_config_hash = chunker_config.config_hash()
    if (
        current_record.chunker_version != chunker_config.chunker_version
        or current_record.chunker_config_hash != current_config_hash
    ):
        return ProcessingDecision.NEEDS_RECHUNK_AND_REEMBED

    # Check embedding model, version & dimensions
    if (
        current_record.embedding_model != embedding_provider.name
        or current_record.embedding_model_version != embedding_provider.version
        or current_record.embedding_dimensions != embedding_provider.dimensions
    ):
        return ProcessingDecision.NEEDS_REEMBED_ONLY

    return ProcessingDecision.UP_TO_DATE
