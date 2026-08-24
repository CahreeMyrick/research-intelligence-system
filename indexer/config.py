from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class IndexerConfig:
    """Centralized configuration for the indexing layer."""
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "postgresql://localhost:5432/research_intel")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    )
    embedding_dimensions: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
    )
    embedding_model_version: str = "1.0"
    
    # Chunking parameters
    chunk_target_tokens: int = 512
    chunk_overlap_tokens: int = 64
    chunk_max_tokens: int = 512
    chunker_version: str = "1.0"
    
    # Parser version expected from ingestion layer
    parser_version: str = "1.0"
    
    # Batch sizes and operational limits
    embedding_batch_size: int = 32
    max_retries: int = 3

    @classmethod
    def from_env(cls, **overrides) -> IndexerConfig:
        """Create configuration merging defaults, environment variables, and explicit overrides."""
        db_url = overrides.get("database_url") or os.getenv("DATABASE_URL", "postgresql://localhost:5432/research_intel")
        model = overrides.get("embedding_model") or os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        dims = overrides.get("embedding_dimensions") or int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
        target_tokens = overrides.get("chunk_target_tokens") or int(os.getenv("CHUNK_TARGET_TOKENS", "512"))
        overlap_tokens = overrides.get("chunk_overlap_tokens") or int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))
        max_tokens = overrides.get("chunk_max_tokens") or int(os.getenv("CHUNK_MAX_TOKENS", "512"))
        chunker_ver = overrides.get("chunker_version", "1.0")
        parser_ver = overrides.get("parser_version", "1.0")
        model_ver = overrides.get("embedding_model_version", "1.0")
        batch_size = overrides.get("embedding_batch_size") or int(os.getenv("INDEXER_BATCH_SIZE", "32"))

        return cls(
            database_url=db_url,
            embedding_model=model,
            embedding_dimensions=dims,
            embedding_model_version=model_ver,
            chunk_target_tokens=target_tokens,
            chunk_overlap_tokens=overlap_tokens,
            chunk_max_tokens=max_tokens,
            chunker_version=chunker_ver,
            parser_version=parser_ver,
            embedding_batch_size=batch_size,
        )
