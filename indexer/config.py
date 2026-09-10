from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class IndexerConfig:
    database_url: str

    embedding_model: str
    embedding_dimensions: int
    embedding_model_version: str

    chunk_target_tokens: int
    chunk_overlap_tokens: int
    chunk_max_tokens: int
    chunker_version: str

    parser_version: str

    embedding_batch_size: int
    max_retries: int

    llm_model: str

    top_k: int
    candidate_k: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IndexerConfig":
        path = Path(path)

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return cls(**data)
