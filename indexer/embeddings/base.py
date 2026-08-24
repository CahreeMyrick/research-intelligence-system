from __future__ import annotations

from typing import Protocol, runtime_checkable
from indexer.chunking.tokenization import TokenizerProtocol


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for pluggable embedding models."""
    @property
    def name(self) -> str:
        ...

    @property
    def version(self) -> str:
        ...

    @property
    def dimensions(self) -> int:
        ...

    @property
    def tokenizer(self) -> TokenizerProtocol:
        ...

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        ...

    def embed_single(self, text: str) -> list[float]:
        ...
