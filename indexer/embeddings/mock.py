from __future__ import annotations

import hashlib
import math
from indexer.chunking.tokenization import RegexFallbackTokenizer, TokenizerProtocol
from indexer.embeddings.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic pseudo-embedding provider for fast unit tests without PyTorch weights.
    Generates reproducible unit-norm vectors from text hashes.
    """
    def __init__(
        self,
        model_name: str = "mock-embedding-model",
        model_version: str = "1.0",
        dimensions: int = 768,
    ):
        self._model_name = model_name
        self._model_version = model_version
        self._dimensions = dimensions
        self._tokenizer = RegexFallbackTokenizer()

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def version(self) -> str:
        return self._model_version

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def tokenizer(self) -> TokenizerProtocol:
        return self._tokenizer

    def _hash_to_vector(self, text: str) -> list[float]:
        vec: list[float] = []
        base_hash = hashlib.sha256(text.encode("utf-8")).digest()
        
        # Expand bytes deterministically to fill dimensions
        for i in range(self._dimensions):
            idx = i % len(base_hash)
            val = (base_hash[idx] + (i * 31)) % 256
            vec.append((val - 128) / 128.0)
            
        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_single(self, text: str) -> list[float]:
        return self._hash_to_vector(text)
