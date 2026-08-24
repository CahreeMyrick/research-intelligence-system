from __future__ import annotations

import logging
from typing import Optional
from indexer.chunking.tokenization import HuggingFaceTokenizer, TokenizerProtocol
from indexer.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformersProvider(EmbeddingProvider):
    """
    Embedding provider using local SentenceTransformers models.
    Supports lazy loading and tokenization.
    """
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        model_version: str = "1.0",
        expected_dimensions: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self._model_name = model_name
        self._model_version = model_version
        self._expected_dimensions = expected_dimensions
        self._device = device
        self._model = None
        self._tokenizer = None
        self._dimensions = expected_dimensions

    def _load_model(self):
        if self._model is None:
            logger.info("Loading SentenceTransformers model: %s", self._model_name)
            from sentence_transformers import SentenceTransformer
            
            kwargs = {}
            if self._device:
                kwargs["device"] = self._device
                
            self._model = SentenceTransformer(self._model_name, **kwargs)
            self._tokenizer = HuggingFaceTokenizer(self._model.tokenizer)
            
            # Detect dimensions dynamically if not provided
            test_emb = self._model.encode("dim_check", normalize_embeddings=True)
            self._dimensions = int(len(test_emb))
            
            if self._expected_dimensions and self._dimensions != self._expected_dimensions:
                raise ValueError(
                    f"Model {self._model_name} produces {self._dimensions} dimensions, "
                    f"but expected {self._expected_dimensions}"
                )

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def version(self) -> str:
        return self._model_version

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._load_model()
        return self._dimensions

    @property
    def tokenizer(self) -> TokenizerProtocol:
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        self._load_model()
        # Clean texts
        cleaned = [t.replace("\x00", " ") for t in texts]
        embeddings = self._model.encode(
            cleaned,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [e.tolist() for e in embeddings]

    def embed_single(self, text: str) -> list[float]:
        return self.embed_texts([text], batch_size=1)[0]
