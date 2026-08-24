from __future__ import annotations

from dataclasses import dataclass

import requests

from indexer.chunking.tokenization import TokenizerProtocol
from indexer.embeddings.base import EmbeddingProvider


@dataclass
class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider backed by Ollama.

    Intended for models such as:
        nomic-embed-text:latest
    """

    model: str = "nomic-embed-text:latest"
    base_url: str = "http://localhost:11434"
    model_version: str = "latest"
    expected_dimensions: int | None = None
    timeout_seconds: float = 120.0

    @property
    def name(self) -> str:
        return self.model

    @property
    def version(self) -> str:
        return self.model_version

    @property
    def dimensions(self) -> int:
        if self.expected_dimensions is None:
            raise RuntimeError(
                "Embedding dimensions are unknown. "
                "Set expected_dimensions when constructing "
                "OllamaEmbeddingProvider."
            )

        return self.expected_dimensions

    @property
    def tokenizer(self) -> TokenizerProtocol:
        raise NotImplementedError(
            "OllamaEmbeddingProvider does not expose a tokenizer."
        )

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        if not texts:
            return []

        cleaned = [text.replace("\x00", " ") for text in texts]

        response = requests.post(
            f"{self.base_url.rstrip('/')}/api/embed",
            json={
                "model": self.model,
                "input": cleaned,
            },
            timeout=self.timeout_seconds,
        )

        response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings")

        if not isinstance(embeddings, list):
            raise RuntimeError(
                "Ollama response did not contain valid 'embeddings'."
            )

        result = []

        for embedding in embeddings:
            if not isinstance(embedding, list):
                raise RuntimeError(
                    "Ollama returned an invalid embedding."
                )

            result.append([float(value) for value in embedding])

        if len(result) != len(cleaned):
            raise RuntimeError(
                f"Ollama returned {len(result)} embeddings "
                f"for {len(cleaned)} inputs."
            )

        if result:
            actual_dimensions = len(result[0])

            for embedding in result:
                if len(embedding) != actual_dimensions:
                    raise RuntimeError(
                        "Ollama returned embeddings with inconsistent dimensions."
                    )

            if (
                self.expected_dimensions is not None
                and actual_dimensions != self.expected_dimensions
            ):
                raise ValueError(
                    f"Model {self.model} produced {actual_dimensions} dimensions, "
                    f"expected {self.expected_dimensions}."
                )

        return result

    def embed_single(self, text: str) -> list[float]:
        return self.embed_texts([text], batch_size=1)[0]
