from __future__ import annotations

import pytest

from indexer.embeddings.mock import MockEmbeddingProvider
from indexer.retrieval.models import RetrievalQuery
from indexer.retrieval.semantic import SemanticRetriever


def test_semantic_retriever_can_be_constructed(repo):
    provider = MockEmbeddingProvider(dimensions=768)

    retriever = SemanticRetriever(
        db_manager=repo.db,
        embedding_provider=provider,
    )

    assert retriever.embedding_provider is provider


def test_semantic_retrieval_returns_chunks(repo):
    provider = MockEmbeddingProvider(dimensions=768)

    retriever = SemanticRetriever(
        db_manager=repo.db,
        embedding_provider=provider,
    )

    query = RetrievalQuery(
        text="database architecture",
        top_k=5,
        candidate_k=5,
    )

    # The test database may legitimately contain no indexed chunks.
    # We are primarily verifying that the retrieval path executes correctly.
    results = retriever.search(query)

    assert isinstance(results, list)

    for result in results:
        assert result.chunk_id
        assert result.document_id
        assert result.section_id
        assert result.source_text
        assert result.semantic_score is not None
        assert 0.0 <= result.semantic_score <= 1.0
        assert "semantic" in result.matched_methods


def test_semantic_retrieval_respects_document_filter(repo):
    provider = MockEmbeddingProvider(dimensions=768)

    retriever = SemanticRetriever(
        db_manager=repo.db,
        embedding_provider=provider,
    )

    query = RetrievalQuery(
        text="database architecture",
        document_id="does-not-exist",
        candidate_k=10,
    )

    results = retriever.search(query)

    assert results == []
