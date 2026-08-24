from __future__ import annotations

from indexer.retrieval.lexical import LexicalRetriever
from indexer.retrieval.models import RetrievalQuery


def test_lexical_retriever_can_be_constructed(repo):
    retriever = LexicalRetriever(db_manager=repo.db)

    assert retriever.db is repo.db


def test_lexical_retrieval_returns_valid_candidates(repo):
    retriever = LexicalRetriever(db_manager=repo.db)

    query = RetrievalQuery(
        text="database architecture",
        candidate_k=10,
    )

    results = retriever.search(query)

    assert isinstance(results, list)

    for result in results:
        assert result.chunk_id
        assert result.document_id
        assert result.section_id
        assert result.source_text
        assert result.lexical_score is not None
        assert result.lexical_score >= 0.0
        assert "lexical" in result.matched_methods


def test_lexical_retrieval_respects_document_filter(repo):
    retriever = LexicalRetriever(db_manager=repo.db)

    query = RetrievalQuery(
        text="database architecture",
        document_id="does-not-exist",
        candidate_k=10,
    )

    results = retriever.search(query)

    assert results == []
