from __future__ import annotations

import pytest

from indexer.retrieval.models import (
    RetrievalMethod,
    RetrievalQuery,
)
from indexer.retrieval.repository import RetrievalRepository


class FakeEmbeddingProvider:
    name = "fake-model"
    version = "1.0"
    dimensions = 3
    tokenizer = None

    def embed_single(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_query_validation():
    query = RetrievalQuery(text="  software architecture  ")

    assert query.text == "software architecture"
    assert query.top_k == 10
    assert query.candidate_k == 50
    assert query.semantic_weight == 0.7
    assert query.lexical_weight == 0.3


def test_empty_query_rejected():
    with pytest.raises(ValueError):
        RetrievalQuery(text="   ")


def test_metadata_filter_generation():
    query = RetrievalQuery(
        text="architecture",
        year_from=2020,
        year_to=2025,
        document_id="doc-123",
        doi="10.1234/test",
        arxiv_id="1234.5678",
    )

    sql, params = RetrievalRepository._build_metadata_filter(query)

    assert "d.publication_year >= %s" in sql
    assert "d.publication_year <= %s" in sql
    assert "d.id = %s" in sql
    assert "d.doi = %s" in sql
    assert "d.arxiv_id = %s" in sql

    assert params == [
        2020,
        2025,
        "doc-123",
        "10.1234/test",
        "1234.5678",
    ]


def test_metadata_filter_is_empty_when_no_filters():
    query = RetrievalQuery(text="architecture")

    sql, params = RetrievalRepository._build_metadata_filter(query)

    assert sql == ""
    assert params == []


def test_score_normalization():
    values = [0.2, 0.5, 0.8]

    normalized = RetrievalRepository._normalize_scores(values)

    assert normalized == pytest.approx(
        [0.0, 0.5, 1.0]
    )


def test_score_normalization_equal_values():
    values = [0.5, 0.5, 0.5]

    normalized = RetrievalRepository._normalize_scores(values)

    assert normalized == [1.0, 1.0, 1.0]


def test_score_normalization_empty():
    assert RetrievalRepository._normalize_scores([]) == []


def test_document_diversity():
    from indexer.retrieval.models import CandidateChunk

    candidates = [
        CandidateChunk(
            chunk_id="a1",
            document_id="doc-a",
            section_id="s1",
            section_title="Section 1",
            source_text="A1",
            final_score=1.0,
        ),
        CandidateChunk(
            chunk_id="a2",
            document_id="doc-a",
            section_id="s2",
            section_title="Section 2",
            source_text="A2",
            final_score=0.9,
        ),
        CandidateChunk(
            chunk_id="a3",
            document_id="doc-a",
            section_id="s3",
            section_title="Section 3",
            source_text="A3",
            final_score=0.8,
        ),
        CandidateChunk(
            chunk_id="b1",
            document_id="doc-b",
            section_id="s1",
            section_title="Section 1",
            source_text="B1",
            final_score=0.7,
        ),
    ]

    selected = RetrievalRepository._apply_document_diversity(
        candidates,
        max_results_per_document=2,
        limit=4,
    )

    assert [c.chunk_id for c in selected] == [
        "a1",
        "a2",
        "b1",
    ]


def test_candidate_merging():
    from indexer.retrieval.models import CandidateChunk

    destination = {}

    semantic = CandidateChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        section_id="section-1",
        section_title="Introduction",
        source_text="architecture systems",
        semantic_score=0.9,
        matched_methods={"semantic"},
    )

    lexical = CandidateChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        section_id="section-1",
        section_title="Introduction",
        source_text="architecture systems",
        lexical_score=0.7,
        matched_methods={"lexical"},
    )

    RetrievalRepository._merge_candidates(
        destination,
        [semantic],
    )

    RetrievalRepository._merge_candidates(
        destination,
        [lexical],
    )

    assert len(destination) == 1

    candidate = destination["chunk-1"]

    assert candidate.semantic_score == 0.9
    assert candidate.lexical_score == 0.7
    assert candidate.matched_methods == {
        "semantic",
        "lexical",
    }


def test_hybrid_method_detection():
    from indexer.retrieval.models import CandidateChunk

    candidate = CandidateChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        section_id="section-1",
        section_title="Introduction",
        source_text="test",
        semantic_score=0.9,
        lexical_score=0.8,
        matched_methods={"semantic", "lexical"},
        final_score=0.95,
    )

    result = RetrievalRepository._candidate_to_result(
        candidate,
        RetrievalQuery(text="test"),
    )

    assert result.retrieval_method == RetrievalMethod.HYBRID
    assert result.chunk_id == "chunk-1"
    assert result.source_text == "test"
    assert result.score == 0.95


def test_semantic_method_detection():
    from indexer.retrieval.models import CandidateChunk

    candidate = CandidateChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        section_id="section-1",
        section_title="Introduction",
        source_text="test",
        semantic_score=0.9,
        matched_methods={"semantic"},
        final_score=0.9,
    )

    result = RetrievalRepository._candidate_to_result(
        candidate,
        RetrievalQuery(text="test"),
    )

    assert result.retrieval_method == RetrievalMethod.SEMANTIC


def test_lexical_method_detection():
    from indexer.retrieval.models import CandidateChunk

    candidate = CandidateChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        section_id="section-1",
        section_title="Introduction",
        source_text="test",
        lexical_score=0.9,
        matched_methods={"lexical"},
        final_score=0.9,
    )

    result = RetrievalRepository._candidate_to_result(
        candidate,
        RetrievalQuery(text="test"),
    )

    assert result.retrieval_method == RetrievalMethod.LEXICAL
