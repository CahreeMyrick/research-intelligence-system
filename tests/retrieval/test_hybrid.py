from __future__ import annotations

from indexer.retrieval.hybrid import HybridRetriever
from indexer.retrieval.models import CandidateChunk, RetrievalQuery


def make_candidate(
    chunk_id: str,
    document_id: str,
    *,
    semantic: float | None = None,
    lexical: float | None = None,
) -> CandidateChunk:
    methods = set()

    if semantic is not None:
        methods.add("semantic")

    if lexical is not None:
        methods.add("lexical")

    return CandidateChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        section_id=f"section-{chunk_id}",
        section_title="Introduction",
        source_text=f"Source text for {chunk_id}",
        semantic_score=semantic,
        lexical_score=lexical,
        matched_methods=methods,
    )


def test_hybrid_merges_candidates():
    retriever = HybridRetriever()

    query = RetrievalQuery(
        text="database architecture",
        top_k=10,
        semantic_weight=0.7,
        lexical_weight=0.3,
    )

    semantic = [
        make_candidate(
            "chunk-1",
            "doc-1",
            semantic=0.9,
        ),
        make_candidate(
            "chunk-2",
            "doc-2",
            semantic=0.5,
        ),
    ]

    lexical = [
        make_candidate(
            "chunk-1",
            "doc-1",
            lexical=0.8,
        ),
        make_candidate(
            "chunk-3",
            "doc-3",
            lexical=0.7,
        ),
    ]

    results = retriever.combine(
        query,
        semantic,
        lexical,
    )

    assert len(results) == 3

    chunk_1 = next(r for r in results if r.chunk_id == "chunk-1")

    assert chunk_1.semantic_score == 0.9
    assert chunk_1.lexical_score == 0.8
    assert chunk_1.matched_methods == {"semantic", "lexical"}


def test_hybrid_applies_weights():
    retriever = HybridRetriever()

    query = RetrievalQuery(
        text="database architecture",
        top_k=10,
        semantic_weight=1.0,
        lexical_weight=0.0,
    )

    semantic = [
        make_candidate("chunk-1", "doc-1", semantic=0.9),
        make_candidate("chunk-2", "doc-2", semantic=0.1),
    ]

    lexical = []

    results = retriever.combine(
        query,
        semantic,
        lexical,
    )

    assert results[0].chunk_id == "chunk-1"
    assert results[0].final_score == 1.0
    assert results[1].final_score == 0.0


def test_hybrid_enforces_document_diversity():
    retriever = HybridRetriever()

    query = RetrievalQuery(
        text="database architecture",
        top_k=10,
        max_results_per_document=1,
    )

    semantic = [
        make_candidate("chunk-1", "doc-1", semantic=0.9),
        make_candidate("chunk-2", "doc-1", semantic=0.8),
        make_candidate("chunk-3", "doc-2", semantic=0.7),
    ]

    results = retriever.combine(
        query,
        semantic,
        [],
    )

    assert len(results) == 2

    document_ids = [r.document_id for r in results]

    assert document_ids.count("doc-1") == 1
    assert document_ids.count("doc-2") == 1


def test_normalization_handles_identical_scores():
    retriever = HybridRetriever()

    normalized = retriever._normalize_scores(
        [0.5, 0.5, 0.5]
    )

    assert normalized == [1.0, 1.0, 1.0]


def test_normalization_returns_zero_to_one():
    retriever = HybridRetriever()

    normalized = retriever._normalize_scores(
        [0.0, 0.5, 1.0]
    )

    assert normalized == [0.0, 0.5, 1.0]
