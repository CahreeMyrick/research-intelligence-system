from __future__ import annotations

import logging
from typing import Optional

from psycopg.rows import dict_row

from indexer.database.connection import DatabaseManager
from indexer.embeddings.base import EmbeddingProvider
from indexer.retrieval.models import (
    CandidateChunk,
    CitationDirection,
    RetrievalMethod,
    RetrievalQuery,
    SearchResponse,
    SearchResult,
)

logger = logging.getLogger(__name__)


class RetrievalRepository:
    """
    PostgreSQL/pgvector retrieval layer.

    This class is intentionally responsible only for retrieval.
    Query interpretation, reranking, and LLM orchestration belong above it.

    Supported retrieval modes:
      - semantic
      - lexical
      - hybrid

    Citation expansion is represented in the query model but is intentionally
    kept as a separate extension point so the core retrieval path remains simple.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_provider: EmbeddingProvider,
    ):
        self.db = db_manager
        self.embedding_provider = embedding_provider

    def search(
        self,
        query: RetrievalQuery,
    ) -> SearchResponse:
        """
        Execute retrieval according to the supplied query.

        The default path is hybrid retrieval:
          1. semantic candidate retrieval using pgvector
          2. lexical candidate retrieval using PostgreSQL full-text search
          3. candidate fusion
          4. metadata filtering
          5. document diversity filtering
          6. final SearchResult construction
        """
        import time

        start = time.perf_counter()

        candidates: dict[str, CandidateChunk] = {}

        if query.semantic_weight > 0:
            semantic_candidates = self._semantic_search(query)
            self._merge_candidates(candidates, semantic_candidates)

        if query.lexical_weight > 0:
            lexical_candidates = self._lexical_search(query)
            self._merge_candidates(candidates, lexical_candidates)

        self._calculate_final_scores(candidates, query)

        ranked = sorted(
            candidates.values(),
            key=lambda candidate: candidate.final_score,
            reverse=True,
        )

        selected = self._apply_document_diversity(
            ranked,
            max_results_per_document=query.max_results_per_document,
            limit=query.top_k,
        )

        results = [
            self._candidate_to_result(candidate, query)
            for candidate in selected
        ]

        duration = time.perf_counter() - start

        return SearchResponse(
            original_query=query.text,
            candidate_count=len(candidates),
            returned_count=len(results),
            duration_seconds=duration,
            results=results,
        )

    # ------------------------------------------------------------------
    # Semantic retrieval
    # ------------------------------------------------------------------

    def _semantic_search(
        self,
        query: RetrievalQuery,
    ) -> list[CandidateChunk]:
        """
        Retrieve chunk candidates using pgvector cosine distance.

        Only chunk-level embeddings are searched because chunks are the
        grounded evidence units returned to callers.
        """
        query_vector = self.embedding_provider.embed_single(query.text)

        where_sql, params = self._build_metadata_filter(query)

        sql = f"""
            SELECT
                c.id AS chunk_id,
                c.document_id,
                d.title AS document_title,
                d.publication_year,
                c.section_id,
                s.title AS section_title,
                c.source_text,
                c.page_start,
                c.page_end,
                1 - (e.embedding <=> %s::vector) AS semantic_score
            FROM embeddings e
            JOIN chunks c
                ON c.id = e.chunk_id
            JOIN documents d
                ON d.id = c.document_id
            JOIN sections s
                ON s.id = c.section_id
            WHERE e.level = 'chunk'
              AND e.model_name = %s
              {where_sql}
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s;
        """

        # Vector appears twice in the query.
        final_params = [
            query_vector,
            self.embedding_provider.name,
            *params,
            query_vector,
            query.candidate_k,
        ]

        candidates: list[CandidateChunk] = []

        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, final_params)

                for row in cur.fetchall():
                    candidates.append(
                        CandidateChunk(
                            chunk_id=row["chunk_id"],
                            document_id=row["document_id"],
                            document_title=row["document_title"],
                            publication_year=row["publication_year"],
                            section_id=row["section_id"],
                            section_title=row["section_title"],
                            section_path=[],
                            source_text=row["source_text"],
                            page_start=row["page_start"],
                            page_end=row["page_end"],
                            semantic_score=float(row["semantic_score"]),
                            matched_methods={"semantic"},
                        )
                    )

        return candidates

    # ------------------------------------------------------------------
    # Lexical retrieval
    # ------------------------------------------------------------------

    def _lexical_search(
        self,
        query: RetrievalQuery,
    ) -> list[CandidateChunk]:
        """
        PostgreSQL full-text retrieval.

        We use websearch_to_tsquery because it handles ordinary user queries
        more gracefully than requiring callers to construct tsquery syntax.
        """
        where_sql, metadata_params = self._build_metadata_filter(query)

        sql = f"""
            SELECT
                c.id AS chunk_id,
                c.document_id,
                d.title AS document_title,
                d.publication_year,
                c.section_id,
                s.title AS section_title,
                c.source_text,
                c.page_start,
                c.page_end,
                ts_rank_cd(
                    to_tsvector('english', c.source_text),
                    websearch_to_tsquery('english', %s)
                ) AS lexical_score
            FROM chunks c
            JOIN documents d
                ON d.id = c.document_id
            JOIN sections s
                ON s.id = c.section_id
            WHERE to_tsvector('english', c.source_text)
                  @@ websearch_to_tsquery('english', %s)
              {where_sql}
            ORDER BY lexical_score DESC
            LIMIT %s;
        """

        params = [
            query.text,
            query.text,
            *metadata_params,
            query.candidate_k,
        ]

        candidates: list[CandidateChunk] = []

        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)

                for row in cur.fetchall():
                    candidates.append(
                        CandidateChunk(
                            chunk_id=row["chunk_id"],
                            document_id=row["document_id"],
                            document_title=row["document_title"],
                            publication_year=row["publication_year"],
                            section_id=row["section_id"],
                            section_title=row["section_title"],
                            section_path=[],
                            source_text=row["source_text"],
                            page_start=row["page_start"],
                            page_end=row["page_end"],
                            lexical_score=float(row["lexical_score"]),
                            matched_methods={"lexical"},
                        )
                    )

        return candidates

    # ------------------------------------------------------------------
    # Candidate merging
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_candidates(
        destination: dict[str, CandidateChunk],
        incoming: list[CandidateChunk],
    ) -> None:
        """
        Merge candidates from semantic and lexical retrieval.

        A chunk is uniquely identified by chunk_id.
        """
        for candidate in incoming:
            existing = destination.get(candidate.chunk_id)

            if existing is None:
                destination[candidate.chunk_id] = candidate
                continue

            if candidate.semantic_score is not None:
                existing.semantic_score = candidate.semantic_score

            if candidate.lexical_score is not None:
                existing.lexical_score = candidate.lexical_score

            if candidate.citation_score is not None:
                existing.citation_score = candidate.citation_score

            existing.matched_methods.update(candidate.matched_methods)

    # ------------------------------------------------------------------
    # Score fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_final_scores(
        candidates: dict[str, CandidateChunk],
        query: RetrievalQuery,
    ) -> None:
        """
        Normalize each retrieval signal and calculate the weighted score.

        Normalization is performed independently across the candidate pool
        using min-max normalization.
        """
        semantic_values = [
            c.semantic_score
            for c in candidates.values()
            if c.semantic_score is not None
        ]

        lexical_values = [
            c.lexical_score
            for c in candidates.values()
            if c.lexical_score is not None
        ]

        citation_values = [
            c.citation_score
            for c in candidates.values()
            if c.citation_score is not None
        ]

        semantic_norm = RetrievalRepository._normalize_scores(semantic_values)
        lexical_norm = RetrievalRepository._normalize_scores(lexical_values)
        citation_norm = RetrievalRepository._normalize_scores(citation_values)

        semantic_index = 0
        lexical_index = 0
        citation_index = 0

        for candidate in candidates.values():
            score = 0.0

            if candidate.semantic_score is not None:
                score += (
                    query.semantic_weight
                    * semantic_norm[semantic_index]
                )
                semantic_index += 1

            if candidate.lexical_score is not None:
                score += (
                    query.lexical_weight
                    * lexical_norm[lexical_index]
                )
                lexical_index += 1

            if candidate.citation_score is not None:
                score += citation_norm[citation_index]
                citation_index += 1

            candidate.final_score = max(0.0, min(1.0, score))

    @staticmethod
    def _normalize_scores(values: list[Optional[float]]) -> list[float]:
        """
        Min-max normalize scores to [0, 1].

        If all values are equal, every value receives 1.0.
        """
        if not values:
            return []

        clean = [float(v) for v in values if v is not None]

        if not clean:
            return []

        minimum = min(clean)
        maximum = max(clean)

        if maximum == minimum:
            return [1.0 for _ in clean]

        return [
            (value - minimum) / (maximum - minimum)
            for value in clean
        ]

    # ------------------------------------------------------------------
    # Metadata filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metadata_filter(
        query: RetrievalQuery,
    ) -> tuple[str, list]:
        """
        Build safe SQL predicates for metadata filtering.

        The returned SQL fragment always begins with AND predicates and
        contains no user-controlled SQL identifiers.
        """
        clauses: list[str] = []
        params: list = []

        if query.year_from is not None:
            clauses.append("AND d.publication_year >= %s")
            params.append(query.year_from)

        if query.year_to is not None:
            clauses.append("AND d.publication_year <= %s")
            params.append(query.year_to)

        if query.document_id is not None:
            clauses.append("AND d.id = %s")
            params.append(query.document_id)

        if query.doi is not None:
            clauses.append("AND d.doi = %s")
            params.append(query.doi)

        if query.arxiv_id is not None:
            clauses.append("AND d.arxiv_id = %s")
            params.append(query.arxiv_id)

        return "\n".join(clauses), params

    # ------------------------------------------------------------------
    # Diversity
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_document_diversity(
        candidates: list[CandidateChunk],
        max_results_per_document: int,
        limit: int,
    ) -> list[CandidateChunk]:
        """
        Prevent a single document from dominating the final result set.
        """
        counts: dict[str, int] = {}
        selected: list[CandidateChunk] = []

        for candidate in candidates:
            count = counts.get(candidate.document_id, 0)

            if count >= max_results_per_document:
                continue

            selected.append(candidate)
            counts[candidate.document_id] = count + 1

            if len(selected) >= limit:
                break

        return selected

    # ------------------------------------------------------------------
    # Result conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _candidate_to_result(
        candidate: CandidateChunk,
        query: RetrievalQuery,
    ) -> SearchResult:
        if candidate.matched_methods == {"semantic"}:
            method = RetrievalMethod.SEMANTIC
        elif candidate.matched_methods == {"lexical"}:
            method = RetrievalMethod.LEXICAL
        else:
            method = RetrievalMethod.HYBRID

        return SearchResult(
            document_id=candidate.document_id,
            document_title=candidate.document_title,
            publication_year=candidate.publication_year,
            section_id=candidate.section_id,
            section_title=candidate.section_title,
            section_path=candidate.section_path,
            chunk_id=candidate.chunk_id,
            source_text=candidate.source_text,
            page_start=candidate.page_start,
            page_end=candidate.page_end,
            score=candidate.final_score,
            retrieval_method=method,
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def semantic_search(
        self,
        text: str,
        top_k: int = 10,
        **kwargs,
    ) -> SearchResponse:
        """
        Convenience wrapper for semantic-only retrieval.
        """
        query = RetrievalQuery(
            text=text,
            top_k=top_k,
            semantic_weight=1.0,
            lexical_weight=0.0,
            **kwargs,
        )
        return self.search(query)

    def lexical_search(
        self,
        text: str,
        top_k: int = 10,
        **kwargs,
    ) -> SearchResponse:
        """
        Convenience wrapper for lexical-only retrieval.
        """
        query = RetrievalQuery(
            text=text,
            top_k=top_k,
            semantic_weight=0.0,
            lexical_weight=1.0,
            **kwargs,
        )
        return self.search(query)

    def hybrid_search(
        self,
        text: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
        **kwargs,
    ) -> SearchResponse:
        """
        Convenience wrapper for hybrid retrieval.
        """
        query = RetrievalQuery(
            text=text,
            top_k=top_k,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
            **kwargs,
        )
        return self.search(query)
