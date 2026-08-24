from __future__ import annotations

import logging
import time
from typing import Optional

import psycopg
from psycopg.rows import dict_row

from indexer.database.connection import DatabaseManager
from indexer.retrieval.models import (
    CandidateChunk,
    RetrievalMethod,
    RetrievalQuery,
    SearchResponse,
    SearchResult,
)

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Hybrid retrieval engine over the PostgreSQL + pgvector index.

    Retrieval flow:

        query
          |
          +--> semantic search
          |
          +--> lexical search
          |
          v
        candidate fusion
          |
          v
        diversity filtering
          |
          v
        SearchResponse

    The engine deliberately returns grounded chunk text and structural
    provenance. An LLM can consume SearchResponse later without needing
    direct database access.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_provider,
    ):
        self.db = db_manager
        self.embedding_provider = embedding_provider

    def search(self, query: RetrievalQuery) -> SearchResponse:
        """
        Execute hybrid retrieval according to the supplied query.

        Semantic and lexical retrieval are performed independently, then
        candidates are fused using the configured weights.
        """
        start = time.time()

        candidates: dict[str, CandidateChunk] = {}

        semantic_candidates = self._semantic_search(query)
        lexical_candidates = self._lexical_search(query)

        for candidate in semantic_candidates:
            candidates[candidate.chunk_id] = candidate

        for candidate in lexical_candidates:
            existing = candidates.get(candidate.chunk_id)

            if existing is None:
                candidates[candidate.chunk_id] = candidate
            else:
                existing.lexical_score = candidate.lexical_score
                existing.matched_methods.add(RetrievalMethod.LEXICAL.value)

        self._fuse_scores(
            candidates,
            semantic_weight=query.semantic_weight,
            lexical_weight=query.lexical_weight,
        )

        ranked = sorted(
            candidates.values(),
            key=lambda candidate: candidate.final_score,
            reverse=True,
        )

        ranked = self._apply_document_diversity(
            ranked,
            max_results_per_document=query.max_results_per_document,
        )

        ranked = ranked[: query.top_k]

        results = [
            SearchResult(
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
                retrieval_method=self._determine_result_method(candidate),
            )
            for candidate in ranked
        ]

        return SearchResponse(
            original_query=query.text,
            candidate_count=len(candidates),
            returned_count=len(results),
            duration_seconds=time.time() - start,
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
        Retrieve chunks using pgvector cosine distance.

        The embedding provider is responsible for producing the query vector
        using the same model family used during indexing.
        """
        query_embedding = self.embedding_provider.embed_single(query.text)

        where_sql, params = self._build_filters(query)

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

                1.0 - (e.embedding <=> %s::vector) AS semantic_score

            FROM chunks c

            JOIN documents d
                ON d.id = c.document_id

            JOIN sections s
                ON s.id = c.section_id

            JOIN embeddings e
                ON e.chunk_id = c.id

            WHERE e.level = 'chunk'
              AND e.model_name = %s
              AND e.dimensions = %s
              {where_sql}

            ORDER BY e.embedding <=> %s::vector
            LIMIT %s;
        """

        final_params = [
            query_embedding,
            self.embedding_provider.name,
            self.embedding_provider.dimensions,
            *params,
            query.candidate_k,
        ]

        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, final_params)
                rows = cur.fetchall()

        return [
            self._row_to_candidate(
                row,
                semantic_score=float(row["semantic_score"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Lexical retrieval
    # ------------------------------------------------------------------

    def _lexical_search(
        self,
        query: RetrievalQuery,
    ) -> list[CandidateChunk]:
        """
        Retrieve chunks using PostgreSQL full-text search.

        websearch_to_tsquery is used so ordinary user questions such as

            "Google distributed systems architecture"

        can be passed directly into PostgreSQL's text search machinery.
        """
        where_sql, params = self._build_filters(query)

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

        final_params = [
            query.text,
            query.text,
            *params,
            query.candidate_k,
        ]

        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, final_params)
                rows = cur.fetchall()

        return [
            self._row_to_candidate(
                row,
                lexical_score=float(row["lexical_score"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _build_filters(
        self,
        query: RetrievalQuery,
    ) -> tuple[str, list[object]]:
        """
        Build SQL predicates for RetrievalQuery metadata filters.

        Returns a SQL fragment beginning with AND predicates plus their
        parameter values.
        """
        predicates: list[str] = []
        params: list[object] = []

        if query.year_from is not None:
            predicates.append("AND d.publication_year >= %s")
            params.append(query.year_from)

        if query.year_to is not None:
            predicates.append("AND d.publication_year <= %s")
            params.append(query.year_to)

        if query.document_id is not None:
            predicates.append("AND d.id = %s")
            params.append(query.document_id)

        if query.doi is not None:
            predicates.append("AND d.doi = %s")
            params.append(query.doi)

        if query.arxiv_id is not None:
            predicates.append("AND d.arxiv_id = %s")
            params.append(query.arxiv_id)

        return "\n".join(predicates), params

    # ------------------------------------------------------------------
    # Score fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _fuse_scores(
        candidates: dict[str, CandidateChunk],
        semantic_weight: float,
        lexical_weight: float,
    ) -> None:
        """
        Normalize semantic and lexical scores independently, then fuse them.

        Normalization is min-max over the candidate pool. This avoids raw
        pgvector cosine scores and PostgreSQL ts_rank scores being treated as
        directly comparable quantities.
        """
        semantic_scores = [
            candidate.semantic_score
            for candidate in candidates.values()
            if candidate.semantic_score is not None
        ]

        lexical_scores = [
            candidate.lexical_score
            for candidate in candidates.values()
            if candidate.lexical_score is not None
        ]

        semantic_min = min(semantic_scores) if semantic_scores else 0.0
        semantic_max = max(semantic_scores) if semantic_scores else 0.0

        lexical_min = min(lexical_scores) if lexical_scores else 0.0
        lexical_max = max(lexical_scores) if lexical_scores else 0.0

        for candidate in candidates.values():
            semantic_normalized = RetrievalEngine._normalize(
                candidate.semantic_score,
                semantic_min,
                semantic_max,
            )

            lexical_normalized = RetrievalEngine._normalize(
                candidate.lexical_score,
                lexical_min,
                lexical_max,
            )

            candidate.final_score = (
                semantic_weight * semantic_normalized
                + lexical_weight * lexical_normalized
            )

    @staticmethod
    def _normalize(
        value: Optional[float],
        minimum: float,
        maximum: float,
    ) -> float:
        if value is None:
            return 0.0

        if maximum <= minimum:
            return 1.0 if value > 0 else 0.0

        normalized = (value - minimum) / (maximum - minimum)

        return max(0.0, min(1.0, normalized))

    # ------------------------------------------------------------------
    # Diversity
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_document_diversity(
        candidates: list[CandidateChunk],
        max_results_per_document: int,
    ) -> list[CandidateChunk]:
        """
        Prevent one paper from dominating the result set.
        """
        counts: dict[str, int] = {}
        results: list[CandidateChunk] = []

        for candidate in candidates:
            count = counts.get(candidate.document_id, 0)

            if count >= max_results_per_document:
                continue

            results.append(candidate)
            counts[candidate.document_id] = count + 1

        return results

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_candidate(
        row: dict,
        semantic_score: Optional[float] = None,
        lexical_score: Optional[float] = None,
    ) -> CandidateChunk:
        """
        Convert a database row into the internal retrieval representation.
        """
        candidate = CandidateChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            document_title=row.get("document_title"),
            publication_year=row.get("publication_year"),
            section_id=row["section_id"],
            section_title=row["section_title"],
            source_text=row["source_text"],
            page_start=row.get("page_start"),
            page_end=row.get("page_end"),
            semantic_score=semantic_score,
            lexical_score=lexical_score,
        )

        if semantic_score is not None:
            candidate.matched_methods.add(RetrievalMethod.SEMANTIC.value)

        if lexical_score is not None:
            candidate.matched_methods.add(RetrievalMethod.LEXICAL.value)

        return candidate

    @staticmethod
    def _determine_result_method(
        candidate: CandidateChunk,
    ) -> RetrievalMethod:
        """
        Determine which retrieval mechanism contributed to the result.
        """
        methods = candidate.matched_methods

        if (
            RetrievalMethod.SEMANTIC.value in methods
            and RetrievalMethod.LEXICAL.value in methods
        ):
            return RetrievalMethod.HYBRID

        if RetrievalMethod.SEMANTIC.value in methods:
            return RetrievalMethod.SEMANTIC

        return RetrievalMethod.LEXICAL
