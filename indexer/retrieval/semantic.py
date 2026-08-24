from __future__ import annotations

import logging
from typing import Optional

from psycopg.rows import dict_row

from indexer.database.connection import DatabaseManager
from indexer.embeddings.base import EmbeddingProvider
from indexer.retrieval.models import CandidateChunk, RetrievalQuery

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Retrieves chunks using pgvector semantic similarity.

    The query text is embedded using the same embedding provider used during
    indexing. Only chunk-level embeddings are searched.
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
        *,
        candidate_k: Optional[int] = None,
    ) -> list[CandidateChunk]:
        """
        Execute semantic retrieval against chunk embeddings.

        Args:
            query: Structured retrieval query.
            candidate_k: Optional override for the candidate pool size.

        Returns:
            Candidate chunks ordered by semantic similarity descending.
        """
        limit = candidate_k or query.candidate_k

        query_vector = self.embedding_provider.embed_single(query.text)

        filters: list[str] = [
            "e.level = 'chunk'",
            "e.chunk_id IS NOT NULL",
        ]
        params: list[object] = [query_vector]

        if query.year_from is not None:
            filters.append("d.publication_year >= %s")
            params.append(query.year_from)

        if query.year_to is not None:
            filters.append("d.publication_year <= %s")
            params.append(query.year_to)

        if query.document_id is not None:
            filters.append("d.id = %s")
            params.append(query.document_id)

        if query.doi is not None:
            filters.append("d.doi = %s")
            params.append(query.doi)

        if query.arxiv_id is not None:
            filters.append("d.arxiv_id = %s")
            params.append(query.arxiv_id)

        where_clause = " AND ".join(filters)

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

                (
                    1.0 - (e.embedding <=> %s::vector)
                ) AS semantic_score

            FROM embeddings e
            JOIN chunks c
                ON c.id = e.chunk_id
            JOIN sections s
                ON s.id = c.section_id
            JOIN documents d
                ON d.id = c.document_id

            WHERE {where_clause}

            ORDER BY e.embedding <=> %s::vector
            LIMIT %s;
        """

        # The vector is needed twice:
        #   1. semantic_score calculation
        #   2. ORDER BY similarity
        execute_params = [
            query_vector,
            *params[1:],
            query_vector,
            limit,
        ]

        # params[0] is already query_vector. The construction above deliberately
        # avoids duplicating it in the filter parameters.
        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, execute_params)
                rows = cur.fetchall()

        results: list[CandidateChunk] = []

        for row in rows:
            score = float(row["semantic_score"])

            # Numerical noise can occasionally push cosine similarity
            # marginally outside the expected range.
            score = max(0.0, min(1.0, score))

            results.append(
                CandidateChunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    document_title=row["document_title"],
                    publication_year=row["publication_year"],
                    section_id=row["section_id"],
                    section_title=row["section_title"],
                    source_text=row["source_text"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    semantic_score=score,
                    matched_methods={"semantic"},
                )
            )

        logger.debug(
            "Semantic retrieval returned %d candidates for query %r",
            len(results),
            query.text,
        )

        return results
