from __future__ import annotations

import logging
from typing import Optional

from psycopg.rows import dict_row

from indexer.database.connection import DatabaseManager
from indexer.retrieval.models import CandidateChunk, RetrievalQuery

logger = logging.getLogger(__name__)


class LexicalRetriever:
    """
    Retrieves chunks using PostgreSQL full-text search.

    The user's query is converted into a PostgreSQL tsquery and matched
    against chunk source text. Ranking uses PostgreSQL ts_rank_cd.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def search(
        self,
        query: RetrievalQuery,
        *,
        candidate_k: Optional[int] = None,
    ) -> list[CandidateChunk]:
        """
        Execute lexical retrieval against chunk source text.

        Args:
            query: Structured retrieval query.
            candidate_k: Optional override for candidate pool size.

        Returns:
            Candidate chunks ordered by lexical relevance descending.
        """
        limit = candidate_k or query.candidate_k

        filters: list[str] = [
            """
            to_tsvector(
                'english',
                coalesce(c.source_text, '')
            ) @@ websearch_to_tsquery(
                'english',
                %s
            )
            """
        ]

        params: list[object] = [query.text]

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

                ts_rank_cd(
                    to_tsvector(
                        'english',
                        coalesce(c.source_text, '')
                    ),
                    websearch_to_tsquery(
                        'english',
                        %s
                    )
                ) AS lexical_score

            FROM chunks c
            JOIN sections s
                ON s.id = c.section_id
            JOIN documents d
                ON d.id = c.document_id

            WHERE {where_clause}

            ORDER BY lexical_score DESC
            LIMIT %s;
        """

        # Query text is required once for ranking and once for the WHERE clause.
        execute_params = [
            query.text,
            *params,
            limit,
        ]

        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, execute_params)
                rows = cur.fetchall()

        results: list[CandidateChunk] = []

        for row in rows:
            score = float(row["lexical_score"])

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
                    lexical_score=score,
                    matched_methods={"lexical"},
                )
            )

        logger.debug(
            "Lexical retrieval returned %d candidates for query %r",
            len(results),
            query.text,
        )

        return results
