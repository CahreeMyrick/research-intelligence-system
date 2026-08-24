from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import psycopg
from psycopg.rows import dict_row

from ingest.schema import Document as IngestDocument
from indexer.chunking.structural import IndexedSectionNode
from indexer.database.connection import DatabaseManager
from indexer.schemas.chunk import ChunkArtifact
from indexer.schemas.citation import ResolvedCitation, UnresolvedReference
from indexer.schemas.embedding import EmbeddingRecord
from indexer.schemas.processing import DocumentProcessingRecord, ProcessingStatus

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


class IndexRepository:
    """Database repository for persisting and querying index artifacts."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def init_schema(self) -> None:
        """Execute initial schema migration DDL."""
        schema_file = MIGRATIONS_DIR / "init_schema.sql"
        sql = schema_file.read_text(encoding="utf-8")

        with self.db.get_connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(sql)

            conn.commit()

        logger.info("Database schema initialized.")

    # -------------------------------------------------------------------------
    # Processing state
    # -------------------------------------------------------------------------

    def get_document_processing(
        self,
        document_id: str,
    ) -> Optional[DocumentProcessingRecord]:
        """Retrieve processing state for a single document."""
        query = """
            SELECT
                document_id,
                parser_version,
                chunker_version,
                chunker_config_hash,
                embedding_model,
                embedding_model_version,
                embedding_dimensions,
                status,
                last_indexed_at,
                error_message
            FROM document_processing
            WHERE document_id = %s;
        """

        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, (document_id,))
                row = cur.fetchone()

                if row:
                    return DocumentProcessingRecord(**row)

        return None

    def get_all_processing_records(
        self,
    ) -> dict[str, DocumentProcessingRecord]:
        """Retrieve processing records for all documents in the database."""
        query = """
            SELECT
                document_id,
                parser_version,
                chunker_version,
                chunker_config_hash,
                embedding_model,
                embedding_model_version,
                embedding_dimensions,
                status,
                last_indexed_at,
                error_message
            FROM document_processing;
        """

        records: dict[str, DocumentProcessingRecord] = {}

        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query)

                for row in cur.fetchall():
                    records[row["document_id"]] = DocumentProcessingRecord(**row)

        return records

    # -------------------------------------------------------------------------
    # Corpus catalog
    # -------------------------------------------------------------------------

    def get_corpus_catalog_identifiers(self) -> list[dict]:
        """
        Retrieve identifiers of all indexed documents for citation resolution.
        """
        query = """
            SELECT
                id,
                doi,
                arxiv_id,
                title
            FROM documents;
        """

        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query)
                return cur.fetchall()

    # -------------------------------------------------------------------------
    # Atomic indexing persistence
    # -------------------------------------------------------------------------

    def save_document_atomic(
        self,
        document: IngestDocument,
        sections: list[IndexedSectionNode],
        chunks: list[ChunkArtifact],
        embeddings: list[EmbeddingRecord],
        citations: list[ResolvedCitation],
        unresolved_references: list[UnresolvedReference],
        processing_record: DocumentProcessingRecord,
    ) -> None:
        """
        Atomically write document, sections, chunks, embeddings, citations,
        unresolved references, and processing record in a single transaction.
        """
        doc_id = document.identity.id
        source = document.identity.source
        meta = document.metadata

        with self.db.transaction() as conn:
            with conn.cursor() as cur:
                # -----------------------------------------------------------------
                # 1. Upsert document
                # -----------------------------------------------------------------

                cur.execute(
                    """
                    INSERT INTO documents (
                        id,
                        content_hash,
                        filename,
                        source_path,
                        title,
                        abstract,
                        publication_year,
                        venue,
                        doi,
                        arxiv_id,
                        other_identifiers,
                        page_count,
                        parser_version,
                        updated_at
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        NOW()
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        content_hash = EXCLUDED.content_hash,
                        filename = EXCLUDED.filename,
                        source_path = EXCLUDED.source_path,
                        title = EXCLUDED.title,
                        abstract = EXCLUDED.abstract,
                        publication_year = EXCLUDED.publication_year,
                        venue = EXCLUDED.venue,
                        doi = EXCLUDED.doi,
                        arxiv_id = EXCLUDED.arxiv_id,
                        other_identifiers = EXCLUDED.other_identifiers,
                        page_count = EXCLUDED.page_count,
                        parser_version = EXCLUDED.parser_version,
                        updated_at = NOW();
                    """,
                    (
                        doc_id,
                        source.content_hash,
                        source.filename,
                        source.path,
                        meta.title,
                        meta.abstract,
                        meta.publication_year,
                        meta.venue,
                        meta.doi,
                        meta.arxiv_id,
                        json.dumps(meta.other_identifiers),
                        document.page_count,
                        processing_record.parser_version,
                    ),
                )

                # -----------------------------------------------------------------
                # Delete dependent records before rebuilding the document index.
                #
                # This is intentional for now. The indexing layer treats a
                # document as the unit of replacement when its processing state
                # changes.
                # -----------------------------------------------------------------

                cur.execute(
                    """
                    DELETE FROM citations
                    WHERE source_document_id = %s;
                    """,
                    (doc_id,),
                )

                cur.execute(
                    """
                    DELETE FROM unresolved_references
                    WHERE source_document_id = %s;
                    """,
                    (doc_id,),
                )

                cur.execute(
                    """
                    DELETE FROM embeddings
                    WHERE document_id = %s
                       OR section_id IN (
                           SELECT id
                           FROM sections
                           WHERE document_id = %s
                       )
                       OR chunk_id IN (
                           SELECT id
                           FROM chunks
                           WHERE document_id = %s
                       );
                    """,
                    (doc_id, doc_id, doc_id),
                )
                cur.execute(
                    """
                    DELETE FROM chunks
                    WHERE document_id = %s;
                    """,
                    (doc_id,),
                )

                cur.execute(
                    """
                    DELETE FROM sections
                    WHERE document_id = %s;
                    """,
                    (doc_id,),
                )

                # -----------------------------------------------------------------
                # 2. Insert sections in top-down order.
                #
                # Parent sections must exist before child sections because
                # parent_section_id is a self-referencing foreign key.
                # -----------------------------------------------------------------

                for sec in sections:
                    cur.execute(
                        """
                        INSERT INTO sections (
                            id,
                            document_id,
                            parent_section_id,
                            title,
                            level,
                            page_start,
                            page_end,
                            section_order,
                            updated_at
                        )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            NOW()
                        );
                        """,
                        (
                            sec.id,
                            sec.document_id,
                            sec.parent_section_id,
                            sec.title,
                            sec.level,
                            sec.page_start,
                            sec.page_end,
                            sec.section_order,
                        ),
                    )

                # -----------------------------------------------------------------
                # 3. Insert chunks
                # -----------------------------------------------------------------

                for chunk in chunks:
                    cur.execute(
                        """
                        INSERT INTO chunks (
                            id,
                            document_id,
                            section_id,
                            chunk_index,
                            source_text,
                            embedding_text,
                            page_start,
                            page_end,
                            token_count,
                            chunker_version,
                            chunker_config_hash,
                            updated_at
                        )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            NOW()
                        );
                        """,
                        (
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.section_id,
                            chunk.chunk_index,
                            chunk.source_text,
                            chunk.embedding_text,
                            chunk.page_start,
                            chunk.page_end,
                            chunk.token_count,
                            chunk.chunker_version,
                            chunk.chunker_config_hash,
                        ),
                    )

                # -----------------------------------------------------------------
                # 4. Insert embeddings
                # -----------------------------------------------------------------

                for emb in embeddings:
                    cur.execute(
                        """
                        INSERT INTO embeddings (
                            id,
                            document_id,
                            section_id,
                            chunk_id,
                            level,
                            model_name,
                            model_version,
                            dimensions,
                            embedding
                        )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s
                        );
                        """,
                        (
                            emb.id,
                            emb.document_id,
                            emb.section_id,
                            emb.chunk_id,
                            (
                                emb.level.value
                                if hasattr(emb.level, "value")
                                else str(emb.level)
                            ),
                            emb.model_name,
                            emb.model_version,
                            emb.dimensions,
                            emb.embedding,
                        ),
                    )

                # -----------------------------------------------------------------
                # 5. Insert resolved citations
                # -----------------------------------------------------------------

                for cit in citations:
                    cur.execute(
                        """
                        INSERT INTO citations (
                            id,
                            source_document_id,
                            target_document_id,
                            reference_id,
                            raw_text,
                            doi,
                            arxiv_id,
                            resolution_method,
                            resolution_confidence,
                            updated_at
                        )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            NOW()
                        );
                        """,
                        (
                            cit.id,
                            cit.source_document_id,
                            cit.target_document_id,
                            cit.reference_id,
                            cit.raw_text,
                            cit.doi,
                            cit.arxiv_id,
                            cit.resolution_method,
                            cit.resolution_confidence,
                        ),
                    )

                # -----------------------------------------------------------------
                # 6. Insert unresolved references
                # -----------------------------------------------------------------

                for unres in unresolved_references:
                    cur.execute(
                        """
                        INSERT INTO unresolved_references (
                            id,
                            source_document_id,
                            reference_id,
                            raw_text,
                            doi,
                            arxiv_id,
                            title,
                            authors,
                            year,
                            updated_at
                        )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            NOW()
                        );
                        """,
                        (
                            unres.id,
                            unres.source_document_id,
                            unres.reference_id,
                            unres.raw_text,
                            unres.doi,
                            unres.arxiv_id,
                            unres.title,
                            json.dumps(unres.authors),
                            unres.year,
                        ),
                    )

                # -----------------------------------------------------------------
                # 7. Upsert document processing state
                # -----------------------------------------------------------------

                cur.execute(
                    """
                    INSERT INTO document_processing (
                        document_id,
                        parser_version,
                        chunker_version,
                        chunker_config_hash,
                        embedding_model,
                        embedding_model_version,
                        embedding_dimensions,
                        status,
                        last_indexed_at,
                        error_message
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        NOW(),
                        NULL
                    )
                    ON CONFLICT (document_id) DO UPDATE SET
                        parser_version = EXCLUDED.parser_version,
                        chunker_version = EXCLUDED.chunker_version,
                        chunker_config_hash = EXCLUDED.chunker_config_hash,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_model_version = EXCLUDED.embedding_model_version,
                        embedding_dimensions = EXCLUDED.embedding_dimensions,
                        status = 'completed',
                        last_indexed_at = NOW(),
                        error_message = NULL;
                    """,
                    (
                        doc_id,
                        processing_record.parser_version,
                        processing_record.chunker_version,
                        processing_record.chunker_config_hash,
                        processing_record.embedding_model,
                        processing_record.embedding_model_version,
                        processing_record.embedding_dimensions,
                        ProcessingStatus.COMPLETED.value,
                    ),
                )

    # -------------------------------------------------------------------------
    # Failure handling
    # -------------------------------------------------------------------------

    def record_document_failure(
        self,
        document_id: str,
        error_message: str,
        parser_version: Optional[str] = None,
        chunker_version: Optional[str] = None,
        chunker_config_hash: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_model_version: Optional[str] = None,
        embedding_dimensions: Optional[int] = None,
    ) -> None:
        """Record processing failure for a document."""

        with self.db.transaction() as conn:
            with conn.cursor() as cur:
                # Ensure placeholder in documents table so the FK constraint
                # on document_processing succeeds.
                cur.execute(
                    """
                    INSERT INTO documents (
                        id,
                        content_hash,
                        filename,
                        source_path,
                        updated_at
                    )
                    VALUES (
                        %s,
                        '',
                        %s,
                        '',
                        NOW()
                    )
                    ON CONFLICT (id) DO NOTHING;
                    """,
                    (
                        document_id,
                        f"{document_id}.pdf",
                    ),
                )

                cur.execute(
                    """
                    INSERT INTO document_processing (
                        document_id,
                        parser_version,
                        chunker_version,
                        chunker_config_hash,
                        embedding_model,
                        embedding_model_version,
                        embedding_dimensions,
                        status,
                        last_indexed_at,
                        error_message
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        'failed',
                        NOW(),
                        %s
                    )
                    ON CONFLICT (document_id) DO UPDATE SET
                        parser_version = EXCLUDED.parser_version,
                        chunker_version = EXCLUDED.chunker_version,
                        chunker_config_hash = EXCLUDED.chunker_config_hash,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_model_version = EXCLUDED.embedding_model_version,
                        embedding_dimensions = EXCLUDED.embedding_dimensions,
                        status = 'failed',
                        last_indexed_at = NOW(),
                        error_message = EXCLUDED.error_message;
                    """,
                    (
                        document_id,
                        parser_version,
                        chunker_version,
                        chunker_config_hash,
                        embedding_model,
                        embedding_model_version,
                        embedding_dimensions,
                        error_message,
                    ),
                )

    # -------------------------------------------------------------------------
    # Citation persistence
    # -------------------------------------------------------------------------

    def update_citations_atomic(
        self,
        document_id: str,
        citations: list[ResolvedCitation],
        unresolved_references: list[UnresolvedReference],
    ) -> None:
        """Update only citations and unresolved references for a document."""

        with self.db.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM citations
                    WHERE source_document_id = %s;
                    """,
                    (document_id,),
                )

                cur.execute(
                    """
                    DELETE FROM unresolved_references
                    WHERE source_document_id = %s;
                    """,
                    (document_id,),
                )

                for cit in citations:
                    cur.execute(
                        """
                        INSERT INTO citations (
                            id,
                            source_document_id,
                            target_document_id,
                            reference_id,
                            raw_text,
                            doi,
                            arxiv_id,
                            resolution_method,
                            resolution_confidence,
                            updated_at
                        )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            NOW()
                        );
                        """,
                        (
                            cit.id,
                            cit.source_document_id,
                            cit.target_document_id,
                            cit.reference_id,
                            cit.raw_text,
                            cit.doi,
                            cit.arxiv_id,
                            cit.resolution_method,
                            cit.resolution_confidence,
                        ),
                    )

                for unres in unresolved_references:
                    cur.execute(
                        """
                        INSERT INTO unresolved_references (
                            id,
                            source_document_id,
                            reference_id,
                            raw_text,
                            doi,
                            arxiv_id,
                            title,
                            authors,
                            year,
                            updated_at
                        )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            NOW()
                        );
                        """,
                        (
                            unres.id,
                            unres.source_document_id,
                            unres.reference_id,
                            unres.raw_text,
                            unres.doi,
                            unres.arxiv_id,
                            unres.title,
                            json.dumps(unres.authors),
                            unres.year,
                        ),
                    )

    # -------------------------------------------------------------------------
    # Retrieval: semantic
    # -------------------------------------------------------------------------

    def semantic_search_chunks(
        self,
        query_embedding: list[float],
        candidate_k: int = 50,
        *,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        document_id: Optional[str] = None,
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve chunk candidates using pgvector cosine distance.

        The returned semantic_score is:

            1 - cosine_distance

        Higher is therefore better.

        Chunks are the primary evidence units returned to the answering layer.
        """

        filters: list[str] = []

        # pgvector embedding is required twice:
        #
        #   1. once to calculate the semantic score
        #   2. once to order by vector distance
        #
        # Keep these parameters first, followed by metadata filters and LIMIT.
        params: list = [
            query_embedding,
            query_embedding,
        ]

        if year_from is not None:
            filters.append("d.publication_year >= %s")
            params.append(year_from)

        if year_to is not None:
            filters.append("d.publication_year <= %s")
            params.append(year_to)

        if document_id is not None:
            filters.append("d.id = %s")
            params.append(document_id)

        if doi is not None:
            filters.append("d.doi = %s")
            params.append(doi)

        if arxiv_id is not None:
            filters.append("d.arxiv_id = %s")
            params.append(arxiv_id)

        where_clause = ""

        if filters:
            where_clause = "AND " + " AND ".join(filters)

        params.append(candidate_k)

        query = f"""
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

                1 - (e.embedding <=> %s) AS semantic_score

            FROM embeddings e

            JOIN chunks c
                ON c.id = e.chunk_id

            JOIN documents d
                ON d.id = c.document_id

            JOIN sections s
                ON s.id = c.section_id

            WHERE e.level = 'chunk'
              AND e.embedding IS NOT NULL
              {where_clause}

            ORDER BY e.embedding <=> %s

            LIMIT %s;
        """

        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)
                return cur.fetchall()

    # -------------------------------------------------------------------------
    # Retrieval: lexical
    # -------------------------------------------------------------------------

    def lexical_search_chunks(
        self,
        query_text: str,
        candidate_k: int = 50,
        *,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        document_id: Optional[str] = None,
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve chunk candidates using PostgreSQL full-text search.

        embedding_text is intentionally searched rather than source_text.

        The indexing layer enriches embedding_text with document/section
        context, so lexical retrieval benefits from that same representation.
        """

        filters: list[str] = []

        # Query text appears twice:
        #
        #   1. ranking
        #   2. matching
        #
        params: list = [
            query_text,
            query_text,
        ]

        if year_from is not None:
            filters.append("d.publication_year >= %s")
            params.append(year_from)

        if year_to is not None:
            filters.append("d.publication_year <= %s")
            params.append(year_to)

        if document_id is not None:
            filters.append("d.id = %s")
            params.append(document_id)

        if doi is not None:
            filters.append("d.doi = %s")
            params.append(doi)

        if arxiv_id is not None:
            filters.append("d.arxiv_id = %s")
            params.append(arxiv_id)

        where_clause = ""

        if filters:
            where_clause = "AND " + " AND ".join(filters)

        params.append(candidate_k)

        query = f"""
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
                    to_tsvector('english', c.embedding_text),
                    websearch_to_tsquery('english', %s)
                ) AS lexical_score

            FROM chunks c

            JOIN documents d
                ON d.id = c.document_id

            JOIN sections s
                ON s.id = c.section_id

            WHERE to_tsvector('english', c.embedding_text)
                  @@ websearch_to_tsquery('english', %s)

              {where_clause}

            ORDER BY lexical_score DESC

            LIMIT %s;
        """

        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)
                return cur.fetchall()

    # -------------------------------------------------------------------------
    # Retrieval: section hierarchy
    # -------------------------------------------------------------------------

    def get_chunk_section_path(
        self,
        section_id: str,
    ) -> list[str]:
        """
        Return the hierarchical section path for a section.

        Example:

            ["3 ViTGAN", "3.1 Discriminator"]

        The recursive traversal walks from the selected section toward its
        ancestors and then reverses the result into document order.
        """

        query = """
            WITH RECURSIVE section_tree AS (
                SELECT
                    id,
                    parent_section_id,
                    title,
                    0 AS depth
                FROM sections
                WHERE id = %s

                UNION ALL

                SELECT
                    parent.id,
                    parent.parent_section_id,
                    parent.title,
                    child.depth + 1
                FROM sections parent
                JOIN section_tree child
                    ON child.parent_section_id = parent.id
            )

            SELECT title
            FROM section_tree
            ORDER BY depth DESC;
        """

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (section_id,))
                return [row[0] for row in cur.fetchall()]

    # -------------------------------------------------------------------------
    # Retrieval: citation graph
    # -------------------------------------------------------------------------

    def get_citation_neighbors(
        self,
        document_id: str,
        *,
        direction: str = "both",
        depth: int = 1,
        max_candidates: int = 25,
    ) -> list[dict]:
        """
        Retrieve documents connected to a document through resolved citations.

        direction:
            outgoing = documents cited by the source document
            incoming = documents that cite the source document
            both     = both directions

        depth:
            Maximum traversal depth. Currently limited to 2.

        Important:
            This uses only explicit resolved citations. Semantic similarity
            and future inferred relationships are intentionally separate from
            this graph.
        """

        if direction not in {"outgoing", "incoming", "both"}:
            raise ValueError(
                "direction must be 'outgoing', 'incoming', or 'both'"
            )

        if depth < 1:
            raise ValueError("depth must be >= 1")

        depth = min(depth, 2)

        # Build the directional edge predicate dynamically.
        if direction == "outgoing":
            edge_sql = """
                SELECT
                    source_document_id AS from_document_id,
                    target_document_id AS to_document_id
                FROM citations
                WHERE target_document_id IS NOT NULL
            """

        elif direction == "incoming":
            edge_sql = """
                SELECT
                    target_document_id AS from_document_id,
                    source_document_id AS to_document_id
                FROM citations
                WHERE target_document_id IS NOT NULL
            """

        else:
            edge_sql = """
                SELECT
                    source_document_id AS from_document_id,
                    target_document_id AS to_document_id
                FROM citations
                WHERE target_document_id IS NOT NULL

                UNION ALL

                SELECT
                    target_document_id AS from_document_id,
                    source_document_id AS to_document_id
                FROM citations
                WHERE target_document_id IS NOT NULL
            """

        query = f"""
            WITH RECURSIVE citation_graph AS (
                -- Direct neighbors.
                SELECT
                    to_document_id AS document_id,
                    1 AS traversal_depth
                FROM (
                    {edge_sql}
                ) edges
                WHERE from_document_id = %s

                UNION

                -- Optional second hop.
                SELECT
                    edges.to_document_id AS document_id,
                    graph.traversal_depth + 1 AS traversal_depth
                FROM citation_graph graph
                JOIN (
                    {edge_sql}
                ) edges
                    ON edges.from_document_id = graph.document_id
                WHERE graph.traversal_depth < %s
            )

            SELECT
                graph.document_id,
                d.title AS document_title,
                d.publication_year,
                MIN(graph.traversal_depth) AS traversal_depth
            FROM citation_graph graph

            JOIN documents d
                ON d.id = graph.document_id

            WHERE graph.document_id <> %s

            GROUP BY
                graph.document_id,
                d.title,
                d.publication_year

            ORDER BY
                MIN(graph.traversal_depth),
                d.publication_year NULLS LAST,
                d.title

            LIMIT %s;
        """

        params = [
            document_id,
            depth,
            document_id,
            max_candidates,
        ]

        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)
                return cur.fetchall()

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Compute summary statistics of the database index."""

        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM documents;
                    """
                )
                docs = cur.fetchone()["count"]

                cur.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM sections;
                    """
                )
                secs = cur.fetchone()["count"]

                cur.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM chunks;
                    """
                )
                chunks = cur.fetchone()["count"]

                cur.execute(
                    """
                    SELECT
                        level,
                        COUNT(*) AS count
                    FROM embeddings
                    GROUP BY level;
                    """
                )
                embs = {
                    row["level"]: row["count"]
                    for row in cur.fetchall()
                }

                cur.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM citations
                    WHERE target_document_id IS NOT NULL;
                    """
                )
                cits = cur.fetchone()["count"]

                cur.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM unresolved_references;
                    """
                )
                unres = cur.fetchone()["count"]

                cur.execute(
                    """
                    SELECT
                        status,
                        COUNT(*) AS count
                    FROM document_processing
                    GROUP BY status;
                    """
                )
                statuses = {
                    row["status"]: row["count"]
                    for row in cur.fetchall()
                }

                return {
                    "documents": docs,
                    "sections": secs,
                    "chunks": chunks,
                    "embeddings": embs,
                    "citations": cits,
                    "unresolved_references": unres,
                    "statuses": statuses,
                }
