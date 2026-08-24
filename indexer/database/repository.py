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

    def get_document_processing(self, document_id: str) -> Optional[DocumentProcessingRecord]:
        """Retrieve processing state for a single document."""
        query = """
            SELECT document_id, parser_version, chunker_version, chunker_config_hash,
                   embedding_model, embedding_model_version, embedding_dimensions,
                   status, last_indexed_at, error_message
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

    def get_all_processing_records(self) -> dict[str, DocumentProcessingRecord]:
        """Retrieve processing records for all documents in the database."""
        query = """
            SELECT document_id, parser_version, chunker_version, chunker_config_hash,
                   embedding_model, embedding_model_version, embedding_dimensions,
                   status, last_indexed_at, error_message
            FROM document_processing;
        """
        records = {}
        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query)
                for row in cur.fetchall():
                    records[row["document_id"]] = DocumentProcessingRecord(**row)
        return records

    def get_corpus_catalog_identifiers(self) -> list[dict]:
        """Retrieve identifiers of all indexed documents for citation resolution."""
        query = "SELECT id, doi, arxiv_id, title FROM documents;"
        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query)
                return cur.fetchall()

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
        unresolved_references, and processing record in a single transaction.
        """
        doc_id = document.identity.id
        source = document.identity.source
        meta = document.metadata

        with self.db.transaction() as conn:
            with conn.cursor() as cur:
                # 1. Upsert document
                cur.execute(
                    """
                    INSERT INTO documents (
                        id, content_hash, filename, source_path, title, abstract,
                        publication_year, venue, doi, arxiv_id, other_identifiers,
                        page_count, parser_version, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
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

                # Delete dependent records prior to re-inserting
                cur.execute("DELETE FROM citations WHERE source_document_id = %s;", (doc_id,))
                cur.execute("DELETE FROM unresolved_references WHERE source_document_id = %s;", (doc_id,))
                cur.execute("DELETE FROM embeddings WHERE document_id = %s;", (doc_id,))
                cur.execute("DELETE FROM chunks WHERE document_id = %s;", (doc_id,))
                cur.execute("DELETE FROM sections WHERE document_id = %s;", (doc_id,))

                # 2. Insert sections in top-down order (parents before children)
                for sec in sections:
                    cur.execute(
                        """
                        INSERT INTO sections (
                            id, document_id, parent_section_id, title, level,
                            page_start, page_end, section_order, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW());
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

                # 3. Insert chunks
                for chunk in chunks:
                    cur.execute(
                        """
                        INSERT INTO chunks (
                            id, document_id, section_id, chunk_index, source_text,
                            embedding_text, page_start, page_end, token_count,
                            chunker_version, chunker_config_hash, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW());
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

                # 4. Insert embeddings
                for emb in embeddings:
                    cur.execute(
                        """
                        INSERT INTO embeddings (
                            id, document_id, section_id, chunk_id, level,
                            model_name, model_version, dimensions, embedding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                        """,
                        (
                            emb.id,
                            emb.document_id,
                            emb.section_id,
                            emb.chunk_id,
                            emb.level.value if hasattr(emb.level, "value") else str(emb.level),
                            emb.model_name,
                            emb.model_version,
                            emb.dimensions,
                            emb.embedding,
                        ),
                    )

                # 5. Insert citations
                for cit in citations:
                    cur.execute(
                        """
                        INSERT INTO citations (
                            id, source_document_id, target_document_id, reference_id,
                            raw_text, doi, arxiv_id, resolution_method,
                            resolution_confidence, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW());
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

                # 6. Insert unresolved references
                for unres in unresolved_references:
                    cur.execute(
                        """
                        INSERT INTO unresolved_references (
                            id, source_document_id, reference_id, raw_text,
                            doi, arxiv_id, title, authors, year, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW());
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

                # 7. Upsert document_processing state
                cur.execute(
                    """
                    INSERT INTO document_processing (
                        document_id, parser_version, chunker_version, chunker_config_hash,
                        embedding_model, embedding_model_version, embedding_dimensions,
                        status, last_indexed_at, error_message
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NULL
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
        """Record processing failure for a document in document_processing table."""
        with self.db.transaction() as conn:
            with conn.cursor() as cur:
                # Ensure placeholder in documents table if not existing so FK constraint succeeds
                cur.execute(
                    """
                    INSERT INTO documents (id, content_hash, filename, source_path, updated_at)
                    VALUES (%s, '', %s, '', NOW())
                    ON CONFLICT (id) DO NOTHING;
                    """,
                    (document_id, f"{document_id}.pdf"),
                )
                cur.execute(
                    """
                    INSERT INTO document_processing (
                        document_id, parser_version, chunker_version, chunker_config_hash,
                        embedding_model, embedding_model_version, embedding_dimensions,
                        status, last_indexed_at, error_message
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, 'failed', NOW(), %s
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

    def update_citations_atomic(
        self,
        document_id: str,
        citations: list[ResolvedCitation],
        unresolved_references: list[UnresolvedReference],
    ) -> None:
        """Update only citations and unresolved references for a document."""
        with self.db.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM citations WHERE source_document_id = %s;", (document_id,))
                cur.execute("DELETE FROM unresolved_references WHERE source_document_id = %s;", (document_id,))

                for cit in citations:
                    cur.execute(
                        """
                        INSERT INTO citations (
                            id, source_document_id, target_document_id, reference_id,
                            raw_text, doi, arxiv_id, resolution_method,
                            resolution_confidence, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW());
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
                            id, source_document_id, reference_id, raw_text,
                            doi, arxiv_id, title, authors, year, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW());
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

    def get_stats(self) -> dict:
        """Compute summary statistics of the database index."""
        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT COUNT(*) as count FROM documents;")
                docs = cur.fetchone()["count"]
                
                cur.execute("SELECT COUNT(*) as count FROM sections;")
                secs = cur.fetchone()["count"]
                
                cur.execute("SELECT COUNT(*) as count FROM chunks;")
                chunks = cur.fetchone()["count"]
                
                cur.execute(
                    "SELECT level, COUNT(*) as count FROM embeddings GROUP BY level;"
                )
                embs = {r["level"]: r["count"] for r in cur.fetchall()}
                
                cur.execute("SELECT COUNT(*) as count FROM citations WHERE target_document_id IS NOT NULL;")
                cits = cur.fetchone()["count"]
                
                cur.execute("SELECT COUNT(*) as count FROM unresolved_references;")
                unres = cur.fetchone()["count"]
                
                cur.execute(
                    "SELECT status, COUNT(*) as count FROM document_processing GROUP BY status;"
                )
                statuses = {r["status"]: r["count"] for r in cur.fetchall()}
                
                return {
                    "documents": docs,
                    "sections": secs,
                    "chunks": chunks,
                    "embeddings": embs,
                    "citations": cits,
                    "unresolved_references": unres,
                    "statuses": statuses,
                }
