from __future__ import annotations

import pytest
from ingest.schema import Document, DocumentIdentity, DocumentMetadata, DocumentSource, Paragraph, Section
from indexer.chunking.structural import StructuralSemanticChunker
from indexer.database.connection import DatabaseManager
from indexer.database.repository import IndexRepository
from indexer.embeddings.mock import MockEmbeddingProvider
from indexer.schemas.citation import ResolvedCitation, UnresolvedReference
from indexer.schemas.embedding import EmbeddingLevel, EmbeddingRecord
from indexer.schemas.processing import DocumentProcessingRecord, ProcessingStatus


def test_schema_init_and_stats(repo):
    stats = repo.get_stats()
    assert "documents" in stats
    assert "sections" in stats
    assert "chunks" in stats
    assert "embeddings" in stats


def test_atomic_document_save_and_rollback(repo):
    doc_id = "doc_test_db_001"
    doc = Document(
        identity=DocumentIdentity(
            id=doc_id,
            source=DocumentSource(
                filename="paper1.pdf",
                path="/test/paper1.pdf",
                content_hash="hash_paper1",
            ),
        ),
        metadata=DocumentMetadata(
            title="Database Test Paper",
            authors=["Tester"],
            publication_year=2025,
        ),
        sections=[
            Section(
                title="1 Introduction",
                level=1,
                paragraphs=[Paragraph(text="Intro database content.", page=0, position=1)],
                subsections=[],
            )
        ],
        references=[],
        page_count=1,
        warnings=[],
    )

    chunker = StructuralSemanticChunker()
    sections, chunks = chunker.chunk_document(doc)

    emb_provider = MockEmbeddingProvider(dimensions=768)
    embeddings = [
        EmbeddingRecord(
            id="emb_doc_001",
            document_id=doc_id,
            section_id=None,
            chunk_id=None,
            level=EmbeddingLevel.DOCUMENT,
            model_name="mock-model",
            model_version="1.0",
            dimensions=768,
            embedding=emb_provider.embed_single("Test paper"),
        )
    ]

    proc_record = DocumentProcessingRecord(
        document_id=doc_id,
        parser_version="1.0",
        chunker_version="1.0",
        chunker_config_hash="abc",
        embedding_model="mock-model",
        embedding_model_version="1.0",
        embedding_dimensions=768,
        status=ProcessingStatus.COMPLETED,
    )

    # Save atomically
    repo.save_document_atomic(
        document=doc,
        sections=sections,
        chunks=chunks,
        embeddings=embeddings,
        citations=[],
        unresolved_references=[],
        processing_record=proc_record,
    )

    # Verify saved
    loaded_proc = repo.get_document_processing(doc_id)
    assert loaded_proc is not None
    assert loaded_proc.status == ProcessingStatus.COMPLETED

    # Verify document failure recording
    fail_doc_id = "doc_test_failed"
    repo.record_document_failure(
        document_id=fail_doc_id,
        error_message="Corrupted PDF bytes",
        parser_version="1.0",
    )
    failed_proc = repo.get_document_processing(fail_doc_id)
    assert failed_proc is not None
    assert failed_proc.status == ProcessingStatus.FAILED
    assert failed_proc.error_message == "Corrupted PDF bytes"
