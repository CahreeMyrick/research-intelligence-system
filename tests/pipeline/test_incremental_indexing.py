from __future__ import annotations

from ingest.schema import Document, DocumentIdentity, DocumentMetadata, DocumentSource
from indexer.chunking.base import ChunkerConfig
from indexer.embeddings.mock import MockEmbeddingProvider
from indexer.pipeline.state import ProcessingDecision, determine_processing_state
from indexer.schemas.processing import DocumentProcessingRecord, ProcessingStatus


def _make_sample_doc():
    return Document(
        identity=DocumentIdentity(
            id="doc_sample_inc",
            source=DocumentSource(
                filename="sample.pdf",
                path="/test/sample.pdf",
                content_hash="hash_sample",
            ),
        ),
        metadata=DocumentMetadata(
            title="Sample Incremental Paper",
            publication_year=2024,
        ),
        sections=[],
        references=[],
        page_count=1,
    )


def test_incremental_state_decisions():
    doc = _make_sample_doc()
    chunker_cfg = ChunkerConfig(target_tokens=512, overlap_tokens=64, chunker_version="1.0")
    config_hash = chunker_cfg.config_hash()
    emb_provider = MockEmbeddingProvider(model_name="mock-model", model_version="1.0", dimensions=768)

    # 1. New document (no record) -> NEEDS_FULL_INDEXING
    assert (
        determine_processing_state(doc, None, chunker_cfg, emb_provider)
        == ProcessingDecision.NEEDS_FULL_INDEXING
    )

    # 2. Completed record matching all versions -> UP_TO_DATE
    current_rec = DocumentProcessingRecord(
        document_id="doc_sample_inc",
        parser_version="1.0",
        chunker_version="1.0",
        chunker_config_hash=config_hash,
        embedding_model="mock-model",
        embedding_model_version="1.0",
        embedding_dimensions=768,
        status=ProcessingStatus.COMPLETED,
    )
    assert (
        determine_processing_state(doc, current_rec, chunker_cfg, emb_provider)
        == ProcessingDecision.UP_TO_DATE
    )

    # 3. Chunker config hash changed (target_tokens changed) -> NEEDS_RECHUNK_AND_REEMBED
    new_chunker_cfg = ChunkerConfig(target_tokens=256, overlap_tokens=32, chunker_version="1.0")
    assert (
        determine_processing_state(doc, current_rec, new_chunker_cfg, emb_provider)
        == ProcessingDecision.NEEDS_RECHUNK_AND_REEMBED
    )

    # 4. Chunker version changed -> NEEDS_RECHUNK_AND_REEMBED
    v2_chunker_cfg = ChunkerConfig(target_tokens=512, overlap_tokens=64, chunker_version="2.0")
    assert (
        determine_processing_state(doc, current_rec, v2_chunker_cfg, emb_provider)
        == ProcessingDecision.NEEDS_RECHUNK_AND_REEMBED
    )

    # 5. Embedding model changed -> NEEDS_REEMBED_ONLY
    new_emb_provider = MockEmbeddingProvider(model_name="new-model", model_version="1.0", dimensions=768)
    assert (
        determine_processing_state(doc, current_rec, chunker_cfg, new_emb_provider)
        == ProcessingDecision.NEEDS_REEMBED_ONLY
    )

    # 6. Parser version changed -> NEEDS_FULL_INDEXING
    assert (
        determine_processing_state(doc, current_rec, chunker_cfg, emb_provider, parser_version="2.0")
        == ProcessingDecision.NEEDS_FULL_INDEXING
    )

    # 7. Previously failed -> NEEDS_FULL_INDEXING
    failed_rec = current_rec.model_copy(update={"status": ProcessingStatus.FAILED})
    assert (
        determine_processing_state(doc, failed_rec, chunker_cfg, emb_provider)
        == ProcessingDecision.NEEDS_FULL_INDEXING
    )

    # 8. Force flag -> NEEDS_FULL_INDEXING
    assert (
        determine_processing_state(doc, current_rec, chunker_cfg, emb_provider, force=True)
        == ProcessingDecision.NEEDS_FULL_INDEXING
    )
