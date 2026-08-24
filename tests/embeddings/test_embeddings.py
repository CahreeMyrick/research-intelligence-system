from __future__ import annotations

import pytest
from indexer.chunking.enrichment import (
    build_chunk_embedding_text,
    build_document_embedding_text,
    build_section_embedding_text,
)
from indexer.embeddings.mock import MockEmbeddingProvider
from indexer.embeddings.sentence_transformers import SentenceTransformersProvider


def test_mock_embedding_provider_dimensions():
    provider = MockEmbeddingProvider(dimensions=768)
    assert provider.dimensions == 768
    vec = provider.embed_single("Test sentence")
    assert len(vec) == 768
    assert isinstance(vec[0], float)


def test_embedding_enrichment_templates():
    doc_emb = build_document_embedding_text(
        doc_title="ViTGAN",
        abstract="We present ViTGAN architecture.",
        section_titles=["1 Introduction", "2 Related Work", "3 ViTGAN"],
    )
    assert "Document: ViTGAN" in doc_emb
    assert "Abstract:\nWe present ViTGAN architecture." in doc_emb
    assert "Sections:\n- 1 Introduction\n- 2 Related Work\n- 3 ViTGAN" in doc_emb

    sec_emb = build_section_embedding_text(
        doc_title="ViTGAN",
        section_path=["3 ViTGAN", "3.1 Generator"],
        section_body_text="Generator details.",
    )
    assert "Document: ViTGAN" in sec_emb
    assert "Section: 3 ViTGAN > 3.1 Generator" in sec_emb
    assert "Generator details." in sec_emb

    chk_emb = build_chunk_embedding_text(
        doc_title="ViTGAN",
        section_path=["3 ViTGAN", "3.1 Generator"],
        chunk_source_text="Patch projection.",
    )
    assert "Document: ViTGAN" in chk_emb
    assert "Section: 3 ViTGAN" in chk_emb
    assert "Subsection: 3.1 Generator" in chk_emb
    assert "Patch projection." in chk_emb


def test_sentence_transformers_provider_live():
    provider = SentenceTransformersProvider(
        model_name="BAAI/bge-base-en-v1.5",
        expected_dimensions=768,
    )
    assert provider.name == "BAAI/bge-base-en-v1.5"
    assert provider.dimensions == 768
    
    vecs = provider.embed_texts(["First text", "Second text"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 768
    assert len(vecs[1]) == 768
