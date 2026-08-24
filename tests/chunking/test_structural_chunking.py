from __future__ import annotations

from ingest.schema import Document, DocumentIdentity, DocumentMetadata, DocumentSource, Paragraph, Section
from indexer.chunking import ChunkerConfig, StructuralSemanticChunker, build_chunk_embedding_text
from indexer.chunking.tokenization import RegexFallbackTokenizer


def _make_dummy_doc(sections: list[Section]) -> Document:
    return Document(
        identity=DocumentIdentity(
            id="doc_test_123",
            source=DocumentSource(
                filename="test.pdf",
                path="/dummy/test.pdf",
                content_hash="dummy_hash_123",
            ),
        ),
        metadata=DocumentMetadata(
            title="A Test Paper on Transformers",
            authors=["Alice", "Bob"],
            publication_year=2024,
            abstract="This is the abstract of the test paper.",
        ),
        sections=sections,
        references=[],
        page_count=3,
        warnings=[],
    )


def test_chunking_preserves_section_boundaries():
    sec1 = Section(
        title="1 Introduction",
        level=1,
        paragraphs=[Paragraph(text="Intro paragraph text.", page=0, position=1)],
        subsections=[],
    )
    sec2 = Section(
        title="2 Methods",
        level=1,
        paragraphs=[Paragraph(text="Methods paragraph text.", page=1, position=2)],
        subsections=[],
    )
    doc = _make_dummy_doc([sec1, sec2])
    
    config = ChunkerConfig(target_tokens=500, overlap_tokens=50)
    chunker = StructuralSemanticChunker(config=config, tokenizer=RegexFallbackTokenizer())
    
    section_nodes, chunks = chunker.chunk_document(doc)
    
    assert len(section_nodes) == 2
    assert len(chunks) == 2
    
    # Ensure chunks never cross section boundaries
    assert chunks[0].section_id == section_nodes[0].id
    assert chunks[0].section_title == "1 Introduction"
    assert "Intro" in chunks[0].source_text
    assert "Methods" not in chunks[0].source_text

    assert chunks[1].section_id == section_nodes[1].id
    assert chunks[1].section_title == "2 Methods"
    assert "Methods" in chunks[1].source_text
    assert "Intro" not in chunks[1].source_text


def test_short_section_produces_one_chunk():
    sec = Section(
        title="5 Conclusion",
        level=1,
        paragraphs=[Paragraph(text="Very brief conclusion.", page=2, position=10)],
        subsections=[],
    )
    doc = _make_dummy_doc([sec])
    
    config = ChunkerConfig(target_tokens=512, overlap_tokens=64)
    chunker = StructuralSemanticChunker(config=config, tokenizer=RegexFallbackTokenizer())
    
    _, chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].section_title == "5 Conclusion"
    assert chunks[0].source_text == "Very brief conclusion."


def test_oversized_paragraph_split_at_sentences():
    # Construct paragraph with multiple distinct sentences exceeding target token count (e.g. 15 tokens target)
    long_sentences = [
        "First sentence discussing neural architectures in depth with details.",
        "Second sentence focusing on self attention mechanism complexity.",
        "Third sentence describing experimental benchmark evaluation results.",
    ]
    p_text = " ".join(long_sentences)
    sec = Section(
        title="3 Deep Dive",
        level=1,
        paragraphs=[Paragraph(text=p_text, page=1, position=5)],
        subsections=[],
    )
    doc = _make_dummy_doc([sec])
    
    # Target of 12 tokens forces splitting across sentences
    config = ChunkerConfig(target_tokens=15, overlap_tokens=0, max_tokens=15)
    chunker = StructuralSemanticChunker(config=config, tokenizer=RegexFallbackTokenizer())
    
    _, chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.section_title == "3 Deep Dive"


def test_overlap_works_within_same_section():
    p1 = Paragraph(text="Sentence A is first. Sentence B is second.", page=0, position=1)
    p2 = Paragraph(text="Sentence C is third. Sentence D is fourth.", page=0, position=2)
    sec = Section(title="1 Overview", level=1, paragraphs=[p1, p2], subsections=[])
    doc = _make_dummy_doc([sec])
    
    config = ChunkerConfig(target_tokens=12, overlap_tokens=6)
    chunker = StructuralSemanticChunker(config=config, tokenizer=RegexFallbackTokenizer())
    
    _, chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2


def test_chunk_ids_are_deterministic():
    sec = Section(
        title="1 Introduction",
        level=1,
        paragraphs=[Paragraph(text="Determinism test paragraph.", page=0, position=1)],
        subsections=[],
    )
    doc = _make_dummy_doc([sec])
    
    chunker1 = StructuralSemanticChunker(ChunkerConfig(), tokenizer=RegexFallbackTokenizer())
    chunker2 = StructuralSemanticChunker(ChunkerConfig(), tokenizer=RegexFallbackTokenizer())
    
    _, chunks1 = chunker1.chunk_document(doc)
    _, chunks2 = chunker2.chunk_document(doc)
    
    assert chunks1[0].chunk_id == chunks2[0].chunk_id == "chunk_doc_test_123_0000"
    assert chunks1[0].chunker_config_hash == chunks2[0].chunker_config_hash


def test_chunker_config_change_alters_hash():
    cfg1 = ChunkerConfig(target_tokens=512, overlap_tokens=64)
    cfg2 = ChunkerConfig(target_tokens=256, overlap_tokens=32)
    cfg3 = ChunkerConfig(target_tokens=512, overlap_tokens=64, chunker_version="2.0")
    
    assert cfg1.config_hash() != cfg2.config_hash()
    assert cfg1.config_hash() != cfg3.config_hash()


def test_enrichment_does_not_mutate_source_text():
    source = "Raw text content."
    enriched = build_chunk_embedding_text("Paper Title", ["Section 1", "Subsection 1.1"], source)
    
    assert "Document: Paper Title" in enriched
    assert "Section: Section 1" in enriched
    assert "Subsection: Subsection 1.1" in enriched
    assert enriched.endswith(source)
    assert source == "Raw text content."
