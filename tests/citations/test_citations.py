from __future__ import annotations

from ingest.schema import Reference
from indexer.citations.matching import normalize_arxiv, normalize_doi, normalize_title
from indexer.citations.resolver import CitationResolver, CorpusCitationCatalog


def test_normalization():
    assert normalize_doi("https://doi.org/10.1145/3422622") == "10.1145/3422622"
    assert normalize_doi("10.1145/3422622") == "10.1145/3422622"
    assert normalize_arxiv("arXiv:2107.04589v2") == "2107.04589"
    assert normalize_arxiv("2107.04589") == "2107.04589"
    assert normalize_title("  ViTGAN: Generative Adversarial Networks!  ") == "vitgan generative adversarial networks"


def test_citation_resolution_doi_and_arxiv():
    catalog = CorpusCitationCatalog()
    catalog.register_document(
        doc_id="doc_target_doi",
        doi="10.1145/1234567",
        arxiv_id=None,
        title="Some DOI Target Paper",
    )
    catalog.register_document(
        doc_id="doc_target_arxiv",
        doi=None,
        arxiv_id="2107.04589",
        title="Some arXiv Target Paper",
    )
    catalog.register_document(
        doc_id="doc_target_title",
        doi=None,
        arxiv_id=None,
        title="Attention Is All You Need In Deep Learning",
    )

    resolver = CitationResolver(catalog)

    refs = [
        # Match by DOI
        Reference(
            id=1,
            raw_text="[1] Target with DOI https://doi.org/10.1145/1234567",
            doi="10.1145/1234567",
        ),
        # Match by arXiv in raw_text
        Reference(
            id=2,
            raw_text="[2] Target with arXiv: 2107.04589v1 transformer paper",
        ),
        # Match by Title
        Reference(
            id=3,
            raw_text="[3] Vaswani et al. Attention Is All You Need In Deep Learning",
            title="Attention Is All You Need In Deep Learning",
        ),
        # Unresolved reference
        Reference(
            id=4,
            raw_text="[4] External Unknown Book 1999",
            title="External Unknown Book",
            year=1999,
        ),
    ]

    resolved, unresolved = resolver.resolve_document_references("doc_source_1", refs)

    assert len(resolved) == 3
    assert len(unresolved) == 1

    # Check DOI resolution
    cit_doi = next(c for c in resolved if c.reference_id == 1)
    assert cit_doi.target_document_id == "doc_target_doi"
    assert cit_doi.resolution_method == "doi"
    assert cit_doi.resolution_confidence == 1.0

    # Check arXiv resolution
    cit_arxiv = next(c for c in resolved if c.reference_id == 2)
    assert cit_arxiv.target_document_id == "doc_target_arxiv"
    assert cit_arxiv.resolution_method == "arxiv"
    assert cit_arxiv.resolution_confidence == 1.0

    # Check Title resolution
    cit_title = next(c for c in resolved if c.reference_id == 3)
    assert cit_title.target_document_id == "doc_target_title"
    assert cit_title.resolution_method == "title"
    assert cit_title.resolution_confidence == 0.9

    # Check Unresolved reference preservation
    unres = unresolved[0]
    assert unres.reference_id == 4
    assert unres.raw_text == "[4] External Unknown Book 1999"
    assert unres.title == "External Unknown Book"
    assert unres.year == 1999
