from __future__ import annotations

import logging
from typing import Iterable, Optional
from ingest.schema import Document as IngestDocument, Reference as IngestReference
from indexer.citations.matching import normalize_arxiv, normalize_doi, normalize_title
from indexer.schemas.citation import ResolvedCitation, UnresolvedReference

logger = logging.getLogger(__name__)


class CorpusCitationCatalog:
    """Index of known documents in the corpus for citation resolution."""
    def __init__(self):
        self.doi_to_doc_id: dict[str, str] = {}
        self.arxiv_to_doc_id: dict[str, str] = {}
        self.title_to_doc_id: dict[str, str] = {}

    def register_document(self, doc_id: str, doi: Optional[str], arxiv_id: Optional[str], title: Optional[str]):
        norm_d = normalize_doi(doi)
        if norm_d:
            self.doi_to_doc_id[norm_d] = doc_id

        norm_a = normalize_arxiv(arxiv_id)
        if norm_a:
            self.arxiv_to_doc_id[norm_a] = doc_id

        norm_t = normalize_title(title)
        # Avoid indexing trivially short titles that could cause false positive matches
        if norm_t and len(norm_t.split()) >= 3:
            self.title_to_doc_id[norm_t] = doc_id

    @classmethod
    def build_from_documents(cls, documents: Iterable[IngestDocument]) -> CorpusCitationCatalog:
        catalog = cls()
        for doc in documents:
            catalog.register_document(
                doc_id=doc.identity.id,
                doi=doc.metadata.doi,
                arxiv_id=doc.metadata.arxiv_id,
                title=doc.metadata.title,
            )
        return catalog


class CitationResolver:
    """
    Resolves references against documents in the corpus.
    Priority:
        1. DOI match (1.0 confidence)
        2. arXiv ID match (1.0 confidence)
        3. Normalized title match (0.9 confidence)
    """
    def __init__(self, catalog: CorpusCitationCatalog):
        self.catalog = catalog

    def resolve_document_references(
        self,
        source_doc_id: str,
        references: list[IngestReference],
    ) -> tuple[list[ResolvedCitation], list[UnresolvedReference]]:
        resolved: list[ResolvedCitation] = []
        unresolved: list[UnresolvedReference] = []

        for ref in references:
            target_id = None
            method = None
            confidence = 1.0

            # 1. Check DOI
            norm_doi = normalize_doi(ref.doi)
            if norm_doi and norm_doi in self.catalog.doi_to_doc_id:
                target_id = self.catalog.doi_to_doc_id[norm_doi]
                method = "doi"
                confidence = 1.0

            # 2. Check arXiv in ref raw_text or doi field
            if not target_id:
                # Look for arXiv pattern if not explicit
                norm_arxiv = None
                if ref.doi and "arxiv" in ref.doi.lower():
                    norm_arxiv = normalize_arxiv(ref.doi)
                if not norm_arxiv and ref.raw_text:
                    import re
                    m = re.search(r"\barXiv:\s*(\d{4}\.\d{4,5})(v\d+)?\b", ref.raw_text, re.IGNORECASE)
                    if m:
                        norm_arxiv = normalize_arxiv(m.group(1))

                if norm_arxiv and norm_arxiv in self.catalog.arxiv_to_doc_id:
                    target_id = self.catalog.arxiv_to_doc_id[norm_arxiv]
                    method = "arxiv"
                    confidence = 1.0

            # 3. Check Title
            if not target_id and ref.title:
                norm_title = normalize_title(ref.title)
                if norm_title and norm_title in self.catalog.title_to_doc_id:
                    target_id = self.catalog.title_to_doc_id[norm_title]
                    method = "title"
                    confidence = 0.9

            # Don't resolve a document as citing itself
            if target_id == source_doc_id:
                target_id = None

            if target_id and method:
                resolved.append(
                    ResolvedCitation(
                        id=f"cit_{source_doc_id}_{ref.id:04d}",
                        source_document_id=source_doc_id,
                        target_document_id=target_id,
                        reference_id=ref.id,
                        raw_text=ref.raw_text,
                        doi=ref.doi,
                        arxiv_id=norm_arxiv if 'norm_arxiv' in locals() else None,
                        resolution_method=method,
                        resolution_confidence=confidence,
                    )
                )
            else:
                unresolved.append(
                    UnresolvedReference(
                        id=f"unres_{source_doc_id}_{ref.id:04d}",
                        source_document_id=source_doc_id,
                        reference_id=ref.id,
                        raw_text=ref.raw_text,
                        doi=ref.doi,
                        arxiv_id=norm_arxiv if 'norm_arxiv' in locals() else None,
                        title=ref.title,
                        authors=ref.authors,
                        year=ref.year,
                    )
                )

        return resolved, unresolved
