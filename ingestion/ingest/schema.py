"""
Canonical schema for the ingestion pipeline.

This is the target of `canonicalization`: no matter how a source PDF is laid
out internally, every paper is coerced into this same shape.

Layers (per the ingestion contract):
    1. DocumentSource / DocumentIdentity  -> stable identity
    2. DocumentMetadata                   -> bibliographic metadata
    3. Section / Paragraph                -> document structure + provenance
    4. Reference                          -> structured reference list
    5. Document                           -> everything above, for one paper
    6. Corpus                             -> collection of Documents
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Identity
# ---------------------------------------------------------------------------

class DocumentSource(BaseModel):
    """Where this document came from on disk."""
    filename: str
    path: str
    content_hash: str  # sha256 of raw file bytes -> used for dedup


class DocumentIdentity(BaseModel):
    id: str  # stable id, derived from content_hash
    source: DocumentSource


# ---------------------------------------------------------------------------
# 2. Bibliographic metadata
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)
    publication_year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    abstract: Optional[str] = None
    other_identifiers: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 3. Structure + provenance
# ---------------------------------------------------------------------------

class Paragraph(BaseModel):
    text: str
    page: int          # 0-indexed page number this paragraph starts on
    position: int       # global order index within the document, for stable sort/citation


class Section(BaseModel):
    title: str
    level: int  # 1 = top-level section, 2 = subsection, ...
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    paragraphs: list[Paragraph] = Field(default_factory=list)
    subsections: list["Section"] = Field(default_factory=list)


Section.model_rebuild()


# ---------------------------------------------------------------------------
# 4. References
# ---------------------------------------------------------------------------

class Reference(BaseModel):
    id: int                      # local reference number, e.g. [1] -> 1
    raw_text: str                # always preserved, even if nothing else resolves
    authors: list[str] = Field(default_factory=list)
    title: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None


# ---------------------------------------------------------------------------
# 5. Document
# ---------------------------------------------------------------------------

class Document(BaseModel):
    identity: DocumentIdentity
    metadata: DocumentMetadata
    sections: list[Section] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    page_count: int = 0
    warnings: list[str] = Field(default_factory=list)  # non-fatal issues during ingestion


# ---------------------------------------------------------------------------
# 6. Manifest
# ---------------------------------------------------------------------------

class DocumentManifestEntry(BaseModel):
    id: str
    filename: str
    content_hash: str
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    publication_year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    page_count: int = 0
    raw_path: str
    parsed_path: str
    chunks_path: Optional[str] = None
    embeddings_path: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


class CorpusManifest(BaseModel):
    version: str = "1.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    document_count: int = 0
    documents: list[DocumentManifestEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 7. Corpus
# ---------------------------------------------------------------------------

class Corpus(BaseModel):
    documents: list[Document] = Field(default_factory=list)

    def by_id(self, doc_id: str) -> Optional[Document]:
        return next((d for d in self.documents if d.identity.id == doc_id), None)

    def content_hashes(self) -> set[str]:
        return {d.identity.source.content_hash for d in self.documents}

    def manifest(self) -> CorpusManifest:
        entries = [
            DocumentManifestEntry(
                id=doc.identity.id,
                filename=doc.identity.source.filename,
                content_hash=doc.identity.source.content_hash,
                title=doc.metadata.title,
                authors=doc.metadata.authors,
                publication_year=doc.metadata.publication_year,
                doi=doc.metadata.doi,
                arxiv_id=doc.metadata.arxiv_id,
                page_count=doc.page_count,
                raw_path=f"raw/pdf/{doc.identity.id}.pdf",
                parsed_path=f"parsed/{doc.identity.id}.json",
                chunks_path=f"chunks/{doc.identity.id}.json",
                warnings=doc.warnings,
            )
            for doc in self.documents
        ]
        return CorpusManifest(
            document_count=len(self.documents),
            documents=entries,
        )

    def save(self, corpus_dir: str | Path) -> CorpusManifest:
        corpus_dir = Path(corpus_dir)
        raw_pdf_dir = corpus_dir / "raw" / "pdf"
        parsed_dir = corpus_dir / "parsed"
        chunks_dir = corpus_dir / "chunks"

        raw_pdf_dir.mkdir(parents=True, exist_ok=True)
        parsed_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir.mkdir(parents=True, exist_ok=True)

        for doc in self.documents:
            # 1. Copy raw PDF
            src_path = Path(doc.identity.source.path)
            dest_pdf = raw_pdf_dir / f"{doc.identity.id}.pdf"
            if src_path.exists() and src_path.is_file():
                if not dest_pdf.exists() or dest_pdf.resolve() != src_path.resolve():
                    shutil.copy2(src_path, dest_pdf)

            # 2. Write parsed JSON
            dest_json = parsed_dir / f"{doc.identity.id}.json"
            dest_json.write_text(doc.model_dump_json(indent=2), encoding="utf-8")

        # 3. Write manifest.json
        manifest = self.manifest()
        manifest_path = corpus_dir / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        return manifest

    @classmethod
    def load(cls, corpus_dir: str | Path) -> Corpus:
        corpus_dir = Path(corpus_dir)
        manifest_path = corpus_dir / "manifest.json"
        parsed_dir = corpus_dir / "parsed"

        documents: list[Document] = []
        if manifest_path.exists():
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in manifest_data.get("documents", []):
                doc_path = corpus_dir / entry.get("parsed_path", f"parsed/{entry['id']}.json")
                if doc_path.exists():
                    doc = Document.model_validate_json(doc_path.read_text(encoding="utf-8"))
                    documents.append(doc)
        elif parsed_dir.exists():
            for doc_path in sorted(parsed_dir.glob("*.json")):
                doc = Document.model_validate_json(doc_path.read_text(encoding="utf-8"))
                documents.append(doc)

        return cls(documents=documents)
