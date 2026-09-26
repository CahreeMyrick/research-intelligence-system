from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ============================================================================
# Runtime document schemas
# ============================================================================

__all__ = [
    "Page",
    "DocumentText",
    "Paragraph",

]

class Page(BaseModel):
    """
    Structured representation of a single PDF page.

    Fields:
        number:
            Human-readable page number. Starts at 1.

        text:
            Text extracted from the page.
    """

    number: int
    text: str


class DocumentText(BaseModel):
    """
    Physical textual representation of a document.

    Fields:
        pages:
            Ordered collection of pages extracted from the document.
    """

    pages: list[Page] = Field(default_factory=list)

    def full_text(self) -> str:
        """
        Return all page text as one string.

        Input:
            None

        Output:
            str
                Complete document text with pages joined in their
                original order.
        """

        return "\n\n".join(
            page.text
            for page in self.pages
        )


class Paragraph(BaseModel):
    """
    Semantic paragraph extracted from a document.

    Fields:
        text:
            Paragraph content.

        page_start:
            First source page containing this paragraph.

        page_end:
            Last source page containing this paragraph.
    """

    text: str

    page_start: int
    page_end: int


class Section(BaseModel):
    """
    Semantic section extracted from a document.

    Fields:
        heading:
            Section heading.

            Examples:
                "Introduction"
                "3 Model Architecture"
                "3.2 Attention"

        level:
            Depth of the section in the document hierarchy.

            Example:
                1 -> top-level section
                2 -> subsection
                3 -> sub-subsection

        page_start:
            First page occupied by the section.

        page_end:
            Last page occupied by the section.

        paragraphs:
            Ordered paragraphs directly belonging to this section.

        subsections:
            Child sections nested beneath this section.
    """

    heading: str
    level: int

    page_start: int
    page_end: int

    paragraphs: list[Paragraph] = Field(
        default_factory=list
    )

    subsections: list["Section"] = Field(
        default_factory=list
    )


class DocumentBody(BaseModel):
    """
    Semantic representation of the document body.

    Fields:
        sections:
            Ordered collection of top-level document sections.
    """

    sections: list[Section] = Field(
        default_factory=list
    )


class Document(BaseModel):
    """
    Structured runtime representation of one processed document.

    Fields:
        id:
            Identifier assigned to the document.

        title:
            Extracted PDF title, if available.

        authors:
            Extracted document authors.

        year:
            Extracted publication/creation year, if available.

        text:
            Physical page-level textual representation.

        body:
            Semantic document structure containing sections
            and paragraphs.

        source_path:
            Original filesystem path of the PDF.

        page_count:
            Number of pages in the PDF.
    """

    id: str

    title: Optional[str] = None

    authors: list[str] = Field(
        default_factory=list
    )

    year: Optional[int] = None

    # Physical representation:
    #
    # document
    #   -> pages
    #       -> text
    text: DocumentText

    # Semantic representation:
    #
    # document
    #   -> sections
    #       -> paragraphs
    #       -> subsections
    body: DocumentBody

    source_path: str
    page_count: int


class Corpus(BaseModel):
    """
    Collection of structured runtime documents.

    Fields:
        documents:
            All successfully processed documents.
    """

    documents: list[Document] = Field(
        default_factory=list
    )


# ============================================================================
# Processing schemas
# ============================================================================


class ProcessingSummary(BaseModel):
    """
    Summary describing a document-processing run.

    Fields:
        discovered:
            Number of supported documents discovered.

        successful:
            Number of documents processed successfully.

        failed:
            Number of documents that failed processing.

        elapsed_seconds:
            Total wall-clock processing time.
    """

    discovered: int
    successful: int
    failed: int
    elapsed_seconds: float


class ProcessingResult(BaseModel):
    """
    Complete result of processing a directory.

    Fields:
        corpus:
            Structured corpus containing successfully processed documents.

        summary:
            Performance and success information about the processing run.
    """

    corpus: Corpus
    summary: ProcessingSummary


# ============================================================================
# Persistent document schemas
# ============================================================================


class PersistedPage(BaseModel):
    """
    Persistent representation of one extracted PDF page.

    Fields:
        number:
            Page number in the original document.

        text:
            Extracted textual content of the page.
    """

    number: int
    text: str


class PersistedDocumentText(BaseModel):
    """
    Persistent physical textual representation of a document.

    Fields:
        pages:
            Ordered pages extracted from the source document.
    """

    pages: list[PersistedPage] = Field(
        default_factory=list
    )


class PersistedParagraph(BaseModel):
    """
    Persistent representation of a semantic paragraph.

    Fields:
        text:
            Paragraph content.

        page_start:
            First source page containing this paragraph.

        page_end:
            Last source page containing this paragraph.
    """

    text: str

    page_start: int
    page_end: int


class PersistedSection(BaseModel):
    """
    Persistent representation of a semantic document section.

    Fields:
        heading:
            Section heading.

        level:
            Heading depth in the document hierarchy.

        page_start:
            First source page occupied by the section.

        page_end:
            Last source page occupied by the section.

        paragraphs:
            Paragraphs directly contained by this section.

        subsections:
            Nested semantic sections.
    """

    heading: str
    level: int

    page_start: int
    page_end: int

    paragraphs: list[PersistedParagraph] = Field(
        default_factory=list
    )

    subsections: list["PersistedSection"] = Field(
        default_factory=list
    )


class PersistedDocumentBody(BaseModel):
    """
    Persistent semantic representation of the document body.

    Fields:
        sections:
            Ordered collection of top-level semantic sections.
    """

    sections: list[PersistedSection] = Field(
        default_factory=list
    )


class PersistedDocumentMetadata(BaseModel):
    """
    Persistent metadata extracted from the document.

    Fields:
        title:
            Document title, if known.

        authors:
            Extracted author names.

        year:
            Publication year, if known.

        doi:
            DOI, if available.

        arxiv_id:
            arXiv identifier, if available.
    """

    title: str | None = None

    authors: list[str] = Field(
        default_factory=list
    )

    year: int | None = None

    doi: str | None = None
    arxiv_id: str | None = None


class PersistedDocument(BaseModel):
    """
    Canonical representation saved after document processing.

    Fields:
        document_id:
            Stable document identifier.

        metadata:
            Extracted bibliographic metadata.

        text:
            Physical page-level representation of the document.

        body:
            Semantic representation containing sections,
            subsections, and paragraphs.

        source_path:
            Path from which the original document was processed.

        page_count:
            Number of pages in the original PDF.

        parser_name:
            Name of the parser used to process the document.

        parser_version:
            Version of the parser used to process the document.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    document_id: str

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    metadata: PersistedDocumentMetadata

    # ------------------------------------------------------------------
    # Physical document structure
    # ------------------------------------------------------------------

    text: PersistedDocumentText

    # ------------------------------------------------------------------
    # Semantic document structure
    # ------------------------------------------------------------------

    body: PersistedDocumentBody

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    source_path: str
    page_count: int

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    parser_name: str
    parser_version: str
