from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Optional
import re

import pymupdf # PyMuPDF
from pydantic import BaseModel, Field

from schemas import Document, Paragraph, Section, DocumentBody, Page, DocumentText, Corpus, ProcessingSummary, ProcessingResult
from schemas import PersistedPage, PersistedDocumentText, PersistedParagraph, PersistedSection, PersistedDocumentBody, PersistedDocumentMetadata, PersistedDocument


# ============================================================================
# Public API
# ============================================================================

__all__ = [
        "Extraction"
        "StructureExtraction"
        "save_document",
        "process_documents",
]

# ============================================================================
# Extraction
# ============================================================================


class Extraction:
    """
    Functions responsible for extracting information from a PDF.

    These functions operate on an already-open PyMuPDF document whenever
    possible so that a PDF does not have to be repeatedly opened.
    """

    @staticmethod
    def extract_id(file_path: str | Path) -> str:
        """
        Construct an identifier for a document.

        Input:
            file_path: str | Path
                Filesystem path to the PDF document.

        Output:
            str
                Document identifier.

        Current strategy:
            Use the filename without its extension.
        """

        path = Path(file_path)

        return path.stem

    @staticmethod
    def extract_title(
        pdf: pymupdf.Document,
        file_path: str | Path,
    ) -> Optional[str]:
        """
        Extract the title of a PDF.

        Input:
            pdf: pymupdf.Document
                Open PDF document.

            file_path: str | Path
                Filesystem path to the PDF.

        Output:
            Optional[str]
                PDF title if available.

        Strategy:
            1. Try the PDF metadata.
            2. Fall back to the filename.
        """

        metadata = pdf.metadata or {}

        title = metadata.get("title")

        if title:
            title = title.strip()

        if title:
            return title

        # If PDF metadata does not contain a title,
        # use the filename as a basic fallback.
        return Path(file_path).stem

    @staticmethod
    def extract_authors(pdf: pymupdf.Document) -> list[str]:
        """
        Extract authors from PDF metadata.

        Input:
            pdf: pymupdf.Document
                Open PDF document.

        Output:
            list[str]
                Extracted author names.

        Note:
            PDF metadata is inconsistent across documents, so this is
            only a first-pass extraction method.
        """

        metadata = pdf.metadata or {}

        author_string = metadata.get("author")

        if not author_string:
            return []

        # Different PDFs may separate authors with commas or semicolons.
        authors = re.split(r"[,;]", author_string)

        return [
            author.strip()
            for author in authors
            if author.strip()
        ]

    @staticmethod
    def extract_year(pdf: pymupdf.Document) -> Optional[int]:
        """
        Extract a year from PDF metadata.

        Input:
            pdf: pymupdf.Document
                Open PDF document.

        Output:
            Optional[int]
                Extracted year, or None if a reasonable year
                cannot be found.

        Strategy:
            Search common metadata fields for a four-digit year.
        """

        metadata = pdf.metadata or {}

        candidate_fields = [
            metadata.get("creationDate"),
            metadata.get("modDate"),
            metadata.get("subject"),
        ]

        for value in candidate_fields:
            if not value:
                continue

            # Find a plausible four-digit year.
            match = re.search(r"\b(19|20)\d{2}\b", value)

            if match:
                return int(match.group())

        return None

    @staticmethod
    def extract_source_path(file_path: str | Path) -> str:
        """
        Normalize the document's source path.

        Input:
            file_path: str | Path
                Filesystem path to the PDF.

        Output:
            str
                Absolute normalized path.
        """

        return str(Path(file_path).resolve())

    @staticmethod
    def extract_text(pdf: pymupdf.Document) -> DocumentText:
        """
        Extract page-level text from a PDF.

        Input:
            pdf: pymupdf.Document
                Open PDF document.

        Output:
            DocumentText
                Ordered page-level textual representation.

        Notes:
            page.get_text(..., sort=True) asks PyMuPDF to attempt to
            return text in a more natural reading order.

            This extracts embedded PDF text. It does NOT perform OCR
            on scanned/image-only PDFs.
        """

        pages: list[Page] = []

        # PyMuPDF iterates through pages in document order.
        for page_index, pdf_page in enumerate(pdf):

            # Extract textual content from this individual page.
            text = pdf_page.get_text(
                "text",
                sort=True,
            )

            # Preserve the association:
            #
            #     document -> page -> text
            #
            # instead of immediately flattening the entire PDF.
            page = Page(
                number=page_index + 1,
                text=text,
            )

            pages.append(page)

        return DocumentText(
            pages=pages,
        )

class StructureExtraction:
    """
    Recover semantic structure from a PDF.

    The goal of this stage is:

        PDF layout
            ↓
        headings + text blocks
            ↓
        sections + subsections + paragraphs
            ↓
        DocumentBody

    Important:
        PDFs generally do not explicitly say:

            "this is a section heading"
            "this is a paragraph"

        Therefore, semantic structure must be inferred from evidence
        such as:

            - section numbering
            - common heading names
            - font size
            - boldness
            - layout position

        This implementation should therefore be considered a
        first-pass heuristic structure extractor.
    """

    # Common top-level headings found in research papers.
    _KNOWN_HEADINGS = {
        "abstract",
        "introduction",
        "background",
        "related work",
        "method",
        "methods",
        "methodology",
        "experiments",
        "experimental setup",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "references",
        "acknowledgments",
        "acknowledgements",
        "appendix",
    }

    # Matches:
    #
    #   1 Introduction
    #   2. Methods
    #   3.1 Attention
    #   4.2.1 Experimental Setup
    #
    _NUMBERED_HEADING = re.compile(
        r"^(?P<number>\d+(?:\.\d+)*)(?:\.)?\s+(?P<title>\S.*)$"
    )

    @staticmethod
    def _estimate_body_font_size(
        pdf: pymupdf.Document,
    ) -> float:
        """
        Estimate the font size used for normal body text.

        Input:
            pdf: pymupdf.Document
                Open PDF document.

        Output:
            float
                Estimated body-text font size.

        Strategy:
            Find the font size responsible for the largest amount
            of textual content in the document.

        Example:
            If most of the document uses 10-point text, while
            headings use 14-point text, this should return
            approximately 10.
        """

        # Maps:
        #
        #     font_size -> number of characters using that size
        #
        character_counts: dict[float, int] = {}

        for pdf_page in pdf:

            # "dict" preserves layout and font information that is
            # unavailable from simple page.get_text("text").
            page_data = pdf_page.get_text(
                "dict",
                sort=True,
            )

            for block in page_data.get("blocks", []):

                # type == 0 indicates a text block.
                if block.get("type") != 0:
                    continue

                for line in block.get("lines", []):

                    for span in line.get("spans", []):

                        text = span.get("text", "").strip()

                        if not text:
                            continue

                        # Round slightly so tiny floating-point
                        # differences do not create separate font sizes.
                        font_size = round(
                            float(span.get("size", 0.0)),
                            1,
                        )

                        character_counts[font_size] = (
                            character_counts.get(font_size, 0)
                            + len(text)
                        )

        # Safe fallback in case no usable text was extracted.
        if not character_counts:
            return 12.0

        return max(
            character_counts,
            key=character_counts.get,
        )

    @staticmethod
    def _extract_block_information(
        block: dict,
    ) -> tuple[str, float, bool]:
        """
        Extract useful information from one PyMuPDF text block.

        Input:
            block: dict
                Text block returned from page.get_text("dict").

        Output:
            tuple[str, float, bool]

            Returns:
                text:
                    Combined textual content of the block.

                max_font_size:
                    Largest font size appearing in the block.

                is_bold:
                    True if any text span in the block is marked bold.
        """

        lines: list[str] = []

        max_font_size = 0.0
        is_bold = False

        for line in block.get("lines", []):

            span_texts: list[str] = []

            for span in line.get("spans", []):

                text = span.get("text", "").strip()

                if not text:
                    continue

                span_texts.append(text)

                max_font_size = max(
                    max_font_size,
                    float(span.get("size", 0.0)),
                )

                flags = int(
                    span.get("flags", 0)
                )

                # PyMuPDF exposes a font flag that indicates
                # whether a span is bold.
                if flags & pymupdf.TEXT_FONT_BOLD:
                    is_bold = True

            if span_texts:
                lines.append(
                    " ".join(span_texts)
                )

        # A block may contain several visual lines belonging to
        # the same logical paragraph.
        text = " ".join(lines).strip()

        return (
            text,
            max_font_size,
            is_bold,
        )

    @classmethod
    def _detect_heading_level(
        cls,
        text: str,
        font_size: float,
        is_bold: bool,
        body_font_size: float,
    ) -> int | None:
        """
        Determine whether a text block appears to be a heading.

        Input:
            text: str
                Text contained in the block.

            font_size: float
                Largest font size used by the block.

            is_bold: bool
                Whether the block contains bold text.

            body_font_size: float
                Estimated normal body-text font size.

        Output:
            int | None
                Heading level if the block appears to be a heading.

                None if the block appears to be ordinary body text.

        Examples:
            "1 Introduction"
                -> 1

            "3.2 Attention"
                -> 2

            "4.2.1 Training"
                -> 3
        """

        normalized = text.strip()

        # Very long blocks are probably paragraphs rather than headings.
        if not normalized or len(normalized) > 160:
            return None

        # --------------------------------------------------------------
        # Numbered headings
        # --------------------------------------------------------------

        match = cls._NUMBERED_HEADING.match(
            normalized
        )

        if match:

            section_number = match.group(
                "number"
            )

            # Examples:
            #
            #   1       -> level 1
            #   3.2     -> level 2
            #   4.2.1   -> level 3
            #
            return section_number.count(".") + 1

        # --------------------------------------------------------------
        # Common unnumbered research-paper headings
        # --------------------------------------------------------------

        normalized_heading = normalized.lower().strip(
            " :.-"
        )

        if normalized_heading in cls._KNOWN_HEADINGS:
            return 1

        # --------------------------------------------------------------
        # Formatting-based heuristic
        # --------------------------------------------------------------

        # A short block that is substantially larger than normal body
        # text and bold is likely to be a heading.
        if (
            len(normalized) <= 120
            and is_bold
            and font_size >= body_font_size * 1.15
        ):
            return 1

        return None

    @classmethod
    def extract_body(
        cls,
        pdf: pymupdf.Document,
    ) -> DocumentBody:
        """
        Recover semantic document structure from a PDF.

        Input:
            pdf: pymupdf.Document
                Open PDF document.

        Output:
            DocumentBody
                Semantic representation containing sections,
                subsections, and paragraphs.

        Transformation:

            PDF
                ↓
            layout-aware text blocks
                ↓
            classify heading / paragraph
                ↓
            construct section hierarchy
                ↓
            DocumentBody
        """

        body_font_size = cls._estimate_body_font_size(
            pdf
        )

        # Top-level document sections.
        sections: list[Section] = []

        # Stack representing the currently open section hierarchy.
        #
        # Example:
        #
        #   [
        #       Section("3 Model", level=1),
        #       Section("3.2 Attention", level=2),
        #   ]
        #
        section_stack: list[Section] = []

        for page_index, pdf_page in enumerate(pdf):

            page_number = page_index + 1

            page_data = pdf_page.get_text(
                "dict",
                sort=True,
            )

            for block in page_data.get(
                "blocks",
                [],
            ):

                # Ignore image blocks.
                if block.get("type") != 0:
                    continue

                (
                    block_text,
                    font_size,
                    is_bold,
                ) = cls._extract_block_information(
                    block
                )

                if not block_text:
                    continue

                heading_level = cls._detect_heading_level(
                    text=block_text,
                    font_size=font_size,
                    is_bold=is_bold,
                    body_font_size=body_font_size,
                )

                # ------------------------------------------------------
                # Heading
                # ------------------------------------------------------

                if heading_level is not None:

                    # Close sections that are at the same or deeper
                    # hierarchy level.
                    #
                    # Example:
                    #
                    #     current: 3.2
                    #     new:     3.3
                    #
                    # 3.2 must close before 3.3 begins.
                    while (
                        section_stack
                        and section_stack[-1].level
                        >= heading_level
                    ):
                        section_stack.pop()

                    section = Section(
                        heading=block_text,
                        level=heading_level,
                        page_start=page_number,
                        page_end=page_number,
                    )

                    # If another section remains on the stack,
                    # the new section is its subsection.
                    if section_stack:

                        parent = section_stack[-1]

                        parent.subsections.append(
                            section
                        )

                        parent.page_end = page_number

                    else:

                        # No parent means this is a top-level section.
                        sections.append(section)

                    section_stack.append(section)

                    continue

                # ------------------------------------------------------
                # Paragraph
                # ------------------------------------------------------

                paragraph = Paragraph(
                    text=block_text,
                    page_start=page_number,
                    page_end=page_number,
                )

                # Some PDFs contain text before the first recognizable
                # heading: title-area text, author information, etc.
                #
                # Preserve it rather than discarding it.
                if not section_stack:

                    front_matter = Section(
                        heading="Front Matter",
                        level=1,
                        page_start=page_number,
                        page_end=page_number,
                    )

                    sections.append(
                        front_matter
                    )

                    section_stack.append(
                        front_matter
                    )

                # Attach the paragraph to the currently active section.
                section_stack[-1].paragraphs.append(
                    paragraph
                )

                # Any ancestor containing this paragraph also extends
                # through the current page.
                for section in section_stack:
                    section.page_end = page_number

        return DocumentBody(
            sections=sections
        )

# ============================================================================
# Single-document processing
# ============================================================================
def _to_persisted_section(
    section: Section,
) -> PersistedSection:
    """
    Convert a runtime Section into its canonical persistence schema.

    Input:
        section: Section
            Runtime semantic section.

    Output:
        PersistedSection
            Serializable semantic section.

    Note:
        This function is recursive because sections may contain
        nested subsections.
    """

    return PersistedSection(
        heading=section.heading,
        level=section.level,
        page_start=section.page_start,
        page_end=section.page_end,

        paragraphs=[
            PersistedParagraph(
                text=paragraph.text,
                page_start=paragraph.page_start,
                page_end=paragraph.page_end,
            )
            for paragraph in section.paragraphs
        ],

        subsections=[
            _to_persisted_section(
                subsection
            )
            for subsection in section.subsections
        ],
    )

def _to_persisted_document(
    document: Document,
) -> PersistedDocument:
    """
    Convert a runtime Document into the canonical persistence schema.

    Input:
        document: Document
            Processed runtime representation of a document.

    Output:
        PersistedDocument
            Canonical representation that will be saved to disk.
    """

    return PersistedDocument(
        document_id=document.id,

        metadata=PersistedDocumentMetadata(
            title=document.title,
            authors=document.authors,
            year=document.year,
            doi=None,
            arxiv_id=None,
        ),

        text=PersistedDocumentText(
            pages=[
                PersistedPage(
                    number=page.number,
                    text=page.text,
                )
                for page in document.text.pages
            ]
        ),

        body=PersistedDocumentBody(
            sections=[
                _to_persisted_section(
                    section
                )
                for section in document.body.sections
            ]
        ),

        source_path=document.source_path,
        page_count=document.page_count,

        parser_name="pymupdf",
        parser_version=pymupdf.VersionBind,
    )

def save_document(
    document: PersistedDocument,
    output_dir: str | Path,
) -> Path:
    """
    Save a canonical persisted document as JSON.

    Input:
        document: PersistedDocument
            Validated canonical document representation.

        output_dir: str | Path
            Directory where the JSON file should be stored.

    Output:
        Path
            Path of the saved JSON file.
    """

    output_directory = Path(output_dir)

    # Ensure the corpus directory exists.
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Give each persisted document its own JSON file.
    output_path = (
        output_directory
        / f"{document.document_id}.json"
    )

    # Serialize exactly the PersistedDocument schema.
    output_path.write_text(
        document.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return output_path


def _process_one_document(
    file_path: str | Path,
) -> Document:
    """
    Process one PDF document.

    This function is intended to execute inside a worker process.

    Input:
        file_path: str | Path
            Filesystem path to one PDF.

    Output:
        Document
            Validated structured representation of the PDF.

    Processing pipeline:

        PDF path
            ↓
        open PDF
            ↓
        extract metadata
            ↓
        extract page-level text
            ↓
        construct Document
            ↓
        Pydantic validation
            ↓
        return Document
    """

    path = Path(file_path)

    # Verify that the file exists before attempting extraction.
    if not path.exists():
        raise FileNotFoundError(
            f"Document does not exist: {path}"
        )

    # The current processor supports PDF documents only.
    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Unsupported document type: {path.suffix}"
        )

    # Open the PDF once.
    #
    # All extraction functions operate on this same PDF object,
    # avoiding unnecessary repeated file opens.
    with pymupdf.open(path) as pdf:

        document_id = Extraction.extract_id(path)

        title = Extraction.extract_title(
            pdf,
            path,
        )

        authors = Extraction.extract_authors(
            pdf,
        )

        year = Extraction.extract_year(
            pdf,
        )

        text = Extraction.extract_text(
            pdf,
        )

        body = StructureExtraction.extract_body(
                pdf,
        )

        source_path = Extraction.extract_source_path(
            path,
        )

        page_count = len(pdf)

    # Construct the final structured document.
    #
    # Because Document inherits from Pydantic BaseModel,
    # the resulting object is validated against the schema.
    document = Document(
        id=document_id,
        title=title,
        authors=authors,
        year=year,
        text=text,
        body=body,
        source_path=source_path,
        page_count=page_count,
    )

    return document


# ============================================================================
# Corpus processing
# ============================================================================

def process_documents(
    dir_path: str | Path,
    output_dir: str | Path,
    max_workers: int | None = None,
) -> ProcessingResult:
    """
    Process and persist all PDF documents in a directory concurrently.

    Input:
        dir_path: str | Path
            Filesystem path to the directory containing raw PDF documents.

        output_dir: str | Path
            Directory where canonical structured documents will be saved.

        max_workers: int | None
            Maximum number of worker processes.

            If None, ProcessPoolExecutor chooses an appropriate
            default based on the machine.

    Output:
        ProcessingResult
            Contains:

                1. Corpus
                   Runtime collection of successfully processed documents.

                2. ProcessingSummary
                   Number of discovered, successful, and failed documents
                   plus total elapsed processing time.

    Side effects:
        - Reads PDF files from the filesystem.
        - Processes documents concurrently.
        - Saves each successfully processed document as JSON.
    """

    directory = Path(dir_path)
    output_directory = Path(output_dir)

    # ------------------------------------------------------------------
    # Validate input directory
    # ------------------------------------------------------------------

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {directory}"
        )

    # ------------------------------------------------------------------
    # Prepare persistence directory
    # ------------------------------------------------------------------

    # Create the canonical corpus directory if it does not exist.
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Discover supported documents
    # ------------------------------------------------------------------

    file_paths = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".pdf"
    ]

    # Deterministic discovery order.
    file_paths.sort()

    start_time = perf_counter()

    documents: list[Document] = []
    failed = 0

    # ------------------------------------------------------------------
    # Process documents concurrently
    # ------------------------------------------------------------------

    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:

        # Submit one independent processing task per PDF.
        #
        # Each Future represents:
        #
        #     _process_one_document(file_path)
        #
        # and is associated with the source file that produced it.
        futures = {
            executor.submit(
                _process_one_document,
                file_path,
            ): file_path
            for file_path in file_paths
        }

        # Handle documents as soon as they finish processing.
        for future in as_completed(futures):

            file_path = futures[future]

            try:
                # ------------------------------------------------------
                # 1. Get runtime Document
                # ------------------------------------------------------

                # Retrieve the validated Document returned by the
                # worker process.
                document = future.result()

                # ------------------------------------------------------
                # 2. Convert to persistence schema
                # ------------------------------------------------------

                # The runtime Document is transformed into the canonical
                # representation that we have explicitly decided should
                # survive after this program terminates.
                persisted_document = _to_persisted_document(
                    document
                )

                # ------------------------------------------------------
                # 3. Persist the document
                # ------------------------------------------------------

                # Serialize the PersistedDocument to JSON and save it
                # in the canonical corpus directory.
                saved_path = save_document(
                    persisted_document,
                    output_directory,
                )

                # ------------------------------------------------------
                # 4. Maintain runtime corpus
                # ------------------------------------------------------

                # Keep the runtime Document in memory so the caller
                # receives a Corpus as part of ProcessingResult.
                documents.append(document)

                print(
                    f"Processed: {file_path.name}"
                )

                print(
                    f"Saved: {saved_path}"
                )

            except Exception as error:

                # A failure for one document should not prevent other
                # independent documents from completing.
                failed += 1

                print(
                    f"Failed: {file_path.name}"
                )

                print(
                    f"Reason: {error}"
                )

    # ------------------------------------------------------------------
    # Restore deterministic corpus ordering
    # ------------------------------------------------------------------

    # as_completed() returns documents in completion order rather
    # than input order, so restore deterministic ordering afterward.
    documents.sort(
        key=lambda document: document.source_path
    )

    elapsed_seconds = perf_counter() - start_time

    # ------------------------------------------------------------------
    # Construct processing result
    # ------------------------------------------------------------------

    corpus = Corpus(
        documents=documents,
    )

    summary = ProcessingSummary(
        discovered=len(file_paths),
        successful=len(documents),
        failed=failed,
        elapsed_seconds=elapsed_seconds,
    )

    return ProcessingResult(
        corpus=corpus,
        summary=summary,
    )

def main() -> None:
    """
    Run the document processing pipeline.

    Input:
        None

    Output:
        None

    Side effects:
        - Reads raw PDFs.
        - Processes PDFs concurrently.
        - Persists canonical structured documents.
        - Prints processing statistics.
    """

    # Raw source documents.
    document_directory = Path(
        "../ingestion/papers"
    )

    # Canonical processed corpus.
    corpus_directory = Path(
        "corpus"
    )

    result = process_documents(
        dir_path=document_directory,
        output_dir=corpus_directory,
    )

    print()
    print("Processing complete")
    print("-------------------")

    print(
        f"Documents discovered: "
        f"{result.summary.discovered}"
    )

    print(
        f"Successfully processed: "
        f"{result.summary.successful}"
    )

    print(
        f"Failed: "
        f"{result.summary.failed}"
    )

    print(
        f"Elapsed time: "
        f"{result.summary.elapsed_seconds:.3f} seconds"
    )

if __name__ == "__main__":
    main()
