"""
ingest(papers) -> Corpus

The single public entry point. Wires together the five stages:

    PDF -> identify -> extract -> reconstruct structure
        -> extract metadata -> extract references -> canonical Document

A failure on any single PDF is isolated: it's recorded and skipped rather
than aborting the whole run, since a corpus of N papers should still
produce N-1 results if one file is corrupt.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ingest import identity, extraction, structure, metadata, references
from ingest.schema import Corpus, Document

logger = logging.getLogger("ingest")


def _strip_title_section(sections: list, title: str | None) -> list:
    """
    The paper title is sometimes picked up as the first top-level "section"
    by structure reconstruction (large font, own block). It already lives
    in `Document.metadata.title`, so drop it here to avoid duplicating it
    as a section with no real body content.
    """
    if not sections or not title:
        return sections
    first = sections[0]
    if first.level == 1 and first.title.strip() == title.strip() and not first.subsections:
        return sections[1:]
    return sections


def ingest_one(path: Path) -> Document:
    """Run the full pipeline on a single PDF. Raises on unrecoverable errors."""
    doc_identity = identity.identify(path)

    raw = extraction.extract(path)
    sections = structure.reconstruct(raw)

    doc_metadata = metadata.extract_metadata(raw, sections)
    doc_references = references.extract_references(sections)
    body_sections = references.strip_reference_section(sections)
    body_sections = _strip_title_section(body_sections, doc_metadata.title)

    warnings: list[str] = []
    if not doc_metadata.title:
        warnings.append("title extraction failed")
    if not body_sections:
        warnings.append("no sections detected -- structure reconstruction may have failed")
    if not doc_references:
        warnings.append("no references detected")

    return Document(
        identity=doc_identity,
        metadata=doc_metadata,
        sections=body_sections,
        references=doc_references,
        page_count=raw.page_count,
        warnings=warnings,
    )


def ingest(
    papers_dir: str | Path,
    output_dir: str | Path | None = None,
    recursive: bool = False,
) -> Corpus:
    """
    Ingest every PDF in `papers_dir` into a Corpus.
    Set `recursive=True` to include PDFs in subdirectories.

    Papers whose content_hash has already been seen earlier in the same
    run are skipped as duplicates (stable across filename/path changes).

    If `output_dir` is provided, the corpus is saved in the canonical structure:
        output_dir/
        ├── raw/
        │   └── pdf/
        │       └── {doc_id}.pdf
        ├── parsed/
        │   └── {doc_id}.json
        ├── chunks/
        ├── embeddings/
        └── manifest.json
    """
    papers_dir = Path(papers_dir)
    if not papers_dir.is_dir():
        raise NotADirectoryError(f"{papers_dir} is not a directory")

    pdf_paths = sorted(papers_dir.rglob("*.pdf") if recursive else papers_dir.glob("*.pdf"))
    corpus = Corpus()
    seen_hashes: set[str] = set()

    for path in pdf_paths:
        try:
            content_hash = identity.hash_file(path)
            if content_hash in seen_hashes:
                logger.info("Skipping duplicate (already ingested): %s", path.name)
                continue

            doc = ingest_one(path)
            seen_hashes.add(content_hash)
            corpus.documents.append(doc)
            logger.info(
                "Ingested %s -> id=%s title=%r (%d sections, %d references, %d warnings)",
                path.name,
                doc.identity.id,
                doc.metadata.title,
                len(doc.sections),
                len(doc.references),
                len(doc.warnings),
            )
        except Exception as e:  # noqa: BLE001 -- deliberately broad: isolate per-file failures
            logger.error("Failed to ingest %s: %s", path.name, e)
            continue

    if output_dir is not None:
        corpus.save(output_dir)

    return corpus
