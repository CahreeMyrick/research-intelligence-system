"""
Stage 4: Scientific metadata extraction.

Heuristic, offline extraction of the bibliographic facts about a paper:
title, authors, abstract, DOI, arXiv id, publication year.

Unknown fields are left as None rather than guessed -- per the ingestion
contract, `venue = null` is preferable to inventing information. This is
also the natural extension point for a networked enrichment step later
(e.g. resolving DOI -> full Crossref record), kept out of v1 on purpose
since ingestion should not require network access.
"""

from __future__ import annotations

import re

from ingest import structure
from ingest.extraction import RawDocument
from ingest.schema import DocumentMetadata, Section

DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>]+\b")
ARXIV_RE = re.compile(r"\barXiv:\s*(\d{4}\.\d{4,5})(v\d+)?\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Lines that are almost certainly not part of the author list even if they
# sit in the right position on the page.
AUTHOR_BLOCK_STOPWORDS = ("abstract", "@", "http", "university", "department")


def _first_page_text(raw: RawDocument) -> str:
    return "\n".join(b.text for b in raw.blocks if b.page == 0)


def _find_abstract(sections: list[Section]) -> str | None:
    for sec in sections:
        if sec.title.strip().lower() == "abstract":
            text = " ".join(p.text for p in sec.paragraphs).strip()
            return text or None
        found = _find_abstract(sec.subsections)
        if found:
            return found
    return None


def _guess_title_and_authors(raw: RawDocument) -> tuple[str | None, list[str]]:
    # Reuse structure's wrapped-heading merge so a title that PyMuPDF split
    # across two blocks (no native paragraph markup in PDFs) is treated as
    # one block here too, consistent with how it's treated in the section tree.
    body_size = structure._estimate_body_size(raw.blocks)
    merged_blocks = structure._merge_wrapped_titles(raw.blocks, body_size)
    page0 = [b for b in merged_blocks if b.page == 0]
    if not page0:
        return None, []

    # Title: the largest-font block within the first handful of blocks on
    # page 1 (titles are almost always near the top and visually dominant).
    candidates = page0[:8]
    title_block = max(candidates, key=lambda b: b.size)
    title = title_block.text.strip() if title_block else None

    # Authors: the next block(s) after the title, up until something that
    # looks like an affiliation line, the abstract heading, or a big drop
    # back to body-text size.
    authors: list[str] = []
    after_title = False
    for b in page0:
        if b is title_block:
            after_title = True
            continue
        if not after_title:
            continue
        low = b.text.strip().lower()
        if low.startswith("abstract") or any(sw in low for sw in AUTHOR_BLOCK_STOPWORDS):
            break
        # Author lines are typically short and comma/"and"-separated names.
        if len(b.text.split()) > 25:
            break
        authors.extend(_split_author_line(b.text))
        # Only take the first plausible author block -- affiliations
        # usually follow immediately after on the next block.
        break

    return title, authors


def _split_author_line(line: str) -> list[str]:
    line = re.sub(r"[\*\u2020\u2021\d]", "", line)  # strip footnote markers
    parts = re.split(r",| and |&", line)
    names = [p.strip() for p in parts if p.strip()]
    return names


def extract_metadata(raw: RawDocument, sections: list[Section]) -> DocumentMetadata:
    page0_text = _first_page_text(raw)
    first_two_pages_text = "\n".join(
        b.text for b in raw.blocks if b.page in (0, 1)
    )

    title, authors = _guess_title_and_authors(raw)
    abstract = _find_abstract(sections)

    doi_match = DOI_RE.search(first_two_pages_text)
    doi = doi_match.group(0).rstrip(".") if doi_match else None

    arxiv_match = ARXIV_RE.search(first_two_pages_text)
    arxiv_id = arxiv_match.group(1) if arxiv_match else None

    year = None
    if arxiv_id:
        # arXiv ids encode YYMM -> derive a plausible 4-digit year.
        yy = int(arxiv_id[:2])
        year = 2000 + yy
    else:
        year_match = YEAR_RE.search(page0_text)
        if year_match:
            year = int(year_match.group(0))

    return DocumentMetadata(
        title=title,
        authors=authors,
        affiliations=[],  # left unresolved in v1 -- see module docstring
        publication_year=year,
        venue=None,       # left unresolved in v1 -- see module docstring
        doi=doi,
        arxiv_id=arxiv_id,
        abstract=abstract,
        other_identifiers={},
    )
