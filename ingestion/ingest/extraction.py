"""
Stage 2: Content extraction.

Pulls raw content out of the PDF using PyMuPDF. This stage knows nothing
about "sections" or "papers" -- it just turns bytes into a flat, ordered
list of text blocks, each carrying the layout signal (font size / boldness /
page / position) that stage 3 (structure reconstruction) needs.

Granularity note: we extract at PyMuPDF's *block* level rather than line
level. For single/double-column academic PDFs a block corresponds closely
to a paragraph or a heading, which makes stage 3 much simpler than trying
to re-merge individual lines into paragraphs ourselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class TextBlock:
    text: str          # full block text, lines joined with spaces
    page: int          # 0-indexed
    size: float        # dominant font size in the block (points)
    bold: bool         # majority of the block's characters are bold
    order: int         # global reading-order index across the whole document


@dataclass
class RawDocument:
    blocks: list[TextBlock]
    page_count: int


def _is_bold(font_name: str, flags: int) -> bool:
    # PyMuPDF flag bit 2**4 (16) marks bold; font name often also says "Bold".
    return bool(flags & 2**4) or "bold" in font_name.lower()


def extract(path: Path) -> RawDocument:
    """
    Extract every text block from the PDF, in reading order, with font
    metadata attached. Raises on encrypted/corrupt PDFs -- the caller
    (pipeline.py) is responsible for catching and isolating that failure
    to a single document.
    """
    doc = pymupdf.open(path)
    if doc.is_encrypted:
        # Try an empty-password unlock (common for "restricted but not
        # really protected" PDFs); if that fails, surface it to the caller.
        if not doc.authenticate(""):
            raise ValueError(f"Encrypted PDF could not be opened: {path.name}")

    blocks: list[TextBlock] = []
    order = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_dict = page.get_text("dict")

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 = text block, 1 = image block
                continue

            line_texts: list[str] = []
            sizes: list[float] = []
            char_counts: list[int] = []
            bold_char_counts: list[int] = []

            for line in block.get("lines", []):
                span_texts = []
                for s in line.get("spans", []):
                    # PDF text may contain NULs, which PostgreSQL cannot store.
                    txt = s.get("text", "").replace("\x00", " ")
                    if not txt.strip():
                        continue
                    span_texts.append(txt)
                    n_chars = len(txt)
                    sizes.append(s.get("size", 0.0))
                    char_counts.append(n_chars)
                    bold_char_counts.append(
                        n_chars if _is_bold(s.get("font", ""), s.get("flags", 0)) else 0
                    )
                if span_texts:
                    line_texts.append("".join(span_texts).strip())

            if not line_texts:
                continue

            block_text = " ".join(t for t in line_texts if t).strip()
            if not block_text:
                continue

            total_chars = sum(char_counts) or 1
            # weight font size by character count so a long body line
            # dominates over e.g. a single stray superscript character
            dominant_size = (
                sum(sz * n for sz, n in zip(sizes, char_counts)) / total_chars
            )
            is_bold = sum(bold_char_counts) > total_chars / 2

            blocks.append(
                TextBlock(
                    text=block_text,
                    page=page_index,
                    size=round(dominant_size, 1),
                    bold=is_bold,
                    order=order,
                )
            )
            order += 1

    page_count = len(doc)
    doc.close()
    return RawDocument(blocks=blocks, page_count=page_count)
