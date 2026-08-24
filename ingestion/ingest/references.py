"""
Stage 5 (references): turn the References/Bibliography section into a
structured list, one entry per cited work.

Per the ingestion contract: even when we can't fully resolve a reference
(authors/title/venue), we always preserve `raw_text` so nothing is lost.
Full resolution (matching to a canonical DOI/Semantic-Scholar record) is
deliberately out of scope here -- that belongs to a later graph-building
stage (`Paper A -> cites -> Paper B`), not ingestion.
"""

from __future__ import annotations

import re

from ingest.schema import Reference, Section

DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>]+\b")
YEAR_RE = re.compile(r"\((19|20)\d{2}[a-z]?\)|\b(19|20)\d{2}\b")

# Matches a leading numbered marker like "[12] " or "12. " starting an entry.
BRACKET_MARKER_RE = re.compile(r"\[(\d+)\]\s*")
DOT_MARKER_RE = re.compile(r"^(\d+)\.\s+")

REFERENCE_SECTION_NAMES = {"references", "bibliography"}


def _find_reference_section(sections: list[Section]) -> Section | None:
    for sec in sections:
        if sec.title.strip().lower().rstrip(":") in REFERENCE_SECTION_NAMES:
            return sec
        found = _find_reference_section(sec.subsections)
        if found:
            return found
    return None


def _split_entries(full_text: str) -> list[str]:
    """Split the concatenated references text into individual entries."""
    # Preferred: numbered bracket markers, e.g. "[1] ... [2] ...".
    marker_positions = [m.start() for m in BRACKET_MARKER_RE.finditer(full_text)]
    if len(marker_positions) >= 2:
        entries = []
        for i, start in enumerate(marker_positions):
            end = marker_positions[i + 1] if i + 1 < len(marker_positions) else len(full_text)
            entries.append(full_text[start:end].strip())
        return entries

    # Fallback: "1. ", "2. " style numbering, one per line-start.
    lines = full_text.split("\n")
    if sum(1 for l in lines if DOT_MARKER_RE.match(l)) >= 2:
        entries = []
        current: list[str] = []
        for line in lines:
            if DOT_MARKER_RE.match(line) and current:
                entries.append(" ".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            entries.append(" ".join(current).strip())
        return entries

    # Last resort: treat each non-empty paragraph/line as its own entry.
    return [l.strip() for l in full_text.split("\n") if l.strip()]


def extract_references(sections: list[Section]) -> list[Reference]:
    ref_section = _find_reference_section(sections)
    if not ref_section or not ref_section.paragraphs:
        return []

    full_text = "\n".join(p.text for p in ref_section.paragraphs)
    raw_entries = _split_entries(full_text)

    references: list[Reference] = []
    for i, raw in enumerate(raw_entries, start=1):
        if not raw.strip():
            continue

        # Prefer an explicit [n] marker as the id if present, else fall
        # back to sequential position.
        bracket_match = re.match(r"\[(\d+)\]", raw)
        ref_id = int(bracket_match.group(1)) if bracket_match else i

        clean = BRACKET_MARKER_RE.sub("", raw, count=1).strip()
        clean = re.sub(r"^\d+\.\s+", "", clean).strip()

        doi_match = DOI_RE.search(clean)
        year_match = YEAR_RE.search(clean)
        year = None
        if year_match:
            digits = re.search(r"(19|20)\d{2}", year_match.group(0))
            if digits:
                year = int(digits.group(0))

        references.append(
            Reference(
                id=ref_id,
                raw_text=clean,
                authors=[],   # left unresolved in v1 -- see module docstring
                title=None,   # left unresolved in v1 -- see module docstring
                year=year,
                venue=None,
                doi=doi_match.group(0).rstrip(".") if doi_match else None,
            )
        )

    return references


def strip_reference_section(sections: list[Section]) -> list[Section]:
    """Return `sections` with the References/Bibliography node removed --
    it's exposed as `Document.references` instead, per the schema."""
    kept = []
    for sec in sections:
        if sec.title.strip().lower().rstrip(":") in REFERENCE_SECTION_NAMES:
            continue
        sec.subsections = strip_reference_section(sec.subsections)
        kept.append(sec)
    return kept
