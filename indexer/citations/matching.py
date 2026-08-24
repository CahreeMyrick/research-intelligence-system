from __future__ import annotations

import re
from typing import Optional

DOI_PREFIX_RE = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.IGNORECASE)
ARXIV_PREFIX_RE = re.compile(r"^arxiv:\s*", re.IGNORECASE)
ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
PUNCT_COLLAPSE_RE = re.compile(r"[^\w\s]", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Normalize DOI string for stable matching."""
    if not doi:
        return None
    d = doi.strip().lower()
    d = DOI_PREFIX_RE.sub("", d).strip()
    return d or None


def normalize_arxiv(arxiv_id: Optional[str]) -> Optional[str]:
    """Normalize arXiv identifier (strip prefix and version suffix)."""
    if not arxiv_id:
        return None
    a = arxiv_id.strip().lower()
    a = ARXIV_PREFIX_RE.sub("", a).strip()
    a = ARXIV_VERSION_RE.sub("", a).strip()
    return a or None


def normalize_title(title: Optional[str]) -> Optional[str]:
    """Normalize paper title for fuzzy/lexical candidate matching."""
    if not title:
        return None
    t = title.strip().lower()
    t = PUNCT_COLLAPSE_RE.sub(" ", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t or None
