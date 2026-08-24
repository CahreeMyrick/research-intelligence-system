"""
Stage 1: Document identification.

Computes a stable identity for a PDF that does not depend on filename or
path, so re-running ingest() on the same file (even renamed/moved) is
recognized as "already seen".
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ingest.schema import DocumentIdentity, DocumentSource


def hash_file(path: Path) -> str:
    """SHA-256 of raw file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identify(path: Path) -> DocumentIdentity:
    content_hash = hash_file(path)
    source = DocumentSource(
        filename=path.name,
        path=str(path.resolve()),
        content_hash=content_hash,
    )
    # id is derived purely from content, so identical bytes -> identical id,
    # regardless of filename/path.
    doc_id = f"doc_{content_hash[:16]}"
    return DocumentIdentity(id=doc_id, source=source)
