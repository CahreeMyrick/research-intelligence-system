"""
CLI:  python -m ingest.cli --papers_dir papers/ --out corpus
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ingest.pipeline import ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a directory of paper PDFs into a canonical Corpus.")
    parser.add_argument("--papers_dir", help="Directory containing .pdf files")
    parser.add_argument("--recursive", action="store_true", help="Include PDFs in subdirectories")
    parser.add_argument("--out", "-o", default="corpus", help="Output directory or JSON file (default: corpus)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    out_path = Path(args.out)
    if out_path.suffix == ".json":
        corpus = ingest(args.papers_dir, recursive=args.recursive)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(corpus.model_dump_json(indent=2), encoding="utf-8")
        print(f"Ingested {len(corpus.documents)} document(s) -> {args.out}")
    else:
        corpus = ingest(args.papers_dir, output_dir=args.out, recursive=args.recursive)
        print(f"Ingested {len(corpus.documents)} document(s) -> {args.out}/")

    if not corpus.documents:
        sys.exit(1)


if __name__ == "__main__":
    main()
