from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from indexer.config import IndexerConfig
from indexer.database.connection import DatabaseManager
from indexer.database.repository import IndexRepository
from indexer.pipeline.indexer import CorpusIndexer


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="indexer",
        description="Hierarchical research document indexer with PostgreSQL + pgvector.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Subcommand to run",
    )

    # -----------------
    # index command
    # -----------------

    index_parser = subparsers.add_parser(
        "index",
        help="Index a parsed document corpus into PostgreSQL.",
    )

    index_parser.add_argument(
        "corpus_dir",
        help="Path to corpus directory containing parsed documents.",
    )

    index_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if documents are already up to date.",
    )

    index_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    # -----------------
    # status command
    # -----------------

    status_parser = subparsers.add_parser(
        "status",
        help="Show database index statistics.",
    )

    status_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    return parser


def main() -> None:
    # Allow:
    #
    #   python -m indexer.cli.commands ingestion/corpus
    #
    # to behave like:
    #
    #   python -m indexer.cli.commands index ingestion/corpus

    argv = sys.argv[1:]

    if argv and argv[0] not in (
        "index",
        "status",
        "-h",
        "--help",
    ):
        argv = ["index"] + argv

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=(
            logging.DEBUG
            if getattr(args, "verbose", False)
            else logging.INFO
        ),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # -----------------
    # Load configuration
    # -----------------

    try:
        config = IndexerConfig.from_yaml(CONFIG_PATH)

    except Exception as exc:
        print(
            f"Failed to load configuration from {CONFIG_PATH}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # -----------------
    # status command
    # -----------------

    if args.command == "status":
        db = DatabaseManager(
            config.database_url,
        )

        repo = IndexRepository(
            db,
        )

        try:
            stats = repo.get_stats()

            print()
            print("=== Index Database Statistics ===")
            print(f"Documents:  {stats['documents']}")
            print(f"Sections:   {stats['sections']}")
            print(f"Chunks:     {stats['chunks']}")
            print(f"Embeddings: {stats['embeddings']}")
            print(f"Citations:  {stats['citations']}")
            print(f"Unresolved: {stats['unresolved_references']}")
            print(f"Statuses:   {stats['statuses']}")
            print()

        except Exception as exc:
            print(
                f"Failed to query index status: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        return

    # -----------------
    # index command
    # -----------------

    if args.command == "index":
        try:
            indexer = CorpusIndexer(
                config=config,
            )

            summary = indexer.index_corpus(
                args.corpus_dir,
                force=args.force,
            )

            print()
            print("=== Indexing Summary ===")
            print(summary.format_report())
            print(
                f"Completed in {summary.duration_seconds:.2f}s"
            )
            print()

            if summary.failed > 0:
                sys.exit(1)

        except Exception as exc:
            print(
                f"Fatal error during indexing: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
