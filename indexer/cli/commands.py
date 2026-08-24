from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from indexer.config import IndexerConfig
from indexer.database.connection import DatabaseManager
from indexer.database.repository import IndexRepository
from indexer.pipeline.indexer import CorpusIndexer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="indexer",
        description="Hierarchical research document indexer with PostgreSQL + pgvector.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # index command
    index_parser = subparsers.add_parser("index", help="Index a parsed document corpus into PostgreSQL.")
    index_parser.add_argument("corpus_dir", help="Path to corpus directory (containing parsed/)")
    index_parser.add_argument("--db-url", help="PostgreSQL connection URL (default: from DATABASE_URL or postgresql://localhost:5432/research_intel)")
    index_parser.add_argument("--model", help="Embedding model name (default: BAAI/bge-base-en-v1.5)")
    index_parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size (default: 32)")
    index_parser.add_argument("--force", action="store_true", help="Force re-indexing even if documents are up to date")
    index_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")

    # status command
    status_parser = subparsers.add_parser("status", help="Show database index statistics.")
    status_parser.add_argument("--db-url", help="PostgreSQL connection URL")
    status_parser.add_argument("-v", "--verbose", action="store_true")

    return parser


def main() -> None:
    # If invoked directly with 'corpus/' without 'index' subcommand, default to index command
    argv = sys.argv[1:]
    if argv and argv[0] not in ("index", "status", "-h", "--help"):
        argv = ["index"] + argv

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "status":
        config = IndexerConfig.from_env(database_url=args.db_url)
        db = DatabaseManager(config.database_url)
        repo = IndexRepository(db)
        try:
            stats = repo.get_stats()
            print("\n=== Index Database Statistics ===")
            print(f"Documents: {stats['documents']}")
            print(f"Sections:  {stats['sections']}")
            print(f"Chunks:    {stats['chunks']}")
            print(f"Embeddings: {stats['embeddings']}")
            print(f"Citations: {stats['citations']}")
            print(f"Unresolved: {stats['unresolved_references']}")
            print(f"Statuses:  {stats['statuses']}\n")
        except Exception as e:
            print(f"Failed to query index status: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.command == "index":
        config = IndexerConfig.from_env(
            database_url=args.db_url,
            embedding_model=args.model,
            embedding_batch_size=args.batch_size,
        )

        try:
            indexer = CorpusIndexer(config=config)
            summary = indexer.index_corpus(args.corpus_dir, force=args.force)

            print("\n=== Indexing Summary ===")
            print(summary.format_report())
            print(f"Completed in {summary.duration_seconds:.2f}s\n")

            if summary.failed > 0:
                sys.exit(1)
        except Exception as e:
            print(f"Fatal error during indexing: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
