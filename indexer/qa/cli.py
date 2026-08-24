from __future__ import annotations

import argparse
import sys

from indexer.config import IndexerConfig
from indexer.database.connection import DatabaseManager
from indexer.retrieval.repository import RetrievalRepository
from indexer.retrieval.service import RetrievalService
from indexer.embeddings.sentence_transformers import SentenceTransformersProvider

from indexer.qa.ollama import OllamaQwenProvider
from indexer.qa.service import QuestionAnsweringService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask a grounded research question against the indexed corpus."
    )

    parser.add_argument(
        "question",
        help="Research question to answer.",
    )

    parser.add_argument(
        "--model",
        default="qwen3:4b-docmind",
        help="Local Ollama model.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of evidence chunks to provide to Qwen.",
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="Number of retrieval candidates.",
    )

    parser.add_argument(
        "--year-from",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--year-to",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--document-id",
        default=None,
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = IndexerConfig.from_env()

    db_manager = DatabaseManager(
        config.database_url
    )

    retrieval_repository = RetrievalRepository(
        db_manager=db_manager,
    )

    embedding_provider = SentenceTransformersProvider(
        model_name=config.embedding_model,
        model_version=config.embedding_model_version,
        expected_dimensions=config.embedding_dimensions,
    )

    retrieval_service = RetrievalService(
        repository=retrieval_repository,
    )

    llm_provider = OllamaQwenProvider(
        model=args.model,
    )

    qa_service = QuestionAnsweringService(
        retrieval_service=retrieval_service,
        llm_provider=llm_provider,
    )

    try:
        response = qa_service.answer(
            args.question,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            year_from=args.year_from,
            year_to=args.year_to,
            document_id=args.document_id,
        )

    except Exception as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1

    print()
    print(response.answer)
    print()

    if response.citations:
        print("Evidence:")
        print()

        for citation in response.citations:
            print(
                f"[{citation.evidence_id}] "
                f"{citation.document_id} / "
                f"{citation.chunk_id}"
            )
            print(citation.quote)
            print()

    else:
        print("No evidence citations were returned.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
