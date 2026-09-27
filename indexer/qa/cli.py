from __future__ import annotations

import argparse
import sys
from pathlib import Path

from indexer.config import IndexerConfig
from indexer.database.connection import DatabaseManager
from indexer.retrieval.repository import RetrievalRepository
from indexer.retrieval.service import RetrievalService
from indexer.embeddings.sentence_transformers import SentenceTransformersProvider

from indexer.qa.ollama import OllamaQwenProvider
from indexer.qa.service import QuestionAnsweringService


INDEXER_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = INDEXER_DIR / "config.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask a grounded research question against the indexed corpus."
    )

    parser.add_argument(
        "question",
        help="Research question to answer.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to YAML configuration file.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Local Ollama model.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of evidence chunks to provide to Qwen.",
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Number of retrieval candidates.",
    )

    parser.add_argument(
        "--year-from",
        type=int,
        default=None,
        help="Only search documents from this year onward.",
    )

    parser.add_argument(
        "--year-to",
        type=int,
        default=None,
        help="Only search documents up to this year.",
    )

    parser.add_argument(
        "--document-id",
        default=None,
        help="Restrict retrieval to a specific document.",
    )

    parser.add_argument(
        "--knowledge-graph", type=Path, default=None,
        help="Enable knowledge-infused retrieval using a supplied graph JSON file.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = IndexerConfig.from_yaml(args.config)

    db_manager = DatabaseManager(
        config.database_url
    )

    embedding_provider = SentenceTransformersProvider(
        model_name=config.embedding_model,
        model_version=config.embedding_model_version,
        expected_dimensions=config.embedding_dimensions,
    )

    retrieval_repository = RetrievalRepository(
        db_manager=db_manager,
        embedding_provider=embedding_provider,
    )

    retrieval_service = RetrievalService(
        repository=retrieval_repository,
    )

    llm_provider = OllamaQwenProvider(
        model=args.model if args.model is not None else config.llm_model,
    )

    if args.knowledge_graph is not None:
        from importlib import import_module
        knowledge = import_module("indexer.knowledge-infusion")
        retrieval_service = knowledge.KnowledgeInfusedRetrievalService(
            retrieval_service=retrieval_service,
            parser=knowledge.LLMQueryParser(llm_provider),
            graph=knowledge.load_graph(args.knowledge_graph),
        )

    qa_service = QuestionAnsweringService(
        retrieval_service=retrieval_service,
        llm_provider=llm_provider,
    )

    try:
        response = qa_service.answer(
            args.question,
            top_k=args.top_k if args.top_k is not None else config.top_k,
            candidate_k=args.candidate_k if args.candidate_k is not None else config.candidate_k,
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
