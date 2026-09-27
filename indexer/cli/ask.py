from __future__ import annotations

import argparse

from indexer.config import IndexerConfig
from indexer.database.connection import DatabaseManager
from indexer.embeddings.ollama import OllamaEmbeddingProvider
from indexer.embeddings.sentence_transformers import SentenceTransformersProvider
from indexer.llm import AnswerService, OllamaLLM
from indexer.retrieval.repository import RetrievalRepository
from indexer.retrieval.service import RetrievalService

from pathlib import Path

INDEXER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = INDEXER_DIR / "config.yaml"

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question about the indexed research corpus."
    )

    parser.add_argument(
        "question",
        nargs="+",
        help="Question to ask.",
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to YAML configuration file.",
    )

    args = parser.parse_args()

    question = " ".join(args.question)

    config = IndexerConfig.from_yaml(args.config)

    db = DatabaseManager(
        config.database_url,
    )

    # Match the provider selection used by CorpusIndexer.
    if ":" in config.embedding_model:
        embedding_provider = OllamaEmbeddingProvider(
            model=config.embedding_model,
            model_version=config.embedding_model_version,
            expected_dimensions=config.embedding_dimensions,
        )
    else:
        embedding_provider = SentenceTransformersProvider(
            model_name=config.embedding_model,
            model_version=config.embedding_model_version,
            expected_dimensions=config.embedding_dimensions,
        )

    retrieval_repository = RetrievalRepository(
        db_manager=db,
        embedding_provider=embedding_provider,
    )

    retrieval_service = RetrievalService(
        repository=retrieval_repository,
    )

    llm = OllamaLLM(
        model=config.llm_model,
    )

    answer_service = AnswerService(
        llm=llm,
    )

    evidence = retrieval_service.retrieve(
        question,
        top_k=config.top_k,
        candidate_k=config.candidate_k,
    )

    answer = answer_service.answer(
        question,
        evidence,
    )

    print()
    print("ANSWER")
    print("------")
    print(answer)

    print()
    print(f"Evidence retrieved: {len(evidence.items)}")


if __name__ == "__main__":
    main()
